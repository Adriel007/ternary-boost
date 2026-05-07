import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizer

from shared.logging import get_logger

logger = get_logger("shared.checkpoint")

# ── Sharded model save (800 MB chunks, avoids OOM from full state_dict) ──

_SHARD_TARGET_BYTES = int(0.8 * 1024 ** 3)  # 800 MB


def _save_model_sharded(
    model: PreTrainedModel,
    output_dir: str,
    *,
    prefix: str = "model",
    half: bool = True,
) -> None:
    """Save model weights as sharded safetensors (tensor-at-a-time scan).

    Iterates ``model.named_parameters()`` one tensor at a time so the full
    state_dict is never resident in memory.  Tensors are flushed to disk
    in ~800 MB shards with a HuggingFace-compatible index.
    """
    from safetensors.torch import save_file as _save_st

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    shard_batch: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    shard_files: list[tuple[str, list[str]]] = []

    def _flush():
        nonlocal shard_batch, shard_bytes, shard_idx
        if not shard_batch:
            return
        name = f"{prefix}-{shard_idx + 1:05d}-of-XXXXX.safetensors"
        path = str(out / name)
        _save_st(shard_batch, path)
        shard_files.append((name, list(shard_batch.keys())))
        shard_batch.clear()
        shard_bytes = 0
        shard_idx += 1

    total = sum(1 for _ in model.named_parameters())
    for key, param in model.named_parameters():
        t = param.detach()
        if half:
            t = t.half() if t.dtype != torch.bfloat16 else t
        t = t.cpu()
        nbytes = t.numel() * t.element_size()
        shard_batch[key] = t
        shard_bytes += nbytes

        if shard_bytes >= _SHARD_TARGET_BYTES:
            _flush()
            for fname, keys in shard_files[-1:]:
                for k in keys:
                    weight_map[k] = fname

    _flush()
    for fname, keys in shard_files[-1:]:
        for k in keys:
            weight_map[k] = fname

    # Fix XXXXX → real shard count
    total_shards = len(shard_files)
    for old_name, keys in shard_files:
        new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
        if old_name != new_name:
            os.rename(str(out / old_name), str(out / new_name))
        for k in keys:
            weight_map[k] = new_name

    total_params = sum(p.numel() for p in model.parameters())
    index = {
        "metadata": {"total_size": total_params * (2 if half else 4)},
        "weight_map": weight_map,
    }
    with open(str(out / f"{prefix}.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"  Sharded save: {total} params → {total_shards} shard(s)")


# ── Public API ─────────────────────────────────────────────────────


def ensure_pad_token_id(model_id: str | Path, **kwargs):
    """Load config and guarantee ``pad_token_id`` is set.

    Phi-2 and some older models ship without ``pad_token_id`` in their
    config, causing ``AttributeError`` in ``PhiModel.__init__`` with
    recent transformers versions.
    """
    config = AutoConfig.from_pretrained(model_id, **kwargs)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = getattr(config, "eos_token_id", 0) or 0
    return config


def save_checkpoint(
    model: PreTrainedModel,
    output_path: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Save model as sharded safetensors (800 MB chunks) + config + tokenizer.

    The full ``state_dict()`` is never materialised — parameters are streamed
    one at a time via ``named_parameters()`` so peak CPU RAM stays under
    ~1 GB regardless of model size.
    """
    os.makedirs(output_path, exist_ok=True)

    _save_model_sharded(model, output_path, prefix="model")

    model.config.save_pretrained(output_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    if metadata is not None:
        with open(os.path.join(output_path, "pipeline_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return output_path


def save_training_checkpoint(
    output_path: str,
    model: PreTrainedModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    step: int = 0,
    tokens_seen: int = 0,
    extra_metadata: Optional[dict] = None,
) -> str:
    """Save a resumable training checkpoint (model shards + optimizer + metadata).

    Model weights are saved as sharded safetensors.  Optimizer and scheduler
    state are saved as a single ``training_state.pt`` file (small — contains
    only parameter IDs and scalar state, not weight copies).
    """
    os.makedirs(output_path, exist_ok=True)

    _save_model_sharded(model, output_path, prefix="model")
    model.config.save_pretrained(output_path)

    training_state = {
        "step": step,
        "tokens_seen": tokens_seen,
    }
    if optimizer is not None:
        training_state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler_state_dict"] = scheduler.state_dict()
    if extra_metadata is not None:
        training_state["metadata"] = extra_metadata

    torch.save(training_state, os.path.join(output_path, "training_state.pt"))
    logger.info(f"  Training checkpoint saved: step={step}, tokens={tokens_seen:,}")
    return output_path


def load_training_checkpoint(
    output_path: str,
    model: PreTrainedModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
) -> dict:
    """Load a resumable training checkpoint.

    Model weights must already be loaded (e.g. via ``load_checkpoint`` or
    ``AutoModelForCausalLM.from_pretrained``).  This function restores
    optimizer/scheduler state and returns training metadata.

    Returns:
        dict with keys ``step``, ``tokens_seen``.
    """
    state_path = os.path.join(output_path, "training_state.pt")
    if not os.path.exists(state_path):
        logger.warning(f"No training_state.pt found in {output_path}")
        return {"step": 0, "tokens_seen": 0}

    state = torch.load(state_path, map_location="cpu", weights_only=False)

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "step": state.get("step", 0),
        "tokens_seen": state.get("tokens_seen", 0),
    }


def load_checkpoint(
    model_cls,
    checkpoint_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    """Load model from safetensors checkpoint (single or sharded)."""
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = model_cls.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=dtype or torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    return model


def copy_config(src_path: str, dst_path: str) -> None:
    """Copy HuggingFace config files between directories."""
    for fname in ["config.json", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "tokenizer.model"]:
        src = os.path.join(src_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_path, fname))
