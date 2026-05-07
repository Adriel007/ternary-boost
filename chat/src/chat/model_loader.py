"""Model loading with automatic TernaryBoost compression pipeline.

Any standard FP16 HuggingFace causal LM is accepted. On first load, the
compression pipeline (PT-BitNet → optional LoRA-KD) is applied and the result
cached as a sharded safetensors checkpoint. Subsequent loads reuse the cache.

The cache layout (per model) is::

    cache/ternary/<safe_model_name>/
        config.json, tokenizer.json, ...
        model-00001-of-NNNNN.safetensors  (sharded ~800 MB chunks)
        model.safetensors.index.json
        lora_weights.safetensors          (if LoRA was applied)
        stage1_done.txt                   (PT-BitNet checkpoint marker)
        stage4_done.txt                   (LoRA checkpoint marker)
        pipeline_metadata.json
"""

import json
import os
import shutil
import sys
import time
import gc
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from shared.logging import get_logger
from .config import ModelEntry

logger = get_logger("tchat")


def _cache_path(model_path: str, cache_root: Path) -> Path:
    safe_name = model_path.replace("/", "__").replace("\\", "__")
    return cache_root / safe_name


def _stage_done(cache_dir: Path, stage: int) -> bool:
    return (cache_dir / f"stage{stage}_done.txt").exists()


def _mark_stage_done(cache_dir: Path, stage: int, elapsed: float) -> None:
    (cache_dir / f"stage{stage}_done.txt").write_text(f"{elapsed:.1f}s\n")


def _is_fully_cached(cache_dir: Path) -> bool:
    """A model is considered ready if PT-BitNet (stage 1) has run."""
    return _stage_done(cache_dir, 1)


def load_model(entry: ModelEntry, cache_root: Optional[str] = None) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    cache_root = Path(cache_root or "./cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    hf_cache = cache_root / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))

    ternary_cache = cache_root / "ternary"
    cache_dir = _cache_path(entry.path, ternary_cache)

    if _is_fully_cached(cache_dir):
        logger.info(f"Loading cached ternary model from {cache_dir}")
        return _load_cached(cache_dir, entry)

    logger.info(f"Pipeline for {entry.path} — data stored under: {cache_root}")
    return _compress_and_cache(entry, cache_dir, hf_cache)


def _ensure_pipeline_imports():
    _project = Path(__file__).resolve().parent.parent.parent.parent
    for _mod in ("shared", "pt_bitnet", "eval"):
        _src = str(_project / _mod / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)


def _load_cached(cache_dir: Path, entry: ModelEntry) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a cached ternary model. LoRA adapters are re-attached if present."""
    _ensure_pipeline_imports()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        str(cache_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True,
        cache_dir=str(cache_dir.parent.parent / "huggingface"),
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(cache_dir), trust_remote_code=True,
        cache_dir=str(cache_dir.parent.parent / "huggingface"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Re-attach LoRA adapters if present
    lora_path = cache_dir / "lora_weights.safetensors"
    if lora_path.exists():
        from pt_bitnet.lora import LoRAConfig, _add_lora_to_model, load_lora_weights
        _add_lora_to_model(model, LoRAConfig())
        load_lora_weights(model, str(lora_path))
        logger.info("  LoRA adapters re-attached from cache")

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Cached ternary model loaded: {params:,} parameters")
    return model, tokenizer


# ── Save / load intermediate checkpoint ─────────────────────────────────

def _save_model_state(model, tokenizer, cache_dir: Path) -> None:
    """Save model as sharded safetensors to limit peak system RAM.

    Tensors are iterated one at a time from named_parameters() (no full
    state_dict in memory), converted to bf16, and saved in 800 MB shards.
    HuggingFace from_pretrained loads sharded checkpoints natively via
    the index file.
    """
    model.config.save_pretrained(str(cache_dir))
    tokenizer.save_pretrained(str(cache_dir))

    from safetensors.torch import save_file

    target_bytes = int(0.8 * 1024**3)  # 800 MB per shard
    weight_map = {}
    shard_batch = {}
    shard_bytes = 0
    shard_idx = 0
    shard_files = []
    total_params = sum(1 for _ in model.named_parameters())
    param_count = 0
    last_log = [0]

    def _flush_shard():
        nonlocal shard_batch, shard_bytes, shard_idx
        if not shard_batch:
            return
        name = f"model-{shard_idx + 1:05d}-of-XXXXX.safetensors"
        path = str(cache_dir / name)
        save_file(shard_batch, path)
        shard_files.append((name, list(shard_batch.keys())))
        logger.info(f"  Saved shard {shard_idx + 1}: {name} "
                    f"({len(shard_batch)} tensors, {shard_bytes/1e9:.2f} GB)")
        shard_batch.clear()
        shard_bytes = 0
        shard_idx += 1

    logger.info(f"Saving model ({total_params} params, sharded)...")
    for key, param in model.named_parameters():
        param_count += 1
        tensor = param.detach().to(torch.bfloat16).cpu()
        nbytes = tensor.numel() * tensor.element_size()

        shard_batch[key] = tensor
        shard_bytes += nbytes
        weight_map[key] = None

        if shard_bytes >= target_bytes:
            _flush_shard()
            for fname, keys in shard_files[-1:]:
                for k in keys:
                    weight_map[k] = fname

        pct = 100 * param_count // total_params
        if pct >= last_log[0] + 25:
            logger.info(f"  Saving: {pct}% ({param_count}/{total_params} tensors)")
            last_log[0] = pct

    _flush_shard()
    for fname, keys in shard_files[-1:]:
        for k in keys:
            weight_map[k] = fname

    total_shards = len(shard_files)
    for old_name, keys in shard_files:
        new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
        if old_name != new_name:
            os.rename(str(cache_dir / old_name), str(cache_dir / new_name))
        for k in keys:
            weight_map[k] = new_name

    total_params_count = sum(p.numel() for p in model.parameters())
    index = {
        "metadata": {"total_size": total_params_count * 2},
        "weight_map": weight_map,
    }
    with open(str(cache_dir / "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


def _load_model_state(cache_dir: Path, entry: ModelEntry) -> tuple:
    """Load intermediate model state (for pipeline resume)."""
    _ensure_pipeline_imports()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    _cfg = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
    if getattr(_cfg, "pad_token_id", None) is None:
        _cfg.pad_token_id = getattr(_cfg, "eos_token_id", 0) or 0
    model = AutoModelForCausalLM.from_pretrained(
        str(cache_dir), config=_cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(cache_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ── Compression pipeline ───────────────────────────────────────────────

def _compress_and_cache(
    entry: ModelEntry, cache_dir: Path, hf_cache: Path
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Run the TernaryBoost pipeline with incremental checkpoints.

    Stages:
      1. PT-BitNet — symmetric ternary + 1% outliers + lm_head Hessian comp
      4. LoRA fine-tuning — KD from FP16 teacher (only if entry.lora_rank > 0)

    Stage numbers are non-contiguous to preserve compatibility with prior
    cache layouts; stages 2 and 3 (formerly ParetoQ and Tequila) were
    permanently removed — see results/history.md.
    """
    _ensure_pipeline_imports()

    from pt_bitnet import apply_pt_bitnet, PTBitNetConfig
    from shared.data import load_calibration_texts

    has_cuda = torch.cuda.is_available()
    logger.info(f"Device: {'GPU' if has_cuda else 'CPU'}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    if _stage_done(cache_dir, 1):
        logger.info("Stage 1 checkpoint found — resuming from PT-BitNet output")
        model, tokenizer = _load_model_state(cache_dir, entry)
    else:
        logger.info("=" * 50)
        logger.info(f"Loading base model: {entry.path}")
        t0 = time.time()
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            entry.path, trust_remote_code=True, cache_dir=str(hf_cache),
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
        model = AutoModelForCausalLM.from_pretrained(
            entry.path, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="cpu", trust_remote_code=True, cache_dir=str(hf_cache),
            config=config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            entry.path, trust_remote_code=True, cache_dir=str(hf_cache),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Base model loaded in {time.time() - t0:.1f}s "
                    f"({sum(p.numel() for p in model.parameters()):,} params)")

    texts = None

    def _ensure_texts():
        nonlocal texts
        if texts is not None:
            return texts
        logger.info("Preparing calibration data...")
        try:
            texts = load_calibration_texts(source="wikitext", num_samples=200)
        except Exception:
            texts = ["The capital of France is Paris. " * 10] * 70
        return texts

    # Stage 1: PT-BitNet
    if not _stage_done(cache_dir, 1):
        logger.info("=" * 50)
        logger.info("Stage 1: PT-BitNet post-training quantization")
        t0 = time.time()
        comp_steps = 50 if has_cuda else 10
        # Symmetric: ITF (asymmetric=True) is incompatible with outliers
        model = apply_pt_bitnet(
            model,
            PTBitNetConfig(block_size=128, outlier_clip_threshold=3.0,
                           outlier_fraction=0.01, compensation_steps=comp_steps,
                           asymmetric=False),
            tokenizer=tokenizer,
            calibration_texts=_ensure_texts()[:32],
        )
        elapsed = time.time() - t0
        logger.info(f"PT-BitNet complete in {elapsed:.1f}s")
        _save_model_state(model, tokenizer, cache_dir)
        _mark_stage_done(cache_dir, 1, elapsed)
    else:
        logger.info("Stage 1: PT-BitNet [SKIPPED — checkpoint exists]")

    # Stage 4: LoRA fine-tuning (quality recovery)
    # Adds rank-decomposition adapters on top of frozen ternary weights,
    # fine-tuned via KD from the FP16 teacher. LoRA weights are saved
    # separately so the ternary backbone stays untouched for INT2 export.
    # Set lora_rank=0 in ModelEntry to skip.
    lora_rank = getattr(entry, "lora_rank", 0)
    if lora_rank > 0 and not _stage_done(cache_dir, 4):
        logger.info("=" * 50)
        logger.info(f"Stage 4: LoRA fine-tuning (rank={lora_rank}, quality recovery)")
        t0 = time.time()

        if has_cuda:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            est_model = sum(p.numel() for p in model.parameters()) * 2 / 1e9
            if vram_total < est_model * 1.8:
                logger.warning(f"  VRAM tight for LoRA (GPU={vram_total:.1f} GB, "
                               f"model={est_model:.1f} GB). Attempting anyway...")

            from pt_bitnet.lora import finetune_lora, LoRAConfig

            logger.info("  Loading FP16 teacher on CPU for logit pre-computation...")
            # Move student to GPU before loading teacher on CPU.
            # Otherwise student (5.6 GB) + teacher (5.6 GB) = 11.2 GB > Colab T4 RAM (12.7 GB).
            model.cuda()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("  Student moved to GPU, freeing CPU RAM for teacher")

            from transformers import AutoConfig
            teacher_config = AutoConfig.from_pretrained(
                entry.path, trust_remote_code=True, cache_dir=str(hf_cache),
            )
            if not hasattr(teacher_config, "pad_token_id") or teacher_config.pad_token_id is None:
                teacher_config.pad_token_id = 0
            teacher = AutoModelForCausalLM.from_pretrained(
                entry.path, torch_dtype=dtype, low_cpu_mem_usage=True,
                device_map="cpu", trust_remote_code=True, cache_dir=str(hf_cache),
                config=teacher_config,
            )

            texts_data = _ensure_texts()
            lo_cfg = LoRAConfig(
                rank=lora_rank,
                num_steps=getattr(entry, "lora_steps", 1000),
            )
            # finetune_lora pre-computes teacher logits on CPU, frees the teacher,
            # then fine-tunes the student alone. Peak VRAM: student (~6 GB on Phi-2).
            model = finetune_lora(model, tokenizer, texts_data[:50], teacher, lo_cfg)

            # LoRA storage policy:
            #   merge_lora=True  → merge LoRA into dense weights (no re-quantize).
            #                     5.6 GB on disk, PPL ~1.24× — destroys INT2 export.
            #   merge_lora=False → keep LoRA adapters separate (default).
            #                     Required for INT2 export and the hybrid runtime.
            merge = getattr(entry, "merge_lora", False)
            if merge:
                from pt_bitnet.lora import merge_lora_to_weights
                model = merge_lora_to_weights(model)
                logger.info("  LoRA merged into dense weights (no re-quantize)")
            else:
                from pt_bitnet.lora import keep_lora_separate
                model = keep_lora_separate(model)
            _save_model_state(model, tokenizer, cache_dir)

            elapsed = time.time() - t0
            logger.info(f"LoRA fine-tuning complete in {elapsed:.1f}s")
            _mark_stage_done(cache_dir, 4, elapsed)
        else:
            logger.info("  LoRA requires GPU — skipping on CPU")
            _mark_stage_done(cache_dir, 4, 0)
    elif lora_rank == 0:
        logger.info("Stage 4: LoRA [SKIPPED — lora_rank=0]")
        if not _stage_done(cache_dir, 4):
            _mark_stage_done(cache_dir, 4, 0)
    else:
        logger.info("Stage 4: LoRA [SKIPPED — checkpoint exists]")
        lora_path = cache_dir / "lora_weights.safetensors"
        if lora_path.exists() and not any("lora" in n for n, _ in model.named_parameters()):
            from pt_bitnet.lora import LoRAConfig, _add_lora_to_model, load_lora_weights
            _add_lora_to_model(model, LoRAConfig())
            load_lora_weights(model, str(lora_path))
            logger.info("  Loaded LoRA weights from cache")

    total_time = time.time() - total_start
    meta = {
        "source_model": entry.path, "pipeline_version": "2.0",
        "total_time_s": total_time,
        "stages_applied": ["pt_bitnet"] + (["lora"] if lora_rank > 0 else []),
        "params": sum(p.numel() for p in model.parameters()),
        "device": "GPU" if has_cuda else "CPU",
    }
    (cache_dir / "pipeline_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Pipeline complete in {total_time:.0f}s — cached at {cache_dir}")
    return model, tokenizer


# ── Cache management ───────────────────────────────────────────────────

def clear_cache(model_path: Optional[str] = None, cache_root: Optional[str] = None) -> int:
    cache_root = Path(cache_root or "./cache").resolve()
    ternary_cache = cache_root / "ternary"
    removed = 0
    if model_path:
        cache_dir = _cache_path(model_path, ternary_cache)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed = 1
    else:
        if ternary_cache.exists():
            for d in ternary_cache.iterdir():
                if d.is_dir():
                    shutil.rmtree(d)
                    removed += 1
    return removed


def list_cache(cache_root: Optional[str] = None) -> list[dict]:
    cache_root = Path(cache_root or "./cache").resolve()
    ternary_cache = cache_root / "ternary"
    entries = []
    if not ternary_cache.exists():
        return entries
    for d in sorted(ternary_cache.iterdir()):
        meta_file = d / "pipeline_metadata.json"
        stages_done = sum(1 for s in (1, 4) if (d / f"stage{s}_done.txt").exists())
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            meta["stages_done"] = stages_done
            entries.append({"path": str(d), **meta})
        elif stages_done > 0:
            entries.append({
                "path": str(d), "source_model": d.name.replace("__", "/"),
                "stages_done": stages_done, "params": 0, "total_time_s": 0,
            })
    return entries


def unload_model(model: PreTrainedModel) -> None:
    if model is not None:
        model.cpu()
        del model
        torch.cuda.empty_cache()
