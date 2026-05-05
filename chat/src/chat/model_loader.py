"""Model loading with automatic TernaryBoost compression pipeline.

Any standard FP16 HuggingFace model is accepted. On first load, the full
compression pipeline (PT-BitNet → ParetoQ+ZeroQAT → Tequila) is applied
and the result cached. Subsequent loads reconstruct custom quantization
layers (QuantizeLinear, UltraQuantLinear) from saved parameters.

Critical: standard nn.Linear layers do NOT accelerate ternary inference.
Custom layers (QuantizeLinear, UltraQuantLinear) must be reconstructed
after loading from safetensors, as HuggingFace from_pretrained always
creates standard architectures.
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
    return _stage_done(cache_dir, 3)


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


# ── Load with custom layer reconstruction ──────────────────────────────

def _ensure_pipeline_imports():
    _project = Path(__file__).resolve().parent.parent.parent.parent
    for _mod in ("shared", "pt_bitnet", "paretoq", "tequila", "eval"):
        _src = str(_project / _mod / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)


def _load_cached(cache_dir: Path, entry: ModelEntry) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load cached ternary model and reconstruct custom quantization layers."""
    _ensure_pipeline_imports()
    from safetensors.torch import load_file

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    # Fix: newer transformers requires pad_token_id on config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        str(cache_dir), trust_remote_code=True,
    )
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

    custom_path = cache_dir / "custom_params.safetensors"
    has_custom = custom_path.exists()

    if _stage_done(cache_dir, 3) and has_custom:
        logger.info("Reconstructing Tequila UltraQuantLinear layers...")
        from tequila.ultraquant import UltraQuantLinear, TequilaConfig
        custom = load_file(str(custom_path))
        config = TequilaConfig()
        _reconstruct_ultraquant(model, custom, config)

    elif _stage_done(cache_dir, 2) and has_custom:
        logger.info("Reconstructing ParetoQ QuantizeLinear layers...")
        from paretoq.utils_quant import QuantizeLinear
        custom = load_file(str(custom_path))
        _reconstruct_quantized(model, custom)

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        model = model.cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Cached ternary model loaded: {params:,} parameters (layers reconstructed)")
    return model, tokenizer


def _reconstruct_ultraquant(model: nn.Module, custom_params: dict, config) -> None:
    """Replace nn.Linear with UltraQuantLinear, load Lambada, then bake to nn.Linear.

    Baking pre-computes the effective ternary weight (A + deadzone contribution)
    so inference uses a single standard matmul instead of the dual-linear
    UltraQuantLinear forward (which is 2× slower on CPU).
    """
    from tequila.ultraquant import UltraQuantLinear, UltraQuantV3
    _replace_with_custom(model, custom_params, config, UltraQuantLinear, "Lambada")
    _bake_ultraquant_to_linear(model)


def _bake_ultraquant_to_linear(model: nn.Module) -> None:
    """Convert UltraQuantLinear layers to standard nn.Linear.

    The Tequila forward pass is:
        out = linear(input, A) + linear(ones, B * Lambada)

    The second term adds a constant bias per output channel (same for all
    tokens). To faithfully reproduce this in a standard nn.Linear:
        weight = A           (ternary, preserves sparsity)
        bias   = sum(B * Lambada, dim=-1)  (deadzone bias)

    The old formula (A + B*L) was incorrect — it turned the deadzone
    contribution into a full token-dependent matrix multiply, which
    neither matches the Tequila forward nor preserves ternary sparsity.
    """
    from tequila.ultraquant import UltraQuantLinear, UltraQuantV3
    skip_modules = ("lm_head", "embed_tokens")
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    baked = 0

    embed_dtype = model.get_input_embeddings().weight.dtype

    for module_name, module in model.named_modules():
        if any(skip in module_name for skip in skip_modules):
            continue
        if not any(target in module_name for target in target_modules):
            continue
        if not isinstance(module, UltraQuantLinear):
            continue

        with torch.no_grad():
            w = module.weight.data.float()
            A, B = UltraQuantV3.apply(w, module.granularity, module.group_size)

            # Weight = ternary A only (deadzone bias handled below)
            effective_weight = A.float()

            if hasattr(module, "Lambada"):
                # linear(ones, B * Lambada) = sum(B * Lambada, dim=-1) per channel
                deadzone_bias = (B.float() * module.Lambada.data.float()).sum(dim=-1)
            else:
                # v1/v2: linear(ones, B) → sum(B, dim=-1)
                deadzone_bias = B.float().sum(dim=-1)

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        device = module.weight.device
        new_linear = nn.Linear(
            module.in_features, module.out_features,
            bias=True, dtype=embed_dtype, device=device,
        )
        new_linear.weight.data.copy_(effective_weight.to(device))

        # Combine original bias + deadzone contribution
        if module.bias is not None:
            combined_bias = module.bias.data.float() + deadzone_bias
        else:
            combined_bias = deadzone_bias
        new_linear.bias.data.copy_(combined_bias.to(device))

        setattr(parent, child_name, new_linear)
        baked += 1

    logger.info(f"  Baked {baked} UltraQuantLinear → nn.Linear (dtype={embed_dtype})")


def _reconstruct_quantized(model: nn.Module, custom_params: dict) -> None:
    """Replace nn.Linear with QuantizeLinear and load weight_clip_val."""
    from paretoq.utils_quant import QuantizeLinear
    _replace_with_custom(model, custom_params, None, QuantizeLinear, "weight_clip_val")


def _replace_with_custom(model, custom_params, config, layer_cls, param_key):
    """Generic layer replacement: nn.Linear → custom layer, loading custom params."""
    skip_modules = ("lm_head", "embed_tokens")
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    replaced = 0
    cls_name = layer_cls.__name__

    for module_name, module in model.named_modules():
        if any(skip in module_name for skip in skip_modules):
            continue
        if not any(target in module_name for target in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        if cls_name == "UltraQuantLinear":
            new_layer = layer_cls(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                quant_method="ultraquantv3",
                granularity="per_channel",
                group_size=128,
                range_of_lambada=0.01,
                lambada_granularity=getattr(config, "lambada_granularity", "per_channel"),
            )
        else:
            new_layer = layer_cls(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                w_bits=1,
            )

        new_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_layer.bias.data.copy_(module.bias.data)

        # Load custom param if available
        full_key = f"{module_name}.{param_key}"
        if full_key in custom_params:
            getattr(new_layer, param_key).data.copy_(custom_params[full_key])

        setattr(parent, child_name, new_layer)
        replaced += 1

    logger.info(f"  Reconstructed {replaced} layers with {layer_cls.__name__}")


# ── Save with custom params ────────────────────────────────────────────

def _save_model_state(model, tokenizer, cache_dir: Path) -> None:
    """Save model as sharded safetensors to limit peak system RAM.

    Tensors are iterated one at a time from named_parameters() (no full
    state_dict in memory), converted to bf16, and saved in 1 GB shards.
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
    last_log = [0]  # mutable for progress tracking

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
    # Iterate tensors one at a time — no full state_dict in memory
    custom_params = {}
    for key, param in model.named_parameters():
        param_count += 1
        tensor = param.detach().to(torch.bfloat16).cpu()
        nbytes = tensor.numel() * tensor.element_size()

        # Custom params saved separately (small, < 10 MB total)
        if "weight_clip_val" in key or "Lambada" in key:
            custom_params[key] = tensor
            continue

        shard_batch[key] = tensor
        shard_bytes += nbytes
        weight_map[key] = None

        if shard_bytes >= target_bytes:
            _flush_shard()
            for fname, keys in shard_files[-1:]:
                for k in keys:
                    weight_map[k] = fname

        # Progress: log at 25%, 50%, 75%
        pct = 100 * param_count // total_params
        if pct >= last_log[0] + 25:
            logger.info(f"  Saving: {pct}% ({param_count}/{total_params} tensors)")
            last_log[0] = pct

    _flush_shard()
    for fname, keys in shard_files[-1:]:
        for k in keys:
            weight_map[k] = fname

    # Save or remove custom params
    cp_path = str(cache_dir / "custom_params.safetensors")
    if custom_params:
        save_file(custom_params, cp_path)
    elif os.path.exists(cp_path):
        os.remove(cp_path)  # Clean stale custom_params from previous failed runs

    # Fix the "XXXXX" placeholder in shard filenames
    total_shards = len(shard_files)
    import os as _os
    for old_name, keys in shard_files:
        new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
        if old_name != new_name:
            _os.rename(str(cache_dir / old_name), str(cache_dir / new_name))
        for k in keys:
            weight_map[k] = new_name

    # Write index file (HuggingFace-compatible sharded checkpoint)
    import json as _json
    total_params = sum(p.numel() for p in model.parameters())
    index = {
        "metadata": {"total_size": total_params * 2},  # bf16 = 2 bytes
        "weight_map": weight_map,
    }
    with open(str(cache_dir / "model.safetensors.index.json"), "w") as f:
        _json.dump(index, f, indent=2)


def _load_model_state(cache_dir: Path, entry: ModelEntry) -> tuple:
    """Load intermediate model state (for pipeline resume)."""
    _ensure_pipeline_imports()
    from safetensors.torch import load_file

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        str(cache_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(cache_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reconstruct layers that were applied in previous stages
    custom_path = cache_dir / "custom_params.safetensors"
    if custom_path.exists():
        custom = load_file(str(custom_path))
        if _stage_done(cache_dir, 3):
            from tequila.ultraquant import TequilaConfig
            _reconstruct_ultraquant(model, custom, TequilaConfig())
        elif _stage_done(cache_dir, 2):
            _reconstruct_quantized(model, custom)

    return model, tokenizer


# ── Compression pipeline ───────────────────────────────────────────────

def _compress_and_cache(
    entry: ModelEntry, cache_dir: Path, hf_cache: Path
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Run the full TernaryBoost pipeline with incremental checkpoints."""
    _ensure_pipeline_imports()

    from pt_bitnet import apply_pt_bitnet, PTBitNetConfig
    from shared.data import load_calibration_texts

    has_cuda = torch.cuda.is_available()
    logger.info(f"Device: {'GPU' if has_cuda else 'CPU'}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(entry.dtype, torch.bfloat16)

    # Load base or resume
    if _stage_done(cache_dir, 1):
        logger.info("Stage 1 checkpoint found — resuming from PT-BitNet output")
        model, tokenizer = _load_model_state(cache_dir, entry)
    else:
        logger.info("=" * 50)
        logger.info(f"Loading base model: {entry.path}")
        t0 = time.time()
        # Fix: newer transformers requires pad_token_id on config
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

    # Calibration data (cached after first load, used by all stages)
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
        logger.info("Stage 1/3: PT-BitNet post-training quantization")
        t0 = time.time()
        comp_steps = 50 if has_cuda else 10
        model = apply_pt_bitnet(
            model,
            PTBitNetConfig(block_size=128, outlier_clip_threshold=3.0,
                           outlier_fraction=0.01, compensation_steps=comp_steps,
                           asymmetric=False),  # Symmetric: ITF incompatible with outliers
            tokenizer=tokenizer,
            calibration_texts=_ensure_texts()[:32],
        )
        elapsed = time.time() - t0
        logger.info(f"PT-BitNet complete in {elapsed:.1f}s")
        _save_model_state(model, tokenizer, cache_dir)
        _mark_stage_done(cache_dir, 1, elapsed)
    else:
        logger.info("Stage 1/3: PT-BitNet [SKIPPED]")

    # Stage 2: Intentionally skipped — Tequila provides ternary-aware QAT (see README)
    # ParetoQ code is retained in paretoq/ as a research artifact.
    # QuantizeLinear lacks a native ternary mode: w_bits=1 is binary {-1, +1}
    # and w_bits=0 uses StretchedElasticQuant which scales weights by 0.667,
    # both incompatible with PT-BitNet's {-α, 0, +α} output.
    if not _stage_done(cache_dir, 2):
        logger.info("Stage 2/3: ParetoQ/ZeroQAT [SKIPPED — Tequila provides ternary-aware optimization]")
        _mark_stage_done(cache_dir, 2, 0)
    else:
        logger.info("Stage 2/3: ParetoQ/ZeroQAT [SKIPPED]")
        model, tokenizer = _load_model_state(cache_dir, entry)

    # Stage 3: Tequila intentionally skipped.
    # PT-BitNet denormalizes weights (w*std + mean), shifting them away from
    # zero. UltraQuantV3 recomputes the ternary decomposition from scratch
    # using its own threshold (mean(|w|)/2), which produces a DIFFERENT
    # ternary pattern than PT-BitNet. Combined with the incorrect baking
    # formula that was present in earlier versions, this destroyed quality.
    # PT-BitNet sym+comp alone is sufficient — ablation proves it matches
    # FP16 output. Tequila code is retained in tequila/ as a research
    # artifact for future quantization-aware training.
    if not _stage_done(cache_dir, 3):
        logger.info("Stage 3/3: Tequila [SKIPPED — PT-BitNet denorm incompatible with UltraQuantV3 re-decomposition]")
        _mark_stage_done(cache_dir, 3, 0)
    else:
        logger.info("Stage 3/3: Tequila [SKIPPED]")
        model, tokenizer = _load_model_state(cache_dir, entry)

    # Stage 4: LoRA fine-tuning (quality recovery)
    # Adds small trainable rank-decomposition adapters on top of frozen
    # ternary weights. Fine-tuned with knowledge distillation from the
    # FP16 teacher. LoRA weights are saved separately — the ternary
    # backbone stays untouched for fast inference.
    # Set lora_rank=0 in ModelEntry to skip.
    lora_rank = getattr(entry, "lora_rank", 0)
    if lora_rank > 0 and not _stage_done(cache_dir, 4):
        logger.info("=" * 50)
        logger.info(f"Stage 4/4: LoRA fine-tuning (rank={lora_rank}, quality recovery)")
        t0 = time.time()

        # Check VRAM: teacher and student alternate on GPU (not simultaneous)
        # Peak: teacher alone (~6 GB for Phi-2) or student+LoRA (~6 GB)
        if has_cuda:
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            est_model = sum(p.numel() for p in model.parameters()) * 2 / 1e9  # bf16
            if vram_total < est_model * 1.8:  # Need ~1.8x model size for safe operation
                logger.warning(f"  VRAM tight for LoRA (GPU={vram_total:.1f} GB, "
                               f"model={est_model:.1f} GB). Attempting anyway...")

            from pt_bitnet.lora import finetune_lora, LoRAConfig

            logger.info("  Loading FP16 teacher on CPU for logit pre-computation...")
            from transformers import AutoConfig
            teacher_config = AutoConfig.from_pretrained(
                entry.path, trust_remote_code=True, cache_dir=str(hf_cache),
            )
            if not hasattr(teacher_config, "pad_token_id") or teacher_config.pad_token_id is None:
                teacher_config.pad_token_id = 0
            # Teacher stays on CPU — avoids GPU OOM. CPU forward is ~0.5s per text.
            teacher = AutoModelForCausalLM.from_pretrained(
                entry.path, torch_dtype=dtype, low_cpu_mem_usage=True,
                device_map="cpu", trust_remote_code=True, cache_dir=str(hf_cache),
                config=teacher_config,
            )

            texts_data = _ensure_texts()
            lo_cfg = LoRAConfig(
                rank=lora_rank,
                num_steps=getattr(entry, "lora_steps", 500),
                distill_weight=0.5,
                max_seq_length=64,  # Shorter seq = less VRAM for activations
                batch_size=1,       # Batch 1 = safer memory
                gradient_accumulation=8,  # More accumulation = same effective batch
            )
            # finetune_lora: pre-computes teacher logits → frees teacher →
            # fine-tunes with student only. Peak VRAM: one model at a time.
            model = finetune_lora(model, tokenizer, texts_data[:50], teacher, lo_cfg)

            from pt_bitnet.lora import save_lora_weights
            save_lora_weights(model, str(cache_dir / "lora_weights.safetensors"))

            elapsed = time.time() - t0
            logger.info(f"LoRA fine-tuning complete in {elapsed:.1f}s")
            _mark_stage_done(cache_dir, 4, elapsed)
        else:
            logger.info("  LoRA requires GPU — skipping on CPU")
            _mark_stage_done(cache_dir, 4, 0)
    elif lora_rank == 0:
        logger.info("Stage 4/4: LoRA [SKIPPED — lora_rank=0]")
        if not _stage_done(cache_dir, 4):
            _mark_stage_done(cache_dir, 4, 0)
    else:
        logger.info("Stage 4/4: LoRA [SKIPPED]")
        # Try loading LoRA weights if they exist
        lora_path = cache_dir / "lora_weights.safetensors"
        if lora_path.exists() and not any("lora" in n for n, _ in model.named_parameters()):
            from pt_bitnet.lora import LoRAConfig, _add_lora_to_model, load_lora_weights
            _add_lora_to_model(model, LoRAConfig())
            load_lora_weights(model, str(lora_path))
            logger.info("  Loaded LoRA weights from cache")
        else:
            model, tokenizer = _load_model_state(cache_dir, entry)

    # Final metadata
    total_time = time.time() - total_start
    meta = {
        "source_model": entry.path, "pipeline_version": "1.0",
        "total_time_s": total_time,
        "stages_applied": ["pt_bitnet", "lora"],
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
        stages_done = sum(1 for s in (1, 2, 3) if (d / f"stage{s}_done.txt").exists())
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
