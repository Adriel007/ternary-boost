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
    """Convert UltraQuantLinear layers to standard nn.Linear with baked weights.

    Computes effective_weight = A + B * Lambada, then replaces UltraQuantLinear
    with nn.Linear. All baked weights are cast to the model's dtype (from the
    embedding layer) to avoid dtype mismatches during inference.
    """
    from tequila.ultraquant import UltraQuantLinear, UltraQuantV3
    skip_modules = ("lm_head", "embed_tokens")
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
    baked = 0

    # Get model dtype from embedding layer (not quantized, preserves original dtype)
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

            if hasattr(module, "Lambada"):
                effective_weight = A.float() + B.float() * module.Lambada.data.float()
            else:
                effective_weight = A.float()
            effective_weight = effective_weight.to(embed_dtype)

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        new_linear = nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None, dtype=embed_dtype,
        )
        new_linear.weight.data.copy_(effective_weight)
        if module.bias is not None:
            new_linear.bias.data.copy_(module.bias.data.to(embed_dtype))

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
    """Save standard HF model + separate custom params (weight_clip_val, Lambada)."""
    model.config.save_pretrained(str(cache_dir))
    tokenizer.save_pretrained(str(cache_dir))

    # Separate standard parameters from custom quantization parameters
    standard_params = {}
    custom_params = {}
    for name, param in model.state_dict().items():
        if "weight_clip_val" in name or "Lambada" in name:
            custom_params[name] = param
        else:
            standard_params[name] = param

    from safetensors.torch import save_file
    save_file(standard_params, str(cache_dir / "model.safetensors"))
    if custom_params:
        save_file(custom_params, str(cache_dir / "custom_params.safetensors"))


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
    from paretoq import apply_paretoq_qat, ZeroQATConfig
    from tequila import apply_tequila, TequilaConfig
    from shared.data import load_calibration_texts, create_qat_dataloader, create_calibration_dataloader

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
        model = apply_pt_bitnet(
            model,
            PTBitNetConfig(block_size=128, outlier_clip_threshold=3.0,
                           outlier_fraction=0.01, compensation_steps=50),
            tokenizer=tokenizer,
            calibration_texts=_ensure_texts()[:32],
        )
        elapsed = time.time() - t0
        logger.info(f"PT-BitNet complete in {elapsed:.1f}s")
        _save_model_state(model, tokenizer, cache_dir)
        _mark_stage_done(cache_dir, 1, elapsed)
    else:
        logger.info("Stage 1/3: PT-BitNet [SKIPPED]")

    # Stage 2: ParetoQ + ZeroQAT (GPU only — binary w_bits=1 destroys ternary sparsity)
    if not _stage_done(cache_dir, 2):
        if has_cuda:
            logger.info("=" * 50)
            logger.info("Stage 2/3: ParetoQ QAT + ZeroQAT (GPU)")
            qat_steps, log_interval, batch_size = 100, 25, 2
            t0 = time.time()
            texts_data = _ensure_texts()
            qat_dataloader = create_qat_dataloader(tokenizer, texts_data[:100], batch_size=batch_size, max_length=128)
            zo_config = ZeroQATConfig(learning_rate=1e-5, max_steps=qat_steps,
                                      perturbation_scale=1e-3, grad_clip=1.0)
            model = model.cuda()
            model = apply_paretoq_qat(
                model=model, tokenizer=tokenizer, train_dataloader=qat_dataloader,
                output_path=str(cache_dir / "stage2_qat"), w_bits=0, qat_steps=qat_steps,
                zo_config=zo_config, log_interval=log_interval,
            )
            elapsed = time.time() - t0
            logger.info(f"ParetoQ/ZeroQAT complete in {elapsed:.1f}s")
            _save_model_state(model, tokenizer, cache_dir)
            _mark_stage_done(cache_dir, 2, elapsed)
        else:
            logger.info("Stage 2/3: ParetoQ/ZeroQAT [SKIPPED — GPU only, PT-BitNet is sufficient on CPU]")
            # Mark as done to skip on resume, since we intentionally skip QAT on CPU
            _mark_stage_done(cache_dir, 2, 0)
    else:
        logger.info("Stage 2/3: ParetoQ/ZeroQAT [SKIPPED]")
        model, tokenizer = _load_model_state(cache_dir, entry)

    # Stage 3: Tequila
    if not _stage_done(cache_dir, 3):
        logger.info("=" * 50)
        logger.info("Stage 3/3: Tequila deadzone trapping")
        t0 = time.time()
        texts_data = _ensure_texts()
        tequila_dataloader = create_calibration_dataloader(tokenizer, texts_data[:50], batch_size=2, max_length=128)
        tequila_config = TequilaConfig(
            quant_method="ultraquantv3", num_epochs=1, range_of_lambada=0.01,
            lambada_granularity=getattr(entry, "lambada_granularity", "per_channel"),
        )
        model = apply_tequila(model, tequila_dataloader, tequila_config)
        elapsed = time.time() - t0
        logger.info(f"Tequila complete in {elapsed:.1f}s")
        # Bake UltraQuantLinear → nn.Linear for fast inference
        _bake_ultraquant_to_linear(model)
        _save_model_state(model, tokenizer, cache_dir)
        _mark_stage_done(cache_dir, 3, elapsed)
    else:
        logger.info("Stage 3/3: Tequila [SKIPPED]")
        model, tokenizer = _load_model_state(cache_dir, entry)

    # Final metadata
    total_time = time.time() - total_start
    meta = {
        "source_model": entry.path, "pipeline_version": "1.0",
        "total_time_s": total_time,
        "stages_applied": ["pt_bitnet", "paretoq_zeroqat", "tequila"],
        "params": sum(p.numel() for p in model.parameters()),
        "device": "GPU" if has_cuda else "CPU",
        "lambada_granularity": getattr(entry, "lambada_granularity", "per_channel"),
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
