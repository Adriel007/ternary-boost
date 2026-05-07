"""Export and load pipeline for separate ternary + LoRA inference format.

Export flow:
  1. Iterate model layers. For each LoRALinear / ternary nn.Linear:
     - Extract ternary mask T = sign(W) → pack to INT2
     - Compute alpha (and mu for asymmetric)
     - Extract LoRA A, B if present
  2. Save INT2 + alpha + mu → ternary_params.pt
  3. Save LoRA weights → lora_weights.safetensors
  4. Save unquantized layers (embed, lm_head, norms) → base_model.safetensors
  5. Save config.json with "ternary_lora" marker

Load flow:
  1. Load config.json → create skeleton HF model
  2. Replace target Linear layers with TernaryInferenceLinear
  3. Load ternary_params.pt → populate INT2 + alpha + mu buffers
  4. Load lora_weights.safetensors → populate lora_A/B buffers
  5. Load base_model.safetensors → populate embed, lm_head, norms
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pt_bitnet.int2_packing import pack_int2, unpack_int2
from shared.logging import get_logger

logger = get_logger("pt_bitnet.export")


# Modules that should NOT be ternarized
_SKIP_MODULES = ("lm_head", "embed_tokens")

# Target modules for ternary quantization
_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def export_ternary_lora(
    model: nn.Module,
    tokenizer,
    output_dir: str,
    target_modules: tuple = _TARGET_MODULES,
    skip_modules: tuple = _SKIP_MODULES,
) -> None:
    """Export a model with LoRALinear / ternary layers to the separate format.

    Args:
        model: model with LoRALinear or ternary nn.Linear layers.
        tokenizer: HF tokenizer to save alongside the model.
        output_dir: directory to save exported files.
        target_modules: layer name patterns to export as ternary.
        skip_modules: layer name patterns to keep as FP16.
    """
    from pt_bitnet.lora import LoRALinear

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ternary_params = {}   # layer_name → {int2_packed, alpha, mu, bias, ...}
    lora_weights = {}     # layer_name → {lora_A, lora_B, lora_scale}
    base_weights = {}     # unquantized layer weights

    ternary_count = 0
    lora_count = 0
    skipped_count = 0

    for module_name, module in model.named_modules():
        # Skip children of LoRALinear — already handled by their parent
        if ".base" in module_name.split("."):
            continue

        # Skip modules
        if any(s in module_name for s in skip_modules):
            if isinstance(module, nn.Linear):
                _save_linear_state(base_weights, module_name, module)
                skipped_count += 1
            elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                base_weights[f"{module_name}.weight"] = module.weight.data.clone().cpu()
                if hasattr(module, "bias") and module.bias is not None:
                    base_weights[f"{module_name}.bias"] = module.bias.data.clone().cpu()
            continue

        # Target modules
        is_target = any(t in module_name for t in target_modules)
        if not is_target:
            continue

        if isinstance(module, LoRALinear):
            _export_lora_linear(ternary_params, lora_weights, module_name, module)
            ternary_count += 1
            lora_count += 1
        elif isinstance(module, nn.Linear) and not module.weight.requires_grad:
            _export_ternary_linear(ternary_params, module_name, module)
            ternary_count += 1

    # Also capture LayerNorm and other non-Linear params
    for name, param in model.named_parameters():
        if any(f"{s}." in name or name.endswith(f".{s}") or name == s
               for s in skip_modules):
            continue
        is_ln = "norm" in name.lower() or "layernorm" in name.lower() or "rmsnorm" in name.lower()
        if is_ln and name not in base_weights:
            base_weights[name] = param.data.clone().cpu()

    logger.info(f"Export: {ternary_count} ternary layers ({lora_count} with LoRA), "
                f"{skipped_count} unquantized layers")

    # ── Save ───────────────────────────────────────────────────────
    # 1. Full model weights as standard safetensors (HF compatibility)
    model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    from safetensors.torch import save_file
    save_file(model_state, output_dir / "model.safetensors")
    logger.info(f"  Saved model.safetensors ({len(model_state)} tensors)")

    # 2. Ternary params (INT2 packed + alphas)
    torch.save(ternary_params, output_dir / "ternary_params.pt")
    logger.info(f"  Saved ternary_params.pt ({len(ternary_params)} layers)")

    # 3. LoRA weights (separate safetensors)
    if lora_weights:
        save_file(lora_weights, output_dir / "lora_weights.safetensors")
        logger.info(f"  Saved lora_weights.safetensors ({len(lora_weights)} tensors)")

    # 4. Config
    _save_config(model, output_dir, tokenizer)

    logger.info(f"Export complete → {output_dir}")


def load_ternary_lora(
    path: str,
    device: str = "cpu",
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = True,
) -> nn.Module:
    """Load a ternary+LoRA model from an exported directory.

    Args:
        path: directory containing config.json, ternary_params.pt,
              lora_weights.safetensors, and base_model.safetensors.
        device: device to load the model on.
        torch_dtype: dtype for non-ternary parameters.
        trust_remote_code: passed to HF from_pretrained.

    Returns:
        Model with TernaryInferenceLinear layers, ready for inference.
    """
    path = Path(path)

    # 1. Load config and model via standard HF from_pretrained
    #    model.safetensors contains ALL weights (saved during export).
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0

    logger.info("Loading full model from model.safetensors...")
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch_dtype, device_map=device,
        trust_remote_code=trust_remote_code, config=config,
    )

    # 2. Load ternary params and replace Linear with TernaryInferenceLinear
    ternary_params_path = path / "ternary_params.pt"
    if ternary_params_path.exists():
        ternary_params = torch.load(ternary_params_path, map_location=device,
                                     weights_only=True)
        logger.info(f"Loaded ternary params for {len(ternary_params)} layers")

        lora_weights = {}
        lora_path = path / "lora_weights.safetensors"
        if lora_path.exists():
            from safetensors.torch import load_file
            lora_weights = load_file(str(lora_path))
            logger.info(f"Loaded {len(lora_weights)} LoRA tensors")

        _inject_ternary_layers(model, ternary_params, lora_weights, device)
    else:
        logger.warning(f"No ternary_params.pt found in {path} — using FP16 weights")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_ternary_params(w: torch.Tensor) -> dict:
    """Extract ternary mask, alpha, mu from a weight matrix.

    Args:
        w: [out_f, in_f] float tensor (ternary values).

    Returns:
        dict with T, alpha, mu, asymmetric keys.
    """
    out_f = w.shape[0]
    w_float = w.float()

    # ── Initial T estimate via sign ──────────────────────────────
    # For symmetric ternary (|mu| << alpha), sign(w) ≈ T for active weights.
    # For zero weights where w ≈ mu ≠ 0, sign(w) misclassifies them.
    # We refine in the next step.
    T_init = torch.sign(w_float)
    nonzero_init = T_init.abs()
    row_sum = (w_float.abs() * nonzero_init).sum(dim=-1)
    row_count = nonzero_init.sum(dim=-1).clamp_min(1)
    alpha_init = (row_sum / row_count).unsqueeze(-1)  # [out_f, 1]
    mu_init = w_float.mean(dim=-1, keepdim=True)      # [out_f, 1]

    # Rows with near-zero alpha have all weights ≈ mu → T = 0 everywhere
    degenerate = alpha_init.squeeze(-1) < 1e-6  # [out_f]

    # ── One-pass refinement: assign each weight to closest center ──
    # The three candidate centers per row: {-alpha_i + mu_i, mu_i, +alpha_i + mu_i}
    c_neg = -alpha_init + mu_init   # [out_f, 1]
    c_zero = mu_init                # [out_f, 1]
    c_pos = alpha_init + mu_init    # [out_f, 1]

    w_exp = w_float.unsqueeze(-1)   # [out_f, in_f, 1]
    centers = torch.stack([c_neg, c_zero, c_pos], dim=-1)  # [out_f, 1, 3]
    dists = (w_exp - centers).abs()  # [out_f, in_f, 3]
    T_idx = dists.argmin(dim=-1)     # [out_f, in_f] ∈ {0, 1, 2}
    T = (T_idx.float() - 1.0)        # 0→-1, 1→0, 2→+1
    # Force degenerate rows (alpha ≈ 0) to all zeros
    T[degenerate, :] = 0.0

    # ── Recompute alpha from refined T ─────────────────────────────
    nonzero = T.abs()
    row_sum = (w_float.abs() * nonzero).sum(dim=-1)
    row_count_refined = nonzero.sum(dim=-1).clamp_min(1)
    alpha = (row_sum / row_count_refined).unsqueeze(-1)

    # ── Detect symmetry ───────────────────────────────────────────
    mu_val = w_float.mean(dim=-1, keepdim=True)
    sym_err = (w_float - alpha * T).abs().mean()
    asym_err = (w_float - alpha * T - mu_val).abs().mean()

    if asym_err < sym_err * 0.99:
        asymmetric = True
        mu = mu_val
    else:
        asymmetric = False
        mu = None

    return {
        "T": T.cpu(),
        "alpha": alpha.cpu(),
        "mu": mu.cpu() if asymmetric else None,
        "asymmetric": asymmetric,
    }


def _export_lora_linear(ternary_params, lora_weights, name, module):
    """Export a LoRALinear module to separate ternary + LoRA format."""
    base_w = module.base.weight.data
    out_f, in_f = base_w.shape

    params = _extract_ternary_params(base_w)
    int2_packed = pack_int2(params["T"])

    ternary_params[name] = {
        "int2_packed": int2_packed.cpu(),
        "alpha": params["alpha"],
        "mu": params["mu"],
        "bias": module.base.bias.data.clone().cpu() if module.base.bias is not None else None,
        "in_features": in_f,
        "out_features": out_f,
        "asymmetric": params["asymmetric"],
        "has_lora": True,
        "lora_scale": module.scaling,
    }

    lora_weights[f"{name}.lora_A"] = module.lora_A.data.clone().cpu()
    lora_weights[f"{name}.lora_B"] = module.lora_B.data.clone().cpu()


def _export_ternary_linear(ternary_params, name, module):
    """Export a plain ternary nn.Linear (no LoRA)."""
    w = module.weight.data
    out_f, in_f = w.shape

    params = _extract_ternary_params(w)
    int2_packed = pack_int2(params["T"])

    ternary_params[name] = {
        "int2_packed": int2_packed.cpu(),
        "alpha": params["alpha"],
        "mu": params["mu"],
        "bias": module.bias.data.clone().cpu() if module.bias is not None else None,
        "in_features": in_f,
        "out_features": out_f,
        "asymmetric": params["asymmetric"],
        "has_lora": False,
        "lora_scale": 0.0,
    }


def _save_linear_state(base_weights, name, module):
    """Save a standard nn.Linear's state to base_weights dict."""
    base_weights[f"{name}.weight"] = module.weight.data.clone().cpu()
    if module.bias is not None:
        base_weights[f"{name}.bias"] = module.bias.data.clone().cpu()


def _save_config(model, output_dir, tokenizer):
    """Save config.json with ternary_lora marker and tokenizer."""
    # Copy the model's original config and add our marker
    if hasattr(model, "config"):
        config_dict = model.config.to_dict()
    elif hasattr(model, "model") and hasattr(model.model, "config"):
        config_dict = model.model.config.to_dict()
    else:
        config_dict = {}

    config_dict["ternary_lora"] = True
    config_dict["model_type"] = config_dict.get("model_type", "phi")
    if "architectures" not in config_dict:
        config_dict["architectures"] = [type(model).__name__]

    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))


def _inject_ternary_layers(model, ternary_params, lora_weights, device):
    """Replace nn.Linear layers with TernaryInferenceLinear in-place."""
    from pt_bitnet.ternary_linear import TernaryInferenceLinear

    replaced = 0
    for module_name, module in list(model.named_modules()):
        if module_name not in ternary_params:
            continue

        tp = ternary_params[module_name]
        out_f = tp["out_features"]
        in_f = tp["in_features"]

        # Build TernaryInferenceLinear
        lora_A = lora_B = None
        lora_scale = 1.0
        if tp.get("has_lora"):
            lora_A_key = f"{module_name}.lora_A"
            lora_B_key = f"{module_name}.lora_B"
            if lora_A_key in lora_weights and lora_B_key in lora_weights:
                lora_A = lora_weights[lora_A_key].to(device)
                lora_B = lora_weights[lora_B_key].to(device)
                lora_scale = tp.get("lora_scale", 1.0)

        new_layer = TernaryInferenceLinear(
            int2_packed=tp["int2_packed"].to(device),
            alpha=tp["alpha"].to(device),
            in_features=in_f,
            out_features=out_f,
            mu=tp["mu"].to(device) if tp.get("mu") is not None else None,
            bias=tp["bias"].to(device) if tp.get("bias") is not None else None,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_scale=lora_scale,
        )

        # Find parent and replace
        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)
        setattr(parent, child_name, new_layer)
        replaced += 1

    logger.info(f"Injected {replaced} TernaryInferenceLinear layers")
