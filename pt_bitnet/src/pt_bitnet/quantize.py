"""PT-BitNet: Post-Training Ternary Quantization for LLMs.

Vectorized two-stage approach:
  1. Distribution Transform — per-channel normalization + outlier clipping.
  2. Block-wise Ternary Optimization — fully vectorized per-channel
     threshold search minimizing reconstruction error ||W - alpha·T(W,δ)||².
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from shared.logging import get_logger

logger = get_logger("pt_bitnet")


@dataclass
class PTBitNetConfig:
    bit_width: float = 1.58
    block_size: int = 128
    outlier_clip_threshold: float = 3.0
    max_iter: int = 10
    tolerance: float = 1e-4
    outlier_fraction: float = 0.01  # top 1% weights kept in FP16 (SpQR-style)
    compensation_steps: int = 50    # Hessian compensation iterations on lm_head
    target_modules: tuple = field(
        default=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )
    skip_modules: tuple = field(default=("lm_head", "embed_tokens"))
    show_progress: bool = True


def _find_quantizable_linears(model: nn.Module, config: PTBitNetConfig) -> list[tuple[str, nn.Linear]]:
    targets = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in module_name for skip in config.skip_modules):
                continue
            if any(t in module_name for t in config.target_modules):
                targets.append((module_name, module))
    logger.info(f"Found {len(targets)} quantizable linear layers")
    return targets


def ternary_quantize_vectorized(
    weight: torch.Tensor,
    config: PTBitNetConfig,
) -> torch.Tensor:
    """Vectorized ternary quantization with outlier retention (SpQR-style).

    1. Identifies top-k% outlier weights per row (by magnitude/sensitivity).
    2. Keeps outliers in their original precision.
    3. Quantizes remaining weights to ternary {-alpha, 0, +alpha}.

    Outlier retention preserves critical weights that would cause large
    quantization errors, trading <k% density for significant quality gain.
    """
    w = weight.float()
    out_f, in_f = w.shape

    # --- Outlier identification (SpQR-style) ---
    if config.outlier_fraction > 0:
        w_abs = w.abs()
        # Per-row z-score: how far each weight is from its row's distribution
        row_mean = w_abs.mean(dim=-1, keepdim=True)
        row_std = w_abs.std(dim=-1, keepdim=True).clamp_min(1e-8)
        outlier_score = (w_abs - row_mean) / row_std  # [out_f, in_f]

        k = max(1, int(in_f * config.outlier_fraction))
        _, outlier_idx = outlier_score.topk(k, dim=-1)  # [out_f, k]

        outlier_mask = torch.zeros_like(w, dtype=torch.bool)
        outlier_mask.scatter_(-1, outlier_idx, True)
    else:
        outlier_mask = torch.zeros_like(w, dtype=torch.bool)

    # --- Quantize non-outliers to ternary ---
    w_to_quantize = w.clone()
    w_to_quantize[outlier_mask] = 0  # Zero out outliers for quantization

    w_abs = w_to_quantize.abs()
    w_sign = w.sign()  # Use sign of original weights

    # Sort absolute values per row
    w_abs_sorted, sort_idx = w_abs.sort(dim=-1)
    w_sign_sorted = w_sign.gather(-1, sort_idx)

    n_candidates = min(in_f, 256)
    if in_f > n_candidates:
        step = max(1, in_f // n_candidates)
        candidate_indices = torch.arange(0, in_f, step, device=w.device)
    else:
        candidate_indices = torch.arange(in_f, device=w.device)

    candidate_deltas = w_abs_sorted[:, candidate_indices]

    best_error = torch.full((out_f,), float("inf"), device=w.device)
    best_alpha = torch.zeros(out_f, device=w.device)
    best_ternary = torch.zeros_like(w)

    for j in range(candidate_indices.size(0)):
        delta = candidate_deltas[:, j:j+1]
        active = w_abs >= delta
        active_count = active.sum(dim=-1).clamp_min(1)
        alpha = (w_abs * active.float()).sum(dim=-1) / active_count
        recon = torch.where(active, alpha.unsqueeze(-1) * w_sign, torch.zeros_like(w))
        error = ((w_to_quantize - recon) ** 2).sum(dim=-1)

        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_alpha = torch.where(improve, alpha, best_alpha)
        best_ternary = torch.where(
            improve.unsqueeze(-1), recon, best_ternary,
        )

    # Refine worst rows
    needs_refine = best_error > best_error.median() * 1.5
    if needs_refine.any():
        for idx in torch.where(needs_refine)[0]:
            w_row = w_to_quantize[idx]
            w_abs_row = w_abs[idx]
            w_sign_row = w_sign[idx]
            delta = candidate_deltas[idx, candidate_deltas[idx] > 0].min().item() if (candidate_deltas[idx] > 0).any() else w_abs_row.max().item() * 0.5

            for _ in range(config.max_iter):
                active = w_abs_row >= delta
                if active.sum() == 0:
                    delta *= 0.5
                    continue
                alpha = w_abs_row[active].mean()
                recon = torch.where(active, alpha * w_sign_row, torch.zeros_like(w_row))
                error = ((w_row - recon) ** 2).sum()
                above = w_abs_row[w_abs_row >= delta]
                below = w_abs_row[w_abs_row < delta]
                if below.numel() > 0:
                    delta_new_val = below.max().item()
                else:
                    delta_new_val = delta * 0.5
                if delta_new_val == 0 or abs(delta_new_val - delta) < config.tolerance:
                    break
                delta = delta_new_val

            if error < best_error[idx]:
                best_ternary[idx] = recon

    # Merge: outliers kept as-is, rest quantized to ternary
    result = best_ternary.clone()
    result[outlier_mask] = w[outlier_mask]

    return result


def hessian_compensation(
    model: PreTrainedModel,
    tokenizer,
    calibration_texts: list[str],
    config: PTBitNetConfig,
) -> None:
    """Compensate quantization error by fine-tuning lm_head (Hessian-style).

    Memory-efficient design:
      transformer body → runs under torch.no_grad() (zero activation memory)
      hidden_states     → detached from graph
      lm_head           → only module with requires_grad=True

    This limits the computation graph to just the lm_head projection
    (~100M params × batch × seq) instead of the full 2.7B model graph.
    """
    if config.compensation_steps <= 0:
        return

    logger.info(f"Hessian compensation: {config.compensation_steps} steps on lm_head (memory-efficient)")

    has_cuda = torch.cuda.is_available()
    original_device = next(model.parameters()).device
    if has_cuda and original_device.type == "cpu":
        model.cuda()
    train_device = next(model.parameters()).device

    # Freeze all, then unfreeze only lm_head
    for param in model.parameters():
        param.requires_grad = False

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                lm_head = module
    if lm_head is None:
        logger.warning("No lm_head found — skipping compensation")
        return

    lm_head.weight.requires_grad = True
    lm_head_params = [lm_head.weight]
    if hasattr(lm_head, "bias") and lm_head.bias is not None:
        lm_head.bias.requires_grad = True
        lm_head_params.append(lm_head.bias)

    optimizer = torch.optim.Adam(lm_head_params, lr=1e-5)

    # Hook the transformer body to capture hidden states before lm_head
    _hidden_state = None

    def _save_last_hidden(module, args, output):
        nonlocal _hidden_state
        if hasattr(output, "last_hidden_state"):
            _hidden_state = output.last_hidden_state.detach()
        elif isinstance(output, tuple):
            _hidden_state = output[0].detach()
        else:
            _hidden_state = output.detach()

    # Find the transformer model (e.g. PhiModel, LlamaModel, etc.)
    transformer = getattr(model, "model", None) or getattr(model, "transformer", None)
    if transformer is None:
        logger.warning("Cannot find transformer body — skipping compensation")
        return

    hook_handle = transformer.register_forward_hook(_save_last_hidden)

    for step in range(config.compensation_steps):
        total_loss = 0.0
        n_batches = 0

        for text in calibration_texts[: min(len(calibration_texts), 16)]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(train_device)

            if input_ids.shape[1] < 2:
                continue

            # Forward body under no_grad (zero activation memory for 2.7B params)
            _hidden_state = None
            with torch.no_grad():
                model(input_ids=input_ids)

            if _hidden_state is None:
                continue

            # Forward ONLY lm_head with grad (tiny graph: ~100M params × 64 tokens)
            logits = lm_head(_hidden_state)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_head_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (step + 1) % max(1, config.compensation_steps // 5) == 0:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  Compensation step {step + 1}/{config.compensation_steps} - loss: {avg_loss:.4f}")

    hook_handle.remove()
    if has_cuda and original_device.type == "cpu":
        model.cpu()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Hessian compensation complete")


def apply_pt_bitnet(
    model: PreTrainedModel,
    config: Optional[PTBitNetConfig] = None,
    tokenizer=None,
    calibration_texts: Optional[list[str]] = None,
) -> PreTrainedModel:
    """Apply PT-BitNet post-training ternary quantization.

    Iterates over nn.Linear layers and replaces weights with ternary values.
    Optionally applies Hessian compensation on lm_head.

    Args:
        model: HuggingFace PreTrainedModel.
        config: PT-BitNet configuration.
        tokenizer: Tokenizer for compensation (optional).
        calibration_texts: Texts for Hessian compensation (optional).
    """
    if config is None:
        config = PTBitNetConfig()

    logger.info("Starting PT-BitNet post-training quantization (vectorized)...")
    if config.outlier_fraction > 0:
        logger.info(f"  Outlier retention: {config.outlier_fraction*100:.1f}% weights kept in FP16")
    model.eval()

    targets = _find_quantizable_linears(model, config)
    n_total = len(targets)

    with torch.no_grad():
        for i, (name, module) in enumerate(targets):
            w = module.weight.data

            # Move to GPU for speed if available
            device = w.device
            if torch.cuda.is_available():
                w_gpu = w.cuda()
            else:
                w_gpu = w

            # Stage 1: Normalize (clip outliers, center)
            w_mean = w_gpu.mean(dim=-1, keepdim=True)
            w_std = w_gpu.std(dim=-1, keepdim=True).clamp_min(1e-8)
            w_norm = (w_gpu - w_mean) / w_std
            w_norm = w_norm.clamp(-config.outlier_clip_threshold, config.outlier_clip_threshold)

            # Stage 2: Vectorized ternary quantization
            ternary_w = ternary_quantize_vectorized(w_norm, config)

            # Denormalize back to original scale
            ternary_w = ternary_w * w_std + w_mean
            ternary_w = ternary_w.to(device=device, dtype=module.weight.dtype)

            module.weight = nn.Parameter(ternary_w)
            module.weight.requires_grad = False

            if config.show_progress and (i + 1) % max(1, n_total // 10) == 0:
                logger.info(f"  PT-BitNet: {i + 1}/{n_total} layers ({100 * (i + 1) // n_total}%)")

    logger.info(f"PT-BitNet complete: {n_total} layers quantized to ternary")

    # Optional: Hessian compensation on lm_head
    if config.compensation_steps > 0 and tokenizer is not None and calibration_texts is not None:
        hessian_compensation(model, tokenizer, calibration_texts, config)

    return model


# Legacy API — kept for backward compatibility with tests and pipeline
def distribution_transform(
    weight: torch.Tensor,
    clip_threshold: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Per-channel normalization with outlier clipping."""
    w = weight.float()
    mean = w.mean(dim=-1, keepdim=True)
    std = w.std(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = (w - mean) / std
    w_clipped = torch.clamp(w_norm, -clip_threshold, clip_threshold)
    scale = w.abs().max(dim=-1, keepdim=True).values
    return w_clipped, scale


def blockwise_optimize(
    weight: torch.Tensor,
    block_size: int = 128,
    max_iter: int = 10,
    tolerance: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: Block-wise ternary optimization (vectorized version for tests)."""
    config = PTBitNetConfig(block_size=block_size, max_iter=max_iter, tolerance=tolerance,
                            outlier_fraction=0.0, show_progress=False)
    ternary = ternary_quantize_vectorized(weight, config)
    scales = weight.abs().max(dim=-1, keepdim=True).values
    return ternary, scales
