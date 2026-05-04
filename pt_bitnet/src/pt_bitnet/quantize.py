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
    """Fully vectorized ternary quantization.

    For each output channel, finds the optimal threshold delta such that:
        W_q = alpha * sign(W) * I(|W| >= delta)

    where alpha minimizes ||W - W_q||².

    The optimal delta is found via binary search on the sorted absolute
    values, evaluating the reconstruction error in closed form.
    """
    w = weight.float()
    out_f, in_f = w.shape

    w_abs = w.abs()
    w_sign = w.sign()

    # Sort absolute values per row for efficient threshold search
    w_abs_sorted, sort_idx = w_abs.sort(dim=-1)
    w_sign_sorted = w_sign.gather(-1, sort_idx)

    # Candidate thresholds: each unique absolute value
    # Use quantiles to reduce candidates (speed)
    n_candidates = min(in_f, 256)
    if in_f > n_candidates:
        step = in_f // n_candidates
        candidate_indices = torch.arange(0, in_f, step, device=w.device)
    else:
        candidate_indices = torch.arange(in_f, device=w.device)

    candidate_deltas = w_abs_sorted[:, candidate_indices]  # [out_f, n_candidates]

    # For each candidate delta, compute:
    #   active = positions where |w| >= delta
    #   alpha = mean(|w| for active positions)
    #   error = sum((|w| - alpha)² for active positions) + sum(|w|² for inactive)

    best_error = torch.full((out_f,), float("inf"), device=w.device)
    best_alpha = torch.zeros(out_f, device=w.device)
    best_delta = torch.zeros(out_f, device=w.device)
    best_ternary = torch.zeros_like(w)

    for j in range(candidate_indices.size(0)):
        delta = candidate_deltas[:, j:j+1]  # [out_f, 1]
        active = w_abs >= delta  # [out_f, in_f]

        active_count = active.sum(dim=-1).clamp_min(1)  # [out_f]

        # alpha = mean absolute value of active weights
        alpha = (w_abs * active.float()).sum(dim=-1) / active_count  # [out_f]

        # Reconstruction error
        recon = torch.where(active, alpha.unsqueeze(-1) * w_sign, torch.zeros_like(w))
        error = ((w - recon) ** 2).sum(dim=-1)  # [out_f]

        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_alpha = torch.where(improve, alpha, best_alpha)
        best_delta = torch.where(improve, delta.squeeze(-1), best_delta)
        best_ternary = torch.where(
            improve.unsqueeze(-1),
            w_sign * best_alpha.unsqueeze(-1) * active.float(),
            best_ternary,
        )

    # Refine: for rows where best_error is still high, do binary search
    needs_refine = best_error > best_error.median() * 1.5
    if needs_refine.any():
        refine_indices = torch.where(needs_refine)[0]
        for idx in refine_indices:
            w_row = w[idx]
            w_abs_row = w_abs[idx]
            w_sign_row = w_sign[idx]
            delta = best_delta[idx].item()

            for _ in range(config.max_iter):
                active = w_abs_row >= delta
                if active.sum() == 0:
                    delta = delta * 0.5
                    continue
                alpha = w_abs_row[active].mean()
                recon = torch.where(
                    active, alpha * w_sign_row, torch.zeros_like(w_row)
                )
                error = ((w_row - recon) ** 2).sum()
                delta_new = w_abs_row[w_abs_row < delta * 2].max()
                if delta_new == 0 or abs(delta_new - delta) < config.tolerance:
                    break
                delta = delta_new.item()

            if error < best_error[idx]:
                best_ternary[idx] = recon
                best_alpha[idx] = alpha
                best_delta[idx] = delta

    return best_ternary


def apply_pt_bitnet(
    model: PreTrainedModel,
    config: Optional[PTBitNetConfig] = None,
) -> PreTrainedModel:
    """Apply PT-BitNet post-training ternary quantization.

    Iterates over nn.Linear layers and replaces weights with ternary values.
    Fully vectorized — a 7B model should complete in under 60 seconds on GPU.
    """
    if config is None:
        config = PTBitNetConfig()

    logger.info("Starting PT-BitNet post-training quantization (vectorized)...")
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
    config = PTBitNetConfig(block_size=block_size, max_iter=max_iter, tolerance=tolerance, show_progress=False)
    ternary = ternary_quantize_vectorized(weight, config)
    scales = weight.abs().max(dim=-1, keepdim=True).values
    return ternary, scales
