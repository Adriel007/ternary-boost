"""PT-BitNet: Post-Training Ternary Quantization for LLMs.

Mathematical foundation from PT²-LLM (Yan et al., ICLR 2026):
  Asymmetric Ternary Quantizer with closed-form Iterative Ternary Fitting (ITF)
  and Activation-aware Grid Alignment (AGA).

Quality improvements from:
  SpQR (Dettmers et al., 2023) — outlier retention
  GPTQ (Frantar et al., 2023)   — Hessian compensation via lm_head fine-tuning
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
    """Configuration for post-training ternary quantization."""
    # Core ternary parameters
    asymmetric: bool = True            # Asymmetric grid {-α+μ, μ, +α+μ} (PT²-LLM)
    itf_iterations: int = 10           # Iterative Ternary Fitting steps
    block_size: int = 128
    # Quality improvements
    outlier_fraction: float = 0.01     # Top-k% weights kept in FP16 (SpQR)
    compensation_steps: int = 50       # Hessian compensation on lm_head (GPTQ)
    # Legacy
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


def _find_quantizable_linears(
    model: nn.Module, config: PTBitNetConfig
) -> list[tuple[str, nn.Linear]]:
    targets = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in module_name for skip in config.skip_modules):
                continue
            if any(t in module_name for t in config.target_modules):
                targets.append((module_name, module))
    logger.info(f"Found {len(targets)} quantizable linear layers")
    return targets


# ═══════════════════════════════════════════════════════════════════
# PT²-LLM: Asymmetric Ternary Quantizer with ITF + AGA
# ═══════════════════════════════════════════════════════════════════

def asymmetric_ternary_init(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize asymmetric ternary parameters (α, μ, T).

    Following PT²-LLM Eq. 4-5:
      μ = row-wise mean of W
      Δ = 0.75 * mean(|W - μ|) per row  (TWN threshold approximation)
      α = weighted mean of active weights
      T initialized via threshold Δ on centered weights
    """
    out_f = w.shape[0]
    mu = w.mean(dim=-1, keepdim=True)       # [out_f, 1]
    w_centered = w - mu

    # TWN threshold: Δ = 0.75 * mean(|w_centered|)
    delta = 0.75 * w_centered.abs().mean(dim=-1, keepdim=True)  # [out_f, 1]

    # Initialize T via threshold
    T = torch.zeros_like(w)
    T[w_centered > delta] = 1.0
    T[w_centered < -delta] = -1.0

    # Initialize α via TWN closed form (Eq. 5)
    num = (T * w_centered).sum(dim=-1)          # [out_f]
    den = T.abs().sum(dim=-1).clamp_min(1)      # [out_f]
    alpha = (num / den).unsqueeze(-1)            # [out_f, 1]

    return alpha, mu, T


def build_optimal_grid(
    T: torch.Tensor, w: torch.Tensor,
    alpha_init: torch.Tensor = None,
    mu_init: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form optimal grid (α*, μ*) given fixed T. PT²-LLM Eq. 9.

    Degenerate rows (all-zero T or all-same-sign) have denom ≈ 0.
    For these, keep the initialization values — the closed-form is undefined.
    For all other rows, the raw closed-form is mathematically optimal;
    no clamping needed.
    """
    m = w.shape[1]
    WoT = w * T
    sum_WoT = WoT.sum(dim=-1)
    sum_abs_T = T.abs().sum(dim=-1)       # |T| = T⊙T since T ∈ {-1,0,1}
    sum_T = T.sum(dim=-1)
    sum_W = w.sum(dim=-1)

    denom = m * sum_abs_T - sum_T * sum_T
    safe = denom > 1e-6                      # Cauchy-Schwarz: denom ≥ 0 always

    # Raw closed-form (PT²-LLM Eq. 9)
    denom_safe = denom.clamp_min(1e-6)
    alpha_opt = (m * sum_WoT - sum_T * sum_W) / denom_safe
    mu_opt = (sum_abs_T * sum_W - sum_T * sum_WoT) / denom_safe

    if alpha_init is not None and mu_init is not None:
        ai = alpha_init.squeeze(-1)
        mi = mu_init.squeeze(-1)
        # Use init for degenerate rows; raw optimum for all others
        alpha = torch.where(safe, alpha_opt, ai)
        mu = torch.where(safe, mu_opt, mi)
    else:
        alpha = torch.where(safe, alpha_opt, alpha_opt.abs().clamp_min(1e-6))
        mu = torch.where(safe, mu_opt, torch.zeros_like(mu_opt))

    return alpha.unsqueeze(-1), mu.unsqueeze(-1)


def flexible_rounding(
    w: torch.Tensor, alpha: torch.Tensor, mu: torch.Tensor,
) -> torch.Tensor:
    """Optimal ternary assignment given grid (α, μ).

    PT²-LLM Eq. 10:
      Z_ij = (W_ij - μ_i) / α_i
      T*_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
    """
    z = (w - mu) / (alpha.clamp_min(1e-8))     # [out_f, in_f]
    T = torch.zeros_like(w)
    T[z > 0.5] = 1.0
    T[z < -0.5] = -1.0
    return T


def structural_similarity_reorder(
    w: torch.Tensor, block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSR: Reorder columns by similarity to form compact quantization blocks.

    PT²-LLM Eq. 16 — efficient greedy clustering:
      1. Compute the mean vector of remaining submatrix
      2. Select top-k columns most similar to this mean (cosine similarity)
      3. Those k columns form the next quantization block
      4. Repeat for remaining columns

    This is fully vectorized — no Python loops over columns.
    """
    out_f, in_f = w.shape
    w_f = w.float()
    col_norms = w_f.norm(dim=0).clamp_min(1e-8)  # [in_f]

    mask = torch.ones(in_f, dtype=torch.bool, device=w.device)
    perm_indices = []

    for _ in range(0, in_f, block_size):
        if not mask.any():
            break
        # Compute mean of remaining columns
        w_remaining = w_f[:, mask]  # [out_f, n_remaining]
        w_bar = w_remaining.mean(dim=1)  # [out_f]

        # Cosine similarity of all remaining columns to w_bar
        bar_norm = w_bar.norm().clamp_min(1e-8)
        sim = (w_remaining.T @ w_bar) / (col_norms[mask] * bar_norm)  # [n_remaining]

        # Take top-k most similar (or all if fewer than k remain)
        k = min(block_size, sim.shape[0])
        _, top_local = sim.topk(k)

        # Map local indices back to global column indices
        remaining_global = torch.where(mask)[0]
        top_global = remaining_global[top_local]
        perm_indices.append(top_global)
        mask[top_global] = False

    perm = torch.cat(perm_indices) if perm_indices else torch.arange(in_f, device=w.device)
    inv_perm = torch.zeros(in_f, dtype=torch.long, device=w.device)
    inv_perm[perm] = torch.arange(in_f, device=w.device)

    return w[:, perm], inv_perm


def iterative_ternary_fitting(
    w: torch.Tensor,
    config: PTBitNetConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ITF: alternate between grid optimization and ternary rounding.

    PT²-LLM Algorithm 1:
      1. Initialize α, μ, T
      2. Repeat until T stabilizes:
         a. Build optimal grid (α, μ) given T (Eq. 9)
         b. Flexible rounding T given (α, μ) (Eq. 10)
      3. Return converged (α, μ, T)

    Converges in ~10 iterations. Each step is closed-form — no gradients.
    """
    alpha, mu, T = asymmetric_ternary_init(w)
    alpha_init = alpha.clone()
    mu_init = mu.clone()

    for _ in range(config.itf_iterations):
        T_prev = T.clone()
        alpha, mu = build_optimal_grid(T, w, alpha_init, mu_init)
        T = flexible_rounding(w, alpha, mu)
        if torch.equal(T, T_prev):
            break

    return alpha, mu, T


def activation_aware_grid_alignment(
    w: torch.Tensor,
    T: torch.Tensor,
    X: Optional[torch.Tensor],
    alpha_init: torch.Tensor = None,
    mu_init: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AGA: refine grid parameters to minimize output error.

    PT²-LLM Eq. 13. Includes same numerical safeguards as build_optimal_grid.
    """
    if X is None:
        return build_optimal_grid(T, w, alpha_init, mu_init)

    # Compute activation covariance S = sum over batch of X_b X_b^T
    # X shape: [B, L, m] → reshape to [B*L, m]
    X_flat = X.reshape(-1, X.shape[-1]).float()  # [B*L, m]
    S = X_flat.T @ X_flat  # [m, m]

    m = w.shape[1]
    ones = torch.ones(m, 1, device=w.device, dtype=torch.float32)
    T_float = T.float()
    w_float = w.float()

    # PT²-LLM Eq. 13 (vectorized)
    d = (ones.T @ S @ ones).squeeze()            # scalar
    v = (T_float @ S @ ones).squeeze(-1)         # [out_f]

    WoT = w_float * T_float
    T2 = T_float * T_float

    sum_WoT_S1 = (WoT @ S @ ones).squeeze(-1)    # [out_f]
    sum_WS1 = (w_float @ S @ ones).squeeze(-1)    # [out_f]
    sum_T2_S1 = (T2 @ S @ ones).squeeze(-1)       # [out_f]

    denom = d * sum_T2_S1 - v * v                 # [out_f]
    denom = denom.clamp_min(1e-8)

    alpha = (d * sum_WoT_S1 - v * sum_WS1) / denom
    mu = (sum_T2_S1 * sum_WS1 - v * sum_WoT_S1) / denom

    return alpha.unsqueeze(-1), mu.unsqueeze(-1)


def ternary_quantize_vectorized(
    weight: torch.Tensor,
    config: PTBitNetConfig,
    calibration_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full PT²-LLM quantization pipeline: ITF + AGA + Outlier Retention.

    1. Identify and preserve outliers (SpQR-style)
    2. Compute activation-aware column weights from calibration data
    3. Run ITF on non-outlier weights or symmetric ternary
    4. Merge: outliers kept FP16, rest quantized
    """
    w = weight.float()
    out_f, in_f = w.shape

    # --- Activation-aware column weights ---
    col_weights = None
    if calibration_inputs is not None:
        # Hessian diagonal: mean activation squared per input channel
        col_weights = (calibration_inputs.float() ** 2).mean(dim=0)  # [in_f]
        col_weights = col_weights / col_weights.mean().clamp_min(1e-8)  # normalize

    # --- Outlier identification (SpQR-style) ---
    if config.outlier_fraction > 0:
        w_abs = w.abs()
        row_mean = w_abs.mean(dim=-1, keepdim=True)
        row_std = w_abs.std(dim=-1, keepdim=True).clamp_min(1e-8)
        outlier_score = (w_abs - row_mean) / row_std
        k = max(1, int(in_f * config.outlier_fraction))
        _, outlier_idx = outlier_score.topk(k, dim=-1)
        outlier_mask = torch.zeros_like(w, dtype=torch.bool)
        outlier_mask.scatter_(-1, outlier_idx, True)
    else:
        outlier_mask = torch.zeros_like(w, dtype=torch.bool)

    # Remove outliers before quantization
    w_clean = w.clone()
    w_clean[outlier_mask] = 0

    if config.asymmetric and config.outlier_fraction == 0:
        alpha, mu, T = iterative_ternary_fitting(w_clean, config)

        if calibration_inputs is not None:
            alpha, mu = activation_aware_grid_alignment(
                w_clean, T, calibration_inputs, alpha, mu)

        reconstructed = alpha * T + mu

        if not torch.isfinite(reconstructed).all():
            logger.warning(f"  ITF produced non-finite values — falling back to symmetric")
            reconstructed = _symmetric_ternary(w_clean, config, col_weights)
        elif reconstructed.abs().max() > 1e6:
            logger.warning(f"  ITF produced extreme values — falling back to symmetric")
            reconstructed = _symmetric_ternary(w_clean, config, col_weights)
    else:
        reconstructed = _symmetric_ternary(w_clean, config, col_weights)

    # Merge outliers back (kept in FP16)
    reconstructed[outlier_mask] = w[outlier_mask]

    return reconstructed


def _symmetric_ternary(
    w: torch.Tensor, config: PTBitNetConfig,
    col_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Symmetric ternary fallback: {-α, 0, +α}.

    If col_weights is provided (e.g., Hessian diagonal), the error
    computation weights each column by its activation importance:
      error = Σ_j h_j * (w_ij - w_q_ij)^2
    This is the activation-aware variant — channels with higher activations
    get finer quantization.
    """
    out_f, in_f = w.shape
    w_abs = w.abs()
    w_sign = w.sign()

    if col_weights is None:
        col_weights = torch.ones(in_f, device=w.device)

    w_abs_sorted, _ = w_abs.sort(dim=-1)
    n_candidates = min(in_f, 256)
    step = max(1, in_f // n_candidates)
    candidate_indices = torch.arange(0, in_f, step, device=w.device)
    candidate_deltas = w_abs_sorted[:, candidate_indices]

    best_error = torch.full((out_f,), float("inf"), device=w.device)
    best_result = torch.zeros_like(w)
    cw = col_weights.unsqueeze(0)  # [1, in_f] for broadcasting

    for j in range(candidate_indices.size(0)):
        delta = candidate_deltas[:, j:j+1]
        active = w_abs >= delta
        active_count = active.sum(dim=-1).clamp_min(1)
        alpha = (w_abs * active.float()).sum(dim=-1) / active_count
        recon = torch.where(active, alpha.unsqueeze(-1) * w_sign, torch.zeros_like(w))
        # Weighted error: columns with higher activations contribute more
        sq_error = ((w - recon) ** 2) * cw
        error = sq_error.sum(dim=-1)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_result = torch.where(improve.unsqueeze(-1), recon, best_result)

    return best_result


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def apply_pt_bitnet(
    model: PreTrainedModel,
    config: Optional[PTBitNetConfig] = None,
    tokenizer=None,
    calibration_texts: Optional[list[str]] = None,
) -> PreTrainedModel:
    """Apply PT-BitNet post-training ternary quantization.

    Uses PT²-LLM asymmetric ternary quantizer with ITF + AGA.
    Optionally applies Hessian compensation on lm_head.
    """
    if config is None:
        config = PTBitNetConfig()

    logger.info("Starting PT-BitNet (PT²-LLM asymmetric ternary quantizer)...")
    if config.asymmetric:
        logger.info("  Mode: Asymmetric {-α+μ, μ, +α+μ} with ITF + AGA")
    if config.outlier_fraction > 0:
        logger.info(f"  Outlier retention: {config.outlier_fraction*100:.1f}% FP16")

    model.eval()
    targets = _find_quantizable_linears(model, config)

    # Prepare calibration activations for AGA (if possible)
    calibration_acts = _collect_activations(model, tokenizer, calibration_texts, config)

    with torch.no_grad():
        for i, (name, module) in enumerate(targets):
            w = module.weight.data
            device = w.device

            # Move to GPU for speed if available
            w_clean = w.cuda() if torch.cuda.is_available() else w

            # Get calibration inputs for this layer (for AGA / OBC compensation)
            calib_in = calibration_acts.get(name) if calibration_acts else None
            if calib_in is not None and torch.cuda.is_available():
                calib_in = calib_in.cuda()

            # Normalize before quantization
            w_mean = w_clean.mean(dim=-1, keepdim=True)
            w_std = w_clean.std(dim=-1, keepdim=True).clamp_min(1e-8)
            w_norm = (w_clean - w_mean) / w_std
            w_norm = w_norm.clamp(-config.outlier_clip_threshold, config.outlier_clip_threshold)

            # PT²-LLM quantization
            ternary_w_norm = ternary_quantize_vectorized(w_norm, config, calib_in)

            # Denormalize
            ternary_w = ternary_w_norm * w_std + w_mean

            # ── OBC-style row compensation ──────────────────────
            # Activation-aware bias correction in output space.
            # After ternary quantization, each row has residual error.
            # We project this error through the mean calibration activation
            # to get the optimal per-output-channel bias correction.
            # Closed-form, no training. First-order OBC approximation.
            if calib_in is not None:
                x_mean = calib_in.float().mean(dim=0)  # [in_f]
                # Error in output space: ternary_w - original_w
                error = ternary_w.float() - w.float()   # [out_f, in_f]
                bias_comp = -(error @ x_mean)            # [out_f]
                if module.bias is not None:
                    module.bias.data.add_(
                        bias_comp.to(device=device, dtype=module.bias.dtype))
                else:
                    module.bias = nn.Parameter(
                        bias_comp.to(device=device, dtype=module.weight.dtype))

            ternary_w = ternary_w.to(device=device, dtype=module.weight.dtype)

            module.weight = nn.Parameter(ternary_w)
            module.weight.requires_grad = False

            if config.show_progress and (i + 1) % max(1, len(targets) // 10) == 0:
                logger.info(f"  PT-BitNet: {i + 1}/{len(targets)} layers "
                            f"({100 * (i + 1) // len(targets)}%)")

    logger.info(f"PT-BitNet complete: {len(targets)} layers quantized "
                f"(with activation-aware row compensation)")

    # Optional: Hessian compensation on lm_head
    if config.compensation_steps > 0 and tokenizer is not None and calibration_texts is not None:
        hessian_compensation(model, tokenizer, calibration_texts, config)

    return model


def _collect_activations(
    model: PreTrainedModel,
    tokenizer,
    calibration_texts: Optional[list[str]],
    config: PTBitNetConfig,
) -> Optional[dict[str, torch.Tensor]]:
    """Collect calibration activations for AGA.

    Runs a single forward pass on calibration data and captures the
    input to each quantizable linear layer.
    """
    if tokenizer is None or calibration_texts is None or len(calibration_texts) == 0:
        return None

    logger.info("  Collecting calibration activations for AGA...")
    acts = {}

    def _make_hook(layer_name):
        def _hook(module, input, output):
            if isinstance(input, tuple):
                acts[layer_name] = input[0].detach()
            else:
                acts[layer_name] = input.detach()
        return _hook

    handles = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in module_name for skip in config.skip_modules):
                continue
            if any(t in module_name for t in config.target_modules):
                handles.append(module.register_forward_hook(_make_hook(module_name)))

    # Run one batch through the model
    has_cuda = torch.cuda.is_available()
    text = calibration_texts[0] if calibration_texts else "Hello world"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"]
    model_device = next(model.parameters()).device
    if has_cuda:
        input_ids = input_ids.to(model_device)

    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception as e:
            logger.warning(f"  Activation collection forward failed: {e}")

    for h in handles:
        h.remove()

    logger.info(f"  Collected activations for {len(acts)} layers")
    return acts if acts else None


# ═══════════════════════════════════════════════════════════════════
# Hessian Compensation (unchanged)
# ═══════════════════════════════════════════════════════════════════

def hessian_compensation(
    model: PreTrainedModel,
    tokenizer,
    calibration_texts: list[str],
    config: PTBitNetConfig,
) -> None:
    """Memory-efficient Hessian compensation on lm_head."""
    if config.compensation_steps <= 0:
        return

    logger.info(f"Hessian compensation: {config.compensation_steps} steps (memory-efficient)")

    has_cuda = torch.cuda.is_available()
    original_device = next(model.parameters()).device
    if has_cuda and original_device.type == "cpu":
        model.cuda()
    train_device = next(model.parameters()).device

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

    _hidden_state = None

    def _save_last_hidden(module, args, output):
        nonlocal _hidden_state
        if hasattr(output, "last_hidden_state"):
            _hidden_state = output.last_hidden_state.detach()
        elif isinstance(output, tuple):
            _hidden_state = output[0].detach()
        else:
            _hidden_state = output.detach()

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

            _hidden_state = None
            with torch.no_grad():
                model(input_ids=input_ids)

            if _hidden_state is None:
                continue

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
    # Keep model on GPU if that's where it was trained (avoids CPU RAM spike)
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Hessian compensation complete")


# ═══════════════════════════════════════════════════════════════════
# Legacy API — kept for backward compatibility with tests
# ═══════════════════════════════════════════════════════════════════

def distribution_transform(
    weight: torch.Tensor, clip_threshold: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.float()
    mean = w.mean(dim=-1, keepdim=True)
    std = w.std(dim=-1, keepdim=True).clamp_min(1e-8)
    w_norm = (w - mean) / std
    scale = w.abs().max(dim=-1, keepdim=True).values
    return torch.clamp(w_norm, -clip_threshold, clip_threshold), scale


def blockwise_optimize(
    weight: torch.Tensor, block_size: int = 128,
    max_iter: int = 10, tolerance: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = PTBitNetConfig(
        block_size=block_size, max_iter=max_iter, tolerance=tolerance,
        outlier_fraction=0.0, show_progress=False, asymmetric=False,
    )
    ternary = _symmetric_ternary(weight.float(), config)
    scales = weight.abs().max(dim=-1, keepdim=True).values
    return ternary, scales
