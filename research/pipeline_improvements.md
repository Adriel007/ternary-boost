# Pipeline Improvement Research — Mathematical & Literature Analysis

## Current State (What Works)

```
FP16 Model → PT-BitNet (sym ternary + 1% outliers + lm_head comp) → LoRA KD → Dense Merge
PPL ratio: 1.24x  |  Pipeline: 35 min  |  VRAM: 6 GB
```

- Symmetric ternary quantization with threshold search
- 1% outlier retention (SpQR-style)
- Hessian compensation on lm_head only (2% loss improvement)
- LoRA rank-64 KD from FP16 teacher (7% PPL improvement)

## Gap Analysis (Where Quality Is Lost)

| Source of Error | Contribution | Current Fix | Improvement Potential |
|-----------------|--------------|-------------|----------------------|
| Ternary weight error | ~70% | Threshold search (256 candidates) | Activation-aware thresholds |
| Row-wise quantization error | ~15% | None | OBC-style row compensation |
| Activation outliers | ~10% | Uniform outlier retention | SmoothQuant migration |
| Calibration mismatch | ~5% | WikiText | Self-generated data |

## Technique 1: OBC-Style Row Compensation [HIGHEST IMPACT]

### Mathematical Foundation

For a linear layer y = Wx, the error from quantizing weight w_k is:

```
E_k = min_δ Σ_i ||(W_quantized + δe_k^T)x_i - W_fp16 x_i||^2
```

Where e_k is the k-th unit vector and δ is the optimal correction.

The OBC solution (Frantar et al., 2022):
```
δ_w_j = (w_quantized_k - w_k) / H^{-1}_{k,k} · H^{-1}_{k,j}
```

Where H = Σ x_i x_i^T is the activation covariance (Hessian).

### Adaptation for Ternary (Per-Row)

Instead of per-weight quantization, quantize per ROW. After row i is quantized to {-α_i, 0, +α_i}, compensate:

```
δ_bias = -Σ_k (w_quantized_{i,k} - w_{i,k}) · h_k / h_mean
```

This is a single bias correction per row, computed from the Hessian diagonal h = diag(X @ X^T).

**Implementation:**
1. Collect calibration activations X per layer (already done — 96 layers)
2. For each row i:
   a. Quantize to {-α_i, 0, +α_i} via threshold search
   b. Compute Hessian-weighted error: error_k = w_q_{i,k} - w_{i,k}
   c. Compute optimal bias correction using h_k
   d. Apply correction to bias term
3. Proceed to next row

**Expected improvement:** PPL ratio 1.32x → 1.12-1.18x (replaces lm_head-only comp with full per-row comp)

**Why this works better than lm_head training:**
- Addresses error at SOURCE (each row), not just at output (lm_head)
- Closed-form: one pass, no optimization needed
- Row-wise: correct structure (ternary is per-row)
- Activation-aware: uses real activation statistics

### Algorithm

```
def obc_row_compensation(W, X, alpha, ternary_mask):
    """Compensate ternary quantization error using Hessian.
    
    Args:
        W: original FP16 weight [out_f, in_f]
        X: calibration input [B*L, in_f]
        alpha: row-wise scales [out_f]
        ternary_mask: which weights are active [out_f, in_f] bool
    
    Returns:
        W_adjusted: compensated ternary weight
        bias_comp: row-wise bias compensation [out_f]
    """
    H = X.T @ X  # [in_f, in_f]
    h_diag = H.diag()  # [in_f]
    
    bias_comp = torch.zeros(out_f)
    for i in range(out_f):
        # Error from ternary quantization
        error = W[i] - alpha[i] * ternary_mask[i].float() * sign(W[i])
        
        # Hessian-weighted optimal per-row bias
        # Compensates for the quadrature error
        bias_comp[i] = -(error @ h_diag) / h_diag.mean()
    
    return W, bias_comp
```

## Technique 2: Activation-Aware Ternary Threshold [MEDIUM IMPACT, LOW EFFORT]

### Mathematical Foundation

Current threshold: δ_i = α_i / 2 (uniform, same for all channels in a row)

AWQ insight: weights connected to high-activation input channels contribute more to output error. They need more precise quantization → narrower deadzone.

**Proposed:** Per-channel threshold scaling
```
δ_{i,k} = (α_i / 2) / s_k
```

Where s_k captures the relative importance of input channel k:
```
s_k = sqrt(mean(X[:,k]^2)) / mean_j(sqrt(mean(X[:,j]^2)))
```

High-activation channels (s_k > 1): wider active zone → easier to be non-zero
Low-activation channels (s_k < 1): wider deadzone → more sparsity

**Implementation:**
1. Compute per-channel activation magnitude from calibration data
2. Scale thresholds proportionally
3. Re-quantize with scaled thresholds

**Expected improvement:** PPL ratio improvement 3-5%

## Technique 3: SmoothQuant-Style Activation Migration [MEDIUM IMPACT, LOW EFFORT]

### Mathematical Foundation

SmoothQuant transformation:
```
Y = X @ W^T = (X ⊘ s) @ (W ⊙ s)^T
```

Where s_j = max(|X|_j)^α / max(|W|_j)^(1-α), α ∈ [0, 1]

The key insight: divide activation by s, multiply weight by s. This is mathematically EQUIVALENT but redistributes quantization difficulty.

For ternary: apply BEFORE quantization.
1. Compute smooth factors s from calibration data
2. Scale weights: W_smooth = W ⊙ s
3. Quantize W_smooth → W_ternary
4. After quantization, the smoothing is "baked in" — no runtime overhead

**Why this helps:** The scaling makes weight distributions more uniform, reducing the gap between large and small weights. This improves ternary fit.

**Implementation:**
```
def smooth_weights(W, X, alpha=0.5):
    """Migrate activation difficulty to weights."""
    a_max = X.abs().max(dim=0).values  # [in_f]
    w_max = W.abs().max(dim=0).values  # [in_f]
    s = (a_max ** alpha) / (w_max ** (1 - alpha) + 1e-8)
    W_smooth = W * s.unsqueeze(0)
    return W_smooth, s
```

**Expected improvement:** PPL ratio improvement 3-5%

## Technique 4: Self-Generated Calibration Data [LOW-MEDIUM IMPACT, LOW EFFORT]

### Motivation

WikiText calibration: loss drops only 2% (5.99→5.86). The calibration distribution doesn't match what the model actually sees during generation.

LLM-QAT approach: use FP16 model's own generations as calibration data. Benefits:
- Distribution matches inference
- Diverse topics (varies prompts)
- No external data dependency
- The "knowledge" being preserved is the model's own behavior

**Implementation:**
1. Generate 200 completions from FP16 model with diverse prompts
2. Use these as calibration data for Hessian comp + LoRA KD

## Technique 5: ZO Fine-Tuning of Ternary Weights [EXPERIMENTAL]

### Mathematical Foundation

Zeroth-order optimization: estimate gradient via SPSA (Simultaneous Perturbation Stochastic Approximation):
```
∇f(w) ≈ (f(w + εz) - f(w - εz)) / (2ε) · z
```
where z ~ N(0, I) is a random perturbation.

This requires only 2 FORWARD passes (no backward) → memory-efficient.

For ternary: perturb ternary weights, measure loss change, update. Only touch 0.1% most sensitive weights.

**Why this could work where STE failed:**
- STE gradients for ternary are noisy (gradient is 0 almost everywhere)
- ZO gradients are noisy but UNBIASED (SPSA gives consistent gradient estimates)
- Only need 2 forward passes per step (very memory efficient)

**Implementation sketch:**
1. After PT-BitNet, identify 0.1% most "sensitive" weights (largest abs value, or near threshold)
2. For each fine-tuning step:
   a. Generate random perturbation z (sparse, only on sensitive weights)
   b. Forward pass with W + εz: compute loss
   c. Forward pass with W - εz: compute loss
   d. Update: W -= lr * (loss_pos - loss_neg) / (2ε) * z
   e. Re-quantize to ternary

**Expected improvement:** Unknown (experimental). ZO + ternary is an open research question.

## Implementation Priority

| Priority | Technique | Impact | Effort | Stack |
|----------|-----------|--------|--------|-------|
| **P0** | OBC Row Compensation | HIGH (15-20% PPL) | MEDIUM (1-2 days) | Replaces lm_head comp |
| **P1** | Activation-Aware Threshold | MEDIUM (3-5%) | LOW (few hours) | On top of OBC |
| **P1** | SmoothQuant Preprocessing | MEDIUM (3-5%) | LOW (few hours) | On top of OBC |
| **P2** | Self-Generated Calibration | LOW-MEDIUM (2-4%) | LOW (few hours) | Data change only |
| **P3** | ZO Fine-Tuning | UNKNOWN | HIGH (3-5 days) | Experimental |

## Expected Combined Improvement

```
PPL ratio = 1.32x (current, without LoRA)
  - OBC compensation:  1.32 → 1.15x
  - Act-Aware threshold: 1.15 → 1.12x
  - SmoothQuant:         1.12 → 1.09x
  - Self-gen calibration: 1.09 → 1.07x
  = ~1.07x (competitive with PT²-LLM paper)
```

With LoRA on top: PPL ratio could reach 1.03-1.05x.

## References

- Frantar et al. (2022) "Optimal Brain Compression" — OBC framework, Hessian-based weight compensation
- Frantar et al. (2023) "GPTQ" — Block-wise second-order quantization for LLMs
- Xiao et al. (2023) "SmoothQuant" — Activation smoothing for quantization
- Lin et al. (2024) "AWQ" — Activation-aware per-channel scaling
- Guo et al. (2024) "ZO Fine-Tuning" — Zeroth-order optimization for sparse LLM fine-tuning
- Liu et al. (2024) "LLM-QAT" — Self-generated calibration data for QAT
- Dettmers et al. (2023) "SpQR" — Outlier retention for 4-bit quantization
- Yan et al. (2026) "PT²-LLM" — Asymmetric ternary with ITF and AGA
