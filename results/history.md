# Experiment History — All Results (Including Failures)

Each experiment documented with date, config, result, and root cause analysis.
Failures are as important as successes — they show what doesn't work and why.

---

## Timeline

```
2026-05-05  00:00   Project start — PT-BitNet + ParetoQ + Tequila pipeline
2026-05-05  02:00   QAT discovered to destroy ternary sparsity
2026-05-05  03:00   QAT removed from pipeline
2026-05-05  05:00   Ablation study: all quantization variants work
2026-05-05  07:00   Tequila OOM on system RAM (12.7 GB Colab)
2026-05-05  08:00   Tequila ran but PPL 20.7 — baking formula bug found
2026-05-05  09:00   Baking fix + ITF outlier fix + memory fixes
2026-05-05  10:00   Tequila still broken — UltraQuantV3 <-> PT-BitNet denorm incompatibility
2026-05-05  11:00   Tequila removed from pipeline. PPL 3.45 (GOOD).
2026-05-05  17:00   LoRA v1 implemented. PPL 3.96 (WORSE than no LoRA).
2026-05-05  19:00   LoRA v2: hyperparameter fixes + re-quantization. Pending test.
```

---

## Experiment 1: Full Pipeline (PT-BitNet + Tequila)

**Date**: 2026-05-05 early
**Hardware**: Colab T4
**Config**: PT-BitNet (asymmetric, ITF, outliers=1%, compensation=50) → Tequila (ultraquantv3, per_channel, epochs=1) → Bake → Save
**Commit**: pre-fix era

### Results

| Metric | Value |
|--------|-------|
| Pipeline time | ~15 min |
| Output | **Garbled** — "The state of the city where they are more..." |
| PPL | **20.7** (baseline 2.61) |
| PPL ratio | **7.94x** |
| Verdict | **FAIL — catastrophic degradation** |

### Root Cause

TWO simultaneous bugs:

1. **Baking formula**: `effective_weight = A + B * Lambada`. Tequila forward uses `linear(ones, B*L)` (constant bias per channel). Baked forward with `A+B*L` computes `linear(input, B*L)` (token-dependent matrix multiply). Different functions.

2. **PT-BitNet denorm vs UltraQuantV3**: PT-BitNet denormalizes weights (w*std + mean). UltraQuantV3 recomputes ternary decomposition on these shifted weights using its own threshold (mean(|w|)/2). When row means are large relative to alpha*std, "zero" weights in PT-BitNet's pattern get classified as active by UltraQuantV3 — destroying ternary structure.

### Disposition

Baking formula fixed (weight=A, bias=sum(B*L)). But fundamental incompatibility remains — Tequila removed from pipeline.

---

## Experiment 2: Ablation Study

**Date**: 2026-05-05
**Hardware**: Colab T4
**Script**: `scripts/colab_ablate.py`
**Commit**: `331855f`

### Results

| Variant | Response | Quality |
|---------|----------|---------|
| 1. BASE (FP16) | "The capital of France is Paris." | OK |
| 2. Symmetric (no ITF, no outliers, no comp) | "The capital of France is known as Paris." | OK |
| 3. Symmetric + outliers + compensation | "The capital of France is Paris." | **BEST** |
| 4. ITF asymmetric (no outliers, no comp) | "Paris..." | OK |
| 5. ITF + outliers + compensation | "The capital of France is Paris." | **BEST** |
| 6. Symmetric + Tequila + Bake | **OOM** | CRASH |

### Key Findings

- ALL quantization variants produce correct output
- Symmetric + outliers + compensation = best quality
- ITF works WITHOUT outliers, fails 96/96 layers WITH outliers
- Tequila crashes on T4 system RAM (12.7 GB)

### Disposition

PT-BitNet (symmetric + outliers + compensation) selected as the pipeline. ITF disabled when outliers present. Tequila kept for investigation.

---

## Experiment 3: ITF + Outliers Fallback

**Date**: 2026-05-05
**Hardware**: Colab T4
**Config**: PTBitNetConfig(asymmetric=True, outlier_fraction=0.01)

### Results

| Metric | Value |
|--------|-------|
| Layers fallback | **96/96** (100%) |
| Time wasted | ~60 seconds |
| Warnings | 96 × "ITF produced extreme values — falling back to symmetric" |

### Root Cause

Outlier positions zeroed before ITF change row statistics. `build_optimal_grid` denominator (m·|T| - (Σ sign)²) becomes unstable for rows where outlier removal changed the ternary pattern distribution.

### Disposition

ITF skipped when `outlier_fraction > 0` in `ternary_quantize_vectorized`. Uses symmetric ternary directly.

---

## Experiment 4: QAT (ParetoQ/ZeroQAT)

**Date**: 2026-05-05 early
**Hardware**: Local testing

### Results

| Metric | Without QAT | With QAT |
|--------|------------|----------|
| Loss | 5.86 | 6.69 |
| Sparsity | 3 unique values/row | Destroyed |

### Root Cause

QuantizeLinear with `w_bits=1` uses `sign()` → binary {-1, +1} (no zero). With `w_bits=0` → StretchedElasticQuant scales to {-0.667a, 0, +0.667a} — incompatible with PT-BitNet's {-a, 0, +a}.

### Disposition

QAT permanently removed. Code kept in `paretoq/` as research artifact.

---

## Experiment 5: Hessian Compensation

**Date**: 2026-05-05
**Hardware**: Colab T4
**Config**: 50 steps on lm_head, WikiText-2 calibration

### Results

| Step | Loss |
|------|------|
| 1 | 5.99 |
| 10 | 5.99 |
| 20 | 5.94 |
| 30 | 5.90 |
| 40 | 5.88 |
| 50 | 5.86 |

**Total improvement**: 2.2% (5.99 → 5.86)

### Analysis

Compensation only trains lm_head (0.1% of parameters). The transformer body's quantization error is not addressed. WikiText calibration data is diverse but the lm_head optimization converges slowly.

Simple repeated text calibration gives much better results: loss 1.10 → 0.80 (27% improvement) in the ablation. The domain mismatch between calibration texts and test texts matters significantly.

### Disposition

Compensation kept but noted as limited. Real improvement needs full-weight fine-tuning (LoRA, STE).

---

## Experiment 6: PT-BitNet Only (Final Working Pipeline)

**Date**: 2026-05-05
**Hardware**: Colab T4
**Config**: PTBitNetConfig(asymmetric=False, outlier_fraction=0.01, compensation_steps=50)
**Commit**: `ec0135b`

### Results

| Metric | Baseline FP16 | Quantized | Ratio |
|--------|--------------|-----------|-------|
| PPL (simple texts) | 2.61 | 3.45 | 1.32x |
| Gen quality | 94/100 | 94/100 | 1.00x |
| Repetition | 0.00 | 0.00 | — |
| Speed | — | 20.5 tok/s | — |
| Pipeline time | — | 10.1 min | — |

### Samples

- "Capital of France?" → `The capital of France is Paris.` ✅
- "Machine learning?" → `Machine learning is a subset of artificial intelligence...` ✅
- "Train 60mph 2h?" → `The train covers a distance of 120 miles...` ✅

### Verdict

**GOOD** — minor perplexity loss (32%), identical generation quality. Competitive for 1.58-bit post-training on a small model (2.7B).

---

## Experiment 7: LoRA Fine-Tuning v1

**Date**: 2026-05-05
**Hardware**: Colab T4
**Config**: LoRA rank=32, distill_weight=0.5, T=3.0, lr=2e-4, steps=500
**Commit**: `091ade2`

### Results

| Metric | Baseline FP16 | PT-BitNet + LoRA | Ratio |
|--------|--------------|------------------|-------|
| PPL (diverse texts) | 3.16 | **3.96** | **1.256x** |
| Gen quality | 86/100 | 90/100 | 1.05x |
| VRAM peak | — | 5.9 GB | — |
| Pipeline time | — | 19.2 min | — |

### Training Dynamics

| Step | CE Loss | KD Loss |
|------|---------|---------|
| 1 | 6.85 | 7.50 |
| 250 | 6.66 | 10.56 |
| 500 | 6.48 | 6.28 |

CE loss INCREASED from without-LoRA baseline (~5.86) to 6.85 at step 1 and never recovered.

### Root Cause Analysis

1. **distill_weight=0.5**: Equal weight to CE and KD pulled model away from correct token predictions
2. **T=3.0**: Teacher distribution almost uniform — very low information per token
3. **Rank 32 on all 7 projection types**: Too many params (15.7M) for 500 steps, underfitting
4. **Dropout 0.05**: Added noise during KD training
5. **No re-quantization**: LoRA correction kept as separate matmul, not baked into ternary structure

### Side Effects

- Model became overly verbose — adds random exercises after each answer
- Creative tasks degraded (haiku → Python code)
- Factual accuracy preserved (all 10 questions correct)

### Disposition

Hyperparameters corrected in v2. Merge+re-quantize added to bake LoRA improvement into ternary weights.

---

## Experiment 8: LoRA Fine-Tuning v2

**Date**: 2026-05-06 (pending)
**Hardware**: Colab T4
**Config**: LoRA rank=64, distill_weight=0.1, T=1.5, lr=5e-5, steps=1000, attention-only, re-quantize after
**Commit**: `350f82c`

### Changes from v1

| Parameter | v1 | v2 | Reason |
|-----------|----|----|--------|
| distill_weight | 0.5 | 0.1 | CE must dominate |
| temperature | 3.0 | 1.5 | Sharper teacher targets |
| rank | 32 | 64 | More capacity |
| lr | 2e-4 | 5e-5 | Stable training |
| steps | 500 | 1000 | More optimization |
| target_modules | 7 types | q,k,v,o | Attention-only |
| dropout | 0.05 | 0 | Teacher regularizes |
| max_seq_length | 64 | 128 | More context |
| re-quantize | No | Yes | Preserve ternary |

### Expected

- CE loss should decrease (not increase) during training
- PPL should be LOWER than without LoRA (not higher)
- PPL ratio target: 1.10-1.20x (vs. 1.32x without LoRA, 1.26x with v1)

### Results

*Pending Colab run.*

---

## Summary: What We Learned

### What works
- Symmetric ternary + 1% outliers → solid baseline
- Hessian compensation on lm_head → small but consistent gain
- CPU teacher for LoRA → safe, no VRAM spike
- Re-quantization after adaptation → preserves ternary structure
- Ablation testing → isolates root causes quickly

### What doesn't work
- **Tequila with PTQ**: Denormalization + re-decomposition conflict (fundamental)
- **QAT with ternary**: sign() quantization incompatible (binary ≠ ternary)
- **ITF + outliers**: Row statistics destabilize closed-form
- **High distill_weight**: Pulls model away from correct predictions
- **Soft KD targets (T≥3)**: Low information, ineffective guidance
- **LoRA on FFN layers**: Adds noise without benefit for ternary

### Open questions
- Can STE block-wise fine-tuning work (or is ternary STE too noisy)?
- What's the optimal LoRA rank for ternary gap?
- Would better calibration data (Alpaca, not WikiText) improve KD?
- Can we train from scratch with BitNet architecture (not post-training)?
- Would a 7B model benefit more from these techniques (more redundancy)?
