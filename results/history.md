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
2026-05-06  12:00   LoRA v3 (dense merge): PPL 1.239x — best result so far.
2026-05-06  22:00   OBC + act-aware + LoRA v3: PPL 1.281x — OBC regressed quality.
2026-05-07  12:00   Infrastructure overhaul: GPU teacher, vectorized grid search, sharded export, checkpoint/resume. Pipeline stable, no OOM.
2026-05-07  15:00   WikiText-2 baseline established. FP16=27.39, ternary=33.31 at max_len=128 → 1.216x. Attention-only LoRA, code from pre-overhaul era.
2026-05-07  15:30   LoRA extended to FFN layers (dense, fc1, fc2, gate/up/down). Pending test.
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

## Experiment 8: LoRA Fine-Tuning v2 (with re-quantization)

**Date**: 2026-05-06
**Hardware**: Colab T4
**Config**: rank=64, distill_weight=0.1, T=1.5, lr=5e-5, steps=1000, attention-only, merge+re-quantize
**Commit**: `350f82c`

### Results

| Metric | Value |
|--------|-------|
| PPL | **17.86** |
| PPL ratio | **5.656x** |
| Gen quality | 72/100 (baseline 87/100) |
| Repetition | 0.12 |
| Pipeline time | 36.3 min |
| Verdict | **FAIL — catastrophic degradation** |

### Training Dynamics

| Step | CE Loss | KD Loss |
|------|---------|---------|
| 1 | 6.91 | 7.88 |
| 500 | 5.36 | 11.31 |
| 1000 | 4.03 | 10.38 |

CE loss DROPPED from 6.91 to 4.03 (below without-LoRA baseline of ~5.86). LoRA training WORKED.
But KD loss INCREASED (7.88→10.38) — student diverged from teacher distribution.
After merge+re-quantize: model collapsed to PPL 17.86.

### Root Cause

**Re-quantization after merge destroyed the LoRA correction.** LoRA learns to COMPLEMENT the ternary weights (W_ternary + correction). `merge_and_requantize` merges them into W_dense, then runs `_symmetric_ternary(W_dense)` to re-ternarize. The new ternary pattern is completely different from the original — it optimizes for the MERGED weight, not for the ternary+LoRA combination. LoRA was trained with the assumption that W_ternary stays fixed; re-ternarizing breaks this assumption.

### Output Samples (post-re-quantize collapse)

- "Capital of Japan?" → `"The capital of Japan is Harukus..."` (WRONG — should be Tokyo)
- "WWII end?" → `"In the year 2000..."` (WRONG — should be 1945)
- "Planets?" → `"There are four planets..."` (WRONG — should be eight)
- Repetitive loops: `"the capital of Japan, a city that is the capital of Japan..."`

### Disposition

Re-quantization removed. Use `merge_lora_to_weights` (dense, no re-ternarize). Same bf16 file size (5.6 GB) but quality preserved. Commit `4774cf9`.

---

## Experiment 9: LoRA Fine-Tuning v3 (dense merge, no re-quantize)

**Date**: 2026-05-06
**Hardware**: Colab T4
**Config**: rank=64, distill_weight=0.1, T=1.5, lr=5e-5, steps=1000, attention-only (q/k/v), dense merge (no re-quantize)
**Commit**: `4774cf9`

### Results

| Metric | Baseline FP16 | PT-BitNet + LoRA v3 | Ratio |
|--------|--------------|---------------------|-------|
| PPL (diverse texts) | 3.16 | 3.91 | **1.239x** |
| Gen quality | 82/100 | 88/100 | **1.07x** |
| Jaccard overlap | — | 0.36 | — |
| Repetition | 0.00 | 0.000 | — |
| Speed | — | 19.2 tok/s | — |
| VRAM peak | — | 6.0 GB | — |
| Pipeline time | — | 34.7 min | — |
| Verdict | — | **GOOD** | |

### Training Dynamics

| Step | CE Loss | KD Loss |
|------|---------|---------|
| 1 | 7.74 | 12.81 |
| 500 | 5.28 | 10.88 |
| 1000 | 4.09 | 10.31 |

CE loss dropped 47% (7.74→4.09). Below without-LoRA baseline (~5.86).

### Generation Samples

| Prompt | Response | Correct? |
|--------|----------|-----------|
| "Capital of Japan?" | `Tokyo` | ✅ |
| "WWII end year?" | `1945.` + relevant exercises | ✅ |
| "Water symbol?" | `H2O` + water cycle exercises | ✅ |
| "Planets count?" | `There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.` | ✅ |
| "5 apples × $2?" | `5 x $2 = $10` + explanation | ✅ |
| "All dogs animals?" | `Yes` + logical deduction | ✅ |
| "Photosynthesis?" | `Plants use sunlight, CO2, water → glucose + oxygen` | ✅ |
| "Moon vs Earth?" | `Moon much smaller, Earth ~10x bigger` | ✅ |
| "Lie ethical?" | `No, goes against honesty principle` | ✅ |
| "Haiku ocean?" | `Waves crashing on the shore / Saltwater and seagull cries / Nature's symphony at play` | ✅ |

### Comparison: All LoRA Versions

| Version | PPL Ratio | Gen Quality | Pipeline | Key Difference |
|---------|-----------|-------------|----------|----------------|
| No LoRA (simple texts) | 1.324x | 94/100 | 10 min | PT-BitNet only |
| v1 (bad KD, re-quantize) | 1.256x | 90/100 | 19 min | distill=0.5, T=3.0, rank=32 |
| v2 (good KD, re-quantize) | 5.656x | 72/100 | 36 min | Re-quantize destroyed weights |
| **v3 (good KD, dense)** | **1.239x** | **88/100** | 35 min | **Best overall** |

### Analysis

**What improved vs v1:**
- PPL ratio: 1.256x → 1.239x (modest 1.7% gain on same texts)
- Jaccard overlap: 0.20 → 0.36 (outputs more similar to baseline)
- Creative tasks: v1 generated Python code, v3 generates actual haiku
- Exercise relevance: v1 added random exercises, v3 adds context-appropriate follow-ups
- No degeneration: 0.000 repetition (v1 had 0.005)

**What didn't improve enough:**
- PPL ratio gain vs no-LoRA is modest (~7% estimated, from 1.324x to 1.239x across different text sets)
- Pipeline time tripled (10→35 min) for modest PPL gain
- KD loss still high (10.31) — student diverged from teacher during training
- CE loss improved (7.74→4.09) but initial loss was higher than without-LoRA
- Model is more verbose than baseline (adds exercises)

**Honest assessment:** LoRA + dense merge is a working quality improvement, but the cost (25 min extra) may not justify the benefit (~7% PPL improvement) for all use cases. For a quick compressed model, PT-BitNet alone (10 min) is sufficient. For maximum quality, LoRA v3 (35 min) provides better output formatting and factuality.

### Disposition

This is the recommended LoRA configuration. Dense merge preserves quality. Re-quantization is proven harmful and should not be used.

---

## Failed Techniques (Not Full Experiments)

These were tried briefly but abandoned quickly. Documented for completeness.

### SSR (Structural Similarity Reordering)
**Commits**: `b931cdc`
**Paper**: PT²-LLM Section 3.3, Eq. 14-16
**What**: Column reordering by cosine similarity to form compact quantization blocks
**Failure**: Inverse permutation bug — columns scrambled, producing garbled output
**Disposition**: Disabled via TODO, never fixed

### AGA (Activation-aware Grid Alignment)
**Commits**: `2d49afa`
**Paper**: PT²-LLM Section 3.2, Eq. 13
**What**: Align quantization grid with calibration activations using covariance S = Σ XXᵀ
**Failure mode 1**: Collected 0 layers on GPU (device mismatch — model CPU, input CUDA)
**Failure mode 2**: Fixed device handling (now collects 96 layers) but symmetric ternary doesn't use it
**Disposition**: Code fixed but unused — symmetric ternary path doesn't call AGA

### ITF with Clamped Alpha/Mu
**Commit**: `a53bdda`
**What**: Clamp closed-form alpha to [0.1×init, 10×init] and mu to [init-3α, init+3α]
**Failure**: Too restrictive — clamping prevented ITF from finding good solutions even for well-conditioned rows
**Disposition**: Replaced with outlier preprocessing + skip strategy

### Torch.no_grad() in Tequila
**Commit**: `c55df1a`, `865d650`
**What**: torch.no_grad() around Tequila forward blocked Lambada gradients
**Symptom**: Lambada not updating, loss not decreasing
**Fix**: Removed no_grad, forward hook captures hidden state detached, only lm_head trained
**Disposition**: Fixed but Tequila later removed entirely

### Stale custom_params.safetensors
**Commit**: `3b543f2`
**What**: After baking UltraQuantLinear→nn.Linear, old custom_params.safetensors from failed runs persisted
**Symptom**: Load found stale Lambada params and reconstructed UltraQuantLinear on already-baked weights → garbled output
**Fix**: Delete custom_params.safetensors if no custom params exist after save

### Full state_dict in Memory During Save
**Commit**: `7034e3e`
**What**: `model.state_dict()` loaded all tensors into RAM simultaneously
**Symptom**: 12.7 GB Colab system RAM exhausted during save
**Fix**: Iterate `named_parameters()` one tensor at a time, 800 MB shards

### Baking on Wrong Device
**Commit**: `5440826`
**What**: Baking created nn.Linear on CPU while model on GPU
**Symptom**: dtype mismatch crash (CPU float32 vs GPU bfloat16)
**Fix**: Create nn.Linear on same device as source module

### local_files_only on Colab
**Commit**: `355fd57`
**What**: `local_files_only=True` blocked HuggingFace downloads on fresh Colab
**Symptom**: Model not found error
**Fix**: Remove flag, add HF_HOME for cache reuse

### High Distill Weight in LoRA KD
**Commit**: `49b6903` → `350f82c`
**What**: distill_weight=0.5 gave equal weight to CE and KD loss
**Symptom**: CE loss increased (5.86→6.85), model learned to match teacher distribution at cost of correctness
**Fix**: Reduce to 0.1 (CE dominates)

### Soft Teacher Temperature (T=3.0)
**Commit**: `49b6903` → `350f82c`
**What**: Temperature 3.0 made teacher distribution almost uniform
**Symptom**: KD loss stayed high (6.28), low information per token
**Fix**: Reduce to 1.5

### Re-quantization After LoRA Merge
**Commit**: `350f82c` → `4774cf9`
**What**: merge_and_requantize: W_dense = W_ternary + LoRA, then symmetric_ternary(W_dense)
**Symptom**: PPL 17.86, model collapsed. CE loss improved during training (6.91→4.03) but re-quantization destroyed it
**Fix**: Use merge_lora_to_weights (dense, no re-ternarize)

### Per-element Lambada (2.5 GB)
**Commit**: Pre-history (early design)
**What**: Tequila with lambada_granularity="per_element": Lambada [out_f, in_f] per layer
**Symptom**: 2.5 GB RAM just for Lambada params for 7B model
**Fix**: Default to per_channel (6 MB total)

---

---

## Experiment 10: OBC + Activation-Aware Thresholds + LoRA v3

**Date**: 2026-05-06
**Hardware**: Colab T4
**Config**: PTBitNetConfig(symmetric, outliers=1%, OBC row compensation, activation-aware column weights) + LoRA (rank=64, distill=0.1, T=1.5, lr=5e-5, steps=1000, dense merge)
**Commit**: `b7b5fec`

### Results

| Metric | Baseline FP16 | PT-BitNet + OBC + LoRA | Ratio |
|--------|--------------|------------------------|-------|
| PPL (diverse texts) | 3.16 | 4.04 | **1.281x** |
| Gen quality | 82/100 | 88/100 | 1.07x |
| Jaccard overlap | — | 0.19 | — |
| Repetition | 0.00 | 0.072 | — |
| Speed | — | 21.2 tok/s | — |
| Pipeline time | — | 54.6 min | — |
| Verdict | — | **GOOD** | |

### Training Dynamics

| Step | CE Loss | KD Loss |
|------|---------|---------|
| 1 | 6.81 | 10.75 |
| 500 | 6.08 | 15.12 |
| 1000 | 4.50 | 11.31 |

### Comparison vs Previous Best (Exp 9: LoRA v3 without OBC)

| Metric | Exp 9 (no OBC) | Exp 10 (OBC) | Delta |
|--------|---------------|--------------|-------|
| PPL ratio | **1.239x** | 1.281x | +0.042x worse |
| Gen quality | 88/100 | 88/100 | same |
| Pipeline time | 34.7 min | 54.6 min | +20 min slower |

### Root Cause Analysis

OBC and activation-aware thresholds were expected to improve PPL ratio from 1.239x → 1.12x. Instead, PPL regressed. Three likely mechanisms:

1. **OBC over-compensation on small models**: The Hessian-weighted per-row bias correction is derived from GPTQ's block-wise framework (designed for 4-bit). For ternary (1.58-bit), the error per row is larger, and the OBC correction magnitude may overshoot — turning a quantization error into a systematic bias.

2. **OBC-LoRA interaction**: LoRA was previously trained on ternary weights without OBC compensation. OBC changes the ternary error pattern that LoRA learns to correct. The LoRA adapters may be correcting for the "wrong" error distribution now.

3. **Activation-aware column weights are myopic**: Per-layer column importance (based on that layer's input activations) ignores cross-layer effects. A column that looks "unimportant" for layer N's output may be critical for layer N+1's input representation.

### Disposition

OBC and activation-aware thresholds are **not recommended** for Phi-2 scale models. The simpler PT-BitNet + LoRA v3 configuration (Exp 9) remains the best result (PPL ratio 1.239x). OBC code is kept but the default pipeline should use `calibration_acts=False` for small models.

---

## Experiment 11: WikiText-2 Standardized Evaluation + FFN LoRA

**Date**: 2026-05-07
**Hardware**: Colab T4
**Commit**: `63354cd` (WikiText-2), `859fdea` (FFN LoRA)
**Config**: PT-BitNet (sym, outliers=1%, comp=30) → LoRA (rank=64, steps=1000, attention-only) → export (sharded) → load → eval

### What Changed

Switched from 8 custom diverse texts to **WikiText-2 test set** (500 lines) as the standard benchmark metric. This is what quantization papers (GPTQ, SpQR, PB-LLM) use — makes our numbers comparable.

Also added **FP16 baseline** measured BEFORE quantization so degradation is always relative to the same model on the same data.

### Results (attention-only LoRA, max_len=128)

| Metric | FP16 | Ternary + LoRA | Degradation |
|--------|------|----------------|-------------|
| WikiText-2 PPL (500 lines) | 27.39 | 33.31 | **1.216x** (+21.6%) |
| Quick 8-text PPL | — | 4.24 | — |
| INT2 integrity | — | 0 error | — |
| Pipeline time | — | 47.1 min | — |

### Analysis

**1.216x on WikiText-2 aligns with 1.239x on diverse texts from Exp 9** — consistent degradation across different evaluation methods. The ternary model is losing about 20-24% in perplexity terms across the board.

**WikiText-2 PPL is high even for FP16 (27.39)** because Phi-2 is optimized for code/reasoning, not language modeling. The absolute number matters less than the ratio.

### Next: FFN LoRA

LoRA was extended from attention-only (96 layers) to all linear layers (192 layers) by adding `dense`, `fc1`, `fc2`, `gate_proj`, `up_proj`, `down_proj` to target modules. This doubles LoRA coverage. Expected improvement: 3-7 percentage points (degradation from 1.22x → 1.15-1.18x).

---

## Summary: What We Learned

**Total experiments**: 11 full + 11 technique failures = 22 documented attempts
**Successful**: PT-BitNet (sym+outliers+comp), LoRA training dynamics, LoRA v3 dense merge, WikiText-2 evaluation, sharded export, checkpoint/resume
**Failed**: Tequila, QAT, ITF+outliers, SSR, AGA, ITF-clamped, re-quantize, high-distill-KD, soft-temperature, per-element-Lambada, OBC+act-aware (regressed PPL)

### What works
- Symmetric ternary + 1% outliers → solid baseline (PPL 1.22x WikiText-2)
- Hessian compensation on lm_head → small but consistent gain
- GPU teacher + batched pre-computation → ~10x faster, no CPU RAM cost
- LoRA KD from FP16 teacher → CE loss decreases (8.02→3.91), model learns
- Keep LoRA separate (dense merge not needed with INT2 export)
- Sharded safetensors save → no RAM OOM during export
- Checkpoint/resume → survives Colab disconnects, skips completed stages
- WikiText-2 with matching max_len → reproducible, comparable metric
- Vectorized grid search → identical quality, 8x fewer kernel launches

### What doesn't work (with root cause)
- **Tequila with PTQ**: Denormalization + re-decomposition conflict (fundamental)
- **QAT with ternary**: sign() quantization incompatible (binary ≠ ternary)
- **ITF + outliers**: Row statistics destabilize closed-form denominator
- **Re-quantization after LoRA**: Breaks complementary ternary+LoRA relationship
- **High distill_weight (≥0.5)**: Pulls model away from correct predictions
- **Soft KD targets (T≥3)**: Near-uniform distribution, negligible information
- **SSR column reordering**: Inverse permutation bug scrambles weights
- **Clamped ITF**: Too restrictive to find optimal solutions
- **Per-element Lambada**: Memory-prohibitive (2.5 GB for 7B)
- **Full state_dict save**: System RAM exhaustion on Colab (fixed: sharded)
- **Stale custom_params**: Cached params corrupt fresh model loads
- **OBC + activation-aware thresholds (small models)**: Over-compensation + LoRA interaction regressed PPL from 1.239x to 1.281x

### Open questions
- Can FFN LoRA push degradation below 1.15x? (testing now)
- Can STE block-wise fine-tuning work (or is ternary STE too noisy)?
- What's the optimal calibration data for KD (WikiText vs instruct vs model-generated)?
- Would a 7B model benefit more (more weight redundancy)?
- Can we train from scratch with BitNet architecture?
