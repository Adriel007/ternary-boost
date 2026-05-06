# Phi-2 Ternary Compression Results

**Last updated**: 2026-05-06
**Hardware**: Colab T4 (15.6 GB VRAM, 12.7 GB RAM)

## Active Pipeline Stages

| Stage | Status | Reason |
|-------|--------|--------|
| 1. PT-BitNet | Active | Symmetric ternary + 1% outliers + OBC row compensation + activation-aware thresholds |
| 2. ParetoQ/QAT | Removed | sign() binary destroys ternary sparsity |
| 3. Tequila | Removed | UltraQuantV3 re-decomposition incompatible with PT-BitNet denorm |
| 4. LoRA | Active | Rank-64 KD from FP16 teacher, dense merge (no re-quantize) |

## Model: microsoft/phi-2 (2.78B params)

### PT-BitNet only (no LoRA, no OBC) — Experiment 6

| Metric | Baseline FP16 | Quantized (ternary) | Ratio |
|--------|--------------|---------------------|-------|
| Perplexity (simple texts) | 2.61 | 3.45 | 1.32x |
| Generation quality (avg 0-100) | 94 | 94 | 1.00x |
| Repetition ratio | 0.00 | 0.00 | — |
| Speed | — | 20.5 tok/s | — |
| Pipeline time | — | 10.1 min | — |

### PT-BitNet + LoRA v3 (rank=64, dense merge, no OBC) — Experiment 9

| Metric | Baseline FP16 | Quantized + LoRA | Ratio |
|--------|--------------|------------------|-------|
| Perplexity (diverse texts) | 3.16 | 3.91 | 1.239x |
| Generation quality (avg 0-100) | 82 | 88 | 1.07x |
| Repetition ratio | 0.00 | 0.000 | — |
| Speed | — | 19.2 tok/s | — |
| Pipeline time | — | 34.7 min | — |

### PT-BitNet (OBC + act-aware) + LoRA v3 — Experiment 10 [LATEST]

**Date**: 2026-05-06
**Commit**: `b7b5fec`

| Metric | Baseline FP16 | Quantized + LoRA | Ratio |
|--------|--------------|------------------|-------|
| Perplexity (diverse texts) | 3.16 | **4.04** | **1.281x** |
| Generation quality (avg 0-100) | 82 | 88 | 1.07x |
| Jaccard overlap | — | 0.19 | — |
| Repetition ratio | 0.00 | 0.072 | — |
| Speed | — | 21.2 tok/s | — |
| Pipeline time | — | 54.6 min | — |

**Verdict: GOOD** — minor quality loss, fully usable

### Training Dynamics (Experiment 10)

| Step | CE Loss | KD Loss |
|------|---------|---------|
| 1 | 6.81 | 10.75 |
| 100 | 7.80 | 21.00 |
| 500 | 6.08 | 15.12 |
| 1000 | 4.50 | 11.31 |

### Generation Samples (Experiment 10)

| Prompt | Response | Correct? |
|--------|----------|-----------|
| "Capital of Japan?" | `Tokyo` + exercises | ✅ |
| "WWII end year?" | `1945.` + EU exercise | ✅ |
| "Water symbol?" | `H2O` + formula + conductivity exercise | ✅ |
| "Planets count?" | `There are eight planets...` + Jupiter question | ✅ |
| "5 apples × $2?" | `$10` + discount follow-up | ✅ |
| "All dogs animals?" | `Yes, all dogs are animals.` | ✅ |
| "Photosynthesis?" | `Plants use sunlight → CO2 + water → glucose` | ✅ |
| "Moon vs Earth?" | `Earth larger, ~12,742 km vs ~1,737 km` | ✅ |
| "Lie ethical?" | `It depends on the context` | ✅ |
| "Haiku ocean?" | `Waves crashing on the shore...` (actual haiku!) | ✅ |

## Comparison: All Versions

| Version | PPL Ratio | Gen Quality | Pipeline | Key Difference |
|---------|-----------|-------------|----------|----------------|
| PT-BitNet only (no OBC) | 1.324x* | 94/100 | 10 min | Baseline ternary |
| LoRA v1 (bad KD) | 1.256x | 90/100 | 19 min | distill=0.5, T=3.0, rank=32 |
| LoRA v2 (re-quantize) | 5.656x | 72/100 | 36 min | Re-quantize destroyed |
| **LoRA v3 (dense, no OBC)** | **1.239x** | **88/100** | **35 min** | **Best PPL ratio** |
| LoRA v3 + OBC + act-aware | 1.281x | 88/100 | 55 min | OBC didn't help |

*Simple texts (baseline 2.61). All others use diverse texts (baseline 3.16).

## Analysis

### OBC + Activation-Aware Thresholds: Didn't Help

The OBC row compensation and activation-aware column weights were expected to improve PPL ratio from 1.239x → 1.12x or better. Instead, PPL ratio regressed to 1.281x. Possible reasons:

1. **Over-compensation**: OBC corrects per-row bias using Hessian diagonal. On small models (2.7B), the correction may be too aggressive relative to the ternary error structure.
2. **Interaction with LoRA**: LoRA was trained to compensate for a specific ternary error pattern. OBC changes that pattern → LoRA's learned correction may no longer match.
3. **Column weights distorting thresholds**: Activation-aware scaling of the ternary error may de-prioritize channels that are important for later layers (myopic per-layer optimization).
4. **Small model effect**: PT²-LLM reports best results on 7-70B models. Phi-2 (2.7B) has less parameter redundancy, making compensation techniques less effective.

### What Actually Works

- **LoRA v3 (dense merge, no OBC)** remains the best configuration: PPL ratio 1.239x
- **PT-BitNet only** is the fastest: 10 min, PPL ratio 1.324x
- Generation quality is consistently good across all working versions (88-94/100)
- All factual answers are correct across all versions

### Honest Assessment

The mathematical improvements we researched (OBC, activation-aware thresholds) didn't translate to real gains on Phi-2. The strongest result is **LoRA v3 without OBC** (PPL ratio 1.239x, 35 min).

For 2.7B models, post-training ternary is close to its ceiling. The techniques that work (outlier retention, LoRA KD) are standard. Our novel additions (OBC row compensation, activation-aware column weights) didn't beat the simpler baseline.
