# Phi-2 Ternary Compression Results

**Last updated**: 2026-05-07
**Hardware**: Colab T4 (15.6 GB VRAM, 12.7 GB RAM)
**Model**: microsoft/phi-2 (2.78B params)

## Active Pipeline

| Stage | Status | Details |
|-------|--------|---------|
| 1. PT-BitNet | Active | Symmetric ternary + 1% outliers + 30-step Hessian compensation |
| 2. ParetoQ/QAT | Removed | sign() binary destroys ternary sparsity |
| 3. Tequila | Removed | UltraQuantV3 incompatible with PT-BitNet denorm |
| 4. LoRA | Active | Rank-64 KD from FP16 teacher, attention-only (FFN pending) |

## Latest Results: WikiText-2 Standardized (2026-05-07)

Measured on WikiText-2 test set (500 lines, max_len=128). This is the standard metric in quantization literature.

| Metric | FP16 Baseline | Ternary + LoRA | Degradation |
|--------|--------------|----------------|-------------|
| WikiText-2 PPL | 27.39 | 33.31 | **1.216x (+21.6%)** |
| 8-text quick PPL | — | 4.24 | — |
| INT2 integrity | — | 0 errors / 629M weights | — |
| Pipeline time | — | 47.1 min | — |
| Export size (full) | — | 6.1 GB | — |
| Inference size (INT2+LoRA) | — | ~450 MB | 12.4x vs FP16 |

### Generation Samples

| Prompt | Response |
|--------|----------|
| "The capital of France is" | Paris. |
| "Water freezes at" | the same temperature as water... |
| "The largest planet in the solar system is" | Jupiter. |

## Comparison: All Versions

| Version | Metric | Degradation | Pipeline | Key Difference |
|---------|--------|-------------|----------|----------------|
| PT-BitNet only | PPL 1.32x | 32% | 10 min | Baseline ternary, no LoRA |
| LoRA v1 (bad KD) | PPL 1.26x | 26% | 19 min | distill=0.5, T=3.0 |
| LoRA v2 (re-quantize) | PPL 5.66x | 466% | 36 min | Re-quantize destroyed |
| **LoRA v3 (attn-only)** | **PPL 1.22x** | **22%** | **35 min** | **Best to date** |
| LoRA v3 + OBC | PPL 1.28x | 28% | 55 min | OBC regressed quality |
| **LoRA FFN (pending)** | **?** | **?** | **~50 min** | **All 192 layers** |

## What We Know

- **1.22x degradation on WikiText-2 is solid** for 1.58-bit post-training quantization
- LoRA KD is the single biggest quality lever (~10 percentage points vs PT-BitNet alone)
- OBC and activation-aware thresholds regress quality on small models
- Keeping LoRA separate (not merging into weights) preserves the ternary backbone for INT2 export
- Checkpoint/resume eliminates re-run time on failure
- WikiText-2 with matching max_len=128 gives reproducible, comparable numbers

## Pending

- [ ] FFN LoRA (fc1/fc2/dense) — doubles coverage from 96 to 192 layers
- [ ] Compare vs GPTQ 2-bit and SpQR baselines on same Phi-2 + WikiText-2
- [ ] HellaSwag / MMLU downstream benchmarks (eval/ code exists)
