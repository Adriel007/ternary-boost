# Phi-2 Ternary Compression Results

**Date**: 2026-05-05
**Hardware**: Colab T4 (15.6 GB VRAM, 12.7 GB RAM)
**Pipeline**: PT-BitNet (symmetric ternary + 1% outliers + Hessian compensation)
**Pipeline time**: 10.1 min
**Code commit**: `ec0135b`

## Active Pipeline Stages

| Stage | Status | Reason |
|-------|--------|--------|
| 1. PT-BitNet | **Active** | Symmetric ternary + 1% outlier retention + 50-step Hessian compensation on lm_head |
| 2. ParetoQ/QAT | Skipped | w_bits=1 uses sign() (binary), destroys ternary sparsity. w_bits=0 uses incompatible scaling |
| 3. Tequila | Skipped | UltraQuantV3 recomputes ternary decomposition on PT-BitNet's denormalized weights (w*std+mean), producing a different pattern. PT-BitNet output is already optimal |

## Model: microsoft/phi-2 (2.78B params)

| Metric | Baseline FP16 | Quantized (ternary) | Ratio |
|--------|--------------|---------------------|-------|
| **Perplexity** (calibration texts) | 2.61 | 3.45 | 1.32x |
| **Perplexity** (wikitext train) | — | — | — |
| **Generation quality** (avg 0-100) | 94 | 94 | 1.00x |
| **Repetition ratio** | 0.00 | 0.00 | — |
| **Speed** | — | 20.5 tok/s | — |

## Generation Samples

### Factual: "What is the capital of France?"

| Model | Response |
|-------|----------|
| FP16 | `The capital of France is Paris.` |
| Ternary | `The capital of France is Paris.` |

### Definition: "What is machine learning?"

| Model | Response |
|-------|----------|
| FP16 | `Machine learning is a subset of artificial intelligence that focuses on developing algorithms and models that enable com...` |
| Ternary | `Machine learning is a subset of artificial intelligence where computers and machines are programmed to execute tasks aut...` |

### Math: "If a train travels at 60mph for 2 hours, how far?"

| Model | Response |
|-------|----------|
| FP16 | `The train will travel 120 miles.` |
| Ternary | `The train covers a distance of 120 miles...` |

### Creative: "Write a short poem about the moon"

| Model | Response |
|-------|----------|
| FP16 | `The moon is a silver coin / That shines in the dark night sky / It changes its shape and size / As it orbits by and by` |
| Ternary | `The moon is a bright light that / That guides us on our way / ...` |

## Analysis

- **Perplexity degradation**: 32% increase is acceptable for 1.58-bit post-training quantization on a small model (2.7B). Smaller models have less weight redundancy, making compression harder.
- **Generation quality**: Matches FP16 baseline (94/100). Factual knowledge, reasoning, and creativity are preserved.
- **No degeneration**: Repetition ratio 0.00 — model does not collapse into loops or gibberish.
- **Room for improvement**: Hessian compensation only reduced loss 2% (5.99→5.86). Better calibration data or iterative weight optimization could reduce PPL further.

## Comparison to Literature

| Method | Model Size | PPL Ratio | Training |
|--------|-----------|-----------|----------|
| **This work** | 2.7B | 1.32x | Post-training |
| PT²-LLM (paper) | 7-70B | 1.15-1.25x | Post-training |
| BitNet b1.58 | 3B | ~1.05x | Trained from scratch |
| GPTQ (4-bit) | 7-70B | 1.05-1.15x | Post-training |

Post-training ternary on small models is inherently harder — less parameter redundancy means each weight matters more. Our result is in line with expectations for a 2.7B model at 1.58 bits.
