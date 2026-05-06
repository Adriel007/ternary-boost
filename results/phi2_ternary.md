# Phi-2 Ternary Compression Results

**Date**: 2026-05-05
**Hardware**: Colab T4 (15.6 GB VRAM, 12.7 GB RAM)
**Code commit**: `091ade2`

## Active Pipeline Stages

| Stage | Status | Reason |
|-------|--------|--------|
| 1. PT-BitNet | Active | Symmetric ternary + 1% outlier retention + 50-step Hessian compensation |
| 2. ParetoQ/QAT | Removed | sign() binary destroys ternary sparsity |
| 3. Tequila | Removed | UltraQuantV3 re-decomposition incompatible with PT-BitNet denorm |
| 4. LoRA | Active (v1 results below) | Rank-32 KD from FP16 teacher on CPU |

## Model: microsoft/phi-2 (2.78B params)

### PT-BitNet only (no LoRA)

| Metric | Baseline FP16 | Quantized (ternary) | Ratio |
|--------|--------------|---------------------|-------|
| Perplexity (simple texts) | 2.61 | 3.45 | 1.32x |
| Perplexity (diverse texts) | — | — | — |
| Generation quality (avg 0-100) | 94 | 94 | 1.00x |
| Repetition ratio | 0.00 | 0.00 | — |
| Speed | — | 20.5 tok/s | — |
| Pipeline time | — | 10.1 min | — |

### PT-BitNet + LoRA (rank=32, v1)

| Metric | Baseline FP16 | Quantized + LoRA | Ratio |
|--------|--------------|------------------|-------|
| Perplexity (diverse texts) | 3.16 | 3.96 | 1.256x |
| Generation quality (avg 0-100) | 86 | 90 | 1.05x |
| Repetition ratio | 0.00 | 0.005 | — |
| Speed | — | 16.2 tok/s | — |
| Pipeline time | — | 19.2 min | — |

**LoRA v1 hyperparameters (suboptimal):**
- Rank: 32, alpha: 64, dropout: 0.05
- distill_weight: 0.5, temperature: 3.0, lr: 2e-4, steps: 500
- max_seq_length: 64, accumulation: 8
- Target: all 7 projection types (96 layers)

**Analysis:**
- PPL ratio improved only 5% (1.32x → 1.26x). Expected 15-25%.
- CE loss increased during training (6.85 → 6.48 vs. without LoRA baseline of ~5.86)
- KD loss remained high (6.28 at step 500) — student couldn't match teacher distribution
- Root cause: distill_weight=0.5 pulled model away from correct predictions
- T=3.0 made teacher targets too uniform (low information per token)
- LoRA on FFN layers (gate/up/down) may have added noise without benefit
- Model became verbose — adds exercises/questions after every response

### Planned: LoRA v2 (improved config)

| Parameter | v1 (bad) | v2 (planned) | Reason |
|-----------|----------|--------------|--------|
| distill_weight | 0.5 | 0.1 | CE should dominate, KD as gentle guide |
| temperature | 3.0 | 1.5 | Sharper teacher = more informative |
| rank | 32 | 64 | More capacity for ternary error gap |
| lr | 2e-4 | 5e-5 | More stable, less overshoot |
| steps | 500 | 1000 | More training with lower LR |
| target_modules | All 7 types | q,k,v,o only | Attention benefits most from LoRA |
| dropout | 0.05 | 0.0 | Teacher already regularizes |
| max_seq_length | 64 | 128 | More context = better KD signal |
| re-quantize | No | Yes | Merge LoRA → re-ternarize → preserve sparsity |

## Generation Samples

### PT-BitNet only

| Prompt | Response |
|--------|----------|
| "Capital of France?" | `The capital of France is Paris.` |
| "Machine learning?" | `Machine learning is a subset of artificial intelligence...` |
| "Train 60mph 2h?" | `The train covers a distance of 120 miles...` |

### PT-BitNet + LoRA v1

| Prompt | Response |
|--------|----------|
| "Capital of Japan?" | `Tokyo` (adds exercise about APEC) |
| "WWII end?" | `World War II ended in 1945.` (adds exercise about countries involved) |
| "Chemical symbol water?" | `H2O` (adds exercise about CO2) |
| "Planets solar system?" | `There are eight planets...` (adds question about Jupiter moons) |
| "Photosynthesis?" | `Photosynthesis is a process in plants where they use sunlight...` (correct, adds follow-up) |
| "Haiku about ocean?" | Generates Python code instead of a haiku |

LoRA v1 produces factually correct answers but is overly verbose, adding random exercises after each response.
