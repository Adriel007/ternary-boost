# TODO

## Active Pipeline Status

```
FP16 Model → PT-BitNet (sym ternary + outliers + lm_head compensation) → [LoRA fine-tuning] → Save
```

| Stage | Status | Description |
|-------|--------|-------------|
| 1. PT-BitNet | ✅ Active | Symmetric ternary + 1% outliers + Hessian compensation |
| 2. ParetoQ/QAT | ❌ Removed | w_bits=1 uses sign() (binary), destroys sparsity |
| 3. Tequila | ❌ Removed | UltraQuantV3 re-decomposition incompatible with PT-BitNet denorm |
| 4. LoRA | ✅ Active | Rank-decomposition adapters + KD from FP16 teacher |

## Current Results (Phi-2, T4, 2026-05-05)

| Metric | FP16 | Ternary (no LoRA) | Ternary + LoRA (rank=32) |
|--------|------|-------------------|--------------------------|
| PPL (diverse) | 2.61 | 3.45 (1.32x) | **TBD** |
| Gen quality | 94/100 | 94/100 | **TBD** |
| Pipeline time | — | 10 min | **TBD** (~20 min est.) |
| Repetition | 0.00 | 0.00 | **TBD** |

## Immediate (validate LoRA)

- [ ] Run colab_test.py with ENABLE_LORA=True on T4
- [ ] Measure PPL improvement from LoRA (expect 1.05-1.15x ratio)
- [ ] If LoRA helps: tune rank, steps, distill_weight
- [ ] If LoRA doesn't help: investigate KD temperature, teacher/student mismatch

## Near-term (improve quality)

### STE Block-wise Fine-tuning
Implemented in `pt_bitnet/src/pt_bitnet/ste.py`. Processes one transformer block
at a time, matching hidden states against FP16 teacher via MSE loss + STE gradients.
- [ ] Test on Phi-2 (low priority — LoRA is safer bet)
- [ ] Fix: deadzone gradient scaling may prevent weight recovery
- [ ] Fix: teacher hidden states collected with original model; drift as blocks change

### Better Calibration Data
Current: WikiText-2 samples (diverse but loss drops only 2% during compensation).
Better approaches:
- [ ] Instruction-tuning data (Alpaca, Dolly) for more relevant loss signal
- [ ] Model-generated data (LLM-QAT approach) — use FP16 model's own generations
- [ ] Active sample selection — pick examples with highest gradient variance

### Re-quantize after LoRA Merge
- [ ] Merge LoRA weights into base → re-quantize to ternary
- [ ] This recovers sparsity while keeping LoRA quality gains
- [ ] Risk: re-quantization may lose some of the LoRA benefit

### Adaptive Ternary Thresholds
Current threshold: alpha/2 (mean of active weights divided by 2).
- [ ] Learned per-channel thresholds (small MLP predicting delta from weight stats)
- [ ] Activation-aware thresholds (use calibration data to set delta)

## Medium-term (scale up)

### 7B Model Support
Requires GPU with >24 GB VRAM (A100, L4, RTX 4090).
- [ ] Test LoRA + Ternary on Mistral-7B
- [ ] LoRA with CPU teacher (pre-compute logits) for low-VRAM scenarios
- [ ] Block-wise processing for 7B on 16 GB GPUs

### Native Kernel Integration
Without this, ternary weights run at FP16 speed — no actual speedup.
- [ ] Export ternary weights → GGUF I2_S format (microsoft/BitNet)
- [ ] Integrate with llama.cpp for CPU inference
- [ ] Target: 3-6× speedup on CPU

### Benchmark Suite
- [ ] MMLU, HellaSwag, ARC-Easy, ARC-Challenge (lm-evaluation-harness)
- [ ] WikiText-2 perplexity
- [ ] Compare against PT²-LLM and BitNet b1.58 baselines
- [ ] Publish if results are competitive (PPL ratio <1.10x on 7B+)

## Backlog

### SSR Column Reordering
PT²-LLM Section 3.3. Groups similar columns for compact blocks.
- [ ] Fix inverse permutation (currently produces garbled output)
- [ ] Or remove — SSR provides marginal gains vs complexity

### AGA Activation Alignment
PT²-LLM Section 3.2. Aligns quantization grid with calibration activations.
- [ ] Fix GPU device handling (now collected 96 layers correctly)
- [ ] Test if AGA improves over symmetric alone
- [ ] Blocked by: symmetric is already good, AGA adds complexity

### AirLLM Integration
Layer-by-layer loading for large models on small GPUs.
- [ ] PT-BitNet stage: layer-by-layer quantization (peak RAM = 1 layer)
- [ ] Compensation: forward pass with sequential layer loading
- [ ] LoRA: sequential forward for teacher logit collection

### CPU Pipeline
- [ ] Full CPU-only pipeline (PT-BitNet + LoRA without GPU)
- [ ] Current: PT-BitNet CPU path works, LoRA needs GPU for KD
- [ ] Option: distill on GPU, infer on CPU (offline distillation)

## Done
- [x] PT²-LLM ITF implementation (closed-form with outlier protection)
- [x] SpQR outlier retention (top 1% FP16)
- [x] GPTQ-style Hessian compensation on lm_head
- [x] Sharded safetensors save (800 MB chunks, Colab-safe)
- [x] Incremental checkpointing (stage1-4 markers)
- [x] QAT removed from pipeline (binary sign destroys ternary)
- [x] Tequila removed from pipeline (denorm incompatibility)
- [x] Tequila baking formula fixed (weight=A, bias=sum(B*L))
- [x] ITF + outliers skip (ITF destabilizes with outlier-zeroed positions)
- [x] Activation collection device fix (model CPU + input CUDA → 0 layers)
- [x] tchat CLI with auto-compression
- [x] Colab ablation script (isolates each stage)
- [x] Colab test script (full pipeline with perplexity + generation metrics)
- [x] Realistic quality verdict thresholds (1.32x PPL = GOOD for 1.58-bit)
- [x] Phi-2 benchmark results saved (results/phi2_ternary.md)
- [x] LoRA + Ternary fine-tuning module (lora.py)
- [x] Block-wise STE fine-tuning module (ste.py)
- [x] LoRA integrated into pipeline as Stage 4
