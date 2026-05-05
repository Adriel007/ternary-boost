# TODO

## Critical (pipeline doesn't work end-to-end)

### Save/Load/Bake Round-Trip Corrupts Quality — FOUND 2026-05-05
**Ablation on Colab T4 proved:** All 5 quantization variants (symmetric,
ITF, +outliers, +compensation) correctly answer "Paris":
```
BASE:      "Paris." ✅
SYMMETRIC: "The city of Paris is located in France." ✅
SYM+COMP:  "The capital of France is Paris." ✅ (best)
ITF:       "The city of Paris is a densely populated city..." ✅
ITF+COMP:  "The capital of France is Paris." ✅ (best)
```
**But the full tchat pipeline (PT-BitNet → Tequila → Bake → Save → Load → Chat)
produces garbage.** The quantization itself is validated. The bug is in
one of: Tequila layer replacement, baking effective_weight computation,
sharded save/load round-trip, or layer reconstruction.

### ITF Falls Back to Symmetric for ALL Layers with Outliers
When `outlier_fraction=0.01`, ITF produces extreme values for 96/96 layers
and falls back to symmetric. Reason: outliers create large weight magnitudes
that destabilize the closed-form grid optimization. Without outliers,
ITF works perfectly (test 4). Fix: apply outlier retention BEFORE ITF,
not during.

### No Native Kernel Integration
Baked nn.Linear weights are ternary but inference uses float matmul
(same speed as FP16). The 3-6× speedup from BitNet papers requires
microsoft/BitNet GGUF I2_S format + llama.cpp SIMD kernels. We have
the conversion path documented but not implemented.

## Near-term (post-stability)

### Quality Validation
- Ablation study: base vs symmetric vs ITF vs ITF+outliers vs ITF+compensation
- Perplexity on WikiText-2
- Zero-shot accuracy (MMLU, HellaSwag, ARC)
- Compare against PT²-LLM paper results when code is released

### Symmetric Quantizer Too Slow on CPU
`_symmetric_ternary` iterates 256 candidates × full matrix ops × 96 layers.
CPU: ~18 min. GPU: <1 min. The ITF (closed-form, 10 iterations) is much
faster but unstable. Fix ITF stability, then symmetric becomes unnecessary.

### Adaptive Compensation Steps
CPU: 10 steps (~14 min). GPU: 50 steps (~3 min). Should auto-scale based
on convergence delta, not fixed steps. Currently converges 5.84→5.72 on GPU.

### CPU-Only Full Pipeline Time
~22 min for 2.7B model (PT-BitNet 1.5 min + comp 14 min + Tequila 5 min).
Acceptable for one-time compression. GPU (T4): ~8 min.

## Backlog

### Native Inference Kernel
Integration with `microsoft/BitNet`:
1. Export baked weights → safetensors → `convert-helper-bitnet.py` → GGUF I2_S
2. Call `llama-cli` binary (subprocess, no streaming) OR
3. Write Python bindings for in-process inference
4. Target: 3-6× CPU speedup

### Column Reordering (SSR) — Fix or Remove
PT²-LLM Section 3.3, Eq. 14-16. Groups similar columns for compact blocks.
Current implementation produces incorrect inverse permutation.
Need: unit test for round-trip (reorder → inverse → compare with original).

### Activation-aware Grid Alignment (AGA) — Fix or Remove
PT²-LLM Section 3.2, Eq. 13. Minimizes ||WX - (αT+μ)X|| instead of ||W - (αT+μ)||.
Uses calibration activations X. Currently broken on GPU (0 layers collected).
Need: fix device handling in `_collect_activations`.

### Larger Model Testing
Current: Phi-2 (2.7B). Target: Mistral-7B, Falcon-7B, Phi-3 Medium (14B).
Requires Colab A100 or local GPU with >24 GB VRAM.

### Benchmark Suite
- MMLU, HellaSwag, ARC-Easy, ARC-Challenge (via lm-eval)
- WikiText-2 perplexity
- GSM8K, HumanEval, TruthfulQA
- Latency: TTFT, ITL, throughput

### bitnet.cpp Python Bindings
Replace subprocess call with `ctypes`/`cffi` bindings for:
- In-process streaming generation
- Direct memory access
- Batch inference

## AirLLM Integration (Research)

### What is AirLLM?
[lyogavin/AirLLM](https://github.com/lyogavin/airllm) — Apache 2.0, 7k+ stars. Layer-by-layer
model loading: loads one transformer layer at a time from disk, processes it, frees memory.
Achieves 70B inference on 4GB GPU, 405B on 8GB. CPU-only mode since v2.10.1. Built-in
4bit/8bit compression via bitsandbytes with profiling tools.

### How it helps ternary-boost

**PT-BitNet stage** — Already processes layers one at a time, BUT requires loading the
full model first (~11 GB for 2.7B, ~28 GB for 7B float32). AirLLM-style loading would
eliminate this peak: load layer k → quantize → save → free → load layer k+1. Memory peak
drops from full model size to single layer (~50 MB for 7B). This enables 7B-70B pipeline
on consumer hardware.

**Compensation stage** — AirLLM's layer-by-layer forward pass could enable lm_head
compensation on low-RAM hardware. Forward hook captures last hidden state → lm_head update.
Transformer body loaded/unloaded layer-by-layer during forward. Not yet compatible with
our gradient-based compensation (needs backward pass on lm_head).

**Tequila stage** — Similar: forward passes with per-layer loading. Tequila's per-layer
`update_lambada()` does its own backward internally. Could work if forward is adapted to
AirLLM-style streaming.

**User's hardware (15 GB RAM, 12-core CPU):** With AirLLM integration, could run pipeline
on 7B-14B models. Currently limited to 2.7B due to peak RAM.

### Caveats & Limitations

| Factor | Impact |
|--------|--------|
| Disk I/O overhead | ~2-3× slower than in-memory (reads layers from disk each forward pass) |
| Gradient compatibility | AirLLM designed for inference, not training. Compensation backward needs adaptation |
| Custom layer types | AirLLM uses standard HF layers. Our QuantizeLinear/UltraQuantLinear not natively supported |
| bitsandbytes dep | Only needed if using AirLLM's built-in compression (we have our own) |
| Model splitting pre-processing | AirLLM requires pre-splitting model into per-layer files. One-time cost |

### Implementation Plan (if pursued)

1. **PT-BitNet integration** (lowest effort, highest impact):
   - Use AirLLM for initial model loading: `AirLLM.from_pretrained(model, layer_by_layer=True)`
   - Iterate: for each layer, load weight tensor, run ternary_quantize_vectorized, save
   - Replace our `_find_quantizable_linears` + `apply_pt_bitnet` loop
   - Peak RAM: ~50 MB per layer instead of full model

2. **Compensation adaptation** (medium effort):
   - Implement AirLLM-style forward pass: for each batch, load layers 0..N sequentially
   - Capture last hidden state via hook on final layer
   - Run lm_head backward normally (lm_head is small, <1 GB)
   - Rest of transformer uses no_grad + sequential loading

3. **Full pipeline AirLLM mode** (high effort):
   - `tchat --model llama-7b --airllm` flag
   - Entire pipeline uses sequential layer loading
   - Tequila adapted to process layers one at a time

### Decision
Not implemented yet. The immediate priority is making the ITF produce good quality on
2.7B models. Once that's validated, AirLLM integration would unlock 7B+ models on the
user's hardware. Added to backlog for post-stability phase.

## Done
- [x] Per-channel Lambada (6 MB vs 2.5 GB per-element)
- [x] Sharded safetensors save (800 MB chunks, Colab-safe)
- [x] Incremental checkpointing (stage1/2/3 markers)
- [x] QAT removed from pipeline (incompatible with ternary, Tequila replaces it)
- [x] PT²-LLM paper reviewed and asymmetric ITF implemented
- [x] SpQR outlier retention (top 1% FP16)
- [x] GPTQ-style Hessian compensation on lm_head
- [x] tchat CLI with auto-compression
- [x] Colab notebook with git pull
