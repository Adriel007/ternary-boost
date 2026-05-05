# TODO

## Critical (pipeline doesn't work end-to-end)

### ITF Numerical Stability
The asymmetric `build_optimal_grid` produces extreme alpha/mu values for
degenerate rows (T all-zero or all-same-sign). Current fix clamps alpha to
[0.1×init, 10×init] and mu to [init-3α, init+3α], but this hasn't been
validated end-to-end. Need: proper bounds testing + fallback to symmetric
for unstable rows.

### SSR Column Permutation Bug
`structural_similarity_reorder` reorders columns before quantization but
the inverse permutation is likely incorrect. Output was garbled when SSR
was enabled. Currently **disabled**. Either fix inverse permutation or
remove SSR permanently.

### AGA Activation Collection Broken on GPU
`_collect_activations` registers hooks on nn.Linear but collected 0 layers
on T4 (Colab). Hooks fire on forward pass but model device vs input device
mismatch may cause silent failure. Either fix device handling or skip AGA.

### Colab T4 OOM During Tequila
Pipeline works through PT-BitNet + compensation (loss 5.72, 185s on T4)
but OOMs when Tequila starts. 15.6 GB VRAM filled by model (11 GB) +
optimizer states + forward activations. Need: memory profiling, model.cpu()
before save then back to GPU for Tequila.

### No Successful End-to-End Run
Pipeline has never completed a full run producing valid output. Base FP16
Phi-2 correctly answers "Paris" with "Question:...\nAnswer:" format but
ternary model output is garbage. Root cause: ITF numerical instability.

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
