# TODO

Items that require external infrastructure, upstream changes, or additional
hardware before they can be implemented.

## Chat Module

### Native Thinking/Reasoning Tokens

Ternary models (BitNet b1.58, LLaMA quantized) do not natively support dedicated
reasoning tokens (`<|thinking|>`, `</|thinking|>`) in their tokenizer vocabularies.
The current `/thinking` implementation uses prompt-level chain-of-thought
prefixing, which works but is not as clean as native thinking support.

**Required to implement natively:**
- Fine-tune the ternary model with thinking tokens added to the vocabulary
- Requires QAT-aware training with the extended tokenizer
- ~10B tokens of chain-of-thought training data

### bitnet.cpp Streaming

The current `bitnet.cpp` backend calls the binary as a subprocess (`subprocess.run`),
which means no streaming token output. Streaming requires a Python binding
(waiting on upstream `bitnet.cpp` Python bindings) or implementing a socket-based
streaming protocol.

### Multi-GPU Tensor Parallelism

For models >7B parameters, tensor parallelism across multiple GPUs would be
needed. This requires integration with `accelerate` or `vLLM` serving
infrastructure — both are outside the scope of this compression-focused project.

## Pipeline

### ParetoQ Quantization Function Search

The ParetoQ paper describes a search over quantization functions (uniform,
elastic, stretched-elastic, LSQ, LSQ+) per layer. The current implementation
uses a fixed per-bit-width mapping. Implementing the full search requires:
- Per-layer architecture analysis
- Validation-based quantization function selection
- Additional compute budget (≈2× training time)

### ZeroQAT Sparse Sensitive Parameters

The ZO fine-tuning paper (arXiv:2406.02913) identifies 0.1% "sensitive parameters"
during pre-training. In a post-training pipeline, this identification must be
done via gradient-based sensitivity analysis at the start of QAT:
- Run one forward + backward pass (full precision) per layer
- Rank parameters by gradient magnitude
- Select top-0.1% for ZO updates
- Requires ~1× extra forward pass cost

### Activation Quantization

All current stages are weight-only quantization. Activation quantization
(8-bit or ternary activations) would further reduce memory and compute:
- Requires quantization-aware activation functions
- Needs calibration data for activation range estimation
- Potential 2–4× additional speedup

### KV-Cache Compression

The KV cache in long-context generation dominates memory. Ternary quantization
of KV cache entries is an orthogonal research direction:
- Per-head quantization granularity
- Requires modification to `DynamicCache` / `StaticCache` in transformers

## Evaluation

### Full Benchmark Suite

Current evaluation covers MMLU, HellaSwag, ARC-Easy, and ARC-Challenge.
Additional benchmarks desirable:
- GSM8K (mathematical reasoning)
- HumanEval (code generation)
- TruthfulQA (factuality)
- BBH (broad reasoning)
- IFEval (instruction following)

### Latency Benchmarks

End-to-end latency measurement under realistic serving conditions:
- Time-to-first-token (TTFT)
- Inter-token latency (ITL)
- Throughput (tokens/sec) at batch sizes 1, 4, 8
- Requires dedicated benchmarking harness with warm-up iterations

## Export

### bitnet.cpp Python Bindings

Currently calls `bitnet` binary via subprocess. Native Python bindings via
`ctypes` or `cffi` would enable:
- In-process streaming generation
- Direct memory access (no serialization overhead)
- Batch inference
- Depends on upstream `microsoft/BitNet` exposing a C API

### GGML / llama.cpp Format

In addition to bitnet.cpp, exporting to GGML format would enable inference
on llama.cpp's broader hardware support:
- Metal (Apple Silicon)
- Vulkan (AMD GPUs)
- WebAssembly (browser)
- Requires ternary-specific changes to GGML's quantization format
