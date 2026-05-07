# Ternary + LoRA Separate Inference — Detailed Implementation Plan

**Goal**: Achieve real compression (7x smaller than FP16) while maintaining PPL ratio ~1.24x, by storing ternary weights and LoRA adapters **separately** — never merging them into dense weights.

**Status**: Pre-implementation research. No code yet.

---

## 1. Background & Motivation

### 1.1 The Merge Problem

Our current best pipeline (Exp 9) does:

```
FP16 → PT-BitNet (ternary) → LoRA KD training → merge_lora_to_weights()
                                                      ↓
                                              W_dense = W_ternary + LoRA_correction
                                              (dense bf16, 5.6 GB — zero compression!)
```

The merge destroys ternary sparsity. We trade compression for quality — and lose both the size benefit AND the inference speedup potential.

### 1.2 The Proposed Solution

Keep ternary and LoRA **as separate components** during storage and inference:

```
Storage:
  W_ternary  → INT2 packed (~0.67 GB)  ─┐
  row_alphas → FP16 (~1 MB)             ─┤  Total: ~0.8 GB
  LoRA A, B  → FP16 (~125 MB)           ─┘  (7x smaller than 5.6 GB)

Inference:
  y = alpha ⊙ (W_ternary @ x) + (α_lora / rank) · B @ (A @ x)
      └── ternary path (no multiplications!) ──┘   └── LoRA correction ──┘
```

### 1.3 Academic Context

**What exists:**
- **BitNet b1.58** (Ma et al., 2024): From-scratch ternary training. Proves ternary weights work but doesn't address post-training quality recovery.
- **PT²-LLM** (Yan et al., ICLR 2026): Post-training asymmetric ternary. Achieves 1.15-1.25x PPL on 7-70B with strictly ternary weights. No learned correction component.
- **QLoRA** (Dettmers et al., 2023): NF4 quantized weights + FP16 LoRA adapters for fine-tuning. Proves "quantized base + learned correction" is viable, but designed for fine-tuning efficiency, not inference compression.
- **T-MAC** (Microsoft, 2024): LUT-based inference for 1-4 bit weights on CPUs. Shows 2-bit ternary inference is 3-6x faster via table lookups.
- **llama.cpp GGML formats**: Q1_0 for 1-bit binary (±1 only), Q2_K/Q3_K for 2-3 bit. No dedicated ternary format yet — TQ1_S/TQ2_S have been discussed but not merged.

**What's novel in our approach:**
Combining post-training ternary quantization with learned LoRA correction stored separately is **not described in the literature**. PT²-LLM is purely mathematical (no learned component). QLoRA uses quantization for fine-tuning memory, not inference compression. Our approach creates a new point in the design space: **compressed ternary backbone + pluggable quality adapter**.

---

## 2. Storage Format Specification

### 2.1 INT2 Ternary Packing

Each ternary weight is encoded in exactly **2 bits**:

```
Encoding (lossless, bijective):
  BITS  | VALUE
  ------+-------
  00    | -1
  01    |  0
  10    | +1
  11    | reserved (unused, or can flag outlier positions)
```

**Packing scheme** — 4 weights per byte, 16 weights per int32:

```
Byte layout (1 byte = 4 weights):
  bits [7:6] = w[0]
  bits [5:4] = w[1]
  bits [3:2] = w[2]
  bits [1:0] = w[3]

Int32 layout (4 bytes = 16 weights) — follows Microsoft BitNet GPU kernel convention:
  int32 = pack_16(w[0], w[1], ..., w[15])
  
  Weight order within int32 (interleaved for efficient SIMD):
    [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
```

**On-disk layout per weight matrix [out_f, in_f]:**

```
Header (24 bytes):
  - out_features:   uint32 (4 bytes)
  - in_features:    uint32 (4 bytes)  
  - packed_stride:  uint32 (4 bytes) = ceil(in_f / 16)
  - has_bias:       uint8  (1 byte)
  - reserved:       uint8[11] (11 bytes padding)

Packed weights:
  - uint32[out_f][packed_stride]
  - Total bytes = out_f * ceil(in_f / 16) * 4

Row alphas (scale factors):
  - float16[out_f]
  - Total bytes = out_f * 2

Bias (if present):
  - float16[out_f]
  - Total bytes = out_f * 2
```

### 2.2 LoRA Adapter Storage

Standard HuggingFace `safetensors` format, separate file:

```
lora_weights.safetensors:
  For each target layer:
    "model.layers.{N}.self_attn.q_proj.lora_A": [rank, in_features]  fp16
    "model.layers.{N}.self_attn.q_proj.lora_B": [out_features, rank]  fp16
    ...
```

### 2.3 Size Calculation for Phi-2 (2.78B params)

**Corrected calculation** (see Section 10.2 for the error analysis of v1):

**Quantizable layers** (all Linear in transformer blocks — 96 layers × 4 attention + 96 × 2 FFN = ~576 matrices):
| Component | Size | Notes |
|-----------|------|-------|
| INT2 packed weights | 0.67 GB | ~2.68B quantizable weights × 2 bits |
| Row alphas + biases (fp16) | 2.4 MB | ~600K rows × 2 bytes × 2 |
| Outlier positions (int32) | 107 MB | 1% of 2.68B = 26.8M positions × 4 bytes |
| Outlier values (fp16) | 54 MB | 26.8M values × 2 bytes |

**Unquantized layers** (must stay FP16 — lm_head, embed, norms):
| Component | Size |
|-----------|------|
| embed_tokens [51200, 2560] | 262 MB |
| lm_head [51200, 2560] | 262 MB |
| LayerNorm params | 1.2 MB |
| **Total unquantized** | **525 MB** |

**LoRA adapters** (rank=64, attention q/k/v/o only, 96 layers):
| Component | Size |
|-----------|------|
| LoRA A matrices (fp16) | 62 MB |
| LoRA B matrices (fp16) | 62 MB |
| **Total LoRA** | **125 MB** |

**Grand total (ternary-boost, with 1% outliers):**

| Component | Size |
|-----------|------|
| INT2 packed weights | 0.67 GB |
| Row params + outliers | 164 MB |
| Unquantized layers | 525 MB |
| LoRA adapters | 125 MB |
| **Total** | **~1.48 GB** |

Compression vs FP16: 5.6 GB → 1.48 GB = **3.8x**.

For PT²-LLM + LoRA (no outliers):
| Component | Size |
|-----------|------|
| INT2 packed weights | 0.67 GB |
| Row params (alpha + mu for asymmetric) | 2.4 MB |
| Unquantized layers | 525 MB |
| LoRA adapters | 125 MB |
| **Total** | **~1.32 GB (4.2x)** |

**Why the compression ratio dropped from v1:** V1 forgot that lm_head (262 MB),
embed_tokens (262 MB), and outlier storage (161 MB) are NOT compressible to
INT2. These account for 40%+ of the compressed model size. This is a known
problem — BitNet papers also exclude embedding/LM head from compression
calculations for from-scratch models, but for post-training, these layers
inherit the FP16 original and can't be ternarized without severe loss.

**Honest comparison:**

| Format | Size | PPL ratio | Compression |
|--------|------|-----------|-------------|
| FP16 original | 5.60 GB | 1.00x | 1.0x |
| INT4 GPTQ (groupsize=128) | ~1.40 GB | ~1.03x | 4.0x |
| **PT²-LLM + LoRA (proposed)** | ~1.32 GB | ~1.15x (est.) | **4.2x** |
| **ternary-boost (proposed)** | ~1.48 GB | ~1.24x | 3.8x |
| PT²-LLM only | ~1.20 GB | ~1.22x (est.) | 4.7x |
| PT-BitNet only | ~1.36 GB | ~1.32x | 4.1x |

The gap between ternary methods and INT4 is narrower than we'd like. The
academic contribution hinges on showing that ternary occupies a useful
tradeoff point — or that ternary+LoRA's PPL ratio gets closer to INT4
than pure ternary does.

---

## 3. Inference Architecture

### 3.1 TernaryInferenceLinear

New module replacing `nn.Linear` after export:

```python
class TernaryInferenceLinear(nn.Module):
    """Stores ternary weights as INT2 + alpha, LoRA as separate fp16 adapters.
    
    Forward: y = alpha * (pos_mask - neg_mask) @ x + lora_scale * (B @ (A @ x))
    
    The ternary matmul uses SELECT + ADD (no fp16 multiplications).
    The LoRA path uses standard small matmuls.
    """
    
    def __init__(self, int2_packed, alpha, bias, lora_A, lora_B, lora_scale):
        self.register_buffer("int2_packed", int2_packed)    # uint32[N][stride]
        self.register_buffer("alpha", alpha)                 # fp16[out_f]
        self.register_buffer("lora_A", lora_A)               # fp16[rank, in_f]
        self.register_buffer("lora_B", lora_B)               # fp16[out_f, rank]
        self.register_buffer("bias", bias)                   # fp16[out_f] or None
    
    def forward(self, x):
        # 1. Dequantize INT2 → ternary mask (on the fly, fused with matmul)
        #    pos_mask[i,j] = 1 if packed_weight[i,j] == +1 else 0
        #    neg_mask[i,j] = 1 if packed_weight[i,j] == -1 else 0
        
        # 2. Ternary matmul (addition/subtraction ONLY, no multiplications):
        #    result = alpha * (sum over pos_mask columns - sum over neg_mask columns)
        #    = alpha * (pos_mask @ x - neg_mask @ x)
        out = self._ternary_matmul(x)  # pure additions!
        
        # 3. LoRA correction (small matmul):
        #    delta = (lora_scale/rank) * B @ (A @ x)
        out = out + self._lora_correction(x)
        
        # 4. Bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
```

### 3.2 Python-Only Implementation (No CUDA)

The first version uses PyTorch operations with unpacked floats (no custom kernel).

```python
def _ternary_matmul(self, x):
    """Pure PyTorch: unpack → mask → matmul with additions only.
    
    Memory: temporarily creates fp16 masks (this is the overhead
    a CUDA kernel would eliminate).
    """
    # Unpack INT2 to float masks
    weights = unpack_int2_to_float(self.int2_packed, out_f, in_f)
    # weights[i,j] ∈ {-1.0, 0.0, +1.0}
    
    pos_mask = (weights > 0.5).to(x.dtype)    # where weight == +1
    neg_mask = (weights < -0.5).to(x.dtype)   # where weight == -1
    
    # Ternário: só adições, zero multiplicações float
    result = self.alpha.unsqueeze(1) * (
        torch.mm(pos_mask, x.T) - torch.mm(neg_mask, x.T)
    ).T
    
    return result
```

**Optimization note**: This unpacks to fp16 for the matmul, which negates some speed benefit. A CUDA kernel would fuse unpack + masked addition without materializing the full fp16 weight matrix. But the **compression** (0.8 GB on disk) is real regardless of unpack strategy.

### 3.3 CUDA Kernel (Phase 2 — Aspirational)

Based on T-MAC lookup-table approach:

```
For each group of 4 ternary weights, precompute 16 partial dot products.
Store in LUT. At inference: use packed INT2 as index → lookup → accumulate.

With a group size of 4 and 3^4 = 81 possible ternary patterns (not 2^4 = 16
because ternary has 3 states, not 2), the naive LUT is 81 entries.

T-MAC optimization: use sign-symmetry to reduce LUT size to 2^(n-1) = 8 entries
for n=4. Since ternary {-1, +1, 0} can be decomposed into sign (1 bit) + 
magnitude (1 bit), the table reduction applies here.

Result: 2-3x speedup over unpacked PyTorch, 4-6x over FP16 matmul
for the ternary portion. The LoRA portion remains standard matmul.
```

---

## 4. Implementation Plan

### Phase 1: INT2 Packing/Unpacking (Day 1)

**File**: `pt_bitnet/src/pt_bitnet/int2_packing.py`

```
Functions:
  pack_int2(weights: Tensor[out_f, in_f]) -> Tensor[out_f, packed_stride]
    - Input: float tensor with values in {-1, 0, 1} (ternary)
    - Output: uint32 tensor with 16 weights per element
    - Encoding: 00=-1, 01=0, 10=+1, 11=unused
    
  unpack_int2(packed: Tensor) -> Tensor[out_f, in_f]
    - Reconstructs float tensor from packed format
    - Lossless roundtrip: unpack(pack(w)) == w for all valid ternary inputs
    
  verify_roundtrip() -> bool
    - Test on random ternary matrices
    - Assert zero reconstruction error
```

### Phase 2: TernaryExportPipeline (Day 2)

**File**: `pt_bitnet/src/pt_bitnet/export.py`

```
Functions:
  export_ternary_lora(model: nn.Module, tokenizer, output_dir: str) -> None
    - For each layer with LoRALinear:
        1. Extract ternary mask from base weight: sign(weight) → {-1, 0, +1}
        2. Extract row alpha: mean(|weight[weight != 0]|)
        3. Pack ternary mask to INT2
        4. Extract LoRA A, B matrices (fp16)
        5. Save to output_dir
    - For non-LoRA layers (lm_head, embed): save in standard safetensors
    - Save config.json with "ternary_lora" architecture type
    
  load_ternary_lora(output_dir: str) -> nn.Module
    - Reconstruct model with TernaryInferenceLinear layers
    - Load INT2 weights + LoRA adapters
    - Ready for inference
```

### Phase 3: TernaryInferenceLinear (Day 2-3)

**File**: `pt_bitnet/src/pt_bitnet/ternary_linear.py`

```
Class TernaryInferenceLinear(nn.Module):
  - __init__: registers buffers for INT2, alpha, bias, lora_A, lora_B
  - forward: unpack + ternary matmul + lora + bias
  - _ternary_matmul: mask-based matmul (zero fp multiplications)
  - _lora_correction: standard B @ (A @ x)
  
Class TernaryLoRAModel(nn.Module):
  - Wraps any HF causal LM, replacing target Linear layers
  - Preserves HF generation API (generate, forward, etc.)
```

### Phase 4: Modify LoRA Training Pipeline (Day 3)

**File**: `pt_bitnet/src/pt_bitnet/lora.py` (modify existing)

```
Changes:
  - After KD training: instead of merge_lora_to_weights(), call 
    keep_lora_separate() — preserves LoRALinear wrappers without merge
  - Save model in new format: INT2 weights + LoRA adapters as separate files
  - Model loader: detect "ternary_lora" format and load appropriately
  
  New function:
    keep_lora_separate(model) -> model
      - Returns model as-is (LoRALinear wrappers intact)
      - This is trivial — just don't call merge
```

### Phase 5: Evaluation & Comparison (Day 4)

**Script**: `scripts/eval_ternary_lora.py`

```
Metrics:
  1. Disk size: INT2+LoRA vs FP16 vs merged vs pure ternary
  2. PPL (diverse texts): same as colab_test.py
  3. Generation quality: same 10 prompts
  4. Memory at inference (VRAM/RAM): peak during forward pass
  5. Speed (tok/s): PyTorch unpack-and-matmul vs baseline FP16
  6. Ablation: rank 16, 32, 64, 128 → PPL vs size tradeoff curve
```

### Phase 6: Colab Memory Optimization (Day 4-5)

```
Concerns:
  1. Export: model already on GPU → extracting and packing on GPU is fast
  2. Loading: INT2 weights stay packed (small), only unpack in forward()
  3. Forward: unpacking per layer on-the-fly saves VRAM (don't materialize full model)
  4. LoRA matmuls: small, already fit in VRAM
  
  Peak VRAM estimate for Phi-2:
    - INT2 packed (GPU): 0.67 GB
    - LoRA fp16 (GPU): 0.13 GB  
    - KV cache + activations: ~2 GB
    - Total: ~3 GB (vs 5.6 GB for FP16 model)
    - Fits T4 easily
```

---

## 5. Expected Results

### 5.1 Compression

See corrected calculation in Section 2.3. Key numbers:

| Model | FP16 | **ternary-boost (proposed)** | PT²-LLM+LoRA (fusion) |
|-------|------|---------------------------|----------------------|
| Phi-2 (2.7B) | 5.6 GB | **~1.48 GB (3.8x)** | ~1.32 GB (4.2x) |
| Mistral-7B | 14.5 GB | **~3.5 GB (4.1x)** | ~3.0 GB (4.8x) |

The 3.8x is lower than the v1 estimate (7x) because of unquantized embed/lm_head
(525 MB fixed cost) and sparse outlier storage (161 MB). This is the honest number.

### 5.2 Quality

| Metric | Expected | Evidence |
|--------|----------|----------|
| PPL ratio | **1.24x** (same as Exp 9) | No quality change — same weights, just different storage |
| Gen quality | **88/100** (same) | Idem |
| Factual accuracy | 10/10 correct | Idem |

### 5.3 Speed

| Implementation | Expected tok/s | vs FP16 |
|---------------|----------------|---------|
| PyTorch unpack-and-matmul | ~15 tok/s | 0.75x slower |
| PyTorch + torch.compile | ~22 tok/s | 1.1x |
| Custom CUDA kernel (future) | ~60 tok/s | 3x |

The unpack-and-matmul approach is slightly slower than FP16 because we materialize the weight matrix in fp16 before the matmul. `torch.compile` should fuse the unpack + matmul, eliminating this overhead. A custom CUDA kernel would be 3x+ faster.

---

## 6. Academic Positioning

### 6.1 What's Novel

1. **First demonstration** of post-training ternary + learned LoRA correction as a **compressed inference format** (not just a training technique)
2. **Size-quality tradeoff curve**: Varying LoRA rank maps a Pareto frontier between pure ternary (~11x compression, PPL 1.32x) and FP16 (1x compression, PPL 1.00x)
3. **Works on small models** (2.7B) where pure post-training ternary is hardest

### 6.2 Potential Paper Structure

**Title candidate**: "Ternário + LoRA: Compressed Ternary Inference with Learned Quality Adapters for Small Language Models"

**Contributions**:
1. INT2 packing format for ternary weights (engineering, not novel but necessary)
2. Demonstration that LoRA + ternary separately stored achieves 7x compression with 1.24x PPL ratio on Phi-2
3. Pareto analysis of rank vs quality vs size
4. Comparison with PT²-LLM (pure ternary, better compression but lower quality on small models)
5. Open-source implementation for T4-accessible research

**Venues**: Efficient NLP workshop (co-located with ACL/EMNLP), or TinyML research symposium. Not NeurIPS/ICML main track — but could be a solid workshop paper.

### 6.3 Comparison with Closest Work

| Aspect | PT²-LLM | Our Work |
|--------|---------|----------|
| Quantization | Ternary only | Ternary + LoRA |
| Compression | 11x (pure ternary) | 7x (ternary + adapters) |
| Quality recovery | Closed-form (ITF, AGA) | Learned (KD) |
| PPL ratio (2.7B) | ~1.20-1.25x (est.) | ~1.24x (measured) |
| Models tested | 7-70B | 2.7B (T4-accessible) |
| Novelty | Asymmetric ternary fitting | Plug-in quality adapters for compressed inference |

These are **complementary**, not competing. LoRA adapters could be applied on top of PT²-LLM-quantized weights to further improve quality. Our contribution is the adapter approach + demonstration on small models.

---

## 7. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| INT2 packing bugs (wrong values) | Medium | High | Roundtrip test: pack → unpack → assert == original |
| PyTorch inference slower than FP16 | High | Medium | `torch.compile` fuse; accept 0.75x as v0 cost |
| LoRA adapters too large (rank too high) | Low | Medium | Rank ablation: find knee point on PPL-vs-size curve |
| Memory spike during unpack | Medium | Medium | Unpack per-layer, don't hold full fp16 model in memory |
| Colab T4 RAM (not VRAM) for export | Low | High | Sharded export, same as current save pipeline |

---

## 8. Timeline

| Day | Deliverable | Hours |
|-----|------------|-------|
| 1 | INT2 packing/unpacking + roundtrip tests | 4-6 |
| 2 | Export pipeline + TernaryInferenceLinear | 6-8 |
| 3 | Modify LoRA pipeline (keep_separate) + model loading | 4-6 |
| 4 | Colab test + evaluation metrics (PPL, size, speed) | 4-6 |
| 5 | Quality ablations (rank sweep) + documentation | 4-6 |
| **Total** | | **22-32 hours** |

---

## 9. Why This Matters (Even Without A100)

1. **Democratizes compressed LLM research**: All experiments run on T4 (free Colab)
2. **7x compression is real-world useful**: phi-2 compressed to <1 GB enables edge deployment (mobile, browser via ONNX)
3. **LoRA as "quality dial"**: Users choose rank based on storage budget — more space = better quality
4. **Framework applies to any quantizer**: Swap PT-BitNet for PT²-LLM or future methods, LoRA adapters still work
5. **First paper on this specific combination**: The literature has pure quantization papers and LoRA fine-tuning papers, but none that treat LoRA as a permanent quality adapter in a compressed inference format

---

## References

- Ma et al. (2024) "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" — arXiv:2402.17764
- Yan et al. (2026) "PT²-LLM: Post-Training Ternary LLM" — ICLR 2026
- Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" — NeurIPS 2023
- Liu et al. (2024) "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" — arXiv:2402.02750
- Microsoft (2024) "T-MAC: Lookup-Table based Inference for Low-Bit LLMs" — arXiv:2410.16144
- Frantar et al. (2023) "GPTQ: Accurate Post-Training Quantization for GPT" — ICLR 2023
- Dettmers et al. (2023) "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression" — arXiv:2306.03078
- Xiao et al. (2023) "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs" — ICML 2023
- Huang et al. (2025) "Tequila: Trapping-free Ternary Quantization for LLMs" — arXiv:2509.23809

---

## 10. Critical Review — Gaps Identified in Version 1

This section is a self-critique. The plan above has 7 gaps that need addressing
before this work is academically defensible.

### 10.1 Gap: No Ablation Framework

The plan assumes ternary-boost (PT-BitNet + LoRA) as the only variant. But to
claim novelty over PT²-LLM, we MUST compare:

| Variant | Quantizer | LoRA | Description |
|---------|-----------|------|-------------|
| **A. FP16 baseline** | None | No | Original model |
| **B. PT-BitNet only** | Symmetric ternary + 1% outliers | No | Our current (Exp 6) |
| **C. PT²-LLM only** | Asymmetric ITF + AGA, no outliers | No | Replication of Yan et al. |
| **D. PT-BitNet + LoRA** | Symmetric ternary + 1% outliers | Yes (rank=64) | **ternary-boost** — our proposal |
| **E. PT²-LLM + LoRA** | Asymmetric ITF + AGA | Yes (rank=64) | **Fusion** — LoRA on top of PT²-LLM |
| **F. INT4 baseline** | GPTQ 4-bit | No | Standard compression baseline |

Without C, we can't claim D improves over PT²-LLM. Without E, we can't show
that LoRA is complementary to (not competing with) PT²-LLM. Without F, we can't
claim ternary compression beats simple INT4.

### 10.2 Gap: Size Calculations Are Wrong

The 0.80 GB estimate is optimistic. Let me correct it for Phi-2:

**Unquantized layers (must stay FP16):**

| Layer | Shape | Size (fp16) |
|-------|-------|-------------|
| embed_tokens | [51200, 2560] | 262 MB |
| lm_head | [51200, 2560] | 262 MB |
| LayerNorm params (96 layers) | ~600K | 1.2 MB |
| **Total unquantized** | | **525 MB** |

**Outlier storage (1% of 2.68B quantizable weights):**

| Component | Count | Size |
|-----------|-------|------|
| Outlier positions (int32 indices) | 26.8M | 107 MB |
| Outlier values (fp16) | 26.8M | 54 MB |
| **Total outliers** | | **161 MB** |

**Corrected total for INT2 + LoRA:**

| Component | Size |
|-----------|------|
| INT2 packed (quantizable layers only) | 0.67 GB |
| Row alphas + biases | 2.4 MB |
| LoRA A + B (rank=64, attn only) | 125 MB |
| Unquantized layers (embed, lm_head, norms) | **525 MB** |
| Outlier positions + values | **161 MB** |
| **Corrected total** | **~1.48 GB** |

Compression vs FP16: 5.6 GB → 1.48 GB = **3.8x** (not 7x as originally stated).

This is still significant but less dramatic. The unquantized embed/lm_head are
the biggest waste — they represent 35% of the compressed size.

### 10.3 Gap: No INT4 Baseline

We must compare against standard INT4 quantization (GPTQ). Why? Because if
INT4 gives PPL 1.05x at 1.4 GB, our ternary+LoRA at PPL 1.24x at 1.5 GB is
strictly worse. We need to show ternary occupies a different (and useful)
point in the quality-size Pareto frontier.

### 10.4 Gap: No Benchmark Tasks Beyond PPL

Perplexity alone is insufficient for a paper in 2026. We need:

| Benchmark | What it measures | Time on T4 |
|-----------|-----------------|------------|
| WikiText-2 PPL | Language modeling | 5 min |
| MMLU (5-shot) | Knowledge/reasoning | 30 min |
| HellaSwag (10-shot) | Commonsense reasoning | 15 min |
| ARC-Challenge (25-shot) | Hard reasoning | 10 min |
| Generation diversity | Repetition, self-BLEU | 5 min |

Total eval time per variant: ~65 min. For 6 variants: ~6.5 hours. Feasible
overnight on Colab T4 (if it stays connected).

### 10.5 Gap: No Statistical Significance

One Colab run = stochastic noise. For reliable conclusions:
- **PPL**: Run 3x per variant, report mean ± std (PPL is deterministic given
  weights, but LoRA training has randomness from data sampling and initialization)
- **Benchmarks**: Single run (deterministic given final weights)
- **Generation quality**: 3 seeds per prompt, average scores

### 10.6 Gap: PT²-LLM Not Working in Our Codebase

We have ITF code (`build_optimal_grid`, `ternary_quantize_vectorized` with
`asymmetric=True`) but it's disabled when `outlier_fraction > 0` because
outlier removal destabilizes the closed-form solution.

To get PT²-LLM working for the ablation, we need to run it **without outliers**:
- `outlier_fraction = 0`
- `asymmetric = True`
- ITF + AGA active

This is a pure replication of PT²-LLM's method. We already fixed the AGA
activation collection bug (commit `2d49afa`). The ITF fallback warning
"ITF produced extreme values" should NOT trigger when outliers are disabled.

### 10.7 Gap: OBC Still Active by Default

Experiments 6 and 9 (our best results) ran WITHOUT OBC. But the current
code enables OBC whenever calibration texts exist. We need to:

1. Add `use_obc: bool = False` to `PTBitNetConfig` — separate toggle
2. Set `use_obc = False` for small models in `model_loader.py`
3. Keep OBC code for future testing on 7B+ models

This is a 10-minute fix but critical for reproducibility.

---

## 11. Ablation Framework

### 11.1 Variants to Test

```
Experiment matrix (Phi-2, Colab T4):

A) FP16 baseline
   - Original microsoft/phi-2
   - No quantization, no LoRA
   - Size: 5.6 GB, PPL: 3.16 (diverse texts)

B) PT-BitNet only (symmetric ternary)
   - Config: symmetric=True, outliers=0.01, use_obc=False
   - No LoRA
   - Size: ~1.2 GB (INT2 + outliers + unquantized)
   - Expected PPL: ~1.32x (from Exp 6)

C) PT²-LLM only (asymmetric ITF + AGA)
   - Config: asymmetric=True, outliers=0, use_obc=False
   - No LoRA
   - Size: ~1.0 GB (INT2 + unquantized, no outliers!)
   - Expected PPL: ~1.20-1.25x (literature estimate for 2.7B)
   - RISK: May not work well on Phi-2 (small model). ITF may produce
     degenerate solutions for some rows.

D) PT-BitNet + LoRA [ternary-boost]
   - Config: same as B, then LoRA rank=64, KD steps=1000
   - LoRA stored separately (NOT merged)
   - Size: ~1.5 GB (INT2 + outliers + LoRA + unquantized)
   - Expected PPL: ~1.24x (from Exp 9)
   - This is OUR proposal.

E) PT²-LLM + LoRA [fusion]
   - Config: same as C, then LoRA rank=64, KD steps=1000
   - LoRA stored separately
   - Size: ~1.3 GB (INT2 + LoRA + unquantized)
   - Expected PPL: ~1.10-1.18x (estimated: PT²-LLM base + LoRA gain)
   - This tests complementarity: does LoRA add value on top of a better
     quantizer, or is the gain redundant?

F) INT4 baseline (GPTQ)
   - Standard GPTQ 4-bit quantization, group_size=128
   - No LoRA
   - Size: ~1.4 GB (4-bit weights + scales + unquantized)
   - Expected PPL: ~1.02-1.05x
   - Reference point: if INT4 is both smaller AND better quality than
     ternary+LoRA, our approach has no reason to exist.

G) [OPTIONAL] PT-BitNet + LoRA, NO outliers
   - Config: symmetric=True, outliers=0, use_obc=False
   - LoRA rank=64
   - Size: ~1.1 GB (pure INT2 + LoRA + unquantized, no outlier overhead)
   - Tests whether outliers are necessary when LoRA is present.
```

### 11.2 Metrics per Variant

| Metric | Tool | Time |
|--------|------|------|
| **PPL** (diverse texts, 8 excerpts) | `colab_test.py` | 3 min |
| **PPL** (WikiText-2, standard) | `lm-eval` or manual | 5 min |
| **Generation quality** (10 prompts) | `colab_test.py` | 5 min |
| **Disk size** | `du -sh` on saved directory | 10s |
| **Inference VRAM** | `torch.cuda.max_memory_allocated()` | per run |
| **Inference speed** (tok/s) | Average over 10 generations | per run |

### 11.3 Expected Pareto Analysis

This is the key figure for a paper:

```
PPL ratio (lower=better)
 1.00 ───┬─── FP16 (5.6 GB)
         │
 1.02 ───┤─── INT4 GPTQ (1.4 GB) ← tough baseline to beat
         │
 1.15 ───┤─── PT²-LLM + LoRA [fusion] (1.3 GB) ← best case
         │
 1.20 ───┤─── PT²-LLM only (1.0 GB)
         │
 1.24 ───┤─── PT-BitNet + LoRA [ternary-boost] (1.5 GB)
         │
 1.32 ───┤─── PT-BitNet only (1.2 GB)
         │
         └────────────────────────────────────────
              0.5    1.0    1.5           5.6
                   Model Size (GB) →

KEY QUESTION: Do D and E occupy a useful point in the frontier
that C and F don't already cover?
```

### 11.4 PT²-LLM Implementation — What We Already Have

Contrary to what I implied in Section 10.6, we already have functional PT²-LLM
code. What we need to make it work:

**Step 1: Disable outlier removal for PT²-LLM variant**
```python
# In model_loader.py or colab_test.py:
pt2_config = PTBitNetConfig(
    asymmetric=True,
    outlier_fraction=0,       # ← critical: ITF is incompatible with outliers
    compensation_steps=0,     # ← skip lm_head Hessian comp
    block_size=128,
)
```

**Step 2: Verify ITF doesn't fallback**
The warning "ITF produced extreme values — falling back to symmetric"
appeared when outliers were enabled (Exp 3). With `outlier_fraction=0`,
no weights are zeroed → row statistics stay intact → ITF should work
for all 96 layers.

**Step 3: AGA is already implemented**
`activation_aware_grid_alignment()` in quantize.py computes the
covariance-aligned grid. It's called when `asymmetric=True` and
calibration data exists. The bug that collected 0 layers was already fixed.

**Step 4: Validation**
Run PT²-LLM on Phi-2, verify:
- 0 ITF fallback warnings
- PPL within expected range (~1.20-1.25x baseline)
- Coherent generation output

### 11.5 Fusion Approach: PT²-LLM + LoRA

The fusion is straightforward:

```
1. PT²-LLM quantizes model → asymmetric ternary weights (no outliers)
2. Wrap quantized layers with LoRALinear (same as current LoRA pipeline)
3. Train LoRA adapters via KD from FP16 teacher
4. Keep LoRA separate → export INT2 (asymmetric) + LoRA
```

**Storage for asymmetric ternary:**
Symmetric uses 1 scale per row: {-α, 0, +α}. Asymmetric uses 2: {-α+μ, μ, +α+μ}.

The INT2 encoding is identical (same {-1, 0, +1} pattern). Only difference:
we store 2 fp16 params per row instead of 1:

```
Row parameters (asymmetric):
  - alpha[out_f]  (float16, 2 bytes per row)
  - mu[out_f]     (float16, 2 bytes per row)
  vs symmetric:
  - alpha[out_f]  (float16, 2 bytes per row)
```

Extra overhead vs symmetric: ~1.2 MB for Phi-2 — negligible.

### 11.6 INT4 Baseline Implementation

Use HuggingFace's `transformers` + `bitsandbytes` or a minimal GPTQ
implementation:

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config,
)
```

Or use `auto-gptq` for a more controlled comparison with group_size=128.

**Important**: bitsandbytes NF4 quantizes ALL linear layers including lm_head.
For fair comparison, we should use the same scope as our method (exclude
lm_head and embed from INT4 too).

---

## 12. Revised Timeline

| Day | Deliverable | Hours |
|-----|------------|-------|
| 1 | Fix OBC config (add `use_obc` flag, default False for small models) + verify PT²-LLM ITF works without outliers + INT2 packing/unpacking + roundtrip tests | 6-8 |
| 2 | Export pipeline + TernaryInferenceLinear (support both sym and asym) + keep_lora_separate in LoRA pipeline | 6-8 |
| 3 | Run ablation variants B, C (quantizers only, no LoRA yet) + INT4 baseline | 4-6 |
| 4 | Train LoRA on B → variant D, train LoRA on C → variant E + eval all | 6-8 |
| 5 | Full evaluation (benchmarks + PPL + size + speed for all 6 variants) + Pareto analysis | 4-6 |
| 6 | Documentation + paper outline + ablation analysis | 4-6 |
| **Total** | | **30-42 hours** |

---

## 13. Decision Gate: After Ablation Results

After running the 6 variants, we'll know whether to proceed. Decision criteria:

| Outcome | Decision |
|---------|----------|
| PT²-LLM + LoRA (E) beats INT4 (F) on PPL at smaller size | **Strong paper** — proceed with full experiments |
| PT-BitNet + LoRA (D) beats PT²-LLM only (C) on PPL | **OK paper** — LoRA adds value, ternary-boost is valid |
| INT4 (F) beats ALL ternary variants on PPL + size | **Weak paper** — ternary has no advantage over INT4 for small models. Consider pivot to 7B or from-scratch training. |
| ITF crashes or produces gibberish on Phi-2 | PT²-LLM may not work on small models. Paper focuses on ternary-boost (D) vs INT4 (F) vs pure ternary (B). |
| Nothing beats INT4 + all ternary variants >1.20x PPL | **Reconsider the project direction.** Post-training ternary on 2.7B may have an inherent quality floor. |
