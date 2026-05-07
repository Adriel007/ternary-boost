"""Hybrid runtime: ternary kernel + sparse FP16 outliers + LoRA at inference.

Composes three substrates into a single forward pass:

    y = TernaryKernel(x, W_int2, alpha_per_row)   # ~99% of FLOPs, fast CPU
      + SparseMatMul(x, W_outlier_csr_fp16)        # ~1% sparsity, FP16
      + (alpha_lora / rank) * (B @ (A @ x))        # LoRA correction

This is the union of SpQR (outlier+lowbit) and QLoRA (base+LoRA-at-runtime),
applied to ternary weights. Not previously published as a single artifact.

Three kernel paths, in order of preference:
  Path A — llama.cpp TQ2_0 via llama-cpp-python
  Path B — T-MAC W2A16 GPTQ-format via ctypes
  Path C — custom AVX2 kernel via cffi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.logging import get_logger

logger = get_logger("pt_bitnet.hybrid_runtime")


# ═══════════════════════════════════════════════════════════════════
# Ternary kernel protocol
# ═══════════════════════════════════════════════════════════════════

TernaryKernel = Callable[[torch.Tensor], torch.Tensor]
"""Signature of a ternary kernel: x[batch, seq, in_f] -> y[batch, seq, out_f]."""


# ═══════════════════════════════════════════════════════════════════
# Default PyTorch ternary kernel (Path C fallback, correct but slow)
# ═══════════════════════════════════════════════════════════════════

class PyTorchTernaryKernel(nn.Module):
    """Reference ternary kernel: unpack INT2 + matmul in PyTorch.

    Correctness over speed — use as fallback or for validation.
    """

    def __init__(
        self,
        weight_int2: torch.Tensor,      # [out_f, in_f // 4] uint8 packed
        alpha: torch.Tensor,             # [out_f] per-row scale, FP16
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer("weight_int2", weight_int2)
        self.register_buffer("alpha", alpha)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unpack INT2 weights and compute y = (W_ternary @ x^T)^T."""
        W = _unpack_int2_to_ternary(self.weight_int2, self.alpha, x.device, x.dtype)
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out


def _unpack_int2_to_ternary(
    packed: torch.Tensor, alpha: torch.Tensor, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack INT2-packed weights to dense ternary {-α, 0, +α}.

    packed: [out_f, in_f // 4] uint8 — 4 weights per byte, codebook {00, 01, 10, 11}
            with 00 = -α, 01 = 0, 10 = +α, 11 = unused (→0).
    alpha:  [out_f] — per-row scale factor.
    """
    out_f, packed_in_f = packed.shape
    in_f = packed_in_f * 4

    # Decode pairs of bits
    code = packed.to(torch.uint8)  # [out_f, in_f//4]
    # Expand to [out_f, in_f] via bit manipulation
    indices = torch.arange(in_f, device=device).view(1, -1)  # [1, in_f]
    byte_idx = indices // 4  # [1, in_f]
    bit_shift = (indices % 4) * 2  # [1, in_f]

    code_expanded = code[:, byte_idx.long()]  # [out_f, in_f]
    two_bit = (code_expanded >> bit_shift) & 0b11  # [out_f, in_f]

    # Codebook: 0→-1, 1→0, 2→+1, 3→0 (unused)
    ternary = torch.zeros(out_f, in_f, device=device, dtype=dtype)
    ternary[two_bit == 2] = 1.0
    ternary[two_bit == 0] = -1.0

    # Apply per-row alpha
    return ternary * alpha.to(device=device, dtype=dtype).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════
# Hybrid composition layer
# ═══════════════════════════════════════════════════════════════════

class HybridTernaryLinear(nn.Module):
    """Hybrid runtime linear layer: ternary kernel + sparse outliers + LoRA.

    Three substrates, three computation strategies:

    1. Ternary kernel (fast path, ~99% of weight mass) — dispatches to the
       configured backend (llama.cpp, T-MAC, or PyTorch reference).
    2. Sparse outliers (FP16 correction, ~1% nnz) — stored as dense small
       matrix; applied via matmul.
    3. LoRA adapter (low-rank correction) — rank ≪ dim, plain PyTorch matmul.

    All three are summed at the application layer. No custom CUDA/AVX
    required for correctness; the kernel swap determines speed.
    """

    def __init__(
        self,
        ternary_kernel: TernaryKernel,
        outliers_weight: Optional[torch.Tensor] = None,  # [n_outliers, in_f]
        outliers_indices: Optional[torch.Tensor] = None,  # [n_outliers] row indices
        lora_A: Optional[torch.Tensor] = None,             # [rank, in_f]
        lora_B: Optional[torch.Tensor] = None,             # [out_f, rank]
        lora_scaling: float = 1.0,
        bias: Optional[torch.Tensor] = None,
        out_features: int = 0,
        in_features: int = 0,
    ):
        super().__init__()
        self._ternary_kernel = ternary_kernel
        self.in_features = in_features
        self.out_features = out_features

        # Outliers: stored as sparse-indexed dense for efficient matmul
        if outliers_weight is not None and outliers_indices is not None:
            self.register_buffer("outliers_weight", outliers_weight.half())
            self.register_buffer("outliers_indices", outliers_indices.long())
        else:
            self.outliers_weight = None
            self.outliers_indices = None

        # LoRA adapter
        if lora_A is not None and lora_B is not None:
            self.lora_A = nn.Parameter(lora_A, requires_grad=False)
            self.lora_B = nn.Parameter(lora_B, requires_grad=False)
        else:
            self.lora_A = None
            self.lora_B = None
        self.lora_scaling = lora_scaling

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Ternary base (bulk computation, ~99% of weight mass)
        y = self._ternary_kernel(x)

        # 2. Sparse outlier correction
        if self.outliers_weight is not None and self.outliers_indices is not None:
            # Gather outlier rows: matmul with submatrix
            outlier_out = x @ self.outliers_weight.T  # [batch, seq, n_outliers]
            # Scatter-add to output positions
            idx = self.outliers_indices  # [n_outliers]
            y.index_add_(dim=-1, index=idx, source=outlier_out.to(y.dtype))

        # 3. LoRA correction
        if self.lora_A is not None and self.lora_B is not None:
            A = self.lora_A.to(dtype=x.dtype)
            B = self.lora_B.to(dtype=x.dtype)
            lora_out = self.lora_scaling * (x @ A.T @ B.T)
            y = y + lora_out

        if self.bias is not None:
            y = y + self.bias

        return y

    def extra_repr(self) -> str:
        parts = [
            f"in={self.in_features}, out={self.out_features}",
            f"outliers={'yes' if self.outliers_weight is not None else 'no'}",
            f"lora={'r=' + str(self.lora_A.shape[0]) if self.lora_A is not None else 'no'}",
        ]
        return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Build hybrid layers from exported ternary + outliers + LoRA
# ═══════════════════════════════════════════════════════════════════

def build_hybrid_layer(
    weight_int2: torch.Tensor,       # [out_f, in_f//4] uint8 packed
    alpha: torch.Tensor,              # [out_f] per-row scale
    bias: Optional[torch.Tensor],
    outliers_weight: Optional[torch.Tensor],  # [n_out, in_f]
    outliers_indices: Optional[torch.Tensor], # [n_out]
    lora_A: Optional[torch.Tensor],           # [rank, in_f]
    lora_B: Optional[torch.Tensor],           # [out_f, rank]
    lora_scaling: float,
    kernel_path: str = "pytorch",     # "llamacpp", "tmac", "pytorch"
) -> HybridTernaryLinear:
    """Build a single HybridTernaryLinear from serialized components.

    Args:
        weight_int2: INT2-packed ternary weights [out_f, in_f//4] uint8.
        alpha: Per-row scale [out_f].
        bias: Optional bias [out_f].
        outliers_weight: FP16 outlier weights [n_outliers, in_f].
        outliers_indices: Row indices for outliers [n_outliers].
        lora_A, lora_B: LoRA adapter matrices.
        lora_scaling: α/rank scaling factor.
        kernel_path: Backend for the ternary kernel.

    Returns:
        Configured HybridTernaryLinear ready for inference.
    """
    out_f = weight_int2.shape[0]
    in_f = weight_int2.shape[1] * 4

    # ── Select ternary kernel ─────────────────────────────────────
    if kernel_path == "llamacpp":
        raise NotImplementedError(
            "Path A (llama.cpp TQ2_0) requires llama-cpp-python. "
            "Install with: pip install llama-cpp-python"
        )
    elif kernel_path == "tmac":
        raise NotImplementedError(
            "Path B (T-MAC W2A16) requires T-MAC compiled with TVM. "
            "See https://github.com/microsoft/T-MAC"
        )
    else:
        ternary_kernel = PyTorchTernaryKernel(weight_int2, alpha, bias)

    return HybridTernaryLinear(
        ternary_kernel=ternary_kernel,
        outliers_weight=outliers_weight,
        outliers_indices=outliers_indices,
        lora_A=lora_A,
        lora_B=lora_B,
        lora_scaling=lora_scaling,
        bias=None if kernel_path != "pytorch" else bias,
        out_features=out_f,
        in_features=in_f,
    )


# ═══════════════════════════════════════════════════════════════════
# Load exported model into hybrid runtime
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HybridModelConfig:
    """Configuration for loading a hybrid runtime model from export."""
    kernel_path: str = "pytorch"
    device: str = "cpu"
    dtype: str = "float16"
    target_modules: tuple = field(default=(
        "q_proj", "k_proj", "v_proj", "o_proj", "dense",
        "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
    ))


def load_hybrid_model(
    export_dir: str | Path,
    device: str = "cpu",
    torch_dtype=torch.float16,
    config: Optional[HybridModelConfig] = None,
) -> nn.Module:
    """Load an exported ternary model into the hybrid runtime.

    Reads the same export format as ``pt_bitnet.export.load_ternary_lora``
    (HF safetensors + ternary_params.pt + lora_weights.safetensors) and
    replaces quantized linears with ``HybridTernaryLinear``.

    Outliers are currently *baked into* the INT2-packed weights.  When
    the Week 5 spike adds separate outlier sidecars to the export format,
    this loader will pick them up automatically.

    Args:
        export_dir: directory containing config.json, ternary_params.pt,
                    lora_weights.safetensors, and sharded safetensors.
        device: target device (``"cpu"`` or ``"cuda"``).
        torch_dtype: dtype for non-ternary parameters.
        config: hybrid runtime configuration (kernel path, etc.).

    Returns:
        Model where each quantized linear is a HybridTernaryLinear.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if config is None:
        config = HybridModelConfig()

    export_dir = Path(export_dir)

    # 1. Load config + model via HF (low_cpu_mem_usage for safety)
    hf_config = AutoConfig.from_pretrained(str(export_dir), trust_remote_code=True)
    if getattr(hf_config, "pad_token_id", None) is None:
        hf_config.pad_token_id = getattr(hf_config, "eos_token_id", 0) or 0

    model = AutoModelForCausalLM.from_pretrained(
        str(export_dir), config=hf_config, torch_dtype=torch_dtype,
        device_map=device, trust_remote_code=True, low_cpu_mem_usage=True,
    )

    # 2. Load ternary params (same format as load_ternary_lora)
    tp_path = export_dir / "ternary_params.pt"
    if not tp_path.exists():
        logger.warning("No ternary_params.pt — returning FP16 model unchanged")
        return model

    ternary_params = torch.load(tp_path, map_location=device, weights_only=True)
    logger.info(f"Loaded ternary params for {len(ternary_params)} layers")

    # 3. Load LoRA weights
    lora_weights: dict[str, torch.Tensor] = {}
    lora_path = export_dir / "lora_weights.safetensors"
    if lora_path.exists():
        from safetensors.torch import load_file
        lora_weights = load_file(str(lora_path))
        logger.info(f"Loaded {len(lora_weights)} LoRA tensors")

    # 4. Replace linears with HybridTernaryLinear
    replaced = 0
    for module_name, module in list(model.named_modules()):
        if module_name not in ternary_params:
            continue

        tp = ternary_params[module_name]
        out_f = tp["out_features"]
        in_f = tp["in_features"]

        # Build PyTorchTernaryKernel (reference path — swap for llama.cpp / T-MAC later)
        tern_kernel = PyTorchTernaryKernel(
            weight_int2=tp["int2_packed"].to(device),
            alpha=tp["alpha"].to(device),
            bias=tp["bias"].to(device) if tp.get("bias") is not None else None,
        )

        # LoRA adapter
        lora_A = lora_B = None
        lora_scaling = 1.0
        if tp.get("has_lora"):
            lora_A_key = f"{module_name}.lora_A"
            lora_B_key = f"{module_name}.lora_B"
            if lora_A_key in lora_weights and lora_B_key in lora_weights:
                lora_A = lora_weights[lora_A_key].to(device)
                lora_B = lora_weights[lora_B_key].to(device)
                lora_scaling = tp.get("lora_scale", 1.0)

        # Outliers are currently baked into int2_packed / alpha (see export.py).
        # When Week 5 adds separate outlier sidecars, load them here.
        hybrid = HybridTernaryLinear(
            ternary_kernel=tern_kernel,
            outliers_weight=None,   # TODO: load from outlier sidecar (Week 5)
            outliers_indices=None,  # TODO
            lora_A=lora_A,
            lora_B=lora_B,
            lora_scaling=lora_scaling,
            bias=tp["bias"].to(device) if tp.get("bias") is not None else None,
            out_features=out_f,
            in_features=in_f,
        )

        # Navigate to parent and replace
        parts = module_name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]
        setattr(parent, child_name, hybrid)
        replaced += 1

    logger.info(f"Replaced {replaced} layers with HybridTernaryLinear")
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# Benchmarking helper
# ═══════════════════════════════════════════════════════════════════

def benchmark_hybrid_layer(
    layer: HybridTernaryLinear,
    batch_size: int = 1,
    seq_length: int = 128,
    warmup: int = 5,
    iters: int = 50,
    device: str = "cpu",
) -> dict:
    """Measure tokens/sec for a single hybrid layer.

    Returns a dict with tok/s, mean_ms, p50_ms, p99_ms.
    """
    import time

    dtype = next(layer.parameters()).dtype if list(layer.parameters()) else torch.float32
    x = torch.randn(batch_size, seq_length, layer.in_features, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        _ = layer(x)

    # Measure
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = layer(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    times = np.array(times_ms)
    tok_per_s = batch_size * seq_length / (times.mean() / 1000)

    return {
        "tokens_per_sec": round(tok_per_s, 1),
        "mean_ms": round(times.mean(), 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "n_iters": iters,
    }
