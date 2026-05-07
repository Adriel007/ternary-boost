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
    config: Optional[HybridModelConfig] = None,
) -> nn.Module:
    """Load exported ternary model into the hybrid runtime.

    Reads INT2-packed weights, outlier sidecars, and LoRA adapters from
    a sharded safetensors export directory (as produced by
    ``pt_bitnet.export.export_ternary_lora``).

    Args:
        export_dir: Path to the export directory containing safetensors shards.
        config: Hybrid runtime configuration.

    Returns:
        An nn.Module where each quantized linear is a HybridTernaryLinear.
    """
    if config is None:
        config = HybridModelConfig()

    import json
    from safetensors.torch import load_file

    export_dir = Path(export_dir)

    # Discover shards
    shard_files = sorted(export_dir.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = sorted(export_dir.glob("*.safetensors"))

    # Load metadata (layer map, config)
    meta_path = export_dir / "model.safetensors.index.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"weight_map": {}}

    # Load all shards into CPU memory (size ~2.5 GB for Phi-2, fits T4 CPU RAM)
    all_tensors: dict[str, torch.Tensor] = {}
    for sf in shard_files:
        tensors = load_file(str(sf))
        all_tensors.update(tensors)

    logger.info(f"Loaded {len(all_tensors)} tensors from {len(shard_files)} shard(s)")

    # Parse components: W_int2, alpha, outliers, LoRA per layer
    layers: dict[str, dict] = {}
    for key, tensor in all_tensors.items():
        # Expected keys: model.layers.N.self_attn.q_proj.weight_int2
        #               model.layers.N.mlp.fc1.weight_int2
        #               model.layers.N.self_attn.q_proj.alpha
        #               model.layers.N.self_attn.q_proj.outliers_weight
        #               model.layers.N.self_attn.q_proj.outliers_indices
        #               model.layers.N.self_attn.q_proj.lora_A
        parts = key.rsplit(".", 1)
        if len(parts) == 2 and parts[1] in (
            "weight_int2", "alpha", "bias",
            "outliers_weight", "outliers_indices",
            "lora_A", "lora_B", "lora_scaling",
        ):
            layer_key = parts[0]
            attr = parts[1]
            if layer_key not in layers:
                layers[layer_key] = {}
            layers[layer_key][attr] = tensor

    logger.info(f"Parsed {len(layers)} layers from export")

    # Build a flat nn.Module with HybridTernaryLinear layers
    model = nn.Module()
    for layer_key, components in sorted(layers.items()):
        if "weight_int2" not in components or "alpha" not in components:
            continue

        weight_int2 = components["weight_int2"]
        alpha = components["alpha"]
        bias = components.get("bias")
        outliers_w = components.get("outliers_weight")
        outliers_idx = components.get("outliers_indices")
        lora_A = components.get("lora_A")
        lora_B = components.get("lora_B")

        # Parse lora_scaling from alpha/rank if not explicit
        lora_scaling = 1.0
        if "lora_scaling" in components:
            lora_scaling = components["lora_scaling"].item()
        elif lora_A is not None and lora_B is not None:
            rank = lora_A.shape[0]
            lora_scaling = 128.0 / rank  # default alpha=128

        hybrid = build_hybrid_layer(
            weight_int2=weight_int2,
            alpha=alpha,
            bias=bias,
            outliers_weight=outliers_w,
            outliers_indices=outliers_idx,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_scaling=lora_scaling,
            kernel_path=config.kernel_path,
        )

        # Store as attribute with dots replaced by underscores
        attr_name = layer_key.replace(".", "_")
        setattr(model, attr_name, hybrid)

    logger.info(f"Built {len([m for m in model.modules() if isinstance(m, HybridTernaryLinear)])} hybrid layers")
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
