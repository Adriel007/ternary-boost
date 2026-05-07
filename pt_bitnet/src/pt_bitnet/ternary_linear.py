"""Ternary + LoRA inference modules.

TernaryInferenceLinear keeps ternary weights (INT2 packed) and LoRA adapters
as SEPARATE components. Never merges them into dense weights — the compression
is real: INT2 on disk, LoRA as small fp16 adapters.

Forward: y = alpha*(T @ x) [+ mu*sum(x)] + lora_scale*(B @ (A @ x)) + bias

Where T ∈ {-1,0,+1} is unpacked from INT2 on the fly.
LoRA correction is a standard small matmul.

v1 uses PyTorch unpack-and-matmul (materializes fp16 T, no custom kernel).
The compression benefit is in storage, not inference speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from pt_bitnet.int2_packing import unpack_int2


class TernaryInferenceLinear(nn.Module):
    """Linear layer with INT2-packed ternary weights + optional LoRA adapter.

    Storage (on disk, per layer):
      - INT2 packed weights: out_f * ceil(in_f/16) * 4 bytes
      - alpha (scale):       out_f * 2 bytes (fp16)
      - mu (shift):          out_f * 2 bytes (fp16), only if asymmetric
      - lora_A:              rank * in_f * 2 bytes (fp16)
      - lora_B:              out_f * rank * 2 bytes (fp16)
      - bias:                out_f * 2 bytes (fp16), optional

    At inference, INT2 is unpacked to fp16 T ∈ {-1,0,+1} per forward pass.
    This materializes the full weight matrix temporarily — a custom CUDA
    kernel (Phase 2) would fuse unpack + masked accumulation to avoid this.
    """

    def __init__(
        self,
        int2_packed: torch.Tensor,        # [out_f, ceil(in_f/16)] int32
        alpha: torch.Tensor,               # [out_f, 1] fp16/bf16
        in_features: int,
        out_features: int,
        mu: Optional[torch.Tensor] = None, # [out_f, 1] or None (symmetric)
        bias: Optional[torch.Tensor] = None, # [out_f] or None
        lora_A: Optional[torch.Tensor] = None,  # [rank, in_f]
        lora_B: Optional[torch.Tensor] = None,  # [out_f, rank]
        lora_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Ternary backbone (INT2 packed, stays on device as-is)
        self.register_buffer("int2_packed", int2_packed)
        self.register_buffer("alpha", alpha.view(out_features, 1))
        if mu is not None:
            self.register_buffer("mu", mu.view(out_features, 1))
        else:
            self.register_buffer("mu", torch.zeros(out_features, 1))

        self._asymmetric = mu is not None

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias.view(out_features))
        else:
            self.bias = None

        # LoRA adapter (optional quality recovery)
        if lora_A is not None and lora_B is not None:
            self.register_buffer("lora_A", lora_A)
            self.register_buffer("lora_B", lora_B)
            self.has_lora = True
        else:
            self.register_buffer("lora_A", torch.empty(0))
            self.register_buffer("lora_B", torch.empty(0))
            self.has_lora = False

        self.lora_scale = lora_scale

    @property
    def weight(self):
        """Proxy .weight for HF model introspection / state_dict compatibility.

        Reconstructs the full ternary weight matrix from INT2 + alpha + mu.
        This is for inspection only — forward() uses the packed path.
        """
        T = unpack_int2(self.int2_packed, self.in_features)
        return self.alpha * T + self.mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ternary matmul + LoRA correction + bias.

        Args:
            x: [*, in_features] input tensor.

        Returns:
            [*, out_features] output tensor.
        """
        # 1. Ternary backbone
        T = unpack_int2(self.int2_packed, self.in_features).to(dtype=x.dtype)
        # W = alpha * T + mu  (not materialized — fused into matmul)
        # y_tern = alpha * (T @ x) + mu * sum_j(x_j)
        ternary_out = F.linear(x, T)                # [*, out_f]
        ternary_out = ternary_out * self.alpha.squeeze(-1)  # [*, out_f] * [out_f]
        if self._asymmetric:
            ternary_out = ternary_out + self.mu.squeeze(-1) * x.sum(dim=-1, keepdim=True)

        # 2. LoRA correction
        if self.has_lora:
            lora_A = self.lora_A.to(dtype=x.dtype)
            lora_B = self.lora_B.to(dtype=x.dtype)
            lora_out = F.linear(x, lora_A)          # [*, rank]
            lora_out = F.linear(lora_out, lora_B)    # [*, out_f]
            ternary_out = ternary_out + self.lora_scale * lora_out

        # 3. Bias
        if self.bias is not None:
            ternary_out = ternary_out + self.bias

        return ternary_out

    def extra_repr(self) -> str:
        asym = "asym" if self._asymmetric else "sym"
        lora = f"+LoRA(r={self.lora_A.shape[0]})" if self.has_lora else ""
        return f"in={self.in_features}, out={self.out_features}, {asym}{lora}"


class TernaryLoRAModel(nn.Module):
    """Wraps a HuggingFace causal LM to use TernaryInferenceLinear layers.

    Reconstructs the model from INT2-packed weights + LoRA adapters,
    preserving the HF generation API (generate, forward, etc.).

    Usage:
        model = TernaryLoRAModel.from_pretrained("./exported_model/")
        output = model.generate(input_ids, max_new_tokens=50)
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TernaryLoRAModel":
        """Load a ternary+LoRA model from an exported directory.

        Args:
            path: directory containing config.json, int2_weights.pt,
                  lora_weights.safetensors, and alpha.pt.

        Returns:
            TernaryLoRAModel wrapping the reconstructed HF model.
        """
        from pt_bitnet.export import load_ternary_lora
        model = load_ternary_lora(path, **kwargs)
        return cls(model)

    @staticmethod
    def replace_with_ternary(
        model: nn.Module,
        target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"),
    ) -> nn.Module:
        """Replace nn.Linear layers in-place with TernaryInferenceLinear.

        The model must already have LoRALinear or nn.Linear layers with
        ternary weights + LoRA adapters. This extracts the weights from
        each layer and repackages them as TernaryInferenceLinear.

        Args:
            model: model with LoRALinear or nn.Linear layers.
            target_modules: which layer names to replace.

        Returns:
            The same model with replaced layers (mutated in-place).
        """
        from pt_bitnet.lora import LoRALinear
        from pt_bitnet.int2_packing import pack_int2

        replaced = 0
        for module_name, module in list(model.named_modules()):
            # Determine if this is a target module
            is_target = any(t in module_name for t in target_modules)
            is_skip = ("lm_head" in module_name or "embed_tokens" in module_name)
            if not is_target or is_skip:
                continue

            if isinstance(module, LoRALinear):
                _replace_lora_linear(model, module_name, module)
                replaced += 1
            elif isinstance(module, nn.Linear) and not module.weight.requires_grad:
                _replace_ternary_linear(model, module_name, module)
                replaced += 1

        return model


def _replace_lora_linear(model, module_name, module):
    """Replace a LoRALinear with TernaryInferenceLinear, preserving both
    ternary weights and LoRA adapters separately."""
    from pt_bitnet.int2_packing import pack_int2

    base_w = module.base.weight.data  # ternary float weights
    out_f, in_f = base_w.shape

    # Extract ternary mask and alpha
    w_float = base_w.float()
    T = torch.sign(w_float)  # {-1, 0, +1}
    # alpha = mean absolute value of non-zero weights
    nonzero = T.abs()
    row_sum = (w_float.abs() * nonzero).sum(dim=-1)
    row_count = nonzero.sum(dim=-1).clamp_min(1)
    alpha = (row_sum / row_count).unsqueeze(-1)

    # Detect if asymmetric (row means are non-zero)
    mu = w_float.mean(dim=-1, keepdim=True)
    asymmetric = mu.abs().max() > 1e-6
    if not asymmetric:
        mu = None

    int2_packed = pack_int2(T)

    bias = module.base.bias.data.clone() if module.base.bias is not None else None

    lora_A = module.lora_A.data.clone()
    lora_B = module.lora_B.data.clone()
    lora_scale = module.scaling

    parent_name = ".".join(module_name.split(".")[:-1])
    child_name = module_name.split(".")[-1]
    parent = model if not parent_name else model.get_submodule(parent_name)

    new_layer = TernaryInferenceLinear(
        int2_packed=int2_packed,
        alpha=alpha.to(base_w.dtype),
        in_features=in_f,
        out_features=out_f,
        mu=mu.to(base_w.dtype) if mu is not None else None,
        bias=bias,
        lora_A=lora_A,
        lora_B=lora_B,
        lora_scale=lora_scale,
    )
    setattr(parent, child_name, new_layer)


def _replace_ternary_linear(model, module_name, module):
    """Replace a plain nn.Linear (ternary weights, no LoRA) with
    TernaryInferenceLinear."""
    from pt_bitnet.int2_packing import pack_int2

    w = module.weight.data.float()
    out_f, in_f = w.shape

    T = torch.sign(w)
    nonzero = T.abs()
    row_sum = (w.abs() * nonzero).sum(dim=-1)
    row_count = nonzero.sum(dim=-1).clamp_min(1)
    alpha = (row_sum / row_count).unsqueeze(-1)

    mu = w.mean(dim=-1, keepdim=True)
    asymmetric = mu.abs().max() > 1e-6
    if not asymmetric:
        mu = None

    int2_packed = pack_int2(T)
    bias = module.bias.data.clone() if module.bias is not None else None

    parent_name = ".".join(module_name.split(".")[:-1])
    child_name = module_name.split(".")[-1]
    parent = model if not parent_name else model.get_submodule(parent_name)

    new_layer = TernaryInferenceLinear(
        int2_packed=int2_packed,
        alpha=alpha.to(module.weight.dtype),
        in_features=in_f,
        out_features=out_f,
        mu=mu.to(module.weight.dtype) if mu is not None else None,
        bias=bias,
    )
    setattr(parent, child_name, new_layer)
