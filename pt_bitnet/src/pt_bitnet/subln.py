"""SubLN insertion for BitNet-Distillation-compatible ternarization.

Adds an RMSNorm immediately before the output projection of each MHSA and FFN
block. Required architectural prep before BitDistill quality recovery.

Reference: Wu et al., "BitNet Distillation," arXiv 2510.13998, Oct 2025.
"""

import re

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from shared.logging import get_logger

logger = get_logger("pt_bitnet.subln")


class SubLN(nn.Module):
    """RMSNorm variant inserted before MHSA/FFN output projections.

    Initialized with weight=1 so the model is functionally identical at
    insertion time. The warm-up phase learns the optimal scaling.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ── FQN regex patterns for output-projection discovery ────────────
# Matches output projections across Phi-2, LLaMA, Mistral, Qwen, Pythia, GPT-2.
_MHSA_OUTPUT_RE = re.compile(
    r".*(?:self_attn|attention|attn|mha)\.(?:o_proj|dense|out_proj)$"
)
_FFN_OUTPUT_RE = re.compile(
    r".*(?:mlp|ffn)\.(?:down_proj|fc2|dense_4h_to_h|dense_h_to_4h)$"
)


def _is_output_projection(module_name: str) -> bool:
    """Check if a linear layer is an MHSA or FFN output projection."""
    return bool(_MHSA_OUTPUT_RE.match(module_name) or _FFN_OUTPUT_RE.match(module_name))


def insert_subln(model: PreTrainedModel) -> PreTrainedModel:
    """Insert SubLN before every MHSA and FFN output-projection linear.

    Walks model.named_modules(), identifies output-projection linears by FQN
    regex, and wraps each with ``nn.Sequential(SubLN(dim), linear)``.

    SubLN.weight is initialized to 1 so the model output is unchanged at
    insertion time. The subsequent C4 warm-up phase learns the scale.
    """
    inserted = 0
    skipped = 0

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _is_output_projection(module_name):
            continue

        in_features = module.in_features

        # Navigate to parent and replace
        parts = module_name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]

        # Skip if already wrapped (idempotent)
        if isinstance(getattr(parent, child_name, None), nn.Sequential):
            skipped += 1
            continue

        wrapped = nn.Sequential(
            SubLN(in_features),
            module,
        )
        setattr(parent, child_name, wrapped)
        inserted += 1

    logger.info(f"  SubLN: inserted {inserted} norms, skipped {skipped} (already wrapped)")
    return model


def remove_subln(model: PreTrainedModel) -> PreTrainedModel:
    """Reverse insert_subln: unwrap Sequential(SubLN, linear) → linear.

    SubLN weights are discarded. The model reverts to its original architecture.
    """
    removed = 0

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Sequential):
            continue
        if len(module) != 2:
            continue
        if not isinstance(module[0], SubLN):
            continue
        if not isinstance(module[1], nn.Linear):
            continue

        parts = module_name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            child_name = parts[0]
        else:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]

        setattr(parent, child_name, module[1])
        removed += 1

    if removed > 0:
        logger.info(f"  SubLN: removed {removed} norms")
    return model


def count_subln(model: PreTrainedModel) -> int:
    """Return the number of SubLN modules currently in the model."""
    return sum(1 for m in model.modules() if isinstance(m, SubLN))
