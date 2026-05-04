"""ZeroQAT: Zero-Order Quantization-Aware Training.

Implements memory-efficient gradient estimation using only forward passes,
enabling QAT on hardware with limited VRAM (≥8GB for 2B-parameter models).

Based on MeZO (Malladi et al., 2023) and ZO-QAT extensions for
quantized models. Uses SPSA-style simultaneous perturbation with
per-layer gradient estimation for reduced variance.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from shared.logging import get_logger

logger = get_logger("zeroqat")


@dataclass
class ZeroQATConfig:
    """Configuration for Zero-Order QAT."""

    epsilon: float = 1e-3
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_steps: int = 1000
    grad_clip: float = 1.0
    perturbation_scale: float = 1e-3
    num_perturbations: int = 1
    zo_estimator: str = "spsa"  # "spsa" or "rge" (random gradient estimation)
    target_modules: tuple = field(
        default=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )

    def __post_init__(self):
        if self.zo_estimator not in ("spsa", "rge"):
            raise ValueError(f"Unknown ZO estimator: {self.zo_estimator}")


class ZeroOrderOptimizer:
    """Memory-efficient zero-order optimizer for QAT.

    Estimates gradients via forward-pass-only perturbation,
    avoiding the memory cost of full backpropagation through
    quantized models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ZeroQATConfig,
        named_params: Optional[list[tuple[str, nn.Parameter]]] = None,
    ):
        self.model = model
        self.config = config

        if named_params is None:
            self.named_params = [
                (n, p) for n, p in model.named_parameters()
                if p.requires_grad and any(
                    t in n for t in config.target_modules
                )
            ]
        else:
            self.named_params = named_params

        self.param_shapes = {
            name: param.shape for name, param in self.named_params
        }
        logger.info(
            f"ZeroOrderOptimizer tracking {len(self.named_params)} parameter groups"
        )

    def estimate_gradient(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn,
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient via SPSA two-sided perturbation.

        For each tracked parameter, generates a random perturbation z,
        computes loss difference, and estimates the gradient as:
            g ≈ (L(θ+εz) - L(θ-εz)) / (2ε) * z
        """
        estimated_grads: dict[str, torch.Tensor] = {}

        z_cache: dict[str, torch.Tensor] = {}
        for name, _ in self.named_params:
            z_cache[name] = torch.randn_like(
                self._get_param_by_name(name),
                device=self._get_param_by_name(name).device,
            )

        for name, param in self.named_params:
            z = z_cache[name]
            eps = self.config.perturbation_scale

            param_orig = param.data.clone()

            param.data = param_orig + eps * z
            loss_pos = self._forward_and_loss(batch, loss_fn)

            param.data = param_orig - eps * z
            loss_neg = self._forward_and_loss(batch, loss_fn)

            param.data = param_orig

            grad_est = (loss_pos - loss_neg) / (2 * eps) * z

            if self.config.grad_clip > 0:
                grad_norm = grad_est.norm()
                if grad_norm > self.config.grad_clip:
                    grad_est = grad_est * (self.config.grad_clip / grad_norm)

            estimated_grads[name] = grad_est

        return estimated_grads

    def estimate_gradient_layerwise(
        self,
        batch: dict[str, torch.Tensor],
        loss_fn,
    ) -> dict[str, torch.Tensor]:
        """Layer-wise gradient estimation for reduced variance.

        Groups parameters by layer prefix and perturbs each group
        independently, reducing cross-layer interference in the
        gradient estimate.
        """
        estimated_grads: dict[str, torch.Tensor] = {}

        layer_groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for name, param in self.named_params:
            prefix = ".".join(name.split(".")[:3])
            layer_groups.setdefault(prefix, []).append((name, param))

        for layer_prefix, params in layer_groups.items():
            z_cache = {
                name: torch.randn_like(param.data)
                for name, param in params
            }

            orig_data = {name: param.data.clone() for name, param in params}

            eps = self.config.perturbation_scale

            for name, param in params:
                param.data = orig_data[name] + eps * z_cache[name]
            loss_pos = self._forward_and_loss(batch, loss_fn)

            for name, param in params:
                param.data = orig_data[name] - eps * z_cache[name]
            loss_neg = self._forward_and_loss(batch, loss_fn)

            for name, param in params:
                param.data = orig_data[name]

            for name in params:
                grad_est = (
                    (loss_pos - loss_neg) / (2 * eps) * z_cache[name[0]]
                )
                if self.config.grad_clip > 0:
                    grad_norm = grad_est.norm()
                    if grad_norm > self.config.grad_clip:
                        grad_est = grad_est * (
                            self.config.grad_clip / grad_norm
                        )
                estimated_grads[name[0]] = grad_est

        return estimated_grads

    def step(self, grads: dict[str, torch.Tensor]) -> None:
        """Apply estimated gradients with weight decay."""
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        for name, param in self.named_params:
            if name not in grads:
                continue
            g = grads[name]
            if wd > 0:
                g = g + wd * param.data
            param.data = param.data - lr * g

    def _get_param_by_name(self, name: str) -> nn.Parameter:
        for n, p in self.named_params:
            if n == name:
                return p
        raise KeyError(f"Parameter {name} not found")

    @torch.no_grad()
    def _forward_and_loss(
        self, batch: dict[str, torch.Tensor], loss_fn
    ) -> torch.Tensor:
        """Single forward pass returning scalar loss."""
        self.model.eval()
        model_inputs = {k: v for k, v in batch.items() if k not in ("labels", "label")}
        labels = batch.get("labels", batch.get("label"))

        outputs = self.model(**model_inputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if labels is not None:
            return loss_fn(outputs, labels)
        return loss_fn(outputs, None)

    def get_param_vector(self) -> torch.Tensor:
        """Flatten tracked parameters into a single vector (for RGE)."""
        vecs = []
        for _, param in self.named_params:
            vecs.append(param.data.flatten())
        return torch.cat(vecs)

    def set_param_vector(self, vec: torch.Tensor) -> None:
        """Restore tracked parameters from a flattened vector."""
        offset = 0
        for _, param in self.named_params:
            numel = param.numel()
            param.data.copy_(
                vec[offset : offset + numel].reshape(param.shape)
            )
            offset += numel
