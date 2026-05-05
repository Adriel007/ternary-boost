"""Tequila: Trapping-free Ternary Quantization.

Key innovation: Deadzone Trapping Recovery.

In standard ternary quantization, weights near the decision boundary
receive noisy, uninformative gradients that prevent them from escaping
the "deadzone." Tequila solves this by:

  1. Splitting weights into two matrices:
     - **A**: Large-magnitude weights quantized to {-scale, 0, +scale}
     - **B**: "Trapped" deadzone residuals, reactivated as dynamic biases

  2. In UltraQuant v3/v4, B is modulated by a learnable `Lambada`
     parameter that controls how much each deadzone residual contributes,
     giving trapped weights a path to receive meaningful gradients.

The forward pass becomes:
    output = linear(input, A) + linear(ones, B * Lambada)

This preserves ternary efficiency while allowing deadzone weights to
contribute a continuous signal and receive direct gradients.
"""

import gc
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from shared.logging import get_logger

logger = get_logger("tequila")


@dataclass
class TequilaConfig:
    """Configuration for Tequila deadzone trapping recovery."""

    quant_method: str = "ultraquantv3"  # v1, v2, v3, v4
    granularity: str = "per_channel"  # per_tensor, per_channel, per_group
    group_size: int = 128
    enable_zero_point: bool = False
    range_of_lambada: float = 0.01
    eps: float = 1e-5
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lambada_granularity: str = "per_channel"  # "per_channel" (6 MB) or "per_element" (2.5 GB)
    target_modules: tuple = field(
        default=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )
    skip_modules: tuple = field(default=("lm_head", "embed_tokens"))


def absmean(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and delta from mean absolute value."""
    scale = x.abs().mean(dim=-1, keepdim=True)
    delta = scale / 2
    return scale, delta


class StaticQuaternaryQuant(torch.autograd.Function):
    """UltraQuant v1: Static deadzone with constant epsilon fill."""

    @staticmethod
    def forward(ctx, input, granularity, group_size, eps):
        original_shape = input.shape
        if granularity == "per_tensor":
            x = input.reshape(1, -1)
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)
        else:
            raise NotImplementedError(f"Unknown granularity: {granularity}")

        alpha = x.abs().mean(dim=-1, keepdim=True)
        delta = alpha / 2

        A = torch.zeros_like(x)
        mask_pos = x >= delta
        mask_neg = x <= -delta
        A[mask_pos] = 1
        A[mask_neg] = -1
        A = A * alpha

        B = torch.zeros_like(x)
        mask_B_pos = (x >= 0) & (A == 0)
        mask_B_neg = (x < 0) & (A == 0)
        B[mask_B_pos] = eps
        B[mask_B_neg] = -eps

        A = A.reshape(original_shape)
        B = B.reshape(original_shape)
        mask_B_pos = mask_B_pos.reshape(original_shape)
        mask_B_neg = mask_B_neg.reshape(original_shape)
        ctx.save_for_backward(mask_B_pos, mask_B_neg)
        return A, B

    @staticmethod
    def backward(ctx, grad_A, grad_B):
        mask_B_pos, mask_B_neg = ctx.saved_tensors
        grad_input = grad_A
        grad_input[mask_B_pos] += grad_B[mask_B_pos]
        grad_input[mask_B_neg] += grad_B[mask_B_neg]
        return grad_input, None, None, None


class UltraQuantV2(torch.autograd.Function):
    """UltraQuant v2: Scaled deadzone with proportional epsilon."""

    @staticmethod
    def forward(ctx, input, granularity, group_size, eps):
        original_shape = input.shape
        if granularity == "per_tensor":
            x = input.reshape(1, -1)
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)
        else:
            raise NotImplementedError

        scale, delta = absmean(x)

        A = torch.zeros_like(x)
        mask_A_pos = x >= delta
        mask_A_neg = x <= -delta
        A[mask_A_pos] = 1
        A[mask_A_neg] = -1
        A = A * scale

        B = torch.zeros_like(x)
        mask_B = A == 0
        B[mask_B] = eps * x[mask_B]

        A = A.reshape(original_shape)
        B = B.reshape(original_shape)
        if isinstance(eps, torch.Tensor):
            eps_saved = eps.detach().clone()
        else:
            eps_saved = torch.tensor(eps)
        ctx.save_for_backward(mask_B, eps_saved, scale)
        return A, B

    @staticmethod
    def backward(ctx, grad_A, grad_B):
        mask_B, eps, scale = ctx.saved_tensors
        grad_output = grad_A.reshape(mask_B.shape)
        grad_B_reshaped = grad_B.reshape(mask_B.shape)
        grad_output[mask_B] = grad_output[mask_B] + eps * grad_B_reshaped[mask_B]
        grad_output = grad_output.reshape(grad_A.shape)
        return grad_output, None, None, None


class UltraQuantV3(torch.autograd.Function):
    """UltraQuant v3: Full residual in deadzone with Lambada modulation.

    Deadzone weights are stored directly in B (no scaling), allowing
    them to receive direct, meaningful gradients during backpropagation.
    The Lambada parameter (learned per-channel) modulates the contribution.
    """

    @staticmethod
    def forward(ctx, input, granularity, group_size):
        original_shape = input.shape
        if granularity == "per_tensor":
            x = input.reshape(1, -1)
        elif granularity == "per_channel":
            x = input.reshape(original_shape[0], -1)
        elif granularity == "per_group":
            x = input.reshape(-1, group_size)
        else:
            raise NotImplementedError

        scale, delta = absmean(x)

        A = torch.zeros_like(x)
        mask_A_pos = x >= delta
        mask_A_neg = x <= -delta
        A[mask_A_pos] = 1
        A[mask_A_neg] = -1
        A = A * scale

        B = torch.zeros_like(x)
        mask_B = A == 0
        B[mask_B] = x[mask_B]

        A = A.reshape(original_shape)
        B = B.reshape(original_shape)
        ctx.save_for_backward(mask_B, scale)
        return A, B

    @staticmethod
    def backward(ctx, grad_A, grad_B):
        mask_B, scale = ctx.saved_tensors
        grad_output = grad_A.reshape(mask_B.shape)
        grad_B_reshaped = grad_B.reshape(mask_B.shape)
        grad_output[mask_B] = grad_B_reshaped[mask_B]
        grad_scale = torch.ones_like(grad_output) * scale
        grad_scale[mask_B] = 1
        grad_output = grad_output * grad_scale
        grad_output = grad_output.reshape(grad_A.shape)
        return grad_output, None, None


class UltraQuantLinear(nn.Linear):
    """Linear layer with Tequila deadzone trapping (AngelSlim-compatible).

    Splits weight into:
      A: ternary-quantized large weights [-scale, 0, +scale]
      B: deadzone residuals reactivated as dynamic biases

    Forward: out = linear(x, A) + linear(ones, B * Lambada)

    Lambada optimization follows the original AngelSlim pattern:
    per-layer AdamW optimizer, update_lambada called during forward pass
    with zero_grad + backward + step (self-contained per-layer optimization).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_method: str = "ultraquantv3",
        granularity: str = "per_channel",
        group_size: int = 128,
        enable_zero_point: bool = False,
        range_of_lambada: float = 0.01,
        eps: float = 1e-5,
        lambada_granularity: str = "per_channel",
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.quant_method = quant_method
        self.granularity = granularity
        self.group_size = group_size
        self.enable_zero_point = enable_zero_point
        self.lambada_granularity = lambada_granularity

        if self.quant_method in ("ultraquant", "ultraquantv2"):
            self.eps = eps
        elif self.quant_method in ("ultraquantv3", "ultraquantv4"):
            if lambada_granularity == "per_channel":
                lambada_shape = (out_features, 1)
            else:
                lambada_shape = (out_features, in_features)
            self.Lambada = nn.Parameter(
                torch.randn(lambada_shape) * range_of_lambada,
                requires_grad=True,
            )
            self.optimizer = torch.optim.AdamW([self.Lambada], lr=1e-4)

    def forward(self, input_):
        real_weights = self.weight

        # During training, break autograd graph between layers.
        # Frozen weights (requires_grad=False) already skip grad computation,
        # but hidden states from embeddings have requires_grad=True.
        # Detaching prevents building a giant useless computation graph.
        if self.training:
            input_ = input_.detach()

        if self.quant_method == "ultraquant":
            eps = torch.tensor(self.eps, device=input_.device, dtype=input_.dtype)
            A, B = StaticQuaternaryQuant.apply(
                real_weights, self.granularity, self.group_size, eps
            )
            A = A.to(input_.dtype)
            B = B.to(input_.dtype)
            ones = torch.sign(input_.detach()).to(input_.dtype)
            out = nn.functional.linear(input_, A) + nn.functional.linear(ones, B)

        elif self.quant_method == "ultraquantv2":
            eps = torch.tensor(self.eps, device=input_.device, dtype=input_.dtype)
            A, B = UltraQuantV2.apply(
                real_weights, self.granularity, self.group_size, eps
            )
            A = A.to(input_.dtype)
            B = B.to(input_.dtype)
            ones = torch.ones(1, 1, device=input_.device, dtype=input_.dtype).expand_as(input_)
            out = nn.functional.linear(input_, A) + nn.functional.linear(ones, B)

        elif self.quant_method in ("ultraquantv3", "ultraquantv4"):
            A, B = UltraQuantV3.apply(
                real_weights, self.granularity, self.group_size
            )
            A = A.to(input_.dtype)
            B = B.to(input_.dtype)

            # Per-layer Lambada optimization — matches original AngelSlim pattern
            if self.training:
                self.update_lambada(input_.detach(), B.detach())

            B_modulated = (B * self.Lambada.detach()).to(input_.dtype)
            ones = torch.ones(1, 1, device=input_.device, dtype=input_.dtype).expand_as(input_)
            out = nn.functional.linear(input_, A) + nn.functional.linear(ones, B_modulated)

        else:
            raise NotImplementedError(f"Unknown quant_method: {self.quant_method}")

        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def update_lambada(self, input_batch: torch.Tensor, B: torch.Tensor) -> None:
        """Per-layer Lambada optimization (AngelSlim-compatible).

        Minimizes ||linear(input, B) - sum(Lambada * B)||^2 per output channel.
        Runs full optimization cycle: zero_grad → backward → step.
        """
        input_batch = input_batch.reshape(-1, input_batch.shape[-1])
        T = input_batch.shape[0]
        Y = nn.functional.linear(input_batch, B)
        s = torch.sum(self.Lambada * B, dim=-1)
        loss = torch.sum((Y - s) ** 2) / max(T, 1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def _replace_with_ultraquant(
    model: nn.Module,
    config: TequilaConfig,
) -> nn.Module:
    """Replace target nn.Linear layers with UltraQuantLinear.

    Creates new modules on the SAME device as the originals to avoid
    temporary CPU↔GPU copies that exhaust system RAM on Colab T4.
    """
    replacements = 0
    for module_name, module in model.named_modules():
        if any(skip in module_name for skip in config.skip_modules):
            continue
        if not any(target in module_name for target in config.target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name = ".".join(module_name.split(".")[:-1])
        child_name = module_name.split(".")[-1]
        parent = model if not parent_name else model.get_submodule(parent_name)

        device = module.weight.device
        ultraquant = UltraQuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            quant_method=config.quant_method,
            granularity=config.granularity,
            group_size=config.group_size,
            enable_zero_point=config.enable_zero_point,
            range_of_lambada=config.range_of_lambada,
            eps=config.eps,
            lambada_granularity=config.lambada_granularity,
        ).to(device)
        ultraquant.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            ultraquant.bias.data.copy_(module.bias.data)

        setattr(parent, child_name, ultraquant)
        replacements += 1

    logger.info(f"Replaced {replacements} nn.Linear layers with UltraQuantLinear")
    return model


def apply_tequila(
    model: PreTrainedModel,
    calibration_dataloader,
    config: Optional[TequilaConfig] = None,
) -> PreTrainedModel:
    """Apply Tequila deadzone trapping recovery.

    This reactivates weights trapped in the ternary quantization deadzone
    as dynamic biases, recovering accuracy with near-zero overhead.

    Args:
        model: QAT-fine-tuned model (from ParetoQ stage).
        calibration_dataloader: Small calibration dataset (100-1000 samples).
        config: Tequila configuration.

    Returns:
        Model with Tequila deadzone trapping applied.
    """
    if config is None:
        config = TequilaConfig()

    logger.info(f"Applying Tequila deadzone trapping (method={config.quant_method})")

    # Free memory from previous pipeline stages before creating new modules
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _replace_with_ultraquant(model, config)

    if config.quant_method in ("ultraquantv3", "ultraquantv4"):
        lambada_count = sum(
            1 for n, _ in model.named_parameters() if "Lambada" in n
        )
        logger.info(f"Found {lambada_count} Lambada parameters "
                     f"(each with per-layer AdamW optimizer)")

        has_cuda = torch.cuda.is_available()
        # Only move to CUDA if not already there (avoids double alloc)
        if has_cuda and next(model.parameters()).device.type != "cuda":
            model.cuda()

        for epoch in range(config.num_epochs):
            num_batches = 0

            model.train()
            for batch in calibration_dataloader:
                if not isinstance(batch, dict):
                    continue
                batch = {k: v.cuda() if has_cuda and isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                if input_ids is None:
                    continue

                # Forward pass: model weights are frozen (requires_grad=False),
                # only Lambada params have requires_grad=True.
                # Each UltraQuantLinear.forward() calls update_lambada() which
                # does its own zero_grad → backward → step internally.
                # Residual connections are detached at each layer boundary.
                model(input_ids=input_ids, attention_mask=attention_mask)

                num_batches += 1
                if num_batches >= 10:
                    break

            logger.info(f"Tequila epoch {epoch + 1}/{config.num_epochs} "
                         f"({num_batches} batches)")

    model.eval()
    logger.info("Tequila deadzone trapping applied successfully")
    return model
