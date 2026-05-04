"""LSQ-based quantization functions and QuantizeLinear layer.

Based on the ParetoQ implementation from torchao/prototype/paretoq.
Supports 1-bit (ternary), 2-bit, 3-bit, and 4-bit quantization with
learned step-size parameters (LSQ: Learned Step-size Quantization).
"""

import math
import torch
import torch.nn as nn


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """LSQ for 1-bit (ternary) through 4-bit quantization.

    Reference: Esser et al., "Learned Step-size Quantization", ICLR 2020.
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """Elastic quantization with stretched levels for 0-2 bit regimes.

    Uses a sigmoid-like stretching to handle the transition between
    discrete levels more smoothly than hard clamping.
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1
        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            q_stretched = (
                torch.round(
                    torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_stretched)
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_stretched)
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class QuantizeLinear(nn.Linear):
    """Linear layer with LSQ-based weight quantization.

    During training, weights are quantized on-the-fly using the
    straight-through estimator. A learnable `weight_clip_val` parameter
    controls the quantization scale per output channel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 1,
        weight_layerwise: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(
                torch.Tensor(self.weight.shape[0], 1)
            )

    def forward(self, input_):
        real_weights = self.weight
        if self.w_bits >= 16:
            weight = self.weight
        elif self.w_bits in (0, 2):
            weight = StretchedElasticQuant.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        elif self.w_bits <= 4:
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                self.weight_clip_val,
                self.w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            raise NotImplementedError(f"w_bits={self.w_bits} not supported")
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
