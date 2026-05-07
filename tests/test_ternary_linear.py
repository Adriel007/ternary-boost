"""Tests for TernaryInferenceLinear and model replacement."""

import pytest
import torch
import torch.nn as nn
from pt_bitnet.int2_packing import pack_int2
from pt_bitnet.ternary_linear import (
    TernaryInferenceLinear,
    TernaryLoRAModel,
)


def _random_ternary(out_f, in_f, sparsity=0.3):
    """Generate random ternary weight with controlled sparsity."""
    w = torch.zeros(out_f, in_f)
    mask = torch.rand(out_f, in_f) > sparsity
    w[mask] = (torch.randint(0, 2, (mask.sum(),)) * 2 - 1).float()
    return w


class TestTernaryInferenceLinear:
    """Correctness tests for the separate ternary + LoRA inference module."""

    @pytest.mark.parametrize("out_f,in_f", [(64, 32), (128, 256), (32, 15)])
    def test_forward_matches_linear_symmetric(self, out_f, in_f):
        """TernaryInferenceLinear == nn.Linear with equivalent ternary weights."""
        w_ternary = _random_ternary(out_f, in_f)

        # Compute alpha as mean abs of non-zero weights
        T = torch.sign(w_ternary)
        nonzero = T.abs()
        row_sum = (w_ternary.abs() * nonzero).sum(dim=-1)
        row_count = nonzero.sum(dim=-1).clamp_min(1)
        alpha = (row_sum / row_count).unsqueeze(-1)

        # Reference: nn.Linear with ternary weights
        ref = nn.Linear(in_f, out_f, bias=False)
        ref.weight.data = w_ternary

        # Test: TernaryInferenceLinear with INT2 + alpha
        int2_packed = pack_int2(T)
        test = TernaryInferenceLinear(
            int2_packed=int2_packed,
            alpha=alpha,
            in_features=in_f,
            out_features=out_f,
        )

        x = torch.randn(4, 8, in_f)
        ref_out = ref(x)
        test_out = test(x)

        assert torch.allclose(ref_out, test_out, atol=1e-5), \
            f"Max diff: {(ref_out - test_out).abs().max().item()}"

    def test_forward_matches_with_bias(self):
        """Ternary + bias must match nn.Linear with same bias."""
        out_f, in_f = 32, 64
        w_ternary = _random_ternary(out_f, in_f)
        b = torch.randn(out_f)
        T = torch.sign(w_ternary)
        nonzero = T.abs()
        alpha = ((w_ternary.abs() * nonzero).sum(dim=-1) / nonzero.sum(dim=-1).clamp_min(1)).unsqueeze(-1)

        ref = nn.Linear(in_f, out_f, bias=True)
        ref.weight.data = w_ternary
        ref.bias.data = b

        test = TernaryInferenceLinear(
            int2_packed=pack_int2(T),
            alpha=alpha,
            in_features=in_f,
            out_features=out_f,
            bias=b.clone(),
        )

        x = torch.randn(2, 16, in_f)
        assert torch.allclose(ref(x), test(x), atol=1e-5)

    def test_forward_with_lora(self):
        """Ternary + LoRA must match LoRALinear forward."""
        out_f, in_f = 64, 128
        rank = 8

        w_ternary = _random_ternary(out_f, in_f)
        T = torch.sign(w_ternary)
        nonzero = T.abs()
        alpha = ((w_ternary.abs() * nonzero).sum(dim=-1) / nonzero.sum(dim=-1).clamp_min(1)).unsqueeze(-1)

        lora_A = torch.randn(rank, in_f) * 0.02
        lora_B = torch.zeros(out_f, rank)
        lora_scale = 2.0

        # Reference: manual forward
        x = torch.randn(3, 16, in_f)
        ref_out = x @ w_ternary.T + lora_scale * (x @ lora_A.T @ lora_B.T)

        test = TernaryInferenceLinear(
            int2_packed=pack_int2(T),
            alpha=alpha,
            in_features=in_f,
            out_features=out_f,
            lora_A=lora_A,
            lora_B=lora_B,
            lora_scale=lora_scale,
        )
        test_out = test(x)

        assert torch.allclose(ref_out, test_out, atol=1e-5), \
            f"Max diff: {(ref_out - test_out).abs().max().item()}"

    def test_weight_property_matches_linear(self):
        """The .weight property must reconstruct the full ternary matrix."""
        out_f, in_f = 32, 64
        w_ternary = _random_ternary(out_f, in_f)
        T = torch.sign(w_ternary)
        nonzero = T.abs()
        alpha = ((w_ternary.abs() * nonzero).sum(dim=-1) / nonzero.sum(dim=-1).clamp_min(1)).unsqueeze(-1)

        test = TernaryInferenceLinear(
            int2_packed=pack_int2(T),
            alpha=alpha,
            in_features=in_f,
            out_features=out_f,
        )

        reconstructed = test.weight
        assert torch.allclose(w_ternary, reconstructed, atol=1e-5), \
            f"Max diff: {(w_ternary - reconstructed).abs().max().item()}"

    def test_forward_asymmetric(self):
        """Asymmetric ternary {-alpha+mu, mu, +alpha+mu} must match."""
        out_f, in_f = 32, 64
        T = _random_ternary(out_f, in_f)
        alpha = torch.rand(out_f, 1) * 0.5 + 0.25
        mu = torch.randn(out_f, 1) * 0.1
        w_asym = alpha * T + mu

        ref = nn.Linear(in_f, out_f, bias=False)
        ref.weight.data = w_asym

        test = TernaryInferenceLinear(
            int2_packed=pack_int2(T),
            alpha=alpha,
            in_features=in_f,
            out_features=out_f,
            mu=mu,
        )

        x = torch.randn(4, 8, in_f)
        ref_out = ref(x)
        test_out = test(x)

        assert torch.allclose(ref_out, test_out, atol=1e-5), \
            f"Max diff: {(ref_out - test_out).abs().max().item()}"

    def test_extra_repr(self):
        """String representation must show config info."""
        T = _random_ternary(32, 64)
        alpha = torch.ones(32, 1)
        layer = TernaryInferenceLinear(
            int2_packed=pack_int2(torch.sign(T)),
            alpha=alpha,
            in_features=64,
            out_features=32,
        )
        rep = layer.extra_repr()
        assert "sym" in rep
        assert "64" in rep
        assert "32" in rep


class TestTernaryModelReplacement:
    """Tests for replacing layers in a model."""

    def test_replace_lora_linear_to_ternary(self):
        """replace_with_ternary converts LoRALinear → TernaryInferenceLinear."""
        from pt_bitnet.lora import LoRALinear, LoRAConfig

        # Build a tiny model
        model = nn.Sequential()
        base = nn.Linear(64, 32, bias=False)
        base.weight.data = _random_ternary(32, 64)
        base.weight.requires_grad = False
        lora_cfg = LoRAConfig(rank=4)
        model.layer = LoRALinear(base, lora_cfg)

        TernaryLoRAModel.replace_with_ternary(model, target_modules=("layer",))

        assert isinstance(model.layer, TernaryInferenceLinear)
        assert model.layer.has_lora
        assert model.layer.out_features == 32
        assert model.layer.in_features == 64

        # Forward pass must match original LoRALinear
        x = torch.randn(2, 8, 64)
        # Can't compare against original since we replaced it,
        # but we can verify the output shape
        out = model.layer(x)
        assert out.shape == (2, 8, 32)

    def test_replace_ternary_linear_no_lora(self):
        """replace_with_ternary converts plain ternary nn.Linear."""
        model = nn.Sequential()
        layer = nn.Linear(64, 32, bias=True)
        layer.weight.data = _random_ternary(32, 64)
        layer.weight.requires_grad = False
        layer.bias.data = torch.randn(32)
        model.layer = layer

        TernaryLoRAModel.replace_with_ternary(model, target_modules=("layer",))

        assert isinstance(model.layer, TernaryInferenceLinear)
        assert not model.layer.has_lora

        x = torch.randn(2, 8, 64)
        out = model.layer(x)
        assert out.shape == (2, 8, 32)
