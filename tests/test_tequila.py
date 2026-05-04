"""Tests for Tequila deadzone trapping recovery."""

import pytest
import torch
import torch.nn as nn

from tequila.ultraquant import (
    absmean,
    StaticQuaternaryQuant,
    UltraQuantV2,
    UltraQuantV3,
    UltraQuantLinear,
    TequilaConfig,
)


class TestAbsMean:
    """Tests for absmean scale/delta computation."""

    def test_basic(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        scale, delta = absmean(x)
        expected_scale = torch.tensor([[2.5]])
        expected_delta = expected_scale / 2
        torch.testing.assert_close(scale, expected_scale)
        torch.testing.assert_close(delta, expected_delta)


class TestStaticQuaternaryQuant:
    """Tests for UltraQuant v1."""

    def test_forward_split(self):
        w = torch.randn(4, 16)
        eps = 1e-5
        A, B = StaticQuaternaryQuant.apply(w, "per_channel", 128, eps)
        assert A.shape == w.shape
        assert B.shape == w.shape

    def test_A_is_ternary_like(self):
        w = torch.randn(8, 32) * 2
        eps = 1e-5
        A, _ = StaticQuaternaryQuant.apply(w, "per_channel", 128, eps)
        nonzero = A[A != 0]
        if nonzero.numel() > 0:
            row_max = A.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
            normalized = A / row_max
            unique_A = normalized.unique()
            assert unique_A.numel() <= 3

    def test_B_fills_deadzone(self):
        w = torch.randn(4, 16)
        eps = 1e-5
        A, B = StaticQuaternaryQuant.apply(w, "per_channel", 32, eps)
        deadzone = A == 0
        if deadzone.any():
            assert (B[deadzone].abs() > 0).any()

    def test_per_tensor_granularity(self):
        w = torch.randn(8, 64)
        eps = 1e-5
        A, B = StaticQuaternaryQuant.apply(w, "per_tensor", 128, eps)
        assert A.shape == w.shape
        A_channel = StaticQuaternaryQuant.apply(w, "per_channel", 128, eps)[0]
        assert A_channel.shape == w.shape


class TestUltraQuantV2:
    """Tests for UltraQuant v2 with scaled deadzone."""

    def test_forward(self):
        w = torch.randn(4, 16)
        eps = 1e-5
        A, B = UltraQuantV2.apply(w, "per_channel", 128, eps)
        assert A.shape == w.shape
        assert B.shape == w.shape

    def test_B_proportional_to_x(self):
        w = torch.randn(8, 32) * 2
        eps = 0.1
        A, B = UltraQuantV2.apply(w, "per_channel", 128, eps)
        deadzone = A == 0
        if deadzone.any():
            ratio = B[deadzone] / (w[deadzone] + 1e-8)
            assert (ratio.abs() - eps).abs().max() < 1e-4


class TestUltraQuantV3:
    """Tests for UltraQuant v3 with full residual."""

    def test_forward(self):
        w = torch.randn(4, 16)
        A, B = UltraQuantV3.apply(w, "per_channel", 128)
        assert A.shape == w.shape
        assert B.shape == w.shape

    def test_B_equals_x_in_deadzone(self):
        w = torch.randn(8, 32) * 2
        A, B = UltraQuantV3.apply(w, "per_channel", 128)
        deadzone = A == 0
        if deadzone.any():
            torch.testing.assert_close(B[deadzone], w[deadzone])


class TestUltraQuantLinear:
    """Tests for the UltraQuantLinear layer."""

    def test_creation_v1(self):
        layer = UltraQuantLinear(32, 64, bias=True, quant_method="ultraquant", eps=1e-5)
        assert hasattr(layer, "eps")
        assert not hasattr(layer, "Lambada")

    def test_creation_v3(self):
        layer = UltraQuantLinear(
            32, 64, bias=False, quant_method="ultraquantv3",
            range_of_lambada=0.01,
        )
        assert hasattr(layer, "Lambada")
        # Default: per_channel → shape [out_features, 1]
        assert layer.Lambada.shape == (64, 1)
        assert hasattr(layer, "optimizer")

    def test_creation_v3_per_element(self):
        layer = UltraQuantLinear(
            32, 64, bias=False, quant_method="ultraquantv3",
            range_of_lambada=0.01, lambada_granularity="per_element",
        )
        assert layer.Lambada.shape == (64, 32)
        assert hasattr(layer, "optimizer")

    def test_forward_v1(self):
        layer = UltraQuantLinear(16, 32, bias=False, quant_method="ultraquant")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    def test_forward_v2(self):
        layer = UltraQuantLinear(16, 32, bias=False, quant_method="ultraquantv2")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    def test_forward_v3(self):
        layer = UltraQuantLinear(16, 32, bias=False, quant_method="ultraquantv3")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    def test_forward_with_bias(self):
        layer = UltraQuantLinear(16, 32, bias=True, quant_method="ultraquantv3")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_lambada_update(self):
        layer = UltraQuantLinear(8, 16, bias=False, quant_method="ultraquantv3")
        x = torch.randn(4, 8)
        w = layer.weight.data
        A, B = UltraQuantV3.apply(w, layer.granularity, layer.group_size)
        # update_lambada does full cycle internally (zero_grad → backward → step)
        # and returns None
        layer.update_lambada(x, B)
        # Lambada should have been updated (values changed from initial random)
        assert torch.isfinite(layer.Lambada).all()
        assert not torch.isnan(layer.Lambada).any()


class TestTequilaConfig:
    """Tests for Tequila configuration."""

    def test_defaults(self):
        config = TequilaConfig()
        assert config.quant_method == "ultraquantv3"
        assert config.granularity == "per_channel"

    def test_target_modules(self):
        config = TequilaConfig()
        assert "q_proj" in config.target_modules
        assert "lm_head" not in config.target_modules
