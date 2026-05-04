"""Tests for ParetoQ quantization + ZeroQAT."""

import pytest
import torch
import torch.nn as nn

from paretoq.utils_quant import (
    QuantizeLinear,
    LsqBinaryTernaryExtension,
    StretchedElasticQuant,
)
from paretoq.zo_optim import ZeroOrderOptimizer, ZeroQATConfig


class TestLSQQuantization:
    """Tests for LSQ-based quantization functions."""

    def test_ternary_quantization_forward(self):
        w = torch.randn(64, 128, requires_grad=False)
        alpha = torch.tensor([[0.5]] * 64)
        result = LsqBinaryTernaryExtension.apply(w, alpha, 1, False)
        assert result.shape == w.shape

    def test_ternary_output_values(self):
        w = torch.randn(32, 64)
        alpha = torch.tensor([[0.3]] * 32)
        result = LsqBinaryTernaryExtension.apply(w, alpha, 1, False)
        unique = result.unique()
        assert unique.numel() <= 2

    def test_stretched_elastic_forward(self):
        w = torch.randn(64, 128)
        alpha = torch.tensor([[0.5]] * 64)
        result = StretchedElasticQuant.apply(w, alpha, 0, False)
        assert result.shape == w.shape

    def test_fp16_passthrough(self):
        w = torch.randn(32, 64)
        alpha = torch.tensor([[1.0]] * 32)
        result = LsqBinaryTernaryExtension.apply(w, alpha, 16, False)
        torch.testing.assert_close(result, w)


class TestQuantizeLinear:
    """Tests for QuantizeLinear layer."""

    def test_creation(self):
        layer = QuantizeLinear(64, 128, bias=True, w_bits=1)
        assert hasattr(layer, "weight_clip_val")
        assert layer.weight_clip_val.shape == (128, 1)

    def test_forward_ternary(self):
        layer = QuantizeLinear(32, 64, bias=False, w_bits=1)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_forward_fp16(self):
        layer = QuantizeLinear(32, 64, bias=False, w_bits=16)
        x = torch.randn(4, 32)
        out = layer(x)
        expected = nn.functional.linear(x, layer.weight)
        torch.testing.assert_close(out, expected)

    def test_forward_with_bias(self):
        layer = QuantizeLinear(32, 64, bias=True, w_bits=1)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_weight_clip_val_initialization(self):
        layer = QuantizeLinear(32, 64, w_bits=2)
        manual_scale = torch.randn(64, 1).abs()
        layer.weight_clip_val.data.copy_(manual_scale)
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 64)


class TestZeroOrderOptimizer:
    """Tests for ZeroQAT zero-order optimizer."""

    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    @pytest.fixture
    def zo_config(self):
        return ZeroQATConfig(
            perturbation_scale=1e-3,
            learning_rate=1e-5,
            target_modules=("0", "2"),
        )

    def test_initialization(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        assert len(opt.named_params) >= 2

    def test_gradient_estimation_shape(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        batch = {"input": torch.randn(2, 16), "labels": torch.randint(0, 4, (2,))}

        def loss_fn(outputs, lbls):
            return nn.functional.cross_entropy(outputs, lbls)

        grads = opt.estimate_gradient(batch, loss_fn)
        for name, param in opt.named_params:
            assert name in grads
            assert grads[name].shape == param.shape

    def test_gradient_estimation_finite(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        batch = {"input": torch.randn(4, 16), "labels": torch.randint(0, 4, (4,))}

        grads = opt.estimate_gradient(batch, lambda o, l: nn.functional.cross_entropy(o, l))
        for g in grads.values():
            assert torch.isfinite(g).all()

    def test_step_updates_parameters(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        orig_params = {
            name: param.data.clone() for name, param in opt.named_params
        }
        grads = {name: torch.randn_like(param.data) * 0.01
                 for name, param in opt.named_params}
        opt.step(grads)
        for name, param in opt.named_params:
            assert not torch.equal(param.data, orig_params[name])

    def test_layerwise_estimation(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        batch = {"input": torch.randn(2, 16), "labels": torch.randint(0, 4, (2,))}

        grads = opt.estimate_gradient_layerwise(
            batch, lambda o, l: nn.functional.cross_entropy(o, l)
        )
        assert len(grads) >= 2
        for g in grads.values():
            assert torch.isfinite(g).all()

    def test_param_vector_roundtrip(self, simple_model, zo_config):
        opt = ZeroOrderOptimizer(simple_model, zo_config)
        vec = opt.get_param_vector()
        opt.set_param_vector(vec)
        vec2 = opt.get_param_vector()
        torch.testing.assert_close(vec, vec2)
