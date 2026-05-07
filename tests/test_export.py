"""Tests for export/load pipeline (ternary + LoRA separate format + bitnet.cpp)."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from pt_bitnet.int2_packing import pack_int2
from pt_bitnet.ternary_linear import TernaryInferenceLinear
from pt_bitnet.export import (
    _extract_ternary_params,
    _export_lora_linear,
    _export_ternary_linear,
    _inject_ternary_layers,
)
def _random_ternary(out_f, in_f, sparsity=0.3):
    w = torch.zeros(out_f, in_f)
    mask = torch.rand(out_f, in_f) > sparsity
    w[mask] = (torch.randint(0, 2, (mask.sum(),)) * 2 - 1).float()
    return w


class TestTernaryParamExtraction:
    """Tests for weight → ternary param extraction."""

    def test_symmetric_extraction(self):
        """Symmetric ternary: {-1,0,+1} weights → alpha=1, mu=None."""
        w = _random_ternary(32, 64)  # values in {-1, 0, +1}, alpha=1 implicitly
        params = _extract_ternary_params(w)
        T = params["T"]
        alpha = params["alpha"]
        assert T.shape == w.shape
        assert alpha.shape == (32, 1)
        assert not params["asymmetric"]
        assert params["mu"] is None

        # Reconstruction: alpha*T must match w exactly
        recon = alpha * T
        assert torch.allclose(w, recon, atol=1e-5)

    def test_asymmetric_extraction(self):
        """Asymmetric: w = alpha*T + mu, balanced T so mean(T)≈0."""
        out_f, in_f = 32, 64
        # Balanced T: equal +1 and -1 per row (mean ≈ 0)
        T = torch.zeros(out_f, in_f)
        for i in range(out_f):
            half = min(10, in_f // 3)
            T[i, :half] = 1.0
            T[i, half:2*half] = -1.0
            T[i] = T[i, torch.randperm(in_f)]
        alpha_val = torch.rand(out_f, 1) * 2.0 + 1.0
        mu_val = torch.randn(out_f, 1) * 0.5
        w = alpha_val * T + mu_val

        params = _extract_ternary_params(w)
        # With balanced T, mu from row_mean ≈ actual mu → asymmetric preferred
        assert params["asymmetric"], "balanced T → mu ≈ row_mean → asym better"
        assert params["mu"] is not None
        recon = params["alpha"] * params["T"] + params["mu"]
        assert torch.allclose(w, recon, atol=1e-4)

    def test_weight_with_zeros(self):
        """Sparsity preserved: zero weights → T[i,j] == 0."""
        w = torch.zeros(16, 32)
        w[0, 0] = 1.0
        w[1, 1] = -1.0
        params = _extract_ternary_params(w)
        T = params["T"]
        assert T[0, 0] == 1.0
        assert T[1, 1] == -1.0
        # Rows with mean=0 should have T=0 (no shift from mu)
        assert (T[2:, :] == 0).all()


class TestExportLoraLinear:
    """Tests for exporting LoRALinear → ternary_params + lora_weights."""

    def test_export_roundtrip(self):
        """Export + inject must preserve forward pass output."""
        from pt_bitnet.lora import LoRALinear, LoRAConfig

        tiny = nn.Sequential()
        base = nn.Linear(64, 32, bias=False)
        base.weight.data = _random_ternary(32, 64)
        base.weight.requires_grad = False
        lora_cfg = LoRAConfig(rank=4)
        lora_layer = LoRALinear(base, lora_cfg)
        tiny.layer = lora_layer

        x = torch.randn(2, 8, 64)
        ref_out = tiny.layer(x).detach()

        ternary_params = {}
        lora_weights = {}
        _export_lora_linear(ternary_params, lora_weights, "layer", tiny.layer)

        tp = ternary_params["layer"]
        assert tp["has_lora"]
        assert tp["int2_packed"].shape == (32, 4)  # ceil(64/16)=4
        assert tp["alpha"].shape == (32, 1)
        assert "layer.lora_A" in lora_weights
        assert "layer.lora_B" in lora_weights
        assert lora_weights["layer.lora_A"].shape == (4, 64)
        assert lora_weights["layer.lora_B"].shape == (32, 4)

        tiny2 = nn.Sequential()
        tiny2.layer = nn.Linear(64, 32)
        _inject_ternary_layers(tiny2, ternary_params, lora_weights, device="cpu")

        assert isinstance(tiny2.layer, TernaryInferenceLinear)
        test_out = tiny2.layer(x)
        assert torch.allclose(ref_out, test_out, atol=1e-5), \
            f"Max diff: {(ref_out - test_out).abs().max().item()}"


class TestExportTernaryLinear:
    """Tests for exporting plain ternary nn.Linear (no LoRA)."""

    def test_export_roundtrip(self):
        """Export + inject (ternary only, no LoRA) must preserve output."""
        tiny = nn.Sequential()
        layer = nn.Linear(64, 32, bias=True)
        layer.weight.data = _random_ternary(32, 64)
        layer.weight.requires_grad = False
        layer.bias.data = torch.randn(32)
        tiny.layer = layer

        x = torch.randn(2, 8, 64)
        ref_out = tiny.layer(x).detach()

        ternary_params = {}
        _export_ternary_linear(ternary_params, "layer", tiny.layer)
        tp = ternary_params["layer"]
        assert not tp["has_lora"]
        assert tp["int2_packed"].shape == (32, 4)

        tiny2 = nn.Sequential()
        tiny2.layer = nn.Linear(64, 32)
        _inject_ternary_layers(tiny2, ternary_params, {}, device="cpu")

        assert isinstance(tiny2.layer, TernaryInferenceLinear)
        test_out = tiny2.layer(x)
        assert torch.allclose(ref_out, test_out, atol=1e-5), \
            f"Max diff: {(ref_out - test_out).abs().max().item()}"


class TestInjectTernaryLayers:
    """Tests for layer injection into nested models."""

    def test_nested_module_injection(self):
        """Injection must work on nested modules (transformer blocks)."""
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)

        class MiniModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block() for _ in range(2)])

        model = MiniModel()
        for i, block in enumerate(model.layers):
            block.q_proj.weight.data = _random_ternary(64, 64)
            block.q_proj.weight.requires_grad = False
            block.k_proj.weight.data = _random_ternary(64, 64)
            block.k_proj.weight.requires_grad = False

        ternary_params = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not module.weight.requires_grad:
                _export_ternary_linear(ternary_params, name, module)

        assert len(ternary_params) == 4

        model2 = MiniModel()
        _inject_ternary_layers(model2, ternary_params, {}, device="cpu")

        for i in range(2):
            assert isinstance(model2.layers[i].q_proj, TernaryInferenceLinear)
            assert isinstance(model2.layers[i].k_proj, TernaryInferenceLinear)

    def test_skip_nontarget(self):
        """Non-target layers must NOT be replaced."""
        model = nn.Sequential()
        model.fc = nn.Linear(64, 32)

        ternary_params = {}
        _inject_ternary_layers(model, ternary_params, {}, device="cpu")
        assert isinstance(model.fc, nn.Linear)  # unchanged


# ── Legacy bitnet.cpp export tests ──────────────────────────────────

class TestBitNetPack:
    """Tests for bitnet.cpp-style ternary weight packing (legacy)."""

    def test_pack_basic(self):
        from eval.export_bitnet import _pack_ternary_weights, _unpack_ternary_weights

        w = torch.tensor([
            [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0],
        ])
        packed, scales = _pack_ternary_weights(w)
        assert isinstance(packed, bytes)
        assert len(packed) > 0
        assert scales.shape == (1, 1)

    def test_roundtrip(self):
        from eval.export_bitnet import _pack_ternary_weights, _unpack_ternary_weights

        w = torch.tensor([
            [1.0, -1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0, 1.0],
        ])
        packed, scales = _pack_ternary_weights(w)
        unpacked = _unpack_ternary_weights(packed, w.shape[0], w.shape[1], scales)
        torch.testing.assert_close(unpacked, w)

    def test_large_matrix_roundtrip(self):
        from eval.export_bitnet import _pack_ternary_weights, _unpack_ternary_weights

        w = (torch.randint(-1, 2, (64, 128))).float()
        packed, scales = _pack_ternary_weights(w)
        unpacked = _unpack_ternary_weights(packed, w.shape[0], w.shape[1], scales)
        torch.testing.assert_close(unpacked, w)

    def test_magic_number(self):
        from eval.export_bitnet import BITNET_MAGIC
        assert BITNET_MAGIC == 0x4249544E

    def test_packing_compression_ratio(self):
        from eval.export_bitnet import _pack_ternary_weights

        w = (torch.randint(-1, 2, (256, 512))).float()
        packed, scales = _pack_ternary_weights(w)
        original_size = w.numel() * 2
        packed_size = len(packed) + scales.numel() * 2
        ratio = original_size / packed_size
        assert ratio > 2.0

    def test_default_config(self):
        from eval.export_bitnet import BitNetExportConfig
        config = BitNetExportConfig()
        assert config.pack_weights is True
        assert config.use_fp16_scales is True
        assert "q_proj" in config.target_modules
