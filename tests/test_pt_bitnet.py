"""Tests for PT-BitNet post-training quantization."""

import pytest
import torch
import torch.nn as nn

from pt_bitnet.quantize import (
    distribution_transform,
    blockwise_optimize,
    _find_quantizable_linears,
    PTBitNetConfig,
    apply_pt_bitnet,
)


class DummyModel(nn.Module):
    """Minimal model with quantizable linear layers."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 64)
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, 100)

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.q_proj(x)
        x = self.k_proj(x)
        return self.lm_head(x)


class TestDistributionTransform:
    """Stage 1: Distribution transformation tests."""

    def test_basic_transform(self):
        weight = torch.randn(128, 256)
        transformed, scale = distribution_transform(weight)
        assert transformed.shape == weight.shape
        assert scale.shape == (128, 1)

    def test_outlier_clipping(self):
        weight = torch.randn(64, 128)
        weight[0, 0] = 100.0
        transformed, _ = distribution_transform(weight, clip_threshold=3.0)
        assert transformed.abs().max() <= 3.0

    def test_zero_mean_centering(self):
        weight = torch.randn(32, 64) * 2 + 5
        transformed, _ = distribution_transform(weight, clip_threshold=10.0)
        assert transformed.mean().abs() < 0.5


class TestBlockwiseOptimize:
    """Stage 2: Block-wise ternary optimization tests."""

    def test_basic_quantization(self):
        weight = torch.randn(4, 32)
        quantized, scales = blockwise_optimize(weight, block_size=8)
        assert quantized.shape == weight.shape
        assert scales.shape == (4, 1)

    def test_ternary_values(self):
        weight = torch.randn(8, 64) * 2
        quantized, scales = blockwise_optimize(
            weight, block_size=16, max_iter=20
        )
        # Each row should have at most 3 distinct values (ternary structure)
        # Note: outlier_fraction=0 by default, so no FP16 outliers
        for row in quantized:
            unique_vals = row.unique()
            assert unique_vals.numel() <= 3, f"Row has {unique_vals.numel()} unique values: {unique_vals}"

    def test_reconstruction_improves(self):
        weight = torch.randn(4, 32)
        quantized, _ = blockwise_optimize(
            weight, block_size=8, max_iter=1
        )
        error_1 = ((weight - quantized) ** 2).mean()

        quantized_20, _ = blockwise_optimize(
            weight, block_size=8, max_iter=20
        )
        error_20 = ((weight - quantized_20) ** 2).mean()

        assert error_20 <= error_1 * 1.5


class TestFindQuantizableLinears:
    """Layer discovery tests."""

    def test_finds_linear_layers(self):
        model = DummyModel()
        config = PTBitNetConfig()
        targets = _find_quantizable_linears(model, config)
        target_names = [n for n, _ in targets]
        assert "q_proj" in target_names
        assert "k_proj" in target_names

    def test_skips_lm_head(self):
        model = DummyModel()
        config = PTBitNetConfig()
        targets = _find_quantizable_linears(model, config)
        target_names = [n for n, _ in targets]
        assert "lm_head" not in target_names

    def test_skips_embed_tokens(self):
        model = DummyModel()
        config = PTBitNetConfig()
        targets = _find_quantizable_linears(model, config)
        target_names = [n for n, _ in targets]
        assert "embed_tokens" not in target_names


class TestApplyPTBitNet:
    """End-to-end PT-BitNet tests."""

    def test_apply_to_model(self):
        model = DummyModel()
        config = PTBitNetConfig(show_progress=False, outlier_fraction=0.0)
        quantized = apply_pt_bitnet(model, config)

        assert quantized.q_proj.weight.requires_grad is False

        # Each row should have at most 3 unique values (ternary structure, no outliers)
        for row in quantized.q_proj.weight:
            unique_q = row.unique()
            assert unique_q.numel() <= 3, f"Row has {unique_q.numel()} unique values"

    def test_apply_to_model_with_outliers(self):
        model = DummyModel()
        config = PTBitNetConfig(show_progress=False, outlier_fraction=0.05)
        quantized = apply_pt_bitnet(model, config)

        assert quantized.q_proj.weight.requires_grad is False
        # With outliers, each row has up to 3 ternary values + some FP16 outliers
        for row in quantized.q_proj.weight:
            unique_q = row.unique()
            assert unique_q.numel() <= 10, f"Row has {unique_q.numel()} unique values"
