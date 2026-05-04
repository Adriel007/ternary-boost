"""Tests for bitnet.cpp export utilities."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from eval.export_bitnet import (
    _pack_ternary_weights,
    _unpack_ternary_weights,
    BitNetExportConfig,
    BITNET_MAGIC,
)


class TestPackTernaryWeights:
    """Tests for ternary weight packing/unpacking."""

    def test_pack_basic(self):
        w = torch.tensor([
            [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0],
        ])
        packed, scales = _pack_ternary_weights(w)
        assert isinstance(packed, bytes)
        assert len(packed) > 0
        assert scales.shape == (1, 1)

    def test_roundtrip(self):
        w = torch.tensor([
            [1.0, -1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0, 1.0],
        ])
        packed, scales = _pack_ternary_weights(w)
        unpacked = _unpack_ternary_weights(packed, w.shape[0], w.shape[1], scales)
        torch.testing.assert_close(unpacked, w)

    def test_large_matrix_roundtrip(self):
        w = (torch.randint(-1, 2, (64, 128))).float()
        packed, scales = _pack_ternary_weights(w)
        unpacked = _unpack_ternary_weights(packed, w.shape[0], w.shape[1], scales)
        torch.testing.assert_close(unpacked, w)

    def test_magic_number(self):
        assert BITNET_MAGIC == 0x4249544E

    def test_packing_compression_ratio(self):
        w = (torch.randint(-1, 2, (256, 512))).float()
        packed, scales = _pack_ternary_weights(w)
        original_size = w.numel() * 2
        packed_size = len(packed) + scales.numel() * 2
        ratio = original_size / packed_size
        assert ratio > 2.0


class TestBitNetExportConfig:
    """Tests for export configuration."""

    def test_defaults(self):
        config = BitNetExportConfig()
        assert config.pack_weights is True
        assert config.use_fp16_scales is True
        assert "q_proj" in config.target_modules
