"""Tests for INT2 ternary weight packing/unpacking."""

import pytest
import torch
from pt_bitnet.int2_packing import pack_int2, unpack_int2, verify_roundtrip


class TestInt2Packing:
    """Roundtrip correctness and edge cases for INT2 pack/unpack."""

    @pytest.mark.parametrize("out_f,in_f", [
        (32, 64),
        (128, 256),
        (64, 15),       # non-multiple of 16
        (1, 1),
        (100, 100),
        (5, 16),        # exact multiple
        (5, 32),        # 2 * 16
    ])
    def test_roundtrip_exact(self, out_f, in_f):
        """pack → unpack must reconstruct exactly."""
        w = torch.randint(-1, 2, (out_f, in_f)).float()
        packed = pack_int2(w)
        unpacked = unpack_int2(packed, in_f)
        assert torch.equal(w, unpacked), \
            f"Roundtrip failed for {out_f}x{in_f}"

    def test_packed_shape(self):
        """packed stride must be ceil(in_f / 16)."""
        w = torch.randint(-1, 2, (5, 100)).float()
        packed = pack_int2(w)
        assert packed.shape == (5, 7)  # ceil(100/16) = 7
        assert packed.dtype == torch.int32

    def test_all_values_represented(self):
        """Each ternary value (-1, 0, +1) must survive roundtrip."""
        w = torch.tensor([[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]])
        packed = pack_int2(w)
        unpacked = unpack_int2(packed, 6)
        assert torch.equal(w, unpacked)

    def test_large_random_matrix(self):
        """Stress test: 512x1024 random ternary matrix."""
        w = torch.randint(-1, 2, (512, 1024)).float()
        packed = pack_int2(w)
        unpacked = unpack_int2(packed, 1024)
        assert torch.equal(w, unpacked)
        # Compression ratio: 512*1024*2 bits / 512*64*32 bits = ~2x over fp16
        assert packed.numel() * 4 < w.numel() * 2  # smaller than fp16

    def test_verify_roundtrip_all_pass(self):
        """Built-in verification must pass all sizes."""
        results = verify_roundtrip()
        for key, result in results.items():
            assert result["match"], f"verify_roundtrip failed at {key}"
            assert result["max_error"] == 0.0
            assert result["packed_shape"][1] == result["expected_stride"]

    def test_unused_code_treated_as_zero(self):
        """Code 0b11 (3) must decode as 0 on unpack."""
        # 16 codes of 0b11 = all bits set in 32-bit int = -1 in signed int32
        packed = torch.tensor([[-1]], dtype=torch.int32)
        unpacked = unpack_int2(packed, 16)
        assert torch.all(unpacked == 0.0)

    @pytest.mark.parametrize("device_str", ["cpu"])
    def test_device_consistency(self, device_str):
        """Output must be on the same device as input."""
        if device_str == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device_str)
        w = torch.randint(-1, 2, (32, 64), device=device).float()
        packed = pack_int2(w)
        assert packed.device == device
        unpacked = unpack_int2(packed, 64)
        assert unpacked.device == device
