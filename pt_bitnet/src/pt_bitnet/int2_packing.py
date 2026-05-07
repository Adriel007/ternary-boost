"""INT2 packing/unpacking for ternary weights.

Lossless bijective encoding of ternary values {-1, 0, +1} into 2-bit codes.
4 weights per byte, 16 weights per int32 (interleaved for SIMD efficiency).

Encoding:
  00 = -1
  01 =  0
  10 = +1
  11 = unused (treated as 0 on unpack)

Int32 layout follows Microsoft BitNet GPU kernel convention:
  pack_16(w[0..15]) packs 16 consecutive weights into one int32,
  interleaved as [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15].
  The 4x4 transpose groups weights for efficient SIMD dot-product.
"""

import torch


# Lookup tables for fast encode/decode (indexed by 2-bit code)
_CODE_TO_FLOAT = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
_FLOAT_TO_CODE = {-1.0: 0, 0.0: 1, 1.0: 2}


def _interleave_16(codes: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transpose interleaving to last dim of size 16.

    Sequential [0..15] → interleaved [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
    """
    *batch, n = codes.shape
    assert n == 16, f"Expected last dim 16, got {n}"
    codes = codes.view(*batch, 4, 4).transpose(-2, -1).contiguous()
    return codes.view(*batch, 16)


def _deinterleave_16(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _interleave_16.

    Interleaved → sequential [0..15]
    """
    *batch, n = packed.shape
    assert n == 16, f"Expected last dim 16, got {n}"
    packed = packed.view(*batch, 4, 4).transpose(-2, -1).contiguous()
    return packed.view(*batch, 16)


def pack_int2(weights: torch.Tensor) -> torch.Tensor:
    """Pack ternary float weights into INT2 format.

    Args:
        weights: [out_f, in_f] float tensor with values in {-1, 0, +1}.

    Returns:
        [out_f, ceil(in_f / 16)] int32 tensor.
        16 weights packed per int32, interleaved for SIMD.
        Padding (if in_f not multiple of 16) uses code 01 (= 0).
    """
    out_f, in_f = weights.shape
    packed_stride = (in_f + 15) // 16

    # Map floats to 2-bit codes: -1→0, 0→1, +1→2
    codes = weights.to(torch.int32) + 1  # -1→0, 0→1, 1→2
    codes = codes.clamp(0, 3)            # safety: any stray value → 0

    # Pad to multiple of 16 (pad with code=1 → decoded as 0)
    pad_len = packed_stride * 16 - in_f
    if pad_len > 0:
        codes = torch.nn.functional.pad(codes, (0, pad_len), value=1)

    # Reshape into groups of 16, then interleave
    codes = codes.view(out_f, packed_stride, 16)
    codes = _interleave_16(codes)  # [out_f, packed_stride, 16] interleaved

    # Pack 16 2-bit codes into one int32
    shifts = torch.arange(0, 32, 2, device=codes.device, dtype=torch.int32)
    codes = codes.to(torch.int32)
    packed = (codes << shifts).sum(dim=-1)

    return packed.to(torch.int32)


def unpack_int2(packed: torch.Tensor, in_f: int) -> torch.Tensor:
    """Unpack INT2 format back to ternary float weights.

    Args:
        packed: [out_f, ceil(in_f / 16)] int32 tensor.
        in_f: original in_features (unpadded).

    Returns:
        [out_f, in_f] float32 tensor with values in {-1.0, 0.0, 1.0}.
    """
    out_f = packed.shape[0]
    packed_stride = packed.shape[1]

    # Extract 16 2-bit codes from each int32
    shifts = torch.arange(0, 32, 2, device=packed.device, dtype=torch.int32).view(1, 1, 16)
    codes = ((packed.unsqueeze(-1) >> shifts) & 0x3).to(torch.int32)  # [out_f, stride, 16]

    # De-interleave
    codes = _deinterleave_16(codes)  # [out_f, packed_stride, 16]

    # Flatten and trim padding
    codes = codes.reshape(out_f, packed_stride * 16)
    codes = codes[:, :in_f]

    # Map codes back to floats using LUT
    lut = _CODE_TO_FLOAT.to(device=codes.device)
    return lut[codes]


def verify_roundtrip(device: str = "cpu") -> dict:
    """Test lossless roundtrip on random ternary matrices.

    Returns dict with test results for assertion in unit tests.
    """
    sizes = [
        (32, 64),
        (128, 256),
        (256, 512),
        (64, 15),     # non-multiple of 16
        (1, 1),
        (100, 100),
    ]
    results = {}
    for out_f, in_f in sizes:
        # Random ternary matrix
        w = torch.randint(-1, 2, (out_f, in_f), device=device).float()

        packed = pack_int2(w)
        unpacked = unpack_int2(packed, in_f)

        max_err = (w - unpacked).abs().max().item()
        match = torch.allclose(w, unpacked)

        results[f"{out_f}x{in_f}"] = {
            "max_error": max_err,
            "match": match,
            "packed_shape": list(packed.shape),
            "expected_stride": (in_f + 15) // 16,
        }

    return results
