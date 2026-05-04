"""Export ternary model to bitnet.cpp format for CPU inference.

bitnet.cpp uses a packed representation where ternary weights {-1, 0, +1}
are stored as 2-bit values, achieving up to 16× memory reduction vs FP16.

Format specification:
  - Header: 4-byte magic + model config (JSON)
  - Weight data: Packed 2-bit ternary values (row-major, packed 4 vals/byte)
  - Scales: FP16 per-channel scale factors

Speedups: 1.37× to 6.17× depending on CPU architecture.
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from shared.logging import get_logger

logger = get_logger("export")

BITNET_MAGIC = 0x4249544E  # "BITN"
BITNET_VERSION = 1


@dataclass
class BitNetExportConfig:
    """Configuration for bitnet.cpp export."""

    pack_weights: bool = True
    use_fp16_scales: bool = True
    export_config_json: bool = True
    target_modules: tuple = field(
        default=(
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )


def _pack_ternary_weights(weight: torch.Tensor) -> tuple[bytes, torch.Tensor]:
    """Pack ternary values {-1, 0, +1} into 2-bit encoding.

    Encoding: 00=0, 01=+1, 10=-1

    Args:
        weight: Tensor of shape [out_features, in_features] with values in {-1, 0, +1}.

    Returns:
        (packed_bytes, scale_vector) where scale_vector has shape [out_features, 1].
    """
    w = weight.float().cpu()
    out_features, in_features = w.shape

    scales = w.abs().max(dim=-1, keepdim=True).values

    w_normalized = w / scales.clamp_min(1e-8)
    w_normalized = w_normalized.round().clamp(-1, 1)

    encoded = np.zeros_like(w_normalized.numpy(), dtype=np.uint8)
    encoded[w_normalized.numpy() > 0.5] = 1
    encoded[w_normalized.numpy() < -0.5] = 2

    padded_len = ((in_features + 3) // 4) * 4
    encoded_padded = np.zeros((out_features, padded_len), dtype=np.uint8)
    encoded_padded[:, :in_features] = encoded

    packed = np.zeros((out_features, padded_len // 4), dtype=np.uint8)
    for i in range(4):
        packed |= (encoded_padded[:, i::4] & 0x03) << (i * 2)

    return packed.tobytes(), scales


def _unpack_ternary_weights(
    packed: bytes,
    out_features: int,
    in_features: int,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Unpack 2-bit encoded ternary weights back to float tensor."""
    padded_len = ((in_features + 3) // 4) * 4
    packed_arr = np.frombuffer(packed, dtype=np.uint8).reshape(out_features, padded_len // 4)

    encoded = np.zeros((out_features, padded_len), dtype=np.int8)
    for i in range(4):
        encoded[:, i::4] = (packed_arr >> (i * 2)) & 0x03

    mapping = np.array([0, 1, -1, 0], dtype=np.float32)
    w = mapping[encoded[:, :in_features]]
    return torch.from_numpy(w) * scales


def export_to_bitnet_cpp(
    model: PreTrainedModel,
    output_path: str,
    config: Optional[BitNetExportConfig] = None,
) -> str:
    """Export ternary model weights to bitnet.cpp format.

    Args:
        model: Ternary-quantized model.
        output_path: Output directory path.
        config: Export configuration.

    Returns:
        Path to the exported model file.
    """
    if config is None:
        config = BitNetExportConfig()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to bitnet.cpp format: {output_dir}")

    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
    model_config["model_type"] = "bitnet_ternary"

    if config.export_config_json:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"  Config saved to {config_path}")

    weight_file = output_dir / "model.bitnet"
    with open(weight_file, "wb") as f:
        f.write(struct.pack("<I", BITNET_MAGIC))
        f.write(struct.pack("<I", BITNET_VERSION))

        config_json = json.dumps(model_config, separators=(",", ":"))
        config_bytes = config_json.encode("utf-8")
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)

        exported = 0
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(
                t in module_name for t in config.target_modules
            ):
                w = module.weight.data
                if w.abs().max() <= 1.1 and w.abs().min() >= -0.1:
                    w_ternary = w.round().clamp(-1, 1).to_sparse()
                    dense = w_ternary.to_dense() if w_ternary.is_sparse else w.round()
                else:
                    dense = w

                packed, scales = _pack_ternary_weights(dense)

                name_bytes = module_name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)

                f.write(struct.pack("<II", dense.shape[0], dense.shape[1]))
                f.write(struct.pack("<I", len(packed)))
                f.write(packed)

                scales_np = scales.cpu().half().numpy() if config.use_fp16_scales else scales.cpu().numpy()
                scale_bytes = scales_np.tobytes()
                f.write(struct.pack("<I", len(scale_bytes)))
                f.write(scale_bytes)

                exported += 1

        logger.info(f"  Exported {exported} layers to {weight_file}")

    total_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(t in n for t in config.target_modules)
    )
    original_size_mb = total_params * 2 / (1024 * 1024)
    packed_size_mb = weight_file.stat().st_size / (1024 * 1024)
    compression_ratio = original_size_mb / max(packed_size_mb, 0.001)

    logger.info(
        f"Export complete: {original_size_mb:.1f}MB -> {packed_size_mb:.1f}MB "
        f"({compression_ratio:.1f}× compression)"
    )
    return str(weight_file)
