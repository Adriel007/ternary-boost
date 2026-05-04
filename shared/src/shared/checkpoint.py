import json
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file
from transformers import PreTrainedModel, PreTrainedTokenizer


def save_checkpoint(
    model: PreTrainedModel,
    output_path: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Save model in safetensors format with HuggingFace-compatible structure."""
    os.makedirs(output_path, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(output_path, "model.safetensors"))

    model.config.save_pretrained(output_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    if metadata is not None:
        with open(os.path.join(output_path, "pipeline_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return output_path


def load_checkpoint(
    model_cls,
    checkpoint_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    """Load model from safetensors checkpoint with compatibility handling."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path)
    model = model_cls.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=dtype or torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    return model


def copy_config(src_path: str, dst_path: str) -> None:
    """Copy HuggingFace config files between directories."""
    for fname in ["config.json", "tokenizer_config.json", "tokenizer.json",
                   "special_tokens_map.json", "tokenizer.model"]:
        src = os.path.join(src_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_path, fname))
