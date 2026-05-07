"""Chat configuration and model registry management."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".config" / "tchat"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class ModelEntry:
    name: str
    path: str
    backend: str = "transformers"  # "transformers" or "bitnet_cpp"
    description: str = ""
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    lambada_granularity: str = "per_channel"  # "per_channel" or "per_element"
    lora_rank: int = 32       # LoRA rank (0 = disable LoRA fine-tuning)
    lora_steps: int = 500     # LoRA optimization steps
    merge_lora: bool = False  # Merge LoRA into weights? False = keep separate (compressed)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        default_instance = cls(name="", path="")
        defaults = default_instance.to_dict()
        return cls(**{k: d.get(k, defaults[k]) for k in defaults})


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "mistral-7b": ModelEntry(
        name="mistral-7b",
        path="mistralai/Mistral-7B-Instruct-v0.3",
        backend="transformers",
        description="Mistral 7B Instruct — 7B params, ternary: ~1.2 GB (~12 min GPU, ~40 min CPU)",
        max_length=4096,
    ),
    "olmo-7b": ModelEntry(
        name="olmo-7b",
        path="allenai/OLMo-7B-Instruct",
        backend="transformers",
        description="OLMo 7B Instruct — fully open (data+code+weights), ternary: ~1.2 GB (~12 min GPU)",
        max_length=4096,
    ),
    "falcon-7b": ModelEntry(
        name="falcon-7b",
        path="tiiuae/falcon-7b-instruct",
        backend="transformers",
        description="Falcon 7B Instruct — ternary: ~1.2 GB (~12 min GPU)",
        max_length=2048,
    ),
    "qwen2.5-7b": ModelEntry(
        name="qwen2.5-7b",
        path="Qwen/Qwen2.5-7B-Instruct",
        backend="transformers",
        description="Qwen2.5 7B Instruct — multilingual, ternary: ~1.2 GB (~12 min GPU)",
        max_length=4096,
    ),
    "phi-3-small": ModelEntry(
        name="phi-3-small",
        path="microsoft/Phi-3-small-8k-instruct",
        backend="transformers",
        description="Phi-3 Small 7B — MIT license, ternary: ~1.2 GB (~12 min GPU)",
        max_length=4096,
    ),
    "phi-3-medium": ModelEntry(
        name="phi-3-medium",
        path="microsoft/Phi-3-medium-4k-instruct",
        backend="transformers",
        description="Phi-3 Medium 14B — ternary: ~2.4 GB (~22 min GPU)",
        max_length=4096,
    ),
    "phi-2": ModelEntry(
        name="phi-2",
        path="microsoft/phi-2",
        backend="transformers",
        description="Phi-2 2.7B — test/validation model, ternary: ~0.5 GB (~4 min GPU)",
        max_length=2048,
    ),
}


@dataclass
class ChatConfig:
    model: str = "mistral-7b"
    cache_dir: str = "./cache"
    system_prompt: str = "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
    max_turns: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.1
    enable_thinking: bool = False
    thinking_prefix: str = "Let me think through this step by step.\n\n"
    stream: bool = True
    show_stats: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ChatConfig":
        """Create config from dict, using dataclass defaults for missing keys."""
        default_instance = cls()
        defaults = default_instance.to_dict()
        return cls(**{k: d.get(k, defaults[k]) for k in defaults})


def load_config() -> ChatConfig:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return ChatConfig.from_dict(data)
    return ChatConfig()


def save_config(config: ChatConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config.to_dict(), indent=2))


def list_models() -> dict[str, ModelEntry]:
    custom_dir = CONFIG_DIR / "models"
    registry = dict(MODEL_REGISTRY)
    if custom_dir.exists():
        for f in sorted(custom_dir.glob("*.json")):
            entry = ModelEntry.from_dict(json.loads(f.read_text()))
            registry[entry.name] = entry
    return registry


def add_custom_model(entry: ModelEntry) -> None:
    custom_dir = CONFIG_DIR / "models"
    custom_dir.mkdir(parents=True, exist_ok=True)
    path = custom_dir / f"{entry.name}.json"
    path.write_text(json.dumps(entry.to_dict(), indent=2))
