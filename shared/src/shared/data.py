"""Data preparation utilities for the ternary-boost pipeline.

Supports loading from HuggingFace datasets and local JSONL files
for calibration, QAT training, and evaluation.
"""

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class CalibrationDataset(Dataset):
    """Small calibration dataset for Tequila Lambada optimization.

    Uses short text segments to learn optimal deadzone modulation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }


class QATDataset(Dataset):
    """Dataset for QAT fine-tuning with autoregressive LM loss."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": enc["attention_mask"][0],
        }


def load_calibration_texts(
    source: str = "wikitext",
    split: str = "train",
    num_samples: int = 500,
) -> list[str]:
    """Load calibration texts from common datasets.

    Args:
        source: Dataset name ("wikitext", "c4", "pile") or path to JSONL file.
        split: Dataset split.
        num_samples: Maximum number of samples to load.
    """
    if source.endswith(".jsonl"):
        texts = []
        with open(source, "r") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                import json
                data = json.loads(line)
                texts.append(data.get("text", data.get("content", line)))
        return texts

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    dataset_map = {
        "wikitext": ("wikitext", "wikitext-2-raw-v1"),
        "c4": ("c4", "en"),
        "pile": ("monology/pile-uncopyrighted", None),
    }

    if source in dataset_map:
        ds_name, ds_config = dataset_map[source]
        dataset = load_dataset(ds_name, ds_config, split=split, streaming=True)
    else:
        dataset = load_dataset(source, split=split, streaming=True)

    texts = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        text = example.get("text", example.get("content", str(example)))
        if text.strip():
            texts.append(text.strip())

    return texts


def create_qat_dataloader(
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> DataLoader:
    """Create a DataLoader for QAT training."""
    dataset = QATDataset(tokenizer, texts, max_length=max_length)

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        pad_token_id = tokenizer.pad_token_id or 0
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 256,
) -> DataLoader:
    """Create a DataLoader for Tequila calibration."""
    dataset = CalibrationDataset(tokenizer, texts, max_length=max_length)

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        pad_token_id = tokenizer.pad_token_id or 0
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
