"""Data processing for MiniGPT WikiText-2 experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from experiments.optimization_survey._runtime.compute import pin_memory_for_device
from experiments.optimization_survey._runtime.errors import OptionalDependencyError


class PackedTokenDataset(Dataset):
    """Dataset of fixed-length packed token chunks for causal LM."""

    def __init__(self, token_blocks: torch.Tensor) -> None:
        super().__init__()
        self.token_blocks = token_blocks

    def __len__(self) -> int:
        return int(self.token_blocks.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = self.token_blocks[idx]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _pack_split(split_texts: list[str], tokenizer: Any, seq_len: int) -> torch.Tensor:
    text = "\n\n".join(item for item in split_texts if item and item.strip())
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    usable = (len(token_ids) // seq_len) * seq_len
    if usable == 0:
        raise RuntimeError("Tokenized split is too small to produce even one sequence block.")

    packed = torch.tensor(token_ids[:usable], dtype=torch.long).view(-1, seq_len)
    return packed


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build packed token dataloaders for WikiText-2."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise OptionalDependencyError("hf", "Install extra dependencies: pip install -e .[hf]") from exc

    data_root = Path(str(cfg.data.root)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_name = str(cfg.data.dataset_name)
    dataset_config = str(cfg.data.dataset_config)
    seq_len = int(cfg.data.seq_len)

    dataset = load_dataset(dataset_name, dataset_config, cache_dir=str(data_root), streaming=bool(cfg.data.streaming))
    if bool(cfg.data.streaming):
        raise RuntimeError("streaming=true is not supported in this scaffold. Use streaming=false.")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.data.tokenizer_name), cache_dir=str(data_root))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_blocks = _pack_split(dataset["train"]["text"], tokenizer, seq_len)
    val_blocks = _pack_split(dataset["validation"]["text"], tokenizer, seq_len)
    test_blocks = _pack_split(dataset["test"]["text"], tokenizer, seq_len)

    train_set = PackedTokenDataset(train_blocks)
    val_set = PackedTokenDataset(val_blocks)
    test_set = PackedTokenDataset(test_blocks)

    batch_size = int(cfg.compute.batch_size)
    num_workers = int(cfg.compute.num_workers)
    pin_memory = pin_memory_for_device(str(cfg.compute.device))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "seq_len": seq_len,
    }
