"""Data processing for Dolly 15k LoRA SFT experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from optbench.schemas import OptionalDependencyError


class SftTokenDataset(Dataset):
    """Simple in-memory dataset of tokenized SFT examples."""

    def __init__(self, records: list[dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.records[idx]


def _format_example(example: dict[str, Any], template: str) -> tuple[str, str]:
    instruction = str(example.get("instruction", "")).strip()
    context = str(example.get("context", "")).strip()
    response = str(example.get("response", "")).strip()

    context_block = ""
    if context:
        context_block = f"### Context:\n{context}\n\n"

    full_text = template.format(
        instruction=instruction,
        context=context,
        context_block=context_block,
        response=response,
    )
    prompt_text = template.format(
        instruction=instruction,
        context=context,
        context_block=context_block,
        response="",
    )
    return full_text, prompt_text


def _tokenize_records(dataset_split: Any, tokenizer: Any, cfg: DictConfig) -> list[dict[str, torch.Tensor]]:
    max_len = int(cfg.data.max_seq_len)
    template = str(cfg.data.prompt_template)

    records: list[dict[str, torch.Tensor]] = []
    for item in dataset_split:
        full_text, prompt_text = _format_example(item, template)

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = min(int(prompt_ids.numel()), max_len)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return records


def build_datamodule(cfg: DictConfig) -> dict[str, Any]:
    """Build deterministic train/val loaders for Dolly 15k SFT."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise OptionalDependencyError("lora", "Install extra dependencies: pip install -e .[lora]") from exc

    data_root = Path(str(cfg.data.root)).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(cfg.data.dataset_name), cache_dir=str(data_root))["train"]
    split = dataset.train_test_split(test_size=0.1, seed=int(cfg.seed), shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.model_name), cache_dir=str(data_root), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = _tokenize_records(split["train"], tokenizer, cfg)
    val_records = _tokenize_records(split["test"], tokenizer, cfg)

    train_set = SftTokenDataset(train_records)
    val_set = SftTokenDataset(val_records)

    batch_size = int(cfg.compute.batch_size)
    num_workers = int(cfg.compute.num_workers)
    pin_memory = str(cfg.compute.device) == "cuda"

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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": val_loader,
        "vocab_size": int(tokenizer.vocab_size),
    }
