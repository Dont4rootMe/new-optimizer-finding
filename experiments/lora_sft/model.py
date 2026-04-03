"""Model creation for LoRA/QLoRA SFT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from optbench.schemas import OptionalDependencyError


def _dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _resolve_target_modules(model: nn.Module, preferred: list[str]) -> list[str]:
    linear_modules = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    selected = [item for item in preferred if any(name.endswith(item) for name in linear_modules)]
    if selected:
        return selected

    fallback_names = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query_key_value"]
    fallback = [item for item in fallback_names if any(name.endswith(item) for name in linear_modules)]
    if fallback:
        return fallback

    unique_suffixes: list[str] = []
    for name in linear_modules:
        suffix = name.split(".")[-1]
        if suffix not in unique_suffixes:
            unique_suffixes.append(suffix)
        if len(unique_suffixes) >= 4:
            break
    return unique_suffixes


def build_model(cfg: DictConfig):
    """Build Causal LM with LoRA adapters (and optional 4-bit quantization)."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise OptionalDependencyError("lora", "Install extra dependencies: pip install -e .[lora]") from exc

    precision = str(cfg.compute.precision)
    torch_dtype = _dtype_from_precision(precision)
    model_name = str(cfg.model.model_name)
    quantization = str(cfg.model.quantization).lower()

    if quantization == "4bit":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError("lora", "QLoRA requires bitsandbytes: pip install -e .[lora]") from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

    preferred_targets = [str(item) for item in cfg.lora.target_modules]
    target_modules = _resolve_target_modules(model, preferred_targets)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    if bool(cfg.compute.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    return model
