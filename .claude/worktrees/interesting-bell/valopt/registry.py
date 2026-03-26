"""Experiment protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from omegaconf import DictConfig

from experiments.audio_transformer import AudioTransformerExperiment
from experiments.cifar_convnet import CifarConvnetExperiment
from experiments.ddpm_cifar10 import DdpmCifar10Experiment
from experiments.lora_sft import LoraSftExperiment
from experiments.minigpt_wikitext2 import MiniGptWikiText2Experiment
from valopt.optimizer_api import OptimizerBuilder


class ExperimentProtocol(Protocol):
    """Unified experiment interface for runner compatibility."""

    name: str

    def build_datamodule(self, cfg: DictConfig) -> dict[str, Any]:
        ...

    def build_model(self, cfg: DictConfig) -> torch.nn.Module:
        ...

    def train(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        datamodule: dict[str, Any],
        optimizer_factory: OptimizerBuilder,
    ) -> dict[str, Any]:
        ...

    def evaluate(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        datamodule: dict[str, Any],
    ) -> dict[str, float]:
        ...


_EXPERIMENTS: dict[str, type[ExperimentProtocol]] = {
    "cifar_convnet": CifarConvnetExperiment,
    "audio_transformer": AudioTransformerExperiment,
    "minigpt_wikitext2": MiniGptWikiText2Experiment,
    "ddpm_cifar10": DdpmCifar10Experiment,
    "lora_sft": LoraSftExperiment,
}


def list_experiments() -> list[str]:
    """Return sorted registered experiment names."""

    return sorted(_EXPERIMENTS.keys())


def create_experiment(name: str) -> ExperimentProtocol:
    """Instantiate registered experiment by string name."""

    if name not in _EXPERIMENTS:
        available = ", ".join(list_experiments())
        raise KeyError(f"Unknown experiment '{name}'. Available: {available}")
    return _EXPERIMENTS[name]()
