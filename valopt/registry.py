"""Experiment protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from omegaconf import DictConfig

from experiments.audio_transformer import AudioTransformerExperiment
from experiments.cifar_convnet import CifarConvnetExperiment
from experiments.conv1d_classify import Conv1dClassifyExperiment
from experiments.ddpm_cifar10 import DdpmCifar10Experiment
from experiments.linear_denoiser import LinearDenoiserExperiment
from experiments.lora_sft import LoraSftExperiment
from experiments.matrix_factorization import MatrixFactorizationExperiment
from experiments.minigpt_wikitext2 import MiniGptWikiText2Experiment
from experiments.mnist_mlp import MnistMlpExperiment
from experiments.poly_regression import PolyRegressionExperiment
from experiments.quadratic_bowl import QuadraticBowlExperiment
from experiments.rosenbrock_net import RosenbrockNetExperiment
from experiments.sin_regression import SinRegressionExperiment
from experiments.synthetic_logreg import SyntheticLogregExperiment
from experiments.tiny_autoencoder import TinyAutoencoderExperiment
from experiments.two_spirals import TwoSpiralsExperiment
from experiments.xor_mlp import XorMlpExperiment
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
    "synthetic_logreg": SyntheticLogregExperiment,
    "mnist_mlp": MnistMlpExperiment,
    "poly_regression": PolyRegressionExperiment,
    "rosenbrock_net": RosenbrockNetExperiment,
    "xor_mlp": XorMlpExperiment,
    "sin_regression": SinRegressionExperiment,
    "matrix_factorization": MatrixFactorizationExperiment,
    "tiny_autoencoder": TinyAutoencoderExperiment,
    "two_spirals": TwoSpiralsExperiment,
    "linear_denoiser": LinearDenoiserExperiment,
    "conv1d_classify": Conv1dClassifyExperiment,
    "quadratic_bowl": QuadraticBowlExperiment,
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
