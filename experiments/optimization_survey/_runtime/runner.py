"""Hydra-instantiated optimization-survey experiment evaluator."""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from experiments.optimization_survey._runtime.baselines import (
    load_baseline_profile,
    validate_baseline_profile,
)
from experiments.optimization_survey._runtime.candidate_loader import load_candidate_builder
from experiments.optimization_survey._runtime.errors import OptionalDependencyError
from experiments.optimization_survey._runtime.optimizer_api import (
    NamedParameters,
    OptimizerBuilder,
    OptimizerControllerProtocol,
)
from experiments.optimization_survey._runtime.scoring import compute_score

LOGGER = logging.getLogger(__name__)


class BuiltinOptimizerController:
    """Controller wrapper for builtin torch optimizers with scheduler support."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_steps: int,
        optimizer_name: str,
        base_optimizer_params: dict[str, Any],
        base_scheduler_cfg: dict[str, Any],
    ) -> None:
        self.optimizer_name = optimizer_name.lower()
        self.base_optimizer_params = dict(base_optimizer_params)
        self.base_scheduler_cfg = dict(base_scheduler_cfg)

        self.named_parameters: NamedParameters = [
            (name, param) for name, param in model.named_parameters() if param.requires_grad
        ]

        optimizer_kwargs = dict(self.base_optimizer_params)
        if "betas" in optimizer_kwargs and isinstance(optimizer_kwargs["betas"], list):
            optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])

        params_object = self._build_params_object(optimizer_kwargs)
        self.optimizer = self._build_torch_optimizer(params_object, optimizer_kwargs)

        total_steps = max(1, int(max_steps))
        self.scheduler = self._build_scheduler(self.optimizer, self.base_scheduler_cfg, total_steps=total_steps)

    def _build_params_object(self, optimizer_kwargs: dict[str, Any]) -> Any:
        if self.optimizer_name not in {"adam", "adamw"}:
            return [param for _, param in self.named_parameters]

        weight_decay = float(optimizer_kwargs.get("weight_decay", 0.0))
        if weight_decay <= 0:
            return [param for _, param in self.named_parameters]

        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters:
            lower = name.lower()
            if (
                lower.endswith("bias")
                or "ln_" in lower
                or "layernorm" in lower
                or ".ln" in lower
                or "norm" in lower
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        if not decay or not no_decay:
            return [param for _, param in self.named_parameters]

        optimizer_kwargs.pop("weight_decay", None)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _build_torch_optimizer(self, params_object: Any, optimizer_kwargs: dict[str, Any]) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params_object, **optimizer_kwargs)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params_object, **optimizer_kwargs)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params_object, **optimizer_kwargs)

        raise ValueError(
            f"Unsupported builtin optimizer '{self.optimizer_name}'. "
            "Use sgd, adam, adamw, or provide an organism_dir with implementation.py."
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: dict[str, Any],
        total_steps: int,
    ) -> LambdaLR | None:
        scheduler_type = str(scheduler_cfg.get("type", "constant")).lower()
        if scheduler_type in {"none", "off", "constant"}:
            return None

        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
        if warmup_steps <= 0:
            warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))
            warmup_steps = int(max(0, total_steps * warmup_ratio))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, float(step + 1) / float(warmup_steps))

            if scheduler_type in {"cosine", "warmup_cosine"}:
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return 1.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self, weights, grads, activations, step_fn=None) -> None:
        del weights, grads, activations, step_fn
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)
            return
        for _, param in self.named_parameters:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_baseline_path(cfg: DictConfig) -> Path | None:
    baseline_cfg = cfg.get("baseline")
    if baseline_cfg is None:
        return None
    profile_path = baseline_cfg.get("profile_path")
    if not profile_path:
        return None
    return Path(str(profile_path)).expanduser().resolve()


def _load_baseline_for_run(cfg: DictConfig) -> tuple[dict[str, Any] | None, str | None]:
    path = _resolve_baseline_path(cfg)
    if path is None:
        return None, None
    try:
        payload = _load_json(path)
        validate_baseline_profile(payload, str(cfg.name))
        return payload, None
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def _save_stats_baseline(cfg: DictConfig, report: dict[str, Any]) -> None:
    path = _resolve_baseline_path(cfg)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_payload = {
        "objective_name": report.get("objective_name", "train_loss"),
        "objective_direction": report.get("objective_direction", "min"),
        "objective_last": report.get("objective_last"),
        "steps": report.get("steps"),
    }
    validate_baseline_profile(baseline_payload, str(cfg.name))
    path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True), encoding="utf-8")


def make_builtin_optimizer_builder(exp_cfg: DictConfig) -> tuple[OptimizerBuilder, str, str]:
    """Return builtin optimizer candidate builder for optimization survey."""

    defaults = OmegaConf.to_container(exp_cfg.optimizer_defaults, resolve=True)
    assert isinstance(defaults, dict)

    optimizer_name = str(defaults.get("name", "sgd")).lower()
    base_params = deepcopy(defaults.get("params", {}))
    if not isinstance(base_params, dict):
        raise ValueError(f"optimizer_defaults.params must be dict for {exp_cfg.name}")

    base_scheduler_cfg = deepcopy(defaults.get("scheduler", {}))
    if not isinstance(base_scheduler_cfg, dict):
        base_scheduler_cfg = {}

    def builder(model: torch.nn.Module, max_steps: int) -> OptimizerControllerProtocol:
        return BuiltinOptimizerController(
            model=model,
            max_steps=max_steps,
            optimizer_name=optimizer_name,
            base_optimizer_params=base_params,
            base_scheduler_cfg=base_scheduler_cfg,
        )

    return builder, f"builtin:{optimizer_name}", optimizer_name


class OptimizationSurveyExperimentEvaluator:
    """Hydra-instantiated optimization-survey evaluator with organism-dir contract."""

    def __init__(self, experiment_target: str, **_: Any) -> None:
        self.experiment_target = experiment_target

    def _resolve_optimizer(
        self,
        organism_dir: str | None,
        cfg: DictConfig,
    ) -> tuple[OptimizerBuilder, str | None, str]:
        if organism_dir:
            implementation_path = Path(str(organism_dir)).expanduser().resolve() / "implementation.py"
            builder, path_obj, implementation_name = load_candidate_builder(str(implementation_path))
            return builder, str(path_obj), implementation_name
        return make_builtin_optimizer_builder(cfg)

    def evaluate_organism(self, organism_dir: str | None, cfg: DictConfig) -> dict[str, Any]:
        baseline_profile, baseline_error = _load_baseline_for_run(cfg)
        baseline_path = _resolve_baseline_path(cfg)
        cfg.runtime.baseline_path = str(baseline_path) if baseline_path is not None else None
        cfg.runtime.baseline_last_train_loss = (
            float(baseline_profile["objective_last"]) if baseline_profile is not None else None
        )
        cfg.runtime.baseline_load_error = baseline_error

        experiment_cls = get_class(self.experiment_target)
        experiment = experiment_cls()
        datamodule = experiment.build_datamodule(cfg)
        model = experiment.build_model(cfg)
        optimizer_factory, implementation_path, implementation_name = self._resolve_optimizer(organism_dir, cfg)
        report = experiment.train(cfg, model, datamodule, optimizer_factory)

        if not isinstance(report, dict):
            raise TypeError(f"Experiment '{cfg.name}' must return a dict report.")

        report = dict(report)
        report.setdefault("status", "ok")
        report["organism_dir"] = str(Path(organism_dir).expanduser().resolve()) if organism_dir else None
        report["implementation_path"] = implementation_path
        report["implementation_name"] = implementation_name
        report["baseline_error"] = baseline_error

        if str(cfg.runtime.mode) == "stats":
            _save_stats_baseline(cfg, report)
            report["score"] = 0.0
            return report

        score, score_error = compute_score(
            report,
            experiment_name=str(cfg.name),
            baseline_profile=baseline_profile,
            normalization_cfg=OmegaConf.to_container(cfg.get("normalization", {}), resolve=True),
        )
        if score_error is not None:
            report["status"] = "failed"
            report["error_msg"] = score_error
            report["score"] = None
        else:
            report["score"] = score
        return report
