"""Main orchestration logic for running experiments."""

from __future__ import annotations

import hashlib
import logging
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from optbench.optimizer_api import NamedParameters, OptimizerBuilder, OptimizerControllerProtocol
from optbench.registry import create_experiment
from optbench.schemas import OptionalDependencyError, RunResult, validate_run_result_dict
from optbench.utils.baselines import baseline_path, inject_baseline_threshold
from optbench.utils.import_utils import load_optimizer_builder
from optbench.utils.io import ensure_dir, save_json, save_yaml_text
from optbench.utils.seed import set_seed

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
        """Build optimizer parameter object from named parameters.

        For Adam/AdamW with weight decay, split by parameter name to exclude
        bias and norm-like layers from decay.
        """

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

    def _build_torch_optimizer(
        self,
        params_object: Any,
        optimizer_kwargs: dict[str, Any],
    ) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params_object, **optimizer_kwargs)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params_object, **optimizer_kwargs)
        if self.optimizer_name == "adamw":
            return torch.optim.AdamW(params_object, **optimizer_kwargs)

        raise ValueError(
            f"Unsupported builtin optimizer '{self.optimizer_name}'. "
            "Use sgd, adam, adamw, or provide optimizer_path in mode=run."
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

    def step(
        self,
        weights: dict[str, torch.Tensor],
        grads: dict[str, torch.Tensor],
        activations: dict[str, torch.Tensor],
        step_fn=None,
    ) -> None:
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


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _get_target_quality(target_cfg: DictConfig) -> float:
    target_dict = OmegaConf.to_container(target_cfg, resolve=True)
    if isinstance(target_dict, dict) and "value" in target_dict:
        return float(target_dict["value"])
    if isinstance(target_dict, dict) and "improvement_ratio" in target_dict:
        return float(target_dict["improvement_ratio"])
    return float("nan")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    if output != output:  # NaN check
        return default
    return output


def _select_device(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device_str


def _make_builtin_optimizer_builder(exp_cfg: DictConfig) -> tuple[OptimizerBuilder, str, str]:
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


class ExperimentRunner:
    """Coordinates experiment execution for all supported modes."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(self) -> list[dict[str, Any]]:
        mode = str(self.cfg.mode)
        if mode not in {"run", "stats", "smoke"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use: run, stats, smoke.")

        set_seed(int(self.cfg.seed), bool(self.cfg.deterministic))

        external_optimizer: tuple[OptimizerBuilder, str, str] | None = None
        if mode == "run":
            if not self.cfg.get("optimizer_path"):
                raise ValueError("mode=run requires optimizer_path=/path/to/optimizer.py")
            builder, path_obj, opt_name = load_optimizer_builder(str(self.cfg.optimizer_path))
            external_optimizer = (builder, str(path_obj), opt_name)

        enabled = self._enabled_experiments()
        LOGGER.info("Selected mode=%s. Enabled experiments=%s", mode, [name for name, _ in enabled])

        results: list[dict[str, Any]] = []
        for exp_name, raw_cfg in enabled:
            exp_cfg = self._prepare_experiment_cfg(exp_name, raw_cfg, mode)

            if external_optimizer is None:
                optimizer_factory, optimizer_path, optimizer_name = _make_builtin_optimizer_builder(exp_cfg)
            else:
                optimizer_factory, optimizer_path, optimizer_name = external_optimizer

            run_id, timestamp = self._build_run_id(exp_name, exp_cfg, mode, optimizer_path)
            resolved_cfg_path = self._save_resolved_config(exp_name, run_id, mode, exp_cfg)

            LOGGER.info("Running experiment=%s run_id=%s", exp_name, run_id)
            result_payload = self._execute_single(
                exp_name=exp_name,
                exp_cfg=exp_cfg,
                optimizer_factory=optimizer_factory,
                optimizer_path=optimizer_path,
                optimizer_name=optimizer_name,
                run_id=run_id,
                timestamp=timestamp,
                resolved_cfg_path=resolved_cfg_path,
                mode=mode,
            )

            result_path = self._persist_result(exp_name, run_id, mode, optimizer_path, result_payload)
            result_payload["result_path"] = str(result_path)
            results.append(result_payload)
            LOGGER.info(
                "Finished experiment=%s status=%s result=%s",
                exp_name,
                result_payload.get("status"),
                result_path,
            )

        return results

    def _enabled_experiments(self) -> list[tuple[str, DictConfig]]:
        enabled: list[tuple[str, DictConfig]] = []
        for exp_name, exp_cfg in self.cfg.experiments.items():
            if bool(exp_cfg.get("enabled", False)):
                enabled.append((str(exp_name), exp_cfg))
        if not enabled:
            raise ValueError("No enabled experiments found in cfg.experiments.*.enabled")
        return enabled

    def _prepare_experiment_cfg(self, exp_name: str, exp_cfg: DictConfig, mode: str) -> DictConfig:
        merged = OmegaConf.create(OmegaConf.to_container(exp_cfg, resolve=False))

        merged.seed = int(self.cfg.seed)
        merged.deterministic = bool(self.cfg.deterministic)
        merged.runtime = {"mode": mode, "smoke": mode == "smoke"}

        merged.compute.device = _select_device(str(self.cfg.device))
        merged.compute.precision = str(self.cfg.precision)
        merged.compute.num_workers = int(self.cfg.num_workers)

        if mode == "smoke":
            smoke_steps = int(merged.compute.smoke_steps)
            merged.compute.max_steps = smoke_steps

        if mode == "run":
            self._apply_run_validation_overrides(exp_name=exp_name, merged=merged)
            inject_baseline_threshold(
                merged,
                stats_root=self.cfg.paths.stats_root,
                experiment_name=exp_name,
            )

        merged.paths = OmegaConf.create(OmegaConf.to_container(self.cfg.paths, resolve=True))

        if "data" not in merged:
            merged.data = {}
        if not merged.data.get("root"):
            merged.data.root = str(Path(self.cfg.paths.data_root) / exp_name)

        if "safety" not in merged:
            merged.safety = {}
        for key, value in self.cfg.safety.items():
            if key not in merged.safety or merged.safety[key] is None:
                merged.safety[key] = value

        return merged

    def _apply_run_validation_overrides(self, exp_name: str, merged: DictConfig) -> None:
        run_validation = merged.get("run_validation")
        if run_validation is None:
            return

        max_steps_override = run_validation.get("max_steps")
        if max_steps_override is not None:
            try:
                max_steps_value = int(max_steps_override)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid experiments.{exp_name}.run_validation.max_steps={max_steps_override!r}. "
                    "Expected positive integer or null."
                ) from exc
            if max_steps_value <= 0:
                raise ValueError(
                    f"Invalid experiments.{exp_name}.run_validation.max_steps={max_steps_value}. "
                    "Value must be > 0."
                )
            merged.compute.max_steps = max_steps_value

        target_quality_override = run_validation.get("target_quality")
        if target_quality_override is not None:
            try:
                target_value = float(target_quality_override)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid experiments.{exp_name}.run_validation.target_quality={target_quality_override!r}. "
                    "Expected float or null."
                ) from exc
            if "target" not in merged:
                merged.target = {}
            if "mode" in merged.target:
                merged.target.mode = "absolute"
            merged.target.value = target_value

    def _build_run_id(
        self,
        exp_name: str,
        exp_cfg: DictConfig,
        mode: str,
        optimizer_path: str,
    ) -> tuple[str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        hash_input = (
            f"{exp_name}|{mode}|{optimizer_path}|{OmegaConf.to_yaml(exp_cfg, resolve=True)}"
        )
        short_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]
        return f"{timestamp}_{short_hash}", timestamp

    def _save_resolved_config(self, exp_name: str, run_id: str, mode: str, exp_cfg: DictConfig) -> str:
        try:
            hydra_out = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            hydra_out = Path(self.cfg.paths.runs_root) / "_hydra_fallback"

        cfg_dir = ensure_dir(hydra_out / "resolved_configs" / exp_name)
        resolved_path = cfg_dir / f"{run_id}__{mode}.yaml"

        snapshot = {
            "mode": mode,
            "seed": int(self.cfg.seed),
            "deterministic": bool(self.cfg.deterministic),
            "device": str(self.cfg.device),
            "precision": str(self.cfg.precision),
            "paths": OmegaConf.to_container(self.cfg.paths, resolve=True),
            "experiment": OmegaConf.to_container(exp_cfg, resolve=True),
        }
        yaml_text = OmegaConf.to_yaml(OmegaConf.create(snapshot), resolve=True)
        save_yaml_text(resolved_path, yaml_text)
        return str(resolved_path)

    def _execute_single(
        self,
        exp_name: str,
        exp_cfg: DictConfig,
        optimizer_factory: OptimizerBuilder,
        optimizer_path: str,
        optimizer_name: str,
        run_id: str,
        timestamp: str,
        resolved_cfg_path: str,
        mode: str,
    ) -> dict[str, Any]:
        result_core: dict[str, Any]

        try:
            experiment = create_experiment(exp_name)
            datamodule = experiment.build_datamodule(exp_cfg)
            model = experiment.build_model(exp_cfg)
            result_core = experiment.train(exp_cfg, model, datamodule, optimizer_factory)
        except OptionalDependencyError as dep_error:
            if mode == "smoke":
                result_core = {
                    "status": "skipped",
                    "error_msg": str(dep_error),
                    "final_metrics": {"train_loss": float("nan")},
                    "best_metrics": {"train_loss": float("nan")},
                    "objective_name": "train_loss",
                    "objective_direction": "min",
                    "objective_last": float("nan"),
                    "objective_best": float("nan"),
                    "objective_best_step": 0,
                    "first_step_at_or_below_baseline": None,
                    "steps": 0,
                    "wall_time_sec": 0.0,
                    "samples_or_tokens_seen": 0,
                    "converged": False,
                    "steps_to_target": None,
                    "extra_metrics": {},
                }
            else:
                raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Experiment '%s' failed: %s", exp_name, exc)
            result_core = {
                "status": "failed",
                "error_msg": str(exc),
                "final_metrics": {"train_loss": float("nan")},
                "best_metrics": {"train_loss": float("nan")},
                "objective_name": "train_loss",
                "objective_direction": "min",
                "objective_last": float("nan"),
                "objective_best": float("nan"),
                "objective_best_step": 0,
                "first_step_at_or_below_baseline": None,
                "steps": 0,
                "wall_time_sec": 0.0,
                "samples_or_tokens_seen": 0,
                "converged": False,
                "steps_to_target": None,
                "extra_metrics": {},
            }

        final_metrics = result_core.get("final_metrics", {})
        best_metrics = result_core.get("best_metrics", final_metrics)
        metric_name = str(exp_cfg.primary_metric.name)

        final_quality = _safe_float(final_metrics.get(metric_name), default=0.0)
        best_quality = _safe_float(best_metrics.get(metric_name), default=final_quality)

        run_result = RunResult(
            run_id=run_id,
            timestamp=timestamp,
            experiment_name=exp_name,
            optimizer_path=optimizer_path,
            optimizer_name=optimizer_name,
            final_quality=final_quality,
            target_quality=_safe_float(_get_target_quality(exp_cfg.target), default=0.0),
            converged=bool(result_core.get("converged", False)),
            steps=int(result_core.get("steps", 0)),
            wall_time_sec=float(result_core.get("wall_time_sec", 0.0)),
            best_quality=best_quality,
            objective_name=str(result_core.get("objective_name", "train_loss")),
            objective_direction=str(result_core.get("objective_direction", "min")),
            objective_last=float(result_core.get("objective_last", float("nan"))),
            objective_best=float(result_core.get("objective_best", float("nan"))),
            objective_best_step=int(result_core.get("objective_best_step", 0)),
            first_step_at_or_below_baseline=result_core.get("first_step_at_or_below_baseline"),
            seed=int(self.cfg.seed),
            device=str(exp_cfg.compute.device),
            precision=str(exp_cfg.compute.precision),
            resolved_config_path=resolved_cfg_path,
            extra_metrics=_to_plain(result_core.get("extra_metrics", {})),
            safety_flags=_to_plain(exp_cfg.safety),
            final_metrics=_to_plain(final_metrics),
            best_metrics=_to_plain(best_metrics),
            samples_or_tokens_seen=int(result_core.get("samples_or_tokens_seen", 0)),
            steps_to_target=result_core.get("steps_to_target"),
            status=str(result_core.get("status", "ok")),
            error_msg=result_core.get("error_msg"),
            smoke=mode == "smoke",
        )
        payload = run_result.to_dict()
        validate_run_result_dict(payload)
        return payload

    def _persist_result(
        self,
        exp_name: str,
        run_id: str,
        mode: str,
        optimizer_path: str,
        result_payload: dict[str, Any],
    ) -> Path:
        if mode == "run":
            optimizer_file = Path(optimizer_path).expanduser().resolve()
            target_dir = ensure_dir(optimizer_file.parent / self.cfg.paths.optimizer_results_dirname)
            result_path = target_dir / f"{run_id}__{exp_name}.json"
            save_json(result_path, result_payload)

            copy_dir = ensure_dir(Path(self.cfg.paths.runs_root) / exp_name)
            copy_path = copy_dir / f"{run_id}.json"
            save_json(copy_path, result_payload)
            return result_path

        if mode == "stats":
            result_path = baseline_path(self.cfg.paths.stats_root, exp_name)
            ensure_dir(result_path.parent)
            save_json(result_path, result_payload)
            return result_path

        smoke_dir = ensure_dir(Path(self.cfg.paths.runs_root) / exp_name)
        result_path = smoke_dir / f"{run_id}__smoke.json"
        save_json(result_path, result_payload)
        return result_path
