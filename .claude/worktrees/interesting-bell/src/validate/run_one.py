"""Single-experiment evaluator used by evolve subprocess workers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from valopt.registry import create_experiment
from valopt.utils.import_utils import load_optimizer_builder
from valopt.utils.io import save_json
from valopt.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _prepare_experiment_cfg(cfg: DictConfig, experiment_name: str, mode: str, device: str, precision: str, seed: int) -> DictConfig:
    exp_cfg = OmegaConf.create(OmegaConf.to_container(cfg.experiments[experiment_name], resolve=False))

    exp_cfg.seed = int(seed)
    exp_cfg.deterministic = bool(cfg.deterministic)
    exp_cfg.runtime = {"mode": mode, "smoke": mode == "smoke"}

    exp_cfg.compute.device = str(device)
    exp_cfg.compute.precision = str(precision)
    exp_cfg.compute.num_workers = int(cfg.num_workers)

    if mode == "smoke":
        exp_cfg.compute.max_steps = int(exp_cfg.compute.smoke_steps)

    if "data" not in exp_cfg:
        exp_cfg.data = {}
    if not exp_cfg.data.get("root"):
        exp_cfg.data.root = str(Path(cfg.paths.data_root) / experiment_name)

    exp_cfg.paths = OmegaConf.create(OmegaConf.to_container(cfg.paths, resolve=True))

    if "safety" not in exp_cfg:
        exp_cfg.safety = {}
    for key, value in cfg.safety.items():
        if key not in exp_cfg.safety or exp_cfg.safety[key] is None:
            exp_cfg.safety[key] = value

    return exp_cfg


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    if output != output:
        return default
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--optimizer_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--config_path", default="conf")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    out_path = Path(args.output_json).expanduser().resolve()

    try:
        conf_dir = Path(args.config_path).expanduser().resolve()
        with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
            cfg = compose(config_name="config", overrides=list(args.override))

        experiment_name = str(args.experiment)
        if experiment_name not in cfg.experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found in cfg.experiments")

        set_seed(int(args.seed), bool(cfg.deterministic))

        exp_cfg = _prepare_experiment_cfg(
            cfg=cfg,
            experiment_name=experiment_name,
            mode=str(args.mode),
            device=str(args.device),
            precision=str(args.precision),
            seed=int(args.seed),
        )

        optimizer_builder, resolved_optimizer_path, optimizer_name = load_optimizer_builder(str(args.optimizer_path))

        experiment = create_experiment(experiment_name)
        datamodule = experiment.build_datamodule(exp_cfg)
        model = experiment.build_model(exp_cfg)
        result_core = experiment.train(exp_cfg, model, datamodule, optimizer_builder)

        final_metrics = result_core.get("final_metrics", {})
        metric_name = str(exp_cfg.primary_metric.name)
        metric_direction = str(exp_cfg.primary_metric.direction)
        raw_metric = _safe_float(final_metrics.get(metric_name), default=0.0)
        assert raw_metric is not None

        final_score = raw_metric if metric_direction == "max" else -raw_metric
        status = str(result_core.get("status", "ok"))

        payload = {
            "status": status,
            "experiment_name": experiment_name,
            "optimizer_path": str(resolved_optimizer_path),
            "optimizer_name": optimizer_name,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "raw_metric": raw_metric,
            "final_score": final_score,
            "time_sec": _safe_float(result_core.get("wall_time_sec"), default=0.0),
            "steps": int(result_core.get("steps", 0)),
            "converged": bool(result_core.get("converged", False)),
            "error_msg": result_core.get("error_msg"),
            "raw_result": result_core,
        }

        save_json(out_path, payload)
        sys.exit(0 if status == "ok" else 1)

    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "final_score": None,
            "time_sec": None,
            "steps": None,
            "converged": False,
            "error_msg": str(exc),
        }
        save_json(out_path, payload)
        LOGGER.exception("run_one failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
