"""Generic experiment runner for run/smoke/stats modes."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.evolve.storage import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    """Coordinates generic experiment execution via organism directories."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def _enabled_experiments(self) -> list[tuple[str, DictConfig]]:
        enabled: list[tuple[str, DictConfig]] = []
        for exp_name, exp_cfg in self.cfg.get("experiments", {}).items():
            if bool(exp_cfg.get("enabled", False)):
                enabled.append((str(exp_name), exp_cfg))
        if not enabled:
            raise ValueError("No enabled experiments found in cfg.experiments.*.enabled")
        return enabled

    def _prepare_experiment_cfg(self, exp_name: str, exp_cfg: DictConfig, mode: str) -> DictConfig:
        merged = OmegaConf.create(OmegaConf.to_container(exp_cfg, resolve=False))

        merged.seed = int(self.cfg.seed)
        merged.deterministic = bool(self.cfg.deterministic)
        merged.runtime = {
            "mode": mode,
            "smoke": mode == "smoke",
            "organism_dir": self.cfg.get("organism_dir"),
        }

        merged.compute.device = str(self.cfg.device)
        merged.compute.precision = str(self.cfg.precision)
        merged.compute.num_workers = int(self.cfg.num_workers)

        if mode == "smoke":
            merged.compute.max_steps = int(merged.compute.smoke_steps)
        if mode == "run":
            self._apply_run_validation_overrides(exp_name=exp_name, merged=merged)

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
            max_steps_value = int(max_steps_override)
            if max_steps_value <= 0:
                raise ValueError(
                    f"Invalid experiments.{exp_name}.run_validation.max_steps={max_steps_value}. Value must be > 0."
                )
            merged.compute.max_steps = max_steps_value

        target_quality_override = run_validation.get("target_quality")
        if target_quality_override is not None:
            target_value = float(target_quality_override)
            if "target" not in merged:
                merged.target = {}
            merged.target.value = target_value

    def _build_run_id(self, exp_name: str, exp_cfg: DictConfig, mode: str) -> tuple[str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        hash_input = f"{exp_name}|{mode}|{self.cfg.get('organism_dir')}|{OmegaConf.to_yaml(exp_cfg, resolve=True)}"
        short_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]
        return f"{timestamp}_{short_hash}", timestamp

    def _save_resolved_config(self, exp_name: str, run_id: str, exp_cfg: DictConfig) -> Path:
        out_dir = ensure_dir(Path(self.cfg.paths.runs_root) / exp_name / run_id)
        out_path = out_dir / "resolved_config.yaml"
        out_path.write_text(OmegaConf.to_yaml(exp_cfg, resolve=True), encoding="utf-8")
        return out_path

    def run(self) -> list[dict[str, Any]]:
        mode = str(self.cfg.mode)
        if mode not in {"run", "stats", "smoke"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use: run, stats, smoke.")
        if mode == "run" and not self.cfg.get("organism_dir"):
            raise ValueError("mode=run requires organism_dir=/path/to/organism")

        results: list[dict[str, Any]] = []
        for exp_name, raw_cfg in self._enabled_experiments():
            exp_cfg = self._prepare_experiment_cfg(exp_name, raw_cfg, mode)
            experiment = instantiate(exp_cfg, _recursive_=False)
            if not hasattr(experiment, "evaluate_organism"):
                raise AttributeError(
                    f"Experiment '{exp_name}' must define evaluate_organism(organism_dir, cfg)."
                )

            try:
                report = experiment.evaluate_organism(self.cfg.get("organism_dir"), exp_cfg)
                if not isinstance(report, dict):
                    raise TypeError(f"Experiment '{exp_name}' returned non-dict payload: {type(report).__name__}")
                if "score" not in report:
                    raise ValueError(f"Experiment '{exp_name}' report is missing required field 'score'.")
            except Exception as exc:  # noqa: BLE001
                if mode == "smoke" and exc.__class__.__name__ == "OptionalDependencyError":
                    report = {
                        "status": "skipped",
                        "score": None,
                        "error_msg": str(exc),
                    }
                else:
                    raise

            run_id, timestamp = self._build_run_id(exp_name, exp_cfg, mode)
            out_dir = ensure_dir(Path(self.cfg.paths.runs_root) / exp_name / run_id)
            report_path = out_dir / "report.json"
            resolved_cfg_path = self._save_resolved_config(exp_name, run_id, exp_cfg)
            write_json(report_path, report)

            payload = {
                "run_id": run_id,
                "timestamp": timestamp,
                "experiment_name": exp_name,
                "organism_dir": self.cfg.get("organism_dir"),
                "status": report.get("status", "ok"),
                "score": report.get("score"),
                "report_path": str(report_path),
                "resolved_config_path": str(resolved_cfg_path),
            }
            results.append(payload)
            LOGGER.info("Finished experiment=%s status=%s report=%s", exp_name, payload["status"], report_path)

        return results
