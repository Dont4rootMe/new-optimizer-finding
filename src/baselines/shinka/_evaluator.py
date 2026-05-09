"""Shared helpers for ShinkaEvolve evaluator adapters."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT_ENV = "SHINKA_BASELINE_PROJECT_ROOT"
CONFIG_NAME_ENV = "SHINKA_BASELINE_CONFIG"


def _project_root() -> Path:
    value = os.environ.get(PROJECT_ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"Environment variable {PROJECT_ROOT_ENV} is not set; the ShinkaEvolve "
            "evaluator adapter expects it to be exported by run_shinka_baseline.sh."
        )
    return Path(value).expanduser().resolve()


def _config_name() -> str:
    value = os.environ.get(CONFIG_NAME_ENV)
    if not value:
        raise RuntimeError(
            f"Environment variable {CONFIG_NAME_ENV} is not set; the ShinkaEvolve "
            "evaluator adapter expects it to be exported by run_shinka_baseline.sh."
        )
    return value


def _compose_cfg() -> DictConfig:
    root = _project_root()
    with initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
        cfg = compose(config_name=_config_name())
    return cfg


def _select_experiment_cfg(cfg: DictConfig) -> DictConfig:
    experiments = cfg.get("experiments")
    if experiments is None:
        raise RuntimeError("Composed config has no `experiments` block.")
    keys = list(experiments.keys())
    if len(keys) != 1:
        raise RuntimeError(
            f"ShinkaEvolve baseline expects exactly one experiment under `experiments:`, "
            f"got {len(keys)}: {keys}."
        )
    return experiments[keys[0]]


def _stage_organism_dir(program_path: Path) -> tuple[Path, Path | None]:
    """Place program_path as `<organism_dir>/implementation.py`.

    Returns (organism_dir, cleanup_dir). cleanup_dir is non-None only when we
    created a tmp dir, so the caller can remove it on exit.
    """

    program_path = program_path.expanduser().resolve()
    if program_path.name == "implementation.py":
        return program_path.parent, None
    tmp_dir = Path(tempfile.mkdtemp(prefix="shinka_eval_"))
    shutil.copy2(program_path, tmp_dir / "implementation.py")
    return tmp_dir, tmp_dir


def _normalize_combined_score(result: dict[str, Any]) -> float:
    score = result.get("score")
    if score is None:
        raise RuntimeError(f"Experiment evaluator returned no 'score' field: keys={list(result)}")
    return float(score)


def evaluate_with_host_experiment(program_path: str, results_dir: str) -> dict[str, Any]:
    """Run the host-project experiment evaluator on `program_path` and return shinka format."""

    organism_dir, cleanup_dir = _stage_organism_dir(Path(program_path))
    try:
        cfg = _compose_cfg()
        experiment_cfg = _select_experiment_cfg(cfg)
        experiment = instantiate(experiment_cfg, _recursive_=False)
        result = experiment.evaluate_organism(organism_dir=str(organism_dir), cfg=experiment_cfg)
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    public = {
        "objective_name": result.get("objective_name"),
        "objective_direction": result.get("objective_direction"),
        "objective_last": result.get("objective_last"),
        "status": result.get("status"),
    }
    if "num_cases" in result:
        public["num_cases"] = result["num_cases"]
        public["mean_absolute_score"] = result.get("mean_absolute_score")
        public["total_absolute_score"] = result.get("total_absolute_score")
    if "num_circles" in result:
        public["num_circles"] = result["num_circles"]
        public["reported_sum_of_radii"] = result.get("reported_sum_of_radii")

    return {
        "combined_score": _normalize_combined_score(result),
        "public": public,
        "private": {
            "results_dir": str(Path(results_dir).expanduser().resolve()),
        },
    }


def dump_cfg_for_debug(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg)
