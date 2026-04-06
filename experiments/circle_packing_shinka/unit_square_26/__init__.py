"""Unit-square circle-packing experiment with 26 circles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from experiments.circle_packing_shinka._runtime.candidate_loader import load_run_packing
from experiments.circle_packing_shinka._runtime.validation import (
    coerce_run_output,
    format_centers_string,
    save_extra_artifact,
    validate_circle_packing,
)


class UnitSquare26CirclePackingExperiment:
    """Evaluate `run_packing()` candidates for 26 circles in the unit square."""

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate_organism(self, organism_dir: str | None, cfg: DictConfig) -> dict[str, Any]:
        runtime_cfg = cfg.get("runtime")
        resolved_organism_dir = organism_dir
        if not resolved_organism_dir and runtime_cfg is not None:
            resolved_organism_dir = runtime_cfg.get("organism_dir")

        if not resolved_organism_dir:
            raise ValueError("Circle-packing evaluation requires organism_dir pointing to implementation.py.")

        organism_path = Path(str(resolved_organism_dir)).expanduser().resolve()
        implementation_path = str(organism_path / "implementation.py")
        run_packing, module_path = load_run_packing(implementation_path)
        centers, radii, reported_sum = coerce_run_output(run_packing())

        validate_circle_packing(
            centers=centers,
            radii=radii,
            reported_sum=reported_sum,
            num_circles=int(cfg.validation.num_circles),
            square_size=float(cfg.validation.square_size),
            atol=float(cfg.validation.validation_atol),
        )

        extra_path = save_extra_artifact(
            organism_dir=str(organism_path),
            experiment_name=str(cfg.name),
            centers=centers,
            radii=radii,
            reported_sum=reported_sum,
        )

        return {
            "status": "ok",
            "score": float(reported_sum),
            "objective_name": "sum_of_radii",
            "objective_direction": "max",
            "objective_last": float(reported_sum),
            "num_circles": int(centers.shape[0]),
            "reported_sum_of_radii": float(reported_sum),
            "centers_str": format_centers_string(centers),
            "candidate_module_path": str(module_path),
            "extra_npz_path": extra_path,
        }
