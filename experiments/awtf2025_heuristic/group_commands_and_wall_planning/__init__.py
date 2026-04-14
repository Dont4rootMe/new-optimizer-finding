"""AtCoder AWTF 2025 heuristic experiment: Group Commands and Wall Planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from experiments.awtf2025_heuristic._runtime.candidate_loader import load_solve_case
from experiments.awtf2025_heuristic._runtime.validation import (
    evaluate_case,
    load_case_paths,
    save_extra_artifact,
)


class GroupCommandsAndWallPlanningExperiment:
    """Evaluate `solve_case(input_text) -> output_text` candidates on a fixed awtf2025 corpus."""

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate_organism(self, organism_dir: str | None, cfg: DictConfig) -> dict[str, Any]:
        runtime_cfg = cfg.get("runtime")
        resolved_organism_dir = organism_dir
        if not resolved_organism_dir and runtime_cfg is not None:
            resolved_organism_dir = runtime_cfg.get("organism_dir")

        if not resolved_organism_dir:
            raise ValueError("awtf2025 evaluation requires organism_dir pointing to implementation.py.")

        organism_path = Path(str(resolved_organism_dir)).expanduser().resolve()
        implementation_path = str(organism_path / "implementation.py")
        solve_case, module_path = load_solve_case(implementation_path)

        mode = str(runtime_cfg.get("mode", "full")) if runtime_cfg is not None else "full"
        validation_cfg = cfg.validation
        case_ids = (
            [int(case_id) for case_id in validation_cfg.smoke_case_ids]
            if mode == "smoke"
            else [int(case_id) for case_id in validation_cfg.full_case_ids]
        )
        case_paths = load_case_paths(str(validation_cfg.corpus_dir), case_ids)

        aggregate = str(validation_cfg.get("aggregate", "mean")).lower()
        if aggregate != "mean":
            raise ValueError(f"Unsupported aggregate strategy '{aggregate}'. Only 'mean' is supported.")

        per_case_results = [
            evaluate_case(
                case_name=case_name,
                case_path=case_path,
                solve_case=solve_case,
                per_case_soft_time_limit_sec=float(validation_cfg.get("per_case_soft_time_limit_sec", 0.0) or 0.0),
            )
            for case_name, case_path in case_paths
        ]
        if not per_case_results:
            raise ValueError("awtf2025 evaluation requires at least one configured case.")

        total_absolute_score = int(sum(int(entry["absolute_score"]) for entry in per_case_results))
        mean_absolute_score = float(total_absolute_score) / float(len(per_case_results))
        extra_payload = {
            "experiment_name": str(cfg.name),
            "aggregate": aggregate,
            "total_absolute_score": total_absolute_score,
            "mean_absolute_score": mean_absolute_score,
            "num_cases": len(per_case_results),
            "cases": per_case_results,
        }
        extra_path = save_extra_artifact(
            organism_dir=str(organism_path),
            experiment_name=str(cfg.name),
            payload=extra_payload,
        )

        return {
            "status": "ok",
            "score": -float(mean_absolute_score),
            "objective_name": "mean_absolute_score",
            "objective_direction": "min",
            "objective_last": float(mean_absolute_score),
            "num_cases": int(len(per_case_results)),
            "total_absolute_score": int(total_absolute_score),
            "mean_absolute_score": float(mean_absolute_score),
            "case_scores": {entry["case_id"]: int(entry["absolute_score"]) for entry in per_case_results},
            "candidate_module_path": str(module_path),
            "extra_json_path": extra_path,
        }
