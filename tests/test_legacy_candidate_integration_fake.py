"""LEGACY candidate-first integration coverage with the fake evaluator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.evolve.legacy_orchestrator import LegacyCandidateOrchestrator


def test_legacy_candidate_pipeline_fake(tmp_path: Path) -> None:
    pop_root = tmp_path / "populations"
    stats_root = tmp_path / "stats"
    for exp_name, objective_last, steps in (("exp_a", 1.0, 10), ("exp_b", 1.0, 10)):
        exp_dir = stats_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "baseline.json").write_text(
            json.dumps(
                {
                    "objective_name": "train_loss",
                    "objective_direction": "min",
                    "objective_last": objective_last,
                    "steps": steps,
                }
            ),
            encoding="utf-8",
        )

    cfg = OmegaConf.create(
        {
            "mode": "evolve",
            "seed": 7,
            "deterministic": False,
            "precision": "fp32",
            "num_workers": 0,
            "safety": {
                "detect_nan": True,
                "abort_on_nan": True,
                "grad_clip_norm": None,
                "log_grad_norm": True,
            },
            "resources": {"num_gpus": 1, "gpu_ids": [0]},
            "paths": {
                "population_root": str(pop_root),
                "data_root": str(tmp_path / "data"),
                "stats_root": str(stats_root),
                "runs_root": str(tmp_path / "runs"),
                "optimizer_guesses_root": str(tmp_path / "optimizer_guesses"),
                "optimizer_results_dirname": "results",
            },
            "experiments": {
                "exp_a": {
                    "enabled": True,
                    "primary_metric": {"direction": "min"},
                    "normalization": {"eps": 1.0e-8},
                },
                "exp_b": {
                    "enabled": True,
                    "primary_metric": {"direction": "min"},
                    "normalization": {"eps": 1.0e-8},
                },
            },
            "evolver": {
                "enabled": True,
                "generation": 0,
                "num_candidates": 2,
                "eval_experiments": ["exp_a", "exp_b"],
                "eval_mode": "smoke",
                "timeout_sec_per_eval": 60,
                "max_retries_per_eval": 0,
                "fail_fast": False,
                "resume": False,
                "force": False,
                "max_proposal_jobs": 2,
                "max_evaluation_jobs": None,
                "max_generation_attempts": 1,
                "eval_entrypoint_module": "tests.fixtures.fake_eval",
                "allocation": {
                    "enabled": True,
                    "method": "neyman",
                    "history_window": 10,
                    "sample_size": 1,
                    "min_history_for_variance": 2,
                    "std_floor": 1.0e-6,
                    "fallback": "uniform",
                    "costs": {"exp_a": 1.0, "exp_b": 1.0},
                },
                "llm": {
                    "provider": "mock",
                    "model": "mock-model",
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "reasoning_effort": None,
                    "seed": 123,
                    "fallback_to_chat_completions": True,
                },
            },
        }
    )

    summary = asyncio.run(LegacyCandidateOrchestrator(cfg).run())

    assert summary["requested_candidates"] == 2
    assert summary["completed_candidates"] >= 2

    gen_dir = pop_root / "gen_0000"
    candidate_dirs = sorted(gen_dir.glob("cand_*"))
    assert len(candidate_dirs) >= 2

    for cand_dir in candidate_dirs[:2]:
        summary_payload = json.loads((cand_dir / "summary.json").read_text(encoding="utf-8"))
        selected = summary_payload["selected_experiments"]
        assert selected
        assert set(selected).issubset({"exp_a", "exp_b"})

        allocation = summary_payload["allocation"]
        assert set(allocation["weights"].keys()) == {"exp_a", "exp_b"}
        assert set(allocation["inclusion_prob"].keys()) == {"exp_a", "exp_b"}
