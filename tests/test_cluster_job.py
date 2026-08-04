"""Side-effect-free contracts for the binary 8xH100 job path."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.common import build_evolution_command, build_sglang_command
from scripts.cluster.monitor import collect_progress, monitor, normalize_scheduler_status
from scripts.cluster.submit import build_job_kwargs


def test_sglang_command_is_tp8_dspark_and_hopper_safe() -> None:
    command = build_sglang_command(python="/env/bin/python", model_path="/models/revision")
    rendered = " ".join(command)
    assert "--tp 8" in rendered
    assert "--speculative-algorithm DSPARK" in rendered
    assert "--reasoning-parser deepseek-v4" in rendered
    assert "--tool-call-parser deepseekv4" in rendered
    assert "--moe-runner-backend" not in command
    assert "flashinfer_mxfp4" not in command
    assert "--speculative-draft-model-path" not in command


def test_evolution_command_is_single_backbone_300_generation_resumeable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    command = build_evolution_command(
        project_root=project,
        population_root=tmp_path / "run" / "population",
        hydra_run_dir=tmp_path / "run" / "hydra",
        config_name="config_circle_packing_shinka",
        backbone="deepseek_v4_flash_0731",
        max_generations=300,
        max_parallel_organisms=8,
    )
    assert "--seed" in command
    assert "backbone=deepseek_v4_flash_0731" in command
    assert "evolver.max_generations=300" in command
    assert "evolver.creation.max_parallel_organisms=8" in command


def test_submit_contract_uses_one_binary_worker(tmp_path: Path) -> None:
    kwargs = build_job_kwargs(
        project_root=tmp_path / "project",
        run_dir=tmp_path / "runs" / "one",
        run_id="one",
        config_name="config_circle_packing_shinka",
        backbone="deepseek_v4_flash_0731",
        max_generations=300,
        max_parallel_organisms=8,
        hydra_overrides=[],
    )
    assert kwargs["type"] == "binary"
    assert kwargs["n_workers"] == 1
    assert kwargs["detached"] is True
    assert kwargs["env_variables"]["MAX_GENERATIONS"] == "300"
    assert "queue_name" not in kwargs


def test_monitor_emits_terminal_event_only_after_both_layers_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    population = run_dir / "population"
    population.mkdir(parents=True)
    (population / "population_state.json").write_text(
        json.dumps({"current_generation": 300, "active_organisms": ["a"]}), encoding="utf-8"
    )
    (population / "llm_usage.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    assert normalize_scheduler_status("Job status=Completed") == "completed"
    snapshot = collect_progress(run_dir, scheduler_status="Job status=Completed")
    assert snapshot["current_generation"] == 300
    assert snapshot["usage_events"] == 2
    assert monitor(
        job_name="job", run_dir=run_dir, status_reader=lambda _name: "Job status=Completed",
        interval_sec=10, once=True
    ) == 0
    event = json.loads((run_dir / "completion_event.json").read_text(encoding="utf-8"))
    assert event["success"] is True
