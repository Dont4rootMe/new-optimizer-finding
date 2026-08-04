"""Side-effect-free contracts for the binary 8xH100 job path."""

from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess

from scripts.cluster.common import (
    SGLANG_RUNTIME_ID,
    build_evolution_command,
    build_git_bootstrap_command,
    build_sglang_command,
)
from scripts.cluster.sglang_runtime import cuda126_requirements
from scripts.cluster.monitor import collect_progress, monitor, normalize_scheduler_status
from scripts.cluster.submit import build_job_kwargs
from scripts.cluster.transfer import connector_path, wait_for_transfer

ROOT = Path(__file__).resolve().parents[1]


def test_sglang_command_is_tp8_dspark_and_hopper_safe() -> None:
    command = build_sglang_command(python="/env/bin/python", model_path="/models/revision")
    rendered = " ".join(command)
    assert "--tp 8" in rendered
    assert "--speculative-algorithm DSPARK" in rendered
    assert "--reasoning-parser deepseek-v4" in rendered
    assert "--tool-call-parser deepseekv4" in rendered
    assert "--moe-runner-backend marlin" in rendered
    assert "--cuda-graph-max-bs-decode 8" in rendered
    assert "--cuda-graph-max-bs" not in command
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
    assert kwargs["processes_per_worker"] == 1
    assert kwargs["detached"] is True
    assert "checkpoint_dir" not in kwargs
    assert kwargs["env_variables"]["MAX_GENERATIONS"] == "300"
    assert kwargs["env_variables"]["SGLANG_CUDA_VARIANT"] == "cu126"
    assert kwargs["env_variables"]["DEEPSEEK_ENV_DIR"].endswith(SGLANG_RUNTIME_ID)
    assert "queue_name" not in kwargs


def test_sglang_runtime_rewrites_cuda13_metadata_to_audited_cuda126_wheels() -> None:
    requirements = cuda126_requirements(
        [
            "cuda-python>=13.0",
            'flashinfer_python[cu13]==0.6.14',
            "humming-kernels[cu13]==0.1.10",
            "nvidia-cutlass-dsl[cu13]==4.6.0",
            "sglang-kernel==0.4.5",
            "sgl-deep-gemm==0.1.4.post1",
            "torch==2.11.0",
            'ray[default]>=2.55.1; extra == "ray"',
        ],
        machine="x86_64",
    )
    rendered = "\n".join(requirements)
    assert "cuda-python>=12,<13" in requirements
    assert "flashinfer_python[cu12]==0.6.14" in requirements
    assert "humming-kernels==0.1.10" in requirements
    assert "nvidia-cutlass-dsl==4.6.0" in requirements
    assert "sglang_kernel-0.4.5+cu129" in rendered
    assert "sgl_deep_gemm-0.1.4.post1+cu129" in rendered
    assert "cu13" not in rendered
    assert "ray" not in rendered


def test_submit_contract_can_target_isolated_regional_nfs(tmp_path: Path) -> None:
    job_root = Path("/home/jovyan/evolutionloop-deepseek-v4")
    commit = "a" * 40
    bootstrap = build_git_bootstrap_command(
        repository_url="https://github.com/example/project.git", commit=commit, job_root=job_root
    )
    kwargs = build_job_kwargs(
        project_root=tmp_path / "control-project",
        run_dir=tmp_path / "control-runs" / "one",
        run_id="one",
        config_name="config_circle_packing_shinka",
        backbone="deepseek_v4_flash_0731",
        max_generations=300,
        max_parallel_organisms=8,
        hydra_overrides=[],
        job_project_root=job_root / "source" / commit,
        job_run_dir=job_root / "runs" / "one",
        job_root=job_root,
        source_commit=commit,
        job_script=bootstrap,
    )
    assert kwargs["script"].startswith("python3 -c ")
    assert kwargs["env_variables"]["SOURCE_COMMIT"] == commit
    assert kwargs["env_variables"]["RUN_DIR"] == str(job_root / "runs" / "one")
    environment = os.environ.copy()
    environment["OMPI_COMM_WORLD_RANK"] = "7"
    completed = subprocess.run(
        ["bash", "-c", bootstrap], env=environment, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0
    assert "owned by rank 0" in completed.stdout


def test_binary_entrypoint_nonzero_rank_exits_before_shared_state_access() -> None:
    environment = os.environ.copy()
    environment.update({"OMPI_COMM_WORLD_RANK": "7", "PROJECT_ROOT": "/does/not/exist"})
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "cluster" / "run_deepseek_v4_circle.sh")],
        cwd=str(ROOT),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "owned by rank 0" in completed.stdout


def test_cuda_driver_environment_removes_only_compat_and_empty_paths() -> None:
    helper = ROOT / "scripts" / "cluster" / "cuda_driver_env.sh"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (
        "/opt/hpcx/ompi/lib:/usr/local/cuda-12.6/compat:"
        "/usr/local/cuda-12.6/lib64::/vendor/compat/stubs:/usr/local/nvidia/lib64"
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; sanitize_cuda_driver_path; sanitize_cuda_driver_path; printf "%s" "$LD_LIBRARY_PATH"',
            "bash",
            str(helper),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == (
        "/opt/hpcx/ompi/lib:/usr/local/cuda-12.6/lib64:/usr/local/nvidia/lib64"
    )
    assert completed.stderr.count("using scheduler-mounted libcuda") == 1


def test_data_transfer_paths_and_terminal_logs() -> None:
    assert connector_path(Path("/home/jovyan/team/repo.tar.gz")) == "/team/repo.tar.gz"
    try:
        connector_path(Path("/tmp/outside"))
    except ValueError as exc:
        assert "below /home/jovyan" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("outside paths must be rejected")

    calls = iter([[], [{"status": "completed", "path": "source.tar.gz"}]])
    assert wait_for_transfer(
        "transfer", log_reader=lambda _transfer_id: next(calls), timeout_sec=1, interval_sec=0
    )[0]["status"] == "completed"


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


def test_monitor_accepts_completed_staged_job_before_artifact_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "control-run"
    run_dir.mkdir()
    (run_dir / "submission.json").write_text(
        json.dumps(
            {
                "artifact_namespace": "regional_nfs",
                "request": {"env_variables": {"RUN_DIR": "/home/jovyan/regional/run"}},
            }
        ),
        encoding="utf-8",
    )
    assert monitor(
        job_name="job", run_dir=run_dir, status_reader=lambda _name: "Job status=Completed",
        interval_sec=10, once=True,
    ) == 0
    event = json.loads((run_dir / "completion_event.json").read_text(encoding="utf-8"))
    assert event["success"] is True
    assert event["regional_run_dir"] == "/home/jovyan/regional/run"
