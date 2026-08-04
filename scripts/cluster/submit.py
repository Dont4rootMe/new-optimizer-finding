"""Submit the canonical DeepSeek-V4 circle-packing run through ML Space."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.cluster.common import (
    BASE_IMAGE,
    DEEPGEMM_NVCC_VERSION,
    DEEPGEMM_TOOLCHAIN_ID,
    INSTANCE_TYPE,
    MODEL_REVISION,
    REGION,
    REGIONAL_JOB_ROOT,
    REPOSITORY_URL,
    SGLANG_CUDA_VARIANT,
    SGLANG_RUNTIME_ID,
    SGLANG_VERSION,
    atomic_write_json,
    build_git_bootstrap_command,
    require_absolute_safe_path,
    require_regional_job_path,
    utc_now,
)


def build_job_kwargs(
    *,
    project_root: Path,
    run_dir: Path,
    run_id: str,
    config_name: str,
    backbone: str,
    max_generations: int,
    max_parallel_organisms: int,
    hydra_overrides: list[str],
    job_project_root: Path | None = None,
    job_run_dir: Path | None = None,
    job_root: Path | None = None,
    source_commit: str | None = None,
    job_script: str | None = None,
    base_image: str = BASE_IMAGE,
    instance_type: str = INSTANCE_TYPE,
    region: str = REGION,
    queue_name: str | None = None,
    priority_class: str | None = None,
) -> dict[str, Any]:
    """Build one single-coordinator binary-job request.

    Queue and priority are absent by default because the older A100 policy has
    not been proven for SR008. Callers can opt in explicitly after checking.
    """

    runtime_project_root = job_project_root or project_root
    runtime_run_dir = job_run_dir or run_dir
    runtime_entrypoint = runtime_project_root / "scripts" / "cluster" / "run_deepseek_v4_circle.sh"
    runtime_shared_root = job_root or runtime_project_root.parent
    environment = {
        "PROJECT_ROOT": str(runtime_project_root),
        "RUN_ID": run_id,
        "RUN_DIR": str(runtime_run_dir),
        "JOB_ROOT": str(runtime_shared_root),
        "DEEPSEEK_ENV_DIR": str(runtime_shared_root / "runtime" / SGLANG_RUNTIME_ID),
        "DEEPGEMM_CUDA_TOOLCHAIN_DIR": str(
            runtime_shared_root / "toolchains" / DEEPGEMM_TOOLCHAIN_ID
        ),
        "DEEPGEMM_NVCC_VERSION": DEEPGEMM_NVCC_VERSION,
        "SGLANG_DG_CACHE_DIR": str(
            runtime_shared_root / "kernel_cache" / f"deep_gemm-sm90-{DEEPGEMM_TOOLCHAIN_ID}"
        ),
        "HF_HOME": str(runtime_shared_root / "model_cache" / "huggingface"),
        "SGLANG_VERSION": SGLANG_VERSION,
        "SGLANG_CUDA_VARIANT": SGLANG_CUDA_VARIANT,
        "MODEL_REVISION": MODEL_REVISION,
        "CONFIG_NAME": config_name,
        "BACKBONE": backbone,
        "MAX_GENERATIONS": str(max_generations),
        "MAX_PARALLEL_ORGANISMS": str(max_parallel_organisms),
        "HYDRA_OVERRIDES_JSON": json.dumps(hydra_overrides, separators=(",", ":")),
        "COMET_ENABLED": "false",
        "PYTHONNOUSERSITE": "1",
        "PIP_USER": "no",
    }
    if source_commit is not None:
        environment["SOURCE_COMMIT"] = str(source_commit)
    request: dict[str, Any] = {
        "base_image": base_image,
        "script": job_script or f"bash {runtime_entrypoint}",
        "region": region,
        "instance_type": instance_type,
        "n_workers": 1,
        # One coordinator owns the TP=8 inference server.  Without this ML
        # Space defaults to one MPI process per GPU even for a binary job.
        "processes_per_worker": 1,
        "type": "binary",
        "job_desc": f"echimbulatov | {run_id} #ID0137 #rnd",
        "env_variables": environment,
        "internet": True,
        "shm_size_class": "large",
        "detached": True,
        "preflight_check": True,
    }
    if queue_name:
        request["queue_name"] = queue_name
    if priority_class:
        request["priority_class"] = priority_class
    return request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="absolute NFS path to this branch clone")
    parser.add_argument("--run-base", help="absolute directory for durable run folders")
    parser.add_argument("--run-id", default="deepseek-v4-flash-0731-circle-300")
    parser.add_argument("--config-name", default="config_circle_packing_shinka")
    parser.add_argument("--backbone", default="deepseek_v4_flash_0731")
    parser.add_argument("--max-generations", type=int, default=300)
    parser.add_argument("--max-parallel-organisms", type=int, default=8)
    parser.add_argument("--override", action="append", default=[], dest="hydra_overrides")
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--queue-name")
    parser.add_argument("--priority-class")
    parser.add_argument("--job-root", default=str(REGIONAL_JOB_ROOT))
    parser.add_argument("--repository-url", default=REPOSITORY_URL)
    parser.add_argument("--source-commit", help="full commit id; defaults to project-root HEAD")
    parser.add_argument(
        "--direct-shared-path", action="store_true",
        help="skip the SR008 git bootstrap only when project-root is proven job-visible",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_generations < 1 or args.max_parallel_organisms < 1:
        raise SystemExit("generation and concurrency values must be positive")
    project_root = require_absolute_safe_path(args.project_root, label="project root")
    run_base = require_absolute_safe_path(
        args.run_base or str(project_root.parent / "optimizer_cluster_runs"), label="run base"
    )
    run_dir = run_base / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    staged_kwargs: dict[str, Any] = {}
    source: dict[str, Any] | None = None
    if not args.direct_shared_path:
        worktree_status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=str(project_root), check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if worktree_status:
            raise SystemExit(
                "refusing a git-bootstrap submission from a dirty worktree; commit or isolate changes first"
            )
        commit = args.source_commit
        if not commit:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        job_root = require_regional_job_path(args.job_root, label="regional job root")
        job_project_root = job_root / "source" / commit
        job_run_dir = job_root / "runs" / args.run_id
        job_script = build_git_bootstrap_command(
            repository_url=args.repository_url, commit=commit, job_root=job_root,
        )
        staged_kwargs = {
            "job_project_root": job_project_root,
            "job_run_dir": job_run_dir,
            "job_root": job_root,
            "source_commit": commit,
            "job_script": job_script,
        }
        source = {
            "transport": "git_https",
            "repository_url": args.repository_url,
            "commit": commit,
            "regional_project_root": str(job_project_root),
            "regional_run_dir": str(job_run_dir),
        }
    kwargs = build_job_kwargs(
        project_root=project_root,
        run_dir=run_dir,
        run_id=args.run_id,
        config_name=args.config_name,
        backbone=args.backbone,
        max_generations=args.max_generations,
        max_parallel_organisms=args.max_parallel_organisms,
        hydra_overrides=args.hydra_overrides,
        base_image=args.base_image,
        instance_type=args.instance_type,
        region=args.region,
        queue_name=args.queue_name,
        priority_class=args.priority_class,
        **staged_kwargs,
    )
    request_payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "request": kwargs,
        "artifact_namespace": "shared" if args.direct_shared_path else "regional_nfs",
        "source": source,
    }
    atomic_write_json(run_dir / "submission_request.json", request_payload)
    if args.dry_run:
        print(json.dumps(request_payload, indent=2, sort_keys=True))
        return

    import client_lib  # type: ignore[import-not-found]  # cluster-only dependency

    job = client_lib.Job(**kwargs)
    result = job.submit()
    job_name = getattr(job, "job_name", None)
    if not job_name and isinstance(result, str) and result.startswith("lm-"):
        job_name = result.strip()
    if not job_name:
        raise RuntimeError(f"scheduler accepted submission but exposed no job_name: {result!r}")
    submission = {
        **request_payload,
        "dry_run": False,
        "submitted_at": utc_now(),
        "job_name": str(job_name),
        "submit_result": str(result),
    }
    atomic_write_json(run_dir / "submission.json", submission)
    print(json.dumps({"job_name": str(job_name), "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
