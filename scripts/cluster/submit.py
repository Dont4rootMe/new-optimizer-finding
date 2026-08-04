"""Submit the canonical DeepSeek-V4 circle-packing run through ML Space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.cluster.common import (
    BASE_IMAGE,
    INSTANCE_TYPE,
    MODEL_REVISION,
    REGION,
    SGLANG_VERSION,
    atomic_write_json,
    require_absolute_safe_path,
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

    shared_root = project_root.parent
    environment = {
        "PROJECT_ROOT": str(project_root),
        "RUN_ID": run_id,
        "RUN_DIR": str(run_dir),
        "DEEPSEEK_ENV_DIR": str(shared_root / ".inference_runtime" / f"sglang-{SGLANG_VERSION}"),
        "HF_HOME": str(shared_root / ".model_cache" / "huggingface"),
        "SGLANG_VERSION": SGLANG_VERSION,
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
    request: dict[str, Any] = {
        "base_image": base_image,
        "script": f"bash {project_root}/scripts/cluster/run_deepseek_v4_circle.sh",
        "region": region,
        "instance_type": instance_type,
        "n_workers": 1,
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
    )
    request_payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dry_run": bool(args.dry_run),
        "request": kwargs,
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
