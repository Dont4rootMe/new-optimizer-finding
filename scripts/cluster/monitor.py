"""Persist scheduler + EvolutionLoop progress and emit a terminal event file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

from scripts.cluster.common import REGION, atomic_write_json, read_json_if_present, utc_now

TERMINAL_SUCCESS = {"completed", "succeeded", "success", "done"}
TERMINAL_FAILURE = {
    "failed",
    "error",
    "cancelled",
    "canceled",
    "deleted",
    "killed",
    "terminated",
    "stopped",
    "aborted",
}


def normalize_scheduler_status(value: object) -> str:
    text = str(value).strip()
    if "=" in text:
        text = text.rsplit("=", 1)[-1]
    return text.strip().lower().replace("job status", "").strip(" :") or "unknown"


def _count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def collect_progress(run_dir: Path, *, scheduler_status: object) -> dict[str, Any]:
    population_root = run_dir / "population"
    population_state = read_json_if_present(population_root / "population_state.json") or {}
    run_manifest = read_json_if_present(run_dir / "run_manifest.json") or {}
    submission = read_json_if_present(run_dir / "submission.json") or read_json_if_present(
        run_dir / "submission_request.json"
    ) or {}
    request = submission.get("request") if isinstance(submission.get("request"), dict) else {}
    request_environment = (
        request.get("env_variables") if isinstance(request.get("env_variables"), dict) else {}
    )
    regional_artifacts = submission.get("artifact_namespace") == "regional_nfs"
    active = population_state.get("active_organisms")
    return {
        "observed_at": utc_now(),
        "scheduler_status_raw": str(scheduler_status),
        "scheduler_status": normalize_scheduler_status(scheduler_status),
        "run_status": run_manifest.get("status", "not_started"),
        "current_generation": population_state.get("current_generation"),
        "active_organisms": len(active) if isinstance(active, list) else None,
        "inflight_seed": isinstance(population_state.get("inflight_seed"), dict),
        "inflight_generation": isinstance(population_state.get("inflight_generation"), dict),
        "usage_events": _count_nonempty_lines(population_root / "llm_usage.jsonl"),
        "run_manifest": run_manifest,
        "regional_artifacts": regional_artifacts,
        "regional_run_dir": request_environment.get("RUN_DIR"),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def monitor(
    *,
    job_name: str,
    run_dir: Path,
    status_reader: Callable[[str], object],
    interval_sec: float,
    once: bool,
) -> int:
    """Poll without streaming logs; durable files are the completion trigger."""

    while True:
        raw_status = status_reader(job_name)
        snapshot = collect_progress(run_dir, scheduler_status=raw_status)
        atomic_write_json(run_dir / "monitor_status.json", snapshot)
        append_jsonl(run_dir / "monitor_history.jsonl", snapshot)
        print(json.dumps({key: snapshot[key] for key in (
            "observed_at", "scheduler_status", "run_status", "current_generation", "usage_events"
        )}, sort_keys=True), flush=True)

        scheduler_status = snapshot["scheduler_status"]
        run_status = str(snapshot["run_status"]).lower()
        # For regional SR008 jobs the control-plane NFS cannot see the regional
        # run manifest until a post-run Data Transfer.  A scheduler Completed
        # state still means the job command returned zero; run_job only returns
        # zero after writing its completed manifest.
        success = scheduler_status in TERMINAL_SUCCESS and (
            run_status == "completed" or (snapshot["regional_artifacts"] and run_status == "not_started")
        )
        failure = scheduler_status in TERMINAL_FAILURE or run_status in {"failed", "interrupted"}
        if success or failure:
            event = {**snapshot, "event": "cluster_run_terminal", "success": success}
            atomic_write_json(run_dir / "completion_event.json", event)
            return 0 if success else 1
        if once:
            return 3
        time.sleep(max(10.0, interval_sec))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--interval-sec", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    import client_lib  # type: ignore[import-not-found]  # cluster-only dependency

    return monitor(
        job_name=args.job_name,
        run_dir=run_dir,
        status_reader=lambda job_name: client_lib.get_job_status(job_name, region=args.region),
        interval_sec=args.interval_sec,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
