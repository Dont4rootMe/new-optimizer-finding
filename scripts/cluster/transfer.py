"""Export regional Cloud.ru job artifacts back to the workspace NFS.

SR008 jobs and the submitting Jupyter server have distinct NFS namespaces.
Source code enters a job through the pinned HTTPS git bootstrap in ``submit``;
completed run artifacts leave through ML Space Data Transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from scripts.cluster.common import (
    REGION,
    atomic_write_json,
    require_absolute_safe_path,
    require_regional_job_path,
    utc_now,
)

JUPYTER_MOUNT = Path("/home/jovyan")


def connector_path(path: Path) -> str:
    """Translate an absolute NFS path to a workspace-connector path."""

    resolved = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        relative = resolved.relative_to(JUPYTER_MOUNT)
    except ValueError as exc:
        raise ValueError(f"Data Transfer path must be below {JUPYTER_MOUNT}: {resolved}") from exc
    if not relative.parts:
        raise ValueError("refusing to transfer the whole NFS mount")
    return "/" + relative.as_posix()


def select_nfs_connector(connectors: list[dict[str, Any]]) -> str:
    candidates = [
        item for item in connectors
        if str(item.get("source_type", "")).lower() == "nfs" and item.get("connector_id")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one workspace NFS connector, found {len(candidates)}")
    return str(candidates[0]["connector_id"])


def wait_for_transfer(
    transfer_id: str,
    *,
    log_reader: Callable[[str], list[dict[str, Any]]],
    timeout_sec: float = 3600.0,
    interval_sec: float = 5.0,
) -> list[dict[str, Any]]:
    """Wait for Data Transfer's object log, the public terminal contract."""

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        logs = log_reader(transfer_id)
        if logs:
            failed = [
                item for item in logs
                if str(item.get("status", item.get("state", ""))).lower()
                in {"failed", "error", "cancelled", "canceled"}
                or item.get("error")
            ]
            if failed:
                raise RuntimeError(f"Data Transfer {transfer_id} failed: {failed}")
            return logs
        time.sleep(max(0.05, interval_sec))
    raise TimeoutError(f"Data Transfer {transfer_id} produced no object log in {timeout_sec}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-run-dir", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    regional_run = require_regional_job_path(args.regional_run_dir, label="regional run directory")
    destination = require_absolute_safe_path(args.destination, label="export destination")
    import client_lib  # type: ignore[import-not-found]  # control-plane-only dependency

    connector_id = select_nfs_connector(client_lib.get_connectors())
    transfer = client_lib.copy_from_nfs(
        source_path=connector_path(regional_run),
        destination_path=connector_path(destination),
        from_region=getattr(client_lib.RegionEnum, args.region),
        destination_connector_id=connector_id,
    )
    logs = wait_for_transfer(
        transfer.id,
        log_reader=client_lib.get_transfer_data_logs,
        timeout_sec=args.timeout_sec,
    )
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "region": args.region,
        "regional_run_dir": str(regional_run),
        "destination": str(destination),
        "transfer_id": transfer.id,
        "logs": logs,
    }
    atomic_write_json(destination.parent / f"{destination.name}.export.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
