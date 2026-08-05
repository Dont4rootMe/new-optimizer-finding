"""Read one regional EvolutionLoop run and emit a portable scheduler-log snapshot.

The readback job deliberately writes nothing below the source run.  It is meant
for the case where the control-plane Jupyter server cannot mount the regional
job NFS: compact JSON and selected PNGs are gzip/base64 encoded into bounded
log events and reconstructed by the operator after the job completes.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVENT_PREFIX = "EVOLUTIONLOOP_READBACK "
SOURCE_OVERRIDE = "readback.source_run_id="
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LOG_CHUNK_CHARS = 3_000
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_event(payload: dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def source_run_id_from_overrides(raw: str) -> str:
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("HYDRA_OVERRIDES_JSON must be valid JSON") from exc
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise ValueError("HYDRA_OVERRIDES_JSON must encode a list of strings")
    values = [item[len(SOURCE_OVERRIDE) :] for item in overrides if item.startswith(SOURCE_OVERRIDE)]
    if len(values) != 1:
        raise ValueError(f"exactly one {SOURCE_OVERRIDE}<id> override is required")
    run_id = values[0].strip()
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(f"invalid readback source run id: {run_id!r}")
    return run_id


def resolve_source_run_dir(job_root: str | Path, source_run_id: str) -> Path:
    root = Path(job_root).expanduser().resolve()
    if not root.is_absolute():
        raise ValueError("JOB_ROOT must be absolute")
    runs_root = (root / "runs").resolve()
    source = (runs_root / source_run_id).resolve()
    try:
        source.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("source run must remain below JOB_ROOT/runs") from exc
    return source


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def collect_curve_snapshot(source_run_dir: Path) -> dict[str, Any]:
    population_root = source_run_dir / "population"
    state_path = population_root / "population_state.json"
    state_before = read_json_object(state_path)
    active_ids = {
        str(entry.get("organism_id"))
        for entry in state_before.get("active_organisms", [])
        if isinstance(entry, dict) and entry.get("organism_id")
    }

    records: list[dict[str, Any]] = []
    read_errors: list[dict[str, str]] = []
    meta_paths = sorted(population_root.glob("gen_*/island_*/org_*/organism.json"))
    for meta_path in meta_paths:
        try:
            meta = read_json_object(meta_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            read_errors.append({"path": str(meta_path.relative_to(population_root)), "error": str(exc)})
            continue
        organism_id = str(meta.get("organism_id") or meta_path.parent.name.removeprefix("org_"))
        records.append(
            {
                "organism_id": organism_id,
                "island_id": str(meta.get("island_id", "unknown")),
                "generation": int(meta.get("generation_created", 0)),
                "current_generation_active": int(meta.get("current_generation_active", 0)),
                "operator": str(meta.get("operator", "unknown")),
                "status": str(meta.get("status", "unknown")),
                "pipeline_id": meta.get("pipeline_id"),
                "pipeline_state": meta.get("pipeline_state"),
                "mother_id": meta.get("mother_id"),
                "father_id": meta.get("father_id"),
                "simple_score": finite_score(meta.get("simple_score")),
                "hard_score": finite_score(meta.get("hard_score")),
                "timestamp": meta.get("timestamp"),
                "active": organism_id in active_ids,
                "relative_dir": str(meta_path.parent.relative_to(population_root)),
            }
        )

    records.sort(key=lambda item: (int(item["generation"]), str(item["organism_id"])))
    state_after = read_json_object(state_path)
    scores_by_generation: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record["simple_score"] is not None:
            scores_by_generation[int(record["generation"])].append(float(record["simple_score"]))

    running_best: float | None = None
    generation_summary: list[dict[str, Any]] = []
    for generation in sorted(scores_by_generation):
        scores = scores_by_generation[generation]
        generation_best = max(scores)
        running_best = generation_best if running_best is None else max(running_best, generation_best)
        generation_summary.append(
            {
                "generation": generation,
                "evaluated": len(scores),
                "best": generation_best,
                "cumulative_best": running_best,
            }
        )

    return {
        "schema_version": 1,
        "captured_at": utc_now(),
        "source_run_dir": str(source_run_dir),
        "population_root": str(population_root),
        "state_before": {
            "current_generation": state_before.get("current_generation"),
            "timestamp": state_before.get("timestamp"),
            "inflight_seed": state_before.get("inflight_seed") is not None,
            "inflight_generation": state_before.get("inflight_generation") is not None,
            "active_organisms": len(active_ids),
            "best_organism_id": state_before.get("best_organism_id"),
            "best_simple_score": finite_score(state_before.get("best_simple_score")),
        },
        "state_after": {
            "current_generation": state_after.get("current_generation"),
            "timestamp": state_after.get("timestamp"),
            "inflight_seed": state_after.get("inflight_seed") is not None,
            "inflight_generation": state_after.get("inflight_generation") is not None,
        },
        "state_stable_during_scan": state_before.get("timestamp") == state_after.get("timestamp"),
        "organism_files_seen": len(meta_paths),
        "records": records,
        "generation_summary": generation_summary,
        "read_errors": read_errors,
    }


def summarize_usage(path: Path) -> dict[str, Any]:
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    by_route: dict[str, dict[str, int]] = defaultdict(lambda: dict(totals))
    malformed = 0
    if not path.exists():
        return {**totals, "by_route": {}, "malformed_lines": 0, "present": False}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(payload, dict):
            malformed += 1
            continue
        tokens = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
        route_id = str(payload.get("route_id") or "unknown")
        for bucket in (totals, by_route[route_id]):
            bucket["calls"] += 1
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
                try:
                    bucket[key] += int(tokens.get(key, 0) or 0)
                except (TypeError, ValueError):
                    pass
    return {**totals, "by_route": dict(by_route), "malformed_lines": malformed, "present": True}


def blob_events(name: str, payload: bytes, *, media_type: str) -> Iterable[dict[str, Any]]:
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [encoded[index : index + LOG_CHUNK_CHARS] for index in range(0, len(encoded), LOG_CHUNK_CHARS)]
    sha256 = hashlib.sha256(payload).hexdigest()
    for index, chunk in enumerate(chunks):
        yield {
            "event": "blob_chunk",
            "name": name,
            "media_type": media_type,
            "encoding": "gzip+base64",
            "sha256": sha256,
            "uncompressed_bytes": len(payload),
            "compressed_bytes": len(compressed),
            "index": index,
            "total": len(chunks),
            "data": chunk,
        }


def emit_blob(name: str, payload: bytes, *, media_type: str) -> None:
    for event in blob_events(name, payload, media_type=media_type):
        emit_event(event)


def main() -> int:
    source_run_id = source_run_id_from_overrides(os.environ.get("HYDRA_OVERRIDES_JSON", "[]"))
    source_run_dir = resolve_source_run_dir(os.environ["JOB_ROOT"], source_run_id)
    current_run_id = os.environ.get("RUN_ID")
    if current_run_id == source_run_id:
        raise ValueError("readback source run id must differ from the readback job run id")

    snapshot = collect_curve_snapshot(source_run_dir)
    population_root = source_run_dir / "population"
    usage = summarize_usage(population_root / "llm_usage.jsonl")
    evaluated = sum(1 for record in snapshot["records"] if record["simple_score"] is not None)
    emit_event(
        {
            "event": "snapshot_summary",
            "captured_at": snapshot["captured_at"],
            "source_run_dir": str(source_run_dir),
            "current_generation": snapshot["state_after"]["current_generation"],
            "state_stable_during_scan": snapshot["state_stable_during_scan"],
            "organism_records": len(snapshot["records"]),
            "evaluated_records": evaluated,
            "read_errors": len(snapshot["read_errors"]),
            "best_simple_score": snapshot["state_before"]["best_simple_score"],
            "usage": usage,
        }
    )
    emit_blob(
        "solution_quality_snapshot.json",
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        media_type="application/json",
    )

    artifact_paths = {
        "score_by_generation.png": population_root / "viz" / "overview" / "score_by_generation.png",
        "evolution_overview.png": population_root / "evolution_overview.png",
    }
    for name, path in artifact_paths.items():
        if not path.is_file():
            emit_event({"event": "artifact_missing", "name": name, "path": str(path)})
            continue
        size = path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            emit_event({"event": "artifact_skipped", "name": name, "path": str(path), "bytes": size})
            continue
        emit_blob(name, path.read_bytes(), media_type="image/png")

    emit_event({"event": "readback_complete", "source_run_dir": str(source_run_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
