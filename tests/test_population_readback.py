from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import pytest

from scripts.cluster.population_readback import (
    blob_events,
    collect_curve_snapshot,
    resolve_source_run_dir,
    source_run_id_from_overrides,
)
from scripts.cluster.decode_population_readback import decode_blobs, parse_events


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_run_id_is_explicit_and_path_safe(tmp_path: Path) -> None:
    run_id = source_run_id_from_overrides(json.dumps(["readback.source_run_id=source-run_1"]))
    assert run_id == "source-run_1"
    assert resolve_source_run_dir(tmp_path, run_id) == (tmp_path / "runs" / run_id).resolve()
    with pytest.raises(ValueError, match="exactly one"):
        source_run_id_from_overrides("[]")
    with pytest.raises(ValueError, match="invalid"):
        source_run_id_from_overrides(json.dumps(["readback.source_run_id=../escape"]))


def test_collect_curve_snapshot_preserves_all_finite_scores(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "source"
    population = run / "population"
    _write_json(
        population / "population_state.json",
        {
            "current_generation": 1,
            "timestamp": "stable",
            "active_organisms": [{"organism_id": "child"}],
            "best_organism_id": "child",
            "best_simple_score": 2.5,
            "inflight_seed": None,
            "inflight_generation": None,
        },
    )
    _write_json(
        population / "gen_0000" / "island_a" / "org_seed" / "organism.json",
        {"organism_id": "seed", "island_id": "a", "generation_created": 0, "simple_score": 2.0},
    )
    _write_json(
        population / "gen_0001" / "island_a" / "org_child" / "organism.json",
        {"organism_id": "child", "island_id": "a", "generation_created": 1, "simple_score": 2.5},
    )
    _write_json(
        population / "gen_0001" / "island_a" / "org_bad" / "organism.json",
        {"organism_id": "bad", "island_id": "a", "generation_created": 1, "simple_score": "nan"},
    )

    snapshot = collect_curve_snapshot(run)

    assert snapshot["state_stable_during_scan"] is True
    assert [record["simple_score"] for record in snapshot["records"]] == [2.0, None, 2.5]
    assert snapshot["generation_summary"] == [
        {"generation": 0, "evaluated": 1, "best": 2.0, "cumulative_best": 2.0},
        {"generation": 1, "evaluated": 1, "best": 2.5, "cumulative_best": 2.5},
    ]
    assert next(record for record in snapshot["records"] if record["organism_id"] == "child")["active"] is True


def test_blob_events_round_trip_and_include_integrity_metadata() -> None:
    payload = (b"complete solution-quality snapshot\n" * 1_000) + b"end"
    events = list(blob_events("snapshot.json", payload, media_type="application/json"))
    encoded = "".join(event["data"] for event in sorted(events, key=lambda item: item["index"]))
    assert gzip.decompress(base64.b64decode(encoded)) == payload
    assert all(event["total"] == len(events) for event in events)
    assert len({event["sha256"] for event in events}) == 1


def test_scheduler_log_decoder_handles_mpi_prefixes_and_duplicate_chunks(tmp_path: Path) -> None:
    payload = b"exact curve payload" * 1_000
    events = list(blob_events("snapshot.json", payload, media_type="application/json"))
    lines = [
        f"[1,0]<stdout>:EVOLUTIONLOOP_READBACK {json.dumps(event, separators=(',', ':'))}\n"
        for event in events
    ]
    parsed = parse_events(["unrelated\n", *lines, lines[0]])
    decoded = decode_blobs(parsed, tmp_path)
    assert (tmp_path / "snapshot.json").read_bytes() == payload
    assert decoded[0]["name"] == "snapshot.json"
