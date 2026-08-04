"""Build and validate the terminal result capsule for a cluster evolution run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.cluster.common import atomic_write_json, read_json_if_present, utc_now


def _finite_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _relative_organism_dir(value: object, population_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(population_root).as_posix()
    except ValueError:
        return path.as_posix()


def _survivor_projection(entry: dict[str, Any], population_root: Path) -> dict[str, Any]:
    return {
        "organism_id": str(entry.get("organism_id", "")),
        "island_id": str(entry.get("island_id", "")),
        "organism_dir": _relative_organism_dir(entry.get("organism_dir", ""), population_root),
        "generation_created": entry.get("generation_created"),
        "current_generation_active": entry.get("current_generation_active"),
        "simple_score": _finite_score(entry.get("simple_score")),
        "hard_score": _finite_score(entry.get("hard_score")),
    }


def _best_entry(entries: list[dict[str, Any]], score_field: str) -> dict[str, Any] | None:
    scored = [entry for entry in entries if entry.get(score_field) is not None]
    if not scored:
        return None
    return max(scored, key=lambda entry: float(entry[score_field]))


def summarize_run(
    population_root: str | Path,
    *,
    token_summary: dict[str, Any],
    expected_generation: int,
) -> dict[str, Any]:
    """Return a compact audited summary and reject false terminal success."""

    root = Path(population_root).expanduser().resolve()
    state = read_json_if_present(root / "population_state.json")
    if not isinstance(state, dict):
        raise RuntimeError(f"missing or invalid terminal population state under {root}")
    active_raw = state.get("active_organisms")
    if not isinstance(active_raw, list) or not active_raw:
        raise RuntimeError("terminal population must contain active organisms")
    if not all(isinstance(entry, dict) for entry in active_raw):
        raise RuntimeError("terminal active_organisms must contain JSON objects")

    try:
        current_generation = int(state.get("current_generation"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("terminal population state has no integer current_generation") from exc
    if current_generation != int(expected_generation):
        raise RuntimeError(
            f"EvolutionLoop stopped at generation {current_generation}; expected {expected_generation}"
        )
    if state.get("inflight_seed") is not None or state.get("inflight_generation") is not None:
        raise RuntimeError("terminal population state still contains an inflight transaction")

    parse_errors = token_summary.get("parse_errors")
    if not isinstance(parse_errors, list) or parse_errors:
        raise RuntimeError(f"terminal token telemetry has parse errors: {parse_errors!r}")
    totals = token_summary.get("totals")
    if not isinstance(totals, dict) or int(totals.get("calls", 0)) <= 0:
        raise RuntimeError("terminal token telemetry contains no real LLM calls")

    survivors = [_survivor_projection(entry, root) for entry in active_raw]
    survivors.sort(
        key=lambda entry: (
            entry["island_id"],
            -(entry["simple_score"] if entry["simple_score"] is not None else -math.inf),
            entry["organism_id"],
        )
    )
    island_summary: dict[str, dict[str, Any]] = {}
    for island_id in sorted({entry["island_id"] for entry in survivors}):
        island_entries = [entry for entry in survivors if entry["island_id"] == island_id]
        island_summary[island_id] = {
            "active_organisms": len(island_entries),
            "best_simple": _best_entry(island_entries, "simple_score"),
            "best_hard": _best_entry(island_entries, "hard_score"),
        }

    status_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    generation_counts: Counter[str] = Counter()
    historical_scored: list[dict[str, Any]] = []
    metadata_files = sorted(root.glob("gen_*/island_*/org_*/organism.json"))
    unreadable_metadata = 0
    for metadata_path in metadata_files:
        payload = read_json_if_present(metadata_path)
        if not isinstance(payload, dict):
            unreadable_metadata += 1
            continue
        status_counts[str(payload.get("status", "<missing>"))] += 1
        operator_counts[str(payload.get("operator", "<missing>"))] += 1
        generation_counts[str(payload.get("generation_created", "<missing>"))] += 1
        simple_score = _finite_score(payload.get("simple_score"))
        hard_score = _finite_score(payload.get("hard_score"))
        if simple_score is not None or hard_score is not None:
            historical_scored.append(
                {
                    "organism_id": str(payload.get("organism_id", "")),
                    "island_id": str(payload.get("island_id", "")),
                    "organism_dir": _relative_organism_dir(metadata_path.parent, root),
                    "generation_created": payload.get("generation_created"),
                    "simple_score": simple_score,
                    "hard_score": hard_score,
                }
            )

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "completion_contract": {
            "satisfied": True,
            "expected_generation": int(expected_generation),
            "current_generation": current_generation,
            "inflight_seed": False,
            "inflight_generation": False,
            "token_usage_parse_errors": 0,
            "llm_calls": int(totals["calls"]),
        },
        "population": {
            "active_organisms": len(survivors),
            "best_organism_id": state.get("best_organism_id"),
            "best_simple_score": _finite_score(state.get("best_simple_score")),
            "best_historical_simple": _best_entry(historical_scored, "simple_score"),
            "best_historical_hard": _best_entry(historical_scored, "hard_score"),
            "survivors": survivors,
            "by_island": island_summary,
        },
        "history": {
            "organism_metadata_files": len(metadata_files),
            "unreadable_organism_metadata": unreadable_metadata,
            "status_counts": dict(sorted(status_counts.items())),
            "operator_counts": dict(sorted(operator_counts.items())),
            "organisms_by_generation": dict(
                sorted(
                    generation_counts.items(),
                    key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
                )
            ),
        },
        "token_usage": {
            "totals": totals,
            "by_route": token_summary.get("by_route", {}),
            "by_stage": token_summary.get("by_stage", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--token-summary", type=Path, required=True)
    parser.add_argument("--expected-generation", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token_summary = read_json_if_present(args.token_summary)
    if not isinstance(token_summary, dict):
        raise SystemExit(f"missing or invalid token summary: {args.token_summary}")
    summary = summarize_run(
        args.population_root,
        token_summary=token_summary,
        expected_generation=args.expected_generation,
    )
    atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
