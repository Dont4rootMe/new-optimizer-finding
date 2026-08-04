"""Summarize append-only ``llm_usage.jsonl`` telemetry for one population."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "uncached_prompt_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return int(ordered[index])


def _aggregate(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    counters = {field: 0 for field in TOKEN_FIELDS}
    latencies: list[int] = []
    generations: list[int] = []
    response_chars = 0
    for event in rows:
        tokens = event.get("tokens") if isinstance(event.get("tokens"), dict) else {}
        for field in TOKEN_FIELDS:
            try:
                counters[field] += max(0, int(tokens.get(field, 0)))
            except (TypeError, ValueError):
                pass
        try:
            latencies.append(max(0, int(event.get("elapsed_ms", 0))))
        except (TypeError, ValueError):
            pass
        try:
            generations.append(int(event.get("generation", 0)))
        except (TypeError, ValueError):
            pass
        try:
            response_chars += max(0, int(event.get("response_chars", 0)))
        except (TypeError, ValueError):
            pass
    elapsed_seconds = sum(latencies) / 1000.0
    return {
        "calls": len(rows),
        **counters,
        "response_chars": response_chars,
        "first_generation": min(generations) if generations else None,
        "last_generation": max(generations) if generations else None,
        "latency_ms": {
            "sum": sum(latencies),
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0,
        },
        "completion_tokens_per_call_second": (
            round(counters["completion_tokens"] / elapsed_seconds, 6)
            if elapsed_seconds > 0
            else 0.0
        ),
    }


def load_usage_events(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid usage events and retain malformed-line diagnostics."""

    usage_path = Path(path).expanduser().resolve()
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if not usage_path.exists():
        return events, [f"usage file does not exist: {usage_path}"]
    for line_number, raw_line in enumerate(usage_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if not isinstance(payload, dict) or payload.get("event") != "llm_usage":
            errors.append(f"line {line_number}: not an llm_usage object")
            continue
        events.append(payload)
    return events, errors


def summarize_usage(path: str | Path) -> dict[str, Any]:
    """Return totals plus route, stage, and generation projections."""

    events, errors = load_usage_events(path)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_route": defaultdict(list),
        "by_stage": defaultdict(list),
        "by_generation": defaultdict(list),
    }
    for event in events:
        grouped["by_route"][str(event.get("route_id", "<missing>"))].append(event)
        grouped["by_stage"][str(event.get("stage", "<missing>"))].append(event)
        grouped["by_generation"][str(event.get("generation", "<missing>"))].append(event)

    def _projection_key(group_name: str, key: str) -> tuple[int, int | str]:
        if group_name == "by_generation" and key.isdigit():
            return (0, int(key))
        return (1, key)

    return {
        "schema_version": 1,
        "source": str(Path(path).expanduser().resolve()),
        "parse_errors": errors,
        "totals": _aggregate(events),
        **{
            group_name: {
                key: _aggregate(rows)
                for key, rows in sorted(
                    buckets.items(),
                    key=lambda item: _projection_key(group_name, item[0]),
                )
            }
            for group_name, buckets in grouped.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usage_jsonl", help="path to population llm_usage.jsonl")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    summary = summarize_usage(args.usage_jsonl)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
