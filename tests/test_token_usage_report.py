"""Detailed token-usage aggregation tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.evolve.token_usage_report import summarize_usage


def test_token_usage_report_projects_route_stage_and_generation(tmp_path: Path) -> None:
    usage_path = tmp_path / "llm_usage.jsonl"
    rows = [
        {
            "event": "llm_usage",
            "route_id": "deepseek",
            "stage": "design",
            "generation": 1,
            "elapsed_ms": 1000,
            "response_chars": 50,
            "tokens": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cached_prompt_tokens": 60,
                "uncached_prompt_tokens": 40,
                "reasoning_tokens": 10,
            },
        },
        {
            "event": "llm_usage",
            "route_id": "deepseek",
            "stage": "implementation",
            "generation": 2,
            "elapsed_ms": 3000,
            "response_chars": 150,
            "tokens": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
                "cached_prompt_tokens": 100,
                "uncached_prompt_tokens": 100,
                "reasoning_tokens": 30,
            },
        },
    ]
    usage_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_usage(usage_path)

    assert summary["parse_errors"] == []
    assert summary["totals"]["calls"] == 2
    assert summary["totals"]["total_tokens"] == 400
    assert summary["totals"]["cached_prompt_tokens"] == 160
    assert summary["totals"]["latency_ms"]["p50"] == 1000
    assert summary["totals"]["latency_ms"]["p95"] == 3000
    assert summary["by_stage"]["design"]["reasoning_tokens"] == 10
    assert summary["by_generation"]["2"]["completion_tokens"] == 80


def test_token_usage_report_sorts_numeric_and_malformed_generations(tmp_path: Path) -> None:
    usage_path = tmp_path / "llm_usage.jsonl"
    usage_path.write_text(
        "\n".join(
            json.dumps({"event": "llm_usage", "generation": value, "tokens": {}})
            for value in (10, "missing", 2)
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_usage(usage_path)

    assert list(summary["by_generation"]) == ["2", "10", "missing"]
