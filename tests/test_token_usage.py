"""Provider-neutral token accounting and append-only telemetry contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.evolve.token_usage import (
    USAGE_LOGGER,
    emit_usage_event,
    ensure_usage_jsonl_logging,
    normalize_token_usage,
)


def test_normalize_openai_sglang_usage_preserves_detailed_counters() -> None:
    usage = normalize_token_usage(
        {
            "prompt_tokens": 120,
            "completion_tokens": 45,
            "total_tokens": 165,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {
                "reasoning_tokens": 20,
                "accepted_prediction_tokens": 11,
                "rejected_prediction_tokens": 3,
            },
        }
    )

    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 165,
        "cached_prompt_tokens": 80,
        "uncached_prompt_tokens": 40,
        "cache_write_tokens": 0,
        "reasoning_tokens": 20,
        "accepted_prediction_tokens": 11,
        "rejected_prediction_tokens": 3,
    }


def test_normalize_ollama_and_anthropic_usage_aliases() -> None:
    assert normalize_token_usage({"prompt_eval_count": 7, "eval_count": 5})[
        "total_tokens"
    ] == 12
    anthropic = normalize_token_usage(
        {
            "input_tokens": 30,
            "output_tokens": 9,
            "cache_read_input_tokens": 17,
            "cache_creation_input_tokens": 4,
        }
    )
    assert anthropic["total_tokens"] == 39
    assert anthropic["cached_prompt_tokens"] == 17
    assert anthropic["uncached_prompt_tokens"] == 13
    assert anthropic["cache_write_tokens"] == 4


def test_usage_jsonl_handler_is_idempotent_and_writes_one_raw_event(tmp_path: Path) -> None:
    log_path = ensure_usage_jsonl_logging(tmp_path)
    assert ensure_usage_jsonl_logging(tmp_path) == log_path
    handlers = [
        handler
        for handler in USAGE_LOGGER.handlers
        if getattr(handler, "_evolve_usage_marker", "").endswith(str(log_path))
    ]
    assert len(handlers) == 1

    event = emit_usage_event(
        route_id="deepseek_v4_flash",
        provider="openai_compatible",
        provider_model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        organism_id="org-1",
        generation=3,
        stage="mutation",
        elapsed_sec=1.234,
        system_prompt_chars=100,
        user_prompt_chars=200,
        response_chars=300,
        started_at="2026-08-04T00:00:00Z",
        finished_at="2026-08-04T00:00:01Z",
        usage={"prompt_tokens": 40, "completion_tokens": 10},
    )
    for handler in handlers:
        handler.flush()

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [event]
    assert rows[0]["tokens"]["total_tokens"] == 50

    for handler in handlers:
        USAGE_LOGGER.removeHandler(handler)
        handler.close()
    logging.shutdown()
