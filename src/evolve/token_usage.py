"""Provider-neutral token accounting and append-only usage telemetry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

USAGE_LOGGER_NAME = "evolution.llm_usage"
USAGE_LOGGER = logging.getLogger(USAGE_LOGGER_NAME)


def _numeric_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _first_int(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = _numeric_int(payload.get(key))
        if value:
            return value
    return 0


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def normalize_token_usage(usage: object) -> dict[str, int]:
    """Return stable token counters from OpenAI, Anthropic, Ollama, or SGLang.

    The first three counters are the algorithm's existing stop/accounting
    contract.  Additional counters preserve cache, reasoning, and speculative
    decoding detail when the serving engine reports them.
    """

    payload = usage if isinstance(usage, dict) else {}
    prompt = _first_int(payload, "prompt_tokens", "prompt_eval_count", "input_tokens")
    completion = _first_int(payload, "completion_tokens", "eval_count", "output_tokens")
    total = _first_int(payload, "total_tokens") or prompt + completion

    prompt_details = _nested_dict(
        payload,
        "prompt_tokens_details",
        "input_tokens_details",
        "prompt_token_details",
    )
    completion_details = _nested_dict(
        payload,
        "completion_tokens_details",
        "output_tokens_details",
        "completion_token_details",
    )
    cached = _first_int(
        prompt_details,
        "cached_tokens",
        "cache_read_tokens",
        "cache_read_input_tokens",
    ) or _first_int(payload, "cache_read_input_tokens", "cached_tokens")
    cache_write = _first_int(
        prompt_details,
        "cache_creation_tokens",
        "cache_write_tokens",
        "cache_creation_input_tokens",
    ) or _first_int(payload, "cache_creation_input_tokens", "cache_write_tokens")
    reasoning = _first_int(
        completion_details,
        "reasoning_tokens",
        "reasoning_output_tokens",
    ) or _first_int(payload, "reasoning_tokens")
    accepted = _first_int(completion_details, "accepted_prediction_tokens")
    rejected = _first_int(completion_details, "rejected_prediction_tokens")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_prompt_tokens": min(cached, prompt) if prompt else cached,
        "uncached_prompt_tokens": max(0, prompt - cached),
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "accepted_prediction_tokens": accepted,
        "rejected_prediction_tokens": rejected,
    }


def emit_usage_event(
    *,
    route_id: str,
    provider: str,
    provider_model_id: str,
    organism_id: str,
    generation: int,
    stage: str,
    elapsed_sec: float,
    system_prompt_chars: int,
    user_prompt_chars: int,
    response_chars: int,
    started_at: str,
    finished_at: str,
    usage: object,
) -> dict[str, Any]:
    """Log one machine-readable usage event and return its payload."""

    provider_usage = usage if isinstance(usage, dict) else {}
    event = {
        "event": "llm_usage",
        "route_id": route_id,
        "provider": provider,
        "provider_model_id": provider_model_id,
        "organism_id": organism_id,
        "generation": int(generation),
        "stage": stage,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": int(round(max(0.0, elapsed_sec) * 1000.0)),
        "system_prompt_chars": int(system_prompt_chars),
        "user_prompt_chars": int(user_prompt_chars),
        "response_chars": int(response_chars),
        "tokens": normalize_token_usage(provider_usage),
        "provider_usage": provider_usage,
    }
    USAGE_LOGGER.info(json.dumps(event, sort_keys=True, separators=(",", ":")))
    return event


def ensure_usage_jsonl_logging(population_root: str | Path) -> Path:
    """Attach an idempotent raw-JSON handler at ``<population>/llm_usage.jsonl``."""

    root = Path(population_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "llm_usage.jsonl"
    marker = f"_evolve_usage_handler_{log_path}"
    if any(getattr(handler, "_evolve_usage_marker", None) == marker for handler in USAGE_LOGGER.handlers):
        return log_path
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._evolve_usage_marker = marker  # type: ignore[attr-defined]
    USAGE_LOGGER.addHandler(handler)
    USAGE_LOGGER.setLevel(logging.INFO)
    # Keep propagation enabled: run.log receives the same event with normal
    # timestamps while llm_usage.jsonl stays clean and machine-readable.
    return log_path
