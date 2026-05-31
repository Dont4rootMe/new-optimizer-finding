"""Per-model token-budget guard for the ShinkaEvolve baseline runner.

These tests exercise the helpers in ``src.baselines.shinka.run`` directly with
a fake runner/LLM-client, so they do NOT require the (external, optional)
``shinka`` package to be installed — ``run.py`` imports shinka lazily inside
``_import_shinka`` only.
"""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import OmegaConf

from src.baselines.shinka.run import (
    _TokenBudgetExceeded,
    _install_shinka_token_budget_guard,
    _normalize_model_name,
    _parse_token_caps,
)


class _FakeResult:
    """Mimics shinka's QueryResult token-bearing fields."""

    def __init__(
        self,
        model_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        self.model_name = model_name
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens


class _FakeClient:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result
        self.calls = 0

    async def query(self, *args, **kwargs):
        self.calls += 1
        return self._result


class _FakeRunner:
    def __init__(self, client: _FakeClient) -> None:
        self.llm = client
        self.should_stop = asyncio.Event()


def test_normalize_model_name_strips_local_prefix_and_url() -> None:
    assert _normalize_model_name("local/qwen3.5:122b@http://127.0.0.1:1/v1") == "qwen3.5:122b"
    assert _normalize_model_name("gemma4:31b") == "gemma4:31b"


def test_parse_token_caps_drops_false_and_keeps_int() -> None:
    block = OmegaConf.create(
        {"max_tokens_per_model": {"qwen3.5:122b": False, "gemma4:31b": 1000, "disabled": -1}}
    )
    assert _parse_token_caps(block) == {"gemma4:31b": 1000}


def test_parse_token_caps_absent_is_empty() -> None:
    assert _parse_token_caps(OmegaConf.create({})) == {}
    assert _parse_token_caps(
        OmegaConf.create({"max_tokens_per_model": {"qwen3.5:122b": False}})
    ) == {}


def test_parse_token_caps_rejects_true() -> None:
    block = OmegaConf.create({"max_tokens_per_model": {"gemma4:31b": True}})
    with pytest.raises(SystemExit):
        _parse_token_caps(block)


def test_guard_not_installed_without_caps() -> None:
    runner = _FakeRunner(_FakeClient(_FakeResult("qwen3.5:122b")))
    assert _install_shinka_token_budget_guard(runner, {}) is False


def test_guard_stops_when_model_exceeds_budget() -> None:
    runner = _FakeRunner(_FakeClient(_FakeResult("qwen3.5:122b", input_tokens=300, output_tokens=300)))
    assert _install_shinka_token_budget_guard(runner, {"qwen3.5:122b": 1000}) is True

    async def _drive() -> None:
        await runner.llm.query()  # 600 cumulative
        await runner.llm.query()  # 1200 >= 1000 -> raise

    with pytest.raises(_TokenBudgetExceeded) as excinfo:
        asyncio.run(_drive())

    assert excinfo.value.model_name == "qwen3.5:122b"
    assert excinfo.value.limit == 1000
    # The guard also flips the runner's native stop event.
    assert runner.should_stop.is_set()


def test_guard_ignores_uncapped_model() -> None:
    runner = _FakeRunner(
        _FakeClient(_FakeResult("gemma4:31b", input_tokens=10**6, output_tokens=10**6))
    )
    assert _install_shinka_token_budget_guard(runner, {"qwen3.5:122b": 1000}) is True

    async def _drive() -> None:
        for _ in range(5):
            await runner.llm.query()

    asyncio.run(_drive())  # gemma is uncapped -> never raises
    assert not runner.should_stop.is_set()


def test_guard_matches_url_form_model_name() -> None:
    # Some shinka versions may report the full local/...@url form; a cap keyed
    # by the bare provider_model_id must still match after normalization.
    runner = _FakeRunner(
        _FakeClient(_FakeResult("local/qwen3.5:122b@http://h/v1", input_tokens=1100))
    )
    _install_shinka_token_budget_guard(runner, {"qwen3.5:122b": 1000})
    with pytest.raises(_TokenBudgetExceeded):
        asyncio.run(runner.llm.query())
