"""Contract tests for api_platforms routing, broker reuse, and local concurrency."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error as urllib_error

import pytest
from omegaconf import OmegaConf

from api_platforms import ApiPlatformRegistry, LlmRequest
from api_platforms._core import broker as broker_module
from api_platforms._core import registry as registry_module
from api_platforms._core import providers as provider_backends
from api_platforms._core.config import build_route_config, derive_ollama_instance_configs
from api_platforms._core.providers import generate_direct
from api_platforms._core.types import ApiRouteConfig, LlmResponse
from src.evolve.generator import CandidateGenerator
from src.evolve.types import Island


def _cfg(tmp_path: Path, *, routes: dict, route_weights: dict[str, float]) -> object:
    return OmegaConf.create(
        {
            "seed": 123,
            "paths": {
                "api_platform_runtime_root": str(tmp_path / ".api_platform_runtime"),
            },
            "api_platforms": routes,
            "evolver": {
                "creation": {
                    "max_attempts_to_create_organism": 1,
                    "max_attempts_to_repair_organism_after_error": 1,
                    "max_attempts_to_regenerate_organism_after_novelty_rejection": 1,
                },
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "mutation_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/mutation/system.txt",
                    "mutation_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "crossover_novelty_system": "conf/experiments/optimization_survey/prompts/novelty/crossover/system.txt",
                    "crossover_novelty_user": "conf/experiments/optimization_survey/prompts/novelty/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
                    "repair_system": "conf/experiments/optimization_survey/prompts/repair/system.txt",
                    "repair_user": "conf/experiments/optimization_survey/prompts/repair/user.txt",
                },
                "llm": {
                    "selection_strategy": "random",
                    "route_weights": route_weights,
                    "seed": 123,
                },
            },
        }
    )


def test_unknown_route_in_weights_fails_fast(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        routes={"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
        route_weights={"missing_route": 1.0},
    )
    with pytest.raises(ValueError, match="unknown route ids"):
        CandidateGenerator(cfg)


def test_registry_reuses_singleton_broker_between_clients(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        routes={"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
        route_weights={"mock": 1.0},
    )
    registry_a = ApiPlatformRegistry(cfg)
    registry_b = ApiPlatformRegistry(cfg)
    try:
        registry_a.start()
        pid_a = json.loads((registry_a._route_runtime_dir("mock") / "broker.pid").read_text(encoding="utf-8"))["pid"]
        registry_b.start()
        pid_b = json.loads((registry_b._route_runtime_dir("mock") / "broker.pid").read_text(encoding="utf-8"))["pid"]
        assert pid_a == pid_b
    finally:
        registry_a.stop()
        registry_b.stop()


def test_registry_health_probe_uses_short_timeout_for_startup_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(
        tmp_path,
        routes={"mock": {"_target_": "api_platforms.mock.platform.build_platform", "timeout_sec": 1800}},
        route_weights={"mock": 1.0},
    )
    registry = ApiPlatformRegistry(cfg)
    socket_path = registry._socket_path("mock")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.touch()
    observed: dict[str, float] = {}

    def fake_send_ipc_message(socket_path: str, payload: dict[str, object], timeout_sec: float):
        observed["timeout_sec"] = timeout_sec
        raise TimeoutError("stale broker socket")

    monkeypatch.setattr(registry_module, "send_ipc_message", fake_send_ipc_message)

    assert registry._broker_healthy("mock") is False
    assert observed["timeout_sec"] <= 2.0


def test_mock_local_route_processes_requests_concurrently(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        routes={
            "mock_local": {
                "_target_": "api_platforms.mock_local.platform.build_platform",
                "gpu_ranks": [0, 1],
                "mock_delay_sec": 0.2,
            }
        },
        route_weights={"mock_local": 1.0},
    )
    registry = ApiPlatformRegistry(cfg)
    try:
        registry.start()
        client = registry.client_for("mock_local")

        def run_one(index: int):
            response = client.generate(
                LlmRequest(
                    route_id="mock_local",
                    stage="design",
                    system_prompt="sys",
                    user_prompt=f"user {index}",
                    seed=123,
                    metadata={"organism_id": f"org{index}", "generation": 0},
                )
            )
            return int(response.raw_response["gpu_rank"])

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            ranks = list(pool.map(run_one, range(4)))
        elapsed = time.perf_counter() - started

        assert set(ranks) == {0, 1}
        assert elapsed < 1.1
    finally:
        registry.stop()


def test_build_route_config_normalizes_grouped_ollama_gpu_ranks() -> None:
    route_cfg = build_route_config(
        route_id="ollama_qwen35_122b",
        provider="ollama",
        provider_model_id="qwen3.5:122b",
        backend="ollama",
        base_url="http://127.0.0.1:11434/api",
        gpu_ranks=[[0, 1, 2], [3, 4, 5]],
        max_concurrency=3,
    )

    assert route_cfg.gpu_ranks == [0, 1, 2, 3, 4, 5]
    assert route_cfg.gpu_rank_groups == [[0, 1, 2], [3, 4, 5]]

    instances = derive_ollama_instance_configs(route_cfg)
    assert [instance.base_url for instance in instances] == [
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11435/api",
    ]
    assert [instance.gpu_ranks for instance in instances] == [[0, 1, 2], [3, 4, 5]]
    assert all(instance.gpu_rank_groups == [instance.gpu_ranks] for instance in instances)
    assert all(instance.max_concurrency == 3 for instance in instances)


def test_non_ollama_route_rejects_multi_gpu_grouped_gpu_ranks() -> None:
    with pytest.raises(ValueError, match="Only ollama routes support multi-GPU grouped gpu_ranks"):
        build_route_config(
            route_id="mock_local",
            provider="mock_local",
            provider_model_id="mock-local-model",
            backend="mock_local",
            gpu_ranks=[[0, 1], [2, 3]],
        )


def test_ollama_broker_dispatches_across_grouped_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route_cfg = build_route_config(
        route_id="ollama_qwen35_122b",
        provider="ollama",
        provider_model_id="qwen3.5:122b",
        backend="ollama",
        base_url="http://127.0.0.1:11434/api",
        gpu_ranks=[[0, 1, 2], [3, 4, 5]],
        max_concurrency=1,
        timeout_sec=5.0,
    )

    def fake_generate_direct(instance_cfg: ApiRouteConfig, request: LlmRequest) -> LlmResponse:
        time.sleep(0.15)
        return LlmResponse(
            text=f"ok:{request.metadata['organism_id']}",
            route_id=instance_cfg.route_id,
            provider=instance_cfg.provider,
            provider_model_id=instance_cfg.provider_model_id,
            raw_request={"base_url": instance_cfg.base_url},
            raw_response={"base_url": instance_cfg.base_url, "gpu_ranks": list(instance_cfg.gpu_ranks)},
            usage={},
            started_at="2026-01-01T00:00:00+0000",
            finished_at="2026-01-01T00:00:01+0000",
        )

    monkeypatch.setattr(broker_module, "generate_direct", fake_generate_direct)

    broker = broker_module.RouteBroker(route_cfg, tmp_path / "runtime")
    try:
        def run_one(index: int) -> str:
            request = LlmRequest(
                route_id="ollama_qwen35_122b",
                stage="design",
                system_prompt="sys",
                user_prompt=f"user {index}",
                seed=123,
                metadata={"organism_id": f"org{index}"},
            )
            payload = broker._handle_message({"type": "generate", "request": request.to_dict()})
            assert payload["ok"] is True
            return str(payload["response"]["raw_response"]["base_url"])

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            urls = list(pool.map(run_one, range(2)))
        elapsed = time.perf_counter() - started

        assert set(urls) == {"http://127.0.0.1:11434/api", "http://127.0.0.1:11435/api"}
        assert elapsed < 0.35
    finally:
        broker.stop()


def test_seed_generation_uses_one_route_for_both_stages(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        routes={"mock": {"_target_": "api_platforms.mock.platform.build_platform"}},
        route_weights={"mock": 1.0},
    )
    generator = CandidateGenerator(cfg)
    island = Island(
        island_id="gradient_methods",
        name="gradient methods",
        description_path=str(tmp_path / "gradient_methods.txt"),
        description_text="First-order optimization heuristics.",
    )
    organism_dir = tmp_path / "org_seed"
    organism_dir.mkdir(parents=True, exist_ok=True)
    try:
        generator.generate_seed_organism(
            island=island,
            organism_id="seed01",
            generation=0,
            organism_dir=organism_dir,
        )
    finally:
        generator.close()

    request_payload = json.loads((organism_dir / "llm_request.json").read_text(encoding="utf-8"))
    response_payload = json.loads((organism_dir / "llm_response.json").read_text(encoding="utf-8"))
    assert request_payload["design"]["route_id"] == request_payload["implementation"]["route_id"] == "mock"
    assert response_payload["design"]["route_id"] == response_payload["implementation"]["route_id"] == "mock"


def test_ollama_route_posts_chat_payload_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    route_cfg = ApiRouteConfig(
        route_id="ollama_gemma4_26b",
        provider="ollama",
        provider_model_id="gemma4:26b",
        backend="ollama",
        api_key_env="OLLAMA_API_KEY",
        base_url="http://localhost:11434/api",
        temperature=0.7,
        max_output_tokens=2048,
        timeout_sec=45.0,
        top_p=0.9,
        top_k=32,
        think=False,
        keep_alive="15m",
        request_options={"num_ctx": 65536},
    )
    request = LlmRequest(
        route_id="ollama_gemma4_26b",
        stage="design",
        system_prompt="system prompt",
        user_prompt="user prompt",
        seed=123,
        metadata={"organism_id": "org001"},
    )
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")
    captured: dict[str, object] = {}

    class FakeHttpResponse:
        def __enter__(self) -> "FakeHttpResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "message": {"role": "assistant", "content": "generated text"},
                    "total_duration": 11,
                    "load_duration": 2,
                    "prompt_eval_count": 17,
                    "prompt_eval_duration": 3,
                    "eval_count": 19,
                    "eval_duration": 4,
                }
            ).encode("utf-8")

    def fake_urlopen(http_request, timeout: float):
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["method"] = http_request.get_method()
        captured["headers"] = {key.lower(): value for key, value in http_request.header_items()}
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return FakeHttpResponse()

    monkeypatch.setattr(provider_backends.urllib_request, "urlopen", fake_urlopen)

    response = generate_direct(route_cfg, request)

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 45.0
    assert captured["method"] == "POST"
    assert captured["headers"] == {
        "content-type": "application/json",
        "authorization": "Bearer secret-token",
    }
    assert captured["body"] == {
        "model": "gemma4:26b",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "stream": False,
        "options": {
            "num_ctx": 65536,
            "temperature": 0.7,
            "num_predict": 2048,
            "top_p": 0.9,
            "top_k": 32,
        },
        "think": False,
        "keep_alive": "15m",
    }
    assert response.text == "generated text"
    assert response.route_id == "ollama_gemma4_26b"
    assert response.provider == "ollama"
    assert response.provider_model_id == "gemma4:26b"
    assert response.raw_request == captured["body"]
    assert response.usage == {
        "total_duration": 11,
        "load_duration": 2,
        "prompt_eval_count": 17,
        "prompt_eval_duration": 3,
        "eval_count": 19,
        "eval_duration": 4,
    }


def test_ollama_route_rejects_truncated_thinking_only_response(monkeypatch: pytest.MonkeyPatch) -> None:
    route_cfg = ApiRouteConfig(
        route_id="ollama_gemma4_26b",
        provider="ollama",
        provider_model_id="gemma4:26b",
        backend="ollama",
        base_url="http://localhost:11434/api",
        timeout_sec=45.0,
    )
    request = LlmRequest(
        route_id="ollama_gemma4_26b",
        stage="implementation",
        system_prompt="system prompt",
        user_prompt="user prompt",
        seed=123,
        metadata={"organism_id": "org001"},
    )

    class FakeHttpResponse:
        def __enter__(self) -> "FakeHttpResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "done_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "I am still reasoning about the answer.",
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(_http_request, timeout: float):
        return FakeHttpResponse()

    monkeypatch.setattr(provider_backends.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="exhausted reasoning budget before final answer"):
        generate_direct(route_cfg, request)


def test_ollama_route_applies_stage_specific_generation_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    route_cfg = ApiRouteConfig(
        route_id="ollama_qwen35_27b",
        provider="ollama",
        provider_model_id="qwen3.5:27b",
        backend="ollama",
        base_url="http://127.0.0.1:11435/api",
        temperature=0.7,
        max_output_tokens=4096,
        timeout_sec=30.0,
        top_p=0.92,
        top_k=40,
        think=False,
        keep_alive="15m",
        request_options={
            "num_ctx": 65536,
            "repeat_penalty": 1.05,
            "min_p": None,
        },
        stage_options={
            "design": {
                "think": "high",
                "temperature": 0.2,
                "max_output_tokens": 1536,
                "top_k": 12,
                "raw": True,
                "logprobs": True,
                "top_logprobs": 4,
                "request_options": {
                    "num_ctx": 32768,
                    "repeat_penalty": 1.15,
                    "presence_penalty": 0.1,
                    "min_p": None,
                },
            }
        },
    )
    request = LlmRequest(
        route_id="ollama_qwen35_27b",
        stage="design",
        system_prompt="system prompt",
        user_prompt="user prompt",
        seed=123,
        metadata={"organism_id": "org001"},
    )
    captured: dict[str, object] = {}

    class FakeHttpResponse:
        def __enter__(self) -> "FakeHttpResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"role": "assistant", "content": "generated text"}}).encode("utf-8")

    def fake_urlopen(http_request, timeout: float):
        captured["timeout"] = timeout
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return FakeHttpResponse()

    monkeypatch.setattr(provider_backends.urllib_request, "urlopen", fake_urlopen)

    response = generate_direct(route_cfg, request)

    assert captured["timeout"] == 30.0
    assert captured["body"] == {
        "model": "qwen3.5:27b",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "stream": False,
        "options": {
            "num_ctx": 32768,
            "repeat_penalty": 1.15,
            "presence_penalty": 0.1,
            "temperature": 0.2,
            "num_predict": 1536,
            "top_p": 0.92,
            "top_k": 12,
        },
        "think": "high",
        "keep_alive": "15m",
        "raw": True,
        "logprobs": True,
        "top_logprobs": 4,
    }
    assert response.raw_request == captured["body"]


def test_ollama_route_wraps_network_errors_with_request_context(monkeypatch: pytest.MonkeyPatch) -> None:
    route_cfg = ApiRouteConfig(
        route_id="ollama_qwen35_27b",
        provider="ollama",
        provider_model_id="qwen3.5:27b",
        backend="ollama",
        base_url="http://localhost:11435/api",
        timeout_sec=30.0,
    )
    request = LlmRequest(
        route_id="ollama_qwen35_27b",
        stage="design",
        system_prompt="system prompt",
        user_prompt="user prompt",
        seed=123,
        metadata={"organism_id": "org001"},
    )

    class FakeTagsResponse:
        status = 200

        def __enter__(self) -> "FakeTagsResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self, _size: int = -1) -> bytes:
            return b'{"models":[]}'

    def fake_urlopen(http_request, timeout: float):
        assert timeout in {30.0, 3.0}
        if http_request.full_url == "http://127.0.0.1:11435/api/chat":
            raise urllib_error.URLError(ConnectionRefusedError(111, "Connection refused"))
        if http_request.full_url == "http://127.0.0.1:11435/api/tags":
            return FakeTagsResponse()
        raise AssertionError(f"unexpected URL {http_request.full_url!r}")

    monkeypatch.setattr(provider_backends.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        generate_direct(route_cfg, request)

    message = str(excinfo.value)
    assert "route 'ollama_qwen35_27b'" in message
    assert "http://127.0.0.1:11435/api/chat" in message
    assert "stage='design'" in message
    assert "organism_id='org001'" in message
    assert "ConnectionRefusedError: [Errno 111] Connection refused" in message
    assert "tags_probe=url='http://127.0.0.1:11435/api/tags' ok=True status=200" in message
