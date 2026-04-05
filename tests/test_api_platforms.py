"""Contract tests for api_platforms routing, broker reuse, and local concurrency."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from api_platforms import ApiPlatformRegistry, LlmRequest
from api_platforms._core import providers as provider_backends
from api_platforms._core.providers import generate_direct
from api_platforms._core.types import ApiRouteConfig
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
                "max_generation_attempts": 1,
                "prompts": {
                    "project_context": "conf/experiments/optimization_survey/prompts/shared/project_context.txt",
                    "seed_system": "conf/experiments/optimization_survey/prompts/seed/system.txt",
                    "seed_user": "conf/experiments/optimization_survey/prompts/seed/user.txt",
                    "mutation_system": "conf/experiments/optimization_survey/prompts/mutation/system.txt",
                    "mutation_user": "conf/experiments/optimization_survey/prompts/mutation/user.txt",
                    "crossover_system": "conf/experiments/optimization_survey/prompts/crossover/system.txt",
                    "crossover_user": "conf/experiments/optimization_survey/prompts/crossover/user.txt",
                    "implementation_system": "conf/experiments/optimization_survey/prompts/implementation/system.txt",
                    "implementation_user": "conf/experiments/optimization_survey/prompts/implementation/user.txt",
                    "implementation_template": "conf/experiments/optimization_survey/prompts/implementation/template.txt",
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

    assert captured["url"] == "http://localhost:11434/api/chat"
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
