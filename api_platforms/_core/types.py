"""Core public contracts for route-agnostic LLM generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ApiRouteConfig:
    """Fully materialized route configuration for one enabled LLM platform."""

    route_id: str
    provider: str
    provider_model_id: str
    backend: str
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1024
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None
    timeout_sec: float = 300.0
    max_retries: int = 0
    max_concurrency: int = 1
    model_name_or_path: str | None = None
    tokenizer_name: str | None = None
    gpu_ranks: list[int] = field(default_factory=list)
    torch_dtype: str | None = None
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    top_p: float | None = None
    top_k: int | None = None
    do_sample: bool = True
    request_options: dict[str, Any] = field(default_factory=dict)
    stage_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    think: str | bool | None = None
    keep_alive: str | int | None = None
    raw: bool | None = None
    format: str | dict[str, Any] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    mock_delay_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ApiRouteConfig":
        return cls(**payload)


@dataclass(slots=True)
class LlmRequest:
    """Unified request payload sent to a route broker."""

    route_id: str
    stage: str
    system_prompt: str
    user_prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlmRequest":
        return cls(
            route_id=str(payload["route_id"]),
            stage=str(payload["stage"]),
            system_prompt=str(payload["system_prompt"]),
            user_prompt=str(payload["user_prompt"]),
            seed=int(payload.get("seed", 0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class LlmResponse:
    """Unified LLM response returned by any route broker."""

    text: str
    route_id: str
    provider: str
    provider_model_id: str
    raw_request: dict[str, Any]
    raw_response: dict[str, Any]
    usage: dict[str, Any]
    started_at: str
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LlmResponse":
        return cls(
            text=str(payload["text"]),
            route_id=str(payload["route_id"]),
            provider=str(payload["provider"]),
            provider_model_id=str(payload["provider_model_id"]),
            raw_request=dict(payload.get("raw_request", {})),
            raw_response=dict(payload.get("raw_response", {})),
            usage=dict(payload.get("usage", {})),
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]),
        )


class ApiPlatformClient(Protocol):
    """Client contract for issuing unified generation requests."""

    def generate(self, request: LlmRequest) -> LlmResponse:
        """Return a route-specific LLM response for one request."""


class ApiPlatformBroker(Protocol):
    """Broker lifecycle contract for one route."""

    def start(self) -> None:
        """Initialize route resources and prepare to serve requests."""

    def stop(self) -> None:
        """Release route resources and stop accepting requests."""

    def serve_forever(self) -> None:
        """Block until shutdown while serving IPC requests."""
