"""Helpers for building, normalizing, and hashing route configurations."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence
import json
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from omegaconf import OmegaConf

from api_platforms._core.types import ApiRouteConfig


def normalize_gpu_ranks(value: Any) -> list[int]:
    """Normalize an int-or-list config field into a stable list of GPU ranks."""

    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(rank) for rank in value]
    raise TypeError(f"gpu_ranks must be null, int, or list[int], got {type(value).__name__}")


def _is_sequence_payload(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def normalize_route_gpu_layout(value: Any, *, backend: str) -> tuple[list[int], list[list[int]]]:
    """Normalize route GPU config into flattened ranks plus instance groups.

    Semantics:
    - `int` -> one instance on one GPU
    - `list[int]` -> one multi-GPU Ollama instance, or one GPU per worker for other backends
    - `list[list[int]]` -> multiple Ollama instances, one per inner GPU group
    """

    groups: list[list[int]] = []
    if value is None:
        return [], groups

    if isinstance(value, int):
        groups = [[int(value)]]
    elif _is_sequence_payload(value):
        items = list(value)
        if not items:
            groups = []
        elif all(isinstance(item, int) for item in items):
            ranks = [int(item) for item in items]
            if backend == "ollama":
                groups = [ranks]
            else:
                groups = [[rank] for rank in ranks]
        elif all(_is_sequence_payload(item) for item in items):
            parsed_groups: list[list[int]] = []
            for group in items:
                parsed = [int(rank) for rank in group]
                if not parsed:
                    raise ValueError("gpu_ranks must not contain empty rank groups")
                parsed_groups.append(parsed)
            if backend != "ollama" and any(len(group) != 1 for group in parsed_groups):
                raise ValueError(
                    f"Only ollama routes support multi-GPU grouped gpu_ranks; backend '{backend}' "
                    f"must use int, list[int], or list[list[int]] with singleton groups."
                )
            groups = parsed_groups
        else:
            raise TypeError(
                "gpu_ranks must be null, int, list[int], or list[list[int]] "
                f"for backend '{backend}', got mixed sequence payload"
            )
    else:
        raise TypeError(
            "gpu_ranks must be null, int, list[int], or list[list[int]] "
            f"for backend '{backend}', got {type(value).__name__}"
        )

    flattened: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for rank in group:
            if rank in seen:
                raise ValueError("gpu_ranks must not contain duplicate ranks")
            flattened.append(rank)
            seen.add(rank)
    return flattened, groups


def _normalize_ollama_base_url(base_url: str | None) -> SplitResult:
    value = str(base_url or "http://127.0.0.1:11434/api").strip() or "http://127.0.0.1:11434/api"
    parsed = urlsplit(value)
    if not parsed.scheme:
        parsed = urlsplit(f"http://{value}")
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        netloc_host = "127.0.0.1"
    else:
        netloc_host = host
    port = parsed.port
    if port is not None:
        netloc = f"{netloc_host}:{port}"
    else:
        netloc = netloc_host
    return SplitResult(parsed.scheme or "http", netloc, parsed.path or "/api", parsed.query, parsed.fragment)


def derive_ollama_instance_configs(route_cfg: ApiRouteConfig) -> list[ApiRouteConfig]:
    """Return concrete instance configs for one logical Ollama route."""

    if route_cfg.backend != "ollama":
        raise ValueError(f"derive_ollama_instance_configs only supports ollama routes, got {route_cfg.backend!r}")

    groups = [list(group) for group in route_cfg.gpu_rank_groups]
    if not groups and route_cfg.gpu_ranks:
        groups = [list(route_cfg.gpu_ranks)]
    if not groups:
        return [replace(route_cfg)]

    parsed_base = _normalize_ollama_base_url(route_cfg.base_url)
    if len(groups) > 1 and parsed_base.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            f"Ollama route '{route_cfg.route_id}' uses grouped gpu_ranks but base_url={route_cfg.base_url!r} "
            "is not a local localhost/127.0.0.1 endpoint."
        )

    base_port = parsed_base.port
    if len(groups) > 1 and base_port is None:
        raise ValueError(
            f"Ollama route '{route_cfg.route_id}' must use an explicit port when gpu_ranks has multiple groups."
        )

    instances: list[ApiRouteConfig] = []
    for index, group in enumerate(groups):
        netloc = parsed_base.netloc
        if base_port is not None:
            host = parsed_base.hostname or "127.0.0.1"
            if host == "localhost":
                host = "127.0.0.1"
            netloc = f"{host}:{base_port + index}"
        base_url = urlunsplit((parsed_base.scheme, netloc, parsed_base.path, parsed_base.query, parsed_base.fragment))
        instances.append(
            replace(
                route_cfg,
                base_url=base_url,
                gpu_ranks=list(group),
                gpu_rank_groups=[list(group)],
            )
        )
    return instances


def build_route_config(
    *,
    route_id: str,
    provider: str,
    provider_model_id: str,
    backend: str,
    **kwargs: Any,
) -> ApiRouteConfig:
    """Build one canonical `ApiRouteConfig` from route-specific defaults and overrides."""

    payload = dict(kwargs)
    payload["route_id"] = route_id
    payload["provider"] = provider
    payload["provider_model_id"] = provider_model_id
    payload["backend"] = backend
    gpu_ranks, gpu_rank_groups = normalize_route_gpu_layout(payload.get("gpu_ranks"), backend=backend)
    payload["gpu_ranks"] = gpu_ranks
    payload["gpu_rank_groups"] = gpu_rank_groups
    if "max_concurrency" not in payload and backend in {"transformers", "mock_local"}:
        payload["max_concurrency"] = max(1, len(payload["gpu_ranks"]) or 1)
    if "request_options" not in payload or payload["request_options"] is None:
        payload["request_options"] = {}
    if OmegaConf.is_config(payload["request_options"]):
        payload["request_options"] = OmegaConf.to_container(payload["request_options"], resolve=True)
    else:
        payload["request_options"] = dict(payload["request_options"])
    if "stage_options" not in payload or payload["stage_options"] is None:
        payload["stage_options"] = {}
    if OmegaConf.is_config(payload["stage_options"]):
        payload["stage_options"] = OmegaConf.to_container(payload["stage_options"], resolve=True)
    normalized_stage_options: dict[str, dict[str, Any]] = {}
    for stage_name, stage_cfg in dict(payload["stage_options"]).items():
        if stage_cfg is None:
            normalized_stage_options[str(stage_name)] = {}
            continue
        if OmegaConf.is_config(stage_cfg):
            stage_payload = OmegaConf.to_container(stage_cfg, resolve=True)
        else:
            stage_payload = dict(stage_cfg)
        if not isinstance(stage_payload, dict):
            raise TypeError(
                f"stage_options.{stage_name} must be a mapping, got {type(stage_payload).__name__}"
            )
        request_options = stage_payload.get("request_options")
        if request_options is None:
            stage_payload["request_options"] = {}
        elif OmegaConf.is_config(request_options):
            stage_payload["request_options"] = OmegaConf.to_container(request_options, resolve=True)
        else:
            stage_payload["request_options"] = dict(request_options)
        normalized_stage_options[str(stage_name)] = stage_payload
    payload["stage_options"] = normalized_stage_options
    return ApiRouteConfig(**payload)


def stable_config_hash(route_cfg: ApiRouteConfig) -> str:
    """Return a stable hash for route runtime identity."""

    canonical = json.dumps(route_cfg.to_dict(), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
