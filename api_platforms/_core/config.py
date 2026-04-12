"""Helpers for building and hashing route configurations."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

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
    payload["gpu_ranks"] = normalize_gpu_ranks(payload.get("gpu_ranks"))
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
