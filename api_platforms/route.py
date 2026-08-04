"""Single Hydra factory for every LLM route.

Model identity belongs in configuration.  Keeping one Python package per model
made adding or comparing backbones needlessly expensive and scattered provider
metadata across dozens of identical wrappers.  Existing wrappers remain import
compatible, while new configs should target :func:`build_platform` directly.
"""

from __future__ import annotations

from typing import Any

from api_platforms._core.config import build_route_config
from api_platforms._core.types import ApiRouteConfig


def build_platform(
    *,
    route_id: str,
    provider: str,
    provider_model_id: str,
    backend: str,
    **kwargs: Any,
) -> ApiRouteConfig:
    """Build one route from explicit, configuration-owned identity fields."""

    return build_route_config(
        route_id=route_id,
        provider=provider,
        provider_model_id=provider_model_id,
        backend=backend,
        **kwargs,
    )
