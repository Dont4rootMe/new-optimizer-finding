"""Hydra-backed discovery for enabled API platform routes."""

from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig

from api_platforms._core.types import ApiRouteConfig


def load_route_configs(cfg: DictConfig) -> dict[str, ApiRouteConfig]:
    """Instantiate every enabled route from `cfg.api_platforms`."""

    payload = getattr(cfg, "api_platforms", None)
    if payload is None:
        return {}

    routes: dict[str, ApiRouteConfig] = {}
    for route_id in payload.keys():
        route_cfg = instantiate(payload[route_id], _recursive_=False)
        if not isinstance(route_cfg, ApiRouteConfig):
            raise TypeError(
                f"api_platforms.{route_id} did not instantiate into ApiRouteConfig: "
                f"{type(route_cfg).__name__}"
            )
        if route_cfg.route_id != route_id:
            raise ValueError(
                f"api_platforms.{route_id} instantiated route_id={route_cfg.route_id!r}; expected {route_id!r}"
            )
        routes[route_id] = route_cfg
    return routes
