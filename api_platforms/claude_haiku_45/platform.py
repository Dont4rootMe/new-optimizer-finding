"""Anthropic Claude Haiku 4.5 route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="claude_haiku_45",
        provider="anthropic",
        provider_model_id="claude-haiku-4.5",
        backend="anthropic",
        **kwargs,
    )
