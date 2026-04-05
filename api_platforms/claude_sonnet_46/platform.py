"""Anthropic Claude Sonnet 4.6 route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="claude_sonnet_46",
        provider="anthropic",
        provider_model_id="claude-sonnet-4.6",
        backend="anthropic",
        **kwargs,
    )
