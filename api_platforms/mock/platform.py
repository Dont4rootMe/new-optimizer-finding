"""Mock route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="mock",
        provider="mock",
        provider_model_id="mock-model",
        backend="mock",
        **kwargs,
    )
