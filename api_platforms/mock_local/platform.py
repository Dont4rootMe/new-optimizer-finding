"""Mock local-worker route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="mock_local",
        provider="mock_local",
        provider_model_id="mock-local-model",
        backend="mock_local",
        **kwargs,
    )
