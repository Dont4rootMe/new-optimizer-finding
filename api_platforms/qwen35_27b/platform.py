"""Local Qwen 3.5 27B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="qwen35_27b",
        provider="local_transformers",
        provider_model_id="Qwen/Qwen3.5-27B",
        backend="transformers",
        **kwargs,
    )
