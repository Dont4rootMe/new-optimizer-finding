"""Local Qwen 3.5 35B A3B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="qwen35_35b_a3b",
        provider="local_transformers",
        provider_model_id="Qwen/Qwen3.5-35B-A3B",
        backend="transformers",
        **kwargs,
    )
