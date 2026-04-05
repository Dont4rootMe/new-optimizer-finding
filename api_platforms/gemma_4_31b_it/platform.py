"""Local Gemma 4 31B IT route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="gemma_4_31b_it",
        provider="local_transformers",
        provider_model_id="google/gemma-4-31B-it",
        backend="transformers",
        **kwargs,
    )
