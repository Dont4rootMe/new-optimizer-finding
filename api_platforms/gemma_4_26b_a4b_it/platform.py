"""Local Gemma 4 26B A4B IT route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="gemma_4_26b_a4b_it",
        provider="local_transformers",
        provider_model_id="google/gemma-4-26B-A4B-it",
        backend="transformers",
        **kwargs,
    )
