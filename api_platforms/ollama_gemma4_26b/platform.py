"""Ollama Gemma 4 26B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="ollama_gemma4_26b",
        provider="ollama",
        provider_model_id="gemma4:26b",
        backend="ollama",
        **kwargs,
    )
