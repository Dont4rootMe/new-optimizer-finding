"""Ollama Gemma 4 31B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="ollama_gemma4_31b",
        provider="ollama",
        provider_model_id="gemma4:31b",
        backend="ollama",
        **kwargs,
    )
