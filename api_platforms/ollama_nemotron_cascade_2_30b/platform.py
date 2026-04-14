"""Ollama Nemotron Cascade 2 30B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="ollama_nemotron_cascade_2_30b",
        provider="ollama",
        provider_model_id="nemotron-cascade-2:30b",
        backend="ollama",
        **kwargs,
    )
