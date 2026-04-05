"""Ollama Qwen 3.5 27B route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="ollama_qwen35_27b",
        provider="ollama",
        provider_model_id="qwen3.5:27b",
        backend="ollama",
        **kwargs,
    )
