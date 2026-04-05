"""Local Qwen 3.5 27B Claude 4.6 Opus distilled route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="qwen35_27b_claude46_opus_distilled",
        provider="local_transformers",
        provider_model_id="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
        backend="transformers",
        **kwargs,
    )
