"""OpenAI GPT-5.4 mini route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="gpt_5_4_mini",
        provider="openai",
        provider_model_id="gpt-5.4-mini",
        backend="openai",
        **kwargs,
    )
