"""OpenAI GPT-5.4 route factory."""

from api_platforms._core.config import build_route_config


def build_platform(**kwargs):
    return build_route_config(
        route_id="gpt_5_4",
        provider="openai",
        provider_model_id="gpt-5.4",
        backend="openai",
        **kwargs,
    )
