"""Provider-specific generation backends used by direct brokers."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from api_platforms._core.types import ApiRouteConfig, LlmRequest, LlmResponse


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_mock_implementation(template: str, organism_id: str, generation: int, seed: int) -> str:
    base = seed + generation + sum(ord(ch) for ch in organism_id[:8])
    use_sgd = (base % 2) == 0
    opt_type = "SGD" if use_sgd else "AdamW"
    lr = "0.05" if use_sgd else "3e-4"
    class_name = f"Implementation_{organism_id[:8]}"
    return template.format(
        imports="import math\nfrom torch.optim.lr_scheduler import LambdaLR",
        optimizer_name=class_name,
        class_name=class_name,
        init_body=(
            "        self.model = model\n"
            "        self.max_steps = max_steps\n"
            "        self.named_parameters = [\n"
            "            (n, p) for n, p in model.named_parameters() if p.requires_grad\n"
            "        ]\n"
            "        params = [p for _, p in self.named_parameters]\n"
            f"        self.optimizer = torch.optim.{opt_type}(params, lr={lr})\n"
            "        warmup_steps = max(1, int(max_steps * 0.05))\n"
            "        def _lambda(step):\n"
            "            if step < warmup_steps:\n"
            "                return max(1e-8, float(step + 1) / float(warmup_steps))\n"
            "            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))\n"
            "            progress = min(max(progress, 0.0), 1.0)\n"
            "            return 0.5 * (1.0 + math.cos(math.pi * progress))\n"
            "        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lambda)"
        ),
        step_body=(
            "        del weights, grads, activations, step_fn\n"
            "        self.optimizer.step()\n"
            "        if self.scheduler is not None:\n"
            "            self.scheduler.step()"
        ),
        zero_grad_body="        self.optimizer.zero_grad(set_to_none=set_to_none)",
    )


def build_mock_text(request: LlmRequest) -> str:
    organism_id = str(request.metadata.get("organism_id", "mock"))
    generation = int(request.metadata.get("generation", 0))
    base = int(request.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
    use_sgd = (base % 2) == 0
    opt_type = "SGD" if use_sgd else "AdamW"
    if request.stage == "design":
        return (
            "## CORE_GENES\n"
            f"- mock {opt_type} optimizer with cosine schedule\n"
            "- short warmup before annealing\n"
            "- stable low-overhead update rule for testing\n\n"
            "## INTERACTION_NOTES\n"
            "Use a simple scheduler-driven optimizer with stable defaults for testing.\n\n"
            "## COMPUTE_NOTES\n"
            "Avoid expensive step_fn calls and keep per-step compute low.\n\n"
            "## CHANGE_DESCRIPTION\n"
            f"Mock {opt_type} optimizer with warmup and cosine annealing for testing.\n"
        )

    template = str(request.metadata.get("implementation_template", "")).strip()
    if template:
        return _render_mock_implementation(template, organism_id, generation, request.seed)

    return (
        "import torch.nn as nn\n\n"
        "OPTIMIZER_NAME = 'MockImplementation'\n\n"
        "class MockImplementation:\n"
        "    def __init__(self, model: nn.Module, max_steps: int) -> None:\n"
        "        self.model = model\n"
        "        self.max_steps = max_steps\n\n"
        "    def step(self, weights, grads, activations, step_fn) -> None:\n"
        "        del weights, grads, activations, step_fn\n\n"
        "    def zero_grad(self, set_to_none: bool = True) -> None:\n"
        "        del set_to_none\n\n"
        "def build_optimizer(model: nn.Module, max_steps: int):\n"
        "    return MockImplementation(model, max_steps)\n"
    )


def _usage_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {"raw": str(value)}


def _response_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {"raw": str(value)}


def _ollama_chat_url(base_url: str | None) -> str:
    normalized = str(base_url or "http://localhost:11434/api").rstrip("/")
    if normalized.endswith("/chat"):
        return normalized
    if normalized.endswith("/api"):
        return normalized + "/chat"
    return normalized + "/api/chat"


def _ollama_request_payload(route_cfg: ApiRouteConfig, request: LlmRequest) -> dict[str, Any]:
    options = dict(route_cfg.request_options)
    options.setdefault("temperature", float(route_cfg.temperature))
    options.setdefault("num_predict", int(route_cfg.max_output_tokens))
    if route_cfg.top_p is not None:
        options.setdefault("top_p", float(route_cfg.top_p))
    if route_cfg.top_k is not None:
        options.setdefault("top_k", int(route_cfg.top_k))

    payload: dict[str, Any] = {
        "model": route_cfg.provider_model_id,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        "stream": False,
        "options": options,
    }
    if route_cfg.think is not None:
        payload["think"] = route_cfg.think
    if route_cfg.keep_alive is not None:
        payload["keep_alive"] = route_cfg.keep_alive
    return payload


def _openai_output_text(response_payload: dict[str, Any]) -> str:
    text = ""
    for item in response_payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for chunk in item.get("content", []):
            if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                text += str(chunk.get("text", ""))
    return text


def generate_direct(route_cfg: ApiRouteConfig, request: LlmRequest) -> LlmResponse:
    """Run one unified request against a direct backend."""

    started_at = _utc_now_iso()

    if route_cfg.backend == "mock":
        if route_cfg.mock_delay_sec > 0:
            time.sleep(float(route_cfg.mock_delay_sec))
        text = build_mock_text(request)
        finished_at = _utc_now_iso()
        return LlmResponse(
            text=text,
            route_id=route_cfg.route_id,
            provider=route_cfg.provider,
            provider_model_id=route_cfg.provider_model_id,
            raw_request={"stage": request.stage, "metadata": request.metadata},
            raw_response={"stage": request.stage, "text": text},
            usage={},
            started_at=started_at,
            finished_at=finished_at,
        )

    if route_cfg.backend == "openai":
        api_key_env = route_cfg.api_key_env or "OPENAI_API_KEY"
        if not os.getenv(api_key_env):
            raise RuntimeError(f"{api_key_env} is not set for route '{route_cfg.route_id}'.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI SDK is not installed. Install with: pip install -e .[evolve]") from exc

        client_kwargs: dict[str, Any] = {}
        if route_cfg.base_url:
            client_kwargs["base_url"] = route_cfg.base_url
        client = OpenAI(api_key=os.getenv(api_key_env), **client_kwargs)
        request_payload: dict[str, Any] = {
            "model": route_cfg.provider_model_id,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "temperature": float(route_cfg.temperature),
            "max_output_tokens": int(route_cfg.max_output_tokens),
        }
        if route_cfg.reasoning_effort:
            request_payload["reasoning"] = {"effort": route_cfg.reasoning_effort}
        response = client.responses.create(**request_payload)
        response_payload = _response_to_dict(response)
        text = getattr(response, "output_text", None) or _openai_output_text(response_payload)
        if not text:
            raise RuntimeError("OpenAI response did not contain usable text output.")
        finished_at = _utc_now_iso()
        return LlmResponse(
            text=text,
            route_id=route_cfg.route_id,
            provider=route_cfg.provider,
            provider_model_id=route_cfg.provider_model_id,
            raw_request=request_payload,
            raw_response=response_payload,
            usage=_usage_dict(response_payload.get("usage")),
            started_at=started_at,
            finished_at=finished_at,
        )

    if route_cfg.backend == "anthropic":
        api_key_env = route_cfg.api_key_env or "ANTHROPIC_API_KEY"
        if not os.getenv(api_key_env):
            raise RuntimeError(f"{api_key_env} is not set for route '{route_cfg.route_id}'.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic SDK is not installed. Install with: pip install -e .[evolve]") from exc

        client = Anthropic(api_key=os.getenv(api_key_env), base_url=route_cfg.base_url or None)
        request_payload = {
            "model": route_cfg.provider_model_id,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "max_tokens": int(route_cfg.max_output_tokens),
            "temperature": float(route_cfg.temperature),
        }
        if route_cfg.thinking_budget_tokens is not None:
            request_payload["thinking"] = {"type": "enabled", "budget_tokens": int(route_cfg.thinking_budget_tokens)}
        response = client.messages.create(**request_payload)
        response_payload = _response_to_dict(response)
        chunks = []
        for item in response_payload.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        text = "".join(chunks).strip()
        if not text:
            raise RuntimeError("Anthropic response did not contain usable text output.")
        finished_at = _utc_now_iso()
        return LlmResponse(
            text=text,
            route_id=route_cfg.route_id,
            provider=route_cfg.provider,
            provider_model_id=route_cfg.provider_model_id,
            raw_request=request_payload,
            raw_response=response_payload,
            usage=_usage_dict(response_payload.get("usage")),
            started_at=started_at,
            finished_at=finished_at,
        )

    if route_cfg.backend == "ollama":
        import json

        request_payload = _ollama_request_payload(route_cfg, request)
        encoded = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key_env = str(route_cfg.api_key_env or "").strip()
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        http_request = urllib_request.Request(
            _ollama_chat_url(route_cfg.base_url),
            data=encoded,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=float(route_cfg.timeout_sec)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed for route '{route_cfg.route_id}' with status {exc.code}: {error_body}"
            ) from exc

        message = response_payload.get("message", {})
        if not isinstance(message, dict):
            message = {}
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError("Ollama response did not contain usable message.content text output.")

        usage = {
            key: response_payload.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if key in response_payload
        }
        finished_at = _utc_now_iso()
        return LlmResponse(
            text=text,
            route_id=route_cfg.route_id,
            provider=route_cfg.provider,
            provider_model_id=route_cfg.provider_model_id,
            raw_request=request_payload,
            raw_response=response_payload,
            usage=usage,
            started_at=started_at,
            finished_at=finished_at,
        )

    raise ValueError(f"Unsupported direct backend '{route_cfg.backend}'.")
