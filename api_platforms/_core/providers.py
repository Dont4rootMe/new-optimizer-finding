"""Provider-specific generation backends used by direct brokers."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from api_platforms._core.types import ApiRouteConfig, LlmRequest, LlmResponse

LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_mock_implementation(template: str, organism_id: str, generation: int, seed: int) -> str:
    if "def run_packing():" in template:
        return template.format(
            imports="",
            helpers=(
                "def _build_centers():\n"
                "    rows = [\n"
                "        (0.15, [0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94]),\n"
                "        (0.33, [0.17, 0.31, 0.45, 0.59, 0.73, 0.87]),\n"
                "        (0.51, [0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94]),\n"
                "        (0.69, [0.17, 0.31, 0.45, 0.59, 0.73, 0.87]),\n"
                "    ]\n"
                "    centers = []\n"
                "    for y_coord, xs in rows:\n"
                "        for x_coord in xs:\n"
                "            centers.append((x_coord, y_coord))\n"
                "    return np.asarray(centers, dtype=float)\n"
            ),
            run_packing_body=(
                "    centers = _build_centers()\n"
                "    radii = np.full(26, 0.04, dtype=float)\n"
                "    reported_sum = float(np.sum(radii))\n"
                "    return centers, radii, reported_sum"
            ),
        )
    if "def solve_case(input_text: str) -> str:" in template:
        return template.format(
            imports="",
            helpers=(
                "def _build_empty_solution(input_text: str) -> str:\n"
                "    lines = [line.strip() for line in input_text.splitlines() if line.strip()]\n"
                "    if not lines:\n"
                "        raise ValueError('input_text must not be empty.')\n"
                "    n, k = map(int, lines[0].split())\n"
                "    vertical = ['0' * (n - 1) for _ in range(n)]\n"
                "    horizontal = ['0' * n for _ in range(n - 1)]\n"
                "    groups = [str(i) for i in range(k)]\n"
                "    return '\\n'.join(vertical + horizontal + groups)\n"
            ),
            solve_case_body="    return _build_empty_solution(input_text)",
        )

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
    joined_prompt = f"{request.system_prompt}\n{request.user_prompt}".lower()
    template = str(request.metadata.get("implementation_template", "")).strip()
    if request.stage == "novelty_check":
        return "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n"
    if "circle-packing" in joined_prompt or "circle packing" in joined_prompt or "run_packing" in joined_prompt:
        if request.stage == "design":
            return (
                "## CORE_GENES\n"
                "- Deterministic staggered-row layout with alternating long and short rows inside the unit square\n"
                "- Uniform radius assignment chosen conservatively so border constraints and pairwise distances remain valid\n"
                "- Geometry-first construction that computes centers from explicit row templates instead of random search\n\n"
                "## INTERACTION_NOTES\n"
                "This design favors stable valid packings with simple geometry and should work well as a seed for later refinements.\n\n"
                "## COMPUTE_NOTES\n"
                "The method is purely constructive, uses O(n) memory, and performs no iterative repair loops.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "A deterministic staggered packing program that starts from a valid geometric template for 26 circles.\n"
            )
        if template:
            return _render_mock_implementation(template, organism_id, generation, request.seed)
    if (
        "group commands and wall planning" in joined_prompt
        or "awtf2025" in joined_prompt
        or "solve_case(input_text: str)" in joined_prompt
    ):
        if request.stage == "design":
            return (
                "## CORE_GENES\n"
                "- Deterministic empty-wall baseline that preserves the original traversable board and avoids accidental traps\n"
                "- Identity group assignment so synchronized moves cannot create cross-robot interference in the baseline organism\n"
                "- Output-construction strategy that always emits a legal no-op plan before later generations add smarter routing logic\n\n"
                "## INTERACTION_NOTES\n"
                "This design prioritizes legality and deterministic control over absolute score quality, which makes it a safe seed for later routing mutations.\n\n"
                "## COMPUTE_NOTES\n"
                "The method performs only lightweight input parsing and string construction, with no simulation or search inside the candidate.\n\n"
                "## CHANGE_DESCRIPTION\n"
                "A deterministic legal-output seed that keeps the board unchanged and isolates control decisions so later generations can add structure safely.\n"
            )
        if template:
            return _render_mock_implementation(template, organism_id, generation, request.seed)

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
    normalized = str(base_url or "http://127.0.0.1:11434/api").rstrip("/")
    parsed = urllib_parse.urlparse(normalized)
    if parsed.hostname == "localhost":
        netloc = parsed.netloc.replace("localhost", "127.0.0.1", 1)
        normalized = urllib_parse.urlunparse(parsed._replace(netloc=netloc))
    if normalized.endswith("/chat"):
        return normalized
    if normalized.endswith("/api"):
        return normalized + "/chat"
    return normalized + "/api/chat"


def _ollama_tags_url(base_url: str | None) -> str:
    normalized = str(base_url or "http://127.0.0.1:11434/api").rstrip("/")
    parsed = urllib_parse.urlparse(normalized)
    if parsed.hostname == "localhost":
        netloc = parsed.netloc.replace("localhost", "127.0.0.1", 1)
        normalized = urllib_parse.urlunparse(parsed._replace(netloc=netloc))
    if normalized.endswith("/chat"):
        return normalized[: -len("/chat")] + "/tags"
    if normalized.endswith("/api"):
        return normalized + "/tags"
    return normalized + "/api/tags"


def _single_line(value: str, *, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _ollama_response_summary(response_payload: dict[str, Any]) -> str:
    message = response_payload.get("message", {})
    if not isinstance(message, dict):
        message = {}
    content_text = str(message.get("content", "") or "").strip()
    thinking_text = str(message.get("thinking", "") or "").strip()
    parts = [
        f"content_chars={len(content_text)}",
        f"thinking_chars={len(thinking_text)}",
    ]
    done_reason = str(response_payload.get("done_reason", "") or "").strip()
    if done_reason:
        parts.append(f"done_reason={done_reason!r}")
    for key in (
        "prompt_eval_count",
        "eval_count",
        "total_duration",
        "prompt_eval_duration",
        "eval_duration",
    ):
        value = response_payload.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _ollama_probe_summary(base_url: str | None, timeout_sec: float) -> str:
    probe_url = _ollama_tags_url(base_url)
    probe_timeout = min(3.0, max(1.0, float(timeout_sec)))
    probe_request = urllib_request.Request(probe_url, method="GET")
    try:
        with urllib_request.urlopen(probe_request, timeout=probe_timeout) as response:
            status = getattr(response, "status", None)
            body_prefix = _single_line(response.read(160).decode("utf-8", errors="replace"))
        if body_prefix:
            return f"url={probe_url!r} ok=True status={status} body_prefix={body_prefix!r}"
        return f"url={probe_url!r} ok=True status={status}"
    except urllib_error.HTTPError as exc:
        error_body = _single_line(exc.read(160).decode("utf-8", errors="replace"))
        if error_body:
            return f"url={probe_url!r} ok=False status={exc.code} body_prefix={error_body!r}"
        return f"url={probe_url!r} ok=False status={exc.code}"
    except urllib_error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"url={probe_url!r} ok=False error={type(reason).__name__}: {reason}"
    except Exception as exc:  # noqa: BLE001
        return f"url={probe_url!r} ok=False error={type(exc).__name__}: {exc}"


def _drop_none_entries(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none_entries(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none_entries(item) for item in value if item is not None]
    return value


def _ollama_stage_effective_config(route_cfg: ApiRouteConfig, stage: str) -> dict[str, Any]:
    stage_cfg = route_cfg.stage_options.get(stage, {})
    if not isinstance(stage_cfg, dict):
        stage_cfg = {}

    def _stage_override(key: str, default: Any) -> Any:
        value = stage_cfg.get(key, default)
        return default if value is None else value

    request_options = dict(route_cfg.request_options)
    override_request_options = stage_cfg.get("request_options", {})
    if isinstance(override_request_options, dict):
        request_options.update({key: value for key, value in override_request_options.items()})

    return {
        "temperature": _stage_override("temperature", route_cfg.temperature),
        "max_output_tokens": _stage_override("max_output_tokens", route_cfg.max_output_tokens),
        "top_p": _stage_override("top_p", route_cfg.top_p),
        "top_k": _stage_override("top_k", route_cfg.top_k),
        "think": _stage_override("think", route_cfg.think),
        "keep_alive": _stage_override("keep_alive", route_cfg.keep_alive),
        "raw": _stage_override("raw", route_cfg.raw),
        "format": _stage_override("format", route_cfg.format),
        "logprobs": _stage_override("logprobs", route_cfg.logprobs),
        "top_logprobs": _stage_override("top_logprobs", route_cfg.top_logprobs),
        "request_options": request_options,
    }


def _ollama_request_payload(route_cfg: ApiRouteConfig, request: LlmRequest) -> dict[str, Any]:
    effective = _ollama_stage_effective_config(route_cfg, request.stage)

    options = dict(effective["request_options"])
    options.setdefault("temperature", float(effective["temperature"]))
    options.setdefault("num_predict", int(effective["max_output_tokens"]))
    if effective["top_p"] is not None:
        options.setdefault("top_p", float(effective["top_p"]))
    if effective["top_k"] is not None:
        options.setdefault("top_k", int(effective["top_k"]))

    payload: dict[str, Any] = {
        "model": route_cfg.provider_model_id,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        "stream": False,
        "options": _drop_none_entries(options),
    }
    if effective["think"] is not None:
        payload["think"] = effective["think"]
    if effective["keep_alive"] is not None:
        payload["keep_alive"] = effective["keep_alive"]
    if effective["raw"] is not None:
        payload["raw"] = bool(effective["raw"])
    if effective["format"] is not None:
        payload["format"] = effective["format"]
    if effective["logprobs"] is not None:
        payload["logprobs"] = bool(effective["logprobs"])
    if effective["top_logprobs"] is not None:
        payload["top_logprobs"] = int(effective["top_logprobs"])
    return _drop_none_entries(payload)


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
        request_url = _ollama_chat_url(route_cfg.base_url)
        encoded = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key_env = str(route_cfg.api_key_env or "").strip()
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        http_request = urllib_request.Request(
            request_url,
            data=encoded,
            headers=headers,
            method="POST",
        )
        organism_id = str(request.metadata.get("organism_id", ""))
        timeout_sec = float(route_cfg.timeout_sec)
        try:
            with urllib_request.urlopen(http_request, timeout=timeout_sec) as response:
                raw_response = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            LOGGER.exception(
                "Ollama HTTP error route=%s url=%s stage=%s organism_id=%s timeout=%s",
                route_cfg.route_id,
                request_url,
                request.stage,
                organism_id or "<missing>",
                timeout_sec,
            )
            raise RuntimeError(
                f"Ollama request failed for route '{route_cfg.route_id}' at {request_url!r} "
                f"(stage={request.stage!r}, organism_id={organism_id!r}, timeout={timeout_sec}): "
                f"HTTP {exc.code}: {_single_line(error_body)!r}"
            ) from exc
        except urllib_error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            probe_summary = _ollama_probe_summary(route_cfg.base_url, timeout_sec)
            LOGGER.exception(
                "Ollama network error route=%s url=%s stage=%s organism_id=%s timeout=%s probe={%s}",
                route_cfg.route_id,
                request_url,
                request.stage,
                organism_id or "<missing>",
                timeout_sec,
                probe_summary,
            )
            raise RuntimeError(
                f"Ollama request failed for route '{route_cfg.route_id}' at {request_url!r} "
                f"(stage={request.stage!r}, organism_id={organism_id!r}, timeout={timeout_sec}): "
                f"{type(reason).__name__}: {reason}; tags_probe={probe_summary}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            probe_summary = _ollama_probe_summary(route_cfg.base_url, timeout_sec)
            LOGGER.exception(
                "Ollama unexpected error route=%s url=%s stage=%s organism_id=%s timeout=%s probe={%s}",
                route_cfg.route_id,
                request_url,
                request.stage,
                organism_id or "<missing>",
                timeout_sec,
                probe_summary,
            )
            raise RuntimeError(
                f"Ollama request failed for route '{route_cfg.route_id}' at {request_url!r} "
                f"(stage={request.stage!r}, organism_id={organism_id!r}, timeout={timeout_sec}): "
                f"{type(exc).__name__}: {exc}; tags_probe={probe_summary}"
            ) from exc

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            LOGGER.exception(
                "Ollama returned non-JSON route=%s url=%s stage=%s organism_id=%s timeout=%s",
                route_cfg.route_id,
                request_url,
                request.stage,
                organism_id or "<missing>",
                timeout_sec,
            )
            raise RuntimeError(
                f"Ollama request for route '{route_cfg.route_id}' at {request_url!r} "
                f"returned non-JSON response (stage={request.stage!r}, organism_id={organism_id!r}): "
                f"{_single_line(raw_response)!r}"
            ) from exc

        message = response_payload.get("message", {})
        if not isinstance(message, dict):
            message = {}
        content_text = str(message.get("content", "")).strip()
        thinking_text = str(message.get("thinking", "")).strip()

        done_reason = str(response_payload.get("done_reason", "")).strip() or None

        if not content_text and thinking_text and done_reason == "length":
            LOGGER.warning(
                "Ollama truncated thinking-only response route=%s stage=%s organism_id=%s %s thinking_preview=%r",
                route_cfg.route_id,
                request.stage,
                organism_id or "<missing>",
                _ollama_response_summary(response_payload),
                _single_line(thinking_text, limit=160),
            )
            raise RuntimeError(
                "Ollama response exhausted reasoning budget before final answer "
                f"(route='{route_cfg.route_id}', stage={request.stage!r}, "
                f"organism_id={organism_id!r}, done_reason='length')."
            )

        # Build the usable text from whatever Ollama returned.
        # When think mode is enabled, some models put all output into the
        # thinking field and leave content empty. We keep accepting that only
        # when the response terminated normally; truncated thinking-only output
        # is treated as an error above.
        if content_text and thinking_text:
            text = thinking_text + "\n\n" + content_text
        elif content_text:
            text = content_text
        elif thinking_text:
            text = thinking_text
        else:
            LOGGER.warning(
                "Ollama response missing usable text route=%s stage=%s organism_id=%s %s",
                route_cfg.route_id,
                request.stage,
                organism_id or "<missing>",
                _ollama_response_summary(response_payload),
            )
            raise RuntimeError("Ollama response did not contain usable message.content text output.")

        LOGGER.info(
            "Ollama response route=%s stage=%s organism_id=%s %s",
            route_cfg.route_id,
            request.stage,
            organism_id or "<missing>",
            _ollama_response_summary(response_payload),
        )

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
