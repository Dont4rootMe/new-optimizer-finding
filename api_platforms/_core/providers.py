"""Provider-specific generation backends used by direct brokers."""

from __future__ import annotations

import logging
import os
import re
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


def _mock_circle_region_body(region_name: str) -> str:
    if region_name == "INIT_GEOMETRY":
        return (
            "    centers = np.asarray([\n"
            "        [0.10, 0.15], [0.24, 0.15], [0.38, 0.15], [0.52, 0.15], [0.66, 0.15], [0.80, 0.15], [0.94, 0.15],\n"
            "        [0.17, 0.33], [0.31, 0.33], [0.45, 0.33], [0.59, 0.33], [0.73, 0.33], [0.87, 0.33],\n"
            "        [0.10, 0.51], [0.24, 0.51], [0.38, 0.51], [0.52, 0.51], [0.66, 0.51], [0.80, 0.51], [0.94, 0.51],\n"
            "        [0.17, 0.69], [0.31, 0.69], [0.45, 0.69], [0.59, 0.69], [0.73, 0.69], [0.87, 0.69],\n"
            "    ], dtype=float)\n"
        )
    if region_name == "RADIUS_POLICY":
        return "    radii = np.full(26, 0.04, dtype=float)\n"
    return f"    # {region_name} is already satisfied by the deterministic constructive baseline.\n"


def _mock_optimizer_region_body(region_name: str) -> str:
    if region_name == "STATE_REPRESENTATION":
        return "        lr = 1e-3\n        max_grad_norm = 1.0\n"
    if region_name == "GRADIENT_PROCESSING":
        return (
            "        processed_params = []\n"
            "        for param in params:\n"
            "            if param.grad is None:\n"
            "                continue\n"
            "            grad = param.grad.detach()\n"
            "            processed_params.append((param, grad))\n"
        )
    if region_name == "UPDATE_RULE":
        return (
            "        with torch.no_grad():\n"
            "            for param, grad in processed_params:\n"
            "                param.add_(grad, alpha=-lr)\n"
        )
    if region_name == "STABILITY_POLICY":
        return (
            "        if max_grad_norm > 0:\n"
            "            for param, grad in processed_params:\n"
            "                norm = torch.linalg.vector_norm(grad)\n"
            "                if torch.isfinite(norm) and norm > max_grad_norm:\n"
            "                    param.add_(grad * (max_grad_norm / (norm + 1e-12)) - grad, alpha=0.0)\n"
        )
    return f"        # {region_name} uses the deterministic baseline optimizer behavior.\n"


def _mock_awtf_region_body(region_name: str) -> str:
    if region_name == "STATE_REPRESENTATION":
        return "    parsed_state = {'n': n, 'k': k, 'line_count': len(lines)}\n"
    if region_name == "MACRO_STRATEGY":
        return "    del parsed_state\n"
    if region_name == "CONSTRUCTION_POLICY":
        return "    operations = []\n"
    return f"    # {region_name} keeps the legal no-op baseline unchanged.\n"


def _extract_schema_sections_from_prompt(prompt: str) -> tuple[str, ...]:
    return tuple(re.findall(r"^# ([A-Z][A-Z0-9_]*)$", prompt, flags=re.MULTILINE))


def _extract_template_region_names(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"# === REGION: ([A-Z][A-Z0-9_]*) ===", text))


def _render_mock_sectioned_design(prompt: str) -> str | None:
    section_names = _extract_schema_sections_from_prompt(prompt)
    if not section_names:
        return None

    if "circle-packing" in prompt.lower() or "circle packing" in prompt.lower() or "run_packing" in prompt:
        subject = "circle-packing"
        score_goal = "packing score"
    elif "awtf2025" in prompt.lower() or "solve_case(input_text: str)" in prompt:
        subject = "heuristic"
        score_goal = "absolute contest score"
    else:
        subject = "optimizer"
        score_goal = "training score"

    pieces = ["## CORE_GENES"]
    for index, section_name in enumerate(section_names):
        pieces.append(f"### {section_name}")
        if index == len(section_names) - 1:
            pieces.append("- None.")
        else:
            label = section_name.lower().replace("_", " ")
            pieces.append(f"- Use a deterministic {subject} baseline idea for {label}.")
        pieces.append("")
    pieces.extend(
        [
            "## INTERACTION_NOTES",
            f"This sectioned {subject} design keeps the baseline coherent while leaving clear local regions for later evolution.",
            "",
            "## COMPUTE_NOTES",
            "The method is deterministic and keeps computation intentionally lightweight for mock-route testing.",
            "",
            "## CHANGE_DESCRIPTION",
            f"A sectioned deterministic {subject} baseline that can be compiled safely while later mutations target {score_goal}.",
        ]
    )
    return "\n".join(pieces).rstrip() + "\n"


def _mock_region_body(region_name: str, prompt: str) -> str:
    if "def run_packing" in prompt:
        return _mock_circle_region_body(region_name)
    if "def solve_case(input_text: str)" in prompt:
        return _mock_awtf_region_body(region_name)
    if "def build_optimizer" in prompt or "SectionAlignedOptimizer" in prompt:
        return _mock_optimizer_region_body(region_name)
    return f"    # {region_name} uses the deterministic mock baseline.\n"


def _render_mock_region_patch(user_prompt: str) -> str:
    template_regions = _extract_template_region_names(user_prompt)
    mode_match = re.search(r"=== COMPILATION MODE ===\n(.+?)(?:\n\n|$)", user_prompt, flags=re.DOTALL)
    mode = mode_match.group(1).strip() if mode_match else "FULL"
    changed_match = re.search(r"=== CHANGED_SECTIONS ===\n(.+?)(?:\n\n=== |$)", user_prompt, flags=re.DOTALL)
    changed_text = changed_match.group(1).strip() if changed_match else ""
    if mode == "PATCH":
        regions = tuple(line.strip() for line in changed_text.splitlines() if line.strip() and line.strip() != "NONE")
    else:
        regions = template_regions
    pieces = ["## COMPILATION_MODE", mode]
    for region_name in regions:
        pieces.extend(
            (
                "",
                f"## REGION {region_name}",
                _mock_region_body(region_name, user_prompt).rstrip("\n"),
                "## END_REGION",
            )
        )
    return "\n".join(pieces).rstrip() + "\n"


def build_mock_text(request: LlmRequest) -> str:
    organism_id = str(request.metadata.get("organism_id", "mock"))
    generation = int(request.metadata.get("generation", 0))
    base = int(request.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
    use_sgd = (base % 2) == 0
    opt_type = "SGD" if use_sgd else "AdamW"
    joined_prompt = f"{request.system_prompt}\n{request.user_prompt}".lower()
    template = str(request.metadata.get("implementation_template", "")).strip()
    if request.stage == "novelty_check":
        return "## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n\n## REJECTION_REASON\nN/A\n\n## SECTIONS_AT_ISSUE\nNONE\n"
    if request.stage == "compatibility_check":
        return (
            "## COMPATIBILITY_VERDICT\n"
            "COMPATIBILITY_ACCEPTED\n\n"
            "## REJECTION_REASON\n"
            "N/A\n\n"
            "## SECTIONS_AT_ISSUE\n"
            "NONE\n"
        )
    if request.stage == "design":
        sectioned_design = _render_mock_sectioned_design(f"{request.system_prompt}\n{request.user_prompt}")
        if sectioned_design is not None:
            return sectioned_design
    if request.stage == "implementation" and "compilation mode" in joined_prompt:
        return _render_mock_region_patch(request.user_prompt)
    if "circle-packing" in joined_prompt or "circle packing" in joined_prompt or "run_packing" in joined_prompt:
        if request.stage == "design":
            return (
                "## CORE_GENES\n"
                "### INIT_GEOMETRY\n"
                "- Deterministic staggered-row layout with alternating long and short rows inside the unit square.\n\n"
                "### RADIUS_POLICY\n"
                "- Uniform radius assignment chosen conservatively so border constraints and pairwise distances remain valid.\n\n"
                "### EXPANSION_POLICY\n"
                "- Do not expand after initialization in the baseline organism.\n\n"
                "### CONFLICT_MODEL\n"
                "- Treat pairwise overlap and boundary overflow as invalid.\n\n"
                "### REPAIR_POLICY\n"
                "- Use a conservative geometry template that should not require repair.\n\n"
                "### CONTROL_POLICY\n"
                "- Construct centers and radii once, then return the deterministic result.\n\n"
                "### PARAMETERS\n"
                "- Use radius 0.04 for all circles.\n\n"
                "### OPTIONAL_CODE_SKETCH\n"
                "- None.\n\n"
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
