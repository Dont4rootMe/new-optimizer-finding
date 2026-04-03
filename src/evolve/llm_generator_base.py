"""Shared LLM/code-validation utilities for canonical and legacy generators."""

from __future__ import annotations

import os
import re
from typing import Any

from omegaconf import DictConfig

from src.evolve.template_parser import validate_rendered_code

LATEST_PRO_THINKING_MODEL = "gpt-5.4-pro"
MODEL_ALIASES = {
    "latest_pro_thinking": LATEST_PRO_THINKING_MODEL,
    "latest_pro": LATEST_PRO_THINKING_MODEL,
    "chatgpt_pro_thinking": LATEST_PRO_THINKING_MODEL,
}


class BaseLlmGenerator:
    """Resolve model/provider details and validate generated Python code."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.llm_cfg = cfg.evolver.llm
        self.provider = str(self.llm_cfg.provider).lower()
        self.model_name = self._resolve_model_name()
        self.seed = int(self.llm_cfg.seed)

    def _resolve_model_name(self) -> str:
        raw_model = str(self.llm_cfg.get("model", "")).strip()
        if not raw_model:
            return LATEST_PRO_THINKING_MODEL
        return MODEL_ALIASES.get(raw_model, raw_model)

    def _extract_python(self, text: str) -> str:
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip() + "\n"
        return text.strip() + "\n"

    def _validate_code(self, code: str) -> tuple[bool, str | None]:
        return validate_rendered_code(code)

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if self.provider == "mock":
            raise RuntimeError("Mock provider should not call _call_openai.")
        if self.provider not in {"openai", "chatgpt"}:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                "Use one of: openai, chatgpt, mock."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export OPENAI_API_KEY before running evolve."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK is not installed. Install with: pip install -e .[evolve]"
            ) from exc

        client_kwargs: dict[str, Any] = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        request_payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": float(self.llm_cfg.temperature),
            "max_output_tokens": int(self.llm_cfg.max_output_tokens),
        }

        reasoning_effort = self.llm_cfg.get("reasoning_effort")
        if reasoning_effort:
            request_payload["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**request_payload)
        response_payload = response.model_dump() if hasattr(response, "model_dump") else {"raw": str(response)}

        text = getattr(response, "output_text", None)
        if not text:
            text = ""
            output = response_payload.get("output", [])
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                for chunk in content:
                    if isinstance(chunk, dict) and chunk.get("type") == "output_text":
                        text += str(chunk.get("text", ""))

        if text:
            return text, request_payload, response_payload

        if bool(self.llm_cfg.get("fallback_to_chat_completions", True)):
            chat_request = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(self.llm_cfg.temperature),
                "max_tokens": int(self.llm_cfg.max_output_tokens),
            }
            chat_resp = client.chat.completions.create(**chat_request)
            chat_payload = chat_resp.model_dump() if hasattr(chat_resp, "model_dump") else {"raw": str(chat_resp)}
            message_content = ""
            choices = chat_payload.get("choices", [])
            if choices:
                message_content = str(choices[0].get("message", {}).get("content", ""))
            if message_content:
                return message_content, {"responses": request_payload, "chat_fallback": chat_request}, {
                    "responses": response_payload,
                    "chat_fallback": chat_payload,
                }

        raise RuntimeError("OpenAI response did not contain usable text output.")
