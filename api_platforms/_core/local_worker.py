"""Local worker processes for route backends that keep models in memory."""

from __future__ import annotations

import os
import queue
import time
from typing import Any

from api_platforms._core.providers import build_mock_text
from api_platforms._core.types import ApiRouteConfig, LlmRequest, LlmResponse


def _torch_dtype(dtype_name: str | None) -> Any:
    if not dtype_name:
        return None
    import torch

    normalized = str(dtype_name).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch_dtype '{dtype_name}'.")
    return mapping[normalized]


def _load_transformers_state(route_cfg: ApiRouteConfig, gpu_rank: int | None) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers backend requires transformers/accelerate/safetensors/sentencepiece. "
            "Install with: pip install -e .[hf,evolve]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        route_cfg.tokenizer_name or route_cfg.model_name_or_path,
        trust_remote_code=bool(route_cfg.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(route_cfg.trust_remote_code),
    }
    torch_dtype = _torch_dtype(route_cfg.torch_dtype)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if route_cfg.attn_implementation:
        load_kwargs["attn_implementation"] = route_cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(route_cfg.model_name_or_path, **load_kwargs)
    if torch.cuda.is_available() and gpu_rank is not None:
        device = f"cuda:{gpu_rank}"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    return {
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
    }


def _build_inputs(state: dict[str, Any], request: LlmRequest) -> tuple[Any, int]:
    tokenizer = state["tokenizer"]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt = (
            "System:\n"
            f"{request.system_prompt.strip()}\n\n"
            "User:\n"
            f"{request.user_prompt.strip()}\n\n"
            "Assistant:\n"
        )
        encoded = tokenizer(prompt, return_tensors="pt").input_ids
    device = state["device"]
    encoded = encoded.to(device)
    return encoded, int(encoded.shape[-1])


def _generate_transformers_text(state: dict[str, Any], route_cfg: ApiRouteConfig, request: LlmRequest) -> str:
    import torch

    tokenizer = state["tokenizer"]
    model = state["model"]
    input_ids, prompt_len = _build_inputs(state, request)
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(route_cfg.max_output_tokens),
        "do_sample": bool(route_cfg.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if route_cfg.temperature is not None:
        generate_kwargs["temperature"] = float(route_cfg.temperature)
    if route_cfg.top_p is not None:
        generate_kwargs["top_p"] = float(route_cfg.top_p)
    if route_cfg.top_k is not None:
        generate_kwargs["top_k"] = int(route_cfg.top_k)

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    generated = outputs[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def local_worker_main(
    route_cfg_payload: dict[str, Any],
    gpu_rank: int | None,
    input_queue: Any,
    result_queue: Any,
) -> None:
    """Long-lived worker loop for local backends."""

    route_cfg = ApiRouteConfig.from_dict(route_cfg_payload)
    state: dict[str, Any] | None = None
    if route_cfg.backend == "transformers":
        state = _load_transformers_state(route_cfg, gpu_rank)

    while True:
        task = input_queue.get()
        if task is None:
            break

        task_id = str(task["task_id"])
        try:
            request = LlmRequest.from_dict(dict(task["request"]))
            started_at = task["started_at"]
            if route_cfg.backend == "mock_local":
                if route_cfg.mock_delay_sec > 0:
                    time.sleep(float(route_cfg.mock_delay_sec))
                text = build_mock_text(request)
                usage: dict[str, Any] = {}
                raw_request = {"stage": request.stage, "metadata": request.metadata, "gpu_rank": gpu_rank}
                raw_response = {"stage": request.stage, "text": text, "gpu_rank": gpu_rank}
            elif route_cfg.backend == "transformers":
                text = _generate_transformers_text(state or {}, route_cfg, request)
                usage = {}
                raw_request = {"stage": request.stage, "metadata": request.metadata, "gpu_rank": gpu_rank}
                raw_response = {"stage": request.stage, "gpu_rank": gpu_rank, "text": text}
            else:
                raise ValueError(f"Unsupported local worker backend '{route_cfg.backend}'.")

            response = LlmResponse(
                text=text,
                route_id=route_cfg.route_id,
                provider=route_cfg.provider,
                provider_model_id=route_cfg.provider_model_id,
                raw_request=raw_request,
                raw_response=raw_response,
                usage=usage,
                started_at=str(started_at),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            result_queue.put({"task_id": task_id, "ok": True, "response": response.to_dict()})
        except Exception as exc:  # noqa: BLE001
            result_queue.put({"task_id": task_id, "ok": False, "error": str(exc), "gpu_rank": gpu_rank, "pid": os.getpid()})
