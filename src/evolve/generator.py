"""LLM-driven optimizer candidate generation."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.evolve.operators import (
    GeneticOperator,
    SeedOperator,
    MutationOperator,
    CrossoverOperator,
)
from src.organisms.organism import build_organism_from_response as _build_organism_from_response
from src.evolve.storage import sha1_text, utc_now_iso, write_json
from src.evolve.template_parser import parse_llm_response, render_template, validate_rendered_code
from src.evolve.types import CandidateMeta, OrganismMeta

LATEST_PRO_THINKING_MODEL = "gpt-5.4-pro"
MODEL_ALIASES = {
    "latest_pro_thinking": LATEST_PRO_THINKING_MODEL,
    "latest_pro": LATEST_PRO_THINKING_MODEL,
    "chatgpt_pro_thinking": LATEST_PRO_THINKING_MODEL,
}


class OptimizerGenerator:
    """Generate optimizer.py candidates using configured LLM provider."""

    def __init__(self, cfg: DictConfig, prompts_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.evolver_cfg = cfg.evolver
        self.llm_cfg = cfg.evolver.llm
        self.provider = str(self.llm_cfg.provider).lower()
        self.model_name = self._resolve_model_name()

        base_dir = Path(__file__).resolve().parent
        self.prompts_dir = Path(prompts_dir) if prompts_dir is not None else base_dir / "prompts"

        self.seed = int(self.llm_cfg.seed)

        self.system_template = (self.prompts_dir / "optimizer_system.txt").read_text(encoding="utf-8")
        self.user_template = (self.prompts_dir / "optimizer_user.txt").read_text(encoding="utf-8")

    def _resolve_model_name(self) -> str:
        raw_model = str(self.llm_cfg.get("model", "")).strip()
        if not raw_model:
            return LATEST_PRO_THINKING_MODEL
        return MODEL_ALIASES.get(raw_model, raw_model)

    def _context_block(self, context: list[dict[str, Any]]) -> str:
        if not context:
            return "No successful prior candidates."

        lines: list[str] = []
        for item in context:
            candidate_id = item.get("candidate_id", "unknown")
            score = item.get("aggregate_score")
            status = item.get("status", "unknown")
            lines.append(f"- candidate={candidate_id}, score={score}, status={status}")
        return "\n".join(lines)

    def _build_prompts(self, context: list[dict[str, Any]]) -> tuple[str, str, str]:
        user_prompt = self.user_template.format(context_block=self._context_block(context))
        prompt_hash = sha1_text(self.system_template + "\n" + user_prompt)
        return self.system_template, user_prompt, prompt_hash

    def _extract_python(self, text: str) -> str:
        pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip() + "\n"
        return text.strip() + "\n"

    def _validate_code(self, code: str) -> tuple[bool, str | None]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"

        has_builder = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_optimizer"
            for node in ast.walk(tree)
        )
        if not has_builder:
            return False, "Missing required function: build_optimizer(model, max_steps)"

        has_controller_class = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            method_names = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if {"step", "zero_grad"}.issubset(method_names):
                has_controller_class = True
                break

        if not has_controller_class:
            return False, "Missing optimizer controller class with step/zero_grad methods"

        return True, None

    def _mock_candidate_code(self, candidate_id: str, generation: int) -> str:
        base = int(self.llm_cfg.seed) + generation + sum(ord(ch) for ch in candidate_id[:8])
        use_sgd = (base % 2) == 0

        if use_sgd:
            return (
                "import math\n"
                "import torch\n"
                "import torch.nn as nn\n"
                "from torch.optim.lr_scheduler import LambdaLR\n\n"
                "OPTIMIZER_NAME = 'MockSGDController'\n\n"
                "class MockSGDController:\n"
                "    def __init__(self, model: nn.Module, max_steps: int):\n"
                "        self.named_parameters = [\n"
                "            (n, p) for n, p in model.named_parameters() if p.requires_grad\n"
                "        ]\n"
                "        params = [p for _, p in self.named_parameters]\n"
                "        self.optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=1e-4)\n"
                "        warmup_steps = max(1, int(max_steps * 0.02))\n"
                "        def _lambda(step):\n"
                "            if warmup_steps > 0 and step < warmup_steps:\n"
                "                return max(1e-8, float(step + 1) / float(warmup_steps))\n"
                "            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))\n"
                "            progress = min(max(progress, 0.0), 1.0)\n"
                "            return 0.5 * (1.0 + math.cos(math.pi * progress))\n"
                "        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lambda)\n\n"
                "    def step(self, weights, grads, activations, step_fn):\n"
                "        del weights, grads, activations, step_fn\n"
                "        self.optimizer.step()\n"
                "        if self.scheduler is not None:\n"
                "            self.scheduler.step()\n\n"
                "    def zero_grad(self, set_to_none=True):\n"
                "        self.optimizer.zero_grad(set_to_none=set_to_none)\n\n"
                "def build_optimizer(model: nn.Module, max_steps: int):\n"
                "    return MockSGDController(model, max_steps)\n"
            )

        return (
            "import math\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from torch.optim.lr_scheduler import LambdaLR\n\n"
            "OPTIMIZER_NAME = 'MockAdamWController'\n\n"
            "class MockAdamWController:\n"
            "    def __init__(self, model: nn.Module, max_steps: int):\n"
            "        self.named_parameters = [\n"
            "            (n, p) for n, p in model.named_parameters() if p.requires_grad\n"
            "        ]\n"
            "        params = [p for _, p in self.named_parameters]\n"
            "        self.optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))\n"
            "        warmup_steps = max(1, int(max_steps * 0.05))\n"
            "        def _lambda(step):\n"
            "            if step < warmup_steps:\n"
            "                return max(1e-8, float(step + 1) / float(warmup_steps))\n"
            "            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))\n"
            "            progress = min(max(progress, 0.0), 1.0)\n"
            "            return 0.5 * (1.0 + math.cos(math.pi * progress))\n"
            "        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lambda)\n\n"
            "    def step(self, weights, grads, activations, step_fn):\n"
            "        del weights, grads, activations, step_fn\n"
            "        self.optimizer.step()\n"
            "        if self.scheduler is not None:\n"
            "            self.scheduler.step()\n\n"
            "    def zero_grad(self, set_to_none=True):\n"
            "        self.optimizer.zero_grad(set_to_none=set_to_none)\n\n"
            "def build_optimizer(model: nn.Module, max_steps: int):\n"
            "    return MockAdamWController(model, max_steps)\n"
        )

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

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        org_dir: Path | None = None,
    ) -> str:
        """Public interface for LLM calls used by organism operators.

        Returns the raw text response. Also persists request/response to
        ``org_dir`` if provided.
        """
        if self.provider == "mock":
            raw_text = self._mock_structured_response("mock", 0)
            if org_dir is not None:
                write_json(org_dir / "llm_request.json", {"provider": "mock"})
                write_json(org_dir / "llm_response.json", {"provider": "mock", "text": raw_text})
            return raw_text

        raw_text, request_payload, response_payload = self._call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if org_dir is not None:
            write_json(org_dir / "llm_request.json", request_payload)
            write_json(org_dir / "llm_response.json", response_payload)
        return raw_text

    def generate_candidate(
        self,
        candidate_id: str,
        generation: int,
        candidate_dir: str | Path,
        context: list[dict[str, Any]] | None = None,
    ) -> CandidateMeta:
        """Generate one candidate and persist optimizer + metadata files."""

        context = context or []
        candidate_path = Path(candidate_dir)

        attempts: list[dict[str, Any]] = []
        max_attempts = int(self.evolver_cfg.max_generation_attempts)

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            system_prompt, user_prompt, prompt_hash = self._build_prompts(context)

            if self.provider == "mock":
                raw_text = self._mock_candidate_code(candidate_id=candidate_id, generation=generation)
                request_payload = {
                    "provider": "mock",
                    "model": self.model_name,
                    "attempt": attempt,
                    "seed": int(self.llm_cfg.seed),
                    "prompt_hash": prompt_hash,
                }
                response_payload = {
                    "provider": "mock",
                    "text": raw_text,
                }
            else:
                raw_text, request_payload, response_payload = self._call_openai(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

            code = self._extract_python(raw_text)
            is_valid, validation_error = self._validate_code(code)

            attempts.append(
                {
                    "attempt": attempt,
                    "request": request_payload,
                    "response": response_payload,
                    "validation_error": validation_error,
                    "is_valid": is_valid,
                }
            )

            if is_valid:
                (candidate_path / "optimizer.py").write_text(code, encoding="utf-8")

                write_json(candidate_path / "llm_request.json", attempts[-1]["request"])
                write_json(candidate_path / "llm_response.json", attempts[-1]["response"])

                meta = CandidateMeta(
                    candidate_id=candidate_id,
                    generation=generation,
                    timestamp=utc_now_iso(),
                    model_name=self.model_name,
                    prompt_hash=prompt_hash,
                    seed=int(self.llm_cfg.seed),
                    candidate_dir=str(candidate_path),
                    optimizer_path=str(candidate_path / "optimizer.py"),
                )
                write_json(candidate_path / "meta.json", meta.to_dict())
                return meta

            last_error = validation_error

        write_json(
            candidate_path / "llm_request.json",
            {"attempts": [item["request"] for item in attempts]},
        )
        write_json(
            candidate_path / "llm_response.json",
            {"attempts": [item["response"] for item in attempts], "errors": [item["validation_error"] for item in attempts]},
        )

        raise RuntimeError(f"Failed to generate valid optimizer code after {max_attempts} attempts: {last_error}")

    # ── Structured generation for evolution loop ──────────────────────

    def _mock_structured_response(self, organism_id: str, generation: int) -> str:
        """Return a mock structured LLM response for testing."""
        base = int(self.llm_cfg.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
        use_sgd = (base % 2) == 0
        opt_type = "SGD" if use_sgd else "AdamW"
        lr = "0.05" if use_sgd else "3e-4"
        wd = "1e-4" if use_sgd else "0.01"

        return (
            f"## IDEA_DNA\n"
            f"mock {opt_type} with cosine schedule; warmup phase\n\n"
            f"## CHANGE_DESCRIPTION\n"
            f"Mock {opt_type} optimizer with warmup and cosine annealing for testing.\n\n"
            f"## IMPORTS\n"
            f"import math\n"
            f"from torch.optim.lr_scheduler import LambdaLR\n\n"
            f"## INIT_BODY\n"
            f"        self.named_parameters = [\n"
            f"            (n, p) for n, p in model.named_parameters() if p.requires_grad\n"
            f"        ]\n"
            f"        params = [p for _, p in self.named_parameters]\n"
            f"        self.optimizer = torch.optim.{opt_type}(params, lr={lr})\n"
            f"        warmup_steps = max(1, int(max_steps * 0.05))\n"
            f"        def _lambda(step):\n"
            f"            if step < warmup_steps:\n"
            f"                return max(1e-8, float(step + 1) / float(warmup_steps))\n"
            f"            progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))\n"
            f"            progress = min(max(progress, 0.0), 1.0)\n"
            f"            return 0.5 * (1.0 + math.cos(math.pi * progress))\n"
            f"        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lambda)\n\n"
            f"## STEP_BODY\n"
            f"        del weights, grads, activations, step_fn\n"
            f"        self.optimizer.step()\n"
            f"        if self.scheduler is not None:\n"
            f"            self.scheduler.step()\n\n"
            f"## ZERO_GRAD_BODY\n"
            f"        self.optimizer.zero_grad(set_to_none=set_to_none)\n"
        )

    def generate_organism(
        self,
        operator: GeneticOperator,
        parents: list[OrganismMeta],
        organism_id: str,
        generation: int,
        organism_dir: Path,
    ) -> OrganismMeta:
        """Generate one organism using a genetic operator and LLM.

        Works for seed, mutation, and crossover operators.
        """
        max_attempts = int(self.evolver_cfg.max_generation_attempts)
        parent_ids = operator.parent_ids(parents)

        # Collect parent evolution log for inheritance
        parent_evolution_log: list[dict[str, Any]] = []
        if parents:
            parent_evolution_log = list(parents[0].evolution_log)

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            system_prompt, user_prompt = operator.build_prompts(parents, self.prompts_dir)
            prompt_hash = sha1_text(system_prompt + "\n" + user_prompt)

            if self.provider == "mock":
                raw_text = self._mock_structured_response(organism_id, generation)
                request_payload: dict[str, Any] = {
                    "provider": "mock",
                    "operator": operator.operator_name,
                    "attempt": attempt,
                }
                response_payload: dict[str, Any] = {"provider": "mock", "text": raw_text}
            else:
                raw_text, request_payload, response_payload = self._call_openai(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

            write_json(organism_dir / "llm_request.json", request_payload)
            write_json(organism_dir / "llm_response.json", response_payload)

            try:
                parsed = parse_llm_response(raw_text)
                org = _build_organism_from_response(
                    parsed=parsed,
                    organism_id=organism_id,
                    generation=generation,
                    parent_ids=parent_ids,
                    operator=operator.operator_name,
                    org_dir=organism_dir,
                    model_name=self.model_name,
                    prompt_hash=prompt_hash,
                    seed=int(self.llm_cfg.seed),
                    timestamp=utc_now_iso(),
                    parent_evolution_log=parent_evolution_log,
                )
                return org
            except (ValueError, KeyError) as exc:
                last_error = str(exc)

        raise RuntimeError(
            f"Failed to generate valid organism after {max_attempts} attempts: {last_error}"
        )
