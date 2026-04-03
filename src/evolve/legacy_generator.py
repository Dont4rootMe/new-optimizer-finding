"""LEGACY raw candidate generator for the quarantined candidate-first mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.evolve.llm_generator_base import BaseLlmGenerator
from src.evolve.storage import sha1_text, utc_now_iso, write_json
from src.evolve.types import CandidateMeta


class LegacyCandidateGenerator(BaseLlmGenerator):
    """LEGACY candidate-first generator using raw Python prompt templates."""

    def __init__(self, cfg: DictConfig, prompts_dir: str | Path | None = None) -> None:
        super().__init__(cfg)
        repo_root = Path(__file__).resolve().parents[2]
        default_prompts_dir = repo_root / "conf" / "prompts" / "legacy_candidate"
        self.prompts_dir = Path(prompts_dir) if prompts_dir is not None else default_prompts_dir
        self.system_template = (self.prompts_dir / "system.txt").read_text(encoding="utf-8")
        self.user_template = (self.prompts_dir / "user.txt").read_text(encoding="utf-8")

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

    def generate_candidate(
        self,
        candidate_id: str,
        generation: int,
        candidate_dir: str | Path,
        context: list[dict[str, Any]] | None = None,
    ) -> CandidateMeta:
        """LEGACY: generate one raw-code candidate and persist candidate artifacts."""

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
                response_payload = {"provider": "mock", "text": raw_text}
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
                    prompt_hash=str(prompt_hash),
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
            {
                "attempts": [item["response"] for item in attempts],
                "errors": [item["validation_error"] for item in attempts],
            },
        )
        raise RuntimeError(
            f"Failed to generate valid optimizer code after {max_attempts} attempts: {last_error}"
        )
