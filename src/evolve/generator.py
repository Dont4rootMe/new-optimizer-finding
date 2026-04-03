"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.evolve.llm_generator_base import BaseLlmGenerator
from src.evolve.operators import SeedOperator
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import sha1_text, utc_now_iso, write_json
from src.evolve.template_parser import parse_llm_response, validate_rendered_code
from src.evolve.types import Island, OrganismMeta
from src.organisms.organism import build_organism_from_response

class OptimizerGenerator(BaseLlmGenerator):
    """Generate canonical structured seed organisms from `conf/prompts/*` assets."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.prompt_bundle = load_prompt_bundle(cfg)

    def _mock_structured_response(self, organism_id: str, generation: int) -> str:
        base = int(self.llm_cfg.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
        use_sgd = (base % 2) == 0
        opt_type = "SGD" if use_sgd else "AdamW"
        lr = "0.05" if use_sgd else "3e-4"

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
            f"Mock {opt_type} optimizer with warmup and cosine annealing for testing.\n\n"
            "## IMPORTS\n"
            "import math\n"
            "from torch.optim.lr_scheduler import LambdaLR\n\n"
            "## INIT_BODY\n"
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
            "        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lambda)\n\n"
            "## STEP_BODY\n"
            "        del weights, grads, activations, step_fn\n"
            "        self.optimizer.step()\n"
            "        if self.scheduler is not None:\n"
            "            self.scheduler.step()\n\n"
            "## ZERO_GRAD_BODY\n"
            "        self.optimizer.zero_grad(set_to_none=set_to_none)\n"
        )

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        org_dir: Path | None = None,
    ) -> str:
        """Return raw structured response text and persist the LLM exchange."""

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

    def generate_seed_organism(
        self,
        island: Island,
        organism_id: str,
        generation: int,
        organism_dir: Path,
    ) -> OrganismMeta:
        """Generate one seed organism for a configured island."""

        max_attempts = int(self.evolver_cfg.max_generation_attempts)
        seed_operator = SeedOperator(island)

        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            system_prompt, user_prompt = seed_operator.build_prompts(self.prompt_bundle)
            prompt_hash = sha1_text(system_prompt + "\n" + user_prompt)

            if self.provider == "mock":
                raw_text = self._mock_structured_response(organism_id, generation)
                request_payload: dict[str, Any] = {
                    "provider": "mock",
                    "operator": "seed",
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
                return build_organism_from_response(
                    parsed=parsed,
                    organism_id=organism_id,
                    island_id=island.island_id,
                    generation=generation,
                    mother_id=None,
                    father_id=None,
                    operator="seed",
                    org_dir=organism_dir,
                    model_name=self.model_name,
                    prompt_hash=prompt_hash,
                    seed=int(self.llm_cfg.seed),
                    timestamp=utc_now_iso(),
                    parent_lineage=[],
                )
            except (ValueError, KeyError) as exc:
                last_error = str(exc)

        raise RuntimeError(
            f"Failed to generate valid organism after {max_attempts} attempts: {last_error}"
        )
