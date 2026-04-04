"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.evolve.llm_generator_base import BaseLlmGenerator
from src.evolve.operators import SeedOperator
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import sha1_text, utc_now_iso, write_json
from src.evolve.template_parser import parse_llm_response, render_template
from src.evolve.types import Island, OrganismMeta
from src.organisms.organism import (
    build_implementation_prompt_from_design,
    build_organism_from_response,
)

class OptimizerGenerator(BaseLlmGenerator):
    """Generate canonical structured seed organisms from `conf/prompts/*` assets."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.prompt_bundle = load_prompt_bundle(cfg)

    def _mock_design_response(self, organism_id: str, generation: int) -> str:
        base = int(self.llm_cfg.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
        use_sgd = (base % 2) == 0
        opt_type = "SGD" if use_sgd else "AdamW"

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

    def _mock_implementation_response(self, organism_id: str, generation: int) -> str:
        base = int(self.llm_cfg.seed) + generation + sum(ord(ch) for ch in organism_id[:8])
        use_sgd = (base % 2) == 0
        opt_type = "SGD" if use_sgd else "AdamW"
        lr = "0.05" if use_sgd else "3e-4"
        sections = {
            "IMPORTS": "import math\nfrom torch.optim.lr_scheduler import LambdaLR",
            "INIT_BODY": (
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
            "STEP_BODY": (
                "        del weights, grads, activations, step_fn\n"
                "        self.optimizer.step()\n"
                "        if self.scheduler is not None:\n"
                "            self.scheduler.step()"
            ),
            "ZERO_GRAD_BODY": "        self.optimizer.zero_grad(set_to_none=set_to_none)",
        }
        class_name = f"Optimizer_{organism_id[:8]}"
        return render_template(
            sections,
            optimizer_name=class_name,
            class_name=class_name,
            template_text=self.prompt_bundle.implementation_template,
        )

    def _call_llm_stage(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        *,
        organism_id: str,
        generation: int,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Return raw stage output plus provider payloads without writing artifacts."""

        if self.provider == "mock":
            if stage == "design":
                raw_text = self._mock_design_response(organism_id, generation)
            elif stage == "implementation":
                raw_text = self._mock_implementation_response(organism_id, generation)
            else:
                raise ValueError(f"Unsupported mock LLM stage '{stage}'.")
            payload = {"provider": "mock", "stage": stage}
            return raw_text, payload, {"provider": "mock", "stage": stage, "text": raw_text}

        raw_text, request_payload, response_payload = self._call_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return raw_text, request_payload, response_payload

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
    ) -> tuple[dict[str, str], str, str]:
        """Run design and implementation stages and persist both LLM exchanges."""

        raw_design, design_request, design_response = self._call_llm_stage(
            "design",
            design_system_prompt,
            design_user_prompt,
            organism_id=organism_id,
            generation=generation,
        )
        parsed_design = parse_llm_response(raw_design)
        implementation_system_prompt, implementation_user_prompt = build_implementation_prompt_from_design(
            parsed_design,
            self.prompt_bundle,
        )
        optimizer_code, implementation_request, implementation_response = self._call_llm_stage(
            "implementation",
            implementation_system_prompt,
            implementation_user_prompt,
            organism_id=organism_id,
            generation=generation,
        )

        write_json(
            org_dir / "llm_request.json",
            {
                "design": {
                    "system_prompt": design_system_prompt,
                    "user_prompt": design_user_prompt,
                    "request": design_request,
                },
                "implementation": {
                    "system_prompt": implementation_system_prompt,
                    "user_prompt": implementation_user_prompt,
                    "request": implementation_request,
                },
            },
        )
        write_json(
            org_dir / "llm_response.json",
            {
                "design": {
                    "text": raw_design,
                    "response": design_response,
                },
                "implementation": {
                    "text": optimizer_code,
                    "response": implementation_response,
                },
            },
        )

        prompt_hash = sha1_text(
            "\n".join(
                (
                    design_system_prompt,
                    design_user_prompt,
                    implementation_system_prompt,
                    implementation_user_prompt,
                )
            )
        )
        return parsed_design, optimizer_code, prompt_hash

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

            try:
                parsed, optimizer_code, prompt_hash = self.run_creation_stages(
                    design_system_prompt=system_prompt,
                    design_user_prompt=user_prompt,
                    org_dir=organism_dir,
                    organism_id=organism_id,
                    generation=generation,
                )
                return build_organism_from_response(
                    parsed=parsed,
                    optimizer_code=optimizer_code,
                    implementation_template=self.prompt_bundle.implementation_template,
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
