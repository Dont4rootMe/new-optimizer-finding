"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def _organism_log_label(organism_id: str, island_id: str | None) -> str:
    """Prefix for logs about a single organism; includes island when known."""

    if island_id is None:
        return f"organism {organism_id}"
    return f"organism {organism_id} (island={island_id})"


def _announce(message: str) -> None:
    """Mirror `evolution_loop._announce`: log + flushed stderr line.

    Creation stages block on LLM inference for a long time. Without direct
    stderr prints we go silent for many minutes, which looks like a hang.
    """

    LOGGER.info(message)
    try:
        print(f"[evolve] {message}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass

from api_platforms import ApiPlatformRegistry, LlmRequest
from src.evolve.llm_generator_base import BaseLlmGenerator
from src.evolve.operators import SeedOperator
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import sha1_text, utc_now_iso, write_json
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import CreationStageResult, Island, OrganismMeta
from src.organisms.organism import (
    build_implementation_prompt_from_design,
    build_organism_from_response,
)


class CandidateGenerator(BaseLlmGenerator):
    """Generate canonical structured seed organisms from configured prompt assets."""

    def __init__(self, cfg: DictConfig, llm_registry: ApiPlatformRegistry | None = None) -> None:
        registry = llm_registry or ApiPlatformRegistry(cfg)
        self._owns_llm_registry = llm_registry is None
        super().__init__(cfg, registry)
        self.prompt_bundle = load_prompt_bundle(cfg)

    def close(self) -> None:
        if self._owns_llm_registry:
            self.registry.stop()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _call_llm_stage(
        self,
        route_id: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        *,
        organism_id: str,
        generation: int,
        extra_metadata: dict[str, object] | None = None,
    ):
        """Run one routed LLM stage against the shared broker registry."""

        metadata = {
            "organism_id": organism_id,
            "generation": generation,
            "stage": stage,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        response = self.registry.generate(
            LlmRequest(
                route_id=route_id,
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=self.seed,
                metadata=metadata,
            )
        )
        return response

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        island_id: str | None = None,
    ) -> CreationStageResult:
        """Run design and implementation stages and persist both LLM exchanges."""

        org_label = _organism_log_label(organism_id, island_id)
        route_id = self.sample_route_id(organism_id=organism_id)
        _announce(
            f"{org_label} -> route {route_id}: calling design stage (generation={generation})"
        )
        llm_request_path = org_dir / "llm_request.json"
        llm_response_path = org_dir / "llm_response.json"
        design_response = self._call_llm_stage(
            route_id,
            "design",
            design_system_prompt,
            design_user_prompt,
            organism_id=organism_id,
            generation=generation,
        )
        _announce(
            f"{org_label} design stage returned "
            f"(route={route_id}, provider={design_response.provider}, "
            f"model={design_response.provider_model_id})"
        )
        llm_request_payload = {
            "route_id": route_id,
            "provider": design_response.provider,
            "provider_model_id": design_response.provider_model_id,
            "design": {
                "route_id": route_id,
                "provider": design_response.provider,
                "provider_model_id": design_response.provider_model_id,
                "system_prompt": design_system_prompt,
                "user_prompt": design_user_prompt,
                "request": design_response.raw_request,
            },
        }
        llm_response_payload = {
            "route_id": route_id,
            "provider": design_response.provider,
            "provider_model_id": design_response.provider_model_id,
            "design": {
                "route_id": route_id,
                "provider": design_response.provider,
                "provider_model_id": design_response.provider_model_id,
                "text": design_response.text,
                "response": design_response.raw_response,
                "usage": design_response.usage,
                "started_at": design_response.started_at,
                "finished_at": design_response.finished_at,
            },
        }
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        parsed_design = parse_llm_response(design_response.text)
        implementation_system_prompt, implementation_user_prompt = build_implementation_prompt_from_design(
            parsed_design,
            self.prompt_bundle,
        )
        llm_request_payload["implementation"] = {
            "route_id": route_id,
            "system_prompt": implementation_system_prompt,
            "user_prompt": implementation_user_prompt,
            "request": None,
        }
        write_json(llm_request_path, llm_request_payload)

        _announce(
            f"{org_label} -> route {route_id}: calling implementation stage"
        )
        implementation_response = self._call_llm_stage(
            route_id,
            "implementation",
            implementation_system_prompt,
            implementation_user_prompt,
            organism_id=organism_id,
            generation=generation,
            extra_metadata={"implementation_template": self.prompt_bundle.implementation_template},
        )
        _announce(
            f"{org_label} implementation stage returned "
            f"(route={route_id})"
        )
        implementation_code = self._extract_python(implementation_response.text)
        llm_request_payload["implementation"] = {
            "route_id": route_id,
            "provider": implementation_response.provider,
            "provider_model_id": implementation_response.provider_model_id,
            "system_prompt": implementation_system_prompt,
            "user_prompt": implementation_user_prompt,
            "request": implementation_response.raw_request,
        }
        llm_response_payload["implementation"] = {
            "route_id": route_id,
            "provider": implementation_response.provider,
            "provider_model_id": implementation_response.provider_model_id,
            "text": implementation_code,
            "response": implementation_response.raw_response,
            "usage": implementation_response.usage,
            "started_at": implementation_response.started_at,
            "finished_at": implementation_response.finished_at,
        }
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

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
        return CreationStageResult(
            parsed_design=parsed_design,
            implementation_code=implementation_code,
            prompt_hash=prompt_hash,
            llm_route_id=route_id,
            llm_provider=design_response.provider,
            provider_model_id=design_response.provider_model_id,
        )

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
                creation = self.run_creation_stages(
                    design_system_prompt=system_prompt,
                    design_user_prompt=user_prompt,
                    org_dir=organism_dir,
                    organism_id=organism_id,
                    generation=generation,
                    island_id=island.island_id,
                )
                return build_organism_from_response(
                    parsed=creation.parsed_design,
                    implementation_code=creation.implementation_code,
                    organism_id=organism_id,
                    island_id=island.island_id,
                    generation=generation,
                    mother_id=None,
                    father_id=None,
                    operator="seed",
                    org_dir=organism_dir,
                    llm_route_id=creation.llm_route_id,
                    llm_provider=creation.llm_provider,
                    provider_model_id=creation.provider_model_id,
                    prompt_hash=creation.prompt_hash,
                    seed=self.seed,
                    timestamp=utc_now_iso(),
                    parent_lineage=[],
                )
            except Exception as exc:  # noqa: BLE001
                # Retry on any failure: malformed LLM output (ValueError / KeyError
                # from parse_llm_response), transient provider errors (RuntimeError
                # from ollama / broker / HTTP 5xx), timeouts, etc. The creation
                # stage must be tolerant of upstream flakiness; unrecoverable
                # failures surface after `max_attempts` exhaustion.
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Seed organism %s attempt %d/%d failed on island %s: %s",
                    organism_id,
                    attempt,
                    max_attempts,
                    island.island_id,
                    last_error,
                )

        raise RuntimeError(
            f"Failed to generate valid organism after {max_attempts} attempts: {last_error}"
        )
