"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


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
from src.evolve.storage import read_json, sha1_text, utc_now_iso, write_json, write_organism_meta
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import CreationStageResult, Island, OrganismMeta
from src.organisms.organism import (
    build_repair_prompt,
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
    ) -> CreationStageResult:
        """Run design and implementation stages and persist both LLM exchanges."""

        route_id = self.sample_route_id(organism_id=organism_id)
        _announce(
            f"organism {organism_id} -> route {route_id}: calling design stage (generation={generation})"
        )
        llm_request_path = org_dir / "llm_request.json"
        llm_response_path = org_dir / "llm_response.json"
        llm_request_payload = {
            "route_id": route_id,
            "design": {
                "route_id": route_id,
                "system_prompt": design_system_prompt,
                "user_prompt": design_user_prompt,
                "request": None,
                "status": "in_flight",
                "error_msg": None,
            },
        }
        llm_response_payload = {
            "route_id": route_id,
            "design": {
                "route_id": route_id,
                "text": None,
                "response": None,
                "usage": None,
                "started_at": None,
                "finished_at": None,
                "status": "awaiting_response",
                "error_msg": None,
            },
        }
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        try:
            design_response = self._call_llm_stage(
                route_id,
                "design",
                design_system_prompt,
                design_user_prompt,
                organism_id=organism_id,
                generation=generation,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_request_payload["design"]["status"] = "failed"
            llm_request_payload["design"]["error_msg"] = error_msg
            llm_response_payload["design"]["status"] = "failed"
            llm_response_payload["design"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            _announce(f"organism {organism_id} design stage failed (route={route_id}): {error_msg}")
            raise
        _announce(
            f"organism {organism_id} design stage returned "
            f"(route={route_id}, provider={design_response.provider}, "
            f"model={design_response.provider_model_id})"
        )
        llm_request_payload["provider"] = design_response.provider
        llm_request_payload["provider_model_id"] = design_response.provider_model_id
        llm_request_payload["design"] = {
            "route_id": route_id,
            "provider": design_response.provider,
            "provider_model_id": design_response.provider_model_id,
            "system_prompt": design_system_prompt,
            "user_prompt": design_user_prompt,
            "request": design_response.raw_request,
            "status": "completed",
            "error_msg": None,
        }
        llm_response_payload["provider"] = design_response.provider
        llm_response_payload["provider_model_id"] = design_response.provider_model_id
        llm_response_payload["design"] = {
            "route_id": route_id,
            "provider": design_response.provider,
            "provider_model_id": design_response.provider_model_id,
            "text": design_response.text,
            "response": design_response.raw_response,
            "usage": design_response.usage,
            "started_at": design_response.started_at,
            "finished_at": design_response.finished_at,
            "status": "completed",
            "error_msg": None,
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
            "status": "in_flight",
            "error_msg": None,
        }
        llm_response_payload["implementation"] = {
            "route_id": route_id,
            "text": None,
            "response": None,
            "usage": None,
            "started_at": None,
            "finished_at": None,
            "status": "awaiting_response",
            "error_msg": None,
        }
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        _announce(
            f"organism {organism_id} -> route {route_id}: calling implementation stage"
        )
        try:
            implementation_response = self._call_llm_stage(
                route_id,
                "implementation",
                implementation_system_prompt,
                implementation_user_prompt,
                organism_id=organism_id,
                generation=generation,
                extra_metadata={"implementation_template": self.prompt_bundle.implementation_template},
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_request_payload["implementation"]["status"] = "failed"
            llm_request_payload["implementation"]["error_msg"] = error_msg
            llm_response_payload["implementation"]["status"] = "failed"
            llm_response_payload["implementation"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            _announce(f"organism {organism_id} implementation stage failed (route={route_id}): {error_msg}")
            raise
        _announce(
            f"organism {organism_id} implementation stage returned "
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
            "status": "completed",
            "error_msg": None,
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
            "status": "completed",
            "error_msg": None,
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

    def run_creation_stages_with_retries(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
    ) -> CreationStageResult:
        """Retry a full organism-creation exchange on provider or parse failure."""

        max_attempts = max(1, int(self.evolver_cfg.creation.max_attempts_to_create_organism))
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self.run_creation_stages(
                    design_system_prompt=design_system_prompt,
                    design_user_prompt=design_user_prompt,
                    org_dir=org_dir,
                    organism_id=organism_id,
                    generation=generation,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Creation stages for organism %s attempt %d/%d failed: %s",
                    organism_id,
                    attempt,
                    max_attempts,
                    last_error,
                )

        raise RuntimeError(
            "Failed to generate valid organism after "
            f"{max_attempts} creation attempts: {last_error}"
        )

    def _load_llm_exchange_payloads(self, org_dir: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
        request_path = org_dir / "llm_request.json"
        response_path = org_dir / "llm_response.json"

        request_payload: dict[str, object] = {}
        response_payload: dict[str, object] = {}
        if request_path.exists():
            payload = read_json(request_path)
            if isinstance(payload, dict):
                request_payload = dict(payload)
        if response_path.exists():
            payload = read_json(response_path)
            if isinstance(payload, dict):
                response_payload = dict(payload)
        return request_path, response_path, request_payload, response_payload

    def repair_organism_after_error(
        self,
        *,
        organism: OrganismMeta,
        phase: str,
        experiment_name: str,
        errors: list[dict[str, object]],
    ) -> None:
        """Use the assigned route to repair implementation.py after evaluator failure."""

        route_id = organism.llm_route_id or self.sample_route_id(organism_id=organism.organism_id)
        org_dir = Path(organism.organism_dir)
        system_prompt, user_prompt = build_repair_prompt(
            organism,
            self.prompt_bundle,
            phase=phase,
            experiment_name=experiment_name,
            errors=[dict(entry) for entry in errors],
        )

        request_path, response_path, request_payload, response_payload = self._load_llm_exchange_payloads(org_dir)
        request_payload["route_id"] = route_id
        response_payload["route_id"] = route_id
        repair_requests = list(request_payload.get("repair_attempts", []))
        repair_responses = list(response_payload.get("repair_attempts", []))
        attempt_number = len(repair_requests) + 1
        request_entry = {
            "attempt": attempt_number,
            "route_id": route_id,
            "phase": phase,
            "experiment_name": experiment_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "request": None,
            "status": "in_flight",
            "error_msg": None,
            "errors": [dict(entry) for entry in errors],
        }
        response_entry = {
            "attempt": attempt_number,
            "route_id": route_id,
            "phase": phase,
            "experiment_name": experiment_name,
            "text": None,
            "response": None,
            "usage": None,
            "started_at": None,
            "finished_at": None,
            "status": "awaiting_response",
            "error_msg": None,
        }
        repair_requests.append(request_entry)
        repair_responses.append(response_entry)
        request_payload["repair_attempts"] = repair_requests
        response_payload["repair_attempts"] = repair_responses
        write_json(request_path, request_payload)
        write_json(response_path, response_payload)

        _announce(
            f"organism {organism.organism_id} -> route {route_id}: "
            f"calling repair stage for {phase}/{experiment_name}"
        )
        try:
            repair_response = self._call_llm_stage(
                route_id,
                "repair",
                system_prompt,
                user_prompt,
                organism_id=organism.organism_id,
                generation=organism.generation_created,
                extra_metadata={
                    "phase": phase,
                    "experiment_name": experiment_name,
                    "repair_attempt": attempt_number,
                },
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            request_entry["status"] = "failed"
            request_entry["error_msg"] = error_msg
            response_entry["status"] = "failed"
            response_entry["error_msg"] = error_msg
            write_json(request_path, request_payload)
            write_json(response_path, response_payload)
            _announce(
                f"organism {organism.organism_id} repair stage failed "
                f"(route={route_id}, phase={phase}, experiment={experiment_name}): {error_msg}"
            )
            raise

        repaired_code = self._extract_python(repair_response.text)
        Path(organism.implementation_path).write_text(repaired_code, encoding="utf-8")
        request_entry.update(
            {
                "provider": repair_response.provider,
                "provider_model_id": repair_response.provider_model_id,
                "request": repair_response.raw_request,
                "status": "completed",
                "error_msg": None,
            }
        )
        response_entry.update(
            {
                "provider": repair_response.provider,
                "provider_model_id": repair_response.provider_model_id,
                "text": repaired_code,
                "response": repair_response.raw_response,
                "usage": repair_response.usage,
                "started_at": repair_response.started_at,
                "finished_at": repair_response.finished_at,
                "status": "completed",
                "error_msg": None,
            }
        )
        request_payload["provider"] = repair_response.provider
        request_payload["provider_model_id"] = repair_response.provider_model_id
        response_payload["provider"] = repair_response.provider
        response_payload["provider_model_id"] = repair_response.provider_model_id
        organism.llm_route_id = route_id
        organism.llm_provider = repair_response.provider
        organism.provider_model_id = repair_response.provider_model_id
        write_organism_meta(organism)
        write_json(request_path, request_payload)
        write_json(response_path, response_payload)
        _announce(
            f"organism {organism.organism_id} repair stage returned "
            f"(route={route_id}, phase={phase}, experiment={experiment_name})"
        )

    def generate_seed_organism(
        self,
        island: Island,
        organism_id: str,
        generation: int,
        organism_dir: Path,
    ) -> OrganismMeta:
        """Generate one seed organism for a configured island."""

        seed_operator = SeedOperator(island)

        system_prompt, user_prompt = seed_operator.build_prompts(self.prompt_bundle)
        creation = self.run_creation_stages_with_retries(
            design_system_prompt=system_prompt,
            design_user_prompt=user_prompt,
            org_dir=organism_dir,
            organism_id=organism_id,
            generation=generation,
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
