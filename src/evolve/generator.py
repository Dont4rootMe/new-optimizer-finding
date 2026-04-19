"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

import importlib
import logging
import sys
import time
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
from src.organisms.novelty import (
    NoveltyCheckContext,
    NoveltyRejectionExhaustedError,
    parse_novelty_judgment,
)
from src.organisms.organism import (
    build_repair_prompt,
    build_implementation_prompt_from_design,
    build_organism_from_response,
)


def _single_line(value: object, *, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _prompt_size_summary(system_prompt: str, user_prompt: str) -> str:
    system_chars = len(system_prompt)
    user_chars = len(user_prompt)
    return (
        f"system_chars={system_chars}, user_chars={user_chars}, "
        f"total_chars={system_chars + user_chars}"
    )


def _response_summary(*, text: str, raw_response: object, usage: object | None = None) -> str:
    parts = [f"text_chars={len(text)}"]
    raw_payload = raw_response if isinstance(raw_response, dict) else {}
    message = raw_payload.get("message", {})
    if not isinstance(message, dict):
        message = {}
    content_text = str(message.get("content", "") or "").strip()
    thinking_text = str(message.get("thinking", "") or "").strip()
    if message or content_text or thinking_text:
        parts.append(f"content_chars={len(content_text)}")
        parts.append(f"thinking_chars={len(thinking_text)}")
    done_reason = str(raw_payload.get("done_reason", "") or "").strip()
    if done_reason:
        parts.append(f"done_reason={done_reason!r}")
    usage_payload = usage if isinstance(usage, dict) else {}
    for key in ("prompt_eval_count", "eval_count", "total_duration", "eval_duration"):
        value = usage_payload.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


class CandidateGenerator(BaseLlmGenerator):
    """Generate canonical structured seed organisms from configured prompt assets."""

    def __init__(self, cfg: DictConfig, llm_registry: ApiPlatformRegistry | None = None) -> None:
        registry = llm_registry or ApiPlatformRegistry(cfg)
        self._owns_llm_registry = llm_registry is None
        super().__init__(cfg, registry)
        self.prompt_bundle = load_prompt_bundle(cfg)
        self.hypothesis_schema_provider = self._load_hypothesis_schema_provider()

    def _load_hypothesis_schema_provider(self):
        artifact_cfg = self.evolver_cfg.get("hypothesis_artifact", None)
        if artifact_cfg is None:
            return None
        provider_path = str(artifact_cfg.get("schema_provider", "")).strip()
        if not provider_path:
            return None
        return importlib.import_module(provider_path)

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

        metadata_keys = ",".join(sorted(str(key) for key in metadata.keys()))
        LOGGER.info(
            "Dispatching LLM stage route=%s organism=%s generation=%d stage=%s %s metadata_keys=[%s]",
            route_id,
            organism_id,
            generation,
            stage,
            _prompt_size_summary(system_prompt, user_prompt),
            metadata_keys,
        )
        started_at = time.perf_counter()
        try:
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
        except Exception:
            LOGGER.exception(
                "LLM stage failed route=%s organism=%s generation=%d stage=%s %s metadata_keys=[%s]",
                route_id,
                organism_id,
                generation,
                stage,
                _prompt_size_summary(system_prompt, user_prompt),
                metadata_keys,
            )
            raise
        elapsed_sec = time.perf_counter() - started_at
        LOGGER.info(
            "Completed LLM stage route=%s organism=%s generation=%d stage=%s provider=%s model=%s elapsed_sec=%.2f %s",
            route_id,
            organism_id,
            generation,
            stage,
            response.provider,
            response.provider_model_id,
            elapsed_sec,
            _response_summary(
                text=response.text,
                raw_response=response.raw_response,
                usage=response.usage,
            ),
        )
        return response

    def _resolve_max_novelty_regeneration_attempts(self) -> int:
        value = int(self.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_novelty_rejection)
        if value < 0:
            raise ValueError(
                "evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection must be >= 0"
            )
        return value

    def _run_standard_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
    ) -> CreationStageResult:
        """Run the canonical two-stage design -> implementation exchange."""

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

        try:
            parsed_design = parse_llm_response(design_response.text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_request_payload["design"]["status"] = "failed"
            llm_request_payload["design"]["error_msg"] = error_msg
            llm_response_payload["design"]["status"] = "failed"
            llm_response_payload["design"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            LOGGER.warning(
                "Design parse failed organism=%s route=%s %s preview=%r error=%s",
                organism_id,
                route_id,
                _response_summary(
                    text=design_response.text,
                    raw_response=design_response.raw_response,
                    usage=design_response.usage,
                ),
                _single_line(design_response.text),
                error_msg,
            )
            raise
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
            "text": implementation_response.text,
            "response": implementation_response.raw_response,
            "usage": implementation_response.usage,
            "started_at": implementation_response.started_at,
            "finished_at": implementation_response.finished_at,
            "status": "completed",
            "error_msg": None,
        }
        llm_request_payload["provider"] = implementation_response.provider
        llm_request_payload["provider_model_id"] = implementation_response.provider_model_id
        llm_response_payload["provider"] = implementation_response.provider
        llm_response_payload["provider_model_id"] = implementation_response.provider_model_id
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        try:
            implementation_code = self._extract_python(implementation_response.text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_request_payload["implementation"]["status"] = "failed"
            llm_request_payload["implementation"]["error_msg"] = error_msg
            llm_response_payload["implementation"]["status"] = "failed"
            llm_response_payload["implementation"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            _announce(
                f"organism {organism_id} implementation stage produced unusable code "
                f"(route={route_id}): {error_msg}"
            )
            LOGGER.warning(
                "Implementation extraction failed organism=%s route=%s %s preview=%r error=%s",
                organism_id,
                route_id,
                _response_summary(
                    text=implementation_response.text,
                    raw_response=implementation_response.raw_response,
                    usage=implementation_response.usage,
                ),
                _single_line(implementation_response.text),
                error_msg,
            )
            raise

        llm_response_payload["implementation"]["text"] = implementation_code
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

    def _run_creation_stages_with_novelty(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        novelty_context: NoveltyCheckContext,
        org_dir: Path,
        organism_id: str,
        generation: int,
    ) -> CreationStageResult:
        """Run design -> novelty-check loop -> implementation for non-seed operators."""

        route_id = self.sample_route_id(organism_id=organism_id)
        llm_request_path = org_dir / "llm_request.json"
        llm_response_path = org_dir / "llm_response.json"
        llm_request_payload: dict[str, object] = {
            "route_id": route_id,
            "design": {
                "route_id": route_id,
                "attempt": 0,
                "system_prompt": None,
                "user_prompt": None,
                "request": None,
                "status": "pending",
                "error_msg": None,
            },
            "design_attempts": [],
            "novelty_checks": [],
        }
        llm_response_payload: dict[str, object] = {
            "route_id": route_id,
            "design": {
                "route_id": route_id,
                "attempt": 0,
                "text": None,
                "response": None,
                "usage": None,
                "started_at": None,
                "finished_at": None,
                "status": "pending",
                "error_msg": None,
            },
            "design_attempts": [],
            "novelty_checks": [],
        }
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        max_design_attempts = 1 + self._resolve_max_novelty_regeneration_attempts()
        rejection_feedback: list[str] = []
        prompt_hash_parts: list[str] = []
        design_requests: list[dict[str, object]] = llm_request_payload["design_attempts"]  # type: ignore[assignment]
        design_responses: list[dict[str, object]] = llm_response_payload["design_attempts"]  # type: ignore[assignment]
        novelty_requests: list[dict[str, object]] = llm_request_payload["novelty_checks"]  # type: ignore[assignment]
        novelty_responses: list[dict[str, object]] = llm_response_payload["novelty_checks"]  # type: ignore[assignment]
        accepted_parsed_design: dict[str, str] | None = None
        accepted_design_response = None

        for attempt in range(1, max_design_attempts + 1):
            if attempt == 1:
                current_design_system_prompt = design_system_prompt
                current_design_user_prompt = design_user_prompt
            else:
                current_design_system_prompt, current_design_user_prompt = novelty_context.build_design_prompts(
                    rejection_feedback
                )
            prompt_hash_parts.extend((current_design_system_prompt, current_design_user_prompt))

            request_entry = {
                "attempt": attempt,
                "route_id": route_id,
                "operator": novelty_context.operator,
                "system_prompt": current_design_system_prompt,
                "user_prompt": current_design_user_prompt,
                "request": None,
                "status": "in_flight",
                "error_msg": None,
                "novelty_rejection_feedback": list(rejection_feedback),
            }
            response_entry = {
                "attempt": attempt,
                "route_id": route_id,
                "operator": novelty_context.operator,
                "text": None,
                "response": None,
                "usage": None,
                "started_at": None,
                "finished_at": None,
                "status": "awaiting_response",
                "error_msg": None,
            }
            design_requests.append(request_entry)
            design_responses.append(response_entry)
            llm_request_payload["design"] = {
                "route_id": route_id,
                "attempt": attempt,
                "system_prompt": current_design_system_prompt,
                "user_prompt": current_design_user_prompt,
                "request": None,
                "status": "in_flight",
                "error_msg": None,
            }
            llm_response_payload["design"] = {
                "route_id": route_id,
                "attempt": attempt,
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
                f"organism {organism_id} -> route {route_id}: calling design stage "
                f"(generation={generation}, novelty_attempt={attempt}/{max_design_attempts})"
            )
            try:
                design_response = self._call_llm_stage(
                    route_id,
                    "design",
                    current_design_system_prompt,
                    current_design_user_prompt,
                    organism_id=organism_id,
                    generation=generation,
                    extra_metadata={
                        "operator": novelty_context.operator,
                        "design_attempt": attempt,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                request_entry["status"] = "failed"
                request_entry["error_msg"] = error_msg
                response_entry["status"] = "failed"
                response_entry["error_msg"] = error_msg
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
                f"model={design_response.provider_model_id}, novelty_attempt={attempt})"
            )
            request_entry.update(
                {
                    "provider": design_response.provider,
                    "provider_model_id": design_response.provider_model_id,
                    "request": design_response.raw_request,
                    "status": "completed",
                    "error_msg": None,
                }
            )
            response_entry.update(
                {
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
            )
            llm_request_payload["provider"] = design_response.provider
            llm_request_payload["provider_model_id"] = design_response.provider_model_id
            llm_response_payload["provider"] = design_response.provider
            llm_response_payload["provider_model_id"] = design_response.provider_model_id
            llm_request_payload["design"] = dict(request_entry)
            llm_response_payload["design"] = dict(response_entry)
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            try:
                parsed_design = parse_llm_response(design_response.text)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                request_entry["status"] = "failed"
                request_entry["error_msg"] = error_msg
                response_entry["status"] = "failed"
                response_entry["error_msg"] = error_msg
                llm_request_payload["design"] = dict(request_entry)
                llm_response_payload["design"] = dict(response_entry)
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                LOGGER.warning(
                    "Design parse failed organism=%s route=%s novelty_attempt=%d/%d %s preview=%r error=%s",
                    organism_id,
                    route_id,
                    attempt,
                    max_design_attempts,
                    _response_summary(
                        text=design_response.text,
                        raw_response=design_response.raw_response,
                        usage=design_response.usage,
                    ),
                    _single_line(design_response.text),
                    error_msg,
                )
                raise
            novelty_system_prompt, novelty_user_prompt = novelty_context.build_novelty_prompts(parsed_design)
            prompt_hash_parts.extend((novelty_system_prompt, novelty_user_prompt))

            novelty_request_entry = {
                "attempt": attempt,
                "design_attempt": attempt,
                "route_id": route_id,
                "operator": novelty_context.operator,
                "system_prompt": novelty_system_prompt,
                "user_prompt": novelty_user_prompt,
                "request": None,
                "status": "in_flight",
                "error_msg": None,
            }
            novelty_response_entry = {
                "attempt": attempt,
                "design_attempt": attempt,
                "route_id": route_id,
                "operator": novelty_context.operator,
                "text": None,
                "response": None,
                "usage": None,
                "started_at": None,
                "finished_at": None,
                "status": "awaiting_response",
                "error_msg": None,
                "verdict": None,
                "rejection_reason": None,
            }
            novelty_requests.append(novelty_request_entry)
            novelty_responses.append(novelty_response_entry)
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            _announce(
                f"organism {organism_id} -> route {route_id}: calling novelty_check stage "
                f"(design_attempt={attempt}, operator={novelty_context.operator})"
            )
            try:
                novelty_response = self._call_llm_stage(
                    route_id,
                    "novelty_check",
                    novelty_system_prompt,
                    novelty_user_prompt,
                    organism_id=organism_id,
                    generation=generation,
                    extra_metadata={
                        "operator": novelty_context.operator,
                        "design_attempt": attempt,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                novelty_request_entry["status"] = "failed"
                novelty_request_entry["error_msg"] = error_msg
                novelty_response_entry["status"] = "failed"
                novelty_response_entry["error_msg"] = error_msg
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                _announce(f"organism {organism_id} novelty_check stage failed (route={route_id}): {error_msg}")
                raise

            novelty_request_entry.update(
                {
                    "provider": novelty_response.provider,
                    "provider_model_id": novelty_response.provider_model_id,
                    "request": novelty_response.raw_request,
                    "status": "completed",
                    "error_msg": None,
                }
            )
            novelty_response_entry.update(
                {
                    "provider": novelty_response.provider,
                    "provider_model_id": novelty_response.provider_model_id,
                    "text": novelty_response.text,
                    "response": novelty_response.raw_response,
                    "usage": novelty_response.usage,
                    "started_at": novelty_response.started_at,
                    "finished_at": novelty_response.finished_at,
                    "status": "completed",
                    "error_msg": None,
                }
            )
            llm_request_payload["provider"] = novelty_response.provider
            llm_request_payload["provider_model_id"] = novelty_response.provider_model_id
            llm_response_payload["provider"] = novelty_response.provider
            llm_response_payload["provider_model_id"] = novelty_response.provider_model_id
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            try:
                judgment = parse_novelty_judgment(novelty_response.text)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                novelty_request_entry["status"] = "failed"
                novelty_request_entry["error_msg"] = error_msg
                novelty_response_entry["status"] = "failed"
                novelty_response_entry["error_msg"] = error_msg
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                LOGGER.warning(
                    "Novelty judgment parse failed organism=%s route=%s design_attempt=%d/%d %s preview=%r error=%s",
                    organism_id,
                    route_id,
                    attempt,
                    max_design_attempts,
                    _response_summary(
                        text=novelty_response.text,
                        raw_response=novelty_response.raw_response,
                        usage=novelty_response.usage,
                    ),
                    _single_line(novelty_response.text),
                    error_msg,
                )
                raise
            novelty_request_entry["verdict"] = judgment.verdict
            novelty_request_entry["rejection_reason"] = judgment.rejection_reason
            novelty_response_entry["verdict"] = judgment.verdict
            novelty_response_entry["rejection_reason"] = judgment.rejection_reason
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            if judgment.is_accepted:
                _announce(
                    f"organism {organism_id} novelty_check accepted "
                    f"(route={route_id}, design_attempt={attempt})"
                )
                accepted_parsed_design = parsed_design
                accepted_design_response = design_response
                break

            rejection_reason = judgment.rejection_reason or "Novelty check rejected the candidate."
            rejection_feedback.append(rejection_reason)
            _announce(
                f"organism {organism_id} novelty_check rejected design attempt {attempt}/{max_design_attempts} "
                f"(route={route_id}): {rejection_reason}"
            )
        else:
            accepted_parsed_design = None

        if accepted_parsed_design is None or accepted_design_response is None:
            detail = rejection_feedback[-1] if rejection_feedback else "novelty rejection limit reached"
            raise NoveltyRejectionExhaustedError(
                f"Novelty validation rejected organism after {max_design_attempts} design attempts: {detail}"
            )

        implementation_system_prompt, implementation_user_prompt = build_implementation_prompt_from_design(
            accepted_parsed_design,
            self.prompt_bundle,
        )
        prompt_hash_parts.extend((implementation_system_prompt, implementation_user_prompt))
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
            "text": implementation_response.text,
            "response": implementation_response.raw_response,
            "usage": implementation_response.usage,
            "started_at": implementation_response.started_at,
            "finished_at": implementation_response.finished_at,
            "status": "completed",
            "error_msg": None,
        }
        llm_request_payload["provider"] = implementation_response.provider
        llm_request_payload["provider_model_id"] = implementation_response.provider_model_id
        llm_response_payload["provider"] = implementation_response.provider
        llm_response_payload["provider_model_id"] = implementation_response.provider_model_id
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        try:
            implementation_code = self._extract_python(implementation_response.text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_request_payload["implementation"]["status"] = "failed"
            llm_request_payload["implementation"]["error_msg"] = error_msg
            llm_response_payload["implementation"]["status"] = "failed"
            llm_response_payload["implementation"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            _announce(
                f"organism {organism_id} implementation stage produced unusable code "
                f"(route={route_id}): {error_msg}"
            )
            LOGGER.warning(
                "Implementation extraction failed organism=%s route=%s %s preview=%r error=%s",
                organism_id,
                route_id,
                _response_summary(
                    text=implementation_response.text,
                    raw_response=implementation_response.raw_response,
                    usage=implementation_response.usage,
                ),
                _single_line(implementation_response.text),
                error_msg,
            )
            raise

        llm_response_payload["implementation"]["text"] = implementation_code
        write_json(llm_request_path, llm_request_payload)
        write_json(llm_response_path, llm_response_payload)

        prompt_hash = sha1_text("\n".join(prompt_hash_parts))
        return CreationStageResult(
            parsed_design=accepted_parsed_design,
            implementation_code=implementation_code,
            prompt_hash=prompt_hash,
            llm_route_id=route_id,
            llm_provider=accepted_design_response.provider,
            provider_model_id=accepted_design_response.provider_model_id,
        )

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        novelty_context: NoveltyCheckContext | None = None,
    ) -> CreationStageResult:
        """Run creation stages, optionally inserting a novelty-check loop."""

        if novelty_context is None:
            return self._run_standard_creation_stages(
                design_system_prompt=design_system_prompt,
                design_user_prompt=design_user_prompt,
                org_dir=org_dir,
                organism_id=organism_id,
                generation=generation,
            )
        return self._run_creation_stages_with_novelty(
            design_system_prompt=design_system_prompt,
            design_user_prompt=design_user_prompt,
            novelty_context=novelty_context,
            org_dir=org_dir,
            organism_id=organism_id,
            generation=generation,
        )

    @staticmethod
    def _retry_backoff_sec(attempt: int) -> float:
        """Compute backoff delay for retry *attempt* (1-based, delay applied after failure).

        Schedule: 1 s, 2 s, 3 s, 4 s, 5 s, … (+1 s each step after the second).
        """
        if attempt <= 1:
            return 1.0
        return attempt

    def run_creation_stages_with_retries(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        novelty_context: NoveltyCheckContext | None = None,
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
                    novelty_context=novelty_context,
                )
            except NoveltyRejectionExhaustedError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Creation stages for organism %s attempt %d/%d failed: %s",
                    organism_id,
                    attempt,
                    max_attempts,
                    last_error,
                )
                if attempt < max_attempts:
                    delay = self._retry_backoff_sec(attempt)
                    _announce(
                        f"organism {organism_id} retrying in {delay:.0f}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(delay)

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
            schema_provider=self.hypothesis_schema_provider,
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
                "text": repair_response.text,
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

        try:
            repaired_code = self._extract_python(repair_response.text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            request_entry["status"] = "failed"
            request_entry["error_msg"] = error_msg
            response_entry["status"] = "failed"
            response_entry["error_msg"] = error_msg
            write_json(request_path, request_payload)
            write_json(response_path, response_payload)
            _announce(
                f"organism {organism.organism_id} repair stage produced unusable code "
                f"(route={route_id}, phase={phase}, experiment={experiment_name}): {error_msg}"
            )
            LOGGER.warning(
                "Repair extraction failed organism=%s route=%s phase=%s experiment=%s repair_attempt=%d %s preview=%r error=%s",
                organism.organism_id,
                route_id,
                phase,
                experiment_name,
                attempt_number,
                _response_summary(
                    text=repair_response.text,
                    raw_response=repair_response.raw_response,
                    usage=repair_response.usage,
                ),
                _single_line(repair_response.text),
                error_msg,
            )
            raise

        response_entry["text"] = repaired_code
        Path(organism.implementation_path).write_text(repaired_code, encoding="utf-8")
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
            schema_provider=self.hypothesis_schema_provider,
        )
