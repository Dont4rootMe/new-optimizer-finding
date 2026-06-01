"""Canonical seed-organism generation for the organism-first evolution loop."""

from __future__ import annotations

import ast
import logging
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

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
from src.evolve.prompt_utils import load_prompt_bundle
from src.evolve.storage import read_json, sha1_text, utc_now_iso, write_json, write_organism_meta
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import CreationStageResult, Island, OrganismMeta
from src.organisms.implementation_patch import (
    ImplementationCompilationPlan,
    assemble_implementation_from_patch,
    compute_changed_genome_sections,
    order_changed_sections_by_region_order,
    parse_implementation_scaffold,
    parse_implementation_patch_response,
    resolve_implementation_region_order,
)
from src.organisms.novelty import (
    NoveltyCheckContext,
    NoveltyRejectionExhaustedError,
    parse_novelty_judgment,
)
from src.organisms.organism import (
    build_repair_prompt,
    build_genetic_code_from_design_response,
    build_implementation_prompt_from_design,
    build_implementation_prompt,
    build_organism_from_response,
    format_genetic_code,
    load_expected_core_gene_sections_from_config,
)


@dataclass(frozen=True)
class _PreparedImplementationStage:
    system_prompt: str
    user_prompt: str
    compilation_plan: ImplementationCompilationPlan | None
    base_source_text: str | None


@dataclass(frozen=True)
class _StructuredResponseText:
    full_text: str
    content_text: str
    thinking_text: str
    parse_text: str
    parse_source: str


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


def _normalize_token_usage(usage: object) -> dict[str, int]:
    """Map a provider-specific ``usage`` blob to canonical token counts.

    Different providers expose token counts under different keys:
      * Ollama   -> ``prompt_eval_count`` / ``eval_count``
      * OpenAI   -> ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``
      * Anthropic-> ``input_tokens`` / ``output_tokens``
      * mock     -> ``{}`` (no counts)

    Returns ``{"prompt_tokens", "completion_tokens", "total_tokens"}`` with
    ``total_tokens`` backfilled from prompt+completion when the provider does
    not report a total. All-zero results are returned as-is; the caller skips
    recording them so empty mock usage never pollutes the accounting.
    """

    payload = usage if isinstance(usage, dict) else {}

    def _coerce(*keys: str) -> int:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    prompt = _coerce("prompt_tokens", "prompt_eval_count", "input_tokens")
    completion = _coerce("completion_tokens", "eval_count", "output_tokens")
    total = _coerce("total_tokens")
    if total <= 0:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _structured_response_text(response: object) -> _StructuredResponseText:
    """Return the text that strict structured parsers should consume.

    Some Ollama models return reasoning in message.thinking and the final
    answer in message.content. Strict artifact parsers must consume only the
    final content when it is available; thinking stays available through
    raw_response diagnostics.
    """

    full_text = str(getattr(response, "text", "") or "")
    raw_response = getattr(response, "raw_response", {})
    raw_payload = raw_response if isinstance(raw_response, dict) else {}
    message = raw_payload.get("message", {})
    if not isinstance(message, dict):
        message = {}
    content_text = str(message.get("content", "") or "").strip()
    thinking_text = str(message.get("thinking", "") or "").strip()
    if content_text:
        return _StructuredResponseText(
            full_text=full_text,
            content_text=content_text,
            thinking_text=thinking_text,
            parse_text=content_text,
            parse_source="message.content",
        )
    return _StructuredResponseText(
        full_text=full_text,
        content_text=content_text,
        thinking_text=thinking_text,
        parse_text=full_text,
        parse_source="response.text",
    )


def _structured_response_fields(response_text: _StructuredResponseText) -> dict[str, str]:
    return {
        "content_text": response_text.content_text,
        "thinking_text": response_text.thinking_text,
        "parse_text": response_text.parse_text,
        "parse_source": response_text.parse_source,
    }


def _parse_failure_diagnostics(response_text: _StructuredResponseText) -> str:
    return (
        f"parse_source={response_text.parse_source!r}, "
        f"content_chars={len(response_text.content_text)}, "
        f"thinking_chars={len(response_text.thinking_text)}, "
        f"parse_chars={len(response_text.parse_text)}, "
        f"content_preview={_single_line(response_text.content_text)!r}, "
        f"thinking_preview={_single_line(response_text.thinking_text)!r}, "
        f"full_text_preview={_single_line(response_text.full_text)!r}, "
        f"parse_text_preview={_single_line(response_text.parse_text)!r}"
    )


def _parse_failure_diagnostics_payload(response_text: _StructuredResponseText) -> dict[str, object]:
    return {
        "parse_source": response_text.parse_source,
        "content_chars": len(response_text.content_text),
        "thinking_chars": len(response_text.thinking_text),
        "parse_chars": len(response_text.parse_text),
        "content_preview": _single_line(response_text.content_text, limit=500),
        "thinking_preview": _single_line(response_text.thinking_text, limit=500),
        "full_text_preview": _single_line(response_text.full_text, limit=500),
        "parse_text_preview": _single_line(response_text.parse_text, limit=500),
    }


def _first_non_empty_line(text: str) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def _format_changed_sections_for_prompt(changed_sections: tuple[str, ...]) -> str:
    return "\n".join(changed_sections) if changed_sections else "NONE"


def _format_parsed_design_as_genetic_code_markdown(parsed_design: dict[str, str]) -> str:
    section_names = ("CORE_GENES", "INTERACTION_NOTES", "COMPUTE_NOTES", "CHANGE_DESCRIPTION")
    missing = [name for name in section_names if not str(parsed_design.get(name, "")).strip()]
    if missing:
        raise ValueError(f"Design response is missing required section(s): {', '.join(missing)}")
    return "\n\n".join(f"## {name}\n{str(parsed_design[name]).strip()}" for name in section_names) + "\n"


def _append_rejected_candidate_repair_block(
    user_prompt: str,
    *,
    last_rejected_design: dict[str, str] | None,
    last_rejection_summary: str,
) -> str:
    """Append the previously rejected candidate as a targeted-repair block.

    Without seeing its prior output the model tends to redesign from scratch
    on every retry, which is what produced ~125 of the 313 atcoder-run
    creation failures. Showing the rejected candidate plus the verdict
    converts the retry from "blind redesign" into "patch this specific
    object," which is much cheaper and more likely to recover the lineage.
    """

    if last_rejected_design is None:
        return user_prompt
    try:
        formatted_candidate = _format_parsed_design_as_genetic_code_markdown(last_rejected_design)
    except Exception:  # noqa: BLE001
        return user_prompt
    summary = str(last_rejection_summary).strip() or "(no rejection text available)"
    repair_block = (
        "\n\n=== PRIOR CANDIDATE TO REPAIR ===\n"
        "Your previous design (below) was rejected for the reason listed above. "
        "Your job now is to revise THIS candidate to address that critique. "
        "Do not redesign from scratch; preserve every section the critique did not flag, "
        "and edit the smallest set of bullets needed to remove the rejection.\n\n"
        f"Critique to address: {summary}\n\n"
        f"{formatted_candidate}"
    )
    return user_prompt + repair_block


def _validate_assembled_python_source(source_text: str) -> str:
    """Validate already-assembled Python without altering its bytes."""

    if not str(source_text).strip():
        raise ValueError("Assembled implementation source is empty.")
    try:
        ast.parse(source_text)
    except SyntaxError as exc:
        LOGGER.warning("Assembled implementation has syntax error: %s (line %s)", exc.msg, exc.lineno)
        raise ValueError(
            f"Assembled implementation is syntactically invalid Python: {exc.msg} at line {exc.lineno}"
        ) from exc
    return source_text


def _implementation_request_metadata(
    plan: ImplementationCompilationPlan | None,
) -> dict[str, object]:
    if plan is None:
        return {
            "implementation_strategy": "legacy_full_source",
            "changed_sections": None,
        }
    payload: dict[str, object] = {
        "implementation_strategy": plan.strategy,
        "changed_sections": list(plan.changed_sections),
    }
    if plan.compilation_mode is not None:
        payload["compilation_mode"] = plan.compilation_mode
    return payload


class CandidateGenerator(BaseLlmGenerator):
    """Generate canonical structured seed organisms from configured prompt assets."""

    def __init__(self, cfg: DictConfig, llm_registry: ApiPlatformRegistry | None = None) -> None:
        registry = llm_registry or ApiPlatformRegistry(cfg)
        self._owns_llm_registry = llm_registry is None
        super().__init__(cfg, registry)
        self.prompt_bundle = load_prompt_bundle(cfg)
        self.expected_core_gene_sections = load_expected_core_gene_sections_from_config(cfg)
        self._implementation_region_order_error: ValueError | None = None
        self.expected_implementation_regions = self._load_expected_implementation_regions()
        # Per-organism LLM token accounting. Organisms are created on parallel
        # worker threads (one per organism), so the accumulator is keyed by
        # organism_id and guarded by a lock. Within one organism the stages
        # run sequentially on a single thread. The bucket is *popped* when the
        # organism meta is finalised (creation) and again after each repair
        # attempt, so the dict never holds stale cross-phase data.
        self._token_usage_lock = threading.Lock()
        self._token_usage_by_organism: dict[str, dict[str, dict[str, int]]] = {}

    def _load_expected_implementation_regions(self) -> tuple[str, ...] | None:
        expected_sections = self.expected_core_gene_sections
        if not expected_sections:
            return None
        if not self.prompt_bundle.implementation_template:
            return None
        try:
            return resolve_implementation_region_order(
                self.prompt_bundle.implementation_template,
                expected_section_names=expected_sections,
            )
        except ValueError as exc:
            self._implementation_region_order_error = exc
            return None

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
        self._record_token_usage(organism_id=organism_id, route_id=route_id, usage=response.usage)
        return response

    def _record_token_usage(self, *, organism_id: str, route_id: str, usage: object) -> None:
        """Accumulate one LLM call's token usage into the per-organism bucket."""

        normalized = _normalize_token_usage(usage)
        if not any(normalized.values()):
            # Mock provider / providers that do not report usage. Nothing to
            # record — skip so empty calls don't create noise buckets.
            return
        with self._token_usage_lock:
            org_bucket = self._token_usage_by_organism.setdefault(organism_id, {})
            route_bucket = org_bucket.setdefault(
                route_id,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
            )
            route_bucket["prompt_tokens"] += normalized["prompt_tokens"]
            route_bucket["completion_tokens"] += normalized["completion_tokens"]
            route_bucket["total_tokens"] += normalized["total_tokens"]
            route_bucket["calls"] += 1

    def pop_token_usage(self, organism_id: str) -> dict[str, dict[str, int]]:
        """Return and clear the accumulated token usage for one organism.

        Called once when the organism meta is built (capturing all creation
        stages) and again after each repair attempt (capturing that attempt's
        delta). Returns ``{}`` when the organism made no recorded LLM calls
        (e.g. seed-copy organisms or the mock provider).
        """

        with self._token_usage_lock:
            return self._token_usage_by_organism.pop(organism_id, {})

    def merge_token_usage_into_meta(self, organism: OrganismMeta) -> bool:
        """Fold any pending token-usage delta for ``organism`` into its meta.

        Additive merge keyed by route_id, so it is correct whether the meta's
        existing ``token_usage`` came from this process (creation) or was
        reloaded from disk on resume. Returns ``True`` when the meta changed.
        """

        delta = self.pop_token_usage(organism.organism_id)
        if not delta:
            return False
        merged: dict[str, dict[str, int]] = {
            str(route): dict(counts) for route, counts in organism.token_usage.items()
        }
        for route, counts in delta.items():
            route_bucket = merged.setdefault(
                route,
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
            )
            for metric, value in counts.items():
                route_bucket[metric] = int(route_bucket.get(metric, 0)) + int(value)
        organism.token_usage = merged
        return True

    def _resolve_max_novelty_regeneration_attempts(self) -> int:
        value = int(self.evolver_cfg.creation.max_attempts_to_regenerate_organism_after_novelty_rejection)
        if value < 0:
            raise ValueError(
                "evolver.creation.max_attempts_to_regenerate_organism_after_novelty_rejection must be >= 0"
            )
        return value

    def uses_section_patch_compilation(self) -> bool:
        """Return true when this prompt bundle uses section-aware implementation prompts."""

        migrated = bool(
            self.prompt_bundle.genome_schema
            or self.expected_core_gene_sections
            or "## COMPILATION_MODE" in self.prompt_bundle.implementation_system
        )
        if not migrated:
            return False
        if not self.expected_core_gene_sections:
            raise ValueError("Section-aware implementation compilation requires a parseable genome schema.")
        if self._implementation_region_order_error is not None:
            raise ValueError(
                "Section-aware implementation scaffold is invalid: "
                f"{self._implementation_region_order_error}"
            ) from self._implementation_region_order_error
        if not self.expected_implementation_regions:
            raise ValueError("Section-aware implementation compilation requires a parseable implementation scaffold.")
        if (
            "## COMPILATION_MODE" not in self.prompt_bundle.implementation_system
            and not self._uses_single_file_implementation_rewrite_contract()
        ):
            raise ValueError(
                "Section-aware implementation compilation requires an implementation system prompt "
                "with a supported implementation contract."
            )
        if self._uses_single_file_implementation_rewrite_contract():
            template_text = self.prompt_bundle.implementation_template
            if "EVOLVE-BLOCK-START" not in template_text or "EVOLVE-BLOCK-END" not in template_text:
                raise ValueError(
                    "Section-aware implementation scaffold is invalid: full-source rewrite templates "
                    "must declare EVOLVE-BLOCK-START and EVOLVE-BLOCK-END."
                )
        else:
            try:
                parse_implementation_scaffold(
                    self.prompt_bundle.implementation_template,
                    expected_region_names=self.expected_implementation_regions,
                )
            except ValueError as exc:
                raise ValueError(f"Section-aware implementation scaffold is invalid: {exc}") from exc
        return True

    def _uses_single_file_implementation_rewrite_contract(self) -> bool:
        """Return true when implementation prompts always expect a full Python file."""

        sentinel = "Single rewrite contract:"
        return sentinel in self.prompt_bundle.implementation_system

    def _prepare_implementation_stage(
        self,
        parsed_design: dict[str, str],
        *,
        implementation_base_parent: OrganismMeta | None = None,
    ) -> _PreparedImplementationStage:
        if not self.uses_section_patch_compilation():
            system_prompt, user_prompt = build_implementation_prompt_from_design(
                parsed_design,
                self.prompt_bundle,
                expected_core_gene_sections=self.expected_core_gene_sections,
            )
            return _PreparedImplementationStage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                compilation_plan=None,
                base_source_text=None,
            )

        expected_sections = self.expected_core_gene_sections
        implementation_regions = self.expected_implementation_regions
        if expected_sections is None or implementation_regions is None:
            raise ValueError("Section patch compilation requires expected CORE_GENES sections.")

        child_genetic_code = build_genetic_code_from_design_response(
            parsed_design,
            expected_core_gene_sections=expected_sections,
        )
        child_genetic_code_text = _format_parsed_design_as_genetic_code_markdown(parsed_design)
        change_description = parsed_design.get("CHANGE_DESCRIPTION", "").strip()
        if not change_description:
            raise ValueError("Implementation patch compilation requires CHANGE_DESCRIPTION.")
        full_source_rewrite = self._uses_single_file_implementation_rewrite_contract()

        if implementation_base_parent is None:
            compilation_mode: str | None = None if full_source_rewrite else "FULL"
            strategy = "full_source_rewrite" if full_source_rewrite else "section_patch_artifact"
            changed_sections = implementation_regions
            base_parent_genetic_code = "NONE"
            base_parent_implementation = "NONE"
            base_source_text = None
            maternal_base_required = False
        else:
            compilation_mode = None if full_source_rewrite else "PATCH"
            strategy = "full_source_rewrite" if full_source_rewrite else "section_patch_artifact"
            base_genetic_code_path = Path(implementation_base_parent.genetic_code_path)
            base_implementation_path = Path(implementation_base_parent.implementation_path)
            if not base_genetic_code_path.exists():
                raise FileNotFoundError(f"Maternal base genetic code was not found: {base_genetic_code_path}")
            if not base_implementation_path.exists():
                raise FileNotFoundError(f"Maternal base implementation was not found: {base_implementation_path}")
            base_parent_genetic_code = base_genetic_code_path.read_text(encoding="utf-8")
            base_source_text = base_implementation_path.read_text(encoding="utf-8")
            base_parent_implementation = base_source_text
            changed_genome_sections = compute_changed_genome_sections(
                base_parent_genetic_code,
                child_genetic_code_text,
                expected_section_names=expected_sections,
            )
            changed_sections = order_changed_sections_by_region_order(
                changed_genome_sections,
                region_order=implementation_regions,
            )
            maternal_base_required = True
            # PATCH mode loses to FULL mode when nearly every region is touched: the
            # token budget is the same, the parser is more brittle, and the LLM has
            # to track changed-section bookkeeping it would otherwise skip. Snap to
            # FULL once at most one region would stay unchanged.
            #
            # NOTE: ``base_parent_implementation`` is intentionally kept
            # populated even in FULL mode. The earlier version cleared it
            # to "NONE" here, which forced the implementer-LLM to
            # re-synthesize the entire Python from the prose CORE_GENES
            # alone — losing the parent's helpers, idioms, and bug-fixed
            # corner cases. With the parent code in scope the LLM can
            # diff-from-baseline even in FULL mode (the FULL/PATCH switch
            # is only about output format, not about whether the LLM may
            # consult the parent). ``base_source_text`` and
            # ``maternal_base_required`` are cleared because the
            # implementation-patch parser uses them for region-level
            # post-processing, which FULL mode doesn't need.
            if not full_source_rewrite and len(changed_sections) >= len(implementation_regions) - 1:
                compilation_mode = "FULL"
                changed_sections = implementation_regions
                base_source_text = None
                maternal_base_required = False

        system_prompt, user_prompt = build_implementation_prompt(
            genetic_code=child_genetic_code,
            change_description=change_description,
            prompts=self.prompt_bundle,
            compilation_mode=compilation_mode or "FULL",
            changed_sections=_format_changed_sections_for_prompt(changed_sections),
            base_parent_genetic_code=base_parent_genetic_code,
            base_parent_implementation=base_parent_implementation,
        )
        return _PreparedImplementationStage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            compilation_plan=ImplementationCompilationPlan(
                strategy=strategy,
                compilation_mode=compilation_mode,  # type: ignore[arg-type]
                changed_sections=tuple(changed_sections),
                maternal_base_required=maternal_base_required,
            ),
            base_source_text=base_source_text,
        )

    def _extract_implementation_stage_code(
        self,
        response_text: str,
        *,
        prepared: _PreparedImplementationStage,
    ) -> str:
        if prepared.compilation_plan is None:
            return self._extract_python(response_text)

        expected_sections = self.expected_implementation_regions
        if expected_sections is None:
            raise ValueError("Section patch compilation requires expected CORE_GENES sections.")
        plan = prepared.compilation_plan
        if plan.strategy == "full_source_rewrite":
            return self._extract_python(response_text)

        expected_patch_regions = (
            expected_sections
            if plan.compilation_mode == "FULL"
            else plan.changed_sections
        )
        patch = parse_implementation_patch_response(
            response_text,
            expected_mode=plan.compilation_mode,
            expected_region_names=expected_patch_regions,
        )
        assembled = assemble_implementation_from_patch(
            scaffold_text=self.prompt_bundle.implementation_template,
            patch=patch,
            expected_region_names=expected_sections,
            base_source_text=prepared.base_source_text,
        )
        return _validate_assembled_python_source(assembled)

    def _resolve_max_implementation_stage_attempts(self) -> int:
        repair_budget = int(
            self.evolver_cfg.creation.get(
                "max_attempts_to_repair_organism_after_error",
                0,
            )
        )
        if repair_budget < 0:
            raise ValueError("evolver.creation.max_attempts_to_repair_organism_after_error must be >= 0")
        return 1 + repair_budget

    def _implementation_contract_metadata(
        self,
        prepared: _PreparedImplementationStage,
    ) -> dict[str, object]:
        if prepared.compilation_plan is None:
            return {
                "expected_mode": "legacy_full_source",
                "expected_regions": None,
            }

        expected_regions = (
            list(self.expected_implementation_regions or ())
            if prepared.compilation_plan.compilation_mode == "FULL"
            else list(prepared.compilation_plan.changed_sections)
        )
        return {
            "expected_mode": prepared.compilation_plan.compilation_mode,
            "expected_regions": expected_regions,
        }

    def _implementation_failure_diagnostics(
        self,
        *,
        prepared: _PreparedImplementationStage,
        response_text: _StructuredResponseText,
        error_msg: str,
    ) -> dict[str, object]:
        return {
            **self._implementation_contract_metadata(prepared),
            **_parse_failure_diagnostics_payload(response_text),
            "first_non_empty_line": _first_non_empty_line(response_text.parse_text),
            "error_msg": error_msg,
        }

    def _run_implementation_stage_with_retries(
        self,
        *,
        route_id: str,
        prepared: _PreparedImplementationStage,
        llm_request_path: Path,
        llm_response_path: Path,
        llm_request_payload: dict[str, object],
        llm_response_payload: dict[str, object],
        organism_id: str,
        generation: int,
    ) -> str:
        implementation_system_prompt = prepared.system_prompt
        implementation_user_prompt = prepared.user_prompt
        max_attempts = self._resolve_max_implementation_stage_attempts()
        request_attempts = llm_request_payload.setdefault("implementation_attempts", [])
        response_attempts = llm_response_payload.setdefault("implementation_attempts", [])
        if not isinstance(request_attempts, list):
            request_attempts = []
            llm_request_payload["implementation_attempts"] = request_attempts
        if not isinstance(response_attempts, list):
            response_attempts = []
            llm_response_payload["implementation_attempts"] = response_attempts

        last_error: BaseException | None = None
        last_error_msg: str | None = None
        for attempt in range(1, max_attempts + 1):
            request_entry = {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "route_id": route_id,
                "system_prompt": implementation_system_prompt,
                "user_prompt": implementation_user_prompt,
                "request": None,
                "status": "in_flight",
                "error_msg": None,
                **_implementation_request_metadata(prepared.compilation_plan),
                **self._implementation_contract_metadata(prepared),
            }
            response_entry = {
                "attempt": attempt,
                "max_attempts": max_attempts,
                "route_id": route_id,
                "text": None,
                "raw_text": None,
                "response": None,
                "usage": None,
                "started_at": None,
                "finished_at": None,
                "status": "awaiting_response",
                "error_msg": None,
                **self._implementation_contract_metadata(prepared),
            }
            request_attempts.append(request_entry)
            response_attempts.append(response_entry)
            llm_request_payload["implementation"] = request_entry
            llm_response_payload["implementation"] = response_entry
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            _announce(
                f"organism {organism_id} -> route {route_id}: calling implementation stage "
                f"(attempt={attempt}/{max_attempts})"
            )
            try:
                implementation_response = self._call_llm_stage(
                    route_id,
                    "implementation",
                    implementation_system_prompt,
                    implementation_user_prompt,
                    organism_id=organism_id,
                    generation=generation,
                    extra_metadata={
                        "implementation_template": self.prompt_bundle.implementation_template,
                        "implementation_attempt": attempt,
                        "max_implementation_attempts": max_attempts,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_error_msg = f"{type(exc).__name__}: {exc}"
                request_entry["status"] = "failed"
                request_entry["error_msg"] = last_error_msg
                response_entry["status"] = "failed"
                response_entry["error_msg"] = last_error_msg
                response_entry["error_kind"] = "provider_failure"
                llm_request_payload["implementation"] = request_entry
                llm_response_payload["implementation"] = response_entry
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                _announce(
                    f"organism {organism_id} implementation stage failed "
                    f"(route={route_id}, attempt={attempt}/{max_attempts}): {last_error_msg}"
                )
                if attempt < max_attempts:
                    continue
                raise

            implementation_text = _structured_response_text(implementation_response)
            request_entry.update(
                {
                    "provider": implementation_response.provider,
                    "provider_model_id": implementation_response.provider_model_id,
                    "request": implementation_response.raw_request,
                    "status": "completed",
                    "error_msg": None,
                }
            )
            response_entry.update(
                {
                    "provider": implementation_response.provider,
                    "provider_model_id": implementation_response.provider_model_id,
                    "text": implementation_response.text,
                    "raw_text": implementation_response.text,
                    **_structured_response_fields(implementation_text),
                    "response": implementation_response.raw_response,
                    "usage": implementation_response.usage,
                    "started_at": implementation_response.started_at,
                    "finished_at": implementation_response.finished_at,
                    "status": "completed",
                    "error_msg": None,
                }
            )
            llm_request_payload["provider"] = implementation_response.provider
            llm_request_payload["provider_model_id"] = implementation_response.provider_model_id
            llm_response_payload["provider"] = implementation_response.provider
            llm_response_payload["provider_model_id"] = implementation_response.provider_model_id
            llm_request_payload["implementation"] = request_entry
            llm_response_payload["implementation"] = response_entry
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)

            try:
                implementation_code = self._extract_implementation_stage_code(
                    implementation_text.parse_text,
                    prepared=prepared,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                last_error_msg = f"{type(exc).__name__}: {exc}"
                diagnostics = self._implementation_failure_diagnostics(
                    prepared=prepared,
                    response_text=implementation_text,
                    error_msg=last_error_msg,
                )
                response_entry["error_kind"] = "implementation_extract_failed"
                response_entry["failure_diagnostics"] = diagnostics
                request_entry["status"] = "failed"
                request_entry["error_msg"] = last_error_msg
                response_entry["status"] = "failed"
                response_entry["error_msg"] = last_error_msg
                llm_request_payload["implementation"] = request_entry
                llm_response_payload["implementation"] = response_entry
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                _announce(
                    f"organism {organism_id} implementation stage produced unusable code "
                    f"(route={route_id}, attempt={attempt}/{max_attempts}): {last_error_msg}"
                )
                LOGGER.warning(
                    "Implementation extraction failed organism=%s route=%s attempt=%d/%d expected_mode=%s "
                    "expected_regions=%s first_non_empty_line=%r %s %s error=%s",
                    organism_id,
                    route_id,
                    attempt,
                    max_attempts,
                    diagnostics["expected_mode"],
                    diagnostics["expected_regions"],
                    diagnostics["first_non_empty_line"],
                    _response_summary(
                        text=implementation_response.text,
                        raw_response=implementation_response.raw_response,
                        usage=implementation_response.usage,
                    ),
                    _parse_failure_diagnostics(implementation_text),
                    last_error_msg,
                )
                if attempt < max_attempts:
                    continue
                raise

            response_entry["text"] = implementation_code
            response_entry["status"] = "completed"
            response_entry["error_msg"] = None
            llm_request_payload["implementation"] = request_entry
            llm_response_payload["implementation"] = response_entry
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            return implementation_code

        raise RuntimeError(
            "Failed to extract valid implementation after "
            f"{max_attempts} implementation attempts: {last_error_msg}"
        ) from last_error

    def _run_standard_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        implementation_base_parent: OrganismMeta | None = None,
        pipeline_state_callback: Callable[[str], None] | None = None,
    ) -> CreationStageResult:
        """Run the canonical two-stage design -> implementation exchange."""

        # In pipeline mode different stages of the same organism may route
        # to different models. The "design" route plays the role of the
        # legacy single-route id (logged at the top of the organism JSON
        # and on ``OrganismMeta.llm_route_id``); the implementation route
        # is sampled separately and the per-stage JSON records it.
        route_id = self.sample_route_id(organism_id=organism_id, stage="design")
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
            llm_response_payload["design"]["error_kind"] = "provider_failure"
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            _announce(f"organism {organism_id} design stage failed (route={route_id}): {error_msg}")
            raise
        _announce(
            f"organism {organism_id} design stage returned "
            f"(route={route_id}, provider={design_response.provider}, "
            f"model={design_response.provider_model_id})"
        )
        design_text = _structured_response_text(design_response)
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
            **_structured_response_fields(design_text),
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
            parsed_design = parse_llm_response(design_text.parse_text)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            llm_response_payload["design"]["error_kind"] = "design_contract_parse_failed"
            llm_response_payload["design"]["expected_contract"] = (
                "top-level sections CORE_GENES, INTERACTION_NOTES, COMPUTE_NOTES, CHANGE_DESCRIPTION"
            )
            llm_response_payload["design"]["failure_diagnostics"] = _parse_failure_diagnostics_payload(design_text)
            llm_request_payload["design"]["status"] = "failed"
            llm_request_payload["design"]["error_msg"] = error_msg
            llm_response_payload["design"]["status"] = "failed"
            llm_response_payload["design"]["error_msg"] = error_msg
            write_json(llm_request_path, llm_request_payload)
            write_json(llm_response_path, llm_response_payload)
            LOGGER.warning(
                "Design parse failed organism=%s route=%s expected_contract=%s %s %s error=%s",
                organism_id,
                route_id,
                "top-level sections CORE_GENES, INTERACTION_NOTES, COMPUTE_NOTES, CHANGE_DESCRIPTION",
                _response_summary(
                    text=design_response.text,
                    raw_response=design_response.raw_response,
                    usage=design_response.usage,
                ),
                _parse_failure_diagnostics(design_text),
                error_msg,
            )
            raise
        prepared_implementation = self._prepare_implementation_stage(
            parsed_design,
            implementation_base_parent=implementation_base_parent,
        )
        implementation_system_prompt = prepared_implementation.system_prompt
        implementation_user_prompt = prepared_implementation.user_prompt
        implementation_code = self._run_implementation_stage_with_retries(
            route_id=self.sample_route_id(
                organism_id=organism_id, stage="implementation"
            ),
            prepared=prepared_implementation,
            llm_request_path=llm_request_path,
            llm_response_path=llm_response_path,
            llm_request_payload=llm_request_payload,
            llm_response_payload=llm_response_payload,
            organism_id=organism_id,
            generation=generation,
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
        implementation_base_parent: OrganismMeta | None = None,
        pipeline_state_callback: Callable[[str], None] | None = None,
    ) -> CreationStageResult:
        """Run design -> novelty-check loop -> implementation for non-seed operators."""

        # ``route_id`` plays the legacy single-route role (organism-level
        # log entry, ``OrganismMeta.llm_route_id``). In pipeline mode the
        # novelty + implementation stages route independently — see the
        # per-stage ``sample_route_id`` calls below.
        route_id = self.sample_route_id(organism_id=organism_id, stage="design")
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
        last_rejected_design: dict[str, str] | None = None
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
                current_design_user_prompt = _append_rejected_candidate_repair_block(
                    current_design_user_prompt,
                    last_rejected_design=last_rejected_design,
                    last_rejection_summary=rejection_feedback[-1] if rejection_feedback else "",
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
                response_entry["error_kind"] = "provider_failure"
                llm_request_payload["design"]["status"] = "failed"
                llm_request_payload["design"]["error_msg"] = error_msg
                llm_response_payload["design"]["status"] = "failed"
                llm_response_payload["design"]["error_msg"] = error_msg
                llm_response_payload["design"]["error_kind"] = "provider_failure"
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                _announce(f"organism {organism_id} design stage failed (route={route_id}): {error_msg}")
                raise

            _announce(
                f"organism {organism_id} design stage returned "
                f"(route={route_id}, provider={design_response.provider}, "
                f"model={design_response.provider_model_id}, novelty_attempt={attempt})"
            )
            design_text = _structured_response_text(design_response)
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
                    **_structured_response_fields(design_text),
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
                parsed_design = parse_llm_response(design_text.parse_text)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                response_entry["error_kind"] = "design_contract_parse_failed"
                response_entry["expected_contract"] = (
                    "top-level sections CORE_GENES, INTERACTION_NOTES, COMPUTE_NOTES, CHANGE_DESCRIPTION"
                )
                response_entry["failure_diagnostics"] = _parse_failure_diagnostics_payload(design_text)
                request_entry["status"] = "failed"
                request_entry["error_msg"] = error_msg
                response_entry["status"] = "failed"
                response_entry["error_msg"] = error_msg
                llm_request_payload["design"] = dict(request_entry)
                llm_response_payload["design"] = dict(response_entry)
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                LOGGER.warning(
                    "Design parse failed organism=%s route=%s novelty_attempt=%d/%d expected_contract=%s %s %s error=%s",
                    organism_id,
                    route_id,
                    attempt,
                    max_design_attempts,
                    "top-level sections CORE_GENES, INTERACTION_NOTES, COMPUTE_NOTES, CHANGE_DESCRIPTION",
                    _response_summary(
                        text=design_response.text,
                        raw_response=design_response.raw_response,
                        usage=design_response.usage,
                    ),
                    _parse_failure_diagnostics(design_text),
                    error_msg,
                )
                raise
            novelty_system_prompt, novelty_user_prompt = novelty_context.build_novelty_prompts(parsed_design)
            prompt_hash_parts.extend((novelty_system_prompt, novelty_user_prompt))

            # In pipeline mode the novelty stage may route to a different
            # model than design; resolve once here and use the same route
            # for both the JSON record and the LLM call so the persisted
            # trace matches what actually executed.
            novelty_route_id = self.sample_route_id(
                organism_id=organism_id, stage="novelty_check"
            )
            novelty_request_entry = {
                "attempt": attempt,
                "design_attempt": attempt,
                "route_id": novelty_route_id,
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
                "route_id": novelty_route_id,
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
                f"organism {organism_id} -> route {novelty_route_id}: calling novelty_check stage "
                f"(design_attempt={attempt}, operator={novelty_context.operator})"
            )
            try:
                novelty_response = self._call_llm_stage(
                    novelty_route_id,
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
                novelty_response_entry["error_kind"] = "provider_failure"
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                _announce(f"organism {organism_id} novelty_check stage failed (route={novelty_route_id}): {error_msg}")
                raise

            novelty_text = _structured_response_text(novelty_response)
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
                    **_structured_response_fields(novelty_text),
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
                judgment = parse_novelty_judgment(
                    novelty_text.parse_text,
                    expected_section_names=self.expected_core_gene_sections,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"{type(exc).__name__}: {exc}"
                novelty_response_entry["error_kind"] = "novelty_judgment_parse_failed"
                novelty_response_entry["expected_contract"] = "NOVELTY_VERDICT, REJECTION_REASON, SECTIONS_AT_ISSUE"
                novelty_response_entry["failure_diagnostics"] = _parse_failure_diagnostics_payload(novelty_text)
                novelty_request_entry["status"] = "failed"
                novelty_request_entry["error_msg"] = error_msg
                novelty_response_entry["status"] = "failed"
                novelty_response_entry["error_msg"] = error_msg
                write_json(llm_request_path, llm_request_payload)
                write_json(llm_response_path, llm_response_payload)
                LOGGER.warning(
                    "Novelty judgment parse failed organism=%s route=%s design_attempt=%d/%d expected_contract=%s %s %s error=%s",
                    organism_id,
                    route_id,
                    attempt,
                    max_design_attempts,
                    "NOVELTY_VERDICT, REJECTION_REASON, SECTIONS_AT_ISSUE",
                    _response_summary(
                        text=novelty_response.text,
                        raw_response=novelty_response.raw_response,
                        usage=novelty_response.usage,
                    ),
                    _parse_failure_diagnostics(novelty_text),
                    error_msg,
                )
                raise
            novelty_request_entry["verdict"] = judgment.verdict
            novelty_request_entry["rejection_reason"] = judgment.rejection_reason
            novelty_request_entry["sections_at_issue"] = list(judgment.sections_at_issue)
            novelty_response_entry["verdict"] = judgment.verdict
            novelty_response_entry["rejection_reason"] = judgment.rejection_reason
            novelty_response_entry["sections_at_issue"] = list(judgment.sections_at_issue)
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
            last_rejected_design = parsed_design
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

        prepared_implementation = self._prepare_implementation_stage(
            accepted_parsed_design,
            implementation_base_parent=implementation_base_parent,
        )
        implementation_system_prompt = prepared_implementation.system_prompt
        implementation_user_prompt = prepared_implementation.user_prompt
        prompt_hash_parts.extend((implementation_system_prompt, implementation_user_prompt))
        implementation_code = self._run_implementation_stage_with_retries(
            route_id=self.sample_route_id(
                organism_id=organism_id, stage="implementation"
            ),
            prepared=prepared_implementation,
            llm_request_path=llm_request_path,
            llm_response_path=llm_response_path,
            llm_request_payload=llm_request_payload,
            llm_response_payload=llm_response_payload,
            organism_id=organism_id,
            generation=generation,
        )

        prompt_hash = sha1_text("\n".join(prompt_hash_parts))
        return CreationStageResult(
            parsed_design=accepted_parsed_design,
            implementation_code=implementation_code,
            prompt_hash=prompt_hash,
            llm_route_id=route_id,
            llm_provider=accepted_design_response.provider,
            provider_model_id=accepted_design_response.provider_model_id,
        )

    def run_rationalization_stage(
        self,
        *,
        rationalization_system: str,
        rationalization_user: str,
        organism_id: str,
        generation: int,
        operator: str,
        org_dir: Path | None = None,
    ) -> str | None:
        """Run Step 1 of the two-step design pipeline.

        Returns the raw rationalization text (six ``## ``-headered prose
        sections per the design contract) or ``None`` when the call fails.

        Persistence: when ``org_dir`` is given, a sibling file
        ``llm_rationalization.json`` is written alongside the organism's
        canonical ``llm_request.json`` so post-mortem dumps can read Step 1
        independently of Step 2. Splitting it from ``llm_request.json``
        avoids a read-modify-write merge with the later design stage that
        overwrites that file wholesale.

        Soft-fails: any exception is logged and ``None`` is returned, which
        callers interpret as "single-call mode fallback".
        """

        from src.organisms.rationalization import (
            parse_rationalization_response,
            rationalization_summary,
        )

        route_id = self.sample_route_id(
            organism_id=organism_id, stage="design_rationalization"
        )
        _announce(
            f"organism {organism_id} -> route {route_id}: calling rationalization stage "
            f"(generation={generation}, operator={operator})"
        )
        started_at = utc_now_iso()
        try:
            response = self._call_llm_stage(
                route_id,
                "design_rationalization",
                rationalization_system,
                rationalization_user,
                organism_id=organism_id,
                generation=generation,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "Rationalization stage failed organism=%s generation=%d operator=%s; "
                "falling back to single-call formalization",
                organism_id,
                generation,
                operator,
            )
            return None
        finished_at = utc_now_iso()
        # ``_structured_response_text`` returns a frozen dataclass whose
        # default ``__str__`` is the dataclass repr — using ``str(wrapper)``
        # would inject ``_StructuredResponseText(full_text=..., content_text=...)``
        # into both the persisted JSON and the Step 2 ``{rationalization}``
        # prompt placeholder, derailing the formalizer's parse-rule following.
        # Always read ``.parse_text`` (the cleaned final answer when present,
        # otherwise the raw response text).
        rationale_text = _structured_response_text(response).parse_text
        parsed = parse_rationalization_response(rationale_text)
        _announce(
            f"organism {organism_id} rationalization stage returned "
            f"(route={route_id}, provider={response.provider}, "
            f"model={response.provider_model_id}, "
            f"actionable={parsed.has_actionable_directive})"
        )

        if org_dir is not None:
            try:
                rationalization_payload = {
                    "route_id": route_id,
                    "operator": operator,
                    "system_prompt": rationalization_system,
                    "user_prompt": rationalization_user,
                    "text": rationale_text,
                    "parsed": rationalization_summary(parsed),
                    "provider": response.provider,
                    "provider_model_id": response.provider_model_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "completed",
                }
                write_json(org_dir / "llm_rationalization.json", rationalization_payload)
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "Failed to persist rationalization for organism %s; continuing", organism_id
                )

        return rationale_text

    def run_creation_stages(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        novelty_context: NoveltyCheckContext | None = None,
        implementation_base_parent: OrganismMeta | None = None,
        pipeline_state_callback: Callable[[str], None] | None = None,
    ) -> CreationStageResult:
        """Run creation stages, optionally inserting a novelty-check loop."""

        if novelty_context is None:
            return self._run_standard_creation_stages(
                design_system_prompt=design_system_prompt,
                design_user_prompt=design_user_prompt,
                org_dir=org_dir,
                organism_id=organism_id,
                generation=generation,
                implementation_base_parent=implementation_base_parent,
                pipeline_state_callback=pipeline_state_callback,
            )
        return self._run_creation_stages_with_novelty(
            design_system_prompt=design_system_prompt,
            design_user_prompt=design_user_prompt,
            novelty_context=novelty_context,
            org_dir=org_dir,
            organism_id=organism_id,
            generation=generation,
            implementation_base_parent=implementation_base_parent,
            pipeline_state_callback=pipeline_state_callback,
        )

    def run_creation_stages_with_retries(
        self,
        *,
        design_system_prompt: str,
        design_user_prompt: str,
        org_dir: Path,
        organism_id: str,
        generation: int,
        novelty_context: NoveltyCheckContext | None = None,
        implementation_base_parent: OrganismMeta | None = None,
        pipeline_state_callback: Callable[[str], None] | None = None,
    ) -> CreationStageResult:
        """Retry a full organism-creation exchange on provider or parse failure."""

        max_attempts = max(1, int(self.evolver_cfg.creation.max_attempts_to_create_organism))
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            # FIX-mode style retry feedback: on attempt 2+ append the previous
            # attempt's exception to the user prompt so the LLM can see what
            # it got wrong (missing OPTIONAL_CODE_SKETCH, unterminated fenced
            # code block, bullet-bullet patterns, etc.) instead of re-running
            # blind. The fields are spelled out so the LLM doesn't have to
            # guess what the parser complained about. ``last_error`` is None
            # on the first attempt — that branch is the unchanged code path.
            if attempt > 1 and last_error is not None:
                retry_user_prompt = (
                    design_user_prompt
                    + "\n\n=== PREVIOUS ATTEMPT FAILED ===\n"
                    + f"This is creation retry attempt {attempt}/{max_attempts}. "
                    + "The previous attempt's response was rejected by the parser or "
                    + "validation layer with the following error:\n"
                    + f"\n  {last_error}\n\n"
                    + "Common causes the parser specifically flags:\n"
                    + "- one or more required CORE_GENES subsections is missing "
                    + "(e.g. you forgot `### OPTIONAL_CODE_SKETCH`).\n"
                    + "- a fenced code block opened with ``` ``` `` was never closed.\n"
                    + "- a continuation line is indented but no `- ` bullet precedes it.\n"
                    + "- a `- - ` (dash-bullet-bullet) pattern at indent 0 — fenced code blocks "
                    + "must be either bare (` ``` `python ... ` ``` `) or directly inside one bullet "
                    + "(`- ``` `python ... ` ``` `), never inside a sub-bullet.\n"
                    + "- a `### Heading` subsection name not matching the schema exactly.\n\n"
                    + "Re-produce the FULL response from scratch (not a diff) with the error fixed. "
                    + "Do not paste the previous broken response back."
                )
            else:
                retry_user_prompt = design_user_prompt
            try:
                return self.run_creation_stages(
                    design_system_prompt=design_system_prompt,
                    design_user_prompt=retry_user_prompt,
                    org_dir=org_dir,
                    organism_id=organism_id,
                    generation=generation,
                    novelty_context=novelty_context,
                    implementation_base_parent=implementation_base_parent,
                    pipeline_state_callback=pipeline_state_callback,
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

        # In pipeline mode the repair stage may use a different route than
        # the organism's primary llm_route_id (e.g. a cheap fast model for
        # checks/repairs and the heavy model only for creative stages).
        # When the organism's pipeline assignment is already known
        # (``organism.llm_pipeline_id`` is set at creation time) we resolve
        # the repair route through that specific pipeline so a long-lived
        # process whose bandit state drifted between creation and repair
        # still uses the original pipeline. Fresh sampling is the fallback
        # for organisms created before this feature shipped.
        if self.pipelines:
            stored_pipeline_id = getattr(organism, "llm_pipeline_id", "") or ""
            pipeline = self._pipelines_by_id.get(stored_pipeline_id)
            if pipeline is not None:
                route_id = pipeline.route_for("repair")
            else:
                route_id = self.sample_route_id(
                    organism_id=organism.organism_id, stage="repair"
                )
        else:
            existing_route = organism.llm_route_id or ""
            # ``seed_copy`` is a sentinel stamped on file-copied baseline
            # organisms (no LLM was involved at creation). Repair needs a
            # real LLM route, so we sample one in that case.
            if existing_route and existing_route != "seed_copy":
                route_id = existing_route
            else:
                route_id = self.sample_route_id(
                    organism_id=organism.organism_id, stage="repair"
                )
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
        # Fold this repair attempt's token spend into the organism's running
        # per-route accounting before persisting the refreshed meta.
        self.merge_token_usage_into_meta(organism)
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

    def materialize_seed_from_file(
        self,
        island: Island,
        organism_id: str,
        generation: int,
        organism_dir: Path,
        source_path: Path,
        pipeline_state_callback: Callable[[str], None] | None = None,
    ) -> OrganismMeta:
        """Build a seed organism by copying a handwritten baseline program.

        This is the lightweight path used when ``evolver.islands.mode``
        is set to ``from_seed``. Instead of asking the LLM to author a
        per-island organism from scratch (3 heavy LLM calls × N×K seeds
        = hours on a 122B model), we copy a single working
        ``implementation.py`` into every (island, slot) tuple. Each copy
        runs through the existing ``simple_eval`` path and seeds the
        bandits with a real fitness measurement immediately.

        The synthetic ``genetic_code.md`` lists every schema subsection
        with a single ``- baseline seed (file-copy from <source>)``
        bullet — enough to make the parser happy without inventing
        prose that the implementer's Python doesn't actually realise.
        Lineage starts empty: this organism has no LLM-generated
        ancestors, only the on-disk source.

        Diversity is expected to emerge during the evolutionary stage
        (mutation, crossover, lineage_regime_hint), not on the seed.
        """

        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(
                f"evolver.islands.seed_program_path does not exist: {source_path}"
            )
        implementation_code = source_path.read_text(encoding="utf-8")
        if not implementation_code.strip():
            raise ValueError(
                f"evolver.islands.seed_program_path is empty: {source_path}"
            )
        if pipeline_state_callback is not None:
            pipeline_state_callback("creating")

        # Synthetic CORE_GENES: one schema-conformant bullet per
        # required subsection plus ``- None.`` for the trailing
        # optional section. The exact list of subsections is family-
        # specific and comes from the loaded schema.
        if self.expected_core_gene_sections is None:
            raise ValueError(
                "materialize_seed_from_file requires expected_core_gene_sections; "
                "this family's genome schema did not load."
            )
        baseline_text = f"baseline seed (file-copy from {source_path.name})"
        core_genes_lines: list[str] = []
        for index, section_name in enumerate(self.expected_core_gene_sections):
            core_genes_lines.append(f"### {section_name}")
            is_last_optional = (
                section_name == "OPTIONAL_CODE_SKETCH"
                or index == len(self.expected_core_gene_sections) - 1
            )
            if is_last_optional:
                core_genes_lines.append("- None.")
            else:
                core_genes_lines.append(f"- {baseline_text}")
        core_genes_text = "\n".join(core_genes_lines)

        parsed: dict[str, str] = {
            "CORE_GENES": core_genes_text,
            "INTERACTION_NOTES": (
                f"Seeded directly from {source_path.name}. The CORE_GENES bullets above are "
                "placeholders for the design intent; the executed behaviour is fully described "
                "by the baseline Python below. Subsequent mutations will replace these bullets "
                "with real design rationale as the LLM iterates on the program."
            ),
            "COMPUTE_NOTES": (
                "Compute matches the baseline program: deterministic, bounded by the "
                "evaluator-side per-case timeout. No new operations are introduced by "
                "this seed step."
            ),
            "CHANGE_DESCRIPTION": (
                "Initial population seed copied verbatim from the family's baseline "
                "implementation. Diversity emerges through mutation and crossover on "
                "subsequent generations, not on this step."
            ),
        }

        if pipeline_state_callback is not None:
            pipeline_state_callback("design")
            pipeline_state_callback("implementation")

        # ``prompt_hash`` is normally an LLM-prompt digest used for dedup
        # of identical prompts. Here every copy shares the same source
        # baseline; we include the organism_id so dedup logic that
        # rejects exact-prompt collisions doesn't accidentally
        # garbage-collect the rest of the population.
        prompt_hash = sha1_text(f"seed_copy:{source_path}:{organism_id}")

        return build_organism_from_response(
            parsed=parsed,
            implementation_code=implementation_code,
            organism_id=organism_id,
            island_id=island.island_id,
            generation=generation,
            mother_id=None,
            father_id=None,
            operator="seed_copy",
            org_dir=organism_dir,
            llm_route_id="seed_copy",
            llm_provider="filesystem",
            provider_model_id="",
            llm_pipeline_id="",
            prompt_hash=prompt_hash,
            seed=self.seed,
            timestamp=utc_now_iso(),
            parent_lineage=[],
            expected_core_gene_sections=self.expected_core_gene_sections,
        )
