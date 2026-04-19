"""Novelty-validation helpers for mutation and crossover design stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import OrganismMeta
from src.organisms.hypothesis_artifacts import read_canonical_genome
from src.organisms.organism import (
    build_genetic_code_from_design_response,
    format_genetic_code,
    require_response_section,
    read_organism_hypothesis_for_prompt,
)

_NOVELTY_ACCEPTED = "NOVELTY_ACCEPTED"
_NOVELTY_REJECTED = "NOVELTY_REJECTED"


class NoveltyRejectionExhaustedError(RuntimeError):
    """Raised when novelty validation rejects every allowed design attempt."""


@dataclass(slots=True)
class NoveltyJudgment:
    """Structured result of a novelty-check LLM judgment."""

    verdict: str
    rejection_reason: str | None = None

    @property
    def is_accepted(self) -> bool:
        return self.verdict == _NOVELTY_ACCEPTED


@dataclass(slots=True)
class NoveltyCheckContext:
    """Operator-specific prompt builders for the novelty-validation loop."""

    operator: str
    build_design_prompts: Callable[[list[str]], tuple[str, str]]
    build_novelty_prompts: Callable[[dict[str, Any]], tuple[str, str]]
    parse_design_response: Callable[[str, str], dict[str, Any]] | None = None
    parse_novelty_response: Callable[[str, dict[str, Any]], NoveltyJudgment] | None = None


def format_novelty_rejection_feedback(reasons: list[str]) -> str:
    """Render accumulated novelty rejection reasons for retry prompts."""

    if not reasons:
        return "No prior novelty rejections."
    return "\n".join(f"- {reason.strip()}" for reason in reasons if reason.strip()) or "No prior novelty rejections."


def parse_novelty_judgment(text: str) -> NoveltyJudgment:
    """Parse novelty-judge output into a strict verdict object."""

    parsed = parse_llm_response(text)
    verdict = require_response_section(parsed, "NOVELTY_VERDICT").strip()
    if verdict not in {_NOVELTY_ACCEPTED, _NOVELTY_REJECTED}:
        raise ValueError(
            "Novelty check must return NOVELTY_ACCEPTED or NOVELTY_REJECTED in ## NOVELTY_VERDICT."
        )
    if verdict == _NOVELTY_ACCEPTED:
        return NoveltyJudgment(verdict=verdict, rejection_reason=None)

    rejection_reason = require_response_section(parsed, "REJECTION_REASON")
    return NoveltyJudgment(verdict=verdict, rejection_reason=rejection_reason)


def build_mutation_novelty_prompt(
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    schema_provider=None,
) -> tuple[str, str]:
    """Build mutation novelty-check prompts for one candidate child design."""

    if schema_provider is not None and callable(getattr(schema_provider, "build_novelty_prompt_context", None)):
        parent_genome = read_canonical_genome(Path(parent.organism_dir), schema_provider)
        system = compose_system_prompt(prompts.project_context, prompts.mutation_novelty_system)
        user = prompts.mutation_novelty_user.format(
            selected_child_slot_draft=schema_provider.format_slot_assignments_for_prompt(parent_genome),
            excluded_modules=schema_provider.format_excluded_modules_for_prompt(removed_genes, parent_genome),
            parent_slot_assignments=schema_provider.format_slot_assignments_for_prompt(parent_genome),
            candidate_child_slot_assignments=_format_candidate_assignments(candidate_design),
            candidate_change_description=str(candidate_design["render_fields"]["change_description"]),
            typed_prompt_context=_build_mutation_novelty_context(schema_provider),
        )
        return system, user

    candidate_genetic_code = build_genetic_code_from_design_response(candidate_design)
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.mutation_novelty_system)
    user = prompts.mutation_novelty_user.format(
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        removed_gene_pool=_render_gene_pool(removed_genes),
        parent_genetic_code=format_genetic_code(read_organism_hypothesis_for_prompt(parent, schema_provider=schema_provider)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
        selected_child_slot_draft=_render_gene_pool(inherited_genes),
        excluded_modules=_render_gene_pool(removed_genes),
        parent_slot_assignments=format_genetic_code(read_organism_hypothesis_for_prompt(parent, schema_provider=schema_provider)),
        candidate_child_slot_assignments=format_genetic_code(candidate_genetic_code),
        typed_prompt_context="",
    )
    return system, user


def build_crossover_novelty_prompt(
    *,
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    schema_provider=None,
) -> tuple[str, str]:
    """Build crossover novelty-check prompts for one candidate child design."""

    if schema_provider is not None and callable(getattr(schema_provider, "build_novelty_prompt_context", None)):
        mother_genome = read_canonical_genome(Path(mother.organism_dir), schema_provider)
        father_genome = read_canonical_genome(Path(father.organism_dir), schema_provider)
        system = compose_system_prompt(prompts.project_context, prompts.crossover_novelty_system)
        user = prompts.crossover_novelty_user.format(
            selected_child_slot_draft=schema_provider.format_slot_assignments_for_prompt(mother_genome),
            primary_parent_slot_assignments=schema_provider.format_slot_assignments_for_prompt(mother_genome),
            secondary_parent_slot_assignments=schema_provider.format_slot_assignments_for_prompt(father_genome),
            candidate_child_slot_assignments=_format_candidate_assignments(candidate_design),
            candidate_change_description=str(candidate_design["render_fields"]["change_description"]),
            typed_prompt_context=_build_crossover_novelty_context(schema_provider),
        )
        return system, user

    candidate_genetic_code = build_genetic_code_from_design_response(candidate_design)
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.crossover_novelty_system)
    user = prompts.crossover_novelty_user.format(
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        mother_genetic_code=format_genetic_code(read_organism_hypothesis_for_prompt(mother, schema_provider=schema_provider)),
        father_genetic_code=format_genetic_code(read_organism_hypothesis_for_prompt(father, schema_provider=schema_provider)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
        selected_child_slot_draft=_render_gene_pool(inherited_genes),
        primary_parent_slot_assignments=format_genetic_code(read_organism_hypothesis_for_prompt(mother, schema_provider=schema_provider)),
        secondary_parent_slot_assignments=format_genetic_code(read_organism_hypothesis_for_prompt(father, schema_provider=schema_provider)),
        candidate_child_slot_assignments=format_genetic_code(candidate_genetic_code),
        typed_prompt_context="",
    )
    return system, user


def _render_gene_pool(genes: list[str]) -> str:
    return "\n".join(f"- {gene}" for gene in genes) or "(none)"


def _format_candidate_assignments(candidate_design: dict[str, Any]) -> str:
    return json.dumps(candidate_design["slot_assignments"], indent=2, ensure_ascii=False)


def _build_mutation_novelty_context(schema_provider: Any) -> str:
    builder = getattr(schema_provider, "build_mutation_novelty_prompt_context", None)
    if not callable(builder):
        return schema_provider.build_novelty_prompt_context()
    return builder()


def _build_crossover_novelty_context(schema_provider: Any) -> str:
    builder = getattr(schema_provider, "build_crossover_novelty_prompt_context", None)
    if not callable(builder):
        return schema_provider.build_novelty_prompt_context()
    return builder()
