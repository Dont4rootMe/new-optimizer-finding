"""Novelty-validation helpers for mutation and crossover design stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.template_parser import parse_llm_response
from src.evolve.types import OrganismMeta
from src.organisms.organism import (
    build_genetic_code_from_design_response,
    format_genetic_code,
    require_response_section,
    read_organism_genetic_code,
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
    build_novelty_prompts: Callable[[dict[str, str]], tuple[str, str]]


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
) -> tuple[str, str]:
    """Build mutation novelty-check prompts for one candidate child design."""

    candidate_genetic_code = build_genetic_code_from_design_response(candidate_design)
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.mutation_novelty_system)
    user = prompts.mutation_novelty_user.format(
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        removed_gene_pool=_render_gene_pool(removed_genes),
        parent_genetic_code=format_genetic_code(read_organism_genetic_code(parent)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def build_crossover_novelty_prompt(
    *,
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
) -> tuple[str, str]:
    """Build crossover novelty-check prompts for one candidate child design."""

    candidate_genetic_code = build_genetic_code_from_design_response(candidate_design)
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.crossover_novelty_system)
    user = prompts.crossover_novelty_user.format(
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        mother_genetic_code=format_genetic_code(read_organism_genetic_code(mother)),
        father_genetic_code=format_genetic_code(read_organism_genetic_code(father)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def _render_gene_pool(genes: list[str]) -> str:
    return "\n".join(f"- {gene}" for gene in genes) or "(none)"
