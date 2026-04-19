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
from src.organisms.genetic_code_format import DEFAULT_CORE_GENE_SECTION_NAMES, parse_section_issue_list

_NOVELTY_ACCEPTED = "NOVELTY_ACCEPTED"
_NOVELTY_REJECTED = "NOVELTY_REJECTED"


class NoveltyRejectionExhaustedError(RuntimeError):
    """Raised when novelty validation rejects every allowed design attempt."""


@dataclass(slots=True)
class NoveltyJudgment:
    """Structured result of a novelty-check LLM judgment."""

    verdict: str
    rejection_reason: str | None = None
    sections_at_issue: tuple[str, ...] = ()

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


def parse_novelty_judgment(
    text: str,
    *,
    expected_section_names: tuple[str, ...] | None = None,
) -> NoveltyJudgment:
    """Parse novelty-judge output into a strict verdict object."""

    parsed = (
        _parse_exact_judgment_sections(
            text,
            (
                "NOVELTY_VERDICT",
                "REJECTION_REASON",
                "SECTIONS_AT_ISSUE",
            ),
        )
        if expected_section_names is not None
        else parse_llm_response(text)
    )
    verdict = require_response_section(parsed, "NOVELTY_VERDICT").strip()
    if verdict not in {_NOVELTY_ACCEPTED, _NOVELTY_REJECTED}:
        raise ValueError(
            "Novelty check must return NOVELTY_ACCEPTED or NOVELTY_REJECTED in ## NOVELTY_VERDICT."
        )
    if verdict == _NOVELTY_ACCEPTED:
        if expected_section_names is not None:
            sections_text = require_response_section(parsed, "SECTIONS_AT_ISSUE")
            sections_at_issue = parse_section_issue_list(
                sections_text,
                expected_section_names=expected_section_names,
            )
            if sections_at_issue:
                raise ValueError("Accepted novelty judgments must use NONE for SECTIONS_AT_ISSUE.")
        return NoveltyJudgment(verdict=verdict, rejection_reason=None, sections_at_issue=())

    rejection_reason = require_response_section(parsed, "REJECTION_REASON").strip()
    if not rejection_reason:
        raise ValueError("Rejected novelty judgments require a non-empty REJECTION_REASON.")
    if expected_section_names is None and "SECTIONS_AT_ISSUE" not in parsed:
        sections_at_issue = ()
    else:
        sections_at_issue = parse_section_issue_list(
            require_response_section(parsed, "SECTIONS_AT_ISSUE"),
            expected_section_names=expected_section_names or DEFAULT_CORE_GENE_SECTION_NAMES,
        )
    return NoveltyJudgment(
        verdict=verdict,
        rejection_reason=rejection_reason,
        sections_at_issue=sections_at_issue,
    )


def build_mutation_novelty_prompt(
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build mutation novelty-check prompts for one candidate child design."""

    candidate_genetic_code = build_genetic_code_from_design_response(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.mutation_novelty_system)
    user = prompts.mutation_novelty_user.format(
        genome_schema=prompts.genome_schema,
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
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build crossover novelty-check prompts for one candidate child design."""

    candidate_genetic_code = build_genetic_code_from_design_response(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    system = compose_system_prompt(prompts.project_context, prompts.crossover_novelty_system)
    user = prompts.crossover_novelty_user.format(
        genome_schema=prompts.genome_schema,
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        mother_genetic_code=format_genetic_code(read_organism_genetic_code(mother)),
        father_genetic_code=format_genetic_code(read_organism_genetic_code(father)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def _render_gene_pool(genes: list[str]) -> str:
    return "\n".join(f"- {gene}" for gene in genes) or "(none)"


def _parse_exact_judgment_sections(text: str, expected: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    observed: list[str] = []

    def flush_current() -> None:
        if current_key is not None:
            parsed[current_key] = "\n".join(current_lines).strip()

    for line_number, line in enumerate(str(text).splitlines(), start=1):
        if line.startswith("## "):
            name = line[3:].strip()
            if not name or " " in name:
                raise ValueError(f"Malformed novelty judgment section heading at line {line_number}: {line!r}")
            flush_current()
            current_key = name
            current_lines = []
            observed.append(name)
            continue
        if line.startswith("##"):
            raise ValueError(f"Malformed novelty judgment section heading at line {line_number}: {line!r}")
        if current_key is None:
            if line.strip():
                raise ValueError("Novelty judgment contains text before the first section.")
            continue
        current_lines.append(line)

    flush_current()
    observed_tuple = tuple(observed)
    if observed_tuple != expected:
        raise ValueError(
            "Novelty judgment must contain exactly these sections in order: "
            + ", ".join(f"## {name}" for name in expected)
        )
    return parsed
