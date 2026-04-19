"""Compatibility-validation helpers for sectioned design-stage organisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.organisms.genetic_code_format import DEFAULT_CORE_GENE_SECTION_NAMES, parse_section_issue_list
from src.organisms.organism import (
    build_genetic_code_from_design_response,
    format_genetic_code,
    require_response_section,
    read_organism_genetic_code,
)
from src.evolve.types import OrganismMeta

_COMPATIBILITY_ACCEPTED = "COMPATIBILITY_ACCEPTED"
_COMPATIBILITY_REJECTED = "COMPATIBILITY_REJECTED"
_COMPATIBILITY_SECTIONS = (
    "COMPATIBILITY_VERDICT",
    "REJECTION_REASON",
    "SECTIONS_AT_ISSUE",
)


class CompatibilityRejectionExhaustedError(RuntimeError):
    """Raised when compatibility validation rejects every allowed design attempt."""


@dataclass(frozen=True)
class CompatibilityJudgment:
    verdict: Literal["COMPATIBILITY_ACCEPTED", "COMPATIBILITY_REJECTED"]
    rejection_reason: str | None
    sections_at_issue: tuple[str, ...]

    @property
    def is_accepted(self) -> bool:
        return self.verdict == _COMPATIBILITY_ACCEPTED


@dataclass(frozen=True)
class CompatibilityCheckContext:
    operator_kind: Literal["seed", "mutation", "crossover"]


@dataclass(frozen=True)
class CompatibilityValidationContext:
    check: CompatibilityCheckContext
    build_design_prompts: Callable[[list[str], list[CompatibilityJudgment]], tuple[str, str]]
    build_compatibility_prompts: Callable[[dict[str, str]], tuple[str, str]]


def parse_compatibility_judgment(
    text: str,
    *,
    expected_section_names: tuple[str, ...] | None = None,
) -> CompatibilityJudgment:
    """Parse compatibility-judge output into a strict verdict object."""

    parsed = _parse_exact_judgment_sections(text, _COMPATIBILITY_SECTIONS)
    verdict = require_response_section(parsed, "COMPATIBILITY_VERDICT").strip()
    if verdict not in {_COMPATIBILITY_ACCEPTED, _COMPATIBILITY_REJECTED}:
        raise ValueError(
            "Compatibility check must return COMPATIBILITY_ACCEPTED or COMPATIBILITY_REJECTED "
            "in ## COMPATIBILITY_VERDICT."
        )

    sections_at_issue = parse_section_issue_list(
        require_response_section(parsed, "SECTIONS_AT_ISSUE"),
        expected_section_names=expected_section_names or DEFAULT_CORE_GENE_SECTION_NAMES,
    )
    if verdict == _COMPATIBILITY_ACCEPTED:
        if sections_at_issue:
            raise ValueError("Accepted compatibility judgments must use NONE for SECTIONS_AT_ISSUE.")
        return CompatibilityJudgment(
            verdict=_COMPATIBILITY_ACCEPTED,
            rejection_reason=None,
            sections_at_issue=(),
        )

    rejection_reason = require_response_section(parsed, "REJECTION_REASON").strip()
    if not rejection_reason:
        raise ValueError("Rejected compatibility judgments require a non-empty REJECTION_REASON.")
    return CompatibilityJudgment(
        verdict=_COMPATIBILITY_REJECTED,
        rejection_reason=rejection_reason,
        sections_at_issue=sections_at_issue,
    )


def format_compatibility_rejection_feedback(
    rejection_history: list[CompatibilityJudgment],
) -> str:
    """Render accumulated compatibility rejection feedback for retry prompts."""

    if not rejection_history:
        return "No prior compatibility rejection."

    blocks: list[str] = []
    for index, rejection in enumerate(rejection_history, start=1):
        reason = rejection.rejection_reason or "Compatibility rejected the candidate."
        sections = ", ".join(rejection.sections_at_issue) if rejection.sections_at_issue else "NONE"
        blocks.append(
            "\n".join(
                (
                    f"Compatibility rejection {index}:",
                    f"Reason: {reason.strip()}",
                    f"Sections at issue: {sections}",
                )
            )
        )
    return "\n\n".join(blocks)


def build_seed_compatibility_prompt(
    *,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build seed compatibility-check prompts for one candidate design."""

    candidate_genetic_code, candidate_change_description = _candidate_artifacts(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    system = compose_system_prompt(prompts.project_context, prompts.compatibility_seed_system)
    user = prompts.compatibility_seed_user.format(
        genome_schema=prompts.genome_schema,
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def build_mutation_compatibility_prompt(
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build mutation compatibility-check prompts for one candidate child design."""

    candidate_genetic_code, candidate_change_description = _candidate_artifacts(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    system = compose_system_prompt(prompts.project_context, prompts.compatibility_mutation_system)
    user = prompts.compatibility_mutation_user.format(
        genome_schema=prompts.genome_schema,
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        removed_gene_pool=_render_gene_pool(removed_genes),
        parent_genetic_code=format_genetic_code(read_organism_genetic_code(parent)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def build_crossover_compatibility_prompt(
    *,
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    candidate_design: dict[str, str],
    prompts: PromptBundle,
    expected_core_gene_sections: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Build crossover compatibility-check prompts for one candidate child design."""

    candidate_genetic_code, candidate_change_description = _candidate_artifacts(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    system = compose_system_prompt(prompts.project_context, prompts.compatibility_crossover_system)
    user = prompts.compatibility_crossover_user.format(
        genome_schema=prompts.genome_schema,
        inherited_gene_pool=_render_gene_pool(inherited_genes),
        mother_genetic_code=format_genetic_code(read_organism_genetic_code(mother)),
        father_genetic_code=format_genetic_code(read_organism_genetic_code(father)),
        candidate_genetic_code=format_genetic_code(candidate_genetic_code),
        candidate_change_description=candidate_change_description,
    )
    return system, user


def _candidate_artifacts(
    candidate_design: dict[str, str],
    *,
    expected_core_gene_sections: tuple[str, ...] | None,
) -> tuple[dict[str, object], str]:
    candidate_genetic_code = build_genetic_code_from_design_response(
        candidate_design,
        expected_core_gene_sections=expected_core_gene_sections,
    )
    candidate_change_description = require_response_section(candidate_design, "CHANGE_DESCRIPTION")
    return candidate_genetic_code, candidate_change_description


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
                raise ValueError(f"Malformed compatibility judgment section heading at line {line_number}: {line!r}")
            flush_current()
            current_key = name
            current_lines = []
            observed.append(name)
            continue
        if line.startswith("##"):
            raise ValueError(f"Malformed compatibility judgment section heading at line {line_number}: {line!r}")
        if current_key is None:
            if line.strip():
                raise ValueError("Compatibility judgment contains text before the first section.")
            continue
        current_lines.append(line)

    flush_current()
    observed_tuple = tuple(observed)
    if observed_tuple != expected:
        raise ValueError(
            "Compatibility judgment must contain exactly these sections in order: "
            + ", ".join(f"## {name}" for name in expected)
        )
    return parsed
