"""Novelty-validation helpers for mutation and crossover design stages."""

from __future__ import annotations

import re
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
from src.organisms.genetic_code_format import parse_section_issue_list

_NOVELTY_ACCEPTED = "NOVELTY_ACCEPTED"
_NOVELTY_REJECTED = "NOVELTY_REJECTED"
_NOVELTY_JUDGMENT_SECTIONS = (
    "NOVELTY_VERDICT",
    "REJECTION_REASON",
    "SECTIONS_AT_ISSUE",
)


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
        _parse_novelty_judgment_sections(text, _NOVELTY_JUDGMENT_SECTIONS)
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
            sections_text = parsed.get("SECTIONS_AT_ISSUE", "NONE")
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
    if "SECTIONS_AT_ISSUE" not in parsed:
        sections_at_issue = ()
    else:
        sections_at_issue = parse_section_issue_list(
            require_response_section(parsed, "SECTIONS_AT_ISSUE"),
            expected_section_names=expected_section_names,
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


def _parse_novelty_judgment_sections(text: str, expected: tuple[str, ...]) -> dict[str, str]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Novelty judgment is empty.")

    parsed = _parse_structured_novelty_judgment(raw, expected)
    if parsed is not None:
        return parsed

    parsed = _parse_compact_novelty_judgment(raw)
    if parsed is not None:
        return parsed

    raise ValueError(
        "Novelty judgment must contain the required verdict fields or a recognized compact verdict form."
    )


def _parse_structured_novelty_judgment(
    text: str,
    expected: tuple[str, ...],
) -> dict[str, str] | None:
    section_pattern = "|".join(re.escape(name) for name in expected)
    heading_re = re.compile(rf"(?:^|[\t\r\n ])(?:##\s*)?({section_pattern})\b", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return None

    valid_starts = [
        index
        for index in range(len(matches) - len(expected) + 1)
        if tuple(match.group(1) for match in matches[index : index + len(expected)]) == expected
    ]
    if not valid_starts:
        if all(match.group(1) != expected[0] for match in matches):
            return None
        raise ValueError(
            "Novelty judgment must contain exactly these sections in order: "
            + ", ".join(f"## {name}" for name in expected)
        )

    start = valid_starts[-1]
    selected = matches[start : start + len(expected)]
    prefix = text[: selected[0].start()].strip()
    if prefix and _parse_compact_novelty_judgment(prefix) is None:
        raise ValueError("Novelty judgment contains text before the first section.")

    parsed: dict[str, str] = {}
    for index, match in enumerate(selected):
        name = match.group(1)
        body_start = match.end()
        body_end = selected[index + 1].start() if index + 1 < len(selected) else len(text)
        parsed[name] = text[body_start:body_end].strip()
    return parsed


def _parse_compact_novelty_judgment(text: str) -> dict[str, str] | None:
    tokens = str(text).strip().split()
    if not tokens:
        return None

    verdict_positions = [
        index for index, token in enumerate(tokens) if token in {_NOVELTY_ACCEPTED, _NOVELTY_REJECTED}
    ]
    if not verdict_positions:
        return None

    verdict_index = verdict_positions[-1]
    verdict = tokens[verdict_index]
    tail = tokens[verdict_index + 1 :]
    parsed = {"NOVELTY_VERDICT": verdict}

    if verdict == _NOVELTY_ACCEPTED:
        if not tail:
            parsed["REJECTION_REASON"] = "N/A"
            parsed["SECTIONS_AT_ISSUE"] = "NONE"
            return parsed
        if tail == ["N/A"] or tail == ["NONE"] or tail == ["N/A", "NONE"]:
            parsed["REJECTION_REASON"] = "N/A"
            parsed["SECTIONS_AT_ISSUE"] = "NONE"
            return parsed
        return None

    if tail and tail[0] == "REJECTION_REASON":
        tail = tail[1:]
    sections_at_issue = "NONE"
    if "SECTIONS_AT_ISSUE" in tail:
        split_index = tail.index("SECTIONS_AT_ISSUE")
        section_tokens = tail[split_index + 1 :]
        tail = tail[:split_index]
        if section_tokens:
            sections_at_issue = " ".join(section_tokens).strip()
    rejection_reason = " ".join(tail).strip()
    if not rejection_reason:
        return None
    parsed["REJECTION_REASON"] = rejection_reason
    parsed["SECTIONS_AT_ISSUE"] = sections_at_issue
    return parsed
