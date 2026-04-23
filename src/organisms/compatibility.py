"""Compatibility-validation helpers for sectioned design-stage organisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
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

    _ = expected_section_names
    exact_errors: list[ValueError] = []
    for candidate_text in _compatibility_judgment_candidates(text):
        try:
            parsed = _parse_exact_judgment_sections(candidate_text, _COMPATIBILITY_SECTIONS)
        except ValueError as exact_error:
            exact_errors.append(exact_error)
            continue
        return _build_compatibility_judgment(parsed)

    exact_error = exact_errors[-1] if exact_errors else ValueError("Compatibility judgment is empty.")
    return _parse_compact_compatibility_judgment(text, exact_error=exact_error)


def _build_compatibility_judgment(parsed: dict[str, str]) -> CompatibilityJudgment:
    verdict = require_response_section(parsed, "COMPATIBILITY_VERDICT").strip()
    if verdict not in {_COMPATIBILITY_ACCEPTED, _COMPATIBILITY_REJECTED}:
        raise ValueError(
            "Compatibility check must return COMPATIBILITY_ACCEPTED or COMPATIBILITY_REJECTED "
            "in ## COMPATIBILITY_VERDICT."
        )

    rejection_reason = require_response_section(parsed, "REJECTION_REASON").strip()
    if verdict == _COMPATIBILITY_ACCEPTED:
        if _first_nonempty_line(rejection_reason) != "N/A":
            raise ValueError("Accepted compatibility judgments must use exactly N/A in ## REJECTION_REASON.")
        return CompatibilityJudgment(
            verdict=_COMPATIBILITY_ACCEPTED,
            rejection_reason=None,
            sections_at_issue=(),
        )

    if not rejection_reason or rejection_reason == "N/A":
        raise ValueError("Rejected compatibility judgments require a non-empty REJECTION_REASON.")
    return CompatibilityJudgment(
        verdict=_COMPATIBILITY_REJECTED,
        rejection_reason=rejection_reason,
        sections_at_issue=(),
    )


def _compatibility_judgment_candidates(text: str) -> tuple[str, ...]:
    raw = str(text).strip()
    if not raw:
        return ()

    candidates = [raw]
    marker = "## COMPATIBILITY_VERDICT"
    marker_index = raw.rfind(marker)
    if marker_index > 0:
        candidates.append(raw[marker_index:].strip())
    return tuple(dict.fromkeys(candidates))


def _parse_compact_compatibility_judgment(text: str, *, exact_error: ValueError) -> CompatibilityJudgment:
    """Accept common compact verdict forms without relaxing the verdict contract."""

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        raise exact_error

    first = lines[0]
    if first == "## COMPATIBILITY_VERDICT":
        tail_parts = lines[1:]
    elif first.startswith("## COMPATIBILITY_VERDICT "):
        tail_parts = [first.removeprefix("## COMPATIBILITY_VERDICT ").strip(), *lines[1:]]
    else:
        compact_from_tokens = _parse_tokenized_compact_compatibility_judgment(text)
        if compact_from_tokens is not None:
            return compact_from_tokens
        raise exact_error

    compact_text = "\n".join(tail_parts).strip()
    for verdict in (_COMPATIBILITY_ACCEPTED, _COMPATIBILITY_REJECTED):
        if compact_text == verdict or compact_text.startswith(verdict + "\n") or compact_text.startswith(verdict + " "):
            remainder = compact_text[len(verdict) :].strip()
            remainder = _strip_reason_heading_alias(remainder)
            if verdict == _COMPATIBILITY_ACCEPTED:
                if remainder and _first_nonempty_line(remainder) != "N/A":
                    raise ValueError(
                        "Accepted compatibility judgments must use N/A or no reason after COMPATIBILITY_ACCEPTED."
                    )
                return CompatibilityJudgment(
                    verdict=_COMPATIBILITY_ACCEPTED,
                    rejection_reason=None,
                    sections_at_issue=(),
                )
            if not remainder or _first_nonempty_line(remainder) == "N/A":
                raise ValueError("Rejected compatibility judgments require a non-empty REJECTION_REASON.")
            return CompatibilityJudgment(
                verdict=_COMPATIBILITY_REJECTED,
                rejection_reason=remainder,
                sections_at_issue=(),
            )

    raise exact_error


def _parse_tokenized_compact_compatibility_judgment(text: str) -> CompatibilityJudgment | None:
    tokens = str(text).strip().split()
    verdict_positions = [
        index for index, token in enumerate(tokens) if token in {_COMPATIBILITY_ACCEPTED, _COMPATIBILITY_REJECTED}
    ]
    if not verdict_positions:
        return None

    verdict_index = verdict_positions[-1]
    verdict = tokens[verdict_index]
    tail = tokens[verdict_index + 1 :]

    if verdict == _COMPATIBILITY_ACCEPTED:
        accepted_tail = [token for token in tail if token not in {"REJECTION_REASON", "SECTIONS_AT_ISSUE"}]
        if not accepted_tail or all(token in {"N/A", "NONE"} for token in accepted_tail):
            return CompatibilityJudgment(
                verdict=_COMPATIBILITY_ACCEPTED,
                rejection_reason=None,
                sections_at_issue=(),
            )
        return None

    if tail and tail[0] == "REJECTION_REASON":
        tail = tail[1:]
    if not tail or tail[0] == "N/A":
        raise ValueError("Rejected compatibility judgments require a non-empty REJECTION_REASON.")
    if "SECTIONS_AT_ISSUE" in tail:
        tail = tail[: tail.index("SECTIONS_AT_ISSUE")]
    reason = " ".join(tail).strip()
    if not reason or reason == "N/A":
        raise ValueError("Rejected compatibility judgments require a non-empty REJECTION_REASON.")
    return CompatibilityJudgment(
        verdict=_COMPATIBILITY_REJECTED,
        rejection_reason=reason,
        sections_at_issue=(),
    )


def _strip_reason_heading_alias(text: str) -> str:
    remainder = str(text).strip()
    for heading in (
        "## REJECTION_REASON",
        "## REASON",
        "REJECTION_REASON",
        "REASON",
    ):
        if remainder == heading:
            return ""
        if remainder.startswith(heading + "\n") or remainder.startswith(heading + " "):
            return remainder[len(heading) :].strip()
    return remainder


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def format_compatibility_rejection_feedback(
    rejection_history: list[CompatibilityJudgment],
) -> str:
    """Render accumulated compatibility rejection feedback for retry prompts."""

    if not rejection_history:
        return "No prior compatibility rejection."

    blocks: list[str] = []
    for index, rejection in enumerate(rejection_history, start=1):
        reason = rejection.rejection_reason or "Compatibility rejected the candidate."
        blocks.append(
            "\n".join(
                (
                    f"Compatibility rejection {index}:",
                    f"Reason: {reason.strip()}",
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
    system = compose_system_prompt("", prompts.compatibility_seed_system)
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
    system = compose_system_prompt("", prompts.compatibility_mutation_system)
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
    system = compose_system_prompt("", prompts.compatibility_crossover_system)
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
