"""Unit tests for novelty-check parsing."""

from __future__ import annotations

import pytest

from src.organisms.genetic_code_format import DEFAULT_CORE_GENE_SECTION_NAMES
from src.organisms.novelty import parse_novelty_judgment


def test_parse_novelty_judgment_accepts_exact_code_phrase() -> None:
    judgment = parse_novelty_judgment("## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n")

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_section_aware_novelty_judgment_accepts_none_sections() -> None:
    judgment = parse_novelty_judgment(
        (
            "## NOVELTY_VERDICT\n"
            "NOVELTY_ACCEPTED\n\n"
            "## REJECTION_REASON\n"
            "N/A\n\n"
            "## SECTIONS_AT_ISSUE\n"
            "NONE\n"
        ),
        expected_section_names=DEFAULT_CORE_GENE_SECTION_NAMES,
    )

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_section_aware_novelty_judgment_rejected_with_one_section() -> None:
    judgment = parse_novelty_judgment(
        (
            "## NOVELTY_VERDICT\n"
            "NOVELTY_REJECTED\n\n"
            "## REJECTION_REASON\n"
            "The repair change is unsupported.\n\n"
            "## SECTIONS_AT_ISSUE\n"
            "REPAIR_POLICY\n"
        ),
        expected_section_names=DEFAULT_CORE_GENE_SECTION_NAMES,
    )

    assert judgment.is_accepted is False
    assert judgment.rejection_reason == "The repair change is unsupported."
    assert judgment.sections_at_issue == ("REPAIR_POLICY",)


def test_parse_section_aware_novelty_judgment_rejected_with_multiple_sections() -> None:
    judgment = parse_novelty_judgment(
        (
            "## NOVELTY_VERDICT\n"
            "NOVELTY_REJECTED\n\n"
            "## REJECTION_REASON\n"
            "The claimed repair novelty depends on an absent conflict model.\n\n"
            "## SECTIONS_AT_ISSUE\n"
            "CONFLICT_MODEL, REPAIR_POLICY\n"
        ),
        expected_section_names=DEFAULT_CORE_GENE_SECTION_NAMES,
    )

    assert judgment.sections_at_issue == ("CONFLICT_MODEL", "REPAIR_POLICY")


def test_parse_novelty_judgment_requires_rejection_reason() -> None:
    with pytest.raises(ValueError, match="REJECTION_REASON"):
        parse_novelty_judgment("## NOVELTY_VERDICT\nNOVELTY_REJECTED\n")


def test_parse_novelty_judgment_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="NOVELTY_ACCEPTED or NOVELTY_REJECTED"):
        parse_novelty_judgment(
            "## NOVELTY_VERDICT\nACCEPT\n\n## REJECTION_REASON\nNope.\n"
        )


def test_parse_section_aware_novelty_judgment_rejects_malformed_sections_at_issue() -> None:
    with pytest.raises(ValueError, match="SECTIONS_AT_ISSUE"):
        parse_novelty_judgment(
            (
                "## NOVELTY_VERDICT\n"
                "NOVELTY_REJECTED\n\n"
                "## REJECTION_REASON\n"
                "Bad section list.\n\n"
                "## SECTIONS_AT_ISSUE\n"
                "- REPAIR_POLICY\n"
            ),
            expected_section_names=DEFAULT_CORE_GENE_SECTION_NAMES,
        )


def test_parse_section_aware_novelty_judgment_rejects_unknown_section_name() -> None:
    with pytest.raises(ValueError, match="unknown section name"):
        parse_novelty_judgment(
            (
                "## NOVELTY_VERDICT\n"
                "NOVELTY_REJECTED\n\n"
                "## REJECTION_REASON\n"
                "Bad section name.\n\n"
                "## SECTIONS_AT_ISSUE\n"
                "GEOMETRY\n"
            ),
            expected_section_names=DEFAULT_CORE_GENE_SECTION_NAMES,
        )
