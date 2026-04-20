"""Unit tests for compatibility-check parsing and feedback."""

from __future__ import annotations

import pytest

from src.organisms.compatibility import (
    CompatibilityJudgment,
    format_compatibility_rejection_feedback,
    parse_compatibility_judgment,
)

SECTION_NAMES = (
    "INIT_GEOMETRY",
    "RADIUS_POLICY",
    "EXPANSION_POLICY",
    "CONFLICT_MODEL",
    "REPAIR_POLICY",
    "CONTROL_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)
OPTIMIZER_SECTION_NAMES = (
    "STATE_REPRESENTATION",
    "GRADIENT_PROCESSING",
    "UPDATE_RULE",
    "PARAMETER_GROUP_POLICY",
    "STEP_CONTROL_POLICY",
    "STABILITY_POLICY",
    "PARAMETERS",
    "OPTIONAL_CODE_SKETCH",
)


def _accepted_text() -> str:
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_ACCEPTED\n\n"
        "## REJECTION_REASON\n"
        "N/A\n"
    )


def _rejected_text(reason: str) -> str:
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_REJECTED\n\n"
        "## REJECTION_REASON\n"
        f"{reason}\n"
    )


def test_parse_compatibility_judgment_accepts_none_sections() -> None:
    judgment = parse_compatibility_judgment(_accepted_text())

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_rejected_with_reason() -> None:
    judgment = parse_compatibility_judgment(
        _rejected_text("The repair policy depends on an absent conflict ranking.")
    )

    assert judgment.is_accepted is False
    assert judgment.rejection_reason == "The repair policy depends on an absent conflict ranking."
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_rejects_malformed_verdict() -> None:
    with pytest.raises(ValueError, match="COMPATIBILITY_ACCEPTED or COMPATIBILITY_REJECTED"):
        parse_compatibility_judgment(
            (
                "## COMPATIBILITY_VERDICT\n"
                "ACCEPT\n\n"
                "## REJECTION_REASON\n"
                "N/A\n"
            )
        )


def test_parse_compatibility_judgment_rejects_missing_section() -> None:
    with pytest.raises(ValueError, match="before the first section"):
        parse_compatibility_judgment("COMPATIBILITY_ACCEPTED\nN/A\n")


def test_parse_compatibility_judgment_accepts_compacted_accepted_with_required_heading() -> None:
    judgment = parse_compatibility_judgment("## COMPATIBILITY_VERDICT COMPATIBILITY_ACCEPTED N/A\n")

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_accepts_compacted_accepted_without_reason() -> None:
    judgment = parse_compatibility_judgment("## COMPATIBILITY_VERDICT\nCOMPATIBILITY_ACCEPTED\n")

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_accepts_reason_alias_for_accepted() -> None:
    judgment = parse_compatibility_judgment(
        "## COMPATIBILITY_VERDICT\nCOMPATIBILITY_ACCEPTED\n\n## REASON\nN/A\n"
    )

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_accepts_same_line_reason_alias_for_accepted() -> None:
    judgment = parse_compatibility_judgment("## COMPATIBILITY_VERDICT COMPATIBILITY_ACCEPTED ## REASON N/A\n")

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_accepts_accepted_with_extra_commentary_after_na() -> None:
    judgment = parse_compatibility_judgment(
        (
            "## COMPATIBILITY_VERDICT\n"
            "COMPATIBILITY_ACCEPTED\n\n"
            "## REJECTION_REASON\n"
            "N/A\n"
            "The design appears internally coherent.\n"
        )
    )

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_rejects_accepted_reason_other_than_na() -> None:
    with pytest.raises(ValueError, match="exactly N/A"):
        parse_compatibility_judgment(
            (
                "## COMPATIBILITY_VERDICT\n"
                "COMPATIBILITY_ACCEPTED\n\n"
                "## REJECTION_REASON\n"
                "Looks good.\n"
            )
        )


def test_parse_compatibility_judgment_rejects_rejected_na_reason() -> None:
    with pytest.raises(ValueError, match="non-empty REJECTION_REASON"):
        parse_compatibility_judgment(
            (
                "## COMPATIBILITY_VERDICT\n"
                "COMPATIBILITY_REJECTED\n\n"
                "## REJECTION_REASON\n"
                "N/A\n"
            )
        )


def test_parse_compatibility_judgment_does_not_require_sections_at_issue() -> None:
    judgment = parse_compatibility_judgment(
        _rejected_text("The parameter schedule refers to an absent step controller."),
        expected_section_names=OPTIMIZER_SECTION_NAMES,
    )

    assert judgment.sections_at_issue == ()


def test_format_compatibility_rejection_feedback_empty_history() -> None:
    assert format_compatibility_rejection_feedback([]) == "No prior compatibility rejection."


def test_format_compatibility_rejection_feedback_includes_reason_and_sections() -> None:
    text = format_compatibility_rejection_feedback(
        [
            CompatibilityJudgment(
                verdict="COMPATIBILITY_REJECTED",
                rejection_reason="Repair depends on a missing conflict ranking.",
                sections_at_issue=("CONFLICT_MODEL", "REPAIR_POLICY"),
            ),
            CompatibilityJudgment(
                verdict="COMPATIBILITY_REJECTED",
                rejection_reason="The optional sketch is too global.",
                sections_at_issue=("OPTIONAL_CODE_SKETCH",),
            ),
        ]
    )

    assert "Compatibility rejection 1:" in text
    assert "Reason: Repair depends on a missing conflict ranking." in text
    assert "Compatibility rejection 2:" in text
    assert "Sections at issue" not in text
