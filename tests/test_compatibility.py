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
        "N/A\n\n"
        "## SECTIONS_AT_ISSUE\n"
        "NONE\n"
    )


def _rejected_text(reason: str, sections: str) -> str:
    return (
        "## COMPATIBILITY_VERDICT\n"
        "COMPATIBILITY_REJECTED\n\n"
        "## REJECTION_REASON\n"
        f"{reason}\n\n"
        "## SECTIONS_AT_ISSUE\n"
        f"{sections}\n"
    )


def test_parse_compatibility_judgment_accepts_none_sections() -> None:
    judgment = parse_compatibility_judgment(_accepted_text())

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None
    assert judgment.sections_at_issue == ()


def test_parse_compatibility_judgment_rejected_with_one_section() -> None:
    judgment = parse_compatibility_judgment(
        _rejected_text(
            "The repair policy depends on an absent conflict ranking.",
            "REPAIR_POLICY",
        )
    )

    assert judgment.is_accepted is False
    assert judgment.rejection_reason == "The repair policy depends on an absent conflict ranking."
    assert judgment.sections_at_issue == ("REPAIR_POLICY",)


def test_parse_compatibility_judgment_rejected_with_multiple_sections() -> None:
    judgment = parse_compatibility_judgment(
        _rejected_text(
            "The repair policy depends on an absent conflict model.",
            "CONFLICT_MODEL, REPAIR_POLICY",
        )
    )

    assert judgment.sections_at_issue == ("CONFLICT_MODEL", "REPAIR_POLICY")


def test_parse_compatibility_judgment_rejects_malformed_verdict() -> None:
    with pytest.raises(ValueError, match="COMPATIBILITY_ACCEPTED or COMPATIBILITY_REJECTED"):
        parse_compatibility_judgment(
            (
                "## COMPATIBILITY_VERDICT\n"
                "ACCEPT\n\n"
                "## REJECTION_REASON\n"
                "N/A\n\n"
                "## SECTIONS_AT_ISSUE\n"
                "NONE\n"
            )
        )


def test_parse_compatibility_judgment_rejects_missing_section() -> None:
    with pytest.raises(ValueError, match="exactly these sections"):
        parse_compatibility_judgment("## COMPATIBILITY_VERDICT\nCOMPATIBILITY_ACCEPTED\n")


def test_parse_compatibility_judgment_rejects_malformed_sections_at_issue() -> None:
    with pytest.raises(ValueError, match="SECTIONS_AT_ISSUE"):
        parse_compatibility_judgment(_rejected_text("Bad section list.", "- REPAIR_POLICY"))


def test_parse_compatibility_judgment_rejects_unknown_section_name() -> None:
    with pytest.raises(ValueError, match="unknown section name"):
        parse_compatibility_judgment(
            _rejected_text("Bad section name.", "GEOMETRY"),
            expected_section_names=SECTION_NAMES,
        )


def test_parse_compatibility_judgment_uses_family_local_sections() -> None:
    judgment = parse_compatibility_judgment(
        _rejected_text(
            "The parameter schedule refers to an absent step controller.",
            "STEP_CONTROL_POLICY, PARAMETERS",
        ),
        expected_section_names=OPTIMIZER_SECTION_NAMES,
    )

    assert judgment.sections_at_issue == ("STEP_CONTROL_POLICY", "PARAMETERS")


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
    assert "Sections at issue: CONFLICT_MODEL, REPAIR_POLICY" in text
    assert "Compatibility rejection 2:" in text
    assert "Sections at issue: OPTIONAL_CODE_SKETCH" in text
