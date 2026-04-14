"""Unit tests for novelty-check parsing."""

from __future__ import annotations

import pytest

from src.organisms.novelty import parse_novelty_judgment


def test_parse_novelty_judgment_accepts_exact_code_phrase() -> None:
    judgment = parse_novelty_judgment("## NOVELTY_VERDICT\nNOVELTY_ACCEPTED\n")

    assert judgment.is_accepted is True
    assert judgment.rejection_reason is None


def test_parse_novelty_judgment_requires_rejection_reason() -> None:
    with pytest.raises(ValueError, match="REJECTION_REASON"):
        parse_novelty_judgment("## NOVELTY_VERDICT\nNOVELTY_REJECTED\n")


def test_parse_novelty_judgment_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="NOVELTY_ACCEPTED or NOVELTY_REJECTED"):
        parse_novelty_judgment(
            "## NOVELTY_VERDICT\nACCEPT\n\n## REJECTION_REASON\nNope.\n"
        )
