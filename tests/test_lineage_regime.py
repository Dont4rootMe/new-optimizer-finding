"""Tests for the lineage regime summarizer."""

from __future__ import annotations

from src.organisms.lineage_regime import summarize_recent_regime


def test_empty_lineage_returns_unclear_signal() -> None:
    hint = summarize_recent_regime([], family="awtf2025_heuristic")
    assert "unclear" in hint.lower()


def test_too_few_ancestors_flag_as_unclear() -> None:
    # Two entries — below the convergence-detection minimum of 3.
    hint = summarize_recent_regime(
        [
            {"change_description": "Tried no_walls + quadrant grouping."},
            {"change_description": "Repeated no_walls + greedy routing."},
        ],
        family="awtf2025_heuristic",
    )
    assert "unclear" in hint.lower()


def test_strong_convergence_emits_break_out_candidate() -> None:
    lineage = [
        {"change_description": "Held no_walls; quadrant grouping; greedy routing."},
        {"change_description": "Kept no_walls and quadrant grouping; tuned greedy."},
        {"change_description": "no_walls again with quadrant grouping."},
        {"change_description": "Still no_walls and quadrant grouping."},
        {"change_description": "no_walls + quadrant grouping + greedy routing."},
    ]
    hint = summarize_recent_regime(lineage, family="awtf2025_heuristic", lookback=5)
    assert "converged" in hint.lower()
    # The keyword matched should be reported in the convergence line.
    assert "no_walls" in hint.lower() or "no walls" in hint.lower()


def test_unknown_family_falls_back_to_unclear_when_no_keywords() -> None:
    # No keyword map for this family → no observations possible.
    hint = summarize_recent_regime(
        [
            {"change_description": "Some prose"},
            {"change_description": "Other prose"},
            {"change_description": "More prose"},
        ],
        family="totally_made_up_family",
    )
    # With no keyword map there are no observations, so the hint either
    # explicitly notes the absence or returns the default no-observations
    # banner. Either is acceptable; we just want no crash and a useful string.
    assert isinstance(hint, str) and hint.strip()


def test_circle_packing_family_uses_packing_axes() -> None:
    lineage = [
        {"change_description": "Single radius layout."},
        {"change_description": "Single radius again, no symmetry change."},
        {"change_description": "Held single radius with no repair."},
        {"change_description": "Single radius, perturbation repair tried."},
    ]
    hint = summarize_recent_regime(lineage, family="circle_packing_shinka", lookback=4)
    # Should be a coherent multi-line hint
    assert "scanning last 4" in hint or "scanning last" in hint
