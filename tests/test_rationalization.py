"""Tests for the Step-1 rationalization artifact parser."""

from __future__ import annotations

from src.organisms.rationalization import (
    REQUIRED_SECTIONS,
    format_rationalization_for_step2,
    parse_rationalization_response,
    rationalization_summary,
)


_CANONICAL_TEXT = """## SCORE_BEARING_CORE
- Parent's score is driven by quadrant grouping plus greedy routing.

## LINEAGE_REGIME_DIAGNOSIS
- All 5 recent ancestors use no walls and the same greedy routing family.

## WEAKNESS_HYPOTHESIS
- The no-walls regime is leaking pairs through the central bottleneck.

## WHAT_TO_REMOVE
- Drop the greedy routing fallback; it duplicates the staged release.

## WHAT_TO_ADD_OR_INVENT
- Add a central spine wall family with a 2x2 hub opening.
- ~~~python
  def add_spine(n):
      mid = n // 2
      for r in range(n):
          if abs(r - mid) > 1:
              vertical[r] = vertical[r][:mid] + '1' + vertical[r][mid:]
  ~~~

## CHILD_DIRECTION
- Break out of the no-walls regime with a central-spine wall + hub.
- Expected regime shift: wall family no-walls → central-spine.
- Constraint compliance: spine adds O(N) walls; group commands stay ≤ 2N.
"""


def test_parse_canonical_text_returns_all_required_sections() -> None:
    output = parse_rationalization_response(_CANONICAL_TEXT)
    assert set(output.sections) >= set(REQUIRED_SECTIONS)
    assert "central spine" in output.sections["WHAT_TO_ADD_OR_INVENT"]
    assert "wall family no-walls → central-spine" in output.sections["CHILD_DIRECTION"]
    assert output.has_actionable_directive is True


def test_parse_empty_text_returns_missing_markers() -> None:
    output = parse_rationalization_response("")
    for name in REQUIRED_SECTIONS:
        assert output.sections[name] == "(not produced)"
    assert output.has_actionable_directive is False


def test_parse_partial_text_fills_missing_with_marker() -> None:
    partial = "## SCORE_BEARING_CORE\nSome diagnosis.\n\n## CHILD_DIRECTION\nGo for spine walls."
    output = parse_rationalization_response(partial)

    assert output.sections["SCORE_BEARING_CORE"] == "Some diagnosis."
    assert output.sections["CHILD_DIRECTION"] == "Go for spine walls."
    assert output.sections["WHAT_TO_REMOVE"] == "(not produced)"
    # CHILD_DIRECTION non-empty → directive is actionable
    assert output.has_actionable_directive is True


def test_parse_unstructured_prose_returns_missing_markers() -> None:
    """A response without any `## ` headers is unparseable: every required
    section degrades to the not-produced marker (callers may still feed the
    raw text to Step 2)."""

    output = parse_rationalization_response("just some text without headings")
    assert all(value == "(not produced)" for value in output.sections.values())
    assert output.has_actionable_directive is False


def test_format_for_step2_normalizes_order_and_fills_missing() -> None:
    partial = "## CHILD_DIRECTION\nGo for spine walls.\n\n## SCORE_BEARING_CORE\nDiagnosed."
    output = parse_rationalization_response(partial)
    formatted = format_rationalization_for_step2(output)
    # Canonical ordering: SCORE_BEARING_CORE first, CHILD_DIRECTION last.
    score_idx = formatted.index("## SCORE_BEARING_CORE")
    direction_idx = formatted.index("## CHILD_DIRECTION")
    assert score_idx < direction_idx
    # Missing sections appear with the marker body.
    assert "(not produced)" in formatted


def test_rationalization_summary_lists_sections_present() -> None:
    output = parse_rationalization_response(_CANONICAL_TEXT)
    summary = rationalization_summary(output)
    assert summary["has_actionable_directive"] is True
    assert set(summary["sections_present"]) == set(REQUIRED_SECTIONS)
