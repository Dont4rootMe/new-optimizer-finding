"""Smoke tests for the Plotly Best vs Evaluations interactive graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evolve.visualization import OrganismVizRecord
from src.evolve.visualization_plotly import render_best_vs_evaluations_plotly


def _record(
    *,
    organism_id: str,
    simple_score: float | None,
    mother_id: str | None = None,
    generation_created: int = 0,
) -> OrganismVizRecord:
    return OrganismVizRecord(
        organism_id=organism_id,
        island_id="island",
        model_label="model",
        generation_created=generation_created,
        current_generation_active=0,
        operator="seed",
        status="evaluated",
        pipeline_state="simple_complete",
        mother_id=mother_id,
        father_id=None,
        simple_score=simple_score,
        hard_score=None,
        created_at=None,
        simple_eval_finished_at=None,
        active=False,
    )


def _plotly_available() -> bool:
    try:
        import plotly.graph_objects as _go  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _plotly_available(), reason="plotly not installed")
def test_render_best_vs_evaluations_plotly_writes_html_with_hover_callback(tmp_path: Path) -> None:
    """The HTML must contain the hover post-script and a customdata block."""

    records = [
        _record(organism_id="grand", simple_score=0.1),
        _record(organism_id="parent", simple_score=0.4, mother_id="grand", generation_created=1),
        _record(organism_id="child", simple_score=0.9, mother_id="parent", generation_created=2),
    ]

    out_path = tmp_path / "best_vs_evaluations.html"
    result = render_best_vs_evaluations_plotly(records, out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    # Hover callback injected by post_script.
    assert "plotly_hover" in html
    assert "plotly_unhover" in html
    # The interactive trace owns customdata so the hover handler can read ancestor xs/ys.
    assert "customdata" in html
    # The ancestor overlay trace exists by name (label visible in HTML payload).
    assert "Hover Ancestors" in html


@pytest.mark.skipif(not _plotly_available(), reason="plotly not installed")
def test_render_best_vs_evaluations_plotly_returns_none_when_no_evaluated(tmp_path: Path) -> None:
    records = [_record(organism_id="dead", simple_score=None)]
    out_path = tmp_path / "no_eval.html"

    result = render_best_vs_evaluations_plotly(records, out_path=out_path)

    assert result is None
    assert not out_path.exists()
