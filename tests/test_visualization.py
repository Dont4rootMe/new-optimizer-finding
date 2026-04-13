"""Tests for evolution overview helpers."""

from __future__ import annotations

from src.evolve.visualization import (
    OrganismVizRecord,
    _best_active_simple_score,
    _format_score,
)


def _record(*, organism_id: str, simple_score: float | None, active: bool) -> OrganismVizRecord:
    return OrganismVizRecord(
        organism_id=organism_id,
        island_id="island",
        model_label="model",
        generation_created=0,
        current_generation_active=0,
        operator="seed",
        status="evaluated",
        pipeline_state="simple_complete",
        mother_id=None,
        father_id=None,
        simple_score=simple_score,
        hard_score=None,
        created_at=None,
        simple_eval_finished_at=None,
        active=active,
    )


def test_best_active_simple_score_prefers_active_population() -> None:
    records = [
        _record(organism_id="hist_best", simple_score=0.9, active=False),
        _record(organism_id="active_a", simple_score=0.4, active=True),
        _record(organism_id="active_b", simple_score=0.7, active=True),
    ]

    assert _best_active_simple_score(records) == 0.7


def test_best_active_simple_score_falls_back_to_best_tracked_score() -> None:
    records = [
        _record(organism_id="hist_best", simple_score=0.9, active=False),
        _record(organism_id="active_pending", simple_score=None, active=True),
    ]

    assert _best_active_simple_score(records) == 0.9


def test_format_score_renders_compact_header_value() -> None:
    assert _format_score(0.123456) == "0.1235"
    assert _format_score(None) == "n/a"
