"""Tests for evolution overview helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt

from src.evolve.visualization import (
    OrganismVizRecord,
    _best_active_simple_score,
    _format_score,
    _maternal_lineage,
    _offspring_operator_counts_by_generation,
    _plot_best_vs_evaluations,
    _plot_best_vs_runtime,
    _plot_operator_mix_by_generation,
)


def _record(
    *,
    organism_id: str,
    simple_score: float | None,
    active: bool,
    island_id: str = "island",
    generation_created: int = 0,
    operator: str = "seed",
    mother_id: str | None = None,
    father_id: str | None = None,
    simple_eval_finished_at: datetime | None = None,
) -> OrganismVizRecord:
    return OrganismVizRecord(
        organism_id=organism_id,
        island_id=island_id,
        model_label="model",
        generation_created=generation_created,
        current_generation_active=0,
        operator=operator,
        status="evaluated",
        pipeline_state="simple_complete",
        mother_id=mother_id,
        father_id=father_id,
        simple_score=simple_score,
        hard_score=None,
        created_at=None,
        simple_eval_finished_at=simple_eval_finished_at,
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


def test_maternal_lineage_follows_best_active_record() -> None:
    records = [
        _record(organism_id="seed", simple_score=0.2, active=False),
        _record(
            organism_id="mut_a",
            simple_score=0.5,
            active=False,
            generation_created=1,
            operator="mutation",
            mother_id="seed",
        ),
        _record(
            organism_id="cross_a",
            simple_score=0.8,
            active=True,
            generation_created=2,
            operator="crossover",
            mother_id="mut_a",
            father_id="other_parent",
        ),
        _record(
            organism_id="other_parent",
            simple_score=0.7,
            active=False,
            generation_created=1,
        ),
        _record(organism_id="historic_best", simple_score=0.95, active=False, generation_created=3),
    ]

    assert [record.organism_id for record in _maternal_lineage(records)] == ["seed", "mut_a", "cross_a"]


def test_offspring_operator_counts_split_within_inter_and_mutation() -> None:
    records = [
        _record(organism_id="seed_a", simple_score=0.1, active=False, island_id="island_a"),
        _record(organism_id="seed_b", simple_score=0.2, active=False, island_id="island_b"),
        _record(
            organism_id="mut_a",
            simple_score=0.3,
            active=False,
            island_id="island_a",
            generation_created=1,
            operator="mutation",
            mother_id="seed_a",
        ),
        _record(
            organism_id="cross_within",
            simple_score=0.4,
            active=False,
            island_id="island_a",
            generation_created=1,
            operator="crossover",
            mother_id="seed_a",
            father_id="mut_a",
        ),
        _record(
            organism_id="cross_inter",
            simple_score=0.5,
            active=True,
            island_id="island_a",
            generation_created=2,
            operator="crossover",
            mother_id="mut_a",
            father_id="seed_b",
        ),
    ]

    counts = _offspring_operator_counts_by_generation(records)

    assert counts["mutation"] == {1: 1}
    assert counts["within_island_crossover"] == {1: 1}
    assert counts["inter_island_crossover"] == {2: 1}


def test_best_vs_evaluations_plots_current_best_maternal_line() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        _record(organism_id="seed", simple_score=0.2, active=False, simple_eval_finished_at=t0 + timedelta(seconds=5)),
        _record(
            organism_id="mut_a",
            simple_score=0.5,
            active=False,
            generation_created=1,
            operator="mutation",
            mother_id="seed",
            simple_eval_finished_at=t0 + timedelta(seconds=15),
        ),
        _record(
            organism_id="cross_a",
            simple_score=0.8,
            active=True,
            generation_created=2,
            operator="crossover",
            mother_id="mut_a",
            father_id="other_parent",
            simple_eval_finished_at=t0 + timedelta(seconds=25),
        ),
        _record(
            organism_id="other_parent",
            simple_score=0.7,
            active=False,
            generation_created=1,
            simple_eval_finished_at=t0 + timedelta(seconds=10),
        ),
    ]

    fig, ax = plt.subplots()
    try:
        _plot_best_vs_evaluations(ax, records)
        labels = ax.get_legend_handles_labels()[1]
        assert "Current Best Maternal Line" in labels
        assert len(ax.lines) == 2
        lineage_line = ax.lines[1]
        assert list(lineage_line.get_xdata()) == [1, 3, 4]
        assert list(lineage_line.get_ydata()) == [0.2, 0.5, 0.8]
    finally:
        plt.close(fig)


def test_best_vs_runtime_plots_current_best_maternal_line() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        _record(organism_id="seed", simple_score=0.2, active=False, simple_eval_finished_at=t0 + timedelta(seconds=5)),
        _record(
            organism_id="mut_a",
            simple_score=0.5,
            active=False,
            generation_created=1,
            operator="mutation",
            mother_id="seed",
            simple_eval_finished_at=t0 + timedelta(seconds=15),
        ),
        _record(
            organism_id="cross_a",
            simple_score=0.8,
            active=True,
            generation_created=2,
            operator="crossover",
            mother_id="mut_a",
            father_id="other_parent",
            simple_eval_finished_at=t0 + timedelta(seconds=25),
        ),
        _record(
            organism_id="other_parent",
            simple_score=0.7,
            active=False,
            generation_created=1,
            simple_eval_finished_at=t0 + timedelta(seconds=10),
        ),
    ]

    fig, ax = plt.subplots()
    try:
        _plot_best_vs_runtime(ax, records)
        labels = ax.get_legend_handles_labels()[1]
        assert "Current Best Maternal Line" in labels
        assert len(ax.lines) == 2
        lineage_line = ax.lines[1]
        assert list(lineage_line.get_xdata()) == [0.0, 10.0, 20.0]
        assert list(lineage_line.get_ydata()) == [0.2, 0.5, 0.8]
    finally:
        plt.close(fig)


def test_operator_mix_plot_no_longer_draws_maternal_lineage_overlay() -> None:
    records = [
        _record(organism_id="seed_a", simple_score=0.1, active=False, island_id="island_a"),
        _record(organism_id="seed_b", simple_score=0.2, active=False, island_id="island_b"),
        _record(
            organism_id="mut_a",
            simple_score=0.3,
            active=False,
            island_id="island_a",
            generation_created=1,
            operator="mutation",
            mother_id="seed_a",
        ),
        _record(
            organism_id="cross_best",
            simple_score=0.5,
            active=True,
            island_id="island_a",
            generation_created=2,
            operator="crossover",
            mother_id="mut_a",
            father_id="seed_b",
        ),
    ]

    fig, ax = plt.subplots()
    try:
        _plot_operator_mix_by_generation(ax, records)
        labels = ax.get_legend_handles_labels()[1]
        assert "Current Best Maternal Line" not in labels
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)
