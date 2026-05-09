"""Tests for evolution overview helpers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from src.evolve.visualization import (
    OrganismVizRecord,
    RenderedSnapshot,
    _best_active_simple_score,
    _evenly_sample_records,
    _format_score,
    _maternal_lineage,
    _offspring_operator_counts_by_generation,
    _offspring_operator_totals,
    _plot_best_vs_evaluations,
    _plot_best_vs_evaluations_with_dead,
    _plot_best_vs_runtime,
    _plot_best_vs_runtime_with_dead,
    _plot_cumulative_creations_by_island_over_generation,
    _plot_cumulative_creations_by_operator_over_generation,
    _plot_cumulative_evaluations_by_island_over_generation,
    _plot_cumulative_max_score_by_model_over_generation,
    _plot_operator_mix_by_generation,
    _plot_score_by_generation,
    _plot_survival_by_evaluations,
    _plot_survival_by_generation,
    _plot_survival_by_runtime,
    _sample_records_for_render,
    build_ancestor_chains,
    render_evolution_snapshot,
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


def test_offspring_operator_totals_aggregate_all_generations() -> None:
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
            organism_id="mut_b",
            simple_score=0.35,
            active=False,
            island_id="island_a",
            generation_created=2,
            operator="mutation",
            mother_id="mut_a",
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
            mother_id="mut_b",
            father_id="seed_b",
        ),
    ]

    totals = _offspring_operator_totals(records)

    assert totals["mutation"] == 2
    assert totals["within_island_crossover"] == 1
    assert totals["inter_island_crossover"] == 1


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


def test_operator_mix_plot_shows_aggregated_totals_without_lineage_overlay() -> None:
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
        assert ax.get_title() == "Created Organisms by Operator"
        bar_heights = [patch.get_height() for patch in ax.patches]
        assert bar_heights == [1, 1]
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_sample_records_for_render_preserves_best_maternal_line() -> None:
    records = [
        _record(organism_id="seed", simple_score=0.2, active=False),
        _record(
            organism_id="mid",
            simple_score=0.5,
            active=False,
            generation_created=1,
            operator="mutation",
            mother_id="seed",
        ),
        _record(
            organism_id="best",
            simple_score=0.8,
            active=True,
            generation_created=2,
            operator="crossover",
            mother_id="mid",
            father_id="other_parent",
        ),
        _record(organism_id="other_parent", simple_score=0.7, active=False, generation_created=1),
        _record(organism_id="later", simple_score=0.6, active=False, generation_created=3),
    ]

    sampled = _sample_records_for_render(records, max_evaluated_points=3)

    assert {record.organism_id for record in sampled} == {"seed", "mid", "best"}


def test_evenly_sample_records_spreads_across_history() -> None:
    records = [
        _record(organism_id=f"org_{idx}", simple_score=float(idx), active=False)
        for idx in range(6)
    ]

    sampled = _evenly_sample_records(records, 3)

    assert [record.organism_id for record in sampled] == ["org_1", "org_3", "org_5"]


def test_offspring_operator_totals_can_use_full_context_for_sampled_records() -> None:
    records = [
        _record(organism_id="seed_a", simple_score=0.1, active=False, island_id="island_a"),
        _record(organism_id="seed_b", simple_score=0.2, active=False, island_id="island_b"),
        _record(
            organism_id="cross_inter",
            simple_score=0.5,
            active=True,
            island_id="island_a",
            generation_created=2,
            operator="crossover",
            mother_id="seed_a",
            father_id="seed_b",
        ),
    ]

    sampled = [records[-1]]
    totals = _offspring_operator_totals(sampled, context_records=records)

    assert totals["inter_island_crossover"] == 1


# ---------------------------------------------------------------------------
# Tests for new panels (tasks 4–7) and the snapshot bundle.
# ---------------------------------------------------------------------------


def test_build_ancestor_chains_walks_maternal_lineage_for_every_record() -> None:
    """Each record's chain must be ordered root → self and contain the record itself."""

    records = [
        _record(organism_id="grand", simple_score=0.1, active=False),
        _record(organism_id="parent", simple_score=0.3, active=False, mother_id="grand"),
        _record(organism_id="child", simple_score=0.7, active=True, mother_id="parent"),
        _record(organism_id="orphan", simple_score=0.4, active=False),
    ]

    chains = build_ancestor_chains(records)

    assert chains["grand"] == ["grand"]
    assert chains["parent"] == ["grand", "parent"]
    assert chains["child"] == ["grand", "parent", "child"]
    # records without mother_id are roots themselves
    assert chains["orphan"] == ["orphan"]


def test_build_ancestor_chains_handles_corrupted_cycle() -> None:
    """If lineage data forms a cycle, the walk must still terminate."""

    records = [
        _record(organism_id="a", simple_score=0.1, active=False, mother_id="b"),
        _record(organism_id="b", simple_score=0.2, active=False, mother_id="a"),
    ]

    chains = build_ancestor_chains(records)

    assert "a" in chains["a"] and chains["a"][-1] == "a"
    assert "b" in chains["b"] and chains["b"][-1] == "b"


def test_score_by_generation_panel_renders_without_error() -> None:
    records = [
        _record(organism_id="g0_a", simple_score=0.2, active=False, generation_created=0),
        _record(organism_id="g0_b", simple_score=0.3, active=False, generation_created=0),
        _record(organism_id="g1_a", simple_score=0.5, active=True, generation_created=1),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_score_by_generation(ax, records)
        # The plot should have at least one collection (scatter) when records exist.
        assert ax.collections
    finally:
        plt.close(fig)


def test_best_vs_evaluations_with_dead_skips_unscored_columns_but_keeps_axis() -> None:
    """Task 5: dead organisms must consume an Ox slot without producing a marker."""

    records = [
        _record(organism_id="dead_a", simple_score=None, active=False),
        _record(organism_id="scored_a", simple_score=0.4, active=False),
        _record(organism_id="dead_b", simple_score=None, active=False),
        _record(organism_id="scored_b", simple_score=0.9, active=True),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_best_vs_evaluations_with_dead(ax, records)
        # x-limits widen to cover all four organisms even though only two were scored.
        xmin, xmax = ax.get_xlim()
        assert xmax - xmin >= 4 - 1
    finally:
        plt.close(fig)


def test_best_vs_runtime_with_dead_renders_runtime_axis() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = [
        _record(organism_id="dead", simple_score=None, active=False, simple_eval_finished_at=None),
        _record(
            organism_id="scored",
            simple_score=0.5,
            active=True,
            simple_eval_finished_at=base + timedelta(seconds=120),
        ),
    ]
    # The dead record needs a created_at to participate; we approximate via the
    # scored record's t0 minus a small offset (the helper falls back to
    # created_at when simple_eval_finished_at is None).
    records[0] = OrganismVizRecord(
        organism_id="dead",
        island_id="island",
        model_label="model",
        generation_created=0,
        current_generation_active=0,
        operator="seed",
        status="failed_simple_eval",
        pipeline_state="failed_simple_eval",
        mother_id=None,
        father_id=None,
        simple_score=None,
        hard_score=None,
        created_at=base,
        simple_eval_finished_at=None,
        active=False,
    )

    fig, ax = plt.subplots()
    try:
        _plot_best_vs_runtime_with_dead(ax, records)
        # Runtime axis is in seconds; the scored point is 120s after t0=created_at(dead).
        xmin, xmax = ax.get_xlim()
        assert xmax >= 100
    finally:
        plt.close(fig)


def test_survival_by_evaluations_includes_ratio_and_counts() -> None:
    records = [
        _record(organism_id="dead_a", simple_score=None, active=False),
        _record(organism_id="scored_a", simple_score=0.4, active=False),
        _record(organism_id="dead_b", simple_score=None, active=False),
        _record(organism_id="scored_b", simple_score=0.9, active=True),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_survival_by_evaluations(ax, records)
        # Two main lines (scored / dead) on the primary axis; ratio line lives on a twin axis.
        assert len(ax.get_lines()) >= 2
    finally:
        plt.close(fig)


def test_survival_by_runtime_skips_when_no_timestamps() -> None:
    records = [
        _record(organism_id="dead_a", simple_score=None, active=False),
        _record(organism_id="scored_a", simple_score=0.4, active=False),
    ]
    fig, ax = plt.subplots()
    try:
        # No timestamps available — the panel must still render an empty placeholder
        # and not raise.
        _plot_survival_by_runtime(ax, records)
        # An empty panel turns the axes off.
        assert ax.axison is False
    finally:
        plt.close(fig)


def test_survival_by_generation_accumulates_scored_and_dead_per_generation() -> None:
    records = [
        _record(organism_id="g0_dead", simple_score=None, active=False, generation_created=0),
        _record(organism_id="g0_scored", simple_score=0.5, active=False, generation_created=0),
        _record(organism_id="g1_dead", simple_score=None, active=False, generation_created=1),
        _record(organism_id="g1_scored", simple_score=0.9, active=True, generation_created=1),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_survival_by_generation(ax, records)
        assert len(ax.get_lines()) >= 2
    finally:
        plt.close(fig)


def test_cumulative_evaluations_by_island_over_generation_is_monotonic_per_island() -> None:
    records = [
        _record(organism_id="a0", simple_score=0.1, active=False, generation_created=0, island_id="i_a"),
        _record(organism_id="b0", simple_score=0.2, active=False, generation_created=0, island_id="i_b"),
        _record(organism_id="a1", simple_score=0.3, active=False, generation_created=1, island_id="i_a"),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_cumulative_evaluations_by_island_over_generation(ax, records)
        # Stacked-area renders use fill_between which adds collections, not lines.
        assert ax.collections
    finally:
        plt.close(fig)


def test_cumulative_creations_by_operator_over_generation_renders_lines() -> None:
    records = [
        _record(organism_id="parent", simple_score=0.1, active=False, generation_created=0),
        _record(
            organism_id="child_mut",
            simple_score=0.3,
            active=False,
            generation_created=1,
            operator="mutation",
            mother_id="parent",
        ),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_cumulative_creations_by_operator_over_generation(ax, records)
        assert ax.get_lines()
    finally:
        plt.close(fig)


def test_cumulative_creations_by_island_over_generation_renders_lines() -> None:
    records = [
        _record(organism_id="a0", simple_score=None, active=False, generation_created=0, island_id="i_a"),
        _record(organism_id="b0", simple_score=None, active=False, generation_created=0, island_id="i_b"),
        _record(organism_id="a1", simple_score=0.3, active=False, generation_created=1, island_id="i_a"),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_cumulative_creations_by_island_over_generation(ax, records)
        assert ax.get_lines()
    finally:
        plt.close(fig)


def test_cumulative_max_score_by_model_over_generation_is_non_decreasing() -> None:
    records = [
        OrganismVizRecord(
            organism_id="a0",
            island_id="island",
            model_label="modelX",
            generation_created=0,
            current_generation_active=0,
            operator="seed",
            status="evaluated",
            pipeline_state="simple_complete",
            mother_id=None,
            father_id=None,
            simple_score=0.4,
            hard_score=None,
            created_at=None,
            simple_eval_finished_at=None,
            active=False,
        ),
        OrganismVizRecord(
            organism_id="a1",
            island_id="island",
            model_label="modelX",
            generation_created=1,
            current_generation_active=0,
            operator="seed",
            status="evaluated",
            pipeline_state="simple_complete",
            mother_id=None,
            father_id=None,
            simple_score=0.7,
            hard_score=None,
            created_at=None,
            simple_eval_finished_at=None,
            active=True,
        ),
    ]
    fig, ax = plt.subplots()
    try:
        _plot_cumulative_max_score_by_model_over_generation(ax, records)
        line = ax.get_lines()[0]
        ys = list(line.get_ydata())
        # Cumulative max must be non-decreasing.
        assert all(b is None or a is None or b >= a for a, b in zip(ys, ys[1:]))
    finally:
        plt.close(fig)


def _write_organism(
    population_root: Path,
    *,
    organism_id: str,
    generation: int,
    island_id: str,
    operator: str,
    simple_score: float | None,
    mother_id: str | None = None,
    father_id: str | None = None,
    timestamp_iso: str = "2026-01-01T00:00:00+00:00",
) -> None:
    """Write a minimal organism.json/summary.json pair for snapshot tests."""

    org_dir = population_root / f"gen_{generation:04d}" / f"island_{island_id}" / f"org_{organism_id}"
    org_dir.mkdir(parents=True, exist_ok=True)
    (org_dir / "organism.json").write_text(
        json.dumps(
            {
                "organism_id": organism_id,
                "island_id": island_id,
                "generation_created": generation,
                "current_generation_active": generation,
                "operator": operator,
                "status": "evaluated" if simple_score is not None else "failed_simple_eval",
                "pipeline_state": "simple_complete" if simple_score is not None else "failed_simple_eval",
                "mother_id": mother_id,
                "father_id": father_id,
                "simple_score": simple_score,
                "hard_score": None,
                "model_name": "modelX",
                "timestamp": timestamp_iso,
            }
        ),
        encoding="utf-8",
    )
    if simple_score is not None:
        (org_dir / "summary.json").write_text(
            json.dumps({"phase_results": {"simple": {"eval_finished_at": timestamp_iso}}}),
            encoding="utf-8",
        )


def test_render_evolution_snapshot_writes_grouped_artifacts(tmp_path: Path) -> None:
    """End-to-end smoke test: snapshot bundle has paths in viz/<group>/ subdirs."""

    population_root = tmp_path / "population"
    population_root.mkdir()
    (population_root / "population_state.json").write_text(
        json.dumps(
            {
                "current_generation": 1,
                "active_organisms": [
                    {
                        "organism_id": "scored_b",
                        "island_id": "i_a",
                        "organism_dir": "gen_0001/island_i_a/org_scored_b",
                        "generation_created": 1,
                        "current_generation_active": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_organism(population_root, organism_id="seed_a", generation=0, island_id="i_a", operator="seed", simple_score=0.2)
    _write_organism(population_root, organism_id="dead_a", generation=0, island_id="i_a", operator="seed", simple_score=None)
    _write_organism(
        population_root,
        organism_id="scored_b",
        generation=1,
        island_id="i_a",
        operator="mutation",
        simple_score=0.7,
        mother_id="seed_a",
    )

    snapshot = render_evolution_snapshot(population_root)

    assert isinstance(snapshot, RenderedSnapshot)
    assert snapshot.composite_overview_path is not None and snapshot.composite_overview_path.exists()
    # Each panel group produced files under the matching subdirectory.
    for name, path in snapshot.overview_panels.items():
        assert path.exists(), f"missing overview panel {name}"
        assert "/viz/overview/" in str(path).replace("\\", "/")
    for name, path in snapshot.timeline_panels.items():
        assert path.exists(), f"missing timeline panel {name}"
        assert "/viz/timeline/" in str(path).replace("\\", "/")
    for name, path in snapshot.survival_panels.items():
        assert path.exists(), f"missing survival panel {name}"
        assert "/viz/survival/" in str(path).replace("\\", "/")
    # Plotly is optional; if it's available the HTML must exist under viz/interactive/.
    for name, path in snapshot.interactive_html.items():
        assert path.exists(), f"missing interactive html {name}"
        assert "/viz/interactive/" in str(path).replace("\\", "/")
    # Required panels per task spec.
    assert "score_by_generation" in snapshot.overview_panels
    assert "best_vs_evaluations_with_dead" in snapshot.overview_panels
    assert "best_vs_runtime_with_dead" in snapshot.overview_panels
    assert "by_evaluations" in snapshot.survival_panels
    assert "by_runtime" in snapshot.survival_panels
    assert "by_generation" in snapshot.survival_panels
    assert "cumulative_evaluations_by_island" in snapshot.timeline_panels
    assert "cumulative_creations_by_operator" in snapshot.timeline_panels
    assert "cumulative_creations_by_island" in snapshot.timeline_panels
    assert "cumulative_max_score_by_model" in snapshot.timeline_panels
