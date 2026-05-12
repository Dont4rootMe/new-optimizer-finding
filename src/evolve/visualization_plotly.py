"""Interactive Plotly graphs for the evolution dashboard.

Each panel rendered as a matplotlib PNG in ``visualization.py`` also has a
sibling interactive Plotly HTML so the user can hover for organism
metadata, zoom into a region, and toggle traces on/off. Both forms ship
to Comet — PNGs give instant thumbnails, the HTML gives the interactive
deep-dive.

The headline graph ``best_vs_evaluations`` is the most elaborate one: it
ships an extra JS post-script that listens to ``plotly_hover`` /
``plotly_unhover`` events, reads the hovered point's full ancestor chain
from ``customdata``, and toggles a dedicated overlay trace so the user
sees the lineage of *any* point on hover (not just the current best). The
post-script is inlined into the saved HTML so the file works offline.

The remaining panels carry rich per-point ``hovertemplate`` content
(organism id, island, model, generation, operator, score) but no JS
extras.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.evolve.visualization import (
    OrganismVizRecord,
    _evaluated_records,
    _maternal_lineage_points_by_evaluations,
    build_ancestor_chains,
)

LOGGER = logging.getLogger(__name__)

# Operator route ids — duplicated from visualization.py to avoid pulling in
# private helpers across modules.
_WITHIN_ISLAND_CROSSOVER = "within_island_crossover"
_INTER_ISLAND_CROSSOVER = "inter_island_crossover"
_MUTATION = "mutation"
_OPERATOR_PLOT_ORDER = (_WITHIN_ISLAND_CROSSOVER, _INTER_ISLAND_CROSSOVER, _MUTATION)
_OPERATOR_PLOT_LABELS = {
    _WITHIN_ISLAND_CROSSOVER: "Crossbreeding",
    _INTER_ISLAND_CROSSOVER: "Crossbreeding Across Islands",
    _MUTATION: "Mutation",
}
_OPERATOR_PLOT_COLORS = {
    _WITHIN_ISLAND_CROSSOVER: "#ff7f0e",
    _INTER_ISLAND_CROSSOVER: "#9467bd",
    _MUTATION: "#2ca02c",
}


def _try_import_plotly():
    """Return the plotly.graph_objects module, or ``None`` if missing.

    Plotly is an optional dependency under the ``evolve`` extra. Returning
    ``None`` lets each caller decide whether to soft-skip the HTML render
    (preferred) or surface a louder warning.
    """

    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        LOGGER.warning("plotly not installed; skipping interactive HTML (%s)", exc)
        return None
    return go


def _write_plotly(figure: Any, out_path: Path, *, post_script: str | None = None) -> Path:
    """Write a plotly Figure to HTML with CDN-loaded plotly.js."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        post_script=post_script,
    )
    return out_path


def _organism_hover_payload(record: OrganismVizRecord) -> list[Any]:
    """customdata payload shared by every panel that scatters organisms.

    Keep the column order in sync with ``_ORGANISM_HOVER_TEMPLATE``.
    """

    return [
        record.organism_id,
        record.island_id,
        record.model_label,
        record.generation_created,
        record.operator,
    ]


_ORGANISM_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b><br>"
    "island: %{customdata[1]}<br>"
    "model: %{customdata[2]}<br>"
    "generation: %{customdata[3]}<br>"
    "operator: %{customdata[4]}<br>"
    "score: %{y:.4f}"
    "<extra></extra>"
)


def render_best_vs_evaluations_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
) -> Path | None:
    """Render the interactive Best Score vs Evaluations graph.

    Adds the hover-ancestor-trace JS extra. Returns the written file path,
    or ``None`` if Plotly is unavailable or no evaluated organisms exist yet.
    """

    go = _try_import_plotly()
    if go is None:
        return None

    evaluated = _evaluated_records(records)
    if not evaluated:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_index_by_id = {record.organism_id: idx + 1 for idx, record in enumerate(evaluated)}
    ancestor_chains = build_ancestor_chains(records)

    # Per-point ancestor positions on the evaluation axis. Only ancestors that
    # were themselves evaluated have a defined x-coordinate; we filter the
    # rest out so the highlight trace only points at scored ancestors.
    customdata: list[list[Any]] = []
    point_ids: list[str] = []
    for idx, record in enumerate(evaluated):
        chain = ancestor_chains.get(record.organism_id, [record.organism_id])
        ancestor_xs = [
            eval_index_by_id[ancestor_id]
            for ancestor_id in chain
            if ancestor_id in eval_index_by_id
        ]
        ancestor_ys = [
            float(next(r.simple_score for r in evaluated if r.organism_id == ancestor_id))
            for ancestor_id in chain
            if ancestor_id in eval_index_by_id
        ]
        customdata.append(
            [
                record.organism_id,
                record.island_id,
                record.model_label,
                record.generation_created,
                record.operator,
                ancestor_xs,
                ancestor_ys,
            ]
        )
        point_ids.append(record.organism_id)

    fig = go.Figure()

    # Cumulative best line (matches the matplotlib panel's red curve)
    best_running = -float("inf")
    best_xs: list[int] = []
    best_ys: list[float] = []
    for idx, record in enumerate(evaluated, start=1):
        if record.simple_score is None:
            continue
        best_running = max(best_running, float(record.simple_score))
        best_xs.append(idx)
        best_ys.append(best_running)

    fig.add_trace(
        go.Scatter(
            x=best_xs,
            y=best_ys,
            mode="lines",
            name="Best Score",
            line={"color": "#d62728", "width": 2.0},
            hoverinfo="skip",
        )
    )

    # Current best maternal line (matches the dashed blue overlay)
    lineage_points = _maternal_lineage_points_by_evaluations(records, evaluated)
    if lineage_points:
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in lineage_points],
                y=[point[1] for point in lineage_points],
                mode="lines+markers",
                name="Current Best Maternal Line",
                line={"color": "#1f77b4", "width": 2.0, "dash": "dash"},
                marker={"size": 7, "color": "#1f77b4"},
                hoverinfo="skip",
            )
        )

    # Ancestor highlight overlay — kept invisible by default, JS toggles on hover.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="Hover Ancestors",
            line={"color": "#ff7f0e", "width": 3.0},
            marker={"size": 11, "color": "#ff7f0e", "line": {"color": "#000000", "width": 1.0}},
            showlegend=True,
            hoverinfo="skip",
            visible=True,
        )
    )

    # Per-point scatter that owns the hover events.
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(evaluated) + 1)),
            y=[float(record.simple_score) for record in evaluated],
            mode="markers",
            name="Individual Evals",
            marker={"size": 8, "color": "black", "opacity": 0.8},
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "island: %{customdata[1]}<br>"
                "model: %{customdata[2]}<br>"
                "generation: %{customdata[3]}<br>"
                "operator: %{customdata[4]}<br>"
                "score: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Best Score vs Evaluations (interactive)",
        xaxis_title="# Evaluated Organisms",
        yaxis_title="Simple Score",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": -0.15},
    )

    # Trace order is fixed by add_trace order:
    #   0 = best score line
    #   1 = best maternal line (optional)
    #   2 = ancestor overlay (the one we mutate on hover)
    #   3 = individual evals (the one that emits hover events)
    ancestor_trace_index = 2 if lineage_points else 1
    point_trace_index = ancestor_trace_index + 1

    post_script = _build_hover_post_script(
        ancestor_trace_index=ancestor_trace_index,
        point_trace_index=point_trace_index,
        ancestor_chains=ancestor_chains,
        eval_index_by_id=eval_index_by_id,
        point_ids=point_ids,
    )

    fig.write_html(
        str(out_path),
        include_plotlyjs="cdn",
        full_html=True,
        post_script=post_script,
    )
    return out_path


def _build_hover_post_script(
    *,
    ancestor_trace_index: int,
    point_trace_index: int,
    ancestor_chains: dict[str, list[str]],
    eval_index_by_id: dict[str, int],
    point_ids: list[str],
) -> str:
    """Generate the JS that highlights ancestors on hover.

    The script binds to ``plotly_hover``/``plotly_unhover`` events. On hover
    it reads ``customdata`` from the hovered point — which already carries
    the ancestor x/y arrays computed at render time — and pushes them into
    the dedicated overlay trace via ``Plotly.restyle``.
    """

    payload = {
        "ancestor_trace_index": ancestor_trace_index,
        "point_trace_index": point_trace_index,
    }
    return (
        "const __ancestorViz = "
        + json.dumps(payload)
        + ";\n"
        + """
const __plotlyDiv = document.querySelectorAll('div.plotly-graph-div')[
    document.querySelectorAll('div.plotly-graph-div').length - 1
];
if (__plotlyDiv) {
    __plotlyDiv.on('plotly_hover', function (event) {
        if (!event || !event.points || !event.points.length) { return; }
        const point = event.points[0];
        if (typeof point.curveNumber !== 'number' ||
            point.curveNumber !== __ancestorViz.point_trace_index) {
            return;
        }
        const cd = point.customdata;
        if (!cd) { return; }
        const ancestorXs = cd[5] || [];
        const ancestorYs = cd[6] || [];
        Plotly.restyle(__plotlyDiv, {
            x: [ancestorXs],
            y: [ancestorYs],
        }, [__ancestorViz.ancestor_trace_index]);
    });
    __plotlyDiv.on('plotly_unhover', function () {
        Plotly.restyle(__plotlyDiv, {
            x: [[]],
            y: [[]],
        }, [__ancestorViz.ancestor_trace_index]);
    });
}
"""
    )


# ---------------------------------------------------------------------------
# Best Score vs Runtime — same shape as best_vs_evaluations but with the
# x-axis showing elapsed seconds since the first eval finished.
# ---------------------------------------------------------------------------


def render_best_vs_runtime_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    evaluated = [r for r in _evaluated_records(records) if r.simple_eval_finished_at is not None]
    if not evaluated:
        return None
    evaluated.sort(key=lambda r: (r.simple_eval_finished_at, r.organism_id))
    t0 = evaluated[0].simple_eval_finished_at

    xs = [(r.simple_eval_finished_at - t0).total_seconds() for r in evaluated]
    ys = [float(r.simple_score) for r in evaluated]

    best_xs: list[float] = []
    best_ys: list[float] = []
    running = -float("inf")
    for x, y in zip(xs, ys, strict=True):
        running = max(running, y)
        best_xs.append(x)
        best_ys.append(running)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=best_xs,
            y=best_ys,
            mode="lines",
            name="Best Score",
            line={"color": "#d62728", "width": 2.0},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Individual Evals",
            marker={"size": 8, "color": "black", "opacity": 0.8},
            customdata=[_organism_hover_payload(r) for r in evaluated],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "island: %{customdata[1]}<br>"
                "model: %{customdata[2]}<br>"
                "generation: %{customdata[3]}<br>"
                "operator: %{customdata[4]}<br>"
                "elapsed: %{x:.0f}s<br>"
                "score: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Best Score vs Runtime (interactive)",
        xaxis_title="Elapsed Runtime (s)",
        yaxis_title="Simple Score",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": -0.15},
    )
    return _write_plotly(fig, out_path)


# ---------------------------------------------------------------------------
# Score-by-X column scatters (island / model / generation). Each point gets
# the rich organism tooltip so the user can click through anomalies fast.
# ---------------------------------------------------------------------------


def _render_score_by_category_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
    category_label: str,
    category_of: Any,
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    evaluated = _evaluated_records(records)
    if not evaluated:
        return None
    categories = sorted({category_of(r) for r in evaluated})
    category_index = {name: idx for idx, name in enumerate(categories)}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[category_index[category_of(r)] for r in evaluated],
            y=[float(r.simple_score) for r in evaluated],
            mode="markers",
            name="Evaluated",
            marker={
                "size": 9,
                "color": "black",
                "opacity": 0.65,
            },
            customdata=[_organism_hover_payload(r) for r in evaluated],
            hovertemplate=_ORGANISM_HOVER_TEMPLATE,
        )
    )
    active_records = [r for r in evaluated if r.active]
    if active_records:
        fig.add_trace(
            go.Scatter(
                x=[category_index[category_of(r)] for r in active_records],
                y=[float(r.simple_score) for r in active_records],
                mode="markers",
                name="Active",
                marker={
                    "size": 14,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "#2ca02c", "width": 2.0},
                },
                customdata=[_organism_hover_payload(r) for r in active_records],
                hovertemplate=_ORGANISM_HOVER_TEMPLATE,
            )
        )
    fig.update_layout(
        title=f"Score by {category_label} (interactive)",
        xaxis={
            "tickmode": "array",
            "tickvals": list(category_index.values()),
            "ticktext": categories,
            "title": category_label,
        },
        yaxis_title="Simple Score",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "y": -0.2},
    )
    return _write_plotly(fig, out_path)


def render_score_by_island_plotly(records: list[OrganismVizRecord], *, out_path: Path) -> Path | None:
    return _render_score_by_category_plotly(
        records, out_path=out_path, category_label="Island", category_of=lambda r: r.island_id
    )


def render_score_by_model_plotly(records: list[OrganismVizRecord], *, out_path: Path) -> Path | None:
    return _render_score_by_category_plotly(
        records, out_path=out_path, category_label="LLM Model", category_of=lambda r: r.model_label
    )


def render_score_by_generation_plotly(records: list[OrganismVizRecord], *, out_path: Path) -> Path | None:
    return _render_score_by_category_plotly(
        records,
        out_path=out_path,
        category_label="Generation",
        category_of=lambda r: r.generation_created,
    )


# ---------------------------------------------------------------------------
# Cumulative timeline panels — line / stacked-area over generation index.
# ---------------------------------------------------------------------------


def _operator_category(record: OrganismVizRecord, records_by_id: dict[str, OrganismVizRecord]) -> str | None:
    if record.operator == _MUTATION:
        return _MUTATION
    if record.operator != "crossover":
        return None
    if record.father_id:
        father = records_by_id.get(record.father_id)
        if father is not None and father.island_id != record.island_id:
            return _INTER_ISLAND_CROSSOVER
    return _WITHIN_ISLAND_CROSSOVER


def render_cumulative_evaluations_by_island_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    evaluated = _evaluated_records(records)
    if not evaluated:
        return None
    islands = sorted({r.island_id for r in evaluated})
    generations = sorted({r.generation_created for r in evaluated})
    per_gen_per_island: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in evaluated:
        per_gen_per_island[r.generation_created][r.island_id] += 1

    cumulative: dict[str, list[int]] = {island: [] for island in islands}
    running: dict[str, int] = {island: 0 for island in islands}
    for gen in generations:
        for island in islands:
            running[island] += per_gen_per_island[gen].get(island, 0)
            cumulative[island].append(running[island])

    fig = go.Figure()
    for island in islands:
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=cumulative[island],
                mode="lines",
                name=island,
                stackgroup="islands",
                hovertemplate=f"<b>{island}</b><br>gen: %{{x}}<br>cumulative: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cumulative Evaluations by Island (interactive)",
        xaxis_title="Generation Created",
        yaxis_title="# Evaluated Organisms (cumulative)",
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plotly(fig, out_path)


def render_cumulative_creations_by_operator_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
    context_records: list[OrganismVizRecord] | None = None,
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    lookup = {r.organism_id: r for r in (context_records or records)}

    per_op_per_gen: dict[str, dict[int, int]] = {op: defaultdict(int) for op in _OPERATOR_PLOT_ORDER}
    for r in records:
        op = _operator_category(r, lookup)
        if op is None:
            continue
        per_op_per_gen[op][r.generation_created] += 1
    all_generations = sorted(
        {gen for buckets in per_op_per_gen.values() for gen in buckets.keys()}
    )
    if not all_generations:
        return None
    fig = go.Figure()
    for op in _OPERATOR_PLOT_ORDER:
        if not per_op_per_gen[op]:
            continue
        running = 0
        ys: list[int] = []
        for gen in all_generations:
            running += per_op_per_gen[op].get(gen, 0)
            ys.append(running)
        fig.add_trace(
            go.Scatter(
                x=all_generations,
                y=ys,
                mode="lines",
                name=_OPERATOR_PLOT_LABELS[op],
                line={"color": _OPERATOR_PLOT_COLORS[op], "width": 2.0},
                hovertemplate=(
                    f"<b>{_OPERATOR_PLOT_LABELS[op]}</b><br>"
                    "gen: %{x}<br>cumulative: %{y}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Cumulative Creations by Operator (interactive)",
        xaxis_title="Generation Created",
        yaxis_title="# Created Organisms (cumulative)",
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plotly(fig, out_path)


def render_cumulative_creations_by_island_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    if not records:
        return None
    islands = sorted({r.island_id for r in records})
    generations = sorted({r.generation_created for r in records})
    per_gen_per_island: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        per_gen_per_island[r.generation_created][r.island_id] += 1
    cumulative: dict[str, list[int]] = {island: [] for island in islands}
    running: dict[str, int] = {island: 0 for island in islands}
    for gen in generations:
        for island in islands:
            running[island] += per_gen_per_island[gen].get(island, 0)
            cumulative[island].append(running[island])
    fig = go.Figure()
    for island in islands:
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=cumulative[island],
                mode="lines+markers",
                name=island,
                hovertemplate=f"<b>{island}</b><br>gen: %{{x}}<br>cumulative: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cumulative Creations by Island (interactive)",
        xaxis_title="Generation Created",
        yaxis_title="# Created Organisms (cumulative)",
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plotly(fig, out_path)


def render_cumulative_max_score_by_model_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    evaluated = _evaluated_records(records)
    if not evaluated:
        return None
    models = sorted({r.model_label for r in evaluated})
    generations = sorted({r.generation_created for r in evaluated})

    per_model_per_gen_max: dict[str, dict[int, float]] = {m: {} for m in models}
    for r in evaluated:
        bucket = per_model_per_gen_max[r.model_label]
        prev = bucket.get(r.generation_created)
        if prev is None or r.simple_score > prev:
            bucket[r.generation_created] = float(r.simple_score)

    fig = go.Figure()
    for model in models:
        running: float | None = None
        ys: list[float | None] = []
        for gen in generations:
            sample = per_model_per_gen_max[model].get(gen)
            if sample is not None and (running is None or sample > running):
                running = sample
            ys.append(running)
        if not any(value is not None for value in ys):
            continue
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=[float("nan") if value is None else value for value in ys],
                mode="lines+markers",
                name=model,
                hovertemplate=f"<b>{model}</b><br>gen: %{{x}}<br>best so far: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cumulative Max Score by Model (interactive)",
        xaxis_title="Generation Created",
        yaxis_title="Best Simple Score so far",
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plotly(fig, out_path)


# ---------------------------------------------------------------------------
# Survival panels (ratio + cumulative counts). Three Ox bases.
# ---------------------------------------------------------------------------


def _record_time(record: OrganismVizRecord) -> datetime | None:
    return record.simple_eval_finished_at or record.created_at


def _render_survival_plotly(
    *,
    xs: list[Any],
    scored_series: list[int],
    dead_series: list[int],
    ratio_series: list[float],
    out_path: Path,
    title: str,
    xaxis_title: str,
    hover_x_format: str,
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=scored_series,
            mode="lines",
            name="Scored (cumulative)",
            line={"color": "#2ca02c", "width": 2.0},
            hovertemplate=(
                f"<b>scored</b><br>{xaxis_title}: %{{x{hover_x_format}}}<br>count: %{{y}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=dead_series,
            mode="lines",
            name="Dead (cumulative)",
            line={"color": "#d62728", "width": 2.0},
            hovertemplate=(
                f"<b>dead</b><br>{xaxis_title}: %{{x{hover_x_format}}}<br>count: %{{y}}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ratio_series,
            mode="lines",
            name="Survival Ratio",
            line={"color": "#1f77b4", "width": 1.8, "dash": "dash"},
            yaxis="y2",
            hovertemplate=(
                f"<b>ratio</b><br>{xaxis_title}: %{{x{hover_x_format}}}<br>"
                "scored/(scored+dead): %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis={"title": "# Organisms (cumulative)"},
        yaxis2={
            "title": "scored / (scored + dead)",
            "overlaying": "y",
            "side": "right",
            "range": [0.0, 1.0],
        },
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return _write_plotly(fig, out_path)


def render_survival_by_evaluations_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    ordered = sorted(
        records,
        key=lambda r: (
            r.created_at.timestamp() if r.created_at else float("inf"),
            r.generation_created,
            r.organism_id,
        ),
    )
    if not ordered:
        return None
    xs = list(range(1, len(ordered) + 1))
    cumulative_scored = 0
    cumulative_dead = 0
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    for r in ordered:
        if r.simple_score is None:
            cumulative_dead += 1
        else:
            cumulative_scored += 1
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)
    return _render_survival_plotly(
        xs=xs,
        scored_series=scored_series,
        dead_series=dead_series,
        ratio_series=ratio_series,
        out_path=out_path,
        title="Survival by Creation Index (interactive)",
        xaxis_title="# Tracked Organisms",
        hover_x_format="",
    )


def render_survival_by_runtime_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    timed = [r for r in records if _record_time(r) is not None]
    if not timed:
        return None
    timed.sort(key=lambda r: (_record_time(r), r.organism_id))
    t0 = _record_time(timed[0])
    xs: list[float] = []
    cumulative_scored = 0
    cumulative_dead = 0
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    for r in timed:
        elapsed = (_record_time(r) - t0).total_seconds()
        if r.simple_score is None:
            cumulative_dead += 1
        else:
            cumulative_scored += 1
        xs.append(elapsed)
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)
    return _render_survival_plotly(
        xs=xs,
        scored_series=scored_series,
        dead_series=dead_series,
        ratio_series=ratio_series,
        out_path=out_path,
        title="Survival by Runtime (interactive)",
        xaxis_title="Elapsed Runtime (s)",
        hover_x_format=":.0f",
    )


def render_survival_by_generation_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    if not records:
        return None
    per_generation: dict[int, dict[str, int]] = defaultdict(lambda: {"scored": 0, "dead": 0})
    for r in records:
        bucket = per_generation[r.generation_created]
        if r.simple_score is None:
            bucket["dead"] += 1
        else:
            bucket["scored"] += 1
    generations = sorted(per_generation)
    cumulative_scored = 0
    cumulative_dead = 0
    scored_series: list[int] = []
    dead_series: list[int] = []
    ratio_series: list[float] = []
    for gen in generations:
        cumulative_scored += per_generation[gen]["scored"]
        cumulative_dead += per_generation[gen]["dead"]
        scored_series.append(cumulative_scored)
        dead_series.append(cumulative_dead)
        total = cumulative_scored + cumulative_dead
        ratio_series.append(cumulative_scored / total if total else 0.0)
    return _render_survival_plotly(
        xs=generations,
        scored_series=scored_series,
        dead_series=dead_series,
        ratio_series=ratio_series,
        out_path=out_path,
        title="Survival by Generation (interactive)",
        xaxis_title="Generation Created",
        hover_x_format="",
    )


# ---------------------------------------------------------------------------
# Operators total bar — kept as plotly for parity with the matplotlib panel.
# ---------------------------------------------------------------------------


def render_operators_total_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
    context_records: list[OrganismVizRecord] | None = None,
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    lookup = {r.organism_id: r for r in (context_records or records)}
    totals: dict[str, int] = {op: 0 for op in _OPERATOR_PLOT_ORDER}
    for r in records:
        op = _operator_category(r, lookup)
        if op is not None:
            totals[op] += 1
    categories = [op for op in _OPERATOR_PLOT_ORDER if totals[op] > 0]
    if not categories:
        return None
    fig = go.Figure(
        data=[
            go.Bar(
                x=[_OPERATOR_PLOT_LABELS[op] for op in categories],
                y=[totals[op] for op in categories],
                marker_color=[_OPERATOR_PLOT_COLORS[op] for op in categories],
                hovertemplate="<b>%{x}</b><br>count: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Created Organisms by Operator (interactive)",
        xaxis_title="Operator",
        yaxis_title="# Created Organisms",
        template="plotly_white",
    )
    return _write_plotly(fig, out_path)


# ---------------------------------------------------------------------------
# Evaluations per generation — stacked bars per island.
# ---------------------------------------------------------------------------


def render_evaluations_per_generation_plotly(
    records: list[OrganismVizRecord], *, out_path: Path
) -> Path | None:
    go = _try_import_plotly()
    if go is None:
        return None
    evaluated = _evaluated_records(records)
    if not evaluated:
        return None
    islands = sorted({r.island_id for r in evaluated})
    generation_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in evaluated:
        generation_counts[r.generation_created][r.island_id] += 1
    generations = sorted(generation_counts)
    fig = go.Figure()
    for island in islands:
        fig.add_trace(
            go.Bar(
                x=generations,
                y=[generation_counts[gen].get(island, 0) for gen in generations],
                name=island,
                hovertemplate=f"<b>{island}</b><br>gen: %{{x}}<br>count: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Evaluations per Generation (interactive)",
        xaxis_title="Generation Created",
        yaxis_title="# Evaluated Organisms",
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plotly(fig, out_path)
