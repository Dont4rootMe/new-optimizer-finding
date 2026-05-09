"""Interactive Plotly graphs for the evolution dashboard.

The headline graph mirrors `_plot_best_vs_evaluations` from
`src.evolve.visualization` but adds a hover-trace: hovering any point
highlights its full maternal-ancestor chain, so the user can inspect the
lineage of every organism, not just the current best.

Plotly itself has no built-in "highlight ancestors on hover" — we ship a
small JS post-script that listens to ``plotly_hover``/``plotly_unhover``,
reads the hovered point's ``customdata.ancestors`` list, and toggles
opacity + colour on a dedicated highlight trace. The post-script is
inlined into the saved HTML so the file works offline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.evolve.visualization import (
    OrganismVizRecord,
    _evaluated_records,
    _maternal_lineage_points_by_evaluations,
    build_ancestor_chains,
)

LOGGER = logging.getLogger(__name__)


def render_best_vs_evaluations_plotly(
    records: list[OrganismVizRecord],
    *,
    out_path: Path,
) -> Path | None:
    """Render the interactive Best Score vs Evaluations graph.

    Returns the written file path, or ``None`` if Plotly is unavailable or no
    evaluated organisms exist yet.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dep
        LOGGER.warning("plotly not installed; skipping interactive HTML (%s)", exc)
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
