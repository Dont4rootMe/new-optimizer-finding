"""Lineage-aware regime summarizer (Tier 2 of the bandit-era rework).

The two-step design pipeline needs an explicit signal that says "the last
few ancestors all converged on the same wall family / radius regime /
routing family — break out". The rationalization stage (Step 1) reads this
hint and uses it to inform its WHAT_TO_REMOVE / WHAT_TO_ADD_OR_INVENT
directives.

The hint is computed cheaply by scanning each ancestor's
``change_description`` (already aggregated on the parent's lineage) for
family-specific mechanism keywords. The keyword map is family-owned and
lives next to the prompts at
``conf/experiments/<family>/prompts/shared/regime_keywords.yaml``.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_DEFAULT_LOOKBACK = 8
_MIN_LINEAGE_FOR_CONVERGENCE = 3
_CONVERGENCE_THRESHOLD = 0.7  # share of recent ancestors using the same family

# Built-in fallback maps. If the family ships a YAML override we use that;
# otherwise these defaults keep both production families usable out of the
# box without requiring an additional asset.
_DEFAULT_KEYWORD_MAPS: dict[str, dict[str, list[str]]] = {
    "awtf2025_heuristic": {
        "wall_family": [
            "no_walls", "no-walls", "no walls", "no added walls",
            "central spine", "central cross", "corridor",
            "boundary frame", "wall family", "added walls",
        ],
        "grouping_family": [
            "single group", "quadrant grouping", "quadrant-grouping",
            "direction-based grouping", "target-cluster", "axis-bucket",
            "per-robot group", "individual group",
        ],
        "routing_family": [
            "greedy", "greedy routing", "pressure-aware", "hub-staged",
            "direction-profile", "per-round priority", "phased release",
            "manhattan", "centroid routing",
        ],
        "repair_regime": [
            "no repair", "individual-fallback", "side-step",
            "single-step backtrack", "wait counter", "stagnation trigger",
            "deferred repair", "bottleneck repair",
        ],
    },
    "circle_packing_shinka": {
        "packing_family": [
            "hex packing", "hexagonal", "square grid", "triangular scaffold",
            "centered grid", "concentric rings", "rings",
        ],
        "radius_regime": [
            "single radius", "uniform radii", "uniform radius",
            "mixed radii", "graded radii", "central large radius",
            "radius variance",
        ],
        "symmetry_family": [
            "rotational symmetry", "4-fold", "6-fold", "mirror symmetry",
            "axis symmetry", "asymmetric placement",
        ],
        "repair_family": [
            "no repair", "worst-conflict repair", "global shrink",
            "local relax", "perturbation repair", "feasibility pass",
        ],
    },
}


def _load_keyword_map(family: str | None) -> dict[str, list[str]]:
    """Resolve the family-specific keyword map.

    Order of precedence:
      1. YAML at ``conf/experiments/<family>/prompts/shared/regime_keywords.yaml``
      2. Built-in default in ``_DEFAULT_KEYWORD_MAPS``
      3. Empty map (then we just return an "unclear" hint)
    """

    if not family:
        return {}
    candidate = Path("conf") / "experiments" / family / "prompts" / "shared" / "regime_keywords.yaml"
    if candidate.exists():
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            LOGGER.warning("PyYAML not available; using built-in regime keywords for %s", family)
        else:
            try:
                payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to read regime keywords for %s", family)
            else:
                if isinstance(payload, dict):
                    return {
                        str(category): [str(keyword) for keyword in keywords]
                        for category, keywords in payload.items()
                        if isinstance(keywords, list)
                    }
    return _DEFAULT_KEYWORD_MAPS.get(family, {})


def _entry_text(entry: Any) -> str:
    """Pull comparable text out of a lineage entry (dict-shaped)."""

    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        parts: list[str] = []
        for key in ("change_description", "summary", "novelty_summary"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value)
        return "\n".join(parts).lower()
    # OrganismMeta-like objects expose ``lineage`` as list of dicts already; the
    # caller should pre-extract those before getting here. Fall back to repr().
    return str(entry).lower()


def summarize_recent_regime(
    lineage: list[Any],
    *,
    family: str | None = None,
    lookback: int = _DEFAULT_LOOKBACK,
) -> str:
    """Return a short, human-readable regime hint based on the last ``lookback`` ancestors.

    Empty or shallow lineages emit a "signal: unclear" hint so the
    rationalization stage doesn't over-interpret. When recent ancestors
    share the same family along an axis with high probability we mark it
    "converged" and explicitly suggest a break-out target.
    """

    if not lineage:
        return _format_hint(family=family, recent_count=0, observations={}, gaps={})

    recent = list(lineage)[-lookback:] if lookback > 0 else list(lineage)
    keyword_map = _load_keyword_map(family)
    if not keyword_map:
        return _format_hint(
            family=family,
            recent_count=len(recent),
            observations={},
            gaps={},
            note="no regime keyword map available — hint suppressed",
        )

    observations: dict[str, Counter[str]] = {category: Counter() for category in keyword_map}
    for entry in recent:
        text = _entry_text(entry)
        if not text:
            continue
        for category, keywords in keyword_map.items():
            matched = next((keyword for keyword in keywords if keyword in text), None)
            if matched is not None:
                observations[category][matched] += 1

    # Identify under-explored axes: categories where NO keyword was observed
    # in the lookback window.
    gaps = {
        category: sorted(keyword_map[category])[:3]
        for category, counter in observations.items()
        if not counter
    }
    return _format_hint(
        family=family,
        recent_count=len(recent),
        observations=observations,
        gaps=gaps,
    )


def _format_hint(
    *,
    family: str | None,
    recent_count: int,
    observations: dict[str, Counter[str]],
    gaps: dict[str, list[str]],
    note: str | None = None,
) -> str:
    if recent_count < _MIN_LINEAGE_FOR_CONVERGENCE:
        return (
            "signal: unclear — too few ancestors to detect convergence "
            f"(observed {recent_count}, need >= {_MIN_LINEAGE_FOR_CONVERGENCE})."
        )

    lines: list[str] = [f"signal: scanning last {recent_count} ancestors for regime convergence"]
    if family:
        lines[0] = f"signal: scanning last {recent_count} ancestors of family {family!r} for regime convergence"
    if note:
        lines.append(f"  note: {note}")

    converged: list[str] = []
    for category, counter in observations.items():
        if not counter:
            continue
        top_keyword, top_count = counter.most_common(1)[0]
        share = top_count / max(recent_count, 1)
        if share >= _CONVERGENCE_THRESHOLD:
            converged.append(f"  - {category}: CONVERGED on {top_keyword!r} ({top_count}/{recent_count})")
        else:
            distribution = ", ".join(f"{kw}={cnt}" for kw, cnt in counter.most_common(3))
            lines.append(f"  - {category}: mixed ({distribution})")
    if converged:
        lines.append("converged axes (break-out candidates):")
        lines.extend(converged)
    if gaps:
        gap_lines = ", ".join(f"{category}={', '.join(samples)}" for category, samples in gaps.items())
        lines.append(f"under-explored axes (no recent ancestor used these): {gap_lines}")
    if not converged and not gaps:
        lines.append("regime appears diverse across the observed window — no specific break-out target")
    return "\n".join(lines)
