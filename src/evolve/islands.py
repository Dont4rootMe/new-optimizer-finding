"""Helpers for building research islands from a flat id list (Shinka-style)."""

from __future__ import annotations

from src.evolve.types import Island


def synthesize_islands_from_ids(island_ids: list[str]) -> list[Island]:
    """Build :class:`Island` objects from a flat list of ids.

    Used by the ``from_seed`` initialization mode (Shinka / FunSearch
    style): the framework copies a single baseline ``implementation.py``
    into one organism per (island, slot) tuple instead of asking the
    LLM to author handwritten organisms per-island. Each island still
    needs an ``Island`` record so that downstream code paths
    (``parent_island_sampler``, ``select_top_k_per_island``,
    ``cross_island_partner_sampler``, visualisation, etc.) keep working.

    See ``conf/evolver/awtf2025_heuristic.yaml`` ``evolver.islands.mode``
    for the user-facing switch.
    """

    if not island_ids:
        raise ValueError(
            "synthesize_islands_from_ids needs at least one island_id; "
            "configure ``evolver.islands.island_ids`` in the yaml."
        )

    seen: set[str] = set()
    islands: list[Island] = []
    for raw_id in island_ids:
        island_id = str(raw_id).strip()
        if not island_id:
            raise ValueError("Empty island_id in evolver.islands.island_ids")
        if island_id in seen:
            raise ValueError(f"Duplicate island_id {island_id!r} in evolver.islands.island_ids")
        seen.add(island_id)
        islands.append(
            Island(
                island_id=island_id,
                name=island_id.replace("_", " ").strip(),
            )
        )
    return islands
