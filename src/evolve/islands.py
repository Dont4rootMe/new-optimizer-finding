"""Helpers for loading research islands from text assets."""

from __future__ import annotations

from pathlib import Path

from src.evolve.types import Island


def load_islands(islands_dir: str | Path) -> list[Island]:
    """Load canonical island definitions from the configured task-family island directory."""

    root = Path(islands_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Island directory was not found: {root}")

    islands: list[Island] = []
    for path in sorted(root.glob("*.txt")):
        description_text = path.read_text(encoding="utf-8").strip()
        if not description_text:
            raise ValueError(f"Island file is empty: {path}")
        island_id = path.stem
        islands.append(
            Island(
                island_id=island_id,
                name=island_id.replace("_", " ").strip(),
                description_path=str(path),
                description_text=description_text,
            )
        )

    if not islands:
        raise ValueError(f"No island descriptions found in {root}")

    return islands


def synthesize_islands_from_ids(
    island_ids: list[str],
    *,
    description_template: str = "Seeded from baseline implementation; diversity emerges via evolution.",
) -> list[Island]:
    """Build :class:`Island` objects from a flat list of ids without prompt files.

    Used by the ``from_seed`` initialization mode (Shinka / FunSearch
    style): the framework copies a single baseline ``implementation.py``
    into one organism per (island, slot) tuple instead of asking the
    LLM to author handwritten organisms per-island. Each island still
    needs an ``Island`` record so that downstream code paths
    (``parent_island_sampler``, ``select_top_k_per_island``,
    ``cross_island_partner_sampler``, visualisation, etc.) keep working
    unchanged — ``description_text`` is left as a short generic placeholder
    because nothing reads it in this mode.

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
                description_path="",
                description_text=description_template,
            )
        )
    return islands
