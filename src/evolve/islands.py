"""Helpers for loading research islands from text assets."""

from __future__ import annotations

from pathlib import Path

from src.evolve.types import Island


def load_islands(islands_dir: str | Path) -> list[Island]:
    """Load canonical island definitions from `conf/prompts/islands/*.txt`."""

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
