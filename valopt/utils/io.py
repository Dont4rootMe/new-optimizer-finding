"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create directory recursively and return as Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save dictionary to JSON with stable formatting."""

    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return out_path


def save_yaml_text(path: str | Path, yaml_text: str) -> Path:
    """Persist pre-rendered YAML text to disk."""

    out_path = Path(path)
    ensure_dir(out_path.parent)
    out_path.write_text(yaml_text, encoding="utf-8")
    return out_path
