"""Task-blind canonical hypothesis artifact IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.organisms.hypothesis_render import render_genetic_code_markdown


GENOME_FILENAME = "genome.json"
GENETIC_CODE_FILENAME = "genetic_code.md"

_TOP_LEVEL_KEYS = {
    "schema_name",
    "schema_version",
    "task_family",
    "task_name",
    "representation",
    "organism_id",
    "slot_order",
    "linkage_groups",
    "global_hypothesis",
    "slots",
    "render_fields",
}


def _provider_value(schema_provider: Any, name: str) -> Any:
    if not hasattr(schema_provider, name):
        raise ValueError(f"Schema provider is missing required constant {name}.")
    return getattr(schema_provider, name)


def _find_null(value: Any, path: str = "$") -> str | None:
    if value is None:
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]"
            found = _find_null(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found = _find_null(child, f"{path}[{idx}]")
            if found is not None:
                return found
    return None


def _schema_validation_callable(schema_provider: Any):
    candidate = getattr(schema_provider, "validate_genome", None)
    if callable(candidate):
        return candidate

    for name in sorted(dir(schema_provider)):
        if name.startswith("validate_") and name.endswith("_genome_v1"):
            candidate = getattr(schema_provider, name)
            if callable(candidate):
                return candidate

    raise ValueError("Schema provider must expose validate_genome or a validate_*_genome_v1 function.")


def _validate_generic_top_level(genome: dict[str, Any], schema_provider: Any) -> None:
    keys = set(genome.keys())
    missing = sorted(_TOP_LEVEL_KEYS.difference(keys))
    extra = sorted(keys.difference(_TOP_LEVEL_KEYS))
    if missing:
        raise ValueError(f"Canonical genome is missing top-level keys: {', '.join(missing)}.")
    if extra:
        raise ValueError(f"Canonical genome contains extra top-level keys: {', '.join(extra)}.")

    null_path = _find_null(genome)
    if null_path is not None:
        raise ValueError(f"Canonical genome contains null at {null_path}.")

    expected_constants = {
        "schema_name": "SCHEMA_NAME",
        "schema_version": "SCHEMA_VERSION",
        "task_family": "TASK_FAMILY",
        "task_name": "TASK_NAME",
        "representation": "REPRESENTATION",
    }
    for genome_key, provider_key in expected_constants.items():
        value = genome.get(genome_key)
        if value != _provider_value(schema_provider, provider_key):
            raise ValueError(
                f"Canonical genome has wrong {genome_key}: "
                f"expected {_provider_value(schema_provider, provider_key)!r}, got {value!r}."
            )

    organism_id = genome.get("organism_id")
    if not isinstance(organism_id, str) or organism_id.strip() != organism_id or not organism_id:
        raise ValueError("Canonical genome organism_id must be a non-empty stripped string.")

    slot_order = genome.get("slot_order")
    expected_slot_order = list(_provider_value(schema_provider, "SLOT_ORDER"))
    if slot_order != expected_slot_order:
        raise ValueError(
            f"Canonical genome slot_order must be exactly {expected_slot_order!r}; got {slot_order!r}."
        )

    if not isinstance(genome.get("global_hypothesis"), dict):
        raise ValueError("Canonical genome global_hypothesis must be an object.")
    if not isinstance(genome.get("slots"), dict):
        raise ValueError("Canonical genome slots must be an object.")
    if not isinstance(genome.get("render_fields"), dict):
        raise ValueError("Canonical genome render_fields must be an object.")
    if not isinstance(genome.get("linkage_groups"), list):
        raise ValueError("Canonical genome linkage_groups must be a list.")


def validate_canonical_genome(genome: dict, schema_provider) -> None:
    """Validate a canonical genome using task-family-provided schema rules."""

    if not isinstance(genome, dict):
        raise ValueError("Canonical genome must be a JSON object.")

    _validate_generic_top_level(genome, schema_provider)
    _schema_validation_callable(schema_provider)(genome)


def read_canonical_genome(organism_dir: Path, schema_provider) -> dict:
    """Read and validate ``genome.json`` from an organism directory."""

    path = Path(organism_dir) / GENOME_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Canonical genome is missing at {path}.")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Canonical genome at {path} is not valid JSON: {exc}") from exc

    validate_canonical_genome(payload, schema_provider)
    return payload


def write_canonical_genome(organism_dir: Path, genome: dict, schema_provider) -> None:
    """Validate and write ``genome.json`` plus rendered ``genetic_code.md``."""

    validate_canonical_genome(genome, schema_provider)

    target_dir = Path(organism_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    genome_path = target_dir / GENOME_FILENAME
    markdown_path = target_dir / GENETIC_CODE_FILENAME

    genome_path.write_text(
        json.dumps(genome, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_genetic_code_markdown(genome, schema_provider),
        encoding="utf-8",
    )
