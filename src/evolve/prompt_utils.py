"""Canonical prompt asset loading helpers for organism-first evolution."""

from __future__ import annotations

from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass(slots=True)
class PromptBundle:
    project_context: str
    seed_system: str
    seed_user: str
    mutation_system: str
    mutation_user: str
    crossover_system: str
    crossover_user: str
    implementation_system: str
    implementation_user: str
    implementation_template: str

from pathlib import Path


def _read_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required canonical prompt asset was not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _resolve_prompt_path(cfg: DictConfig, key: str) -> Path:
    prompts_cfg = cfg.evolver.get("prompts")
    if prompts_cfg is None or key not in prompts_cfg:
        raise KeyError(f"evolver.prompts is missing required key '{key}'.")
    candidate = Path(str(prompts_cfg[key])).expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Configured prompt file for '{key}' was not found: {candidate}")


def load_prompt_bundle(cfg: DictConfig) -> PromptBundle:
    """Load canonical prompt texts from explicit config paths."""

    project_context_path = _resolve_prompt_path(cfg, "project_context")
    project_context = _read_path(project_context_path)

    return PromptBundle(
        project_context=project_context,
        seed_system=_read_path(_resolve_prompt_path(cfg, "seed_system")),
        seed_user=_read_path(_resolve_prompt_path(cfg, "seed_user")),
        mutation_system=_read_path(_resolve_prompt_path(cfg, "mutation_system")),
        mutation_user=_read_path(_resolve_prompt_path(cfg, "mutation_user")),
        crossover_system=_read_path(_resolve_prompt_path(cfg, "crossover_system")),
        crossover_user=_read_path(_resolve_prompt_path(cfg, "crossover_user")),
        implementation_system=_read_path(_resolve_prompt_path(cfg, "implementation_system")),
        implementation_user=_read_path(_resolve_prompt_path(cfg, "implementation_user")),
        implementation_template=_read_path(_resolve_prompt_path(cfg, "implementation_template")),
    )


def compose_system_prompt(project_context: str, operator_system_prompt: str) -> str:
    """Compose constant project context with an operator-specific system prompt."""

    pieces = [piece.strip() for piece in (project_context, operator_system_prompt) if piece and piece.strip()]
    return "\n\n".join(pieces).strip()
