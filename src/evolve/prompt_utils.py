"""Canonical prompt asset loading helpers for organism-first evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def _default_conf_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "conf" / "prompts"


def _default_prompt_relative_path(key: str) -> Path:
    mapping = {
        "project_context": Path("shared/project_context.txt"),
        "seed_system": Path("seed/system.txt"),
        "seed_user": Path("seed/user.txt"),
        "mutation_system": Path("mutation/system.txt"),
        "mutation_user": Path("mutation/user.txt"),
        "crossover_system": Path("crossover/system.txt"),
        "crossover_user": Path("crossover/user.txt"),
    }
    try:
        return mapping[key]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt bundle key: {key}") from exc


def _read_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required canonical prompt asset was not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _resolve_prompt_path(
    cfg: DictConfig,
    key: str,
) -> Path:
    prompts_cfg = cfg.evolver.get("prompts")
    if prompts_cfg is not None and key in prompts_cfg:
        candidate = Path(str(prompts_cfg[key])).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Configured prompt file for '{key}' was not found: {candidate}")

    return (_default_conf_prompts_dir() / _default_prompt_relative_path(key)).resolve()


def load_prompt_bundle(cfg: DictConfig) -> PromptBundle:
    """Load canonical prompt texts from `conf/prompts/**` or explicit overrides."""

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
    )


def compose_system_prompt(project_context: str, operator_system_prompt: str) -> str:
    """Compose constant project context with an operator-specific system prompt."""

    pieces = [piece.strip() for piece in (project_context, operator_system_prompt) if piece and piece.strip()]
    return "\n\n".join(pieces).strip()
