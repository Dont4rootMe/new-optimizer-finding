"""Canonical prompt asset loading helpers for organism-first evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from omegaconf import DictConfig

from pathlib import Path
from typing import Any

# Sentinel placeholder kept inside the Step-2 user template until the
# generator either substitutes the actual Step-1 rationale text (two-step
# mode) or a "rationalization disabled" stub (single-call mode). Using a
# bare placeholder string lets str.format(...) leave it untouched as long
# as the kwarg explicitly carries the same string.
RATIONALIZATION_PLACEHOLDER = "{rationalization}"
RATIONALIZATION_SINGLE_CALL_STUB = "(rationalization disabled — single-call mode)"

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class PromptBundle:
    project_context: str
    seed_system: str
    seed_user: str
    mutation_system: str
    mutation_user: str
    mutation_novelty_system: str
    mutation_novelty_user: str
    crossover_system: str
    crossover_user: str
    crossover_novelty_system: str
    crossover_novelty_user: str
    implementation_system: str
    implementation_user: str
    implementation_template: str
    repair_system: str
    repair_user: str
    genome_schema: str = ""
    compatibility_seed_system: str = ""
    compatibility_seed_user: str = ""
    compatibility_mutation_system: str = ""
    compatibility_mutation_user: str = ""
    compatibility_crossover_system: str = ""
    compatibility_crossover_user: str = ""
    # Step-1 prompts for the two-step design pipeline (rationalization →
    # formalization). Empty strings mean the family hasn't migrated and the
    # single-call legacy path is used unconditionally.
    mutation_rationalization_system: str = ""
    mutation_rationalization_user: str = ""
    crossover_rationalization_system: str = ""
    crossover_rationalization_user: str = ""

    @property
    def supports_two_step_mutation(self) -> bool:
        return bool(self.mutation_rationalization_system and self.mutation_rationalization_user)

    @property
    def supports_two_step_crossover(self) -> bool:
        return bool(self.crossover_rationalization_system and self.crossover_rationalization_user)


@dataclass(slots=True)
class DesignPromptBundle:
    """Carries the prompts for one design-stage attempt.

    Step 1 (rationalization) prompts are ``None`` when the family hasn't
    migrated to the two-step pipeline or when single-call mode is forced
    by config. Step 2 (formalization) prompts always exist; the user
    template carries the ``{rationalization}`` placeholder unsubstituted —
    the generator fills it in either with the Step-1 LLM response (two
    step) or with the single-call stub before issuing the LLM call.
    """

    formalization_system: str
    formalization_user_template: str
    rationalization_system: str | None = None
    rationalization_user: str | None = None
    # Lineage regime hint used at Step 1 prompt-render time. Empty when no
    # convergence diagnosis is available or single-call mode is active.
    lineage_regime_hint: str = ""

    @property
    def is_two_step(self) -> bool:
        return bool(self.rationalization_system and self.rationalization_user)

    def render_formalization(self, rationale_text: Any) -> tuple[str, str]:
        """Substitute the rationalization placeholder and return ``(system, user)``.

        ``rationale_text`` is accepted as ``Any`` (not strictly ``str``)
        because upstream callers may pass an LLM-response wrapper that
        renders to a string via ``str(...)``. Coercing here is defense in
        depth: a single non-string sneaking through used to kill the entire
        evolution run with ``TypeError: replace() argument 2 must be str``.
        """

        if rationale_text is None:
            substitution = RATIONALIZATION_SINGLE_CALL_STUB
        else:
            substitution = str(rationale_text) or RATIONALIZATION_SINGLE_CALL_STUB
        rendered_user = self.formalization_user_template.replace(
            RATIONALIZATION_PLACEHOLDER,
            substitution,
        )
        return self.formalization_system, rendered_user


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
    if not candidate.is_absolute():
        repo_relative = (REPO_ROOT / candidate).resolve()
        if repo_relative.exists():
            return repo_relative
    raise FileNotFoundError(f"Configured prompt file for '{key}' was not found: {candidate}")


def _read_optional_prompt_asset(cfg: DictConfig, key: str) -> str:
    prompts_cfg = cfg.evolver.get("prompts")
    if prompts_cfg is None or key not in prompts_cfg:
        return ""
    return _read_path(_resolve_prompt_path(cfg, key))


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
        mutation_novelty_system=_read_path(_resolve_prompt_path(cfg, "mutation_novelty_system")),
        mutation_novelty_user=_read_path(_resolve_prompt_path(cfg, "mutation_novelty_user")),
        crossover_system=_read_path(_resolve_prompt_path(cfg, "crossover_system")),
        crossover_user=_read_path(_resolve_prompt_path(cfg, "crossover_user")),
        crossover_novelty_system=_read_path(_resolve_prompt_path(cfg, "crossover_novelty_system")),
        crossover_novelty_user=_read_path(_resolve_prompt_path(cfg, "crossover_novelty_user")),
        implementation_system=_read_path(_resolve_prompt_path(cfg, "implementation_system")),
        implementation_user=_read_path(_resolve_prompt_path(cfg, "implementation_user")),
        implementation_template=_read_path(_resolve_prompt_path(cfg, "implementation_template")),
        repair_system=_read_path(_resolve_prompt_path(cfg, "repair_system")),
        repair_user=_read_path(_resolve_prompt_path(cfg, "repair_user")),
        genome_schema=_read_optional_prompt_asset(cfg, "genome_schema"),
        compatibility_seed_system=_read_optional_prompt_asset(cfg, "compatibility_seed_system"),
        compatibility_seed_user=_read_optional_prompt_asset(cfg, "compatibility_seed_user"),
        compatibility_mutation_system=_read_optional_prompt_asset(cfg, "compatibility_mutation_system"),
        compatibility_mutation_user=_read_optional_prompt_asset(cfg, "compatibility_mutation_user"),
        compatibility_crossover_system=_read_optional_prompt_asset(cfg, "compatibility_crossover_system"),
        compatibility_crossover_user=_read_optional_prompt_asset(cfg, "compatibility_crossover_user"),
        mutation_rationalization_system=_read_optional_prompt_asset(cfg, "mutation_rationalization_system"),
        mutation_rationalization_user=_read_optional_prompt_asset(cfg, "mutation_rationalization_user"),
        crossover_rationalization_system=_read_optional_prompt_asset(cfg, "crossover_rationalization_system"),
        crossover_rationalization_user=_read_optional_prompt_asset(cfg, "crossover_rationalization_user"),
    )


def compose_system_prompt(project_context: str, operator_system_prompt: str) -> str:
    """Compose constant project context with an operator-specific system prompt."""

    pieces = [piece.strip() for piece in (project_context, operator_system_prompt) if piece and piece.strip()]
    return "\n\n".join(pieces).strip()
