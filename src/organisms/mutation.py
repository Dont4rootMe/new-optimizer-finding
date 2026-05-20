"""Mutation operator for canonical organism genetic code."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.evolve.prompt_utils import (
    DesignPromptBundle,
    PromptBundle,
    RATIONALIZATION_PLACEHOLDER,
    compose_system_prompt,
)
from src.evolve.storage import utc_now_iso
from src.evolve.types import OrganismMeta
from src.organisms.compatibility import (
    CompatibilityCheckContext,
    CompatibilityJudgment,
    CompatibilityValidationContext,
    build_mutation_compatibility_prompt,
    format_compatibility_rejection_feedback,
)
from src.organisms.lineage_regime import summarize_recent_regime
from src.organisms.novelty import (
    NoveltyCheckContext,
    build_mutation_novelty_prompt,
    format_novelty_rejection_feedback,
)
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_implementation_code,
    format_lineage_summary,
    format_parent_fitness_signal,
    read_organism_genetic_code,
    read_organism_implementation,
    read_organism_lineage,
)

LOGGER = logging.getLogger(__name__)


def _detect_family_id(generator: Any) -> str | None:
    """Best-effort detection of the active experiment family.

    Used by the lineage regime summarizer to pick the right keyword map.
    Resolves via the generator's Hydra cfg (``cfg.evolver.family_id``) or
    falls back to the ``conf/evolver/<family>.yaml`` path encoded into the
    PromptBundle's prompt asset roots.
    """

    cfg = getattr(generator, "cfg", None)
    if cfg is not None:
        try:
            from omegaconf import OmegaConf  # local import to avoid hard dep at module load

            value = OmegaConf.select(cfg, "evolver.family_id", default=None)
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to resolve evolver.family_id from cfg", exc_info=True)

    # Fallback: scan a known prompt path for "experiments/<family>/" — this
    # always works for the canonical Hydra-composed configs.
    bundle = getattr(generator, "prompt_bundle", None)
    candidate_assets = (
        getattr(bundle, "mutation_rationalization_system", "") or "",
        getattr(bundle, "mutation_system", "") or "",
        getattr(bundle, "genome_schema", "") or "",
    )
    for asset in candidate_assets:
        marker = "experiments/"
        if marker in asset:
            try:
                tail = asset.split(marker, 1)[1]
                family = tail.split("/", 1)[0]
                if family:
                    return family
            except Exception:  # noqa: BLE001
                continue
    return None


def _maybe_run_step1(
    *,
    generator: Any,
    bundle: Any,
    organism_id: str,
    generation: int,
    operator: str,
    org_dir: Path | None = None,
) -> str | None:
    """Run the Step-1 rationalization LLM call if the family supports it.

    Returns the rationale text on success, ``None`` when:
      - the family hasn't migrated (bundle.is_two_step is False), or
      - the generator doesn't expose ``run_rationalization_stage`` (legacy
        callers / test fakes), or
      - the call itself fails (logged; Step 2 proceeds in single-call mode).
    """

    if not getattr(bundle, "is_two_step", False):
        return None
    runner = getattr(generator, "run_rationalization_stage", None)
    if runner is None:
        return None
    try:
        return runner(
            rationalization_system=bundle.rationalization_system,
            rationalization_user=bundle.rationalization_user,
            organism_id=organism_id,
            generation=generation,
            operator=operator,
            org_dir=org_dir,
        )
    except Exception:  # noqa: BLE001
        LOGGER.exception(
            "Step 1 rationalization failed for organism %s (operator=%s); "
            "falling back to single-call formalization",
            organism_id,
            operator,
        )
        return None


def prune_gene_pool(
    genes: list[str],
    delete_probability: float = 0.2,
    rng: random.Random | None = None,
) -> tuple[list[str], list[str]]:
    """Delete genes from a pool with probability `delete_probability`."""

    if rng is None:
        rng = random.Random()

    surviving: list[str] = []
    removed: list[str] = []

    for trait in genes:
        if rng.random() < delete_probability:
            removed.append(trait.strip())
        else:
            surviving.append(trait.strip())

    if not surviving and genes:
        rescued = rng.choice(genes).strip()
        surviving.append(rescued)
        if rescued in removed:
            removed.remove(rescued)

    return surviving, removed


def _build_mutate_prompt(
    inherited_genes: list[str],
    removed_genes: list[str],
    parent: OrganismMeta,
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for mutation LLM call (legacy path).

    Kept for callers that still want flat strings — internally it derives the
    formalization prompts from :func:`build_mutation_design_bundle` and
    substitutes the rationalization placeholder with the single-call stub.
    """

    bundle = build_mutation_design_bundle(
        parent=parent,
        prompts=prompts,
        novelty_feedback=novelty_feedback,
        compatibility_feedback=compatibility_feedback,
    )
    return bundle.render_formalization(rationale_text=None)


def build_mutation_design_bundle(
    *,
    parent: OrganismMeta,
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
    family_id: str | None = None,
) -> DesignPromptBundle:
    """Build the full two-step prompt bundle for a mutation child.

    When the family's prompt bundle does not ship rationalization prompts
    (``supports_two_step_mutation`` is False) the returned bundle has
    ``rationalization_*`` fields set to None and only the formalization
    prompts populated — the generator then routes the call as single-step.
    """

    parent_genetic_code = read_organism_genetic_code(parent)
    parent_lineage = read_organism_lineage(parent)
    # Ground-truth Python that the parent actually executes. Previously
    # only the implementation stage saw this; Step 1 (rationalization) and
    # Step 2 (formalization) operated on the prose ``parent_genetic_code``
    # alone, which let the LLM confidently "diagnose" mechanisms that
    # might not even exist in the parent's Python. Reading it here so
    # ``build_mutation_prompt_from_artifacts`` can inject it into all
    # stages.
    parent_implementation = read_organism_implementation(parent)

    return build_mutation_prompt_from_artifacts(
        parent_genetic_code=parent_genetic_code,
        parent_lineage=parent_lineage,
        prompts=prompts,
        novelty_feedback=novelty_feedback,
        compatibility_feedback=compatibility_feedback,
        family_id=family_id,
        parent_simple_score=parent.simple_score,
        parent_implementation=parent_implementation,
    )


def build_mutation_prompt_from_artifacts(
    *,
    parent_genetic_code: dict[str, Any],
    parent_lineage: list[dict[str, Any]],
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
    family_id: str | None = None,
    parent_simple_score: float | None = None,
    parent_implementation: str | None = None,
    # Legacy kwargs accepted for backward compatibility with manual_pipeline.py;
    # both are intentionally unused since pre-LLM gene sampling is disabled.
    inherited_genes: list[str] | None = None,
    removed_genes: list[str] | None = None,
) -> DesignPromptBundle:
    """Build mutation design prompts (Step 1 + Step 2 templates)."""

    parent_genetic_code_str = format_genetic_code(dict(parent_genetic_code))
    parent_lineage_str = format_lineage_summary(list(parent_lineage))
    parent_fitness_signal = format_parent_fitness_signal(
        primary_label="parent",
        primary_score=parent_simple_score,
        primary_lineage=list(parent_lineage),
    )
    # ``parent_implementation`` is the ground-truth Python the parent runs.
    # It's optional so callers that don't have a parent (seeds) or that
    # don't have the implementation loaded can still build the prompts;
    # the ``(unavailable)`` fallback keeps ``.format()`` from raising on a
    # missing placeholder while still being a readable signal to the LLM.
    parent_implementation_str = (
        format_implementation_code(parent_implementation)
        if parent_implementation is not None
        else "(unavailable)"
    )
    novelty_feedback_str = format_novelty_rejection_feedback(list(novelty_feedback or []))
    compatibility_block = ""
    if compatibility_feedback:
        compatibility_block = (
            "\n\n=== COMPATIBILITY REJECTION FEEDBACK ===\n"
            f"{format_compatibility_rejection_feedback(compatibility_feedback)}"
        )

    # Step 2 (formalization) — the {rationalization} placeholder stays literal
    # until the generator substitutes it for either the Step-1 output or the
    # single-call stub. Passing the placeholder back through .format() keeps
    # the brace untouched.
    formalization_system = compose_system_prompt(prompts.project_context, prompts.mutation_system)
    formalization_user_template = prompts.mutation_user.format(
        genome_schema=prompts.genome_schema,
        parent_genetic_code=parent_genetic_code_str,
        parent_lineage_summary=parent_lineage_str,
        parent_fitness_signal=parent_fitness_signal,
        parent_implementation=parent_implementation_str,
        novelty_rejection_feedback=novelty_feedback_str,
        rationalization=RATIONALIZATION_PLACEHOLDER,
        # Legacy placeholders kept for backward compat with optimization_survey
        # prompts that still reference the pre-bandit gene-sampling shape.
        # Production awtf2025 + circle_packing prompts have dropped them.
        inherited_gene_pool="(none)",
        removed_gene_pool="(none)",
    ) + compatibility_block

    # Step 1 (rationalization) — only rendered when the family ships the
    # prompts. The lineage regime hint feeds in here exclusively.
    rationalization_system: str | None = None
    rationalization_user: str | None = None
    lineage_regime_hint = summarize_recent_regime(list(parent_lineage), family=family_id)
    if prompts.supports_two_step_mutation:
        rationalization_system = compose_system_prompt(
            prompts.project_context,
            prompts.mutation_rationalization_system,
        )
        rationalization_user = prompts.mutation_rationalization_user.format(
            parent_genetic_code=parent_genetic_code_str,
            parent_lineage_summary=parent_lineage_str,
            parent_fitness_signal=parent_fitness_signal,
            parent_implementation=parent_implementation_str,
            lineage_regime_hint=lineage_regime_hint,
            novelty_rejection_feedback=novelty_feedback_str,
        )
        if compatibility_feedback:
            rationalization_user = (
                f"{rationalization_user}\n\n=== COMPATIBILITY REJECTION FEEDBACK ===\n"
                f"{format_compatibility_rejection_feedback(compatibility_feedback)}"
            )

    return DesignPromptBundle(
        formalization_system=formalization_system,
        formalization_user_template=formalization_user_template,
        rationalization_system=rationalization_system,
        rationalization_user=rationalization_user,
        lineage_regime_hint=lineage_regime_hint,
    )


class MutationOperator:
    """Mutation operator: hand the LLM the parent's full genome and let it design the child."""

    def __init__(self, q: float = 0.2, seed: int | None = None) -> None:
        # `q` and `seed` are retained for backward compatibility with config/tests but
        # the random gene-pruning phase is disabled — the LLM now designs the mutation
        # directly from the parent's full genome.
        self.q = q
        self.rng = random.Random(seed)

    def produce(
        self,
        parent: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
        pipeline_state_callback: Any = None,
    ) -> OrganismMeta:
        """Create a child organism via mutation."""

        inherited_genes: list[str] = []
        removed_genes: list[str] = []
        LOGGER.info("Mutate %s: full-genome mutation (no pre-LLM gene pruning)", parent.organism_id[:8])

        family_id = _detect_family_id(generator)
        initial_bundle = build_mutation_design_bundle(
            parent=parent,
            prompts=generator.prompt_bundle,
            family_id=family_id,
        )

        # Step 1 (rationalization) runs once per organism creation attempt.
        # The result is cached and reused across novelty / compatibility
        # retries: rejections almost always reflect formalization drift, not
        # a wrong strategic direction, so re-running Step 1 wastes a call
        # and risks the planner contradicting itself across iterations.
        rationale_text = _maybe_run_step1(
            generator=generator,
            bundle=initial_bundle,
            organism_id=organism_id,
            generation=generation,
            operator="mutation",
            org_dir=org_dir,
        )

        system_prompt, user_prompt = initial_bundle.render_formalization(rationale_text)
        inherited_genes_for_validators: list[str] = []
        removed_genes_for_validators: list[str] = []

        def _rebuild_step2(novelty_feedback=None, compatibility_feedback=None):
            retry_bundle = build_mutation_design_bundle(
                parent=parent,
                prompts=generator.prompt_bundle,
                novelty_feedback=novelty_feedback,
                compatibility_feedback=compatibility_feedback,
                family_id=family_id,
            )
            return retry_bundle.render_formalization(rationale_text)

        novelty_context = NoveltyCheckContext(
            operator="mutation",
            build_design_prompts=lambda feedback: _rebuild_step2(novelty_feedback=feedback),
            build_novelty_prompts=lambda candidate_design: build_mutation_novelty_prompt(
                inherited_genes=inherited_genes_for_validators,
                removed_genes=removed_genes_for_validators,
                parent=parent,
                candidate_design=candidate_design,
                prompts=generator.prompt_bundle,
                expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
            ),
        )
        compatibility_context = None
        if (
            getattr(generator.prompt_bundle, "compatibility_mutation_system", "")
            and getattr(generator.prompt_bundle, "compatibility_mutation_user", "")
        ):
            compatibility_context = CompatibilityValidationContext(
                check=CompatibilityCheckContext(operator_kind="mutation"),
                build_design_prompts=lambda novelty_feedback, compatibility_feedback: _rebuild_step2(
                    novelty_feedback=novelty_feedback,
                    compatibility_feedback=compatibility_feedback,
                ),
                build_compatibility_prompts=lambda candidate_design: build_mutation_compatibility_prompt(
                    inherited_genes=inherited_genes_for_validators,
                    removed_genes=removed_genes_for_validators,
                    parent=parent,
                    candidate_design=candidate_design,
                    prompts=generator.prompt_bundle,
                    expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
                ),
            )
        run_creation = getattr(generator, "run_creation_stages_with_retries", generator.run_creation_stages)
        creation_kwargs = {
            "design_system_prompt": system_prompt,
            "design_user_prompt": user_prompt,
            "org_dir": org_dir,
            "organism_id": organism_id,
            "generation": generation,
            "novelty_context": novelty_context,
        }
        if compatibility_context is not None:
            creation_kwargs["compatibility_context"] = compatibility_context
        if getattr(generator, "uses_section_patch_compilation", lambda: False)():
            creation_kwargs["implementation_base_parent"] = parent
        if pipeline_state_callback is not None:
            creation_kwargs["pipeline_state_callback"] = pipeline_state_callback
        creation = run_creation(**creation_kwargs)
        parent_lineage = read_organism_lineage(parent)
        return build_organism_from_response(
            parsed=creation.parsed_design,
            implementation_code=creation.implementation_code,
            organism_id=organism_id,
            island_id=parent.island_id,
            generation=generation,
            mother_id=parent.organism_id,
            father_id=None,
            operator="mutation",
            org_dir=org_dir,
            llm_route_id=creation.llm_route_id,
            llm_provider=creation.llm_provider,
            provider_model_id=creation.provider_model_id,
            llm_pipeline_id=(
                generator.pipeline_id_for_organism(organism_id) or ""
                if hasattr(generator, "pipeline_id_for_organism")
                else ""
            ),
            prompt_hash=creation.prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_lineage=parent_lineage,
            ancestor_ids=parent.ancestor_ids,
            cross_island=False,
            father_island_id=None,
            expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
        )
