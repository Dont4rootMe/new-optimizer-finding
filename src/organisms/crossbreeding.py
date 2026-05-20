"""Crossbreeding operator for canonical organism genetic code."""

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
    build_crossover_compatibility_prompt,
    format_compatibility_rejection_feedback,
)
from src.organisms.lineage_regime import summarize_recent_regime
from src.organisms.mutation import _detect_family_id, _maybe_run_step1
from src.organisms.novelty import (
    NoveltyCheckContext,
    build_crossover_novelty_prompt,
    format_novelty_rejection_feedback,
)
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_implementation_code,
    format_inspiration_organisms,
    format_lineage_summary,
    format_parent_fitness_signal,
    read_organism_genetic_code,
    read_organism_implementation,
    read_organism_lineage,
)

LOGGER = logging.getLogger(__name__)


def merge_gene_pools(
    mother_genes: list[str],
    father_genes: list[str],
    inherit_probability: float = 0.7,
    rng: random.Random | None = None,
) -> list[str]:
    """Recombine maternal and paternal gene pools with maternal bias."""

    if rng is None:
        rng = random.Random()

    seen_lower: set[str] = set()
    child: list[str] = []

    for trait in mother_genes:
        if rng.random() < inherit_probability:
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    for trait in father_genes:
        if rng.random() < (1.0 - inherit_probability):
            key = trait.strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                child.append(trait.strip())

    if not child:
        fallback = mother_genes if mother_genes else father_genes
        if fallback:
            child.append(rng.choice(fallback).strip())

    return child


def _build_crossbreed_prompt(
    inherited_genes: list[str],
    mother: OrganismMeta,
    father: OrganismMeta,
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for crossover LLM call (legacy path).

    Kept for callers that still want flat strings — internally derives the
    formalization prompts from :func:`build_crossover_design_bundle` and
    substitutes the rationalization placeholder with the single-call stub.
    """

    bundle = build_crossover_design_bundle(
        mother=mother,
        father=father,
        prompts=prompts,
        novelty_feedback=novelty_feedback,
        compatibility_feedback=compatibility_feedback,
    )
    return bundle.render_formalization(rationale_text=None)


def build_crossover_design_bundle(
    *,
    mother: OrganismMeta,
    father: OrganismMeta,
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
    family_id: str | None = None,
    inspirations: list[OrganismMeta] | None = None,
) -> DesignPromptBundle:
    """Build the full two-step prompt bundle for a crossover child."""

    return build_crossover_prompt_from_artifacts(
        mother_genetic_code=read_organism_genetic_code(mother),
        mother_lineage=read_organism_lineage(mother),
        father_genetic_code=read_organism_genetic_code(father),
        father_lineage=read_organism_lineage(father),
        prompts=prompts,
        novelty_feedback=novelty_feedback,
        compatibility_feedback=compatibility_feedback,
        family_id=family_id,
        mother_simple_score=mother.simple_score,
        father_simple_score=father.simple_score,
        # Ground-truth Python both parents actually execute. Previously
        # only the implementation stage saw this; Step 1 (rationalization)
        # and Step 2 (formalization) had to reason about the parents from
        # the prose genetic_code alone, which let the LLM diagnose
        # mechanisms that may not even exist in the parents' Python.
        mother_implementation=read_organism_implementation(mother),
        father_implementation=read_organism_implementation(father),
        inspirations=inspirations,
    )


def build_crossover_prompt_from_artifacts(
    *,
    mother_genetic_code: dict[str, Any],
    mother_lineage: list[dict[str, Any]],
    father_genetic_code: dict[str, Any],
    father_lineage: list[dict[str, Any]],
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
    compatibility_feedback: list[CompatibilityJudgment] | None = None,
    family_id: str | None = None,
    mother_simple_score: float | None = None,
    father_simple_score: float | None = None,
    mother_implementation: str | None = None,
    father_implementation: str | None = None,
    inspirations: list[OrganismMeta] | None = None,
    # Legacy kwarg accepted for backward compatibility with manual_pipeline.py;
    # intentionally unused since pre-LLM gene merging is disabled.
    inherited_genes: list[str] | None = None,
) -> DesignPromptBundle:
    """Build crossover design prompts (Step 1 + Step 2 templates)."""

    mother_code_str = format_genetic_code(dict(mother_genetic_code))
    mother_lineage_str = format_lineage_summary(list(mother_lineage))
    father_code_str = format_genetic_code(dict(father_genetic_code))
    father_lineage_str = format_lineage_summary(list(father_lineage))
    # Optional ground-truth implementations for each parent. Same shape as
    # the mutation builder: ``(unavailable)`` fallback keeps ``.format()``
    # happy on legacy callers that don't have the implementations loaded.
    mother_implementation_str = (
        format_implementation_code(mother_implementation)
        if mother_implementation is not None
        else "(unavailable)"
    )
    father_implementation_str = (
        format_implementation_code(father_implementation)
        if father_implementation is not None
        else "(unavailable)"
    )
    inspirations_str = format_inspiration_organisms(inspirations)
    parent_fitness_signal = format_parent_fitness_signal(
        primary_label="mother",
        primary_score=mother_simple_score,
        primary_lineage=list(mother_lineage),
        secondary_label="father",
        secondary_score=father_simple_score,
        secondary_lineage=list(father_lineage),
    )
    novelty_feedback_str = format_novelty_rejection_feedback(list(novelty_feedback or []))
    compatibility_block = ""
    if compatibility_feedback:
        compatibility_block = (
            "\n\n=== COMPATIBILITY REJECTION FEEDBACK ===\n"
            f"{format_compatibility_rejection_feedback(compatibility_feedback)}"
        )

    formalization_system = compose_system_prompt(prompts.project_context, prompts.crossover_system)
    formalization_user_template = prompts.crossover_user.format(
        genome_schema=prompts.genome_schema,
        mother_genetic_code=mother_code_str,
        mother_lineage_summary=mother_lineage_str,
        father_genetic_code=father_code_str,
        father_lineage_summary=father_lineage_str,
        parent_fitness_signal=parent_fitness_signal,
        mother_implementation=mother_implementation_str,
        father_implementation=father_implementation_str,
        inspirations=inspirations_str,
        novelty_rejection_feedback=novelty_feedback_str,
        rationalization=RATIONALIZATION_PLACEHOLDER,
        # Legacy placeholder for optimization_survey prompts that still
        # reference the pre-bandit gene-sampling shape.
        inherited_gene_pool="(none)",
    ) + compatibility_block

    rationalization_system: str | None = None
    rationalization_user: str | None = None
    # Regime hint is computed from the primary parent's (mother's) lineage —
    # the primary parent dominates the recombination by design.
    lineage_regime_hint = summarize_recent_regime(list(mother_lineage), family=family_id)
    if prompts.supports_two_step_crossover:
        rationalization_system = compose_system_prompt(
            prompts.project_context,
            prompts.crossover_rationalization_system,
        )
        rationalization_user = prompts.crossover_rationalization_user.format(
            mother_genetic_code=mother_code_str,
            mother_lineage_summary=mother_lineage_str,
            father_genetic_code=father_code_str,
            father_lineage_summary=father_lineage_str,
            parent_fitness_signal=parent_fitness_signal,
            mother_implementation=mother_implementation_str,
            father_implementation=father_implementation_str,
            inspirations=inspirations_str,
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


class CrossbreedingOperator:
    """Crossover operator: hand the LLM both parent genomes and let it design the child."""

    def __init__(self, p: float = 0.7, seed: int | None = None) -> None:
        # `p` and `seed` are retained for backward compatibility with config/tests but
        # the random gene-merging phase is disabled — the LLM now designs the crossover
        # directly from both parents' full genomes.
        self.p = p
        self.rng = random.Random(seed)

    def produce(
        self,
        mother: OrganismMeta,
        father: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
        pipeline_state_callback: Any = None,
        inspirations: list[OrganismMeta] | None = None,
    ) -> OrganismMeta:
        """Create a child organism via crossbreeding.

        ``inspirations`` — see :meth:`MutationOperator.produce`. Top-K
        archive sample from the mother's island, formatted into Step 1
        / Step 2 prompts for FunSearch-style multi-program reference.
        """

        child_dna: list[str] = []
        LOGGER.info(
            "Crossbreed %s x %s: full-genome crossover (no pre-LLM gene merging)",
            mother.organism_id[:8],
            father.organism_id[:8],
        )

        family_id = _detect_family_id(generator)
        initial_bundle = build_crossover_design_bundle(
            mother=mother,
            father=father,
            prompts=generator.prompt_bundle,
            family_id=family_id,
            inspirations=inspirations,
        )

        rationale_text = _maybe_run_step1(
            generator=generator,
            bundle=initial_bundle,
            organism_id=organism_id,
            generation=generation,
            operator="crossover",
            org_dir=org_dir,
        )

        system_prompt, user_prompt = initial_bundle.render_formalization(rationale_text)

        def _rebuild_step2(novelty_feedback=None, compatibility_feedback=None):
            retry_bundle = build_crossover_design_bundle(
                mother=mother,
                father=father,
                prompts=generator.prompt_bundle,
                novelty_feedback=novelty_feedback,
                compatibility_feedback=compatibility_feedback,
                inspirations=inspirations,
                family_id=family_id,
            )
            return retry_bundle.render_formalization(rationale_text)

        novelty_context = NoveltyCheckContext(
            operator="crossover",
            build_design_prompts=lambda feedback: _rebuild_step2(novelty_feedback=feedback),
            build_novelty_prompts=lambda candidate_design: build_crossover_novelty_prompt(
                inherited_genes=child_dna,
                mother=mother,
                father=father,
                candidate_design=candidate_design,
                prompts=generator.prompt_bundle,
                expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
            ),
        )
        compatibility_context = None
        if (
            getattr(generator.prompt_bundle, "compatibility_crossover_system", "")
            and getattr(generator.prompt_bundle, "compatibility_crossover_user", "")
        ):
            compatibility_context = CompatibilityValidationContext(
                check=CompatibilityCheckContext(operator_kind="crossover"),
                build_design_prompts=lambda novelty_feedback, compatibility_feedback: _rebuild_step2(
                    novelty_feedback=novelty_feedback,
                    compatibility_feedback=compatibility_feedback,
                ),
                build_compatibility_prompts=lambda candidate_design: build_crossover_compatibility_prompt(
                    inherited_genes=child_dna,
                    mother=mother,
                    father=father,
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
        # Always pass the mother as the implementation base (was previously
        # gated on ``uses_section_patch_compilation``). See the matching
        # change in ``src/organisms/mutation.py`` for the rationale: the
        # parent's working Python being in scope at the implementation
        # stage lets the LLM reuse helpers instead of re-synthesizing
        # from the genetic_code prose.
        creation_kwargs["implementation_base_parent"] = mother
        if pipeline_state_callback is not None:
            creation_kwargs["pipeline_state_callback"] = pipeline_state_callback
        creation = run_creation(**creation_kwargs)
        mother_lineage = read_organism_lineage(mother)
        ancestor_ids = list(mother.ancestor_ids)
        for parent_id in (mother.organism_id, father.organism_id, *father.ancestor_ids):
            if parent_id and parent_id not in ancestor_ids:
                ancestor_ids.append(parent_id)
        return build_organism_from_response(
            parsed=creation.parsed_design,
            implementation_code=creation.implementation_code,
            organism_id=organism_id,
            island_id=mother.island_id,
            generation=generation,
            mother_id=mother.organism_id,
            father_id=father.organism_id,
            operator="crossover",
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
            parent_lineage=mother_lineage,
            ancestor_ids=ancestor_ids,
            cross_island=mother.island_id != father.island_id,
            father_island_id=father.island_id,
            expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
        )
