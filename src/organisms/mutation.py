"""Mutation operator for canonical organism genetic code."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.storage import utc_now_iso
from src.evolve.types import OrganismMeta
from src.organisms.novelty import (
    NoveltyCheckContext,
    build_mutation_novelty_prompt,
    format_novelty_rejection_feedback,
)
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_lineage_summary,
    read_organism_genetic_code,
    read_organism_lineage,
)

LOGGER = logging.getLogger(__name__)


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
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for mutation LLM call."""

    parent_genetic_code = read_organism_genetic_code(parent)
    parent_lineage = read_organism_lineage(parent)

    return build_mutation_prompt_from_artifacts(
        inherited_genes=inherited_genes,
        removed_genes=removed_genes,
        parent_genetic_code=parent_genetic_code,
        parent_lineage=parent_lineage,
        prompts=prompts,
        novelty_feedback=novelty_feedback,
    )


def build_mutation_prompt_from_artifacts(
    *,
    inherited_genes: list[str],
    removed_genes: list[str],
    parent_genetic_code: dict[str, Any],
    parent_lineage: list[dict[str, Any]],
    prompts: PromptBundle,
    novelty_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """Build mutation prompts from raw canonical artifacts."""

    system = compose_system_prompt(prompts.project_context, prompts.mutation_system)
    user = prompts.mutation_user.format(
        genome_schema=prompts.genome_schema,
        inherited_gene_pool="\n".join(f"- {gene}" for gene in inherited_genes) or "(none)",
        removed_gene_pool="\n".join(f"- {gene}" for gene in removed_genes) or "(none)",
        parent_genetic_code=format_genetic_code(dict(parent_genetic_code)),
        parent_lineage_summary=format_lineage_summary(list(parent_lineage)),
        novelty_rejection_feedback=format_novelty_rejection_feedback(list(novelty_feedback or [])),
    )
    return system, user


class MutationOperator:
    """Two-phase mutation: probabilistic gene removal + LLM rewrite."""

    def __init__(self, q: float = 0.2, seed: int | None = None) -> None:
        self.q = q
        self.rng = random.Random(seed)

    def produce(
        self,
        parent: OrganismMeta,
        organism_id: str,
        generation: int,
        org_dir: Path,
        generator: Any,
    ) -> OrganismMeta:
        """Create a child organism via mutation."""

        parent_genetic_code = read_organism_genetic_code(parent)
        parent_genes = list(parent_genetic_code.get("core_genes", []))
        inherited_genes, removed_genes = prune_gene_pool(parent_genes, self.q, self.rng)
        LOGGER.info(
            "Mutate %s: kept %d/%d genes, removed: %s",
            parent.organism_id[:8],
            len(inherited_genes),
            len(parent_genes),
            removed_genes or "(none)",
        )

        system_prompt, user_prompt = _build_mutate_prompt(
            inherited_genes,
            removed_genes,
            parent,
            generator.prompt_bundle,
        )
        novelty_context = NoveltyCheckContext(
            operator="mutation",
            build_design_prompts=lambda feedback: _build_mutate_prompt(
                inherited_genes,
                removed_genes,
                parent,
                generator.prompt_bundle,
                novelty_feedback=feedback,
            ),
            build_novelty_prompts=lambda candidate_design: build_mutation_novelty_prompt(
                inherited_genes=inherited_genes,
                removed_genes=removed_genes,
                parent=parent,
                candidate_design=candidate_design,
                prompts=generator.prompt_bundle,
                expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
            ),
        )
        run_creation = getattr(generator, "run_creation_stages_with_retries", generator.run_creation_stages)
        creation = run_creation(
            design_system_prompt=system_prompt,
            design_user_prompt=user_prompt,
            org_dir=org_dir,
            organism_id=organism_id,
            generation=generation,
            novelty_context=novelty_context,
        )
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
            prompt_hash=creation.prompt_hash,
            seed=generator.seed,
            timestamp=utc_now_iso(),
            parent_lineage=parent_lineage,
            ancestor_ids=parent.ancestor_ids,
            cross_island=False,
            father_island_id=None,
            expected_core_gene_sections=getattr(generator, "expected_core_gene_sections", None),
        )
