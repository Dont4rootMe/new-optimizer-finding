"""Mutation operator for canonical organism genetic code."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.storage import utc_now_iso
from src.evolve.types import OrganismMeta
from src.organisms.organism import (
    build_organism_from_response,
    format_genetic_code,
    format_implementation_code,
    format_lineage_summary,
    read_organism_genetic_code,
    read_organism_implementation,
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
) -> tuple[str, str]:
    """Build `(system_prompt, user_prompt)` for mutation LLM call."""

    parent_genetic_code = read_organism_genetic_code(parent)
    parent_lineage = read_organism_lineage(parent)
    parent_implementation = read_organism_implementation(parent)

    system = compose_system_prompt(prompts.project_context, prompts.mutation_system)
    user = prompts.mutation_user.format(
        inherited_gene_pool="\n".join(f"- {gene}" for gene in inherited_genes) or "(none)",
        removed_gene_pool="\n".join(f"- {gene}" for gene in removed_genes) or "(none)",
        parent_genetic_code=format_genetic_code(parent_genetic_code),
        parent_lineage_summary=format_lineage_summary(parent_lineage),
        parent_implementation_code=format_implementation_code(parent_implementation),
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
        creation = generator.run_creation_stages(
            design_system_prompt=system_prompt,
            design_user_prompt=user_prompt,
            org_dir=org_dir,
            organism_id=organism_id,
            generation=generation,
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
        )
