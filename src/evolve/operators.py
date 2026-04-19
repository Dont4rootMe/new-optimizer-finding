"""Seed prompt abstraction for canonical organism generation."""

from __future__ import annotations

from src.evolve.prompt_utils import PromptBundle, compose_system_prompt
from src.evolve.types import Island


class SeedOperator:
    """Create a new organism from scratch inside one island."""

    def __init__(self, island: Island) -> None:
        self.island = island

    def build_prompts(
        self,
        prompts: PromptBundle,
        schema_provider=None,
    ) -> tuple[str, str]:
        system = compose_system_prompt(prompts.project_context, prompts.seed_system)
        context_builder = getattr(schema_provider, "build_seed_prompt_context", None)
        typed_prompt_context = context_builder() if callable(context_builder) else ""
        user = prompts.seed_user.format(
            island_id=self.island.island_id,
            island_name=self.island.name,
            island_description=self.island.description_text,
            typed_prompt_context=typed_prompt_context,
        )
        return system, user
