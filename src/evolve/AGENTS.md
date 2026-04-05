# Evolution Engine Guide

This file applies to `src/evolve/`.

## Scope

This directory implements the canonical organism-first evolution engine.

Key modules:

- `run.py`: public entrypoint for canonical runs
- `evolution_loop.py`: generation-to-generation population lifecycle
- `generator.py`: seed organism generation from structured prompts
- `orchestrator.py`: evaluation seam that runs experiment jobs and aggregates phase scores
- `prompt_utils.py`: prompt-bundle loading from configured prompt assets
- `template_parser.py`: generic structured LLM response parsing
- `storage.py`: canonical filesystem layout and strict resume helpers
- `selection.py`: island-local survival and parent sampling helpers

## Non-Negotiable Invariants

- `run_evolution(...)` must always use `EvolutionLoop`.
- Canonical evolution reads the canonical `evolver.*` schema. Do not add hidden fallback reads from older config trees.
- `src/evolve` must stay task-blind. It operates on configured prompt paths, organism folders, and experiment report scores.
- Canonical prompt loading comes from `cfg.evolver.prompts.*`, not embedded prompt strings.
- Canonical organism layout is strict. Missing `population_state.json`, `genetic_code.md`, or `lineage.json` is a real error during canonical resume.
- Great Filter selection is island-local and phase-specific. Preserve the meaning of:
  - `simple_score`
  - `hard_score`

## Editing Rules

- If you change prompt file names or placeholders, update `prompt_utils.py`, operator builders, and prompt tests together.
- If you change organism storage layout or population-state semantics, update resume and contract tests together.
- If you change scoring, allocation, or phase selection semantics, update both integration tests and targeted unit tests.

## Verification

- `pytest -q tests/test_run_evolution.py`
- `pytest -q tests/test_evolution_loop_semantics.py`
- `pytest -q tests/test_evolution_resume.py`
- `pytest -q tests/test_optimizer_generator.py`
- `pytest -q tests/test_selection.py`
