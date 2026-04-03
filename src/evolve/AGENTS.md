# Evolution Engine Guide

This file applies to `src/evolve/`.

## Scope

This directory implements the canonical organism-first evolution engine plus a quarantined legacy candidate-first path.

Key modules:

- `run.py`: public entrypoints for canonical and explicit legacy runs
- `evolution_loop.py`: generation-to-generation population lifecycle
- `generator.py`: seed organism generation from structured prompts
- `orchestrator.py`: evaluation seam that runs experiment jobs and aggregates phase scores
- `prompt_utils.py`: prompt-bundle loading from `conf/prompts/`
- `template_parser.py`: optimizer template rendering and contract validation
- `storage.py`: canonical filesystem layout and strict resume helpers
- `selection.py`: island-local survival and parent sampling helpers
- `legacy_*`: explicit legacy surfaces only

## Non-Negotiable Invariants

- `run_evolution(...)` must always use `EvolutionLoop`; legacy execution requires explicit `run_legacy_single_generation(...)`.
- Canonical evolution reads the canonical `evolver.*` schema. Do not add hidden fallback reads from older config trees.
- Canonical prompt loading comes from `conf/prompts/**`, not embedded prompt strings.
- Canonical organism layout is strict. Missing `population_manifest.json`, `genetic_code.md`, or `lineage.json` is a real error during canonical resume.
- Great Filter selection is island-local and phase-specific. Preserve the meaning of:
  - `simple_reward`
  - `hard_reward`
  - `selection_reward`
- Legacy compatibility may exist in explicit helpers, but canonical code should not silently downgrade to legacy behavior.

## Editing Rules

- If you change prompt file names or placeholders, update `prompt_utils.py`, operator builders, and prompt tests together.
- If you change organism storage layout or manifest semantics, update resume and contract tests together.
- If you change scoring, allocation, or phase selection semantics, update both integration tests and targeted unit tests.
- Prefer adding new canonical functionality before expanding legacy modules.

## Verification

- `pytest -q tests/test_run_evolution.py`
- `pytest -q tests/test_evolution_loop_semantics.py`
- `pytest -q tests/test_evolution_resume.py`
- `pytest -q tests/test_optimizer_generator.py`
- `pytest -q tests/test_selection.py`
