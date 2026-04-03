# Prompt Asset Guide

This file applies to `conf/prompts/` unless a deeper `AGENTS.md` overrides it.

## Layout

- `shared/`: shared system context prepended to canonical task-specific system prompts
- `seed/`: paired prompts for creating a brand-new organism
- `mutation/`: paired prompts for mutating an organism
- `crossover/`: paired prompts for crossbreeding two organisms
- `islands/`: plain-text research directions, one file per island
- `legacy_candidate/`: explicit legacy raw-code prompt pair

## Canonical Prompt Rules

- Canonical task folders use exactly two files:
  - `system.txt`
  - `user.txt`
- `src.evolve.prompt_utils.compose_system_prompt(...)` combines `shared/project_context.txt` with the task-specific `system.txt`.
- Placeholders in `user.txt` must stay aligned with the Python format calls in:
  - `src/evolve/operators.py`
  - `src/organisms/mutation.py`
  - `src/organisms/crossbreeding.py`
- Canonical responses are structured section payloads, not raw Python files.

## Contract Sensitivity

If you change prompt wording that affects structure, also review:

- `src/evolve/template_parser.py`
- `src/organisms/organism.py`
- `tests/test_prompt_bundle.py`
- `tests/test_optimizer_generator.py`
- `tests/test_organism_contract.py`

## Legacy Boundary

- `legacy_candidate/` is for the explicit legacy candidate-first path only.
- Canonical organism generation must never depend on `legacy_candidate/`.
