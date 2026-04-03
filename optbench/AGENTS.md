# Runtime Guide

This file applies to `optbench/` unless a deeper `AGENTS.md` overrides it.

## Purpose

- `main.py`: Hydra entrypoint for `run`, `smoke`, and `stats`
- `runner.py`: central orchestration and result persistence
- `registry.py`: source of truth for registered experiments
- `optimizer_api.py`: runtime optimizer-controller contract
- `schemas.py`: persisted run-result schema and validation

## Core Invariants

- `optbench` is the validation runtime, not the evolution engine.
- Keep `main.py` thin. Business logic belongs in `runner.py` or lower-level helpers.
- The optimizer contract is strict:
  - `build_optimizer(model, max_steps)`
  - controller with `step(weights, grads, activations, step_fn)`
  - `zero_grad(set_to_none=True)`
- Builtin optimizer support is intentionally narrow (`sgd`, `adam`, `adamw`).
- Optional dependency failures should surface as `OptionalDependencyError`, not random import errors leaking upward.

## Editing Rules

- If you change result payload shape, update `RunResult`, its validator, and any consumers in `src/validate/` or `src/evolve/`.
- If you add an experiment, register it in `registry.py` and ensure Hydra composes it.
- Keep run IDs, timestamps, and persisted artifact paths stable unless migration work explicitly requires a format change.

## Verification

- `pytest -q tests/test_import_optimizer.py`
- `pytest -q tests/test_result_schema.py`
- `pytest -q tests/test_run_validation_overrides.py`
