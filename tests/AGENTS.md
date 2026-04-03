# Test Suite Guide

This file applies to `tests/`.

## Purpose

The suite protects strict contracts across runtime, prompt loading, organism artifacts, selection logic, and canonical-vs-legacy boundaries.

## What The Tests Intentionally Enforce

- Hydra composition and canonical config shape
- strict optimizer import signatures
- prompt bundle layout and placeholder wiring
- canonical organism artifact structure and lineage semantics
- canonical evolution entrypoints and resume behavior
- explicit separation between canonical and legacy paths

## Editing Rules

- When changing behavior, prefer updating or adding the narrowest relevant test instead of weakening broad invariants.
- Do not update tests merely to bless accidental regressions in contracts or artifact layout.
- If you change:
  - config shape: update Hydra tests
  - optimizer contract: update import/schema/template tests
  - prompt layout: update prompt bundle and generator tests
  - organism artifacts: update organism contract and evolution resume tests
  - canonical-vs-legacy routing: update run-evolution and legacy integration tests

## Useful Test Targets

- `tests/test_hydra_compose.py`
- `tests/test_import_optimizer.py`
- `tests/test_prompt_bundle.py`
- `tests/test_optimizer_generator.py`
- `tests/test_organism_contract.py`
- `tests/test_run_evolution.py`
- `tests/test_evolution_loop_semantics.py`
