# Experiment Package Guide

This file applies to `experiments/` unless a deeper `AGENTS.md` overrides it.

## Purpose

Each experiment package implements task-specific behavior. The evaluator referenced by the experiment YAML must expose:

- `evaluate_organism(organism_dir, cfg) -> dict`

That evaluator may delegate to package-local helpers such as:

- `build_datamodule(cfg)`
- `build_model(cfg)`
- `train(cfg, model, datamodule, implementation_factory)`
- `evaluate(cfg, model, datamodule)`

## Rules

- Keep task-specific runtime helpers inside the owning experiment family, not in `src/`.
- Reuse `experiments/optimization_survey/_shared/` train loops where possible instead of cloning them.
- Returned reports must include `score`. Everything else is opaque to `src`.
- Optional dependency experiments should raise [`OptionalDependencyError`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/optimization_survey/_runtime/errors.py) with a clear install hint.
- If you add a new experiment package, also update:
  - `conf/experiments/<family>/<name>.yaml`
  - `conf/config.yaml`
  - the experiment YAML `_target_`
  - tests

## Verification

- `pytest -q tests/test_hydra_compose.py`
- any experiment-specific or integration tests affected by the change
