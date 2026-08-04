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
- A successful report must carry a finite numeric score; use the shared report
  contract rather than allowing NaN/Infinity into selection.
- Keep `experiments/catalog.py` exhaustive. Every added, removed, renamed, or
  re-phased task requires a matching catalog entry and knowledge-base update.
- Optional dependency experiments should raise the family-local
  `OptionalDependencyError` with a clear install hint.
- If you add a new experiment package, also update:
  - `conf/experiments/<family>/<name>.yaml`
  - the relevant top-level preset such as `conf/config_optimization_survey.yaml` or `conf/config_circle_packing_shinka.yaml`
  - the experiment YAML `_target_`
  - `experiments/catalog.py`
  - `REPOSITORY_KNOWLEDGE_BASE.md`
  - tests

## Verification

- `pytest -q tests/test_hydra_compose.py`
- any experiment-specific or integration tests affected by the change
