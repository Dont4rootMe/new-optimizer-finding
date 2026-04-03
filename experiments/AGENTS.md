# Experiment Package Guide

This file applies to `experiments/` unless a deeper `AGENTS.md` overrides it.

## Purpose

Each experiment package implements the protocol consumed by `optbench.registry`:

- `build_datamodule(cfg) -> dict`
- `build_model(cfg) -> torch.nn.Module`
- `train(cfg, model, datamodule, optimizer_factory) -> dict`
- `evaluate(cfg, model, datamodule) -> dict[str, float]`

The package `__init__.py` typically exposes a `<Name>Experiment` class that delegates to those functions.

## Common Layout

Most experiments use some subset of:

- `data.py`
- `model.py`
- `train.py`
- `metrics.py`
- `__init__.py`

Parameter-only experiments may omit data loaders and return an empty datamodule from `__init__.py`.

## Rules

- Reuse `experiments/_shared/` train loops where possible instead of cloning them.
- Keep train-loop return payloads compatible with the runtime and evolve-side consumers:
  - `final_metrics`
  - `best_metrics`
  - `objective_*`
  - `steps`
  - `wall_time_sec`
  - `samples_or_tokens_seen`
  - `status`
- Optional dependency experiments should raise `optbench.schemas.OptionalDependencyError` with a clear extra name and install hint.
- If you add a new experiment package, also update:
  - `conf/experiments/<name>.yaml`
  - `conf/config.yaml`
  - `optbench/registry.py`
  - tests

## Verification

- `pytest -q tests/test_hydra_compose.py`
- any experiment-specific or integration tests affected by the change
