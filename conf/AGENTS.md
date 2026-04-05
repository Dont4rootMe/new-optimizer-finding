# Config Guide

This file applies to `conf/` and its subdirectories unless a deeper `AGENTS.md` overrides it.

## Purpose

- `config.yaml` is the root Hydra composition file.
- `evolver/default.yaml` is the canonical schema for organism-first evolution.
- `experiments/<family>/*.yaml` are per-experiment runtime configs.
- task-specific prompts live next to their experiment family configs.

## Rules

- Keep canonical evolution settings under `evolver.*`. Do not introduce parallel schema families for the same concept.
- Runtime-facing config shape is flat at `cfg.experiments.<name>`, even if files physically live under `conf/experiments/<family>/`.
- Keep prompt paths repo-relative.
- Each experiment config must be Hydra-instantiable via `_target_`.
- Keep `smoke_steps` materially smaller than full `max_steps`.
- Keep compute budgets honest for a single-device environment.
- Optimization-survey builtin optimizer defaults must stay compatible with [`experiments/optimization_survey/_runtime/runner.py`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/optimization_survey/_runtime/runner.py).

## When Adding Or Renaming Experiments

- Add the Hydra default in `conf/config.yaml`.
- Add `conf/experiments/<family>/<name>.yaml`.
- Point that config at a Hydra `_target_` evaluator.
- Add or update tests that compose the config and exercise run-time overrides.

## Verification

- `pytest -q tests/test_hydra_compose.py`
- `pytest -q tests/test_run_validation_overrides.py`
