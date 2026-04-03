# Config Guide

This file applies to `conf/` and its subdirectories unless a deeper `AGENTS.md` overrides it.

## Purpose

- `config.yaml` is the root Hydra composition file.
- `evolver/default.yaml` is the canonical schema for organism-first evolution.
- `experiments/*.yaml` are per-experiment runtime configs.
- `prompts/` stores prompt assets consumed by the canonical generation paths.

## Rules

- Keep canonical evolution settings under `evolver.*`. Do not introduce parallel schema families for the same concept.
- Keep prompt paths repo-relative and rooted under `conf/prompts/`.
- Keep experiment config filenames aligned with experiment names because Hydra defaults and registry names rely on that convention.
- Keep `smoke_steps` materially smaller than full `max_steps`.
- Keep compute budgets honest for a single-device environment.
- Builtin optimizer defaults must stay compatible with `optbench.runner.BuiltinOptimizerController`, which supports only `sgd`, `adam`, and `adamw`.

## When Adding Or Renaming Experiments

- Add the Hydra default in `conf/config.yaml`.
- Add `conf/experiments/<name>.yaml`.
- Register the experiment class in `optbench/registry.py`.
- Add or update tests that compose the config and exercise run-time overrides.

## Verification

- `pytest -q tests/test_hydra_compose.py`
- `pytest -q tests/test_run_validation_overrides.py`
