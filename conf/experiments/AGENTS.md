# Experiment Config Guide

This file applies to `conf/experiments/`.

## Required Shape

Each experiment config should define the fields that its evaluator expects. In practice the common shape includes:

- `_target_`
- `enabled`
- `name`
- `compute`
- optional task-specific blocks such as `data`, `model`, `train`, `normalization`, `baseline`, `optimizer_defaults`
- optional task-specific runtime blocks such as `safety`

## Important Constraints

- `name` must match the experiment key used in Hydra composition.
- `compute.device`, `precision`, `smoke_steps`, and `max_steps` must reflect real single-device limits.
- Optimization-survey configs that use builtin optimizers must keep `optimizer_defaults.params` as a mapping.
- Optional dependency experiments should still compose cleanly even when their Python extras are absent.
- Task-specific knobs belong here, not in top-level presets under `conf/`.

## Change Discipline

- Prefer updating existing config keys over inventing parallel variants.
- If a new config field is consumed in code, add tests that exercise both default composition and runtime override behavior.
- If you add a new experiment config, update the relevant top-level preset (for optimizer-search this is `conf/config_optimization_survey.yaml`), the matching experiment package, and the YAML `_target_` in the same change.
