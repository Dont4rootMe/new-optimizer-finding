# Experiment Config Guide

This file applies to `conf/experiments/`.

## Required Shape

Each experiment config should define the fields that the runtime expects:

- `enabled`
- `name`
- `data`
- `model`
- `train`
- `compute`
- `safety`
- `primary_metric`
- `target`
- `run_validation`
- `normalization`
- `optimizer_defaults`

Some experiments use empty or minimal blocks, but the runtime contract should still remain consistent.

## Important Constraints

- `name` must match the experiment key used in Hydra and the registry.
- `compute.device`, `precision`, `smoke_steps`, and `max_steps` must reflect real single-device limits.
- `optimizer_defaults.params` must be a mapping.
- `optimizer_defaults.scheduler.type` must stay compatible with the builtin scheduler logic in `optbench/runner.py`.
- `primary_metric` and `target` must agree on direction semantics.
- Optional dependency experiments should still compose cleanly even when their Python extras are absent.

## Change Discipline

- Prefer updating existing config keys over inventing parallel variants.
- If a new config field is consumed in code, add tests that exercise both default composition and runtime override behavior.
- If you add a new experiment config, update `conf/config.yaml`, `experiments/`, and `optbench/registry.py` in the same change.
