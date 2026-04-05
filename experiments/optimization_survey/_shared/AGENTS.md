# Shared Train Loop Guide

This file applies to `experiments/optimization_survey/_shared/`.

## Purpose

- `train_supervised.py`: generic supervised training loop for classification and regression experiments
- `train_paramonly.py`: generic loop for parameter-only objective experiments
- `metrics.py`: shared primary-metric and target checks

## Important Invariants

- Preserve honest `step_fn()` accounting through `StepBudget`.
- Preserve activation capture and cleanup through `ActivationRecorder`.
- Preserve baseline-relative train-loss tracking through `TrainObjectiveTracker`.
- Evaluation helpers should return metric dictionaries only; orchestration and persistence happen higher up.

## Editing Rules

- Prefer adding parameters or small hooks over forking entire loops for one experiment.
- Keep shared loops generic and config-driven.
- Avoid experiment-specific branches here unless multiple experiments genuinely need them.
- Any change to result payload keys must stay aligned with [`src/validate/run_one.py`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/src/validate/run_one.py), [`src/validate/runner.py`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/src/validate/runner.py), and evolve-side scoring.

## Verification

- `pytest -q tests/test_baselines.py`
- `pytest -q tests/test_metrics_normalization.py`
- relevant integration tests for experiments that use the shared loops
