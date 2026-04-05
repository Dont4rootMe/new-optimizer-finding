# External Optimizer Example Guide

This file applies to `optimizer_guesses/`.

## Purpose

- This directory contains example optimizer implementations for the `optimization_survey` task family.
- Files here are not special-cased by `src`; they are loaded only by optimization-specific runtime code under `experiments/optimization_survey/_runtime/`.

## Rules

- Example optimizers must satisfy the optimization-survey runtime contract:
  - `build_optimizer(model, max_steps)`
  - controller `step(weights, grads, activations, step_fn)`
  - controller `zero_grad(set_to_none=True)`
- Keep examples small, readable, and safe.
- Avoid file I/O, subprocesses, network access, and hidden side effects.
- Treat `step_fn()` as an expensive extra step that consumes budget.

## Verification

- `pytest -q tests/test_import_optimizer.py`
