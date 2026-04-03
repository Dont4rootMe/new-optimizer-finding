# External Optimizer Example Guide

This file applies to `optimizer_guesses/`.

## Purpose

- This directory contains example external optimizers used by docs, smoke checks, and contract tests.
- Files here are not special-cased by the runtime; they are imported through the same strict path-based loader as user-provided optimizers.

## Rules

- Example optimizers must satisfy the strict runtime contract:
  - `build_optimizer(model, max_steps)`
  - controller `step(weights, grads, activations, step_fn)`
  - controller `zero_grad(set_to_none=True)`
- Keep examples small, readable, and safe.
- Avoid file I/O, subprocesses, network access, and hidden side effects.
- Treat `step_fn()` as an expensive extra step that consumes budget.

## Verification

- `pytest -q tests/test_import_optimizer.py`
- `python -m optbench.main mode=run optimizer_path=optimizer_guesses/examples/sgd_baseline.py`
