# Runtime Utility Guide

This file applies to `optbench/utils/`.

## Purpose

This directory contains low-level helpers shared by the train loops and runtime:

- import/signature validation
- baseline loading
- objective tracking
- optimizer step-budget bookkeeping
- seed control
- stable file I/O
- timing
- numerical safety checks

## Important Invariants

- `import_utils.py` is intentionally strict and rejects outdated optimizer signatures.
- `optimizer_runtime.py` owns honest step accounting for `step_fn()` calls. Do not break `StepBudget`, activation capture, or named-parameter collection semantics.
- `objective_tracking.py` and `baselines.py` feed evolve-side scoring and normalization. Field names and meanings must stay consistent.
- Helpers should remain deterministic, side-effect-light, and easy to reuse from both runtime and evolve paths.

## Editing Rules

- Prefer pure functions or small stateful helpers with clear contracts.
- When changing validation logic, add or update tests rather than weakening callers.
- Keep error messages actionable; many tests assert on them.
