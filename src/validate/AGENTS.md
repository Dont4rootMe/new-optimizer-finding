# Validation Worker Guide

This file applies to `src/validate/`.

## Purpose

- `run_one.py` is the subprocess entrypoint used by evolution-time evaluation jobs.
- It loads one config snapshot, resolves one experiment, runs one evaluator against one organism directory, and writes one result JSON.

## Rules

- Keep the CLI contract stable:
  - `--experiment`
  - `--organism_dir`
  - `--output_json`
  - `--seed`
  - `--device`
  - `--precision`
  - `--mode`
  - `--config_path`
  - `--override`
- Always write `output_json`, including failure cases.
- The written payload must be a dict with at least `score`.
- This worker should prepare and run one experiment only. Heavy orchestration belongs elsewhere.

## Verification

- `pytest -q tests/test_run_validation_overrides.py`
- `pytest -q tests/test_evolve_integration_fake.py`
