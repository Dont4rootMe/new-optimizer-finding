# Validation Worker Guide

This file applies to `src/validate/`.

## Purpose

- `run_one.py` is the subprocess entrypoint used by evolution-time evaluation jobs.
- It composes Hydra config, resolves one experiment, imports one optimizer, runs training, and writes one result JSON.

## Rules

- Keep the CLI contract stable:
  - `--experiment`
  - `--optimizer_path`
  - `--output_json`
  - `--seed`
  - `--device`
  - `--precision`
  - `--mode`
  - `--config_path`
  - `--override`
- Always write `output_json`, including failure cases.
- Keep the output payload shape compatible with `src/evolve/orchestrator.py` and downstream scoring.
- This worker should prepare and run one experiment only. Heavy orchestration belongs elsewhere.

## Verification

- `pytest -q tests/test_run_validation_overrides.py`
- `pytest -q tests/test_evolve_integration_fake.py`
