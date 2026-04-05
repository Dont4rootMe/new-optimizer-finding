# Repository Guide

This file applies to the whole repository unless a deeper `AGENTS.md` overrides it.

## What This Repo Contains

- `src/`: task-blind runtime, validation worker, and organism-first evolution engine
- `experiments/`: task-specific experiment packages and runtimes
- `conf/`: Hydra composition, experiment configs, and task-specific prompt assets
- `tests/`: contract, regression, and integration coverage

## Global Invariants

- The project is single-device by design. Do not introduce DDP, model parallelism, or hidden multi-GPU assumptions.
- `src` must stay blind to task-specific runtime details. It operates on organism folders, experiment lists, and report `score`s.
- Every experiment evaluator must implement `evaluate_organism(organism_dir, cfg) -> dict` and every returned report must include `score`.
- Canonical evolution is organism-first and island-aware. `run_evolution(...)` must remain wired to the current `EvolutionLoop`.
- Canonical organism artifacts are first-class data. Keep these filenames and meanings stable:
  - `implementation.py`
  - `genetic_code.md`
  - `lineage.json`
  - `organism.json`
  - `summary.json`
  - `llm_request.json`
  - `llm_response.json`
- Population resume state is stored in `population_state.json`.
- Optimization-specific contracts such as `build_optimizer(model, max_steps)` belong only under `experiments/optimization_survey/`.

## Editing Rules

- Keep config, code, and tests in sync. This repo relies heavily on strict contracts.
- Prefer extending existing shared helpers before adding one-off implementations.
- If you add a new experiment, update all of:
  - `conf/config.yaml`
  - `conf/experiments/<family>/<name>.yaml`
  - `experiments/<family>/<name>/`
  - the experiment YAML `_target_`
  - relevant tests
- If you change prompt placeholders, prompt file layout, or structured response sections, update the prompt builders and parser/contract tests together.

## Verification

- Full regression suite: `pytest -q`
- Common focused checks:
  - `pytest -q tests/test_hydra_compose.py`
  - `pytest -q tests/test_import_optimizer.py`
  - `pytest -q tests/test_prompt_bundle.py`
  - `pytest -q tests/test_organism_contract.py`
  - `pytest -q tests/test_run_evolution.py`
