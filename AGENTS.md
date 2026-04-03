# Repository Guide

This file applies to the whole repository unless a deeper `AGENTS.md` overrides it.

## What This Repo Contains

- `optbench/`: validation and benchmarking runtime for `run`, `smoke`, and `stats`.
- `src/evolve/` + `src/organisms/`: canonical organism-first optimizer evolution pipeline.
- `experiments/`: concrete ML tasks that share a common experiment protocol.
- `conf/`: Hydra composition, experiment configs, and prompt assets.
- `tests/`: contract, regression, and integration coverage for both runtime and evolution.

## Global Invariants

- The project is single-device by design. Do not introduce DDP, model parallelism, or hidden multi-GPU assumptions.
- External optimizers must implement `build_optimizer(model, max_steps)` and return a controller with:
  - `step(weights, grads, activations, step_fn)`
  - `zero_grad(set_to_none=True)`
- Canonical evolution is organism-first and island-aware. `run_evolution(...)` must remain wired to the current `EvolutionLoop`.
- Canonical prompt assets live under `conf/prompts/`:
  - `shared/` for shared system context
  - `seed/`, `mutation/`, `crossover/` for paired task prompts
  - `islands/` for research directions
- Canonical organism artifacts are first-class data. Keep these filenames and meanings stable:
  - `optimizer.py`
  - `genetic_code.md`
  - `lineage.json`
  - `organism.json`
  - `summary.json`
  - `llm_request.json`
  - `llm_response.json`

## Editing Rules

- Keep config, code, and tests in sync. This repo relies heavily on strict contracts.
- Prefer extending existing shared helpers before adding one-off implementations.
- If you add a new experiment, update all of:
  - `conf/config.yaml`
  - `conf/experiments/<name>.yaml`
  - `experiments/<name>/`
  - `optbench/registry.py`
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
