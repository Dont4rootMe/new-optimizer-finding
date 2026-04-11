# Organism Framework

Generic AI-for-science framework with:

- a task-blind core in [`src/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/src)
- task-specific experiment/runtime code in [`experiments/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments)
- Hydra configuration in [`conf/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/conf)

## Core contracts

- `src` only knows organism directories and experiment reports.
- Each experiment config is Hydra-instantiated via `_target_`.
- Each instantiated evaluator must implement `evaluate_organism(organism_dir, cfg) -> dict`.
- Every experiment report must contain `score`.

## Canonical organism layout

```text
<organism_dir>/
  implementation.py
  genetic_code.md
  lineage.json
  organism.json
  llm_request.json
  llm_response.json
  summary.json
  results/
  logs/
```

`organism.json` is the source of truth for core orchestration. Population-level resume state lives in `population_state.json`.

## Layout

- `src/`: generic runtime, validation worker, and organism-first evolution engine
- `experiments/optimization_survey/`: optimization-specific experiments and runtime helpers
- `experiments/circle_packing_shinka/`: circle-packing task package inspired by ShinkaEvolve
- `conf/experiments/optimization_survey/`: optimization-specific experiment configs and prompts
- `conf/experiments/circle_packing_shinka/`: circle-packing configs and prompts
- `tests/`: contract and integration coverage

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[audio]
pip install -e .[hf]
pip install -e .[lora]
pip install -e .[evolve]
```

## Validation runtime

Smoke enabled experiments:

```bash
python -m src.main --config-name config_optimization_survey mode=smoke
```

Collect baseline stats:

```bash
python -m src.main --config-name config_optimization_survey mode=stats
```

Run experiments against a concrete organism:

```bash
python -m src.main --config-name config_optimization_survey mode=run organism_dir=/absolute/path/to/organism
```

## Evolution

```bash
export OPENAI_API_KEY=...
python -m src.main --config-name config_optimization_survey mode=evolve
```

The evolution engine is task-blind. It operates on organism folders and delegates all task-specific behavior to the configured experiments.

## Optimization survey

The shipped task package is [`experiments/optimization_survey/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/optimization_survey). Its runtime knows how to interpret `implementation.py` as an optimizer candidate with the contract:

```python
def build_optimizer(model, max_steps):
    ...
```

and a controller exposing:

```python
class CandidateController:
    def step(self, weights, grads, activations, step_fn): ...
    def zero_grad(self, set_to_none=True): ...
```

## Circle Packing Shinka

The repository also ships [`experiments/circle_packing_shinka/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/circle_packing_shinka), a task family based on the ShinkaEvolve circle-packing benchmark.

Use the dedicated preset:

```bash
python -m src.main --config-name config_circle_packing_shinka mode=run organism_dir=/absolute/path/to/organism
./scripts/seed_population.sh --config-name config_circle_packing_shinka
./scripts/run_evolution.sh --config-name config_circle_packing_shinka
./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka
```

That preset isolates the two local Ollama routes by default:

- `gemma4:26b` on `http://127.0.0.1:11434/api` with `gpu_ranks=[0]`
- `qwen3.5:27b` on `http://127.0.0.1:11435/api` with `gpu_ranks=[1]`

The `scripts/*` wrappers auto-start one local Ollama service per distinct local `base_url`, pin it to the configured route GPU when `gpu_ranks` is set, and pull missing models before `seed` or `evolve` starts.

Its organism contract is:

```python
def run_packing():
    return centers, radii, reported_sum
```

where `centers.shape == (26, 2)`, `radii.shape == (26,)`, and `reported_sum == sum(radii)`.

All user-facing entrypoints now require an explicit Hydra preset via `--config-name`.
