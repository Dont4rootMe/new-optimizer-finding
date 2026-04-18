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
- `experiments/awtf2025_heuristic/`: AtCoder AWTF 2025 heuristic task package
- `conf/experiments/optimization_survey/`: optimization-specific experiment configs and prompts
- `conf/experiments/circle_packing_shinka/`: circle-packing configs and prompts
- `conf/experiments/awtf2025_heuristic/`: awtf2025 configs, prompts, and research islands
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
python -m src.main --config-name config_optimization_survey mode=run +organism_dir=/absolute/path/to/organism
```

## Evolution

```bash
./scripts/seed_population.sh --config-name config_optimization_survey
./scripts/run_evolution.sh --config-name config_optimization_survey
./scripts/run_evolution.sh --seed --config-name config_optimization_survey
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
python -m src.main --config-name config_circle_packing_shinka mode=run +organism_dir=/absolute/path/to/organism
./scripts/seed_population.sh --config-name config_circle_packing_shinka
./scripts/run_evolution.sh --config-name config_circle_packing_shinka
./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka
```

That preset now uses one logical local Ollama route by default:

- `qwen3.5:122b` rooted at `http://127.0.0.1:12434/api` with `gpu_ranks=[[0,1,2],[3,4,5]]`

For grouped local Ollama routes, the `scripts/*` wrappers auto-start one local service per GPU group. Instance 0 keeps the configured `base_url`, later instances use incremented localhost ports, and each service receives its own `CUDA_VISIBLE_DEVICES` group before models are pulled and warmed.
By default, local Ollama weights are stored in `./ollama_cache` at the project root. Override that with `paths.ollama_cache_root=/absolute/path` or `OLLAMA_MODELS=/absolute/path`.

## Concurrency knobs

- `evolver.creation.max_parallel_organisms`: maximum number of organisms being created in parallel. Each creation task includes both LLM stages for one organism.
- `evolver.creation.max_attempts_to_create_organism`: retry budget for one organism creation when LLM output or provider calls fail.
- `evolver.creation.max_attempts_to_repair_organism_after_error`: maximum number of LLM repair passes after explicit evaluator errors for one organism phase evaluation.
- `resources.evaluation.cpu_parallel_jobs`: number of CPU-only evaluation tasks that can run at once.
- `resources.evaluation.gpu_ranks`: GPU evaluation worker slots. The number of concurrent GPU evaluation tasks equals the number of configured ranks.
- `api_platforms.<route>.gpu_ranks`: route GPU allocation. For Ollama routes, `int` means one GPU, `list[int]` means one multi-GPU instance, and `list[list[int]]` means multiple instances of the same model.
- `api_platforms.<route>.max_concurrency`: backend request concurrency per concrete routed model service, such as one local Ollama instance.

Its organism contract is:

```python
def run_packing():
    return centers, radii, reported_sum
```

where `centers.shape == (26, 2)`, `radii.shape == (26,)`, and `reported_sum == sum(radii)`.

## AWTF2025 Heuristic

The repository also ships [`experiments/awtf2025_heuristic/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/awtf2025_heuristic), a task family based on AtCoder World Tour Finals 2025 Heuristic A: Group Commands and Wall Planning.

Use the dedicated preset:

```bash
python -m src.main --config-name config_awtf2025_heuristic mode=run +organism_dir=/absolute/path/to/organism
./scripts/seed_population.sh --config-name config_awtf2025_heuristic
./scripts/run_evolution.sh --config-name config_awtf2025_heuristic
./scripts/run_evolution.sh --seed --config-name config_awtf2025_heuristic
```

Its organism contract is:

```python
def solve_case(input_text: str) -> str:
    ...
```

The evaluator runs that function on a vendored fixed corpus of official `seed=0..99` inputs and maximizes `-mean_absolute_score`, where each per-case absolute score is the official `T + 100 * sum_k Manhattan(final_position_k, target_k)`.

All user-facing entrypoints now require an explicit Hydra preset via `--config-name`.
