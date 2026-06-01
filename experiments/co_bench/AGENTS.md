# co_bench — CO-Bench combinatorial-optimization experiment family

Task-blind organism evolution against [CO-Bench](https://github.com/sunnweiwei/CO-Bench).
Each organism is a single candidate function scored by CO-Bench's own per-task
evaluator (higher normalized score = better).

## Candidate contract

Every organism's `implementation.py` must define:

```python
def solve(**kwargs) -> dict:
    ...
```

CO-Bench invokes it as `solve(**instance)` (every instance field is passed by
keyword), then scores the result with the task's `eval_func(**instance,
**solution)`. The per-task input keys and required output keys are described in
that task's `project_context.txt` prompt (e.g. TSP receives `nodes` and returns
`{"tour": [...]}`). A `**kwargs` parameter is mandatory so unexpected instance
keys are tolerated; required positional-only parameters are rejected by the
candidate loader.

The report returned by `CoBenchExperimentEvaluator.evaluate_organism` uses
`score = feedback.dev_score` (CO-Bench's dev split, normalized so higher is
better); `objective_name="cobench_dev_score"`, `objective_direction="max"`.

## Checkout + dataset (not pip-installable)

CO-Bench ships top-level packages `evaluation` and `agents` and is not a pip
package. The runtime bridge (`_runtime/cobench_bridge.py`) puts the checkout
root on `sys.path` just long enough to import `evaluation`, then caches it.

- Checkout root: `$COBENCH_ROOT` if set, else `<project_root>/third_party/CO-Bench`.
- Dataset root: `data.cobench_src_dir` (defaults to `${paths.data_root}/co-bench`),
  resolved to an **absolute** path before `get_data` — CO-Bench re-imports
  `eval_func` from `config.py` inside spawned subprocesses, so a relative
  `src_dir` would break once a worker changes directory.
- Each task is a directory named **exactly** as CO-Bench names it (passed
  verbatim as `co_bench_task`), e.g. `Travelling salesman problem`, containing a
  `config.py` (with `solve`/`load_data`/`eval_func`/`norm_score`/`DESCRIPTION`)
  plus instance files.

A missing checkout or a missing solver dependency (ortools/networkx/pulp/...)
raises `OptionalDependencyError("co_bench", ...)`, which the host runner turns
into `status="skipped"` in smoke mode instead of crashing.

## Bootstrap

```bash
scripts/bootstrap_cobench.sh
```

Clones the repo into `third_party/CO-Bench` (skipped if present), runs
`pip install -e ".[co_bench]"`, and downloads the dataset via
`huggingface_hub.snapshot_download` into `${AIFS_DATA_ROOT:-./data}/co-bench`.
Override the checkout with `COBENCH_ROOT` and the data root with `AIFS_DATA_ROOT`.

## ONE config, task chosen by a field

The whole family runs from a SINGLE top-level preset, `conf/config_co-bench.yaml`.
There are NO per-task presets and NO `conf/runs/` tree. The active task is
selected by the field `experiments.co_bench.CO_BENCH_TASK` — an UPPER-CASE
identifier — which can be set in the preset or overridden on the command line
(it defaults to `TSP`):

```bash
python -m src.main --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=JOB_SHOP mode=smoke
./scripts/seed_population.sh --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=TSP
./scripts/run_evolution.sh  --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=SET_COVERING
python -m src.main --config-name baselines/co-bench experiments.co_bench.CO_BENCH_TASK=TSP   # baseline
```

The identifier &rarr; `{dataset task name, slug}` mapping lives in the
`co_bench_registry` in `conf/experiments/co-bench/co_bench.yaml`. That node is
self-contained: `co_bench_task` and `name` (the slug) resolve from the registry
via **node-relative** interpolation (`${.co_bench_registry.${.CO_BENCH_TASK}.*}`)
so they survive the per-experiment config detachment in `src/validate/runner.py`.

Shipped identifiers: `TSP`, `BIN_PACKING_1D`, `MULTI_KNAPSACK`, `SET_COVERING`,
`GRAPH_COLOURING`, `JOB_SHOP`.

`conf/config_co-bench.yaml` derives a `co_bench_slug` from the selected task's
`name` and uses it to point the seed program and the per-task project-context
prompt at the right files:

- `evolver.islands.seed_program_path` &rarr; `experiments/co_bench/_baselines/shinka/<slug>/initial_program.py`
- `evolver.prompts.project_context` &rarr; `conf/experiments/co-bench/prompts/<slug>/project_context.txt`

All OTHER prompts (`genome_schema`, `seed/*`, `mutation/*`, `crossover/*`,
`novelty/*`, `rationalization/*`, `implementation/*`, `repair/*`, `shared/*`)
are shared across the family and live directly under
`conf/experiments/co-bench/prompts/`. The single evolver is
`conf/evolver/co-bench.yaml`; its `phases.*.experiments` reference the static
experiment KEY `co_bench` (not the slug).

## Adding one of CO-Bench's other ~30 tasks

The runtime code (`_runtime/*`), the single evaluator, the single evolver, and
the shinka adapter (`_baselines/shinka/evaluate.py`) are all task-agnostic.
Adding a task is three local additions — no new preset, no new evolver/baseline:

1. **Registry row** — append to `co_bench_registry` in
   `conf/experiments/co-bench/co_bench.yaml`:
   `NEW_TASK: { task: "<CO-Bench Task Name>", slug: <slug> }`. The `task` value
   must match the dataset folder name **verbatim** (it is passed to
   `evaluation.get_data`); the `slug` is the lowercase identifier used in paths.
2. **Seed** — `_baselines/shinka/<slug>/initial_program.py`: a trivial valid
   `solve(**kwargs)` wrapped in `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.
3. **Per-task project context** — `conf/experiments/co-bench/prompts/<slug>/project_context.txt`:
   the problem description plus the `solve` input keys, required output keys,
   and objective. Source it from `data/co-bench/<CO-Bench Task Name>/config.py`
   (`DESCRIPTION` + the `solve` docstring).

Then select it with `experiments.co_bench.CO_BENCH_TASK=NEW_TASK`. Nothing else
changes per task.
