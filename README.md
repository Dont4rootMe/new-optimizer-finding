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
- `experiments/co_bench/`: CO-Bench combinatorial-optimization task family (one evaluator, many tasks)
- `conf/experiments/optimization_survey/`: optimization-specific experiment configs and prompts
- `conf/experiments/circle_packing_shinka/`: circle-packing configs and prompts
- `conf/experiments/awtf2025_heuristic/`: awtf2025 configs, prompts, and research islands
- `conf/experiments/co-bench/`: CO-Bench task registry, shared prompts, and per-task project contexts
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

## Pipeline architecture

One organism creation cycle. Each `╔═...═╗` block is one LLM call with its
own pipeline-routed model. Italicised items are this framework's
differentiators over FunSearch / ShinkaEvolve / AlphaEvolve.

```text
┌─────────────────────────────────────────────────────────────────────┐
│  EvolutionLoop._execute_planned_creations  (evolution_loop.py)      │
│  - Bandit-island sampler picks parent island                        │
│  - Operator router (mutation / within-island XO / inter-island XO)  │
│  - For each operator: _pick_inspirations(parent) — top-K survivors  │
│    from parent.island_id by simple_score, excluding parent          │
│    (default N=2, knob: evolver.prompts.num_inspirations)            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  MutationOperator.produce(parent, inspirations)                    │
   │   or CrossbreedingOperator.produce(mother, father, inspirations)   │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  STEP 1: RATIONALIZATION    (* framework differentiator *)                  ║
║  ───────────────────────                                                    ║
║                                                                             ║
║  Route: pipeline.rationalization                                            ║
║  Prompts: rationalization/{mutation,crossover}/{system,user}.txt            ║
║                                                                             ║
║  User-prompt context:                                                       ║
║    • PARENT FITNESS SIGNAL    (parent_score, best_ancestor, gap)            ║
║    • PARENT GENETIC CODE      (prose, 6 sections)                           ║
║    • PARENT IMPLEMENTATION    raw parent.py — anchors weakness diagnosis    ║
║                               against ground-truth code, not just prose     ║
║    • PARENT LINEAGE SUMMARY   (last N ancestors + scores + change_desc)     ║
║    • TOP-K INSPIRATION        2 best siblings on the same island with       ║
║                               their scores + impl.py + change_desc          ║
║    • LINEAGE REGIME HINT      * differentiator * — auto-detected family    ║
║                               the lineage is converging to (wall /          ║
║                               grouping / routing / repair). The LLM is      ║
║                               forced to break out of that family.           ║
║    • NOVELTY REJECTION FEEDBACK (only on retry)                             ║
║                                                                             ║
║  Output: 6-section prose plan                                               ║
║    1. SCORE_BEARING_CORE       what mechanism currently earns the score     ║
║    2. LINEAGE_REGIME_DIAGNOSIS what family the lineage sits in              ║
║    3. WEAKNESS_HYPOTHESIS      what the parent is failing at                ║
║    4. WHAT_TO_REMOVE           parent material to drop                      ║
║    5. WHAT_TO_ADD_OR_INVENT    new mechanism — must change ≥1 family axis   ║
║    6. CHILD_DIRECTION          one coherent score-bearing hypothesis        ║
║                                                                             ║
║  → cached and reused across novelty/compatibility retries                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  STEP 2: FORMALIZATION   (* differentiator: 2-layer abstraction *)          ║
║  ──────────────────────                                                     ║
║                                                                             ║
║  Route: pipeline.design                                                     ║
║  Prompts: {mutation,crossover}/{system,user}.txt                            ║
║                                                                             ║
║  User-prompt context:                                                       ║
║    • GENOME SECTION SCHEMA    (authoritative format)                        ║
║    • STEP 1 DESIGN RATIONALE  ← from Step 1 output                          ║
║    • PARENT GENETIC CODE                                                    ║
║    • PARENT IMPLEMENTATION    raw parent.py                                 ║
║    • PARENT LINEAGE SUMMARY                                                 ║
║    • TOP-K INSPIRATION                                                      ║
║    • NOVELTY REJECTION FEEDBACK (only on retry)                             ║
║                                                                             ║
║  Output: canonical sectioned genetic_code.md                                ║
║    ## CORE_GENES                                                            ║
║      ### STATE_REPRESENTATION                                               ║
║      ### MACRO_STRATEGY                                                     ║
║      ### CONSTRUCTION_POLICY                                                ║
║      ### LOCAL_REPAIR_POLICY                                                ║
║      ### OPTIONAL_CODE_SKETCH                                               ║
║    ## INTERACTION_NOTES                                                     ║
║    ## COMPUTE_NOTES                                                         ║
║    ## CHANGE_DESCRIPTION                                                    ║
║                                                                             ║
║  Failure path: parser-rejection retries see the previous attempt's error    ║
║  inside a PREVIOUS ATTEMPT FAILED block with a list of common parser flags. ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
              ┌─────────────────────────────────────┐
              │ Parser: parse_genetic_code_text     │
              │  - validates schema                 │
              │  - auto-completes missing trailing  │
              │    OPTIONAL_CODE_SKETCH as `- None.`│
              └─────────────────────────────────────┘
                                  │
                                  ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  VALIDATION GATES (sequential — each can reject → triggers Step-2 retry)    ║
║  ───────────────────────────────────────────────────────────────────────    ║
║                                                                             ║
║  ┌──────────────────────┐    ┌──────────────────────┐                       ║
║  │ NOVELTY CHECK        │    │ COMPATIBILITY CHECK  │                       ║
║  │ route: pipeline.     │    │ route: pipeline.     │                       ║
║  │   novelty            │    │   compatibility      │                       ║
║  │ Sees: parent + cand. │    │ Sees: same           │                       ║
║  │   genetic_code only  │    │ Rejects: contradic-  │                       ║
║  │ Rejects: paraphrase, │    │   tions, scope creep,│                       ║
║  │   regime not changed │    │   unsupported mechs  │                       ║
║  │   coefficient retune │    │                      │                       ║
║  └──────────────────────┘    └──────────────────────┘                       ║
║                                                                             ║
║  Bounded retries each via max_attempts_to_regenerate_organism_after_*.      ║
║  Step-1 rationale is reused; only Step 2 is regenerated with rejection      ║
║  feedback injected.                                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  STEP 3: IMPLEMENTATION                                                     ║
║  ────────────────────────                                                   ║
║                                                                             ║
║  Route: pipeline.implementation                                             ║
║  Prompts: implementation/{system,user}.txt + shared/template.txt scaffold   ║
║                                                                             ║
║  User-prompt context:                                                       ║
║    • FUNCTION CONTRACT       (def solve_case(input_text) -> str)            ║
║    • COMPILATION_MODE        FULL or PATCH (auto-decided by how many        ║
║                              sections of genetic_code changed)              ║
║    • ORGANISM GENETIC CODE   (the new design from Step 2)                   ║
║    • ORGANISM CHANGE_DESCRIPTION                                            ║
║    • MATERNAL BASE GENETIC CODE     parent's old genetic_code               ║
║    • MATERNAL BASE IMPLEMENTATION   parent's actual Python — always in      ║
║                                     scope (helpers may be reused verbatim)  ║
║    • PRE-DEFINED LOCAL VARIABLES    scaffold pre-executes:                  ║
║                                     n, k, vertical, horizontal,             ║
║                                     groups, operations                      ║
║                                                                             ║
║  Output: Python region bodies that mutate the scaffold's locals.            ║
║                                                                             ║
║  Instruction: "use MATERNAL BASE IMPLEMENTATION as a reference baseline.    ║
║  Reuse helpers and idioms verbatim where the child genes are silent.        ║
║  Re-synthesizing every function from prose is wasted work."                 ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
              ┌─────────────────────────────────────┐
              │ Persistence: organism saved to disk │
              │   organism.json     (meta + scores) │
              │   genetic_code.md   (canonical)     │
              │   implementation.py (canonical)     │
              │   llm_request.json  (audit)         │
              │   llm_response.json (audit)         │
              │   llm_rationalization.json (Step 1) │
              └─────────────────────────────────────┘
                                  │
                                  ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  SIMPLE EVAL                                                                ║
║  ───────────                                                                ║
║                                                                             ║
║  Orchestrator runs solve_case(input_text) per case across the corpus        ║
║  (parallel via resources.evaluation.cpu_parallel_jobs). Score is the        ║
║  task's negated objective (higher = better):                                ║
║                                                                             ║
║    awtf2025:        simple_score = -mean_absolute_score                     ║
║                                  = -(T + 100 * Σ_k Manhattan(final, tgt))   ║
║    circle_packing:  simple_score =  sum_of_radii                            ║
║                                                                             ║
║  Stdout/stderr per case logged for repair-LLM visibility.                   ║
║                                                                             ║
║  Outcomes:                                                                  ║
║    • ok   → simple_score stored on organism → bandit feedback (next block)  ║
║    • fail → _repair_organism_after_eval_error triggers repair-LLM           ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                  │
                        ┌─────────┴──────────┐
                        ▼                    ▼
                     (ok path)          (failure path)
                        │                    │
                        │                    ▼
                        │  ╔═══════════════════════════════════════════════╗
                        │  ║  REPAIR  (only on eval failure)               ║
                        │  ║  ─────────                                    ║
                        │  ║  Route: pipeline.repair                       ║
                        │  ║  Prompts: repair/{system,user}.txt            ║
                        │  ║                                               ║
                        │  ║  Context:                                     ║
                        │  ║    • current_implementation.py                ║
                        │  ║    • organism_genetic_code                    ║
                        │  ║    • change_description                       ║
                        │  ║    • implementation_template (scaffold)       ║
                        │  ║    • error_history with                       ║
                        │  ║        attempt, status, timestamp, error_msg  ║
                        │  ║      AND                                      ║
                        │  ║        LAST STDERR (last 3 kB, raw)           ║
                        │  ║        LAST STDOUT (last 3 kB, raw)           ║
                        │  ║                                               ║
                        │  ║  Output: full new implementation.py           ║
                        │  ║  Retry budget: max_repair_attempts            ║
                        │  ╚═══════════════════════════════════════════════╝
                        │                    │
                        │                    ▼
                        │              (re-evaluate)
                        │                    │
                        └────────┬───────────┘
                                 ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║  BANDIT FEEDBACK (3 parallel bandits get rewarded for this organism)        ║
║  ──────────────                                                             ║
║                                                                             ║
║  reward = score_quantile within a sliding window of recent organisms        ║
║  (clipped to [0, 1], dead organisms = 0).                                   ║
║                                                                             ║
║  ├── parent_island_sampler.observe(island_id, simple_score)                 ║
║  │     — which island produces good organisms                               ║
║  │                                                                          ║
║  ├── cross_island_partner_sampler.observe(island_id, partner_island,        ║
║  │                                        simple_score)                     ║
║  │     — per-source-island: which partner is good for inter-island XO       ║
║  │                                                                          ║
║  └── pipeline_sampler.observe(pipeline_id, simple_score)                    ║
║        — which LLM-pipeline configuration is winning. Lazily sampled per    ║
║          organism on first stage; cached for all remaining stages           ║
║                                                                             ║
║  Algorithm: Discounted Thompson Sampling (γ=0.97 ≈ 33 effective recent      ║
║  observations).                                                             ║
║                                                                             ║
║  Persistence: bandit state → population_state.json (resume-safe)            ║
╚═════════════════════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │ Survivor selection                  │
              │   select_top_k_per_island(          │
              │     population, k=max_per_island)   │
              │   - kills bottom of each island     │
              │   - survivors become next gen's     │
              │     candidate pool for parent       │
              │     sampling + inspiration pool     │
              └─────────────────────────────────────┘
                                 │
                                 └──→ NEXT GENERATION
```

### What each LLM stage sees vs ShinkaEvolve

| Stage | LLM call | What it sees now | ShinkaEvolve analogue | Our differentiator |
|---|---|---|---|---|
| **Step 1: Rationalization** | yes | parent.py + parent_genetic_code + 2 inspirations.py + fitness gap + **lineage regime hint** + lineage scores | — *(no analogous stage)* | **Anti-convergence planner**: regime detector + explicit requirement to switch family (wall / grouping / routing / repair) |
| **Step 2: Formalization** | yes | everything from Step 1 + rationalization output | (part of the single FunSearch / Shinka prompt) | 2-layer abstraction: design is locked in as a structured artifact, not as code |
| Novelty validator | yes | parent + candidate genetic_code | — | Semantic rejection of paraphrased mutations |
| Compatibility validator | yes | parent + candidate genetic_code | — | Internal cross-section consistency check |
| **Step 3: Implementation** | yes | child genetic_code + **parent.py** (always in scope) + change_description + scaffold | full / diff / cross patch over the single Python program | Compiler-LLM sees both layers: new design + old code as a baseline |
| Simple eval | no (host evaluator) | — | yes | Identical pattern |
| Repair | yes (only on eval crash) | implementation + error_msg + **raw stdout + stderr** + genetic_code + change_desc | FIX mode with stderr | Identical pattern |
| Bandit feedback | no | score | UCB1 | DiscountedThompson with 3 parallel bandits (route / island / partner) |

ShinkaEvolve packs everything into one long `task_sys_msg + diff_user_msg` →
Python-patch call. We split it into four linked calls (rationalize → formalize
→ validate × 2 → implement). The trade-off:

- More LLM compute per organism (~3× the calls).
- Step 1 acts as an explicit anti-convergence planner — ShinkaEvolve has
  no equivalent and relies on island isolation alone.
- Step 2 + Step 3 are a two-layer compiler: the design is fixed as a
  structured artifact first, then compiled to Python with the parent's
  code in scope.

The research claim is: the planner stage produces strictly better mutations
than direct code edits, paying back the extra compute via fewer wasted
generations on regime-stuck lineages.

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

To stop by total organism creation attempts instead of generation count, disable the generation cap and set `evolver.max_organism_creations`. Pending and failed organism creations count toward this cap:

```bash
./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka evolver.max_generations=false evolver.max_organism_creations=200
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

## CO-Bench

The repository also ships [`experiments/co_bench/`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/co_bench), a task family that evolves combinatorial-optimization algorithms against [CO-Bench](https://github.com/sunnweiwei/CO-Bench) ("Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization", [arXiv:2504.04310](https://arxiv.org/abs/2504.04310)). Each organism is a single candidate function scored by CO-Bench's own per-task evaluator (higher normalized score = better).

Its organism contract is:

```python
def solve(**kwargs) -> dict:
    ...
```

CO-Bench invokes the candidate as `solve(**instance)` (every instance field is passed by keyword), then scores the returned dict with the task's `eval_func(**instance, **solution)`. The exact input keys, the required output keys, and the objective are described in each task's `project_context.txt` prompt (e.g. TSP receives `nodes` and returns `{"tour": [...]}`). A `**kwargs` parameter is mandatory so unexpected instance keys are tolerated; required positional-only parameters are rejected by the candidate loader.

### Bootstrap (checkout + dataset, not pip-installable)

CO-Bench ships top-level packages `evaluation`/`agents` and is not a pip package, so it must be checked out and its dataset downloaded:

```bash
scripts/bootstrap_cobench.sh
```

That script clones CO-Bench into `third_party/CO-Bench` (skipped if already present), runs `pip install -e ".[co_bench]"` for the solver/dataset deps, and downloads the Hugging Face dataset `CO-Bench/CO-Bench` into `./data/co-bench` via `huggingface_hub.snapshot_download`. Override the checkout location with `COBENCH_ROOT` and the dataset root with `AIFS_DATA_ROOT` (data lands in `<root>/co-bench`). A missing checkout or solver dependency makes `mode=smoke` report `status="skipped"` instead of crashing.

### Usage — ONE preset, task selected by a field

Unlike the other families, CO-Bench uses a **single** top-level preset, `config_co-bench`. There are **no** per-task presets. The active task is chosen by the field `experiments.co_bench.CO_BENCH_TASK`, an UPPER-CASE identifier:

| `CO_BENCH_TASK` | CO-Bench dataset task |
|---|---|
| `TSP` | Travelling salesman problem |
| `BIN_PACKING_1D` | Bin packing - one-dimensional |
| `MULTI_KNAPSACK` | Multidimensional knapsack problem |
| `SET_COVERING` | Set covering |
| `GRAPH_COLOURING` | Graph colouring |
| `JOB_SHOP` | Job shop scheduling |

Set it in `conf/config_co-bench.yaml` or override it on the command line (it defaults to `TSP`):

```bash
python -m src.main --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=JOB_SHOP mode=smoke
./scripts/seed_population.sh --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=TSP
./scripts/run_evolution.sh  --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=SET_COVERING
./scripts/run_evolution.sh  --seed --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=GRAPH_COLOURING
```

Run a concrete organism, or collect the baseline, the same way:

```bash
python -m src.main --config-name config_co-bench experiments.co_bench.CO_BENCH_TASK=MULTI_KNAPSACK mode=run +organism_dir=/absolute/path/to/organism
python -m src.main --config-name baselines/co-bench experiments.co_bench.CO_BENCH_TASK=TSP
```

The identifier &rarr; dataset-task mapping (plus the lowercase slug that drives the seed/prompt/output paths) lives in the `co_bench_registry` in `conf/experiments/co-bench/co_bench.yaml`; CO-Bench's other tasks are added there. See [`experiments/co_bench/AGENTS.md`](/Users/artemon/Library/Mobile%20Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/experiments/co_bench/AGENTS.md) for how to add one.

All user-facing entrypoints now require an explicit Hydra preset via `--config-name`.
