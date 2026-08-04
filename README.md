# Organism Framework

Task-blind, organism-first evolutionary search for algorithms. An organism is
a sectioned hypothesis (`genetic_code.md`), executable candidate
(`implementation.py`), lineage, LLM audit records, and evaluator reports. The
canonical engine is the island-aware `EvolutionLoop`; task code never leaks
into `src/`.

Read `AGENTS.md` for repository invariants and
`REPOSITORY_KNOWLEDGE_BASE.md` for the cross-branch audit, exact EvolutionLoop
history, task readiness, results, and operational notes. `FRAMEWORK.md` is the
long lifecycle/post-mortem walkthrough.

## Repository map

```text
src/                  task-blind evolution, validation, state, telemetry
experiments/          evaluator implementations and machine-readable catalog
conf/evolver/         task mechanics, islands, prompts, retry/selection policy
conf/backbone/        interchangeable LLM routes and stage pipelines
conf/experiments/     task runtime configuration and prompt assets
api_platforms/        provider-neutral LLM registry and backends
scripts/cluster/      reproducible single-job 8xH100 serving + monitoring
scripts/analysis/     post-mortem renderers for persisted LLM traces
scripts/legacy/       compatibility-only historical launchers
tests/                contracts, regressions, and integrations
```

The separation is intentional: change a task through an experiment preset;
change every LLM stage through one backbone override.

## Install and verify

Python 3.10+ is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[evolve]'
.venv/bin/pytest -q
```

Optional extras are `audio`, `hf`, `lora`, `shinka_baseline`, and `co_bench`.
The CO-Bench checkout and data are external and pinned by
`scripts/bootstrap_cobench.sh`.

## Discover and audit tasks

```bash
python -m experiments.catalog list
python -m experiments.catalog markdown
python -m experiments.catalog check
python -m experiments.catalog check --require-ready
```

The catalog records each candidate API, score semantics, preset/selector,
phase, default state, and scientific caveats. `check` distinguishes broken
contracts from missing datasets or baseline profiles.

Shipped families:

- `circle_packing_shinka`: `run_packing()` for 26 circles; maximize the
  evaluator-computed sum of radii subject to geometry constraints.
- `awtf2025_heuristic`: `solve_case(input_text)`; maximize the negative mean
  absolute score on the fixed vendored corpus (a reproducible surrogate, not
  the relative leaderboard score).
- `co_bench`: `solve(**instance)` for six combinatorial tasks; maximize finite
  normalized dev score. Test metrics are hidden during evolution.
- `optimization_survey`: `build_optimizer(model, max_steps)` across 17 training
  tasks; requires generated baseline profiles for meaningful scoring.

## Run one task

Every entrypoint requires an explicit Hydra preset.

```bash
# Validate a concrete organism
python -m src.main --config-name config_circle_packing_shinka \
  mode=run +organism_dir=/absolute/path/to/organism

# Create generation 0, then evolve
./scripts/seed_population.sh --config-name config_circle_packing_shinka
./scripts/run_evolution.sh --config-name config_circle_packing_shinka

# One idempotent command: seed only when needed, then resume/evolve
./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka
```

Population resume state lives in `population_state.json`. A saved inflight plan
is replayed, not resampled. Do not delete or hand-edit organism/state artifacts
to resume a run.

## Compare backbone LLMs

The task remains fixed while `backbone=<profile>` changes all LLM stages:

```bash
./scripts/run_evolution.sh --seed \
  --config-name config_circle_packing_shinka \
  backbone=ollama_qwen122_gemma31

./scripts/run_evolution.sh --seed \
  --config-name config_circle_packing_shinka \
  backbone=deepseek_v4_flash_0731 \
  paths.population_root=/absolute/path/to/a/fresh/population
```

Available canonical profiles are under `conf/backbone/`. Use a fresh population
root for a fair model comparison. Every provider call is appended to
`<population>/llm_usage.jsonl`; summarize it with:

```bash
python -m src.evolve.token_usage_report \
  /absolute/path/to/population/llm_usage.jsonl \
  --output /absolute/path/to/token_usage_summary.json
```

For human-readable trace dumps use `python -m scripts.analysis.dump_llm` for
one generation or `python -m scripts.analysis.dump_llm_excerpt` for a complete
population.

## Add a task

Keep these pieces together in one change:

1. `experiments/<family>/<task>/` with
   `evaluate_organism(organism_dir, cfg) -> dict`;
2. `conf/experiments/<family>/<task>.yaml` with a valid `_target_`;
3. the relevant top-level `conf/config_*.yaml` default/selector;
4. prompts and a seed implementation owned by the family;
5. one `experiments/catalog.py` entry and knowledge-base update;
6. Hydra, evaluator-contract, and task regression tests.

Every successful report must include a finite numeric `score`; larger is always
better to the task-blind engine.

## Add or compare a backbone

Add one `conf/backbone/<name>.yaml` describing route identity, provider backend,
stage options, token budgets, and pipelines. Do not duplicate route identities
inside task evolver configs. The generic `api_platforms.route.build_platform`
factory covers mock, Ollama, OpenAI-compatible, and existing provider backends.

## Cluster job: DeepSeek-V4-Flash-0731 on 8×H100

The production job path is documented in `scripts/cluster/README.md` and
`agents/remote_cluster_access.md`. It uses one `type="binary"` worker with
`processes_per_worker=1` and one coordinator; never use a multi-rank
`pytorch2` launcher for EvolutionLoop. Because SR008 and Jupyter use distinct
NFS namespaces, each job clones and verifies the exact commit into persistent
regional NFS before it starts or resumes.

The job pins the model revision and SGLang version, validates all eight GPUs,
starts TP=8 SGLang with the checkpoint-bundled DSpark head, smoke-tests the
OpenAI endpoint, runs/resumes EvolutionLoop, and writes durable manifests plus
a terminal completion event. Credentials are environment-only and must never
be committed.

## Canonical artifacts

```text
<population>/
  population_state.json
  llm_usage.jsonl
  run.log / seed_run.log
  gen_NNNN/island_<id>/org_<id>/
    implementation.py
    genetic_code.md
    lineage.json
    organism.json
    summary.json
    llm_request.json
    llm_response.json
    llm_rationalization.json   # when the two-step pipeline is used
    results/
    logs/
```

`organism.json` and `population_state.json` are first-class source records, not
disposable caches.
