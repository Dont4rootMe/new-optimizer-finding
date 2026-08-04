# Repository Guide

This file applies to the whole repository unless a deeper `AGENTS.md` overrides it.

`REPOSITORY_KNOWLEDGE_BASE.md` is the living cross-branch knowledge database for
this repository. Read it before work that depends on repository architecture,
experiment status, upstream branches, operational history, or empirical
results. Live code, composed config, and tests remain the ultimate source of
truth; the database records their current interpretation and explicitly marks
historical or upstream-only material.

Every agent must keep `REPOSITORY_KNOWLEDGE_BASE.md` up to date. Before
finishing any repository task, review whether the task or its investigation
changed or revealed anything about behavior, configuration, prompts,
experiments, branches, results, failure modes, artifacts, or operations, and
update the database in the same change. A repository change affecting one of
those areas is incomplete without the corresponding database update. Record
the audit date and relevant commit/ref, distinguish verified facts from
hypotheses, and never copy credentials or secrets into the database.

# CLUSTER JOB NAMES — HUMAN-SELECTED, NEVER AGENT-INVENTED

The human operator must explicitly choose every cluster `run_id`. An agent must
never invent, infer, auto-generate, or silently accept a default run/job label.
If the operator has not supplied the name, stop before submission and ask for
it. Pass the exact approved value through `scripts.cluster.submit --run-id`;
that value is used in the scheduler-visible job description and as the durable
artifact namespace. Cloud.ru separately assigns the immutable technical ID
`lm-mpi-job-<uuid>`, which is not user-selectable.

User-chosen scheduler-visible labels must not contain backbone provider, model,
or revision names such as `deepseek`, `qwen`, or `gemma`. Keep labels accurate
to the actual workload; this naming rule does not authorize presenting a job as
an unrelated project. Never rename or restart an active job merely to apply a
new label: record its historical identifier verbatim and apply this protocol to
the next submission.

For a historical deep walkthrough of the evolutionary pipeline and the 426-,
1516-, and 491-organism atcoder post-mortems, also read `FRAMEWORK.md`. It is a
valuable post-mortem but contains behavior that has since been superseded; use
the knowledge database's drift ledger and live code/config/tests when they
conflict.

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
- Population resume state is stored in `population_state.json`. Bandit posteriors (per `src/evolve/bandit.py`) live inside that file under `bandit_state`; do not move them to a sibling file without updating both `read_population_state` and `EvolutionLoop._restore_bandit_state` together.
- Optimization-specific contracts such as `build_optimizer(model, max_steps)` belong only under `experiments/optimization_survey/`.
- Adaptive sampling (LLM route, parent island, cross-island partner) is configurable per family under `evolver.{llm.route_sampling, reproduction.parent_island_sampling, reproduction.cross_island_partner_sampling}`. All three default to `uniform` / `weighted_static` (legacy behaviour); `strategy: bandit` swaps in discounted Thompson sampling. See FRAMEWORK.md "Adaptive sampling (bandits)" for details.
- The design stage runs as a two-step pipeline (Step 1 rationalization → Step 2 formalization) for families that ship `prompts/rationalization/{mutation,crossover}/{system,user}.txt`. Step 1's output is stored in `org_dir/llm_rationalization.json` (separate from the canonical `llm_request.json`); it is cached across validator retries. See FRAMEWORK.md "Two-step design pipeline" for details.

## Canonical EvolutionLoop Protocol

This is the last author implementation discovered across every reachable local,
upstream, PR, reflog, and cluster checkout as of **2026-08-04**. The canonical
file is `src/evolve/evolution_loop.py`, Git blob
`a4164058053a605e884b91748ab481dd542039df`; its last behavior-changing commit
is `b51afb6abb8067e9013f78ccd03bc13f6fd1c8b5`. The same blob is present in
`origin/master@4ca9bbb` and the later full project snapshot
`origin/adding-co-bench@641777d`. Unless an intentional successor is introduced
with migration tests and this section is updated, "EvolutionLoop" means this
protocol:

1. `run_seed_population(cfg)` and `run_evolution(cfg)` create exactly one
   `ApiPlatformRegistry` and inject it into exactly one `EvolutionLoop`. The
   production entrypoint must not substitute ShinkaEvolve or another loop.
2. Generation 0 is a separate, resumable seed transaction. In the only
   supported `from_seed` mode, one configured handwritten implementation is
   copied into the requested number of organisms per island, each copy is
   simple-evaluated, failed attempts are topped up while budget/progress allow,
   and the selected population is committed to `population_state.json`.
   Calling evolution without that state is an error; silently reseeding is not
   allowed.
3. Before generation `g`, resume restores the active population, persisted
   bandit posteriors, token accounting, and any `inflight_generation`. A saved
   offspring plan is replayed; it is not sampled again. Stop checks run before
   planning and cover `max_generations`, planned organism-creation attempts,
   and per-route/model token budgets.
4. A generation first samples only feasible reproduction routes (mutation,
   same-island crossover, cross-island crossover), parents/islands, and one LLM
   route or named pipeline per child. Every planned child receives a canonical
   directory/stub and the full plan is persisted before concurrent work starts.
5. Child creation is organism-first: optional rationalization → formalized
   sectioned genetic code → novelty gate → implementation compile/extraction.
   The configured retry budgets govern novelty regeneration, implementation
   repair, and post-evaluation repair. Completed children enter simple
   evaluation as soon as they are materialized; unrelated children may remain
   in flight.
6. Parents and successful offspring form one candidate pool. Survival is
   island-local top-K by `simple_score`. On configured Great Filter
   generations, simple survivors receive hard evaluation and survival becomes
   island-local top-H by `hard_score`; otherwise simple survivors are the next
   active population.
7. Lineage/status artifacts, parent-usage counts, pipeline attribution, token
   usage, and adaptive-bandit outcomes are updated from the realized child
   result. The generation is finalized only when `population_state.json` is
   saved with no inflight plan; visualization and optional Comet telemetry are
   observational and must not determine selection.
8. Canonical resume data and organism files are first-class records, not a
   disposable cache. Refactors may relocate orchestration helpers or introduce
   provider backends, task registries, and cluster launchers, but must preserve
   planning, retry, evaluation, selection, state, accounting, and resume
   semantics unless the algorithm change is explicitly requested and covered
   by focused regression tests.

For the evidence trail, branch matrix, drift notes, and historical run results,
see `REPOSITORY_KNOWLEDGE_BASE.md` rather than inferring finality from a local
branch name.

## Editing Rules

- Keep config, code, and tests in sync. This repo relies heavily on strict contracts.
- Prefer extending existing shared helpers before adding one-off implementations.
- If you add a new experiment, update all of:
  - the relevant top-level `conf/config_<preset>.yaml`
  - `conf/experiments/<family>/<name>.yaml`
  - `experiments/<family>/<name>/`
  - the experiment YAML `_target_`
  - relevant tests
  - `REPOSITORY_KNOWLEDGE_BASE.md`
- If you change prompt placeholders, prompt file layout, or structured response sections, update the prompt builders and parser/contract tests together.

## Verification

- Full regression suite: `pytest -q`
- Common focused checks:
  - `pytest -q tests/test_hydra_compose.py`
  - `pytest -q tests/test_import_optimizer.py`
  - `pytest -q tests/test_prompt_bundle.py`
  - `pytest -q tests/test_organism_contract.py`
  - `pytest -q tests/test_run_evolution.py`
