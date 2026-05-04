# Framework Reference

This document explains how the LLM-driven evolutionary heuristic-search framework currently works end-to-end. It is the canonical reference for anyone (human or AI) who needs to reason about the pipeline beyond what `AGENTS.md` covers. Everything below describes the live code at the time of the post-mortem on the 426-organism atcoder run.

The companion plan-file `~/.claude/plans/quiet-munching-kahn.md` records the diagnosis that motivated the recent batch of changes (P1–P6).

## Big picture

The framework evolves *organisms*. An organism is a sectioned, plain-language description of a heuristic ("genetic code") plus the Python implementation it compiles to. Organisms live in islands; islands have a research-style bias. New organisms are produced by three operators (seed, mutation, crossover) executed by an LLM. Each candidate must clear two LLM-judged validators (compatibility, novelty) before its implementation is compiled and evaluated. Surviving organisms feed back into the parent pool for the next generation.

Three currently configured task families share this engine:

- `awtf2025_heuristic` — AtCoder grid-robot routing (the focus of this codebase right now)
- `circle_packing_shinka` — circle packing in the unit square
- `optimization_survey` — neural-network optimizer search

The engine in `src/evolve/` and the contract code in `src/organisms/` are deliberately task-blind. Everything task-specific lives in `conf/experiments/<family>/` (prompts, schema, scaffold, island seeds) and `experiments/<family>/` (evaluator runtime).

## Lifecycle of one organism

```
plan → seed/mutate/crossbreed → design (LLM) → validate → implement (LLM)
                                       ↑                      ↓
                                       └── repair retry ──────┤
                                                              ↓
                                            simple eval → (great_filter) → hard eval
                                                              ↓
                                                         lineage update
```

Pipeline state transitions (recorded in each organism's `organism.json`):

`planned_creation` → `creating` → `compatibility_check` → (loop with novelty) → `pending_simple_eval` → `running_simple_eval` → `simple_complete` | `failed_simple_eval`

Terminal failure states: `failed_creation` (any of: malformed artifact, exhausted compatibility/novelty retries, failed implementation extract), `failed_simple_eval`.

## Genome and implementation artifacts

The genome is a sectioned markdown document. It must contain exactly four top-level sections:

- `## CORE_GENES` — body is broken into subsections (one per canonical region)
- `## INTERACTION_NOTES` — free text about how sections compose
- `## COMPUTE_NOTES` — complexity / budget reasoning
- `## CHANGE_DESCRIPTION` — plain-language novelty summary used in lineage

Each task family declares the canonical CORE_GENES subsections via `conf/experiments/<family>/prompts/shared/genome_schema.txt`, parsed by `src/organisms/genetic_code_format.py`. For `awtf2025_heuristic` the eight subsections are: `STATE_REPRESENTATION`, `MACRO_STRATEGY`, `CONSTRUCTION_POLICY`, `PRIORITY_MODEL`, `LOCAL_REPAIR_POLICY`, `CONTROL_POLICY`, `PARAMETERS`, `OPTIONAL_CODE_SKETCH`.

The compiled implementation is a single Python file (`implementation.py`) defined by a scaffold at `conf/experiments/<family>/prompts/shared/template.txt`. The scaffold uses scaffold-internal markers `# === REGION: NAME ===` / `# === END_REGION: NAME ===` that the parser in `src/organisms/implementation_patch.py` recognizes. **The LLM never produces this scaffold form directly.**

The LLM produces an *artifact* in a strict patch format:

```
## COMPILATION_MODE
FULL

## REGION STATE_REPRESENTATION
    ...python lines, four-space indented...
## END_REGION

## REGION MACRO_STRATEGY
    ...
## END_REGION
```

`COMPILATION_MODE` is `FULL` (emit every region) or `PATCH` (emit only the regions in `CHANGED_SECTIONS`). The parser in `implementation_patch.py` then assembles this artifact into the actual `implementation.py` using the scaffold.

Three formatting conventions therefore coexist in the codebase, by design:

| Artifact                        | Markers                                  | Owner                          |
|---------------------------------|------------------------------------------|--------------------------------|
| Genome schema (`genome_schema.txt`) | `# SECTION_NAME`                       | `parse_genome_schema_text`     |
| Genetic code (`genetic_code.md`)    | `## CORE_GENES`, `### NAME`            | `parse_genetic_code_text`      |
| Implementation patch (LLM output)   | `## REGION NAME`, `## END_REGION`      | `parse_implementation_patch_response` |
| Implementation scaffold (file)      | `# === REGION: NAME ===`               | `parse_implementation_scaffold` |

Small local models stumbled on this distinction; the awtf2025_heuristic implementation prompt was rewritten to lead with one concrete artifact example and never show the scaffold-form markers to the LLM (see `implementation/system.txt` and `implementation/user.txt`).

## Validators

Two LLM judges sit between the design call and the implementation call. Each one runs as part of the same retry loop in `src/evolve/generator.py`.

**Compatibility validator** (`src/organisms/compatibility.py`): asks the LLM whether the candidate's sections are internally consistent. The verdict is `COMPATIBILITY_ACCEPTED` or `COMPATIBILITY_REJECTED` plus a free-text rejection reason. Rejection causes a re-design within the budget. Each operator (`seed`, `mutation`, `crossover`) has its own compatibility prompts under `conf/experiments/<family>/prompts/compatibility/<operator>/`.

**Novelty validator** (`src/organisms/novelty.py`): only runs for mutation and crossover. Asks the LLM whether the child is materially different from its parent(s). The verdict is `NOVELTY_ACCEPTED` or `NOVELTY_REJECTED` plus a rejection reason and an optional `SECTIONS_AT_ISSUE` list. Rejection causes a re-design within the budget.

**Repair-aware retry**: when either validator rejects, the next design attempt now (post-P6) appends a `=== PRIOR CANDIDATE TO REPAIR ===` block carrying the rejected candidate plus the critique, and instructs the LLM to *patch* that candidate rather than redesign from scratch. See `_append_rejected_candidate_repair_block` in `src/evolve/generator.py`. This converts the retry from a blind redesign into a targeted local edit.

**Implementation repair** (`prompts/repair/{system,user}.txt`): runs only after a successful design clears validators but the *implementation* compile-and-run failed (`failed_simple_eval`). Budget controlled by `creation.max_attempts_to_repair_organism_after_error`.

Retry budgets in `conf/evolver/awtf2025_heuristic.yaml`:

```yaml
creation:
  max_attempts_to_create_organism: 2
  max_attempts_to_repair_organism_after_error: 2
  max_attempts_to_regenerate_organism_after_novelty_rejection: 1
  max_attempts_to_regenerate_organism_after_compatibility_rejection: 1
```

These were cut from `(3, 2, 2, 3)` in P3; in the previous run they multiplied to up to 18 LLM calls per failed organism slot. The budgets above are intentionally tight because the new repair-aware retry recovers more per attempt.

## Operators

`src/evolve/operators.py`, `src/organisms/mutation.py`, `src/organisms/crossbreeding.py`.

**Seed** (`SeedOperator`) generates an organism from scratch given an island seed prompt. Used only at generation 0 and to top up under-populated islands.

**Mutation** (`MutationOperator`) prunes a parent's gene pool with probability `operators.mutation.gene_removal_probability` then asks the LLM to evolve the inherited genes into a child. Mutation is intended as hill climbing: change one coherent module, do not paradigm-shift.

**Crossover** (`CrossbreedingOperator`) merges genes from two parents. Primary parent dominance is controlled by `operators.crossover.primary_parent_gene_inheritance_probability` (default `0.7`). Crossover may be within-island or inter-island depending on `reproduction.island_sampling`.

Operator selection per generation lives in `reproduction.operator_weights`. The current weights (post-P4):

```yaml
operator_weights:
  within_island_crossover: 0.4
  inter_island_crossover: 0.1
  mutation: 0.5
```

Mutation now dominates, partly because the mid-generation novelty-rejection rate in the previous run peaked when crossover children collapsed back toward their primary parent.

## Islands

Island definitions live as plain-text seed prompts in `conf/experiments/<family>/prompts/islands/`. Each island:

- gets seeded with `evolver.islands.seed_organisms_per_island` organisms
- can hold up to `evolver.islands.max_organisms_per_island`
- biases the LLM toward a particular research style via its seed prompt

The `awtf2025_heuristic` family currently has two islands:

- `macro_partitioning` — wall planning, corridor decomposition, target-side grouping
- `staged_routing_repair` — greedy per-round progress driven by direction-profile bundling and lightweight conflict tally (rewritten in P2; the previous "plan-then-repair" framing was effectively dead at 0.9% success)

Island sampling for crossover is configured via `reproduction.island_sampling`.

## Selection

`src/evolve/selection.py`. Parents are sampled with one of:

- `uniform_select_organisms`
- `softmax_select_organisms` (and a no-replacement variant), using `simple_score` as the selection score
- `weighted_rule_select_organisms` (current default for awtf2025_heuristic), which combines fitness with parent-usage balance

Selection score weights live in `reproduction.selection_score.weights` (currently `simple_score=1.0`, `inheritance_fitness=0.0`).

## Phases

Defined under `evolver.phases`:

- **simple** — every viable organism runs through this. Aggregated as `simple_score` per organism, written to `summary.json`.
- **great_filter** — optional periodic re-evaluation that ranks the top-H organisms per island. Disabled in the awtf2025_heuristic config (`great_filter.enabled: false`).

Each phase declares the experiments to run (`experiments:`) and an evaluation budget. The actual `evaluate_organism(organism_dir, cfg) -> dict` lives under `experiments/<family>/`.

For `awtf2025_heuristic` the evaluator is `experiments.awtf2025_heuristic.group_commands_and_wall_planning.GroupCommandsAndWallPlanningExperiment`. It loads `implementation.py`, calls `solve_case(input_text)` against each test case in the configured corpus (`conf/experiments/awtf2025_heuristic/group_commands_and_wall_planning.yaml` lists smoke and full case ids), enforces a per-case soft timeout (1 second by default), and returns a report with `score = -mean_absolute_score` (negated because the engine maximizes).

Score formula for the actual contest: `T + 100 * sum_k Manhattan(final_k, target_k)`, lower is better. The trivial baseline (no walls, every robot in group `0`, no operations) scores `100 * sum_k Manhattan(start_k, target_k)`.

## Storage layout

Per `src/evolve/storage.py`, every generation writes to a population root:

```
population_root/
├── gen_0000/
│   └── island_<id>/
│       └── org_<hash>/
│           ├── genetic_code.md          # canonical genome
│           ├── implementation.py        # compiled scaffold
│           ├── lineage.json             # parent chain + scores
│           ├── organism.json            # OrganismMeta
│           ├── summary.json             # aggregated eval result
│           ├── llm_request.json         # all design/validation/implementation requests
│           ├── llm_response.json        # parallel structure of responses
│           ├── results/
│           │   ├── simple/<exp>.json
│           │   └── hard/<exp>.json
│           └── logs/...
├── gen_0001/...
└── population_state.json                # resume manifest
```

Resume reads `population_state.json` and rehydrates `OrganismMeta` for each active organism. Missing canonical files are real errors during resume.

The `dump_llm.py` script collapses one generation directory into per-organism markdown excerpts plus an `INDEX.md`. That tool is what produced the `~/Downloads/llm_excerpt_all/` post-mortem corpus.

## LLM routing

`src/evolve/llm_generator_base.py` plus the broker in `api_platforms/`. Routes are configured under `evolver.llm.route_weights`; the route id maps to an `api_platforms` entry. Selection strategy is currently `random` weighted by `route_weights`.

For `awtf2025_heuristic` the live routing in `conf/evolver/awtf2025_heuristic.yaml`:

```yaml
llm:
  selection_strategy: random
  seed: 123
  route_weights:
    ollama_gemma4_31b: 1.0
    ollama_qwen35_35b: 1.0
```

Both routes hit local Ollama. There is no per-stage routing — design, validation, repair and implementation all draw from the same weighted pool. This means the same model that produced an artifact may also be the one judging it (LLM-as-self-judge); the validator prompts are aware of that risk but the architecture does not anti-route validators yet.

## Configuration

The repo uses Hydra. Top-level entry configs live at `conf/config_<family>.yaml`. Each composes:

- `conf/evolver/<family>.yaml` (creation budgets, reproduction, phases, prompts, llm)
- `conf/experiments/<family>/<experiment>.yaml` (corpus, scoring, eval timeouts)
- `conf/api_platforms/...` (LLM provider definitions)
- `conf/resources/...` (CPU/GPU pool)

`run_evolution(cfg)` (in `src/evolve/run.py`) is the canonical entry point and is what launches a real run.

## Section-aware compilation modes

Defined in `src/organisms/implementation_patch.py` and orchestrated in `src/evolve/generator.py::_prepare_implementation_stage`.

- **FULL** mode: the artifact carries every canonical region. Used for seeds and any organism whose maternal base is missing.
- **PATCH** mode: the artifact carries only the regions in `CHANGED_SECTIONS`; the framework keeps the maternal implementation byte-for-byte for unchanged regions.

A guard added in P5 promotes PATCH to FULL whenever `len(changed_sections) >= len(implementation_regions) - 1`. PATCH at near-full coverage was the worst of both worlds: same token cost as FULL plus more parser fragility.

## Recent changes (the P-fixes)

| Fix | Where                                                                                                                                                                  | Effect |
|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| P1  | `conf/experiments/awtf2025_heuristic/prompts/implementation/system.txt`, `implementation/user.txt`; mock-route hint extractor in `api_platforms/_core/providers.py`     | Implementation prompt rewritten to lead with one canonical artifact example; user prompt no longer shows the scaffold-form markers (replaced with a numbered region-order list and a pre-defined-locals block) |
| P2  | `conf/experiments/awtf2025_heuristic/prompts/islands/staged_routing_repair.txt`                                                                                          | Island rewritten from a brittle "plan-then-repair" framing (0.9% success in the prior run) to a per-round greedy direction-profile bundling style |
| P3  | `conf/evolver/awtf2025_heuristic.yaml` `creation.*`                                                                                                                      | Retry budgets cut from `(3, 2, 2, 3)` to `(2, 2, 1, 1)` to stop multiplying wasted LLM calls on doomed lineages |
| P4  | `conf/evolver/awtf2025_heuristic.yaml` `reproduction.operator_weights`                                                                                                   | Mutation 0.2 → 0.5; within-island crossover 0.7 → 0.4; inter-island 0.2 → 0.1. Reduces the novelty-rejection peak driven by crossover-toward-primary-parent collapse |
| P5  | `src/evolve/generator.py::_prepare_implementation_stage`                                                                                                                 | When changed_sections covers all regions (or all but one), snap PATCH → FULL and drop the maternal-base requirement |
| P6  | `src/evolve/generator.py::_append_rejected_candidate_repair_block`, wired into both `_run_creation_stages_with_novelty` and `_run_creation_stages_with_validation`       | After a novelty/compatibility rejection, the retry user prompt now appends the rejected candidate plus the critique with explicit "patch this, do not redesign" framing |

## Common pitfalls (and where to look)

- **`failed_creation` with "Implementation patch regions must match the expected changed regions exactly"** — usually means the LLM emitted a region count or order that doesn't match `CHANGED_SECTIONS`. With P5 in place, FULL mode handles the all-changed case automatically.
- **`failed_creation` with "Unexpected ## END_REGION at line N"** — the LLM closed a region with `## END_REGION: NAME` (decorated end marker). The current implementation prompt forbids this explicitly with a concrete bad-example callout; if it still happens, check whether the prompt was loaded from the right path.
- **Novelty rejection storm in middle generations** — a sign that the population converged. Lower the crossover share or raise mutation weight in `reproduction.operator_weights`; consider raising `gene_removal_probability` to force more sectional variation.
- **An entire island stops producing successful organisms** — inspect that island's seed prompt for over-strict requirements (the prior `staged_routing_repair` had six "must specify" pillars; the rewrite cut it to two paragraphs of design intent).
- **Hydra-compose changes silently break tests** — many invariants are pinned in `tests/test_section_aware_family_contracts.py`, `tests/test_awtf2025_heuristic.py`, and `tests/test_hydra_compose.py`. When changing prompts or budgets in one family, search those tests for the family name and update both sides together.
- **Ollama appears to hang** — see `~/.claude/projects/.../memory/project_ollama_gpu.md`: `ollama serve` on the H100 node has reported `total_vram=0B` and silently fallen back to CPU for `qwen3.5:35b`. Check the stderr log before assuming the framework is stuck.
- **Long LLM stages look silent** — `_announce()` in `evolution_loop.py` and `generator.py` writes flushed stderr lines for each stage transition. If you don't see them, Hydra is probably swallowing your log output; the stderr prints are the recovery channel.

## Files worth knowing by heart

- `src/evolve/run.py` — public entrypoint, wires `EvolutionLoop`
- `src/evolve/evolution_loop.py` — generation lifecycle, parent sampling integration, allocation, resume
- `src/evolve/generator.py` — design / validation / implementation / repair stage loops; the heart of the per-organism pipeline
- `src/evolve/scoring.py` — phase score aggregation
- `src/evolve/selection.py` — parent sampling strategies
- `src/evolve/orchestrator.py` — eval scheduling on CPU/GPU pools
- `src/organisms/genetic_code_format.py` — genome parser
- `src/organisms/implementation_patch.py` — implementation artifact parser + scaffold assembler
- `src/organisms/compatibility.py`, `novelty.py`, `mutation.py`, `crossbreeding.py`, `operators.py` — operator and validator contracts
- `conf/evolver/<family>.yaml` — per-family budgets, weights, prompts, llm
- `conf/experiments/<family>/prompts/` — every LLM-facing prompt for that family
- `dump_llm.py` — post-mortem dump of one generation directory
