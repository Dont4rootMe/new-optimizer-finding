# Follow-Up Instruction for Closing the Remaining Post-Audit Gaps

## Purpose

This document is the next execution instruction after:

- `agents/8_gpt_current_state_reaudit_and_gap_analysis.md`
- `agents/9_gpt_agent_instruction_for_closing_all_remaining_gaps.md`

The repository is in a much better state now than it was during the `8.md` audit. The canonical organism-first path exists, prompt assets were moved into `conf`, island support exists, resume uses a manifest, lineage is richer, and the test surface is materially stronger.

However, the claim that there are no residual risks left inside the `8.md` scope is not correct.

There are still several real issues. Most of them are no longer broad architectural failures. They are narrower, but they are still important because they affect:

- mathematical correctness of the reward estimator;
- strictness and truthfulness of the canonical runtime;
- isolation between canonical and legacy paths;
- silent fallback behavior that can mask corruption or misconfiguration;
- lingering duplicate or ambiguous APIs that keep the codebase semantically messier than the target architecture.

This file tells the next agent exactly what still has to be finished.

This is not a fresh audit.

This is an execution instruction for the remaining work only.

Do not reopen already-settled decisions unless a change below explicitly says so.

## Current Locked Facts

The following are already considered correct and must remain in place:

1. `mode=evolve` must run the canonical organism-first multi-generation loop.
2. Legacy candidate-first evolution must remain available only through an explicit legacy entrypoint.
3. Canonical prompt assets live in `conf/prompts/*`.
4. Island descriptions live in `conf/islands/*.txt`.
5. Canonical mutation and crossover logic live in `src/organisms/mutation.py` and `src/organisms/crossbreeding.py`.
6. Canonical organism validation must require structured sections and reject thin or malformed `CORE_GENES`.
7. Lineage must store maternal-path history, with optional father metadata and cross-island annotations.
8. The canonical optimizer contract remains `build_optimizer(model, max_steps)` plus `step(weights, grads, activations, step_fn)` and `zero_grad(set_to_none=True)`.
9. The reward formula stays baseline-relative and uses the F1-like aggregation already implemented in `src/evolve/metrics_adapter.py`.
10. Great Filter remains a separate hard-phase selection step and must not overwrite the meaning of `simple_reward`.

Do not undo any of the above.

## What Is Still Open

There are five remaining workstreams.

They are not equivalent in severity.

The order below is mandatory.

### Priority Order

1. P0: fix the Neyman allocation plus Horvitz-Thompson mismatch.
2. P1: remove silent legacy config fallbacks from the canonical loop.
3. P1: remove silent legacy artifact fallbacks from canonical restore and canonical reads.
4. P1: collapse the remaining duplicate canonical-looking generation path.
5. P1: fully isolate the legacy candidate generator from canonical prompt-bundle dependencies.
6. P2: final cleanup of ambiguous aliases, dead helpers, and naming drift.

If you do not finish P0 and the P1 items above, the repository is still not “done”.

## Non-Negotiable Rules

1. Do not reintroduce any implicit fallback from canonical mode into legacy candidate-first mode.
2. Do not silently keep mathematically inconsistent allocation logic.
3. Do not keep canonical runtime behavior that accepts legacy config shapes without an explicit legacy or migration surface.
4. Do not keep canonical resume behavior that silently downgrades into best-effort recovery when required canonical artifacts are missing.
5. Do not leave two mutation/crossover generation paths that both look canonical.
6. Do not remove explicit legacy mode altogether.
7. Do not change the reward formula itself. Fix the sampling and aggregation semantics around it.
8. Do not make the implementation “pass tests” by weakening validations. If tests fail, update the implementation and then rewrite tests to prove the stricter semantics.

## Workstream 1: Fix the Canonical Neyman Allocation and Aggregate Estimator

### Why This Is Still Open

The current canonical allocation stack is still mathematically inconsistent.

The problem is the interaction between:

- `src/evolve/allocation.py`
- `src/evolve/scoring.py`

The current implementation does this:

1. compute Neyman weights;
2. convert them into Poisson-style inclusion probabilities `q_i = min(1, m * w_i)`;
3. sample each experiment independently with probability `q_i`;
4. if the sample is empty, force-pick the max-`q_i` experiment;
5. aggregate with a Horvitz-Thompson-style estimator using the original `q_i`.

Step 4 breaks the design assumed in step 5.

Once you force-pick an experiment after an empty draw, the true first-order inclusion probabilities are no longer the raw `q_i`.

That means the estimator in `mean_score()` is no longer using the actual design probabilities.

So the current code cannot honestly claim unbiased subset evaluation in the sense required by the project specification.

This is not a style issue.

This is a real mathematical bug.

### Required Outcome

The canonical allocation and scoring path must use one coherent design from end to end.

The exact design to implement is:

`conditional Poisson sampling conditioned on non-empty sample`

Do not use the current “sample independently, and if empty then choose argmax” rule.

### Exact Design to Implement

For a phase with experiments `E = {1, ..., N}`:

1. Compute Neyman weights `w_i`.
2. Compute base inclusion probabilities:

```text
q_i = min(1, m * w_i)
```

where `m = sample_size`.

3. Sample experiments by repeated independent Bernoulli draws with probabilities `q_i` until the sampled set is non-empty.

This is rejection sampling from the Poisson design conditioned on non-empty outcome.

4. Compute:

```text
p_nonempty = 1 - ∏_i (1 - q_i)
```

5. Use the conditional first-order inclusion probabilities:

```text
q_i_cond = q_i / p_nonempty
```

This formula is valid because the event “experiment `i` is selected” implies the event “sample is non-empty”.

6. Use the Horvitz-Thompson-style estimator with `q_i_cond`, not with `q_i`.

7. Persist both the base and effective probabilities in the allocation snapshot so the design is auditable.

### Exact Changes

#### `src/evolve/allocation.py`

Make the following changes.

1. Replace the current empty-sample fallback behavior.

The existing `sample_experiments_poisson()` must no longer do:

- one independent Bernoulli pass;
- then “if empty, choose argmax(inclusion_prob)”.

2. Implement conditional Poisson sampling with deterministic rejection.

The simplest acceptable implementation is:

- create `rng = random.Random(seed)`;
- repeatedly draw the subset with the same `rng` object;
- stop at the first non-empty draw;
- return that non-empty subset.

Do not reseed on each retry.

The draw sequence must remain deterministic for a fixed `seed`.

3. Add an explicit helper to compute `p_nonempty`.

Suggested helper:

```python
def compute_nonempty_probability(inclusion_prob: dict[str, float]) -> float:
    ...
```

4. Add an explicit helper to compute conditional inclusion probabilities.

Suggested helper:

```python
def compute_conditional_inclusion_probabilities(
    inclusion_prob: dict[str, float],
    nonempty_probability: float,
) -> dict[str, float]:
    ...
```

5. Change the allocation snapshot payload.

The canonical allocation snapshot must contain these fields:

```json
{
  "method": "neyman",
  "enabled": true,
  "sample_size": 2,
  "sampling_design": "conditional_poisson_nonempty",
  "weights": {...},
  "base_inclusion_prob": {...},
  "prob_nonempty": 0.73,
  "inclusion_prob": {...},
  "stats": {...},
  "selected_experiments": [...]
}
```

Rules:

- `base_inclusion_prob` stores the raw `q_i`.
- `inclusion_prob` stores the conditional `q_i_cond`.
- `sampling_design` must be explicit and stable.
- `prob_nonempty` must be stored for auditability.

6. If `enabled` is false or the allocation degenerates to full evaluation, set:

- `sampling_design = "full_evaluation"`
- `base_inclusion_prob = {exp: 1.0}`
- `inclusion_prob = {exp: 1.0}`
- `prob_nonempty = 1.0`
- `selected_experiments = all experiments`

Do not leave the design ambiguous.

#### `src/evolve/scoring.py`

1. Update the docstring and logic to explicitly state that canonical subset scoring uses the effective first-order inclusion probabilities from the conditional non-empty design.

2. Continue using the Horvitz-Thompson-style aggregate:

```text
mean_hat = (1 / N) * Σ_{i in selected} y_i / π_i
```

where `π_i` is now the effective conditional inclusion probability stored in `allocation_snapshot["inclusion_prob"]`.

3. Do not read `base_inclusion_prob` inside `mean_score()`.

4. If a selected experiment lacks a positive `π_i`, raise a hard error.

That is a design corruption bug, not a soft warning.

5. Keep the existing status behavior:

- `ok` if all selected experiments succeeded;
- `partial` if some selected experiments failed;
- `failed` if none of the selected experiments produced usable scores.

#### `src/evolve/orchestrator.py`

1. No semantic rewrite is needed beyond using the new allocation snapshot fields consistently.

2. Keep passing the effective `allocation_snapshot["inclusion_prob"]` to `mean_score()`.

3. Update comments and any naming that still implies the raw Poisson-with-fallback design.

### What You Must Not Do

1. Do not keep the current argmax fallback and just rename the field names.
2. Do not keep the current argmax fallback and claim the design is “approximately unbiased”.
3. Do not switch to a different estimator without updating the design semantics and tests.
4. Do not quietly change from subset evaluation to full evaluation for small sample sizes.

### Tests Required

Add or update tests in:

- `tests/test_neyman_allocation.py`
- `tests/test_scoring.py`

Add at least these cases.

1. A deterministic test that the conditional sampler never returns an empty set.
2. A test that `prob_nonempty` is computed correctly for a small hand-checkable probability map.
3. A test that `inclusion_prob` equals `base_inclusion_prob / prob_nonempty`.
4. A test that the scoring function uses the effective conditional probabilities.
5. A regression test proving the old argmax-fallback semantics are gone.

### Acceptance Criteria

1. There is no code path in canonical allocation that force-picks argmax after an empty draw.
2. The snapshot explicitly records `sampling_design`, `base_inclusion_prob`, `prob_nonempty`, and effective `inclusion_prob`.
3. The aggregate estimator is consistent with the actual sampling design.
4. The code no longer overclaims unbiasedness while using inconsistent probabilities.

## Workstream 2: Remove Silent Legacy Config Fallbacks from the Canonical Evolution Loop

### Why This Is Still Open

`mode=evolve` is now the canonical organism-first runtime.

But `src/evolve/evolution_loop.py` still silently reads legacy config shapes in multiple places.

This means the repository still has a hidden second config schema for canonical mode.

That is not acceptable for the final target architecture.

The canonical loop must consume only the canonical config layout documented in `conf/evolver/default.yaml`.

Legacy config handling must live only in explicit legacy mode or explicit migration tooling.

### The Current Residual Fallthroughs

These legacy fallbacks are still present in the canonical loop:

1. fallback to `legacy_flat_population_island()` when `evolver.islands.dir` is missing;
2. fallback from `evolver.islands.organisms_per_island` to `evolution.population_size`;
3. fallback from `operators.mutation.probability` to `evolution.mutation_rate`;
4. fallback from `operators.mutation.gene_delete_probability` to `evolution.mutation_q`;
5. fallback from `operators.crossover.inherit_gene_probability_from_mother` to `evolution.crossover_p`;
6. fallback from canonical phase config into `evaluation.simple_experiments`, `evaluation.hard_experiments`, `simple_allocation`, `hard_allocation`, `elite_count`, `great_filter_interval`;
7. fallback from `evolver.max_generations` to `evolution.max_generations`;
8. fallback from phase timeout into top-level `evolver.timeout_sec_per_eval`.

These fallbacks make canonical mode less strict than the documentation now claims.

### Required Outcome

`EvolutionLoop` must read exactly one config schema: the canonical one.

If the canonical fields are missing, it must fail fast with a clear error that names the missing config path.

### Exact Changes

#### `src/evolve/evolution_loop.py`

Make the following rules true.

1. `_load_islands()`

Required behavior:

- require `cfg.evolver.islands.dir`;
- call `load_islands()` on that directory;
- if the field is missing or empty, raise `ValueError` or `FileNotFoundError`;
- do not fallback to `legacy_flat_population_island()`.

2. `_organisms_per_island()`

Required behavior:

- require `cfg.evolver.islands.organisms_per_island`;
- if missing or invalid, raise.

3. `_inter_island_crossover_rate()`

Required behavior:

- read only `cfg.evolver.islands.inter_island_crossover_rate`;
- keep default `0.1` only if that default is explicitly documented as canonical behavior in config composition;
- do not fallback to any legacy field.

4. `_mutation_probability()`

Required behavior:

- read only `cfg.evolver.operators.mutation.probability`;
- do not read `evolution.mutation_rate`.

5. `_gene_delete_probability()`

Required behavior:

- read only `cfg.evolver.operators.mutation.gene_delete_probability`;
- do not read `evolution.mutation_q`.

6. `_inherit_gene_probability()`

Required behavior:

- read only `cfg.evolver.operators.crossover.inherit_gene_probability_from_mother`;
- do not read `evolution.crossover_p`.

7. `_softmax_temperature()`

Required behavior:

- read only `cfg.evolver.operators.crossover.softmax_temperature`;
- do not introduce any legacy fallback.

8. `_phase_cfg()`

Required behavior:

- read only `cfg.evolver.phases.simple` and `cfg.evolver.phases.great_filter`;
- if the requested phase block is missing, raise;
- do not build compatibility payloads from `evaluation.*` and `evolution.*`.

9. `_phase_eval_mode()` and `_phase_timeout_sec()`

Required behavior:

- read only per-phase fields;
- do not fallback to top-level `evolver.timeout_sec_per_eval`.

10. `run()`

Required behavior:

- read only `cfg.evolver.max_generations`;
- do not fallback to `evolution.max_generations`.

### Required Documentation Update

Update any docs or comments that still imply canonical mode accepts both old and new config shapes.

Canonical mode must be strict.

Legacy mode remains explicit.

Compatibility must not be implicit.

### Tests Required

Add or update tests in:

- `tests/test_evolution_loop_semantics.py`
- `tests/test_run_evolution.py`
- `tests/test_hydra_compose.py`

Add at least these cases.

1. Canonical loop fails if `evolver.islands.dir` is missing.
2. Canonical loop fails if canonical operator probabilities are missing even if `evolution.mutation_rate` or `evolution.crossover_p` are present.
3. Canonical loop fails if phase blocks are missing even if legacy `evaluation.*` keys are present.
4. Explicit legacy mode still works through `run_legacy_single_generation()`.

### Acceptance Criteria

1. `mode=evolve` consumes only canonical config keys.
2. Legacy config fields are not silently accepted by the canonical loop.
3. The default Hydra config still composes cleanly because canonical fields are present.
4. The only surviving reader of legacy candidate config shape is the explicit legacy candidate path.

## Workstream 3: Make Canonical Resume and Canonical Artifact Loading Strict

### Why This Is Still Open

The repository now has a canonical manifest-based resume model.

But the canonical restore path still has silent compatibility behavior that can mask corruption.

Examples:

1. if the manifest is missing, `_restore_population_from_manifest()` falls back to scanning generation directories;
2. `read_genetic_code()` silently falls back to `idea_dna.txt`;
3. `read_lineage()` silently falls back to `evolution_log.json`;
4. `read_genetic_code()` returns empty canonical genetic code if the canonical file is missing;
5. `read_organism_meta()` always uses these permissive readers.

This means canonical resume can succeed even when canonical artifacts are damaged or incomplete.

That is the wrong failure mode.

A canonical run must fail loudly on missing canonical artifacts.

Compatibility with older artifacts must be explicit, not silent.

### Required Outcome

There must be a strict canonical read path and, if needed, a separate explicit legacy-compatible migration path.

Canonical resume must use the strict path.

### Exact Changes

#### `src/evolve/storage.py`

Refactor the readers so that strictness is explicit.

Implement one of these two acceptable designs.

Design A:

- `read_genetic_code(path, *, allow_legacy_fallback: bool = False, allow_empty: bool = False)`
- `read_lineage(path, *, allow_legacy_fallback: bool = False)`
- `read_organism_meta(path, *, allow_legacy_artifacts: bool = False)`

Design B:

- `read_canonical_genetic_code(path)`
- `read_legacy_compatible_genetic_code(path)`
- `read_canonical_lineage(path)`
- `read_legacy_compatible_lineage(path)`
- `read_canonical_organism_meta(path)`
- `read_legacy_compatible_organism_meta(path)`

Either design is acceptable.

The behavior must be this:

1. Canonical readers:

- require `genetic_code.md`;
- require `lineage.json`;
- reject missing files;
- reject malformed files;
- never silently read `idea_dna.txt`;
- never silently read `evolution_log.json`;
- never silently return empty genes for a missing file.

2. Legacy-compatible readers:

- may read old artifact names;
- may log warnings;
- must be used only in explicit migration or explicit legacy compatibility tools.

3. Manifest restore:

- must use the canonical strict reader.

4. `load_population()`:

- may remain as a compatibility helper if explicitly marked as non-canonical;
- must not be used as automatic fallback during canonical resume.

#### `src/evolve/evolution_loop.py`

Change `_restore_population_from_manifest()`:

1. If the manifest is missing, raise a hard error.
2. If an `organism_dir` from the manifest is missing, raise a hard error.
3. If the organism metadata exists but canonical organism artifacts are missing or malformed, raise a hard error.
4. Remove the current fallback to `load_population()` from the canonical resume flow.

### Optional but Recommended Explicit Migration Tool

If you want to preserve one-shot migration support, add a dedicated helper instead of keeping silent runtime fallback.

A small helper module is enough.

Suggested location:

- `src/evolve/migration.py`

Suggested functions:

```python
def load_legacy_population_for_migration(...): ...
def migrate_legacy_artifacts_to_canonical(...): ...
def build_manifest_from_explicit_population(...): ...
```

This is optional only in form.

The key requirement is not optional:

canonical resume itself must no longer auto-downgrade into compatibility mode.

### Tests Required

Add or update tests in:

- `tests/test_evolution_resume.py`
- `tests/test_organism_contract.py`

Add at least these cases.

1. Missing manifest causes canonical resume to fail.
2. Manifest pointing to a missing organism directory causes canonical resume to fail.
3. Missing `genetic_code.md` for a canonical organism causes strict read failure.
4. Missing `lineage.json` for a canonical organism causes strict read failure.
5. If you keep a legacy-compatible reader, test it separately and label it explicitly as migration compatibility.

### Acceptance Criteria

1. Canonical resume uses manifest-only restore.
2. Canonical readers do not silently consume legacy or empty artifacts.
3. Corrupted canonical artifacts fail loud instead of degrading into empty state.
4. Legacy compatibility, if kept, is explicit and isolated.

## Workstream 4: Collapse the Remaining Duplicate Canonical-Looking Generation Path

### Why This Is Still Open

The real canonical generation flow is now:

- seed via `OptimizerGenerator`
- mutation via `src/organisms/mutation.py`
- crossover via `src/organisms/crossbreeding.py`

But the codebase still contains a second canonical-looking prompt-generation abstraction in:

- `src/evolve/operators.py`
- `src/evolve/generator.py`

`src/evolve/operators.py` still defines prompt-only `MutationOperator` and `CrossoverOperator`.

`OptimizerGenerator.generate_organism()` still presents itself as a generic organism generator driven by `GeneticOperator`.

In practice, canonical mutation and canonical crossover do not use that path anymore.

So the repository still exposes two mutation/crossover generation models:

1. the real one in `src/organisms/*`;
2. a leftover prompt-only abstraction that still looks like canonical runtime API.

That is source-of-truth drift.

### Required Outcome

The canonical codebase must expose exactly one mutation path and one crossover path.

The prompt-only mutation/crossover abstraction must stop looking canonical.

### Exact Changes

#### `src/evolve/operators.py`

1. Keep only the seed-related prompt abstraction.

You have two acceptable choices.

Choice A:

- keep `SeedOperator`;
- remove prompt-only `MutationOperator`;
- remove prompt-only `CrossoverOperator`;
- remove `OPERATORS` entries for mutation and crossover.

Choice B:

- rename the file to make its role explicit, for example `seed_operator.py`;
- keep only seed-related functionality there.

Do not leave prompt-only mutation and crossover classes in a canonical module.

#### `src/evolve/generator.py`

1. Replace the generic method:

```python
generate_organism(operator, parents, organism_id, generation, organism_dir)
```

with a seed-specific method, for example:

```python
generate_seed_organism(seed_operator, organism_id, generation, organism_dir)
```

or:

```python
generate_seed_organism(island, organism_id, generation, organism_dir)
```

2. Remove generic parent-based logic from the generator that only exists to support the no-longer-canonical prompt-only mutation/crossover path.

3. Remove the fallback:

```python
island_id = "legacy_flat_population"
```

from the generic generation logic.

A seed organism always belongs to a real configured island.

4. Keep the generator responsible for:

- loading prompt bundle;
- calling the LLM;
- parsing the structured response;
- validating rendered code;
- building seed organisms.

Do not let it continue to look like a second full reproduction engine.

#### `src/evolve/evolution_loop.py`

Update the seed flow to call the new seed-specific generator method.

No mutation/crossover logic should route through `src/evolve/operators.py`.

### Tests Required

Add or update tests in:

- `tests/test_optimizer_generator.py`
- `tests/test_run_evolution.py`

Add at least these cases.

1. The canonical generator still seeds organisms correctly.
2. There is no surviving canonical prompt-only mutation/crossover API.
3. Seed generation requires a real island identity and does not fallback to `legacy_flat_population`.

### Acceptance Criteria

1. The canonical codebase has exactly one mutation implementation and one crossover implementation.
2. `src/evolve/operators.py` no longer contains mutation/crossover classes that look canonical.
3. `OptimizerGenerator` is seed-only for canonical runtime.

## Workstream 5: Fully Isolate the Legacy Candidate Generator from Canonical Prompt-Bundle Dependencies

### Why This Is Still Open

The explicit legacy path was mostly quarantined, but one coupling remains.

`LegacyCandidateGenerator` still subclasses `OptimizerGenerator`.

Because of that, its constructor currently inherits canonical initialization behavior that loads the canonical prompt bundle from `conf/prompts/*`.

That means legacy candidate mode is not actually fully isolated from canonical prompt assets.

If canonical prompt files were missing but the raw legacy prompt files still existed, the legacy generator could still fail before reading its own prompt templates.

That is the wrong dependency direction.

### Required Outcome

Explicit legacy mode must not require canonical prompt assets at all.

### Exact Changes

#### `src/evolve/legacy_generator.py`

Refactor `LegacyCandidateGenerator` so it no longer depends on canonical prompt-bundle initialization.

The simplest correct solution is:

1. stop subclassing `OptimizerGenerator`;
2. extract only the genuinely shared LLM client or code-validation pieces into a smaller reusable helper or base class.

Acceptable shared pieces:

- model name resolution;
- OpenAI call helper;
- Python code extraction helper;
- code validation helper.

Unacceptable inherited dependency:

- canonical prompt bundle loading.

Suggested patterns:

Pattern A:

- add `BaseLlmGenerator` with provider/model/openai utilities;
- let `OptimizerGenerator` and `LegacyCandidateGenerator` both inherit from it.

Pattern B:

- move shared LLM utility functions into a helper module;
- use composition instead of inheritance.

Either is fine.

The only hard requirement is:

constructing `LegacyCandidateGenerator` must not read canonical prompt assets.

#### `tests/test_optimizer_generator.py`

Add the reciprocal test that does not currently exist.

The existing test already proves:

- canonical generator must not read legacy raw candidate prompt files.

Add the mirror assertion:

- legacy generator must not read canonical organism prompt bundle files.

You can implement it with a `Path.read_text` monkeypatch that raises if one of:

- `system_project_context.txt`
- `seed_system.txt`
- `seed_user.txt`
- `mutation_system.txt`
- `mutation_user.txt`
- `crossover_system.txt`
- `crossover_user.txt`

is accessed during `LegacyCandidateGenerator(...)` construction.

### Acceptance Criteria

1. Explicit legacy generator initialization succeeds without touching canonical prompt assets.
2. Canonical generator initialization succeeds without touching legacy raw candidate prompt assets.
3. Canonical and legacy generators now depend only on the assets they actually need.

## Workstream 6: Final Cleanup of Ambiguous Aliases, Dead Helpers, and Naming Drift

### Why This Still Matters

These are not the biggest open issues.

But they still matter because the user asked for all remaining bugs and unfinished spots to be closed, not just the largest ones.

### Required Cleanup Items

#### 6.1 Remove `OrganismMeta.score` Alias from Canonical Runtime Surfaces

`src/evolve/types.py` still exposes:

- `score`
- `simple_score`
- `hard_score`

as compatibility aliases on `OrganismMeta`.

The canonical runtime should use:

- `selection_reward`
- `simple_reward`
- `hard_reward`

only.

Do this:

1. remove the alias properties from `OrganismMeta`;
2. keep backward-compatibility only in serialization or old-payload coercion if needed;
3. stop using `score_key="score"` in canonical or quasi-canonical tests.

If a legacy-specific helper still needs old field names, isolate it there.

#### 6.2 Move or Delete Dead Legacy Selection Helpers

`src/evolve/selection.py` still exports:

- `tournament_select`
- `elite_select`
- `select_parents_for_reproduction`

These are legacy candidate-era helpers and are not used by the canonical organism-first loop.

Do one of the following.

Preferred:

- move them into `src/evolve/legacy_selection.py`;
- keep tests there;
- keep `src/evolve/selection.py` canonical.

Acceptable:

- delete them if truly unused.

Do not keep them in the canonical selection module.

#### 6.3 Remove the Outdated `BuildOptimizerCallable` Alias

`valopt/utils/import_utils.py` still defines:

```python
BuildOptimizerCallable = Callable[[Any], OptimizerControllerProtocol]
```

That signature is wrong relative to the current optimizer contract.

Fix it by either:

1. deleting the alias if unused;
2. or changing it to match `OptimizerBuilder`.

Preferred:

- delete it if it is unused.

#### 6.4 Rename the `aggregate_score` Parameter in Lineage Backfill

`src/organisms/organism.py` still uses:

```python
update_latest_lineage_entry(..., aggregate_score=...)
```

But the lineage entry now stores phase-specific scores, not an aggregate lineage-wide score field.

Rename the parameter to something truthful, for example:

- `phase_score`

Update call sites and tests.

This is a naming cleanup, but it removes a misleading semantic leftover.

#### 6.5 Remove Remaining Canonical Runtime Imports of `legacy_flat_population`

After the config and resume cleanup above, the canonical evolution loop should no longer import or instantiate `legacy_flat_population_island()`.

The concept may remain in storage or explicit migration compatibility helpers.

It must not remain part of the canonical loop’s normal control flow.

### Tests Required

Add or update tests in:

- `tests/test_selection.py`
- `tests/test_optimizer_generator.py`
- any affected type or storage tests

Required coverage:

1. no canonical test uses `.score` alias;
2. legacy selection helpers, if retained, are tested from an explicitly legacy module;
3. no canonical module depends on `legacy_flat_population_island()` for normal execution;
4. no outdated builder alias remains.

### Acceptance Criteria

1. Canonical modules expose canonical names only.
2. Legacy compatibility, if kept, is explicit and located in legacy-labeled modules.
3. There is no misleading leftover naming around `aggregate_score` inside lineage backfill.

## Recommended Implementation Order

Follow this order exactly.

### Step 1

Fix the conditional Poisson plus effective inclusion probability design.

Do not postpone this.

It is the most important correctness issue still left open.

### Step 2

Make canonical config reading strict in `EvolutionLoop`.

Do this before storage strictness so the runtime becomes truthfully canonical at the config layer.

### Step 3

Make canonical resume and artifact reads strict.

Remove automatic manifest and artifact fallback behavior from canonical resume.

### Step 4

Collapse the duplicate canonical-looking generation path.

Make seeding explicit and keep mutation/crossover only in `src/organisms/*`.

### Step 5

Decouple `LegacyCandidateGenerator` from `OptimizerGenerator`.

This is important to make the isolation boundary truthful, but it depends less on the previous changes.

### Step 6

Do the final alias and dead-helper cleanup.

### Step 7

Run the full test suite and add the new negative tests described above.

## Grep-Based Sanity Checks

After implementation, run grep-style checks to verify the cleanup really happened.

The following are the expected outcomes.

### Check 1

`src/evolve/evolution_loop.py` must no longer reference:

- `evolution.mutation_rate`
- `evolution.mutation_q`
- `evolution.crossover_p`
- `evaluation.simple_experiments`
- `evaluation.hard_experiments`
- `legacy_flat_population_island`
- `load_population(` as resume fallback

### Check 2

`src/evolve/selection.py` must not contain:

- `tournament_select`
- `elite_select`
- `select_parents_for_reproduction`

unless the file is explicitly redefined as legacy-only, which is not the preferred outcome.

### Check 3

`src/evolve/operators.py` must not contain prompt-only mutation or crossover classes.

### Check 4

`src/evolve/legacy_generator.py` must not import or initialize canonical prompt-bundle loading.

### Check 5

Canonical storage reads must not silently fallback from:

- `genetic_code.md` to `idea_dna.txt`
- `lineage.json` to `evolution_log.json`

inside the normal canonical resume path.

## Final Acceptance Checklist

All of the following must be true before you declare the work finished.

1. Canonical subset evaluation uses a coherent sampling design and a mathematically consistent estimator.
2. No canonical runtime path auto-selects or silently consumes legacy config layout.
3. No canonical resume path silently consumes legacy or missing artifacts.
4. Canonical mutation and crossover each exist in exactly one runtime implementation.
5. Legacy candidate mode no longer depends on canonical prompt assets.
6. Ambiguous aliases and dead legacy helpers are removed from canonical surfaces.
7. `pytest -q` passes.
8. New negative tests exist for strict canonical config and strict canonical resume.
9. New tests exist for the fixed allocation math.
10. Documentation and comments no longer overstate completion or isolation that is not actually true.

## What the Final Repository Should Feel Like

After these changes, the repository should have this property:

If a user runs canonical evolve mode, they are using one strict, truthful, island-aware organism-first architecture.

If a user runs legacy candidate mode, they are using one explicit quarantined legacy architecture.

There should be no hidden bridges between them except the ones intentionally exposed for explicit migration or explicit legacy mode.

The codebase is not done until that statement is true without caveats.
