# Инструкция для агента: полностью закрыть все оставшиеся баги и недоделки после `8.md`

## 0. Назначение документа

Этот файл является не аудитом и не brainstorming.

Это instruction/spec для агента-исполнителя.

Его задача:

- взять текущий репозиторий в состоянии после `8.md`;
- довести систему до semantic closure;
- исправить все remaining bugs and unfinished areas, которые уже установлены;
- не принимать архитектурных решений на ходу;
- не плодить новые transitional branches;
- не оставлять “почти готово”.

Этот документ должен быть достаточен сам по себе.

Агент, который его читает, не должен придумывать:

- что считать canonical;
- какие баги править первыми;
- что удалять, а что сохранять;
- какие конфиг-ключи являются правдой;
- какие тесты должны доказать завершение работы.

Если агенту нужна более подробная доказательная база, он может читать:

- `agents/8_gpt_current_state_reaudit_and_gap_analysis.md`
- `agents/7_gpt_master_execution_spec_for_optimizer_discovery_platform.md`
- `agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md`

Но implementer не имеет права переоткрывать уже принятые решения.

Этот документ фиксирует решения.

## 1. Главная миссия

Ты должен довести репозиторий до состояния, где:

- canonical organism-first island-aware system является единственным mainline runtime для `mode=evolve`;
- legacy candidate-first machinery перестает влиять на canonical behavior;
- reward semantics математически и продуктово соответствуют заявленной модели;
- prompt/source-of-truth migration реально завершена;
- selection semantics и topology semantics правдивы;
- config keys больше не лгут;
- tests доказывают нужные инварианты, а не просто локальные механики;
- README и runtime рассказывают одну и ту же историю.

Важно:

- задача не “сделать чуть лучше”;
- задача не “поправить пару файлов”;
- задача не “оставить совместимость на всякий случай везде”;
- задача именно закрыть все remaining problem areas, установленные в `8.md`.

## 2. Непересматриваемые решения

Ниже решения locked.

Не обсуждать.

Не переоценивать.

Не изобретать альтернативы.

### 2.1 Canonical architecture

Canonical architecture:

- organism-first;
- island-aware;
- multi-generation;
- manifest-driven;
- fixed-template constrained;
- baseline-relative reward driven;
- LLM structured response driven.

Canonical architecture не candidate-first.

### 2.2 Canonical runtime entrypoint

Для `mode=evolve` canonical runtime entrypoint:

- multi-generation `EvolutionLoop`.

Automatic fallback на legacy single-generation path больше не считается допустимым default behavior.

### 2.3 Canonical generation model

Canonical generation model:

- seed organism from island prompt;
- mutate organism via gene deletion plus structured LLM rewrite;
- crossbreed via maternal-biased inherited pool plus structured LLM rewrite.

Raw “return only Python code” candidate generation не является canonical generation model.

### 2.4 Canonical prompt ownership

Canonical prompt assets live only in:

- `conf/prompts/*`

`src/evolve/prompts/*` не может оставаться source of truth для canonical path.

### 2.5 Canonical island ownership

Island descriptions live only in:

- `conf/islands/*.txt`

### 2.6 Canonical organism artifacts

Canonical organism artifacts are:

- `optimizer.py`
- `genetic_code.md`
- `lineage.json`
- `organism.json`
- `summary.json`
- `llm_request.json`
- `llm_response.json`
- `results/simple/*.json`
- `results/hard/*.json`
- `logs/*`

### 2.7 Canonical active-population state

Canonical active population state lives in:

- `population_manifest.json`
- `evolution_state.json`

Generation scan is compatibility helper only.

### 2.8 Canonical optimizer contract

Canonical optimizer contract:

- `build_optimizer(model, max_steps)`
- controller `__init__(self, model, max_steps)`
- `step(self, weights, grads, activations, step_fn)`
- `zero_grad(self, set_to_none=True)`

Old `build_optimizer(cfg)` contract must not be silently supported anywhere in canonical path.

### 2.9 Canonical reward semantics

Canonical reward semantics:

- objective is `train_loss`;
- direction is `min`;
- quality term uses baseline-relative last loss;
- steps term uses baseline-relative step-to-threshold behavior;
- experiment score is F1-like combination of quality and steps terms;
- simple and hard phases are distinct phases;
- simple reward stays as persistent property of organism;
- hard phase influences reproductive selection through hard-specific reward.

### 2.10 Canonical sampling semantics

Canonical sampling semantics:

- mutation parent sampling is uniform;
- crossover mother sampling is softmax over current selection scalar;
- crossover father sampling is chosen from local or foreign pool according to one-shot inter-island decision;
- `parent_sampling` is not a runtime-configurable knob anymore.

### 2.11 Canonical config philosophy

If a key exists in canonical config:

- runtime must actually use it;
- tests must prove it;
- docs must describe it honestly.

If runtime does not use a key:

- remove it;
- or mark it as deprecated and unsupported;
- but do not leave it as implied active control.

## 3. Что ты обязан считать уже установленными фактами

Не трать время на переоценку этих фактов.

Они уже установлены.

### 3.1

`conf/prompts/*` and `conf/islands/*` already exist.

### 3.2

`EvolutionLoop` already exists and is the right structural base.

### 3.3

`population_manifest.json` already exists and is the right control-plane idea.

### 3.4

`genetic_code.md` and `lineage.json` already exist and are the right artifact direction.

### 3.5

`simple_reward`, `hard_reward`, `selection_reward` split already exists.

### 3.6

The remaining work is not to redesign from scratch.

The remaining work is to finish, harden, and cleanly close the migration.

## 4. Главные дефекты, которые ты обязан устранить

Этот список обязателен.

Если хотя бы один пункт остается нерешенным, задача считается незавершенной.

### 4.1 Critical bug: inter-island crossover probability is wrong

Current bug:

- the father-pool selection in `EvolutionLoop` uses two independent random draws;
- configured `inter_island_crossover_rate=r` behaves like approximately `r^2` when local and foreign pools exist.

You must:

- replace this logic with a one-shot inter-island decision;
- guarantee that configured rate is the actual foreign-pool selection probability when foreign pools are available.

### 4.2 Critical bug: mutation-vs-crossover split is not config-truthful

Current issue:

- runtime looks for `operators.mutation.probability`;
- canonical default config does not define it;
- system silently defaults to `0.5`.

You must:

- add explicit canonical operator-mix config;
- use it in runtime;
- test it.

### 4.3 Critical bug: allocation aggregate estimator is semantically wrong for the stated goal

Current issue:

- subset aggregation under Neyman allocation does not currently preserve the semantics “expected aggregate equals full-experiment mean”.

You must:

- replace current estimator and sampling semantics with a mathematically explicit unbiased estimator for the declared target;
- test it;
- document it.

### 4.4 Critical semantic gap: Great Filter is not guaranteed to be hard in runtime regime

Current issue:

- hard phase uses hard experiment list;
- but still inherits global `eval_mode=smoke` by default.

You must:

- introduce phase-specific eval regime;
- make Great Filter default to `full`;
- prove it by tests.

### 4.5 Critical architecture gap: canonical path still depends on residual legacy layers

Current issues:

- legacy prompt files in `src/evolve/prompts/*` remain active for parts of generation;
- candidate-oriented generator/orchestrator logic still shares too much surface with canonical path;
- canonical allocation history is polluted by candidate summaries.

You must:

- isolate legacy candidate mode so it no longer affects canonical organism-first behavior;
- or fully retire it from default runtime path.

### 4.6 Critical test gap: canonical path is still under-proven

You must:

- add tests that directly prove canonical organism-first behavior;
- not just legacy helpers and candidate mode.

## 5. Что ты не имеешь права делать

Запреты.

Жесткие.

### 5.1

Не добавляй новые product features до закрытия remaining gaps.

### 5.2

Не оставляй новые config keys without runtime usage.

### 5.3

Не расширяй legacy candidate mode новыми возможностями.

### 5.4

Не оставляй dual source of truth for prompts.

### 5.5

Не оставляй автоматический fallback на legacy runtime path для `mode=evolve`.

### 5.6

Не объявляй “совместимость” без четко ограниченной области действия.

### 5.7

Не меняй reward math произвольно.

Ты должен реализовать конкретную corrected semantics, описанную ниже.

### 5.8

Не оставляй tests, которые продолжают утверждать ложные legacy semantics как mainline behavior.

## 6. Конечное состояние, которого ты обязан добиться

После завершения твоей работы должны быть истинны все пункты ниже.

### 6.1 Runtime

- `mode=evolve` always runs canonical organism-first evolution loop.
- No automatic fallback to legacy candidate generation path.
- Great Filter runs with phase-specific hard evaluation mode.

### 6.2 Prompts

- Canonical organism generation uses only `conf/prompts/*`.
- No canonical code path requires `src/evolve/prompts/optimizer_system.txt`.
- No canonical code path requires `src/evolve/prompts/optimizer_user.txt`.

### 6.3 Islands

- Configured inter-island crossover rate equals actual runtime behavior.
- Island membership remains maternal for child placement.
- Cross-island ancestry is inspectable in organism artifacts.

### 6.4 Organisms

- Canonical organism generation rejects malformed/too-thin genetic code.
- `IDEA_DNA` is not accepted as normal canonical organism response format.
- `gene_diff_summary` reflects actual semantic change, not only operator stats.

### 6.5 Reward

- Simple phase uses simple experiments only.
- Great Filter uses hard experiments only.
- Great Filter default eval mode is `full`.
- `simple_reward` persists after hard phase.
- `hard_reward` is separate.
- `selection_reward` is clearly defined and documented.
- Allocation estimator is mathematically explicit and tested against intended semantics.

### 6.6 Config

- No active canonical config key is ignored.
- Unused keys removed or explicitly deprecated.
- Legacy experiment normalization fossils removed from canonical configs/tests/docs.

### 6.7 Storage

- Manifest validation is strict enough to catch malformed active population state.
- Canonical history for allocation comes from canonical organism summaries only.
- Legacy readers remain only where explicitly necessary for migration.

### 6.8 Tests

- Canonical organism operators are directly tested.
- Inter-island probability bug is covered by tests.
- Great Filter phase semantics are covered by tests.
- Allocation estimator semantics are covered by tests.
- Prompt ownership closure is covered by tests.

### 6.9 Docs

- README matches actual runtime behavior.
- Legacy mode, if still retained, is explicitly labeled legacy.

## 7. Implementation order

Выполнять только в этом порядке.

Не перепрыгивать через фазы.

Каждая фаза должна оставлять репозиторий в рабочем состоянии.

## 8. Phase 0: freeze canonical and legacy boundaries

### Goal

Сделать архитектурную границу между canonical organism-first system и legacy candidate-first system жесткой и явной.

### Required changes

1. `mode=evolve` must always route to canonical `EvolutionLoop`.
2. Automatic fallback in `src/evolve/run.py` must be removed.
3. Legacy single-generation candidate path must become explicit legacy-only entry.
4. Legacy candidate code must no longer be part of canonical runtime graph.

### Exact decisions

- `run_evolution()` always constructs `EvolutionLoop`.
- `run_single_generation()` may remain only as explicit legacy helper.
- If legacy single-generation mode is retained, it must be invoked only by an explicitly named legacy entrypoint or explicit flag, not by implicit config shape detection.

### File-level intent

- `src/evolve/run.py`
- `src/evolve/orchestrator.py`
- `src/evolve/generator.py`

### Required structural refactor

You must split generation concerns cleanly:

- canonical structured organism generation must live in canonical generator surface;
- legacy raw candidate generation must be isolated behind explicit legacy naming.

The cleanest acceptable result is:

- canonical organism generation stays in a canonical generator module/class;
- legacy raw-code candidate generation moves to explicitly legacy surface;
- `EvolutionLoop` does not import or depend on legacy prompt files or raw candidate prompt templates.

### Mandatory naming rule

Any retained legacy entity must include `legacy` in one of:

- module name;
- class name;
- function name;
- docstring header.

Do not leave ambiguous names for non-canonical code.

### Acceptance criteria

- `mode=evolve` cannot accidentally run candidate-first path.
- `EvolutionLoop` imports only canonical generation/evaluation surfaces.
- legacy candidate mode, if retained, is clearly labeled and not auto-selected.

### Tests required

- test that `run_evolution()` always chooses `EvolutionLoop`;
- test that legacy path requires explicit call and is not activated by missing multigen config.

## 9. Phase 1: make prompt ownership single-source and final

### Goal

Finish prompt migration so canonical organism-first path has exactly one prompt source of truth.

### Required changes

1. Canonical organism path must read only `conf/prompts/*`.
2. No canonical constructor may require legacy prompt files in `src/evolve/prompts/*`.
3. Legacy prompt files may remain only for explicit legacy candidate mode.
4. Canonical generator initialization must succeed without legacy candidate prompt assets.

### Exact decisions

- `PromptBundle` remains canonical.
- `system_project_context.txt`, `seed_system.txt`, `seed_user.txt`, `mutation_system.txt`, `mutation_user.txt`, `crossover_system.txt`, `crossover_user.txt` remain the canonical prompt set.
- `optimizer_system.txt` and `optimizer_user.txt` are not part of canonical organism generation anymore.

### Required code changes

- Remove any canonical-path dependency on `self.prompts_dir / "optimizer_system.txt"` and `self.prompts_dir / "optimizer_user.txt"`.
- If raw candidate generation remains, move those reads into legacy-only code.
- Ensure canonical organism generator builds prompts exclusively through `PromptBundle`.

### Prompt ownership rule

For canonical path:

- prompt text source = `conf/prompts/*`
- prompt composition = `PromptBundle` + `compose_system_prompt`
- no second source

### Acceptance criteria

- deleting or renaming legacy raw candidate prompt files does not break canonical organism evolution tests;
- canonical organism generation still works;
- prompt tests explicitly prove canonical ownership closure.

### Tests required

- canonical generator test that monkeypatches or removes legacy candidate prompt assets and still passes;
- prompt ownership test for organism generation path.

## 10. Phase 2: fix operator mix and selection config truthfulness

### Goal

Make runtime selection semantics exactly match declared semantics and remove lying config fields.

### Required changes

1. Add explicit canonical config for mutation-vs-crossover split.
2. Use it in runtime.
3. Remove or deprecate unused `parent_sampling` config keys.
4. Remove or deprecate any old `selection_strategy` semantics from canonical evolve path.
5. Add runtime validation for `top_h_per_island <= top_k_per_island`.

### Exact config decisions

Canonical operator config must be:

```yaml
operators:
  mutation:
    probability: 0.5
    gene_delete_probability: 0.2
  crossover:
    inherit_gene_probability_from_mother: 0.7
    softmax_temperature: 1.0
```

Not allowed in canonical config:

- `operators.mutation.parent_sampling`
- `operators.crossover.parent_sampling`
- `selection_strategy`

These are removed, not merely ignored.

### Exact runtime decisions

- mutation parent selection is always uniform by design;
- crossover mother selection is always softmax over `selection_reward`;
- father selection is from local or foreign pool based on one-shot island decision;
- these are product semantics, not runtime-configurable strategy names.

### Acceptance criteria

- `default.yaml` contains `operators.mutation.probability`;
- canonical runtime uses it;
- old strategy keys are gone from canonical config, README, and tests;
- top-h/top-k invalid config is rejected early.

### Tests required

- test that high mutation probability produces mutation-only reproduction in a controlled fixture;
- test that low mutation probability permits crossover when enough parents exist;
- config validation test for invalid `top_h_per_island > top_k_per_island`.

## 11. Phase 3: fix inter-island crossover probability bug

### Goal

Make `inter_island_crossover_rate` mathematically equal to actual runtime behavior.

### Exact runtime behavior

When local and foreign father pools are both available:

- draw exactly one Bernoulli with probability `inter_island_crossover_rate`;
- if draw succeeds, choose father pool from foreign islands;
- if draw fails, choose father pool from local island;
- if selected pool is empty, fall back to the non-empty pool;
- if both empty, crossover cannot proceed.

When only local pool exists:

- use local pool.

When only foreign pools exist:

- use foreign pool.

### Required code changes

- rewrite `_select_father_pool()` accordingly;
- do not use two independent random draws for one crossover routing decision.

### Required artifact changes

When child is cross-island:

- capture this in metadata available to lineage/summary.

Exact required fields:

- `father_island_id`
- `cross_island`

These fields must exist at least in lineage entry or organism summary.

### Acceptance criteria

- for a deterministic seeded simulation with both local and foreign pools available, empirical foreign-pool selection frequency converges near configured rate;
- lineage or summary makes cross-island event explicit.

### Tests required

- statistical selection test using many draws with seeded RNG;
- test that child remains on maternal island while recording father island.

## 12. Phase 4: harden genetic code contract and structured response validation

### Goal

Make canonical organism generation reject structurally weak or outdated LLM outputs.

### Exact output contract

Canonical organism response must contain all of:

- `## CORE_GENES`
- `## INTERACTION_NOTES`
- `## COMPUTE_NOTES`
- `## CHANGE_DESCRIPTION`
- `## IMPORTS`
- `## INIT_BODY`
- `## STEP_BODY`
- `## ZERO_GRAD_BODY`

### Exact validation rules

The canonical validator must reject response if any of the following is true:

- `CORE_GENES` has fewer than 3 non-empty bullet lines;
- any core gene contains fewer than 2 words;
- `INTERACTION_NOTES` is empty after strip;
- `COMPUTE_NOTES` is empty after strip;
- `CHANGE_DESCRIPTION` is empty after strip;
- any required code section is missing;
- rendered code fails existing template validation.

### Compatibility rule

- `IDEA_DNA` fallback is not accepted for canonical organism generation.
- If legacy candidate mode still uses old format, it must stay inside legacy-only code.

### Exact storage rule

`genetic_code.md` remains canonical.

Its canonical section model stays:

- `CORE_GENES`
- `INTERACTION_NOTES`
- `COMPUTE_NOTES`

### Exact `gene_diff_summary` rule

For every mutation or crossover child:

- `gene_diff_summary` must be generated from actual parent-vs-child gene delta;
- it must mention added and removed conceptual items when such items exist.

Minimum required semantics:

- for mutation: mention removed genes and newly introduced/rewritten genes;
- for crossover: mention maternal genes preserved, paternal genes introduced, and any major rewrites.

### Acceptance criteria

- malformed structured response triggers retry/failure, not silent weak organism creation;
- canonical organism generation no longer depends on `IDEA_DNA`;
- lineage diff summary is semantic, not only numeric operator statistics.

### Tests required

- test that organism build rejects too-thin `CORE_GENES`;
- test that missing notes or sections trigger failure;
- test that `IDEA_DNA`-only response is rejected in canonical path;
- test that `gene_diff_summary` contains actual added/removed gene text.

## 13. Phase 5: simplify lineage semantics and make prompt history more useful

### Goal

Remove remaining ambiguity from lineage and improve prompt usefulness.

### Exact schema decision

Canonical `LineageEntry` must contain:

- `generation`
- `operator`
- `mother_id`
- `father_id`
- `change_description`
- `gene_diff_summary`
- `selected_simple_experiments`
- `selected_hard_experiments`
- `simple_score`
- `hard_score`
- `cross_island`
- `father_island_id`

Canonical `LineageEntry` must not write `aggregate_score` anymore.

Compatibility rule:

- old entries with `aggregate_score` may be read;
- new writes must not include it.

### Prompt rendering decision

`format_lineage_summary()` must output:

- one line per recent lineage entry;
- generation;
- operator;
- short change description;
- gene diff summary;
- simple score if available;
- hard score if available;
- cross-island flag when true.

It must not output generic unused aggregate field.

### Acceptance criteria

- no new lineage entries contain `aggregate_score`;
- recent prompt summary is materially more informative;
- cross-island history is visible.

### Tests required

- test that new lineage write omits `aggregate_score`;
- test that old lineage with `aggregate_score` can still load;
- test that prompt summary includes cross-island info when present.

## 14. Phase 6: replace current allocation math with an explicit unbiased estimator

### Goal

Make subset evaluation under allocation mathematically honest with respect to the stated target: estimate the full-experiment mean of `exp_score`.

### Canonical sampling decision

You must replace the current fixed-size weighted-without-replacement subset approximation with Poisson-style Neyman sampling with known inclusion probabilities.

Use this exact algorithm:

1. Compute Neyman weights:
   - `w_i ∝ std_i / sqrt(cost_i)`
2. Normalize weights so `sum(w_i)=1`.
3. Given target `sample_size = m` and `N = len(experiments)`, compute inclusion probabilities:
   - `q_i = min(1.0, m * w_i)`
4. Sample each experiment independently with probability `q_i` using deterministic RNG seeded from global seed plus organism id.
5. If zero experiments are selected, select exactly one experiment with largest `q_i`.
6. Store both:
   - normalized Neyman weights
   - actual inclusion probabilities `q_i`

### Canonical aggregate estimator

The aggregate over selected experiments must be:

```text
aggregate = (1 / N) * sum_{i in selected} (exp_score_i / q_i)
```

Where:

- `N` is total number of experiments in the phase list;
- `q_i` is the actual inclusion probability used in sampling;
- experiments not selected contribute zero;
- failed selected experiments still count as zero contribution because no successful `exp_score_i` exists.

### Status rule

- `ok` means all selected experiments succeeded;
- `partial` means at least one selected experiment failed but at least one succeeded;
- `failed` means no selected experiment produced usable score.

### Rationale

This is the locked resolution of the current ambiguity.

Do not invent another estimator.

Do not keep the old `sum(pi * score)/sum(pi)` semantics.

### Required snapshot shape

Allocation snapshot must contain:

- `method`
- `enabled`
- `history_window`
- `sample_size`
- `weights`
- `inclusion_prob`
- `stats`
- `selected_experiments`

`pi` as ambiguous field name must be removed from canonical path.

Compatibility rule:

- read old `pi` if legacy summary is loaded;
- new canonical writes must use `weights` and `inclusion_prob`.

### Required history-source decision

For canonical organism-first evaluation:

- allocation history must be read from organism summaries only.

Legacy candidate history must not influence canonical organism allocation.

If legacy candidate mode is kept:

- it may maintain its own separate history reader.

### Acceptance criteria

- aggregate semantics are explicitly documented in code and tests;
- canonical allocation snapshot no longer uses misleading `pi` name;
- organism-first history is organism-only.

### Tests required

- unit test for inclusion probability computation;
- unit test for deterministic Poisson sampling behavior with seed;
- unit test for Horvitz-Thompson style aggregate formula;
- integration test that canonical organism allocation history ignores candidate summaries.

## 15. Phase 7: make Great Filter truly hard and phase-specific

### Goal

Make hard phase product semantics true at runtime, not just by experiment names.

### Exact config decision

Canonical phase config must support:

```yaml
phases:
  simple:
    eval_mode: smoke
    timeout_sec_per_eval: 7200
    top_k_per_island: ...
    experiments: [...]
    allocation: ...
  great_filter:
    enabled: true
    interval_generations: 5
    eval_mode: full
    timeout_sec_per_eval: 7200
    top_h_per_island: ...
    experiments: [...]
    allocation:
      enabled: false
```

Rules:

- if phase-specific `eval_mode` absent, simple defaults to `smoke`, great_filter defaults to `full`;
- if phase-specific `timeout_sec_per_eval` absent, fall back to global timeout.

### Exact request decision

`OrganismEvaluationRequest` must include:

- `eval_mode`
- `timeout_sec`

Canonical orchestrator must use request-level values, not global evolve default for all phases.

### Exact selection decision after hard phase

After hard evaluation:

- `hard_reward` is set from hard aggregate;
- `selection_reward` is set to `hard_reward` for that generation’s reproductive selection;
- `simple_reward` remains unchanged;
- summaries keep both phase result blocks.

### Acceptance criteria

- Great Filter defaults to full evaluation mode;
- simple phase and hard phase can be configured independently;
- tests prove that `simple_reward` survives and hard phase uses hard experiments only.

### Tests required

- test that simple and hard requests carry different `eval_mode`;
- test that Great Filter uses hard experiment list only;
- test that `simple_reward` is unchanged after hard phase;
- test that `selection_reward` switches to hard reward only for hard-selection semantics.

## 16. Phase 8: clean config truthfulness and remove fossils

### Goal

Make canonical config honest and small.

### Remove from canonical config/tests/docs

- `operators.mutation.parent_sampling`
- `operators.crossover.parent_sampling`
- `selection_strategy`
- `eval_experiments` from canonical organism-first evolve path
- static normalization fields `quality_ref`
- static normalization fields `steps_ref`

### Allowed compatibility behavior

- code may read old keys only in legacy mode or compatibility loaders;
- canonical config files and canonical tests must not use them.

### Exact experiment-config cleanup

You must remove `quality_ref` and `steps_ref` from experiment YAMLs if they are truly unused.

If you decide to retain them for compatibility, then:

- they must be under explicit deprecated block;
- canonical scoring path must not read them;
- tests must not present them as meaningful.

Preferred resolution:

- remove them entirely from experiment configs.

### Exact orchestrator cleanup

For canonical organism-first path:

- phase experiment lists come from `evolver.phases.simple.experiments` and `evolver.phases.great_filter.experiments`;
- no canonical code should need `evolver.eval_experiments`.

### Acceptance criteria

- canonical config files no longer advertise ignored knobs;
- tests no longer use removed fossil keys except in explicit legacy compatibility tests.

### Tests required

- config load test for canonical config without fossil keys;
- optional explicit legacy config compatibility test if legacy mode retained.

## 17. Phase 9: clean storage and history readers

### Goal

Make storage semantics canonical without breaking migration.

### Required changes

1. Split or clearly separate legacy candidate helpers from organism helpers.
2. Make organism-first history readers organism-only by default.
3. Strengthen manifest validation.
4. Preserve read-only compatibility for old artifacts with explicit warnings.

### Exact manifest validation rules

Manifest load must reject:

- missing `organism_id`;
- missing `organism_dir`;
- duplicate `organism_id`;
- duplicate `organism_dir`;
- non-int generation values;
- missing or non-string `island_id`.

Manifest load must warn but still recover for:

- generation mismatch between state file and manifest;
- legacy flat-pop island mapping when explicitly needed.

### Exact history-reader decision

Implement two different history readers:

- canonical organism history reader for organism-first evolution;
- legacy candidate history reader if legacy mode remains.

Do not keep one mixed reader for both worlds.

### Acceptance criteria

- canonical allocation cannot read candidate summaries anymore;
- manifest corruption is caught early;
- compatibility fallback paths are explicit.

### Tests required

- manifest duplicate-entry failure test;
- manifest malformed-entry failure test;
- organism-only history reader test;
- legacy-reader test only if legacy mode retained.

## 18. Phase 10: harden canonical operator coverage

### Goal

Make tests directly cover the code paths that actually matter in canonical evolution.

### You must add direct tests for

- `src/organisms/mutation.py::MutationOperator.produce`
- `src/organisms/crossbreeding.py::CrossbreedingOperator.produce`
- actual structured prompt generation for canonical path
- lineage persistence after canonical mutation and crossover
- semantic `gene_diff_summary`

### Required test style

Use mock LLM provider or controlled mock generator so tests remain fast and deterministic.

Do not rely on network.

### Exact coverage requirements

For mutation operator:

- surviving and removed genes appear in prompt;
- child stays on parent island;
- mother_id equals parent id;
- lineage appended;
- canonical genetic code persisted.

For crossover operator:

- maternal-biased inherited pool used;
- child island equals mother island;
- mother_id and father_id written;
- maternal lineage inherited;
- father island recorded for cross-island cases.

### Acceptance criteria

- canonical organism operators are directly tested end-to-end with artifact writes and lineage assertions.

## 19. Phase 11: rebalance the integration test surface

### Goal

Make the main integration story organism-first, not candidate-first.

### Required changes

1. Add at least one canonical organism-first integration test that runs:
   - seeding
   - simple evaluation
   - selection
   - manifest write
2. Add at least one multi-generation organism-first test with:
   - resume
   - reproduction
   - per-island boundaries
3. Downgrade candidate-first integration tests from “main pipeline proof” to explicit legacy coverage.

### Exact narrative rule

After your changes:

- the first integration test a maintainer sees must validate canonical organism-first evolution;
- candidate integration test, if retained, must be labeled legacy in name/docstring.

### Required test additions

- organism-first fake-eval integration test;
- hard-phase integration test;
- inter-island crossover integration test;
- prompt ownership integration test.

### Acceptance criteria

- test suite’s center of gravity shifts to canonical path.

## 20. Phase 12: finish README and developer truthfulness

### Goal

Bring docs into exact alignment with runtime after cleanup.

### Required README changes

README must explicitly state:

- `mode=evolve` runs organism-first multi-generation evolution loop;
- prompt assets live in `conf/prompts` for canonical path;
- island descriptions live in `conf/islands`;
- Great Filter defaults to hard/full evaluation mode;
- active population restore is manifest-driven;
- legacy candidate mode, if retained, is legacy only and not canonical.

### Required doc cleanup

Do not claim:

- prompt migration complete if canonical path still depends on legacy prompt files;
- hard phase is hard if still using smoke mode;
- config keys are supported if they were removed.

### Acceptance criteria

- README describes current truth, not intended future.

## 21. Exact code changes by subsystem

This section is intentionally direct.

It tells you what outcome each subsystem must reach.

### 21.1 `src/evolve/run.py`

Must end up with:

- canonical `run_evolution()` always uses `EvolutionLoop`;
- no config-shape auto-fallback;
- legacy run, if retained, is explicitly named and not default.

### 21.2 `src/evolve/evolution_loop.py`

Must end up with:

- explicit `operators.mutation.probability` usage;
- one-shot inter-island routing;
- phase-specific eval settings passed to evaluation requests;
- top-h/top-k validation;
- no reliance on unused strategy keys.

### 21.3 `src/evolve/orchestrator.py`

Must end up with:

- clean organism evaluation seam using request-level eval mode and timeout;
- canonical allocation semantics using inclusion probabilities and unbiased aggregate estimator;
- no canonical organism history contamination from candidate summaries.

### 21.4 `src/evolve/allocation.py`

Must end up with:

- Neyman weights;
- inclusion probabilities;
- Poisson-style deterministic sampling;
- no misleading `pi` as aggregate estimator input field.

### 21.5 `src/evolve/scoring.py`

Must end up with:

- full-mean estimator over selected subset using inclusion probabilities;
- explicit documentation of estimator semantics;
- no old weighted-by-raw-probabilities aggregate.

### 21.6 `src/evolve/prompt_utils.py`

Must remain canonical prompt loader.

If fallback behavior retained:

- make it explicit deprecation-only behavior;
- canonical organism tests must not rely on it.

### 21.7 `src/evolve/generator.py`

Must no longer sit ambiguously across two architectures.

Required end state:

- canonical organism generation path uses only canonical prompt bundle and structured response logic;
- raw candidate generation logic is isolated into explicit legacy surface or removed from canonical runtime graph.

### 21.8 `src/organisms/organism.py`

Must end up with:

- strict structured-response validation;
- no canonical `IDEA_DNA` fallback;
- stronger genetic-code validation;
- lineage write without `aggregate_score`;
- better semantic lineage summary formatting.

### 21.9 `src/organisms/mutation.py`

Must end up with:

- semantic `gene_diff_summary`;
- preserved canonical lineage;
- canonical artifact persistence;
- direct tests.

### 21.10 `src/organisms/crossbreeding.py`

Must end up with:

- semantic `gene_diff_summary`;
- explicit cross-island lineage metadata when relevant;
- preserved maternal island;
- direct tests.

### 21.11 `src/evolve/storage.py`

Must end up with:

- strict manifest validation;
- organism-only canonical history reader;
- legacy candidate history reader separated or clearly labeled;
- migration fallbacks explicit and quarantined.

### 21.12 `conf/evolver/default.yaml`

Must end up with:

- explicit `operators.mutation.probability`;
- no fake parent_sampling keys;
- phase-specific `eval_mode`;
- phase-specific optional timeout;
- canonical phases-only experiment ownership.

### 21.13 `conf/experiments/*.yaml`

Must end up with:

- no stale `quality_ref` / `steps_ref` in canonical configs.

### 21.14 `README.md`

Must end up matching actual runtime truth exactly.

### 21.15 `tests/*`

Must end up proving canonical path, not only legacy residue.

## 22. Required test inventory

You are not done until the following tests exist and pass.

### 22.1 Contract tests

- strict `build_optimizer(model, max_steps)` import acceptance;
- strict legacy builder rejection;
- strict missing `step_fn` rejection.

### 22.2 Prompt ownership tests

- canonical organism generator does not require legacy raw candidate prompt files;
- prompt bundle loads from `conf/prompts`.

### 22.3 Island tests

- island loader;
- inter-island probability behavior;
- maternal island preservation;
- cross-island lineage metadata.

### 22.4 Mutation tests

- gene deletion helper;
- canonical mutation operator end-to-end.

### 22.5 Crossover tests

- maternal-biased pool helper;
- canonical crossover operator end-to-end.

### 22.6 Reward math tests

- inclusion probability calculation;
- aggregate estimator formula;
- never-reached-baseline behavior;
- missing baseline failure behavior.

### 22.7 Phase semantics tests

- simple phase sets `simple_reward`;
- hard phase sets `hard_reward`;
- `simple_reward` persists after hard phase;
- Great Filter uses hard experiment list only;
- phase-specific eval mode used.

### 22.8 Resume and manifest tests

- restore older-generation survivors from manifest;
- reject malformed manifest;
- reject duplicate manifest entries.

### 22.9 History-source tests

- canonical organism history reader ignores candidate summaries;
- legacy candidate history reader only used in explicit legacy mode if retained.

### 22.10 Integration tests

- canonical organism-first evolve integration test;
- canonical organism-first multi-generation or resume test;
- candidate integration test labeled legacy if still present.

## 23. Explicit legacy policy

You must choose one of two allowed outcomes.

You are not allowed to choose a third ambiguous middle state.

### Allowed outcome A: full retirement

- delete legacy candidate-first generation mode from canonical runtime and tests;
- keep only minimal read compatibility for old artifacts if needed.

### Allowed outcome B: explicit quarantine

- keep legacy candidate-first mode only behind explicit legacy entrypoints/modules/tests;
- remove all influence on canonical path;
- label everything legacy in names and docs.

### Locked recommendation

Preferred outcome is:

- explicit quarantine now;
- optional later deletion.

Why:

- it closes semantic ambiguity now;
- it is less risky than full deletion in one pass;
- it preserves ability to inspect old artifacts.

## 24. What “done” means

You are done only if all of the following are true.

### 24.1 Runtime truth

- no auto-fallback to legacy evolve path;
- Great Filter runs with phase-specific hard mode by default;
- inter-island rate bug fixed.

### 24.2 Math truth

- canonical allocation estimator is explicit, unbiased for the declared target, and tested.

### 24.3 Config truth

- no canonical key lies about controlling runtime;
- no unused fossil keys remain in canonical config.

### 24.4 Artifact truth

- canonical organism generation writes strict rich genetic code and clean lineage entries;
- cross-island events are inspectable.

### 24.5 Ownership truth

- canonical path uses only canonical prompt assets;
- canonical history readers use only canonical organism artifacts.

### 24.6 Test truth

- canonical path is the best-tested path in the repo;
- all new tests pass;
- existing relevant tests pass after cleanup.

### 24.7 Documentation truth

- README matches actual behavior after cleanup.

## 25. Final delivery requirements for the implementing agent

Когда ты закончишь реализацию, ты обязан:

1. перечислить все ключевые architectural changes;
2. перечислить удаленные или deprecated config keys;
3. отдельно сказать, retained ли legacy candidate mode и как именно он теперь изолирован;
4. дать точный список новых тестов;
5. показать итоговый `pytest` result;
6. отдельно перечислить residual risks, если хоть что-то сознательно не доведено.

Если residual risks остались:

- задача не считается полностью завершенной;
- ты обязан прямо сказать, что именно осталось и почему.

## 26. Short execution checklist

Используй это как финальную проверку перед закрытием работы.

- [ ] `mode=evolve` always runs canonical organism-first loop
- [ ] legacy candidate path is quarantined or retired
- [ ] canonical path depends only on `conf/prompts`
- [ ] explicit mutation probability config exists and is used
- [ ] `parent_sampling` config lies removed
- [ ] inter-island crossover probability bug fixed
- [ ] cross-island ancestry recorded
- [ ] canonical structured response validation strengthened
- [ ] no canonical `IDEA_DNA` fallback
- [ ] `aggregate_score` no longer written in new lineage entries
- [ ] phase-specific `eval_mode` exists
- [ ] Great Filter defaults to `full`
- [ ] `simple_reward` persistence tested
- [ ] allocation estimator replaced with explicit unbiased version
- [ ] canonical allocation history is organism-only
- [ ] stale `quality_ref` / `steps_ref` removed from canonical config/tests
- [ ] manifest validation strengthened
- [ ] canonical mutation operator directly tested
- [ ] canonical crossover operator directly tested
- [ ] organism-first integration test exists
- [ ] README matches current runtime truth

## 27. Final instruction

Не пытайся “минимально поправить, чтобы тесты стали зелеными”.

Твоя задача другая.

Ты должен:

- закрыть все remaining semantic gaps;
- сделать canonical path truly canonical;
- закончить migration truthfully;
- и оставить после себя репозиторий, по которому уже не придется писать `10.md` с разбором тех же самых проблем под новыми именами.
