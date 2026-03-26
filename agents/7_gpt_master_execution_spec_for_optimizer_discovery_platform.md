# Master Execution Spec: переход репозитория к canonical island-based optimizer discovery platform

## 1. Назначение документа

Этот файл не является еще одним аудитом состояния репозитория. Для этого уже существует [agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md](/Users/artemon/Library/Mobile Documents/com~apple~CloudDocs/Programming/python_projects/new-optimizer-finding/agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md).

Разделение ролей документов такое:

- `6.md` отвечает на вопрос: что сейчас сделано, что сломано, где расхождения, почему это проблема, какие есть риски.
- `7.md` отвечает на вопрос: что именно нужно делать по этапам, в каком порядке, какими изменениями, какими схемами, с какими тестами и критериями завершения.

Этот документ должен работать как:

- handoff-документ для другого агента;
- master-plan на несколько последовательных implementation sessions;
- техническая инструкция, по которой можно дробить работу на отдельные задачи без дополнительных архитектурных решений;
- canonical source of execution order, пока система приводится к целевой модели.

Ключевой принцип: implementer, читающий только этот файл, не должен придумывать модель системы заново. Все ключевые decisions в этом файле уже зафиксированы.

## 2. Как пользоваться этим документом

### Если ты новый агент в проекте

Сначала прочитай:

1. Этот файл целиком.
2. Затем при необходимости обратись к `6.md` только как к доказательной базе и расширенному audit-контексту.

Твоя задача после чтения `7.md`:

- понять canonical target state;
- выбрать текущую фазу;
- выполнить изменения строго в пределах этой фазы;
- не открывать архитектурные развилки заново, если они уже locked в этом документе.

### Если ты разбиваешь работу на отдельные задачи

Разбивка должна происходить по фазам и workstreams из этого файла, а не по произвольным кускам кода.

Нельзя:

- одновременно развивать несколько конкурирующих архитектур;
- брать подзадачу, которая зависит от незавершенной предыдущей фазы;
- реализовывать optional improvements раньше, чем закрыты architectural blockers.

### Если ты используешь документ как execution runbook

В каждой фазе ориентируйся на:

- `Goal`
- `Preconditions`
- `Exact changes`
- `Tests required`
- `Acceptance criteria`
- `Definition of done`

Только после выполнения acceptance criteria фаза считается завершенной.

## 3. Статус этого файла относительно других артефактов

Этот файл является:

- canonical execution-spec;
- operational source of truth для порядка реализации;
- основным handoff-документом для implementation work.

Этот файл не является:

- историческим логом;
- общим brainstorming-документом;
- местом для повторного обсуждения уже принятых решений;
- местом, где допустимо оставлять открытые архитектурные развилки.

Связь с другими документами:

- `4_train_loss_baseline_reward_followup.md` — historical context про reward rework.
- `5_*` — предыдущие аудиты и отчеты.
- `6_gpt_revision_merged_comprehensive_audit_and_roadmap.md` — основной audit, upstream diagnosis.
- `7.md` — canonical implementation playbook.

## 4. Locked decisions

Ниже перечислены решения, которые считаются принятыми. В ходе реализации их не нужно повторно обсуждать, если только не появится отдельное явное указание на пересмотр архитектуры.

### 4.1 Canonical architecture = organism-first

Принятое решение:

- canonical эволюционная сущность системы — `Organism`, а не `Candidate`.
- organism-state является главным носителем:
  - genetic code;
  - lineage;
  - island membership;
  - rewards;
  - reproduction eligibility.

Что отвергнуто:

- `candidate-first` как главная модель системы;
- поддержание `cand_*` и `org_*` как равноправных long-term архитектур.

Что временно допустимо:

- migration bridge, при котором `EvolverOrchestrator` еще умеет работать с candidate-like артефактами, пока evolution stack не переведен полностью.

### 4.2 `EvolverOrchestrator` остается evaluation backend

Принятое решение:

- `EvolverOrchestrator` не является second canonical evolution model.
- Он используется как backend для:
  - generation/evaluation jobs там, где это нужно;
  - orchestration of experiment runs;
  - allocation/execution plumbing.

Что отвергнуто:

- идея поддерживать long-term и organism-loop, и candidate-loop как две отдельные product-mode архитектуры.

Что временно допустимо:

- transitional adapters, которые позволяют organism-loop вызывать orchestrator до полного рефакторинга seam API.

### 4.3 Baseline-relative reward на `train_loss` сохраняется

Принятое решение:

- canonical reward objective = `train_loss`;
- baseline-relative quality/speed terms и harmonic mean сохраняются;
- `objective_name` в reward path должен быть `train_loss`;
- scoring semantics не переписываются в этой migration, кроме phase/selection integration.

Что отвергнуто:

- возврат к `quality_ref` / `steps_ref` как primary scoring source;
- simultaneous support of multiple incompatible reward systems.

Что временно допустимо:

- legacy fields в payload’ах, если они не влияют на canonical scoring path.

### 4.4 Fixed optimizer template сохраняется

Принятое решение:

- optimizer template остается fixed contract;
- LLM не получает право переписывать сигнатуру и outer structure произвольным образом;
- editable regions остаются явно выделенными.

Что отвергнуто:

- переход к unconstrained whole-file generation как canonical mode.

Что временно допустимо:

- coexistence legacy generator path, пока contract cleanup не завершен, но только как transitional behavior, не как target state.

### 4.5 Prompts переезжают в `conf`

Принятое решение:

- prompt texts являются конфигурационными и исследовательскими asset’ами;
- constant project/system context живет в `conf`;
- operator prompts живут в `conf/prompts`;
- school/island descriptions живут в `conf/islands`.

Что отвергнуто:

- хранение canonical prompt set внутри `src/evolve/prompts` как code-coupled permanent source of truth.

Что временно допустимо:

- compatibility fallback к старым prompt paths на ограниченный migration period, но только если он явно помечен как legacy.

### 4.6 Islands обязательны как часть эволюционной модели

Принятое решение:

- island model — не optional enhancement, а центральная часть search topology;
- initialization, selection, crossover policy и replenishment становятся island-aware;
- per-island selection — обязательный invariant.

Что отвергнуто:

- single flat population как конечная модель системы.

Что временно допустимо:

- transitional flat-to-island migration для существующих population artifacts.

### 4.7 Lineage является canonical research artifact

Принятое решение:

- lineage — это не декоративный лог;
- lineage хранит evolution history с backfilled scores;
- lineage участвует в prompt conditioning;
- maternal lineage semantics является canonical.

Что отвергнуто:

- merged lineage from both parents как long-term canonical behavior;
- endless unfinished entries with `score=None`.

### 4.8 Optimizer contract унифицируется вокруг нового runtime контракта

Принятое решение:

- canonical external optimizer contract:
  - `build_optimizer(model, max_steps)`
  - controller implements `step(weights, grads, activations, step_fn)`
  - controller implements `zero_grad(set_to_none=True)`
- именно этот контракт должен быть согласован в runtime, README, validator, prompts, tests и examples.

Что отвергнуто:

- старый контракт `build_optimizer(cfg)` / `initialize(...)` как равноправно поддерживаемый canonical mode.

Что временно допустимо:

- explicit legacy rejection paths с понятными ошибками, но не silent backward compatibility.

## 5. Что не подлежит пересмотру без отдельного решения

Ниже перечислено то, что implementer не должен менять по собственной инициативе.

- Reward math не меняется без отдельного решения.
- Fixed template не заменяется свободным codegen.
- Island model не откладывается “на потом” как optional stage.
- `cand_*` path не получает новые canonical capabilities.
- Legacy config knobs не должны silently жить после cleanup.
- README и tests не могут расходиться с runtime contract.
- `OrganismMeta` остается central domain entity после refactor.

## 6. Основные engineering rules

### 6.1 Один canonical путь

После завершения каждой фазы у системы должен быть один все более явный canonical execution path.

Запрещено:

- строить новый behavior одновременно в legacy `cand_*` и canonical `org_*` path;
- добавлять новые функции в обе competing architectures;
- оставлять ambiguous source of truth.

### 6.2 Migration должна быть explicit

Если фаза еще не полностью завершила migration, то нужно:

- явно пометить compatibility bridge;
- явно отметить deprecated path;
- иметь clear criteria, когда bridge удаляется.

### 6.3 Каждая фаза оставляет repo в рабочем состоянии

После каждой фазы должны быть:

- проходящие relevant tests;
- documented invariants;
- no silent broken migration state.

### 6.4 No fake config

Запрещено держать config key как будто он влияет на behavior, если runtime его игнорирует.

Для любого config field после cleanup допускаются только варианты:

- field actively used;
- field rejected with error;
- field accepted with explicit warning and deprecation note;
- field auto-migrated with explicit conversion.

### 6.5 No prompt/persistence contradictions

Если prompt разрешает LLM что-то менять, то canonical persistence path обязан это поддерживать.

Пример правила:

- если mutation prompt говорит, что модель может добавить новый gene, final DNA persistence обязана уметь принять этот gene.

### 6.6 No ambiguous score usage

Не допускается использовать одно поле `score` для нескольких семантически разных ролей без явного перевода в canonical field set.

Должны существовать разные поля для:

- simple reward;
- hard reward;
- selection reward;
- display/reporting aggregate.

## 7. Source-of-truth matrix

Этот раздел определяет, что считать canonical, что legacy, а что transitional.

### 7.1 Canonical now

- `agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md` — audit diagnosis.
- `agents/7_gpt_master_execution_spec_for_optimizer_discovery_platform.md` — execution spec.
- `valopt/*` reward/runtime baseline stack — canonical foundation for validation.
- `src/evolve/template_parser.py` and template-based rendering — canonical template mechanism.

### 7.2 Canonical target after migration

- organism-first evolution model;
- island-aware population state;
- `conf/prompts/*` and `conf/islands/*` as text assets;
- `population_manifest.json` as active population source for resume;
- unified optimizer contract.

### 7.3 Legacy

- candidate-first `cand_*` evolution outputs as main documented evolve model;
- old optimizer contract from README/tests;
- prompt set under `src/evolve/prompts` as canonical prompt source;
- static normalization refs as if they controlled current reward.

### 7.4 Transitional

- orchestrator adapters used by organism-loop;
- fallback reading of old prompt locations, if needed temporarily;
- reading old population layouts while writing new canonical manifests.

## 8. Canonical target state

Этот раздел описывает finished target system. Все фазы ниже существуют только для того, чтобы привести репозиторий к этому состоянию.

## 8.1 Target product definition

Итоговая система — это island-based optimizer discovery platform, в которой:

- каждая особь — это optimizer implementation + canonical genetic code + lineage;
- каждый остров соответствует отдельной школе/направлению исследования;
- стартовая популяция seed’ится по школам из `conf/islands/*.txt`;
- reproduction идет через:
  - mutation,
  - within-island crossover,
  - controlled inter-island crossover;
- simple phase присваивает постоянный simple reward;
- Great Filter оценивает по hard tasks и определяет reproductive eligibility;
- reward path baseline-relative и основан на `train_loss`;
- optimizer template fixed;
- lineage и genetic code являются first-class research artifacts.

## 8.2 Target filesystem layout

### 8.2.1 Config and prompt assets

```text
conf/
  config.yaml
  evolver/
    default.yaml
  prompts/
    system_project_context.txt
    seed_system.txt
    seed_user.txt
    mutation_system.txt
    mutation_user.txt
    crossover_system.txt
    crossover_user.txt
  islands/
    gradient_methods.txt
    second_order.txt
    adaptive_methods.txt
```

### 8.2.2 Population artifacts

```text
populations/
  evolution_state.json
  population_manifest.json
  gen_0000/
    island_gradient_methods/
      org_<id>/
        optimizer.py
        genetic_code.md
        lineage.json
        organism.json
        summary.json
        results/
          simple/
            <experiment>.json
          hard/
            <experiment>.json
        logs/
          simple_<experiment>.out
          simple_<experiment>.err
          hard_<experiment>.out
          hard_<experiment>.err
```

Notes:

- `gen_<N>` keeps creation-time grouping, not active-population truth.
- `population_manifest.json` stores active population regardless of where each organism was created.
- island nesting is explicit in filesystem, not inferred.

### 8.2.3 Baseline artifacts

```text
stats/
  <experiment>/
    baseline.json
```

Baseline path stays canonical and is not reworked in this migration.

## 8.3 Target config schema

Canonical target `evolver` shape:

```yaml
evolver:
  enabled: true
  generation: 0
  resume: true
  fail_fast: false

  prompts:
    project_context: conf/prompts/system_project_context.txt
    seed_system: conf/prompts/seed_system.txt
    seed_user: conf/prompts/seed_user.txt
    mutation_system: conf/prompts/mutation_system.txt
    mutation_user: conf/prompts/mutation_user.txt
    crossover_system: conf/prompts/crossover_system.txt
    crossover_user: conf/prompts/crossover_user.txt

  islands:
    dir: conf/islands
    organisms_per_island: 5
    inter_island_crossover_rate: 0.1

  operators:
    mutation:
      gene_delete_probability: 0.2
      parent_sampling: uniform
    crossover:
      inherit_gene_probability_from_mother: 0.7
      parent_sampling: softmax_reward
      softmax_temperature: 1.0

  phases:
    simple:
      top_k_per_island: 5
      experiments: [...]
      allocation:
        enabled: true
        method: neyman
        sample_size: 2
        history_window: 100
        min_history_for_variance: 3
        std_floor: 1.0e-6
        fallback: uniform
    great_filter:
      enabled: true
      interval_generations: 5
      top_h_per_island: 3
      experiments: [...]
      allocation:
        enabled: false

  llm:
    provider: chatgpt
    model: gpt-5.4-pro
    temperature: 0.5
    max_output_tokens: 8000
    reasoning_effort: xhigh
    seed: 123
    fallback_to_chat_completions: true
```

Non-goals for target config:

- no duplicated `allocation` blocks with conflicting semantics;
- no top-level `eval_experiments` as a second parallel experiment selection model;
- no dead `score_weights`.

## 8.4 Target domain schemas

### 8.4.1 `Island`

```python
@dataclass(slots=True)
class Island:
    island_id: str
    name: str
    description_path: str
    description_text: str
```

### 8.4.2 `OrganismMeta`

```python
@dataclass(slots=True)
class OrganismMeta:
    organism_id: str
    island_id: str
    generation_created: int
    current_generation_active: int
    timestamp: str
    mother_id: str | None
    father_id: str | None
    operator: str  # seed | mutation | crossover
    genetic_code_path: str
    optimizer_path: str
    lineage_path: str
    organism_dir: str
    simple_reward: float | None
    hard_reward: float | None
    selection_reward: float | None
    status: str  # pending | evaluated | eliminated | archived
```

### 8.4.3 `LineageEntry`

```python
@dataclass(slots=True)
class LineageEntry:
    generation: int
    operator: str
    mother_id: str | None
    father_id: str | None
    change_description: str
    gene_diff_summary: str
    selected_simple_experiments: list[str]
    selected_hard_experiments: list[str]
    simple_score: float | None
    hard_score: float | None
    aggregate_score: float | None
```

### 8.4.4 `population_manifest.json`

```json
{
  "generation": 5,
  "active_organisms": [
    {
      "organism_id": "abc123",
      "island_id": "gradient_methods",
      "organism_dir": "populations/gen_0000/island_gradient_methods/org_abc123",
      "generation_created": 0,
      "current_generation_active": 5,
      "simple_reward": 0.81,
      "hard_reward": 0.72,
      "selection_reward": 0.81
    }
  ]
}
```

## 8.5 Target scoring semantics

### 8.5.1 Reward math

Canonical per-experiment score:

- `quality_ratio = baseline.objective_last / candidate.objective_last`
- `steps_ratio = baseline.steps / first_step_at_or_below_baseline`
- `exp_score = harmonic_mean(quality_ratio, steps_ratio)`

If baseline threshold is never reached:

- `steps_ratio = 0`
- `exp_score = 0`

### 8.5.2 Phase semantics

- `simple_reward` is assigned after simple evaluation and remains a permanent organism attribute.
- `hard_reward` is assigned only when organism enters Great Filter.
- `selection_reward` is the field used for selection policy in the current phase.

### 8.5.3 Experiment semantics

- simple phase uses only simple experiments;
- Great Filter uses only hard experiments;
- allocation config is phase-specific;
- no single ambiguous scalar `score` is used as canonical truth.

## 8.6 Target lineage semantics

- Mother lineage is canonical.
- Father contributes genes and ancestry metadata, but does not replace canonical lineage path.
- Every lineage entry becomes score-bearing after evaluation.
- Prompt rendering uses condensed, impact-rich lineage summaries.

## 8.7 Target selection semantics

- Mutation parent sampling: uniform within island.
- Crossover parent sampling: softmax over selected reward field within island, except explicit inter-island crossover.
- Great Filter uses `top-h` per island.
- Simple phase uses `top-k` per island.

## 9. Execution principles

Этот раздел обязателен к соблюдению во всех фазах.

### 9.1 Do not build two futures

Нельзя одновременно:

- развивать canonical organism path;
- и добавлять те же semantics в candidate path.

Candidate path может only:

- serve as compatibility bridge;
- be reduced;
- be deprecated.

### 9.2 Every phase must reduce ambiguity

После каждой фазы должно становиться меньше:

- conflicting types;
- duplicate configs;
- duplicate prompt sources;
- architecture seams without formal contract.

### 9.3 Every phase must define deletion targets

В каждой фазе implementer должен явно фиксировать:

- what stays;
- what becomes transitional;
- what is safe to remove later.

### 9.4 Every phase must be testable in isolation

Фаза не считается завершенной только потому, что код компилируется. Нужны:

- explicit tests;
- explicit acceptance criteria;
- documented invariants.

### 9.5 Migration safety is part of the feature

Resume, existing populations, old prompt locations, old configs и old docs — это не afterthought. Migration behavior является частью definition of done для соответствующих фаз.

### 9.6 Rollback and containment are mandatory

Ни одна migration-heavy фаза не считается безопасной, если implementer не понимает, как остановить rollout без разрушения уже существующих артефактов.

Обязательные правила:

- rollback plan не обязательно оформлять отдельным заголовком, но он обязан быть явно покрыт через `Migration / compatibility notes` и `Failure modes`;
- если фаза меняет filesystem layout, должен существовать либо compatibility reader, либо reversible migration step;
- если фаза меняет schema, старые данные должны либо читаться с warning, либо быть однозначно отклонены с ясной ошибкой;
- если фаза меняет public API, implementer обязан решить, существует ли временный adapter, или legacy path должен падать immediately;
- partial rollout не должен создавать silent corruption population state, lineage state или scoring artifacts;
- если safe rollback невозможен, фаза должна выполняться atomically и сопровождаться explicit cutover note;
- в ambiguous case приоритет у data preservation, а не у агрессивного cleanup.

## 10. Workstreams overview

Чтобы implementer мог параллелить работу без разрушения архитектуры, все изменения делятся на workstreams.

### Workstream A. Contracts and interfaces

- optimizer contract;
- public orchestrator seam;
- storage helpers;
- config schema.

### Workstream B. Domain model

- organisms;
- islands;
- lineage;
- genetic code;
- active population manifest.

### Workstream C. Execution flow

- seeding;
- mutation;
- crossover;
- simple phase;
- Great Filter;
- resume.

### Workstream D. Prompt system

- prompt relocation;
- global project context prompt;
- operator prompts;
- gene richness requirements;
- lineage summary rendering requirements.

### Workstream E. Validation and tests

- unit tests;
- integration tests;
- end-to-end flow checks;
- migration compatibility tests.

### Workstream F. Docs and handoff

- README;
- canonical docs;
- deprecation notes;
- operational handoff doc after migration.

## 11. Phase template

Ниже фиксируется обязательный internal template для каждой implementation phase.

Каждая фаза в дальнейших разделах должна содержать:

- `Goal`
- `Why this phase exists`
- `Preconditions`
- `Files / modules affected`
- `Exact changes`
- `New or changed types / schemas`
- `Data flow after change`
- `Migration / compatibility notes`
- `Failure modes`
- `Tests required`
- `Acceptance criteria`
- `Definition of done`

Если implementer пишет подзадачу по фазе, он обязан сохранять эту структуру.

Отдельный заголовок `Rollback` не вводится, чтобы не раздувать шаблон и не смешивать его с failure analysis. Вместо этого rollback, containment и cutover-стратегия обязаны быть явно описаны внутри:

- `Migration / compatibility notes`
- `Failure modes`

Если там нет ответа на вопрос "что происходит при частично завершенной миграции", значит описание фазы неполно.

## 12. Фаза 0. Working agreement and freeze

### Goal

Заморозить канонические названия, источники истины, target schemas и migration direction до любых глубоких реализационных изменений.

### Why this phase exists

Без этой фазы проект продолжит плодить параллельные partial interpretations:

- candidate-centric vs organism-centric;
- old vs new optimizer contract;
- old prompts in `src` vs config-driven prompts;
- flat population vs island model.

### Preconditions

- `6.md` существует и принят как audit-source.
- implementer ознакомлен с locked decisions из этого документа.

### Files / modules affected

- `agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md` — reference only
- `agents/7_gpt_master_execution_spec_for_optimizer_discovery_platform.md` — this document
- no repo-tracked implementation files should be mutated in this phase if phase is done as planning-only

### Exact changes

В practical implementation program эта фаза означает:

1. Зафиксировать canonical naming:
   - `OrganismMeta`
   - `Island`
   - `LineageEntry`
   - `population_manifest.json`
   - `simple_reward`
   - `hard_reward`
   - `selection_reward`
2. Зафиксировать canonical path:
   - organism-first
   - island-aware
   - baseline-relative reward
3. Зафиксировать legacy list:
   - `cand_*` main-path artifacts
   - old optimizer contract
   - prompts in `src/evolve/prompts` as canonical source
   - config fields with no runtime effect
4. Зафиксировать source-of-truth map:
   - what is canonical
   - what is legacy
   - what is transitional

### New or changed types / schemas

No code change required yet, but type names and schema names are locked.

### Data flow after change

No runtime data flow change in this phase. The output is decision freeze.

### Migration / compatibility notes

- Nothing is removed yet.
- This phase only establishes rules that make later migrations deterministic.

### Failure modes

- Implementer starts coding before canonical names are frozen.
- New code introduces alternative names for same concepts.
- Phases are executed out of order because dependencies are not frozen.

### Tests required

No code tests. The “test” for this phase is consistency of all subsequent tasks with locked decisions.

### Acceptance criteria

- No unresolved architectural branch remains open.
- No core domain concept has more than one accepted name.
- `cand_*` path explicitly marked legacy in implementation planning.

### Definition of done

Phase 0 is done when an implementer can begin Phase 1 without making any domain-model decisions from scratch.

## 13. Фаза 1. Unify public optimizer contract

### Goal

Привести README, runtime, import layer, generator validation, examples, prompts и tests к одному canonical optimizer contract.

### Why this phase exists

Сейчас разные части проекта описывают несовместимые контракты:

- old `build_optimizer(cfg)` / `initialize(...)`
- new `build_optimizer(model, max_steps)` / `step(..., step_fn)`

Пока это не исправлено:

- docs mislead users;
- tests validate wrong interface;
- generator can accept code that runtime should reject.

### Preconditions

- Phase 0 completed.
- Locked decision `optimizer contract = new runtime contract` accepted.

### Files / modules affected

- `README.md`
- `valopt/optimizer_api.py`
- `valopt/utils/import_utils.py`
- `src/evolve/generator.py`
- `src/evolve/template_parser.py`
- `src/evolve/prompts/optimizer_system.txt`
- `src/evolve/prompts/optimizer_user.txt`
- `tests/test_optimizer_generator.py`
- `tests/test_import_optimizer.py`
- `optimizer_guesses/examples/sgd_baseline.py`

### Exact changes

1. README
   - Replace old contract section with canonical one.
   - Remove `initialize(named_parameters, cfg)` references.
   - Show only `build_optimizer(model, max_steps)` examples.
   - Clarify role of `step_fn`.

2. `valopt/optimizer_api.py`
   - Keep canonical contract as primary API.
   - If helper comments remain, align them with README wording exactly.

3. `valopt/utils/import_utils.py`
   - Enforce `build_optimizer(model, max_steps)` at load level.
   - Improve error message for legacy contract usage.
   - Optionally add signature validation if it can be done safely.

4. `src/evolve/generator.py`
   - Strengthen `_validate_code()` so accepted generated code matches runtime contract.
   - Do not accept code with only legacy `initialize(...)`.
   - Require compatibility with `step_fn`.

5. `src/evolve/template_parser.py`
   - Keep strict template validation aligned with canonical runtime contract.
   - If needed, centralize shared validation logic instead of duplicating rules.

6. Prompts
   - `optimizer_system.txt` and `optimizer_user.txt` must document canonical contract only.
   - Remove any ambiguity around builder signature and `step_fn`.

7. Tests
   - Rewrite `tests/test_optimizer_generator.py` to validate canonical contract.
   - Keep tests that ensure malformed legacy code is rejected.
   - Ensure runtime and generator agree on valid/invalid examples.

8. Example optimizer
   - `optimizer_guesses/examples/sgd_baseline.py` must remain canonical and current.

### New or changed types / schemas

No major domain-schema changes in this phase. Public contract is what changes.

Canonical contract:

```python
def build_optimizer(model: torch.nn.Module, max_steps: int):
    ...

class OptimizerController:
    def step(self, weights, grads, activations, step_fn) -> None: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
```

### Data flow after change

After this phase:

- documented example -> load path -> runtime path -> generator-accepted code

must all agree on the same interface.

### Migration / compatibility notes

Recommended strategy:

- do not silently support old contract;
- fail fast with explicit error;
- update docs and tests in the same phase.

### Failure modes

- README updated but tests still accept old code.
- Runtime rejects code that generator accepts.
- Generator becomes stricter than runtime in a different way than intended.

### Tests required

- Contract acceptance test for canonical example.
- Rejection test for old `build_optimizer(cfg)` style.
- Generator validation test for correct `step_fn` usage shape.
- Import/load test that canonical example passes end-to-end.

### Acceptance criteria

- A README example optimizer is loadable without edits.
- Old documented contract no longer appears anywhere as canonical.
- Generator validator and runtime validator agree on the same valid sample set.
- No test still encodes `initialize(...)` as accepted contract.

### Definition of done

Phase 1 is done when a user can follow README and get a working optimizer file without falling into interface mismatch.

## 14. Фаза 2. Canonicalize artifact/storage model

### Goal

Сделать organism-first storage canonical, устранить ambiguity между `cand_*` и `org_*`, и сделать resume semantically correct.

### Why this phase exists

Сейчас:

- `cand_*` and `org_*` coexist;
- current generation directory scan cannot reconstruct active population correctly;
- storage layer does not clearly distinguish created-at vs active-now state.

Without fixing this:

- resume is not trustworthy;
- evolution state is ambiguous;
- context loading and lineage tracking remain inconsistent.

### Preconditions

- Phase 1 complete.
- Canonical architecture fixed as organism-first.

### Files / modules affected

- `src/evolve/storage.py`
- `src/evolve/evolution_loop.py`
- `src/evolve/types.py`
- any summary-writing helpers that still assume candidate-centric truth

### Exact changes

1. Add active-population manifest support.
   - Create helpers:
     - `population_manifest_path(population_root)`
     - `write_population_manifest(...)`
     - `read_population_manifest(...)`

2. Redefine resume flow.
   - `_save_state()` saves scalar state.
   - population manifest saves active organism refs.
   - `_load_state()` restores:
     - generation
     - active organisms via manifest

3. Define canonical meaning of generation directories.
   - `gen_<N>` is creation batch context.
   - It is not by itself the source of truth for current active population.

4. Mark `cand_*` layout as legacy in storage APIs.
   - Keep reading if needed during migration.
   - Stop documenting it as main evolution layout.

5. Add forward-compatible organism summary shape if not already present.

### New or changed types / schemas

Add explicit manifest schema:

```python
ManifestEntry = TypedDict(
    "ManifestEntry",
    {
        "organism_id": str,
        "island_id": str,
        "organism_dir": str,
        "generation_created": int,
        "current_generation_active": int,
        "simple_reward": float | None,
        "hard_reward": float | None,
        "selection_reward": float | None,
    },
)
```

### Data flow after change

Before:

- save state -> scan only current `gen_N/org_*` on resume

After:

- save state -> write manifest of active organisms
- resume -> read scalar state + read active organism refs from manifest -> rehydrate population

### Migration / compatibility notes

- Existing populations without manifest should be handled explicitly.
- Fallback may attempt old behavior only as legacy compatibility mode.
- If fallback is used, it must emit a warning that population may be incomplete.

### Failure modes

- Missing manifest on old population.
- Stale manifest points to missing organism dir.
- Partial generation wrote state but not manifest.
- Manifest and scalar generation disagree.

### Tests required

- Resume with survivor created in older generation.
- Resume with newly created current-generation organism.
- Resume with missing manifest in legacy population.
- Resume with stale path in manifest.
- Resume after partial generation state.

### Acceptance criteria

- Active population after resume matches uninterrupted run semantics.
- `load_population(generation)` is no longer the primary restore mechanism.
- Documentation no longer implies that scanning current generation directory is sufficient.

### Definition of done

Phase 2 is done when resume restores the real active population instead of a generation-local snapshot approximation.

## 15. Фаза 3. Split orchestration responsibilities cleanly

### Goal

Убрать private coupling между `EvolutionLoop` и `EvolverOrchestrator`, зафиксировать публичный seam API и разделить domain policy от execution plumbing.

### Why this phase exists

Сейчас organism-loop reaches into orchestrator internals. This is brittle and prevents clean architecture.

### Preconditions

- Phase 2 complete.
- Manifest-based active population model available.

### Files / modules affected

- `src/evolve/evolution_loop.py`
- `src/evolve/orchestrator.py`
- `src/evolve/types.py`
- possibly `src/evolve/allocation.py`

### Exact changes

1. Define public orchestrator-facing evaluation API.
2. Remove direct usage of private methods:
   - `_build_candidate_allocation`
   - `_register_candidate`
3. Introduce public request/response types for organism evaluation.
4. Make phase-specific allocation explicit in request payload.
5. Clarify responsibility split:
   - loop owns population and lineage
   - orchestrator owns evaluation job submission/execution

### New or changed types / schemas

Introduce request/response abstractions, e.g.:

```python
@dataclass(slots=True)
class OrganismEvaluationRequest:
    organism_id: str
    organism_dir: str
    phase: str  # simple | hard
    experiments: list[str]
    allocation_cfg: dict[str, Any]
    created_at: str

@dataclass(slots=True)
class OrganismEvaluationSummary:
    organism_id: str
    phase: str
    aggregate_score: float | None
    per_experiment: dict[str, dict[str, Any]]
    selected_experiments: list[str]
    allocation_snapshot: dict[str, Any]
    status: str
```

### Data flow after change

After this phase:

1. `EvolutionLoop` builds evaluation request for organism(s).
2. Orchestrator executes via public API.
3. Public summary returns to loop.
4. Loop updates:
   - organism rewards
   - lineage
   - manifest
   - selection state

### Migration / compatibility notes

- During migration, orchestrator may internally still use candidate-like state, but that must be hidden behind the public API.
- The public API must already express organism semantics even if implementation still adapts internally.

### Failure modes

- Public API leaks candidate-centric concepts.
- Phase-specific allocation silently ignored.
- Loop still imports or reaches private methods.

### Tests required

- Integration test: loop evaluates organism via public API.
- Allocation config passed from loop is visible in evaluation behavior.
- No test depends on private orchestrator methods.

### Acceptance criteria

- `EvolutionLoop` no longer directly calls orchestrator private methods.
- Orchestrator can evaluate organism requests through a stable public interface.
- Domain policy decisions live in loop/config, not in orchestrator internals.

### Definition of done

Phase 3 is done when loop/orchestrator integration is explicit, public and testable.

## 16. Фаза 4. Move prompts and text assets into `conf`

### Goal

Сделать prompts и school descriptions configuration assets, а не runtime-code assets inside `src`.

### Why this phase exists

Prompt system is part of research policy and must be versioned/configured like project text assets.

### Preconditions

- Phase 3 complete.
- Public API boundaries no longer rely on hidden prompt path assumptions.

### Files / modules affected

- `conf/`
- `src/evolve/generator.py`
- `src/evolve/operators.py`
- `src/organisms/mutation.py`
- `src/organisms/crossbreeding.py`
- prompt-loading helpers if introduced

### Exact changes

1. Create canonical prompt layout under `conf/prompts/`.
2. Create `conf/islands/` directory even if islands not fully implemented yet.
3. Move or recreate canonical prompt text assets there.
4. Introduce config-driven prompt path resolution.
5. Mark `src/evolve/prompts/*` as legacy.
6. Remove prompt duplication from canonical set:
   - no parallel `mutate_user` / `mutation_user`
   - no parallel `crossbreed_user` / `crossover_user` in canonical layout
7. Add global project context prompt.

### New or changed types / schemas

Config block:

```yaml
evolver:
  prompts:
    project_context: ...
    seed_system: ...
    seed_user: ...
    mutation_system: ...
    mutation_user: ...
    crossover_system: ...
    crossover_user: ...
```

### Data flow after change

After this phase:

- prompt resolution = config -> file path -> text asset
- project context prompt is prepended or composed into all operation-specific prompt builds

### Migration / compatibility notes

- Temporary fallback to old prompt paths is allowed only with explicit warning.
- New code must not introduce any new canonical prompt under `src/evolve/prompts`.

### Failure modes

- Prompt path hardcoded again in generator/operators.
- Old prompt files remain canonical by accident.
- Prompt duplication preserved under new names.

### Tests required

- Prompt loading from config paths.
- Failure on missing required prompt file.
- Legacy fallback warning if old path is used.
- Prompt composition test: project context included for each operator.

### Acceptance criteria

- Canonical prompt source is `conf/prompts`.
- School descriptions have reserved canonical location `conf/islands`.
- No new feature depends on `src/evolve/prompts` as source of truth.

### Definition of done

Phase 4 is done when prompt texts can be changed as config assets without editing Python source.

## 17. Фаза 5. Introduce island model

### Goal

Добавить islands как основную topology search system: seed, population state, selection, crossover policy and replenishment all become island-aware.

### Why this phase exists

Flat population is fundamentally not the target design. Islands are central to diversity, specialization and controlled cross-pollination.

### Preconditions

- Phase 4 complete.
- `conf/islands/` exists and prompt system is config-driven.

### Files / modules affected

- `src/evolve/types.py`
- `src/evolve/evolution_loop.py`
- `src/evolve/selection.py`
- `src/evolve/storage.py`
- island loader helper (new)
- `conf/evolver/default.yaml`
- `conf/islands/*.txt`

### Exact changes

1. Add `Island` domain type.
2. Add `island_id` to organisms.
3. Build island loader:
   - scans `conf/islands/*.txt`
   - derives `island_id` from filename
   - reads description text
4. Seed initial population per island:
   - `organisms_per_island`
   - school-specific prompt context
5. Store organisms under island-aware directory structure.
6. Change selection to operate per island.
7. Introduce inter-island crossover rate.
8. Define replenishment policy as island-aware.

### New or changed types / schemas

See target schemas in section 8; actual code should implement them here.

### Data flow after change

1. Load islands.
2. Seed organisms per island.
3. Evaluate per phase.
4. Select within island.
5. Reproduce mostly within island, optionally across islands.
6. Maintain active population manifest with island membership.

### Migration / compatibility notes

- Existing flat populations without island info are legacy.
- Migration strategy:
  - either reject resume for flat-population states after island migration;
  - or assign them to a special legacy island.

Recommended default:

- use explicit `legacy_flat_population` island only for compatibility, never as long-term canonical mode.

### Failure modes

- island files empty or malformed;
- zero islands found;
- unequal population counts accidentally produced;
- inter-island crossover ignores rate config;
- storage layout does not encode island identity explicitly.

### Tests required

- island loader with multiple txt files;
- failure on missing/empty island dir;
- per-island seeding count;
- per-island selection invariants;
- cross-island crossover respecting configured rate;
- manifest preserving island ids.

### Acceptance criteria

- No flat-population-only selection remains in canonical flow.
- Seeding is driven by island descriptions.
- Organisms always carry `island_id`.
- Filesystem layout makes island membership explicit.

### Definition of done

Phase 5 is done when the system’s search topology is island-aware end-to-end.

## 18. Фаза 6. Rebuild genetic code as canonical artifact

### Goal

Превратить genetic code из бедного trait list в canonical artifact, который действительно служит blueprint для optimizer implementation.

### Why this phase exists

Current flat short DNA is insufficient for your target design, and current mutation/crossover persistence contradicts prompt promises.

### Preconditions

- Phase 5 complete.
- Islands and prompt assets already canonicalized.

### Files / modules affected

- `src/organisms/organism.py`
- `src/organisms/mutation.py`
- `src/organisms/crossbreeding.py`
- `src/evolve/template_parser.py`
- prompt files in `conf/prompts`
- storage helpers for new genetic code artifact

### Exact changes

1. Define canonical genetic code storage format.
   - Recommended canonical artifact: `genetic_code.md`
   - Optional machine-readable derivative later
2. Define what one gene is.
3. Mutation flow:
   - pre-LLM mutation selects removed/surviving inherited pool
   - LLM returns final canonical child genetic code
   - persistence uses LLM-returned final DNA, not silent override
4. Crossover flow:
   - pre-LLM crossover builds inherited pool
   - LLM rewrites inherited pool into coherent child genetic code
   - persistence uses child canonical output
5. Add validation rules for gene richness and malformed outputs.

### New or changed types / schemas

Recommended canonical content model for `genetic_code.md`:

```text
## CORE_GENES
<semicolon-separated rich genes>

## INTERACTION_NOTES
<brief coherence notes>

## COMPUTE_NOTES
<step_fn or budget-relevant constraints>
```

If machine-readable schema is also needed:

```python
@dataclass(slots=True)
class Gene:
    text: str
    category: str | None = None
    rationale: str | None = None
```

### Data flow after change

Before:

- mutation/crossover compute child trait list
- persistence may override LLM DNA

After:

- pre-LLM phase defines inherited/removed pools
- LLM returns final canonical DNA
- persistence writes final DNA artifact
- template implementation is derived from final DNA, not from hidden override

### Migration / compatibility notes

- Existing `idea_dna.txt` may be kept temporarily as derived/legacy representation.
- New canonical artifact should be `genetic_code.md` or equivalent richer format.

### Failure modes

- LLM returns empty DNA section.
- Genes remain too terse.
- Duplicate genes after crossover.
- Contradictory genes.
- Mutation removes everything and LLM output malformed.

### Tests required

- Mutation persists LLM-added gene.
- Crossover persists LLM-rewritten DNA.
- Duplicate gene handling.
- Empty malformed DNA rejection or fallback policy.
- Richness validation test for obviously too-terse genes if validator is introduced.

### Acceptance criteria

- No silent `idea_dna_override` canonical persistence for mutation/crossover.
- Child genetic code comes from accepted final LLM output.
- Genetic code is rich enough to reconstruct optimizer logic from template.

### Definition of done

Phase 6 is done when genetic code is truly the canonical blueprint of an organism rather than a lossy sidecar.

## 19. Фаза 7. Rebuild lineage as canonical research record

### Goal

Сделать lineage завершенным, score-bearing, maternal-path-preserving artifact’ом, пригодным как для human analysis, так и для prompt conditioning.

### Why this phase exists

Current lineage is incomplete and semantically wrong for your design.

### Preconditions

- Phase 6 complete.
- Canonical genetic code persistence in place.

### Files / modules affected

- `src/organisms/organism.py`
- `src/organisms/crossbreeding.py`
- `src/evolve/evolution_loop.py`
- storage helpers for lineage artifact
- prompt formatting helpers

### Exact changes

1. Redefine lineage entry schema.
2. Preserve maternal canonical path.
3. Store father contribution separately from canonical lineage chain.
4. Backfill scores after evaluation.
5. Include selected experiments and phase-specific scoring in entries.
6. Improve prompt rendering to highlight impact and not just log text.

### New or changed types / schemas

Canonical `LineageEntry` from target section 8 should be implemented here.

### Data flow after change

1. Organism created.
2. New lineage entry created with:
   - operator
   - parent ids
   - gene diff summary
   - change description
3. Evaluation completes.
4. Same lineage entry is updated with:
   - simple score and/or hard score
   - aggregate score
5. Prompt renderer summarizes completed lineage entries.

### Migration / compatibility notes

- Legacy lineage entries with only `score` may be readable but should not be considered full canonical entries.
- Migration may backfill new fields as `null` for legacy history.

### Failure modes

- scores not written back after evaluation;
- mother lineage accidentally merged with father lineage;
- lineage summary truncates key information;
- resume loses unfinished lineage entry state.

### Tests required

- New lineage entry created on organism birth.
- Score backfilled after simple evaluation.
- Hard score added after Great Filter.
- Maternal lineage preserved in crossover.
- Prompt summary renders score-bearing history.

### Acceptance criteria

- No endless lineage with `score=None` for newly evaluated organisms.
- Child lineage follows maternal path.
- Father contribution retained as ancestry metadata without corrupting canonical lineage chain.

### Definition of done

Phase 7 is done when lineage can be treated as a true research history artifact and not as a decorative log.

## 20. Фаза 8. Rebuild selection semantics

### Goal

Привести parent sampling, selection fields and selection policies к exact target semantics.

### Why this phase exists

Current tournament-based logic differs materially from target search dynamics.

### Preconditions

- Phase 7 complete.
- Organism rewards and lineage semantics already explicit.

### Files / modules affected

- `src/evolve/selection.py`
- `src/evolve/evolution_loop.py`
- `conf/evolver/default.yaml`
- any helper modules that rely on ambiguous `score`

### Exact changes

1. Replace mutation parent selection with uniform sampling.
2. Replace crossover parent selection with softmax sampling.
3. Add explicit softmax temperature config.
4. Define mother/father assignment rules.
5. Replace ambiguous `score` in selection logic with explicit `selection_reward`.
6. Remove or repurpose dead `selection_strategy` and `crossover_rate`.

### New or changed types / schemas

If explicit helper signatures are introduced:

```python
def uniform_select_organisms(population, k, rng): ...
def softmax_select_organisms(population, score_field, temperature, k, rng): ...
```

### Data flow after change

After this phase:

- simple phase writes simple reward
- selection chooses `selection_reward` for current phase
- mutation samples uniformly within island
- crossover samples mother/father through softmax within allowed population set

### Migration / compatibility notes

- During migration, `score` may remain as derived reporting field, but not as canonical selection field.

### Failure modes

- mutation still indirectly influenced by tournament pressure;
- softmax parent selection falls back to implicit uniform without warning;
- `None` scores break selection or are silently misused;
- cross-island selection ignores island boundaries.

### Tests required

- uniform mutation parent selection behavior;
- softmax crossover selection behavior;
- handling of all-None score cases;
- per-island sampling boundaries;
- softmax temperature changes sampling distribution.

### Acceptance criteria

- Selection logic matches spec exactly.
- No core selection behavior depends on ambiguous aggregate `score`.
- Dead selection knobs removed or made real.

### Definition of done

Phase 8 is done when search dynamics reflect the intended design instead of legacy tournament behavior.

## 21. Фаза 9. Rebuild simple phase and Great Filter semantics

### Goal

Сделать simple phase и Great Filter двумя различными, semantically clean phases with separate scoring and selection behavior.

### Why this phase exists

Current Great Filter incorrectly mixes simple and hard experiments and reuses shared selection scalar semantics.

### Preconditions

- Phase 8 complete.
- Explicit `simple_reward`, `hard_reward`, `selection_reward` model in place.

### Files / modules affected

- `src/evolve/evolution_loop.py`
- `src/evolve/orchestrator.py`
- `src/evolve/scoring.py`
- `src/evolve/metrics_adapter.py`
- `conf/evolver/default.yaml`

### Exact changes

1. Redefine simple phase:
   - evaluate newborns on simple experiments only
   - assign permanent `simple_reward`
   - select `top-k` per island
2. Redefine Great Filter:
   - evaluate survivors on hard experiments only
   - assign `hard_reward`
   - compute `selection_reward` for reproduction eligibility in hard phase
   - select `top-h` per island
3. Introduce phase-specific allocation configs.
4. Ensure Great Filter does not overwrite historical meaning of simple reward.

### New or changed types / schemas

Config:

```yaml
phases:
  simple:
    top_k_per_island: ...
    experiments: ...
    allocation: ...
  great_filter:
    top_h_per_island: ...
    experiments: ...
    allocation: ...
```

### Data flow after change

Simple phase:

1. newborn organism evaluated
2. baseline-relative per-experiment score computed
3. `simple_reward` saved
4. lineage updated with simple-phase score
5. per-island `top-k` selection

Great Filter:

1. selected survivors evaluated on hard experiments
2. `hard_reward` saved
3. `selection_reward` updated for reproduction phase
4. lineage updated with hard-phase score
5. per-island `top-h` selection

### Migration / compatibility notes

- Existing `elite_count` should be replaced or deprecated.
- Old top-level `allocation` config should be migrated into phase-specific allocation.

### Failure modes

- Great Filter accidentally still uses simple + hard union;
- top-k and top-h not truly separate;
- hard score overwrites simple reward;
- phase-specific allocation silently ignored.

### Tests required

- Great Filter uses hard-only experiments.
- Simple phase uses simple-only experiments.
- top-k and top-h both applied correctly.
- phase-specific allocation honored.
- simple reward preserved after Great Filter.

### Acceptance criteria

- No ambiguous shared phase scalar remains.
- Great Filter semantics match target design.
- Simple reward remains stable historical property.

### Definition of done

Phase 9 is done when simple phase and Great Filter have distinct, testable, non-overlapping semantics.

## 22. Фаза 10. Config cleanup and truthfulness

### Goal

Убрать или deprecate misleading config, чтобы every config key has real, documented runtime meaning.

### Why this phase exists

Fake config is dangerous in research infrastructure because it creates false interpretability and false control.

### Preconditions

- Phase 9 complete.

### Files / modules affected

- `conf/evolver/default.yaml`
- `conf/experiments/*.yaml`
- config parsing code in evolve/runtime modules
- README config sections

### Exact changes

1. Remove or deprecate dead keys:
   - `score_weights`
   - top-level duplicated `allocation`
   - duplicate experiment-selection fields
2. Mark or remove static normalization refs from evolve scoring docs.
3. Add explicit warnings or migration behavior for deprecated keys.
4. Ensure no config key is silently ignored.

### New or changed types / schemas

Config schema becomes cleaner and phase-oriented.

### Data flow after change

Config -> runtime behavior should become one-to-one and explainable.

### Migration / compatibility notes

For each deprecated key choose one:

- reject with error
- accept with warning
- auto-migrate

Recommended:

- `score_weights`: accept with warning for one migration cycle, then remove
- old `allocation`: reject or migrate into `phases.simple.allocation` if deterministic

### Failure modes

- docs updated but runtime still accepts dead keys silently;
- deprecated keys remain in examples;
- migration warnings not emitted.

### Tests required

- deprecated key warning tests;
- config rejection tests for invalid conflicting fields;
- config path truthfulness tests.

### Acceptance criteria

- Every documented config key affects runtime or is explicitly deprecated.
- No duplicated config blocks encode conflicting semantics.

### Definition of done

Phase 10 is done when config is honest, minimal and semantically aligned with runtime behavior.

## 23. Фаза 11. Test-surface rebuild

### Goal

Расширить test coverage с helper-level correctness до product-level invariants of the canonical system.

### Why this phase exists

Current tests are green but insufficiently protective against architectural drift.

### Preconditions

- Phases 1-10 complete or near-complete enough to expose stable canonical behavior.

### Files / modules affected

- `tests/`
- fixtures and fake evaluators
- any new integration harnesses needed

### Exact changes

Test families to add or rebuild:

1. Contract tests
2. Resume tests
3. Island initialization tests
4. Selection semantics tests
5. Genetic code persistence tests
6. Lineage score backfill tests
7. Great Filter semantics tests
8. Prompt/config loading tests
9. Migration compatibility tests

### New or changed types / schemas

No major production schema change. Test fixtures should mirror canonical schemas.

### Data flow after change

Test suite becomes the guardrail for:

- architecture invariants;
- migration safety;
- semantic correctness.

### Migration / compatibility notes

- It is acceptable to keep some legacy tests temporarily, but they must be clearly marked and eventually removed if they assert old semantics.

### Failure modes

- tests remain too local and do not catch cross-module semantic drift;
- fake fixtures do not exercise island-aware behaviors;
- old tests still validate legacy contract.

### Tests required

This phase is itself about adding tests, so required output is the test matrix below.

### Acceptance criteria

- Canonical invariants are covered, not just helpers.
- Failing core semantics can no longer hide behind green helper tests.

### Definition of done

Phase 11 is done when test suite meaningfully protects the target architecture and migration behavior.

## 24. Test matrix for canonical system

### 24.1 Contract tests

Invariant proved:

- documented optimizer contract matches runtime and generator expectations.

Must-have tests:

- canonical example optimizer loads;
- legacy `build_optimizer(cfg)` rejected;
- generator validator and import layer accept same good sample;
- generator validator and import layer reject same bad sample.

### 24.2 Resume tests

Invariant proved:

- active population resume is generation-independent and manifest-driven.

Must-have tests:

- survivor created in old generation restored correctly;
- current-generation organism restored correctly;
- stale manifest path handled deterministically;
- missing manifest for legacy population triggers explicit fallback or error path.

### 24.3 Island tests

Invariant proved:

- system is truly island-aware.

Must-have tests:

- multiple island txt files load correctly;
- equal seed counts per island by default;
- malformed island file handled explicitly;
- per-island selection boundaries enforced;
- inter-island crossover rate influences parent pairing.

### 24.4 Genetic code tests

Invariant proved:

- canonical genetic code persists final accepted child DNA.

Must-have tests:

- mutation can add one new gene and persist it;
- crossover can rewrite inherited genes and persist final DNA;
- empty malformed DNA rejected or handled via explicit fallback;
- duplicate genes normalized according to canonical policy.

### 24.5 Lineage tests

Invariant proved:

- lineage is canonical maternal research record with score backfill.

Must-have tests:

- mother lineage preserved in crossover;
- father metadata preserved separately;
- last lineage entry updated after simple evaluation;
- hard-phase score backfill added after Great Filter;
- prompt-summary formatting includes score-bearing entries.

### 24.6 Selection tests

Invariant proved:

- mutation uniform and crossover softmax semantics are real.

Must-have tests:

- mutation parent sampling approximates uniform over many runs;
- crossover parent sampling biased by reward and temperature;
- all-None scores handled deterministically;
- per-island selection never leaks across islands unless explicit inter-island crossover path is used.

### 24.7 Phase semantics tests

Invariant proved:

- simple phase and Great Filter are semantically distinct.

Must-have tests:

- Great Filter uses hard-only experiments;
- simple phase uses simple-only experiments;
- `top-k` and `top-h` differ in effect;
- simple reward preserved after hard phase;
- phase-specific allocation actually affects selected experiments.

### 24.8 Prompt/config loading tests

Invariant proved:

- prompt system is config-driven and truthfully loaded from `conf`.

Must-have tests:

- missing required prompt file fails clearly;
- configured prompt path loaded correctly;
- project context prompt composed into operator prompt;
- legacy fallback path emits warning if still supported.

### 24.9 Migration compatibility tests

Invariant proved:

- old artifacts/configs are either migrated or rejected explicitly.

Must-have tests:

- old population without manifest;
- old prompt location;
- deprecated config key;
- old normalization config fields ignored or warned without affecting reward.

## 25. Фаза 12. Docs and handoff cleanup

### Goal

Привести документацию и handoff-артефакты к новому canonical state.

### Why this phase exists

Even correct code remains dangerous if docs still describe old contracts and old layouts.

### Preconditions

- Phases 1-11 complete or stable enough that docs can be updated without immediate invalidation.

### Files / modules affected

- `README.md`
- `agents/`
- optional short operational doc to create after migration

### Exact changes

1. Rewrite README around canonical optimizer contract and canonical evolve outputs.
2. Remove stale references to old scoring model and old template assumptions.
3. Keep `6.md` as audit.
4. Keep `7.md` as execution spec.
5. Create one short post-migration operational doc for day-to-day usage.
6. Mark outdated `agents` notes as historical if they remain.

### New or changed types / schemas

No production code schema changes here.

### Data flow after change

New engineer or agent should be able to:

- read README for usage;
- read short operational doc for normal work;
- read `6.md` for why the migration happened;
- read `7.md` for implementation logic.

### Migration / compatibility notes

- Historical docs should not be deleted if they contain useful context, but must not masquerade as canonical current docs.

### Failure modes

- README still documents old optimizer contract;
- `cand_*` still shown as canonical output;
- old reward docs still suggest `quality_ref`/`steps_ref` are active.

### Tests required

Docs consistency checks are mostly manual, but README examples should be smoke-tested whenever feasible.

### Acceptance criteria

- A new engineer reading current docs is not misled about contract, scoring, or storage.
- `6.md` and `7.md` have clearly separated roles.

### Definition of done

Phase 12 is done when docs no longer preserve the old mental model of the system.

## 26. Appendices overview

The appendices below are part of the execution spec. They are not optional reading.

## Appendix A. Source-of-truth map

| Category | Canonical | Legacy | Transitional |
|---|---|---|---|
| Evolution entity | `OrganismMeta` | `CandidateMeta` as main model | candidate adapters behind orchestrator seam |
| Prompt source | `conf/prompts/*` | `src/evolve/prompts/*` as source of truth | fallback reading of old path only if warned |
| Island definitions | `conf/islands/*.txt` | none | legacy flat-pop migration island |
| Active population state | `population_manifest.json` | scan current `gen_N/org_*` | manifest + legacy fallback |
| Reward model | baseline-relative `train_loss` | static normalization refs as active scoring inputs | legacy fields in payloads only |
| Selection | uniform mutation + softmax crossover | tournament-based core policy | temporary adapters if needed |
| Docs | README + `7.md` + short operational doc | older followup notes as current docs | historical notes marked as historical |

## Appendix B. Final schemas

### B.1 `evolver` config

Use target schema from section 8.3.

### B.2 `Island`

Use target schema from section 8.4.1.

### B.3 `OrganismMeta`

Use target schema from section 8.4.2.

### B.4 `LineageEntry`

Use target schema from section 8.4.3.

### B.5 `population_manifest.json`

Use target schema from section 8.4.4.

## Appendix C. Final directory layout

### C.1 `conf/`

```text
conf/
  config.yaml
  evolver/default.yaml
  prompts/
    system_project_context.txt
    seed_system.txt
    seed_user.txt
    mutation_system.txt
    mutation_user.txt
    crossover_system.txt
    crossover_user.txt
  islands/
    *.txt
```

### C.2 Population storage

```text
populations/
  evolution_state.json
  population_manifest.json
  gen_0000/
    island_<name>/
      org_<id>/
        optimizer.py
        genetic_code.md
        lineage.json
        organism.json
        summary.json
        results/
          simple/
          hard/
        logs/
```

## Appendix D. Public APIs and contracts

### D.1 Optimizer contract

```python
def build_optimizer(model: torch.nn.Module, max_steps: int): ...
```

Controller methods:

```python
def step(self, weights, grads, activations, step_fn) -> None: ...
def zero_grad(self, set_to_none: bool = True) -> None: ...
```

### D.2 Evaluation backend seam

Recommended public API:

```python
async def evaluate_organisms(
    requests: list[OrganismEvaluationRequest],
) -> list[OrganismEvaluationSummary]:
    ...
```

### D.3 Storage helpers

Required public helpers:

- `population_manifest_path(...)`
- `write_population_manifest(...)`
- `read_population_manifest(...)`
- `read_organism_meta(...)`
- `write_organism_meta(...)`
- `read_lineage(...)`
- `write_lineage(...)`
- `load_best_organism_context(...)`

### D.4 Selection helpers

Required public helpers:

- `uniform_select_organisms(...)`
- `softmax_select_organisms(...)`
- `select_top_k_per_island(...)`
- `select_top_h_per_island(...)`

## Appendix E. Task graph

### E.1 Strict dependency order

```text
Phase 0
  -> Phase 1
  -> Phase 2
  -> Phase 3
  -> Phase 4
  -> Phase 5
  -> Phase 6
  -> Phase 7
  -> Phase 8
  -> Phase 9
  -> Phase 10
  -> Phase 11
  -> Phase 12
```

### E.2 What can be parallelized after dependencies are satisfied

After Phase 1:

- README cleanup and validator alignment can be parallelized if one owner coordinates canonical contract wording.

After Phase 4:

- island prompt drafting and config schema coding can run in parallel if file ownership is separated.

After Phase 9:

- docs cleanup and test-surface expansion can overlap partially.

### E.3 What must not start early

- Island model must not start before prompt relocation and schema freeze.
- Genetic code rebuild must not start before canonical artifact/storage path is decided.
- Test-surface rebuild must not lock in temporary migration behavior as canonical.

## Appendix F. Acceptance checklist

Use this as end-to-end completion checklist.

### Architecture

- [ ] organism-first architecture is canonical
- [ ] candidate-first path is legacy only
- [ ] orchestrator is accessed through public seam

### Contracts

- [ ] one optimizer contract everywhere
- [ ] README matches runtime
- [ ] generator validator matches runtime validator

### Storage

- [ ] active population manifest exists
- [ ] resume does not depend on current generation scan
- [ ] island-aware directory layout exists

### Prompts

- [ ] prompts loaded from `conf`
- [ ] project context prompt exists
- [ ] school descriptions live in `conf/islands`

### Domain model

- [ ] organisms have `island_id`
- [ ] genetic code is canonical rich artifact
- [ ] lineage stores scored entries
- [ ] maternal lineage semantics enforced

### Selection and phases

- [ ] mutation sampling uniform
- [ ] crossover sampling softmax
- [ ] simple phase uses simple experiments only
- [ ] Great Filter uses hard experiments only
- [ ] `top-k` and `top-h` both implemented

### Config

- [ ] no dead config keys left undocumented
- [ ] deprecated keys warn or fail explicitly
- [ ] no duplicated phase semantics in config

### Tests

- [ ] contract tests updated
- [ ] resume tests added
- [ ] island tests added
- [ ] lineage tests added
- [ ] selection tests added
- [ ] phase semantics tests added
- [ ] migration compatibility tests added

### Docs

- [ ] README updated
- [ ] `6.md` remains audit
- [ ] `7.md` remains execution spec
- [ ] short operational doc prepared after migration

## 27. Non-goals of the current execution program

Чтобы implementer не расползался в breadth, ниже явно указано, что не является частью этой migration unless explicitly requested later.

- Re-inventing reward math.
- Moving to unconstrained code generation.
- Adding novelty filter or bandit model selection as new production features before canonical architecture is stable.
- Large-scale experiment-set redesign unrelated to evolution architecture cleanup.
- Multi-GPU support.
- UI/dashboard work unless directly needed for canonical artifact inspection.

## 28. Final operational note

Этот документ intentionally жесткий. Он фиксирует не только что надо сделать, но и что **нельзя** делать во время реализации.

Если implementer сталкивается с локальной трудностью, он должен:

- сначала проверить, не решается ли она в рамках locked decisions и phase plan;
- затем выбрать решение, которое минимально нарушает canonical target state;
- и только если это невозможно, поднять вопрос как explicit exception, а не silently invent new architecture.

Главная идея `7.md`:

- не дать репозиторию еще раз распасться на несколько одновременно существующих partial systems;
- провести migration в одну coherent optimizer discovery platform;
- сделать так, чтобы после завершения фаз другой агент уже не писал "еще один аудит", а мог просто работать по ясной operational documentation.
