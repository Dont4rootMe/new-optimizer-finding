# Repository External Knowledge Base

Эта страница — живая внешняя база знаний о репозитории для людей и агентов. Она
описывает не только текущую checkout-ветку, но и все видимые upstream-ветки,
реальное поведение кода, активные конфигурации, экспериментальные ответвления,
известные результаты и расхождения старой документации с live-кодом.

## Паспорт последнего аудита

- Дата полного cross-branch аудита: **2026-08-04**.
- Последнее targeted-обновление: **2026-08-04** — начата finalization-линия:
  унифицированы backbone profiles и task contracts, добавлены task catalog,
  provider-neutral token telemetry и воспроизводимый 8×H100 job path.
- Авторская рабочая копия остаётся в
  `/Users/enrifermi/Projects/new-optimizer-finding`; она не переключалась и не
  очищалась. Все изменения выполняются в отдельном clone/worktree
  `.worktrees/finalize-evolutionloop-deepseek-v4` на ветке
  `finalize/evolutionloop-deepseek-v4`.
- Remote: `origin = https://github.com/Dont4rootMe/new-optimizer-finding.git`.
- Base finalization-ветки: последний полный авторский snapshot
  `origin/adding-co-bench@641777d`; canonical EvolutionLoop в нём идентичен
  `origin/master` (blob указан в root `AGENTS.md`).
- Утверждения ниже про `HEAD=4ca9bbb` описывают исходный master-аудит; разделы
  с меткой **finalization live** описывают текущую overlay-ветку.
- Последние server-side refs были проверены read-only командой
  `git ls-remote origin` и совпали с имеющимися локально объектами remote refs.
  Обычный `git fetch --all --prune` во время аудита не смог записать
  `.git/FETCH_HEAD` из-за sandbox-ограничения. Поэтому вывод ниже основан на
  актуальных server refs и уже доступных полных Git-объектах.
- Upstream-тегов на момент аудита нет.
- В HEAD: 186 достижимых коммитов, 507 tracked-файлов, 41 файл
  `tests/test_*.py`.
- Рабочее дерево до этого аудита содержало пользовательский untracked-файл
  `scripts/run_circle_packing.sh`; он не является частью HEAD и не изменялся.
- До нового cluster run в workspace не найдено сохранённых `population_state.json`, `summary.json`,
  `metrics.json`, `correct.json` или каталогов `populations/`, `runs/`,
  `outputs/`, `stats/` с результатами реальных запусков.

### Finalization live: что изменено поверх `641777d`

- Алгоритм `EvolutionLoop` не переписывался: сохранены planning/retry/evaluation/
  island-selection/resume semantics, зафиксированные в `AGENTS.md`.
- Модельный слой вынесен из task evolver configs в composable
  `conf/backbone/*.yaml`. Одна Hydra-замена (`backbone=<profile>`) теперь меняет
  все стадии, не трогая задачу. `api_platforms/route.py` — единая route factory;
  `openai_compatible` backend обслуживает локальный SGLang и внешние совместимые
  endpoints. Stage aliases теперь применяются одинаково к pipelines и provider
  options.
- Canonical legacy profile — `ollama_qwen122_gemma31`; `mock` предназначен для
  contracts/tests; `deepseek_v4_flash_0731` направляет rationalization, design,
  implementation, novelty и repair только в один DeepSeek route.
- Каждый LLM call пишет append-only `population/llm_usage.jsonl`: route, stage,
  organism, generation, latency, provider raw usage, prompt/completion/total,
  cache read/write, uncached prompt, reasoning и speculative counters. CLI
  `python -m src.evolve.token_usage_report` строит totals и projections.
- Все evaluator reports проходят общий finite-score contract. Circle packing
  больше не принимает `reported_sum` через относительный tolerance и score
  всегда пересчитывает из radii. CO-Bench по умолчанию не раскрывает test split
  эволюции и отвергает non-finite dev score.
- Удалены восемь сгенерированных job notebooks и их generator: они дублировали
  launcher, использовали ошибочный для single coordinator `pytorch2`, были
  привязаны к Ollama и содержали plaintext credential. Credential не переносился;
  Comet теперь opt-in только через environment.
- Удалены ошибочно tracked `.tmp_manual_pipeline` outputs; runtime path теперь
  ignored. Root dump utilities перенесены в `scripts/analysis/`, а старый
  Ollama/torchrun launcher явно изолирован в `scripts/legacy/` вместо смешения
  с canonical binary-job path.
- `scripts/cluster/` — единый binary-job контур: pinned model revision, pinned
  SGLang, проверка ровно 8 H100, TP=8 server, smoke probe, EvolutionLoop,
  durable manifests и terminal monitor. Авторский server clone не используется.

## Обязательный протокол поддержки этой базы

1. Перед работой, требующей понимания архитектуры, экспериментов, результатов
   или веток, сначала читать этот файл, затем проверять утверждения по live-коду.
2. Любая задача, которая меняет или обнаруживает новое поведение, конфиг,
   prompt-контракт, experiment family, evaluator, ветку, результат, failure mode,
   эксплуатационную процедуру или ограничение, не завершена, пока обновлён
   соответствующий раздел этой базы.
3. При обновлении фиксировать дату и, для межветочного утверждения, commit/ref.
   Не переписывать текущий факт историческим: явно маркировать `live`,
   `upstream-only`, `historical`, `hypothesis` или `unknown`.
4. При конфликте источников приоритет такой:
   **live-код + composed Hydra config + тесты → эта база → `FRAMEWORK.md` /
   `README.md` → исторические документы в `agents/`**.
5. Не переносить сюда секреты, токены и ключи. Достаточно записать сам факт
   утечки и путь к проблемному конфигу.
6. После merge/rebase/fetch обновлять раздел «Git и upstream». После
   полноценного run — раздел «Результаты», включая preset, commit, seed,
   бюджет, лучший score, survival/failure breakdown и путь к артефактам.

## Кратко: что это за проект

`organism-framework` — task-blind платформа для LLM-driven поиска алгоритмов.
Единица эволюции — **organism**: структурированное текстовое описание гипотезы
(`genetic_code.md`) плюс исполняемая Python-реализация (`implementation.py`),
lineage и отчёты оценивания. Organisms живут в islands, рождаются mutation или
crossover, проходят LLM novelty gate, компилируются в код, оцениваются
task-specific evaluator'ами и отбираются по score.

Главная архитектурная граница:

- `src/` ничего не знает о конкретной задаче; он оперирует каталогами organisms,
  списками экспериментов и полем `score`;
- `experiments/` владеет task-specific API, валидацией и scoring;
- каждый evaluator обязан реализовать
  `evaluate_organism(organism_dir, cfg) -> dict`, а report обязан содержать
  `score`;
- canonical path — organism-first, island-aware `EvolutionLoop`;
- проект не должен превращаться в DDP/model-parallel training framework.
  Несколько GPU могут явно обслуживать независимые Ollama instances, но
  evaluator и orchestration остаются однопроцессными/однодевайсными по
  контракту.

В текущем master полностью присутствуют три family:

1. AtCoder AWTF2025 heuristic search;
2. packing 26 circles in a unit square;
3. optimization survey — поиск Python optimizer-controller'ов на наборе
   training-задач.

В `origin/adding-co-bench`, но не в master, добавлена четвёртая family:
CO-Bench для шести combinatorial-optimization задач.

## Git и все upstream-ветки

### Топология

```text
641777d  origin/adding-co-bench              (master + 12 commits)
   |
4ca9bbb  HEAD, origin/master                 (рабочая линия)
   |
b43d94e  origin/atcoder-heuristic-experiment
   |
b51afb6  origin/removing-genetic-sampling
   |
031b14b  origin/refactor-genetic-structure
   |\
   | 76d231c origin/refactor                 (старый side branch)
   |/
00b5e18  origin/square-fill-experiment
   |
dd45fb9  локальный master                    (устарел)
```

### Матрица веток

| Ref | Tip / дата | Отношение к `origin/master` | Смысл |
|---|---|---:|---|
| `origin/master` | `4ca9bbb`, 2026-06-01 | база | Текущий canonical runtime. HEAD совпадает с ним. |
| `origin/adding-co-bench` | `641777d`, 2026-06-26 | 0 behind / 12 ahead | Последний публичный и реально использованный автором project snapshot: CO-Bench, cluster notebooks и bootstrap tooling. Не merged; core `EvolutionLoop` идентичен master. |
| `origin/atcoder-heuristic-experiment` | `b43d94e`, 2026-06-01 | 2 behind / 0 ahead | Предок master; merge ref. Его tree не добавляет содержимого поверх уже merged `removing-genetic-sampling`. |
| `origin/removing-genetic-sampling` | `b51afb6`, 2026-05-31 | 3 behind / 0 ahead | Полностью merged. Название уже ветки её содержимого: здесь находятся P11–P13, bandits, pipelines, two-step design, token budgets, baseline-copy seed и удаление compatibility gate. |
| `origin/refactor-genetic-structure` | `031b14b`, 2026-04-20 | 86 behind / 0 ahead | Полностью merged историческая staged migration к section-aware genome. |
| `origin/square-fill-experiment` | `00b5e18`, 2026-04-14 | 135 behind / 0 ahead | Полностью merged ранняя circle-packing/AWTF линия. |
| `origin/refactor` | `76d231c`, 2026-04-19 | 94 behind / 4 ahead | Отдельный неслитый прототип typed segmented genome для circle packing. Merge-base с master: `0cb38e7`. |

Локальный `master` указывает на старый `dd45fb9` и отстаёт от
`origin/master`; ориентироваться по его имени нельзя. Локальная рабочая ветка,
несмотря на имя `square-fill-experiment`, на 135 коммитов впереди одноимённого
remote ref и совпадает с `origin/master`.

### Что вошло в master исторически

- `square-fill-experiment`: первоначальная circle-packing family, weighted
  species selection, novelty/repair, visualization и Ollama operations; затем
  ранний AWTF код.
- `refactor-genetic-structure`: восемь этапов перехода к структурированному
  `genetic_code.md`, family-specific schemas, section-aware compilation,
  novelty и тогда ещё compatibility gates, удаление legacy fallbacks.
- `removing-genetic-sampling`: анализ крупных AWTF runs, отказ от случайного
  pre-LLM pruning/merging genes, lineage-aware prompts, two-step design,
  adaptive bandits, pipeline routing, Comet/Plotly, queued speed work,
  file-copy seeds, удаление compatibility validator, resume-safe token
  accounting и stop budgets.
- `atcoder-heuristic-experiment` и два последних master merge-коммита в основном
  фиксируют интеграцию уже существующего tree.

### Что считать финальной версией при handoff

У проекта нет release/tag или commit с явной меткой «final», поэтому надо
разделять **финальный engine** и **последний полный project snapshot**.

- Финальный доступный engine — `src/evolve/evolution_loop.py` с blob
  `a4164058053a605e884b91748ab481dd542039df`. Последний изменивший его commit:
  `b51afb6` от 2026-05-31 (`stop criteria: optional per-model token budget`).
- Этот blob побайтно одинаков в `origin/master`, `origin/removing-genetic-sampling`,
  `origin/atcoder-heuristic-experiment` и более позднем
  `origin/adding-co-bench`. В последней ветке нет diff под `src/` или
  `api_platforms/`.
- Единственная альтернативная реализация `EvolutionLoop` находится в старом
  divergent `origin/refactor`; она на 94 master commits позади и не является
  следующей версией.
- Последний полный публичный snapshot автора — **`origin/adding-co-bench` at
  `641777d`**. Он добавляет family/config/cluster tooling поверх того же engine.
  Серверная рабочая копия автора также checkout'нута именно на этот commit.
- Public remote на 2026-08-04 содержит только PR refs 1–3, уже соответствующие
  merged ancestor branches; у `adding-co-bench` нет видимого PR ref.
- На локальном clone и серверном clone `git fsck --unreachable` не нашёл
  dangling commits. Серверный reflog заканчивается fast-forward на `641777d`.
  В доступном NFS найден ровно один clone этого проекта.
- Серверный working tree не содержит незакоммиченных изменений в `src/`,
  `experiments/`, `conf/`, `api_platforms/` или scripts. Отличия — только
  execution outputs в двух job notebooks, замена circle notebook копией с
  outputs и untracked population artifacts.

Практический вывод для финализации: базироваться следует на
`origin/adding-co-bench@641777d`, если CO-Bench входит в deliverable, либо на
`origin/master@4ca9bbb`, если scope — только исходные три family. В обоих
случаях используется один и тот же финальный `EvolutionLoop`; переносить код из
`origin/refactor` целиком не следует.

## Карта текущего master

- `src/main.py` — unified Hydra entrypoint: `mode=evolve` запускает canonical
  evolution, другие mode передаются standalone validation runner'у.
- `src/evolve/run.py`, `seed_run.py` — production entrypoints.
- `src/evolve/evolution_loop.py` — lifecycle generations, resume, planning,
  selection, bandit feedback, snapshots.
- `src/evolve/generator.py` — LLM stages, parsing, compilation, repair и
  persistence LLM artifacts.
- `src/evolve/orchestrator.py` — task-blind evaluation seam и subprocess jobs.
- `src/evolve/gpu_pool.py` — отдельные CPU/GPU resource queues.
- `src/evolve/{bandit,allocation,selection,scoring}.py` — adaptive sampling,
  experiment allocation и aggregation.
- `src/evolve/{storage,types}.py` — строгий filesystem/state contract.
- `src/evolve/{visualization,visualization_plotly,comet}.py` — run telemetry.
- `src/organisms/` — genetic-code parser, mutation/crossover bundles, novelty,
  rationalization, lineage regime, implementation patch compiler.
- `src/validate/` — запуск одного experiment evaluator в отдельном процессе.
- `api_platforms/` — route configs/factories, provider adapters, brokers, IPC,
  local workers и Ollama lifecycle.
- `experiments/` — три live task family и их runtime-контракты.
- `conf/` — явные top-level presets, evolver configs, experiment configs и
  prompt assets.
- `scripts/` — canonical shell wrappers для seed/evolve/Shinka/Ollama.
- `notebooks/manual_simple_scoring.ipynb`,
  `src/evolve/manual_pipeline.py` — ручная сборка prompt context и scoring;
  полезны для отладки, но не являются production lifecycle.
- `agents/` — старые audit/roadmap документы. Наиболее полный исторический
  snapshot — `agents/11_full_project_almanac_post_audit_and_platform_expansion.md`
  (2026-04-19), но он предшествует поздним изменениям.
- `FRAMEWORK.md` — хороший post-mortem и объяснение P-fixes, но уже не точная
  спецификация live runtime; список расхождений дан ниже.

Пакет имеет версию `0.1.0`, требует Python `>=3.10`. Базовые зависимости:
Hydra/OmegaConf, PyTorch/torchvision, matplotlib и pytest. Extras покрывают
audio, Hugging Face, LoRA, LLM/evolution stack, Plotly/Comet и ShinkaEvolve.

## Реальный canonical lifecycle

### Вход и конфигурация

Любой пользовательский entrypoint требует явный Hydra preset:

```bash
./scripts/seed_population.sh --config-name config_awtf2025_heuristic
./scripts/run_evolution.sh --config-name config_awtf2025_heuristic
./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka
python -m src.main --config-name config_optimization_survey mode=smoke
```

`ensure_root_runtime_config` требует `paths`, `experiments`, `resources`,
`api_platforms`, `evolver`. Не существует implicit `conf/config.yaml`.

### Generation 0: только `from_seed`

1. `evolver.islands.mode=from_seed` — единственный поддерживаемый режим.
2. Из плоского `island_ids` синтезируются topology labels. Текущий код не
   загружает research-bias prompts на island.
3. Один handwritten `seed_program_path` дословно копируется в `K × N`
   organisms. LLM на seed-фазе не вызывается.
4. Для каждого seed генерируется schema-valid, но намеренно placeholder
   `genetic_code.md`: по одному `baseline seed (file-copy ...)` bullet на
   обязательный раздел и `None` в optional section.
5. Каждый seed сразу проходит simple evaluation. При неудачах seed loop
   планирует top-up attempts, пока не наберёт заданное число успешных organisms
   на island или не исчерпает `max_organism_creations`.
6. Результат generation 0 и незавершённый `inflight_seed` persist'ятся.
   Повторный seed существующей завершённой population запрещён.

Следствие: стартовые copies внутри и между islands поведенчески одинаковы.
Island в live master — прежде всего topology/selection label; diversity должна
возникнуть после mutation/crossover.

### Каждая последующая generation

1. Loop требует существующий `population_state.json`; seed и evolve нарочно
   разделены.
2. Планируется фиксированное число offspring по доступным reproduction routes:
   mutation, within-island crossover, inter-island crossover. Невозможные routes
   исключаются, а веса остальных нормализуются.
3. Для каждого плана заранее создаётся organism stub, так что запланированная
   попытка считается в creation budget даже при последующей ошибке.
4. Creation tasks идут параллельно под
   `creation.max_parallel_organisms`. Как только один organism готов, его simple
   eval сразу ставится в resource queue, не ожидая окончания остальных LLM
   generations.
5. Evaluation experiments одного organism планируются последовательно;
   разные organisms исполняются конкурентно. CPU и GPU slots разделены.
   Evaluation GPU обязаны не пересекаться с GPU API-platform routes.
6. Old population и новые evaluated offspring объединяются. На каждом island
   остаётся top `max_organisms_per_island` по `simple_score`.
7. Если generation попадает в interval включённого Great Filter, simple
   survivors получают hard eval и затем island-local top-H по `hard_score`.
8. Обновляются lineage, bandits, `population_state.json`, PNG/HTML snapshots и,
   если включён, Comet.

Resume восстанавливает как завершённую population, так и
`inflight_generation`/`inflight_seed`; строгие canonical artifacts считаются
данными, а не кешем, который можно молча восстановить.

### Stop criteria

- `max_generations`;
- `max_organism_creations` — planned attempts, включая failed creation;
- `max_tokens_per_model[route_id]`.

Token usage читается из persisted organism metadata, поэтому лимит survives
restart. Активные shipped configs оставляют model token caps выключенными.

## Creation и LLM pipeline

### Mutation/crossover inputs

Production operators больше не вызывают случайные `prune_gene_pool` и
`merge_gene_pools`. LLM получает полный genome и реальный Python родителя
(для crossover — обоих), lineage, scores и change history, после чего напрямую
проектирует child. Старые helpers остались для manual/backward-compatible
flows. Поэтому `gene_removal_probability` и
`primary_parent_gene_inheritance_probability` фактически не управляют
production gene sampling, хотя config keys сохранены.

`num_inspirations` поддерживается: top-K survivors того же island могут
добавляться как reference programs. В активных AWTF и circle configs значение
равно **0**. Optimization prompts остались в legacy форме и получают `(none)`
для удалённых inherited/removed gene-pool placeholders.

### Two-step design

AWTF и circle mutation/crossover используют:

1. **Rationalization** — свободный plan из шести секций:
   `SCORE_BEARING_CORE`, `LINEAGE_REGIME_DIAGNOSIS`,
   `WEAKNESS_HYPOTHESIS`, `WHAT_TO_REMOVE`, `WHAT_TO_ADD_OR_INVENT`,
   `CHILD_DIRECTION`.
2. **Formalization** — строгий sectioned `genetic_code.md`, который должен
   материализовать plan.

Rationalization получает lineage-regime hint по последним ancestors. Она
persist'ится как `llm_rationalization.json`, кешируется внутри creation attempt
между novelty retries и soft-fail'ится: при проблеме Step 2 продолжает работу
со stub. Optimization survey rationalization prompts не имеет и работает
single-step.

### Структура genetic code

Верхний контракт одинаков:

- `CORE_GENES`;
- `INTERACTION_NOTES`;
- `COMPUTE_NOTES`;
- `CHANGE_DESCRIPTION`.

Family-specific subsections:

| Family | `CORE_GENES` subsections |
|---|---|
| AWTF | `STATE_REPRESENTATION`, `MACRO_STRATEGY`, `CONSTRUCTION_POLICY`, `LOCAL_REPAIR_POLICY`, `OPTIONAL_CODE_SKETCH` |
| Circle | `INIT_GEOMETRY`, `RADIUS_POLICY`, `EXPANSION_POLICY`, `CONFLICT_MODEL`, `REPAIR_POLICY`, `CONTROL_POLICY`, `PARAMETERS`, `OPTIONAL_CODE_SKETCH` |
| Optimization | `STATE_REPRESENTATION`, `GRADIENT_PROCESSING`, `UPDATE_RULE`, `PARAMETER_GROUP_POLICY`, `STEP_CONTROL_POLICY`, `STABILITY_POLICY`, `PARAMETERS`, `OPTIONAL_CODE_SKETCH` |

AWTF parser разрешает fenced code/pseudocode в substantive sections, кроме
plain-language `MACRO_STRATEGY`; поддержка bullet-wrapped fences была добавлена
после cascading failures.

### Gates, compilation и repair

- В live master есть только LLM **novelty validator** для mutation/crossover.
  Compatibility validator удалён из production code в commit `82492bc`.
  Некоторые неиспользуемые optimization compatibility prompt files остались,
  но config и generator их не вызывают.
- Novelty rejection может добавить critique и rejected candidate в следующий
  formalization prompt. В активных AWTF/circle configs
  `max_attempts_to_regenerate_organism_after_novelty_rejection=0`, поэтому
  rejection там сейчас terminal. Код поддерживает retry, если budget override
  больше нуля.
- AWTF и optimization используют scaffold regions и LLM artifacts `FULL` /
  `PATCH`. При изменении всех или всех кроме одного region compiler автоматически
  выбирает `FULL`; maternal implementation при этом всё равно виден модели как
  контекст.
- Circle использует отдельный single-rewrite contract: модель возвращает весь
  source для EVOLVE block/template, а не section patch.
- После extraction выполняется syntax/compile validation. Внутренняя
  implementation stage имеет initial attempt и repair attempts для
  parse/extraction failures.
- Если evaluator вернул runtime/contract error, post-eval repair получает
  Python, genome, change description, error history и последний stdout/stderr,
  переписывает code и запускает eval снова. Shipped budgets дают до двух таких
  recovery attempts.

Активные AWTF/circle creation settings:

```text
create attempts = 1
novelty regenerations = 0
post-eval repairs = 2
```

То есть текущий operational режим предпочитает throughput, а не дорогие design
retries.

## Selection, allocation и bandits

Parent selection использует `weighted_rule`: fitness комбинируется с балансом
parent usage; активный selection score фактически равен `simple_score`, потому
что `inheritance_fitness` имеет вес 0.

Optimization simple phase использует conditional-Poisson/Neyman allocation:
из 12 задач на organism выбираются 2, variance оценивается по истории, а итог
агрегируется с inclusion probabilities. Failed experiments дают нулевой вклад;
если успехов нет, organism phase считается failed.

Adaptive sampling реализован discounted Thompson sampling с Beta posteriors:

- origin/parent island;
- conditional partner island для inter-island crossover;
- LLM arm — либо route, либо целая named pipeline.

Defaults bandit'ов в активных family: discount `0.97`, prior `Beta(1,1)`,
reward `score_quantile`, window 50; доступны `survival` и `hybrid`. Старые
наблюдения discount'ятся, failed organisms получают reward 0. Partner sampler
отдельный для каждого origin island. State и reward windows находятся под
`population_state.json.bandit_state`.

Если `evolver.llm.pipelines` непуст, один pipeline выбирается на organism, и
legacy per-stage route sampler обходится. Все pipeline обязаны заполнить пять
canonical stages: rationalization, design, implementation, novelty, repair.

Дегенеративные конфигурации, которые важно понимать:

- AWTF имеет один pipeline `qwen_creative_gemma_check`; pipeline bandit там
  формально включён, но учить нечего — один arm.
- Circle имеет два осмысленных pipeline arm: `gemma_only` и `qwen_only`.
- Circle имеет один island, поэтому parent/partner bandits одно-arm; inter-island
  crossover невозможен, несмотря на ненулевой config weight.
- Upstream CO-Bench тоже имеет один island и один pipeline.

## Canonical filesystem artifacts

Population layout:

```text
<population_root>/
  population_state.json
  seed_run.log
  run.log
  .eval_config/config.yaml
  gen_0000/
    island_<id>/
      org_<id>/
        implementation.py
        genetic_code.md
        lineage.json
        organism.json
        summary.json
        llm_request.json
        llm_response.json
        llm_rationalization.json   # только two-step offspring
        results/simple/*.json
        results/hard/*.json
        logs/*.out
        logs/*.err
```

Не каждый stage обязан создать каждый optional LLM artifact, но семь файлов из
root `AGENTS.md` являются стабильным canonical contract. `organism.json`
содержит status, parents, score refs, pipeline attribution и token usage.
`summary.json` содержит phase aggregation и experiment report index.
`population_state.json` содержит active organisms, best, relationship history,
inflight plans и bandits.

## Finalization live: реестр и аудит задач

Машиночитаемый source of truth — `experiments/catalog.py`. Команды:

```bash
python -m experiments.catalog list
python -m experiments.catalog markdown
python -m experiments.catalog check
python -m experiments.catalog check --require-ready
```

`check` compose'ит каждый Hydra preset, импортирует evaluator, проверяет точную
сигнатуру `evaluate_organism(organism_dir, cfg)`, seed/prompt paths и полноту
optimization registry. Он разделяет логические ошибки (`ERROR`) и отсутствующие
внешние prerequisites (`BLOCKED`): это важно, чтобы missing dataset не выглядел
как некорректная постановка задачи.

На локальном аудите 2026-08-04: **25 specs, 22 enabled; error=0, blocked=26,
warning=1, info=3**. Blockers — отсутствующие CO-Bench checkout+data для шести
задач (12) и baseline profiles для четырнадцати enabled optimization tasks
(14). Три heavy optimization задачи намеренно disabled. Единственное научное
предупреждение: AWTF fixed-corpus score — воспроизводимый surrogate, но не
AtCoder relative leaderboard score. Отсутствовавший во всех Git refs
optimization seed восстановлен как task-blind `SeedAdam` с canonical
`build_optimizer` contract.

| Task ID | Default | Phase | Candidate contract | Evolution score |
|---|---:|---|---|---|
| `circle_packing_shinka/unit_square_26` | yes | simple | `run_packing() -> (centers[26,2], radii[26], reported_sum)` | `sum(radii)`, max; strict feasibility |
| `awtf2025_heuristic/group_commands_and_wall_planning` | yes | simple | `solve_case(input_text: str) -> str` | negative mean absolute fixed-corpus score, max |
| `co_bench/TSP` | yes | simple | `solve(**instance) -> solution: dict` | finite normalized dev score, max |
| `co_bench/BIN_PACKING_1D` | yes | simple | same CO-Bench contract | finite normalized dev score, max |
| `co_bench/MULTI_KNAPSACK` | yes | simple | same CO-Bench contract | finite normalized dev score, max |
| `co_bench/SET_COVERING` | yes | simple | same CO-Bench contract | finite normalized dev score, max |
| `co_bench/GRAPH_COLOURING` | yes | simple | same CO-Bench contract | finite normalized dev score, max |
| `co_bench/JOB_SHOP` | yes | simple | same CO-Bench contract | finite normalized dev score, max |
| `optimization_survey/synthetic_logreg` | yes | simple | `build_optimizer(model, max_steps) -> controller` | harmonic quality/speed ratio, max |
| `optimization_survey/mnist_mlp` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/poly_regression` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/rosenbrock_net` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/xor_mlp` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/sin_regression` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/matrix_factorization` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/tiny_autoencoder` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/two_spirals` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/linear_denoiser` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/conv1d_classify` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/quadratic_bowl` | yes | simple | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/cifar_convnet` | yes | great_filter | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/audio_transformer` | no | great_filter | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/minigpt_wikitext2` | yes | great_filter | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/ddpm_cifar10` | no | great_filter | same optimizer contract | harmonic quality/speed ratio, max |
| `optimization_survey/lora_sft` | no | great_filter | same optimizer contract | harmonic quality/speed ratio, max |

## Live experiment families

### AWTF2025: group commands and wall planning

- Preset: `conf/config_awtf2025_heuristic.yaml`.
- Candidate API: `solve_case(input_text: str) -> str`.
- Фиксированный corpus: 100 committed inputs, IDs 0–99; smoke — 0–4.
- Validator проверяет walls, robot groups и operation sequence.
- Official absolute objective:
  `T + 100 * Σ Manhattan(final_position_k, target_k)`.
- Experiment minimизирует mean absolute objective, но framework всегда
  максимизирует report `score`, поэтому `score = -mean_absolute_score`.
- Per-case soft timeout: 1 second. Дополнительный per-case report сохраняется
  JSON artifact'ом.
- CPU evaluation: до 25 parallel jobs; LLM routes занимают GPU 0–7.
- Islands: `macro_partitioning`, `staged_routing_repair`; по 5 одинаковых seed
  copies, capacity 10 на island.
- 6 offspring/generation, shipped ceiling 700 generations.
- Operator weights: within `0.4`, inter `0.1`, mutation `0.5`.
- Great Filter выключен.
- Inspirations выключены.
- Default backbone profile `ollama_qwen122_gemma31`: Qwen 3.5 122B для
  rationalization/design/implementation, Gemma 4 31B для novelty/repair.

Названия islands отражают исторические research regimes, но live seed prompts
для них отсутствуют; различие не закодировано в generation-0 organisms.

### Circle packing: 26 circles

- Preset: `conf/config_circle_packing_shinka.yaml`.
- Candidate API:
  `run_packing() -> (centers, radii, reported_sum)`.
- Строгий контракт: ровно 26 centers shape `(26,2)`, radii shape `(26,)`,
  finite/nonnegative values, containment in unit square, no overlap,
  `reported_sum == radii.sum()` только с absolute tolerance (`rtol=0`).
- `score = evaluator-computed sum(radii)`, направление max; self-report никогда
  не является score и сохраняется только как diagnostic.
- Extra artifact — `.npz` с centers/radii/result.
- Один island `packing_default`, 5 seed copies, capacity 10.
- 6 offspring/generation, shipped ceiling 700 generations (300-generation
  DeepSeek run задаётся явным cluster override).
- Config weights: within `0.7`, inter `0.2`, mutation `0.2`; inter route
  недоступен при одном island и исключается планировщиком.
- CPU evaluation до 20 jobs. GPU ownership задаётся backbone profile, а не
  task config; legacy Ollama и 8×H100 DeepSeek используют разные profiles.
- Great Filter и inspirations выключены.
- Default profile использует Qwen для creative stages и Gemma для checks;
  DeepSeek profile заменяет весь pipeline одной моделью.

### Optimization survey

- Preset: `conf/config_optimization_survey.yaml`.
- Candidate API:
  `build_optimizer(model, max_steps) -> controller`.
- Controller обязан иметь
  `step(weights, grads, activations, step_fn)` и `zero_grad(set_to_none=True)`.
- Baseline program:
  `experiments/optimization_survey/_baselines/initial_program.py`.
- 4 islands: gradient, adaptive, quasi-Newton, second-order; по 5 seeds,
  capacity 5.
- 10 offspring/generation, максимум 100 generations.
- Deterministic operator schedule с равными weights; uniform island sampling.
- Shipped LLM route — `mock`, то есть preset пригоден для contracts/tests, но
  не является готовой production-модельной кампанией без override.

Simple phase выбирает 2 из следующих 12 enabled задач:

| Задача | Роль |
|---|---|
| `synthetic_logreg` | synthetic logistic regression |
| `mnist_mlp` | MNIST MLP |
| `poly_regression` | polynomial regression |
| `rosenbrock_net` | Rosenbrock objective network |
| `xor_mlp` | XOR |
| `sin_regression` | sine regression |
| `matrix_factorization` | low-rank factorization |
| `tiny_autoencoder` | small autoencoder |
| `two_spirals` | nonlinear classification |
| `linear_denoiser` | denoising |
| `conv1d_classify` | 1-D convolution classification |
| `quadratic_bowl` | param-only quadratic objective |

Great Filter включён каждые 5 generations, top 3 на island, и перечисляет:

- enabled: `cifar_convnet`, `minigpt_wikitext2`;
- disabled by default: `audio_transformer`, `ddpm_cifar10`, `lora_sft`.

Перед реальным run следует проверить, как disabled entries фильтруются в
composed phase, и подготовить baseline profiles.

Per-experiment score требует `objective_name=train_loss`,
`objective_direction=min`, finite `objective_last` и baseline:

```text
quality_ratio = baseline_last / candidate_last
speed_ratio   = baseline_steps / first_step_at_or_below_baseline
score         = harmonic_mean(quality_ratio, speed_ratio)
```

Если baseline отсутствует, score не вычисляется. Если candidate никогда не
достиг baseline, speed ratio и итоговый score равны 0. `mode=stats` строит
baseline profile, а не оценивает evolutionary quality.

## API platforms, execution и telemetry

В исходном master зарегистрировано 19 route family:

- cloud: Claude Opus/Sonnet/Haiku и GPT-5.4 / mini / nano;
- local/HF: Qwen 3.5 27B, distilled Qwen, Qwen 35B A3B, Gemma 4 26B/31B;
- Ollama: Nemotron Cascade 30B, Qwen 27B/35B/122B, Gemma 26B/31B;
- `mock`, `mock_local`.

Registry запускает по route broker subprocess/Unix-socket service, управляет
leases/concurrency и local workers. Ollama configs поддерживают несколько
явных GPU groups/instances. Orchestrator валидирует, что evaluation GPU не
пересекаются с LLM GPU.

Finalization live добавляет generic config-owned factory и
`openai_compatible` backend, не ломая import compatibility старых wrappers.
Named pipeline canonicalizes конкретные stage labels, например
`design_attempt -> design`, `design_rationalization -> rationalization`,
`novelty_check -> novelty`.

Закрытый drift: provider-side `stage_options` теперь использует тот же alias
fallback. Exact concrete override по-прежнему имеет приоритет, иначе
`design_rationalization` получает `rationalization`, `novelty_check` —
`novelty`, `repair_attempt` — `repair`.

### DeepSeek-V4-Flash-0731 / 8×H100

- Exact model: `deepseek-ai/DeepSeek-V4-Flash-0731`, pinned Hugging Face
  revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Serving runtime: SGLang `0.5.16`, CUDA runtime `cu126`, single node, tensor parallel 8, official
  bundled `DSPARK` draft head, reasoning parser `deepseek-v4`, tool parser
  `deepseekv4`, `temperature=1.0`, `top_p=0.95` in the experiment profile.
- CUDA packaging is deliberately pinned to the persistent runtime ID
  `sglang-0.5.16-cu126`. SR008 is heterogeneous: the 8×H100 allocation below
  exposed CUDA Driver API 12.6, while one later 1×H100 probe exposed driver
  `580.105.08`. The default SGLang PyPI stack (`torch 2.11.0+cu130`) therefore
  fails nondeterministically by node. The bootstrap reproduces the official
  SGLang 0.5.16 CUDA-12 Docker substitutions (`cuda-python<13`,
  `flashinfer[cu12]`, cu126 PyTorch, and published cu129 Hopper
  SGLang-kernel/DeepGEMM wheels), checks real `torch.cuda` initialization, and never reuses
  the incompatible legacy `sglang-0.5.16` environment. Primary references:
  [SGLang v0.5.16 Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.16/docker/Dockerfile),
  [NVIDIA CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/).
- Before any GPU Python process starts, `scripts/cluster/cuda_driver_env.sh`
  removes only `LD_LIBRARY_PATH` components named `compat` (and unsafe empty
  components). ML Space's base image puts CUDA 12.6 forward-compat
  `libcuda.so.560.35.05` ahead of the scheduler-mounted R580 driver; that exact
  mismatch causes CUDA Error 803. Native CUDA/NCCL/HPC-X/NVIDIA paths remain
  intact, and the sanitized environment is recorded in both run and SGLang
  launch manifests.
- The canonical regional runtime is now published at
  `/home/jovyan/evolutionloop-deepseek-v4/runtime/sglang-0.5.16-cu126`.
  Its live H100 gate records Python `3.11.14`, Torch `2.11.0+cu126`, CUDA
  `12.6`, `cuda-python 12.9.7`, SGLang `0.5.16`, SGLang kernel
  `0.4.5+cu129`, DeepGEMM `0.1.4.post1+cu129`, and FlashInfer `0.6.14`;
  `.runtime-freeze.txt` contains 206 distributions. The canonical bootstrap
  reuse gate passed after the atomic promotion. SGLang's installed wheel
  metadata still declares its PyPI default `cuda-python>=13`; `pip check`
  therefore reports that one expected mismatch even though the audited
  CUDA-12 dependency view is intentional. Do not "fix" it by installing
  CUDA 13 on the heterogeneous SR008 pool.
- DeepGEMM JIT is a separate compiler boundary. The exact production job
  `lm-mpi-job-9025f0bf-80f1-4cb6-a32d-2b2fbbeecd1c` proved that the cu126
  serving runtime loads all 48 checkpoint shards on eight H100s, but the base
  image's `/usr/local/cuda-12.6/bin/nvcc` rejects the `__int128_t` `"q"`
  operand in `st.shared.b128` while prewarming DeepSeek-V4 MHC prenorm (all
  eight TP ranks, 21 `n_splits` buckets). SGLang exited before endpoint
  readiness; EvolutionLoop never started, no population was created, and token
  usage remained zero. This is a runtime diagnostic, not an experiment result.
- The corrective path pins an isolated NVIDIA conda compiler prefix
  `toolchains/cuda-nvcc-12.9.86`, selected only through
  `DG_JIT_NVCC_COMPILER`; it does not alter PyTorch cu126 or
  `LD_LIBRARY_PATH`. `bootstrap_cuda_toolchain.sh` requires an SM90a
  128-bit-store cubin smoke, and the job runs a numerical
  `tf32_hc_prenorm_gemm` smoke before loading the model. Compiled kernels use
  the persistent `kernel_cache/deep_gemm-sm90-cuda-nvcc-12.9.86` namespace.
  This design follows DeepGEMM's own `>=12.9` performance recommendation and
  SGLang's NVCC default; the NVRTC alternative remains disabled because
  upstream explicitly warns that it can reduce performance. Cluster validation
  of this fix is in progress and must replace this sentence with the job ID and
  measured outcome.
- H100 constraint: use stock official FP4 checkpoint and explicitly pin
  SGLang's Hopper W4A16/Marlin runner. Do not force `flashinfer_mxfp4` or other
  Blackwell-only FP4 kernels. BF16 compressed state reduces KV-state memory;
  serving starts conservatively at `mem-fraction-static=0.88`, chunked prefill
  8192, context 65536, max running requests/cuda graph batch 8, SWA full-token
  ratio 0.1. Values must be changed only after workload measurements.
- Scheduler: Cloud.ru ML Space (не Slurm), region `SR008`, instance
  `a100plus.8gpu.80vG.96C.1456G`, one worker, `type=binary`, large shared memory,
  detached, `processes_per_worker=1`. Без последнего параметра даже `binary`
  был экспериментально запущен на восьми MPI ranks; rank guard остаётся
  defense-in-depth. `pytorch2` запрещён для этого path.
- Jupyter NFS и SR008 regional NFS — разные namespaces. Canonical submit
  передаёт validation-safe one-line bootstrap, клонирует public Git repository
  в `/home/jovyan/evolutionloop-deepseek-v4/source/<full-commit>`, fetch/checkout
  делает строго по SHA и сверяет `HEAD` до исполнения. Runtime/model/run state
  остаётся на persistent regional NFS; terminal artifacts возвращаются через
  `scripts.cluster.transfer`.
- Run artifacts: `gpu_inventory.json`, `model_snapshot.json`,
  `sglang_launch.json`, `sglang.log`, `smoke.json`, `evolution_launch.json`,
  `run_manifest.json`, population artifacts, token summary и monitor history/
  completion event. Повторный job с тем же run directory использует canonical
  EvolutionLoop resume state.
- На resource check 2026-08-04 обе 8-GPU H100 SKU показывали 0 свободных
  workers; это dynamic capacity, поэтому job допустимо ставить в очередь, но
  факт `Pending` не является подтверждением inference.

Visual output:

- composite `evolution_overview.png`;
- standalone overview/timeline/survival PNG groups;
- score/token/runtime/cumulative plots;
- Plotly interactive lineage view с hover по maternal ancestor chain;
- Comet upload composite, panels и HTML.

AWTF и circle top-level configs сейчас включают Comet. В них также committed
plain-text API credential. Ключ намеренно не воспроизводится здесь; его следует
немедленно revoke/rotate и заменить на `${oc.env:COMET_API_KEY,...}`.

`scripts/run_evolution.sh` умеет `--seed`, проверяет population state,
поднимает/останавливает Ollama и требует preset. `scripts/seed_population.sh`
создаёт только generation 0.

## ShinkaEvolve baseline: что именно сравнивается

ShinkaEvolve здесь не является ещё одной experiment task. Это **альтернативный
search/evolution engine**, задуманный как baseline против собственного
`EvolutionLoop` этого репозитория:

```text
наш EvolutionLoop
  structured genetic code → rationalization → formalization → novelty
  → implementation compiler → post-eval repair → island/bandit selection

против

upstream ShinkaEvolve
  прямые LLM edits EVOLVE-BLOCK в одной Python-программе
  → собственные islands/database/UCB1/model selection
```

Для AWTF и circle обе линии используют один и тот же task evaluator, одно поле
`score` и сейчас стартуют из того же family seed-файла под
`experiments/<family>/_baselines/shinka/initial_program.py`. Shinka adapter
временно кладёт предложенную программу как `implementation.py`, инстанцирует
тот же experiment config и записывает ожидаемые upstream-файлы
`correct.json`/`metrics.json`. Поэтому objective values технически сопоставимы.

Но shipped presets не задают честный budget-matched A/B:

| Family | Собственный loop | Shinka preset |
|---|---|---|
| AWTF | 150 generations, 2 islands, 6 offspring/generation, hybrid Qwen-creative/Gemma-check pipeline | 75 generations, 4 islands, Qwen 122B only, 4 proposal/eval jobs |
| Circle | 150 generations, 1 island, 6 offspring/generation, bandit между Gemma-only и Qwen-only pipelines | 50 generations, 4 islands, Shinka UCB1 по доступным Ollama instances |

Они также различаются числом LLM calls на candidate, retry/repair semantics и
population mechanics. Чтобы получить научное сравнение, нужно отдельно
выровнять хотя бы seed, evaluator corpus, total evaluations, per-model tokens
или wall time, model set и random seeds.

Готового comparator/report builder нет: собственный loop пишет
`populations/...`, Shinka — `shinka_runs/<family>/shinka.db` и snapshots.
Пользователь должен запустить обе линии и затем сопоставить best score,
score-vs-evaluations, token usage, survival rate и wall time. В Git такого
парного результата сейчас нет.

## Upstream-only: `origin/adding-co-bench`

Состояние: commit `641777d`, 12 commits поверх audited master,
70 changed files, примерно `+4845/-82`. В master этого нет.

### Что добавлено

- Одна reusable `co_bench` family и один preset `config_co-bench`.
- Selector `experiments.co_bench.CO_BENCH_TASK`.
- Шесть зарегистрированных задач:
  TSP, 1-D bin packing, multidimensional knapsack, set covering, graph
  colouring, job-shop scheduling.
- Для каждой задачи собственный handwritten seed и project-context prompt;
  evaluator/runtime и остальные prompts shared.
- Candidate contract: `solve(**kwargs) -> dict`. `**kwargs` обязателен;
  required positional-only params запрещены, обычные keyword-fillable required
  params допустимы.
- Evaluator делегирует официальному CO-Bench `Evaluator`; evolution score —
  `dev_score` (max), `test_score` сохраняется как diagnostic.
- External CO-Bench не pip package. Нужен checkout через `COBENCH_ROOT` либо
  `third_party/CO-Bench` и dataset под
  `${AIFS_DATA_ROOT}/co-bench`. Bootstrap script клонирует repo, ставит extra и
  скачивает Hugging Face dataset.
- Optional extra включает Hugging Face Hub, NumPy, OR-Tools, NetworkX, PuLP,
  SciPy.
- Missing checkout/data превращаются в понятную optional-dependency ошибку;
  dataset integration test skip'ается, если ресурса нет.
- 1 island, 5 seeds, capacity 10, 6 offspring/generation, 300 generations;
  Great Filter off, inspirations 0, Qwen-creative/Gemma-check pipeline.
- Shinka baseline adapter.

### Cloud/cluster tooling

- Генератор и восемь job notebooks: AWTF, circle и шесть CO-Bench tasks.
- `scripts/legacy/ollama_torchrun_job.py`: сохранённый rank-0-only launcher
  для historical environments, где
  `torchrun` стартует процесс на каждый GPU. Все nonzero ranks завершаются, а
  один orchestrator получает видимость всех GPU. Это предотвращает N
  конкурирующих evolution loops и не вводит DDP.
- `scripts/create_env.sh`, CO-Bench bootstrap и auto-provision последнего
  official Ollama `.tar.zst`, lock-guarded shared environment/cache setup.
- Branch также повышает AWTF/circle `max_generations` с 150 до 700 и переводит
  circle с Qwen 35B на 122B setup.

Никаких committed CO-Bench score tables или completed population artifacts в
ветке не найдено; наличие code/tests не является доказательством качества
решений.

## Upstream-only: `origin/refactor`

Состояние: commit `76d231c`, 4 unique commits от merge-base `0cb38e7`,
но ветка на 94 master commits позади. Diff своей линии:
45 files, примерно `+6647/-488`. Не merged и не совместима напрямую с поздним
runtime без переноса.

Это исследовательский прототип typed CirclePacking genome:

- canonical machine-readable `genome.json`;
- детерминированно rendered `genetic_code.md`;
- `compatibility_report.json` и `functional_checks.json`;
- schema `typed_segmented_genome` v1;
- восемь slots: layout, selection, radius initialization, growth, conflict,
  repair, boundary, termination;
- linkage groups: placement, dynamics, control;
- закрытая library примерно из четырёх modules на slot (пять repair modules),
  с typed state inputs/outputs, enums, pre/postconditions, invariants,
  compatibility tags и inheritance units;
- hard/soft compatibility и functional checks;
- JSON contracts для seed/mutation/crossover/novelty и task-blind hypothesis
  artifact integration;
- большой набор targeted schema/prompt/compatibility tests.

Live master выбрал другую эволюционную линию: Markdown section schemas,
LLM-authored mechanisms и отсутствие compatibility validator. Поэтому
`origin/refactor` следует рассматривать как библиотеку идей/альтернативный
prototype, а не «новее master». Empirical run results для него не committed.

## Какие результаты действительно известны

### Чего в репозитории нет

Нет committed финальной таблицы best organisms, score trajectory или
post-fix benchmark comparison ни для AWTF, ни для circle, optimization survey,
CO-Bench или typed refactor. Поэтому по текущему checkout нельзя честно
утверждать, что найденный алгоритм превзошёл baseline или что rationalization,
bandits либо pipeline routing улучшили objective.

Это утверждение относится к Git: результаты не закоммичены. На доступном
Cloud.ru NFS 2026-08-04 найдены реальные population artifacts, описанные ниже.

### Remote population runs, не сохранённые в Git

Источник: server checkout
`/home/jovyan/echimbulatov/fork_afedorov/constant_repos/new-optimizer-finding`
на `adding-co-bench@641777d`. Все четыре run использовали `seed=42`,
`max_generations=700`, 6 offspring/generation и pipeline
`qwen_creative_gemma_check`; exact `EvolutionLoop` blob совпадает с master.

| Run | Состояние | Organisms / scored | Seed → best | LLM tokens |
|---|---|---:|---:|---:|
| `populations_first_attempt/atcoder_awt` | interrupted: finalized gen 148, inflight gen 149 | 904 / 457 (50.6%) | absolute cost 12166.87 → **8446.13**; 30.58% lower | 39,065,869 |
| `populations_first_attempt/circle-packing` | interrupted: finalized gen 93, inflight gen 94 | 569 / 438 (77.0%) | radii sum 2.028 → **2.5727894626**; +26.86% | 34,930,220 |
| `populations/atcoder_awt` | completed gen 700 on 2026-07-08 UTC | 4210 / 2149 (51.0%) | absolute cost 12166.87 → **9185.05**; 24.51% lower | 198,427,180 |
| `populations/circle-packing` | completed gen 700 on 2026-07-12 UTC | 4205 / 2084 (49.6%) | radii sum 2.028 → **2.4748327059**; +22.03% | 307,278,041 |

Лучшие remote artifacts:

- first AWTF:
  `gen_0119/island_staged_routing_repair/org_c027e3c915264916b7d632324a4b2421`;
- first circle:
  `gen_0070/island_packing_default/org_c77b3135e2bf4bbd8d031365f012bc9a`;
- completed AWTF:
  `gen_0429/island_macro_partitioning/org_57688f60e1364dca9f02c743955d58be`;
- completed circle:
  `gen_0638/island_packing_default/org_5fde2f7f59c94c09aa534f88af22e28e`.

Два важных вывода:

1. Более короткие interrupted attempts нашли **лучшие** objectives, чем
   последующие completed 700-generation runs. «Последний run» здесь не означает
   «лучший найденный organism».
2. Failure rate остаётся около 49–50% в обоих completed runs. Completed AWTF:
   1085 failed creation + 976 failed simple eval; completed circle:
   1035 failed creation + 1086 failed simple eval. Доминируют malformed
   top-level sections, Markdown/bullet parser failures, invalid Python,
   novelty rejection и exhausted reasoning budget.

Результаты EvolutionLoop достоверно улучшили собственный file-copy seed на том
же evaluator. Они **не доказывают превосходство над ShinkaEvolve**: в обоих
executed cluster notebooks normal-run cells имеют execution counts, а все
Shinka baseline cells остались неисполненными; на сервере отсутствуют
`shinka_runs/` и `shinka.db`.

Перед очисткой/пересозданием NFS эти четыре population roots и особенно четыре
best-organism directories надо архивировать. Сейчас они существуют только на
remote storage и не защищены Git history.

### Active finalization run: DeepSeek-only circle packing

Это operational state, **ещё не результат** до terminal success и проверки
артефактов:

| Поле | Значение |
|---|---|
| Code | `finalize/evolutionloop-deepseek-v4@b3f6383f780794922cbb2d5589ecdf0f46815351` |
| Scheduler job | `lm-mpi-job-34dd8b84-8165-4c05-8674-88448517035e` |
| Submitted | 2026-08-04 01:23 UTC |
| Terminal state | `Failed` at 01:35:57 UTC before Python/model start |
| Preset | `config_circle_packing_shinka`, seed 42 |
| Backbone | только `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01…6062` |
| Budget | generation 0 + EvolutionLoop through generation 300; 6 offspring/generation; max 8 concurrent organisms |
| Run root | `/home/jovyan/echimbulatov/fork_afedorov/constant_repos/optimizer_cluster_runs/deepseek-v4-flash-0731-circle-300-b3f6383` |
| Monitor | detached PID recorded in `monitor.pid`; `monitor_status.json`, append-only history, terminal `completion_event.json` |

Diagnostic result: SR008 did allocate the requested worker, but the job-visible
NFS namespace did not contain the Jupyter-side checkout. Logs also proved that
omitting `processes_per_worker=1` invokes a `binary` command on eight MPI ranks
(`[1,0]`…`[1,7]`). No model/evolution compute ran. Remediation is covered by
regression tests: exact-commit HTTPS bootstrap into regional NFS, one scheduler
process per worker, nonzero-rank defense before shared state, and devices 0–7
for the sole rank-0 TP server. Do not relabel this failed probe as a run.

Remediated submission (active):

| Поле | Значение |
|---|---|
| Code | `3990b5f3e0a589658f5ec7032437e282dff03a79` |
| Scheduler job | `lm-mpi-job-62223f05-33f0-432d-a951-3aa9274db98d` |
| Submitted/state | 2026-08-04 01:41 UTC; `Failed` at 01:43 UTC before workers became READY |
| Requested clone | `/home/jovyan/echimbulatov/new-optimizer-finding-finalize`; not observed by user code |
| Run root | `/home/jovyan/echimbulatov/optimizer_cluster_runs/deepseek-v4-flash-0731-circle-300-3990b5f` |
| Monitor | detached, 60-second scheduler/population snapshots |

The second scheduler allocation reached node assignment but never reached
`All workers are READY` and emitted no user-script line. This is an
infrastructure/startup failure, so it neither validates nor invalidates that
specific top-level path. A subsequent 1×H100 namespace probe
`lm-mpi-job-ac92b7b8-6577-4dff-b322-8518b036075e` completed and settled the
question: inside SR008 only `/home/jovyan` existed; every tested
`/home/jovyan/echimbulatov/...` path was absent.

Bootstrap probes on 2026-08-04:

| Job | Outcome / finding |
|---|---|
| `lm-mpi-job-e7ce9b6a-8ffc-4d67-897f-98fdfaf4dbdd` | Data Transfer destination appeared as a directory but no copied object/object-log was available; inbound Data Transfer is not the source-code path. |
| `lm-mpi-job-4c5533eb-d518-4710-bb16-d2d50da60ab5` | One-line Internet git bootstrap was accepted and scheduler-completed with `processes_per_worker=1`; exact-commit checkout path is on regional NFS. |
| `lm-mpi-job-27f29ff0-b863-4bef-8145-1c283b941663` | Cross-allocation persistence verified: a later worker read exact `HEAD=7beb67474dd8e6855baaf1f04e0cdbfb461ee836` from the earlier job's regional checkout. |

Current production submission:

| Поле | Значение |
|---|---|
| Code | `6b7ae1e24e8fb26afb45acf417e481403db1851d` |
| Scheduler job | `lm-mpi-job-1024dcb2-e505-4e8a-a255-3805acee66e7` |
| Submitted/state | 2026-08-04 02:30:48 UTC; `Failed` at 02:43:16 UTC before server readiness |
| Regional source | `/home/jovyan/evolutionloop-deepseek-v4/source/6b7ae1e24e8fb26afb45acf417e481403db1851d` |
| Regional run | `/home/jovyan/evolutionloop-deepseek-v4/runs/deepseek-v4-flash-0731-circle-300-6b7ae1e` |
| Scheduler contract | one `a100plus.8gpu.80vG.96C.1456G` worker, `processes_per_worker=1`; run state lives directly on regional NFS |
| Control monitor | PID `22377`; 60-second snapshots and terminal completion event under Jupyter-side `runs/optimizer_cluster_runs/...` |

The immutable checkout, exactly one coordinator, eight-H100 inventory, runtime
bootstrap and complete 74-file model snapshot were all validated. The serving
environment resolved `torch 2.11.0+cu130`; this allocation's driver exposed
CUDA Driver API `12.6`, so SGLang stopped before readiness with
`No accelerator ... available`. `EvolutionLoop` never started and emitted zero
usage events. The model snapshot remains cached on regional NFS. The terminal
monitor correctly wrote a failed `completion_event.json` at 02:43:38 UTC.
This is an infrastructure/runtime diagnostic, never an experiment result.

CUDA heterogeneity probe:

| Job | Outcome / finding |
|---|---|
| `lm-mpi-job-5a8a57c5-b24b-493f-aa63-5939eb3ade64` | One 1×H100 allocation completed and reported driver `580.105.08`, H100 compute capability 9.0. Together with the earlier 12.6 API failure this proves allocations cannot assume one driver branch; portable production runtime is cu126. |
| `lm-mpi-job-227b2d27-71af-4892-bed5-b794bfaa2559` | First cu126 bootstrap reached Python packaging, then failed because the CUDA-12 SGLang Dockerfile's historical `sglang-kernel` cu124 URL is no longer a published GitHub asset. No runtime was promoted. The audited resolver now uses the release's published cu129 SM90 wheel. |
| `lm-mpi-job-cf2b9e16-dd7f-4e70-93b7-884c9f6cefff` | Built the complete SGLang 0.5.16 stack (`torch 2.11.0+cu126`, CUDA-12 FlashInfer, cu129 Hopper kernels), but the final live-CUDA gate failed with Error 803. The incomplete environment was intentionally preserved at `/home/jovyan/evolutionloop-deepseek-v4/runtime/sglang-0.5.16-cu126.building.20260804T031427Z.120`. |
| `lm-mpi-job-b280fbee-e4fa-4f94-9607-7edc10bae6b9` | Controlled libcuda experiment on driver `580.105.08`: original path selected `/usr/local/cuda-12.6/compat/libcuda.so.560.35.05` and `torch.cuda.is_available()` was false with Error 803; removing only the `compat` path made the same cu126 runtime immediately detect one H100. Prepending `/usr/lib/x86_64-linux-gnu` also worked but is unnecessarily cluster-specific, so canonical code uses the minimal removal. |
| `lm-mpi-job-92775f0b-26a4-462a-b7c2-c634f8306b19` | Clean post-fix rebuild sanitized `LD_LIBRARY_PATH` correctly but its second `uv` transaction made no filesystem progress and was explicitly stopped after read-only proof; it never published a runtime. |
| `lm-mpi-job-6d3a60e0-657f-481f-8ca1-b0a7a740e914` | Three read-only NFS snapshots over 90 seconds showed the new build fixed at 6,574,684,777 bytes / 21,907 files, while the preserved complete environment was 10,224,077,977 bytes / 65,686 files; about 54 GB remained free. This distinguished an `uv` hang from slow copying or ENOSPC. |
| `lm-mpi-job-04c1ec31-22e7-471e-9bb0-ca92b01846f1` | First promotion gate stopped before mutation because plain `pip check` sees SGLang's deliberate CUDA-13 metadata edge versus the installed audited CUDA-12 dependency. No other inconsistency was reported. |
| `lm-mpi-job-23c7357b-f551-4d32-bdda-271b42e85593` | Strict promotion accepted only that one known metadata override, checked exact CUDA-12/Hopper package versions and native imports, initialized one H100, wrote ready/freeze artifacts, atomically promoted the preserved complete environment, then passed the canonical bootstrap reuse gate. Scheduler `Completed`. |
| `lm-mpi-job-efb3aefc-03e7-4d8a-b484-c24aeab08001` | Read-only 1×H100 toolchain probe: Ubuntu 22.04 job image exposes conda at `/home/user/conda/bin/conda` and system `nvcc 12.6.85`. This independently reproduced the compiler version implicated by the 8×H100 MHC failure and established the bootstrap mechanism for the isolated 12.9 prefix. |
| `lm-mpi-job-9025f0bf-80f1-4cb6-a32d-2b2fbbeecd1c` | Exact `2456f094ec57bc33845d04f87ebdb83ef30964d5` production retry sanitized the driver path, reused the validated cu126 runtime/model cache, initialized NCCL ranks 0–7, and loaded all 48 DeepSeek shards. It then failed before server readiness when `nvcc 12.6.85` rejected DeepGEMM's 128-bit PTX constraint during the 21-bucket MHC prenorm prewarm on every TP rank. No EvolutionLoop state or LLM usage exists. |

Bootstrap smoke `lm-mpi-job-3c02e882-bc28-434f-9ed2-1a3841b66c2a` failed
before workers became READY and emitted no user-code output. Its only new
pre-start dependency was `checkpoint_dir` pointing at a run path that the
bootstrap had not created yet; that scheduler field was removed. Resume does
not depend on it because the explicit regional `RUN_DIR` is persistent.

Acceptance before calling it debugged: inventory proves exactly eight H100;
SGLang model revision and concurrent smoke response are persisted; scheduler is
Running; `population_state.json` advances and `llm_usage.jsonl` contains real
DeepSeek calls. Acceptance before calling it complete: scheduler+run manifests
both completed, finalized generation=300, no inflight transaction, token report
parses cleanly, scores/organism survival are audited and this section is
replaced with final measurements.

### Исторические AWTF post-mortems

| Run / наблюдение | Подтверждённый вывод | Что после этого сделали |
|---|---|---|
| 426 organisms | Малые модели путались в нескольких форматах artifacts; retries умножали цену doomed lineages; crossover сходился к primary parent; один island regime имел около 0.9% success. | P1: один canonical implementation example. P2: historical island prompt упростили. P3: урезали retries. P4: mutation 0.2→0.5, within 0.7→0.4, inter 0.2→0.1. P5: near-full PATCH→FULL. P6: critique-aware redesign retry. |
| 1516 organisms | 1329 `failed_creation`; около 1277/1329 (≈96%) имели один root cause: bullet-wrapped Python fence в `OPTIONAL_CODE_SKETCH`. Один bad gen-0 organism отравил 784 descendants при повторном чтении malformed genome. Ещё примерно 27 gen-0 failures были paragraph prose в `MACRO_STRATEGY`. | P7: parser понимает bare и bullet-wrapped fences. P8: AWTF schema сокращена 8→5 sections. P9: code/pseudocode распределены по owning sections, `MACRO_STRATEGY` закреплён как bullets. P10: prompts требуют сохранять именованные накопленные механизмы. |
| 491 organisms | Mutation children failed в 1.79 раза чаще crossover. 102 organisms (около 20% population) упали из-за fenced Python в `STATE_REPRESENTATION`. Обнаружен «stage collapse»: genome обещал 3–5 stages, code реализовал 1–2. | P11: production random gene pruning/merging выключен, full-parent genomes передаются LLM. P12: fenced-code tolerance расширена на substantive sections. P13: silent mechanic-to-code audit добавлен в AWTF implementation prompt. |

Часть P-fixes позже была superseded:

- P2 island-specific seed prompting больше не live, потому что seed — file-copy;
- P3 budgets стали ещё жёстче (`1/2/0` в AWTF/circle);
- compatibility половина P6/P10 больше не исполняется после удаления gate;
- P11 остаётся экспериментальной гипотезой: post-fix outcome metrics не
  committed.

### Performance/operations observations

- Первый упомянутый в коде 75-generation AWTF Shinka-run показал `0/78 correct`
  и `score=0.0` даже для заведомо рабочего seed. Это **невалидный benchmark
  result**: Shinka не читает Python return evaluator adapter'а, а ожидает
  `correct.json` и `metrics.json`; старый adapter их не записывал, поэтому
  upstream подставлял defaults. В соседних config comments тот же run также
  связывается со слишком общим task prompt. Оба недостатка затем исправили
  (task-specific system prompt и `save_json_results`), но post-fix Shinka-vs-
  EvolutionLoop result не committed.
- В одном предыдущем pipeline run 98.6% calls пришлись на Gemma и 1.4% на Qwen,
  несмотря на ожидаемый `qwen_creative_gemma_check`. Код теперь логирует
  загруженные pipeline IDs на startup, но committed подтверждения, что
  распределение исправлено, нет.
- Для Qwen 3.5 122B средний design call был измерен как 916 seconds и занимал
  51% wall time. После этого для `design` уменьшили thinking/output budget и
  увеличили broker concurrency.
- Эти числа — historical operational telemetry, не objective-quality results.

## Известные расхождения, риски и технический долг

### Критичные

1. **Historical committed credential.** В author refs/generated notebooks был
   plain-text Comet API key. Finalization live удаляет notebooks и использует
   только env interpolation с `COMET_ENABLED=false` по умолчанию. Сам секрет
   всё равно надо revoke вне Git; его значение нигде не повторять.
2. **Remote results не защищены и baseline отсутствует.** Четыре population
   roots и лучшие organisms существуют только на Cloud.ru NFS; в Git нет
   manifests/archive. Shinka jobs не запускались, поэтому head-to-head
   comparison отсутствует.
3. **Historical documentation drift.** README finalization live переписан под
   текущие contracts, но `FRAMEWORK.md` всё ещё местами описывает compatibility
   validator, LLM seed operators и biased island
   prompts; live master этого не делает.

### Поведенческий/config drift

- README говорит `num_inspirations` default N=2; active AWTF/circle configs —
  0. В code fallback 2 срабатывает только при отсутствии key.
- `FRAMEWORK.md` говорит Comet disabled by default; active top-level AWTF/circle
  configs включают его.
- `FRAMEWORK.md` показывает старые retry budgets и compatibility state machine.
- `FRAMEWORK.md` ссылается на отсутствующий
  `src/organisms/compatibility.py` и отсутствующие AWTF/circle compatibility
  prompts. Optimization dead prompts всё ещё лежат на диске.
- `conf/experiments/awtf2025_heuristic/prompts/shared/project_context.txt`
  всё ещё говорит о novelty и compatibility «where configured».
- Optimization prompts всё ещё используют legacy inherited/removed gene-pool
  framing; production operators передают туда `(none)`.
- Rationalization `stage_options` mismatch описан выше.
- AWTF pipeline bandit имеет один arm; circle island bandits — один arm.
- Circle/CO-Bench задают inter-island weight при одном island.
- Top-level configs предполагают 8 GPU для LLM routes. Это явный deployment
  profile, не переносимый default для обычной машины.
- `conf/experiments/awtf2025_heuristic/group_commands_and_wall_planning 2.yaml`
  — tracked duplicate с пробелом в имени, не подключённый Hydra defaults.
- В нескольких guides остались абсолютные ссылки на старый user path
  `/Users/artemon/...`, что делает их непереносимыми.
- Текущая локальная branch naming/tracking вводит в заблуждение; перед push
  обязательно проверять точный target branch.

### Не считать результатом

- Наличие теста подтверждает контракт, но не fitness.
- Prompt-фраза «rationalization лучше» — hypothesis, пока нет A/B run.
- Bandit с одним arm не адаптируется.
- Upstream ref с более поздней датой не обязательно является desired merge:
  `adding-co-bench` — linear extension, `refactor` — старый divergent prototype.

## Проверка и рабочие команды

Рекомендуемая полная проверка:

```bash
pytest -q
```

Фокусные suites:

```bash
pytest -q tests/test_hydra_compose.py
pytest -q tests/test_import_optimizer.py
pytest -q tests/test_prompt_bundle.py
pytest -q tests/test_organism_contract.py
pytest -q tests/test_run_evolution.py
pytest -q tests/test_evolution_resume.py
pytest -q tests/test_bandit.py
pytest -q tests/test_two_step_design.py
```

Finalization live использует изолированный `.venv`. Полный результат на
2026-08-04:

```bash
.venv/bin/pytest -q
# 475 passed, 8 skipped in 162.46s

python -m compileall -q src experiments api_platforms scripts/cluster
git diff --check
bash -n scripts/*.sh scripts/cluster/*.sh
```

Skipped tests требуют optional external dependencies/hardware. До green run
были исправлены только environment/entrypoint defects: Bash 3.2 array/map
portability и Hydra 1.3 no-config crash на Python 3.14. Также проверены
отсутствие trailing whitespace и известного plaintext credential во всём live
tree.

Минимальные read-only проверки Git:

```bash
git status --short --branch
git ls-remote origin
git branch -a -vv
git log --graph --decorate --oneline --all
git rev-list --left-right --count origin/master...origin/<branch>
git diff --stat origin/master...origin/<branch>
```

## Checklist для будущего обновления

- [ ] Обновлены audit date, HEAD и server refs.
- [ ] Новые/merged/deleted upstream branches отражены в матрице.
- [ ] Live lifecycle перепроверен по code и composed config, не только README.
- [ ] Новые artifacts/state keys внесены в filesystem contract.
- [ ] Для нового family записаны API, scoring, dataset, resources и preset.
- [ ] Для нового run записаны commit, config overrides, seed, budgets, models,
      elapsed time, token usage, survival/failures, baseline и best score.
- [ ] Historical telemetry отделена от текущих результатов.
- [ ] Secrets не скопированы.
- [ ] Drift/risk items закрыты либо обновлены.
- [ ] Верификация выполнена и записана честно, включая environment blockers.

## Change log этой базы

- **2026-08-04, `finalize/evolutionloop-deepseek-v4` (base `641777d`):**
  зафиксирован canonical EvolutionLoop protocol; добавлены backbone composition,
  generic OpenAI-compatible inference, exact DeepSeek/SGLang 8×H100 job,
  per-call token telemetry, evaluator finite-score contracts, machine task
  catalog/readiness audit и cluster runbook. Удалены stale generated job
  notebooks с plaintext credential. Полный regression suite: 475 passed,
  8 skipped. Fitness run ещё не считается результатом до terminal monitor event.
- **2026-08-04, `641777d` remote handoff audit:** проверены public heads/PR
  refs, local/server reflogs, unreachable objects, server working tree и NFS
  artifacts. Установлено, что финальный engine одинаков в master и
  adding-co-bench, а последний реально использованный snapshot — `641777d`.
  Добавлены результаты двух interrupted и двух completed 700-generation runs;
  подтверждено, что Shinka baseline cells не запускались.
- **2026-08-04, `4ca9bbb`:** уточнено, что ShinkaEvolve — отдельная baseline
  search line против собственного `EvolutionLoop`, а не автоматический
  comparator; записаны различия shipped budgets и невалидность исторического
  `0/78 correct` AWTF run из-за broken result adapter.
- **2026-08-03, `4ca9bbb`:** первая полная cross-branch база. Проаудированы все
  семь server refs, live architecture/configs/tests, две unmerged линии,
  исторические AWTF post-mortems, отсутствие committed run results и
  documentation/security drift.
