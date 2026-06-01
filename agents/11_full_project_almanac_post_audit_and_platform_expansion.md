# Полноценный альманах проекта: пост-аудит и экспансия платформы

## Назначение

Этот документ продолжает цепочку инструкций и ре-аудитов (`5_…` → `10_…`) и
фиксирует новый, актуальный снимок репозитория. Он не ставит новых задач.
Его роль — собрать в одном месте все изменения, которые накопились **после**
инструкции `agents/10_gpt_followup_instruction_for_finishing_remaining_post_audit_gaps.md`,
и зафиксировать, во что фактически превратился проект.

Предыдущий `10.md` перечислял шесть workstream-ов: P0-исправление Neyman /
Horvitz-Thompson, убирание legacy-фолбэков в конфиге, строгий canonical-read,
схлопывание второго «canonical-looking» генератора, полная изоляция legacy
кандидат-генератора, финальная чистка имён и мёртвых алиасов. Нынешний
снимок показывает: все шесть workstream-ов закрыты, а поверх них проект
вырос в многозадачную (task-family) архитектуру с унифицированным
LLM-API-слоем и полной ручной (notebook) петлёй.

Всё, что ниже, выведено из состояния файлов на `2026-04-19` на ветке
`atcoder-heuristic-experiment`.

---

## 1. Верхнеуровневая структура репозитория

```
new-optimizer-finding/
├── AGENTS.md                 # глобальный гайд репозитория
├── README.md
├── pyproject.toml
├── agents/                   # серия инструкций и аудитов (5–11)
├── api_platforms/            # НОВОЕ: унифицированный LLM-слой
│   ├── _core/                # broker, registry, providers, ipc, …
│   └── <route_id>/           # 19 маршрутов (cloud + ollama + local + mock)
├── conf/                     # Hydra-композиция
│   ├── config_awtf2025_heuristic.yaml
│   ├── config_circle_packing_shinka.yaml
│   ├── config_optimization_survey.yaml
│   ├── evolver/              # per-task-family настройки эволюции
│   ├── experiments/          # per-task-family промпты и эксп. конфиги
│   └── api_platforms/        # per-route YAML-файлы
├── experiments/              # три task-family пакета + _runtime
│   ├── awtf2025_heuristic/
│   ├── circle_packing_shinka/
│   └── optimization_survey/
├── src/
│   ├── main.py               # диспетчер режимов evolve/validate
│   ├── runtime_config.py     # проверка наличия полного preset
│   ├── evolve/               # canonical organism-first loop
│   ├── organisms/            # mutation/crossbreeding/novelty/organism
│   └── validate/             # валидатор одного организма
├── scripts/                  # bash-раннеры и диагностика
├── notebooks/                # manual_simple_scoring.ipynb
├── tests/                    # контрактные + регрессионные тесты
├── populations/              # рабочие population-корни
└── outputs/                  # Hydra-выдачи
```

Главные **новые** блоки по сравнению с состоянием `10.md`:

1. полноценный каталог `api_platforms/` (прежде LLM-вызовы были
   внутри `src/evolve/` через OpenAI-клиент и ручные ollama-хелперы);
2. три полноценных task-family: `awtf2025_heuristic`,
   `circle_packing_shinka`, `optimization_survey` (раньше канонически жил
   один survey);
3. `scripts/` с `seed_population.sh`, `run_evolution.sh`, `kill_ollama.sh`,
   `lib_runtime.sh`;
4. `notebooks/manual_simple_scoring.ipynb` и поддерживающий его
   `src/evolve/manual_pipeline.py`.

---

## 2. Статус всех шести workstream-ов из `10.md`

### Workstream 1 — Conditional Poisson / Horvitz-Thompson. **ЗАКРЫТ.**

- `src/evolve/allocation.py:101` — `compute_nonempty_probability()`.
- `src/evolve/allocation.py:110` — `compute_conditional_inclusion_probabilities()`.
- `src/evolve/allocation.py:125` — `sample_experiments_poisson()` реализует
  отказную выборку до непустого подмножества, без argmax-fallback.
- Snapshot содержит все поля, указанные в `10.md`: `method`, `enabled`,
  `sample_size`, `sampling_design`, `weights`, `base_inclusion_prob`,
  `prob_nonempty`, `inclusion_prob`, `stats`, `selected_experiments`.
- `src/evolve/scoring.py:8` — `mean_score()` использует **эффективные**
  условные вероятности и жёстко падает при `q_i <= 0.0` у выбранного эксп.
- Покрытие: `tests/test_neyman_allocation.py`, `tests/test_scoring.py`.

### Workstream 2 — строгий canonical config. **ЗАКРЫТ.**

- `src/evolve/evolution_loop.py:127` — `_require_cfg_value(path)` поднимает
  `ValueError`, если канонический ключ отсутствует.
- Все legacy-пути (`evolution.mutation_rate`, `evolution.crossover_p`,
  `evolver.timeout_sec_per_eval`, `legacy_flat_population_island`) удалены из
  `src/` — grep не находит ничего.
- Конфиг `conf/evolver/*.yaml` требует новый layout (`creation`, `islands`,
  `reproduction`, `operators`, `phases`, `llm`).

### Workstream 3 — строгий canonical resume / read. **ЗАКРЫТ.**

- В `src/` нет упоминаний `idea_dna.txt`, `evolution_log.json`,
  `legacy_flat_population`, `load_population`.
- `read_population_state()` (`src/evolve/storage.py:262`) валидирует
  manifest по каждому полю (организмы, island_id, generation_created,
  current/pipeline state), поднимает `ValueError` на дубликаты и
  некорректные типы.
- Каноническая directory-layout для организма зафиксирована:
  `results/{simple,hard}`, `logs/`, `implementation.py`, `genetic_code.md`,
  `lineage.json`, `organism.json`, `summary.json`.

### Workstream 4 — единственная каноническая мутация/кросс. **ЗАКРЫТ.**

- `src/evolve/operators.py` — **25 строк**, в них только `SeedOperator`.
  Ни `MutationOperator`, ни `CrossoverOperator` в канонике больше нет.
- Мутация — `src/organisms/mutation.py`, 181 строка.
- Скрещивание — `src/organisms/crossbreeding.py`, 201 строка.
- Генератор теперь называется `CandidateGenerator` (историческое имя) и
  отвечает исключительно за seed-создание — см. `src/evolve/generator.py:85`.

### Workstream 5 — изоляция legacy-кандидат-генератора. **ЗАКРЫТ.**

- Файла `src/evolve/legacy_generator.py` в репозитории больше нет.
- Вместо него введён `src/evolve/llm_generator_base.py` с классом
  `BaseLlmGenerator`, который инкапсулирует только генерические функции:
  routing по `route_weights`, batch-assignment, `_extract_python()`.
- Канонический `CandidateGenerator` наследует `BaseLlmGenerator`; никакой
  второй ветки (legacy) с доступом к `conf/prompts/*` не существует —
  канонические промпты теперь живут в `conf/experiments/<family>/prompts/`.

### Workstream 6 — чистка имён и мёртвых хелперов. **ЗАКРЫТ.**

- `src/evolve/selection.py` содержит **только** канонические функции:
  `uniform_select_organisms`, `softmax_select_organisms[,distinct]`,
  `weighted_rule_select_organisms[,distinct]`, `select_top_k_per_island`,
  `select_top_h_per_island`. Ни `tournament_select`, ни `elite_select`, ни
  `select_parents_for_reproduction` больше нет.
- `OrganismMeta` (`src/evolve/types.py:43`) хранит `simple_score` /
  `hard_score`; устаревший алиас `score` удалён (сохранён только
  backward-compat `generation` как property).
- Lineage backfill: `src/organisms/organism.py` использует
  `update_latest_lineage_entry(...)` без параметра `aggregate_score`.

Итог: все шесть целей `10.md` технически закрыты.

---

## 3. Новая архитектура LLM: слой `api_platforms/`

Это самое заметное **новое** направление, целиком созданное после `10.md`.

### 3.1 Структура

```
api_platforms/
├── __init__.py              # экспортирует ApiPlatformRegistry, Broker,
│                            # Client, ApiRouteConfig, LlmRequest/Response
├── _core/
│   ├── broker.py            # singleton брокер, Unix-socket IPC
│   ├── broker_runner.py
│   ├── config.py            # derive_ollama_instance_configs
│   ├── discovery.py         # load_route_configs
│   ├── ipc.py               # read_json_line / write_json_line
│   ├── local_worker.py      # worker-процесс для локальных (GPU) моделей
│   ├── providers.py         # generate_direct: ollama / openai / anthropic
│   ├── registry.py          # ApiPlatformRegistry — публичный фасад
│   └── types.py             # dataclasses LlmRequest/LlmResponse/…
└── <route>/                 # один каталог на каждый маршрут
```

### 3.2 Каталог маршрутов (19 штук)

Cloud-провайдеры:
- `claude_opus_46`, `claude_sonnet_46`, `claude_haiku_45`
- `gpt_5_4`, `gpt_5_4_mini`, `gpt_5_4_nano`

Локальные HF-модели (через own worker-процесс, `gpu_ranks` из конфига):
- `qwen35_27b`, `qwen35_27b_claude46_opus_distilled`, `qwen35_35b_a3b`
- `gemma_4_26b_a4b_it`, `gemma_4_31b_it`

Ollama-маршруты (через локальный `ollama serve`, по GPU-разбиению):
- `ollama_nemotron_cascade_2_30b`, `ollama_qwen35_122b`,
  `ollama_qwen35_27b`, `ollama_qwen35_35b`,
  `ollama_gemma4_26b`, `ollama_gemma4_31b`

Заглушки:
- `mock`, `mock_local`

Каждому маршруту отвечают:
- `conf/api_platforms/<route>.yaml` — конфиг маршрута (base_url,
  gpu_ranks, max_concurrency, stage_options),
- `api_platforms/<route>/` — пакет-дескриптор.

### 3.3 Как evolve-loop им пользуется

1. `ApiPlatformRegistry(cfg)` строится один раз в `src/evolve/run.py:24` или
   `src/evolve/seed_run.py:24`.
2. `registry.start()` поднимает broker + local-worker-процессы с
   `mp.get_context("spawn")`.
3. `CandidateGenerator.sample_route_id(organism_id=...)` через
   `BaseLlmGenerator` либо возвращает route из
   batch-assignment (`set_batch_route_assignments`), либо хэширует
   `(seed, organism_id)` и сэмплирует по `route_weights`.
4. `generate_direct()` в `api_platforms/_core/providers.py` диспатчит на
   нужного провайдера и возвращает `LlmResponse`.

Ключевое «почему» batch-assignment: две seed-заявки в маленьком батче на
двух равно-весомых маршрутах сталкивались в 50% случаев (простая per-org
хэш-статистика), оставляя один ollama-инстанс простаивать. Теперь
evolution loop заранее кладёт сбалансированное отображение
`organism_id → route_id`, которое `sample_route_id` чтит.

### 3.4 Диагностика

`conf/config_awtf2025_heuristic.yaml:39` показывает рабочий шаблон:
ollama-маршруты получают `base_url`, `gpu_ranks`, `max_concurrency`,
`stage_options.{design,implementation,repair,novelty_check}`.

`/Users/artemon/.claude/.../memory/project_ollama_gpu.md` фиксирует
известный паттерн: `ollama serve` может рапортовать `total_vram=0B` на
H100-ноде и незаметно гнать qwen35 на CPU. Это не баг в коде, но важно
при отладке «зависаний».

---

## 4. Canonical организм-first цикл (текущее состояние)

### 4.1 Entrypoints

- `src/main.py` — диспетчер режимов: `mode=evolve` → `src.evolve.run`,
  иначе → `src.validate.runner.ExperimentRunner`.
- `src/evolve/run.py` — `run_evolution(cfg)`: запускает
  `ApiPlatformRegistry` и вызывает `EvolutionLoop.run()` через
  `asyncio.run`.
- `src/evolve/seed_run.py` — `run_seed_population(cfg)`: выполняет только
  `EvolutionLoop.seed_population()`. Это отдельный вход именно для
  «засеять gen_0 и остановиться», чтобы ручной notebook / debug-сессия
  могли пере-скор-ить сгенерированное.
- Все три входа требуют полноценного preset (`runtime_config.ensure_root_runtime_config`).

### 4.2 `EvolutionLoop` (1814 строк, `src/evolve/evolution_loop.py`)

Что **зафиксировано** как каноническая форма:
- Чтение конфига — только через `_require_cfg_value`; fallback нет.
- Islands грузятся из `evolver.islands.dir` через `load_islands()`
  (`src/evolve/islands.py`), требуются `.txt`-файлы.
- `reproduction.operator_selection_strategy` ∈ {`random`, …};
  `operator_weights` делятся между `within_island_crossover`,
  `inter_island_crossover`, `mutation`.
- `reproduction.species_sampling`:
  - `strategy: weighted_rule` — по формуле `σ(λ·(fitness − median)) ·
    1/(1+offspring_count)` (см. `_weighted_rule_weights` в selection.py),
  - либо `softmax` (с `mutation_softmax_temperature` и парой cross-over
    температур).
- Есть `max_parallel_organisms` для параллельного создания.
- `creation` блок — `max_attempts_to_create_organism`,
  `max_attempts_to_repair_organism_after_error`,
  `max_attempts_to_regenerate_organism_after_novelty_rejection`.

### 4.3 Pipeline organism creation

Организм проходит через фазы, сохраняемые как
`PlannedOrganismCreation.pipeline_state` (`src/evolve/types.py:253`):

1. `planned_creation` — запись плана в `organism.json` до LLM-вызовов;
2. LLM design → genetic_code;
3. `novelty_check` (см. 4.4);
4. LLM implementation → `implementation.py`;
5. Validation of code (syntax via `ast.parse`);
6. Запись `lineage.json` с materal history, отцом (если crossover),
   пометкой `cross_island`;
7. Готовый организм попадает в активную популяцию.

Если шаг падает — `max_attempts_to_repair_organism_after_error` триггерит
repair-prompt; при исчерпании попыток организм помечается ошибочным и
исключается.

### 4.4 Novelty-валидация

Новый модуль `src/organisms/novelty.py`:
- `NoveltyCheckContext` — дескриптор операторной специфики (mutation vs
  crossover);
- `parse_novelty_judgment()` разбирает `NOVELTY_ACCEPTED/REJECTED`
  ответ LLM;
- при отказе — цикл повторной генерации до
  `max_attempts_to_regenerate_organism_after_novelty_rejection`, с
  накоплением формулировок отказов (`format_novelty_rejection_feedback`).

Novelty-промпты живут per-family:
`conf/experiments/<family>/prompts/novelty/{mutation,crossover}/{system,user}.txt`.

### 4.5 Evaluation / Orchestrator

- `src/evolve/orchestrator.py:40` — `EvolverOrchestrator` принимает
  `OrganismEvaluationRequest`, строит `EvalTask`-и, прогоняет их через
  `GpuJobPool` (`src/evolve/gpu_pool.py`).
- Entrypoint внутри subprocess — `src.validate.run_one` по умолчанию
  (`DEFAULT_EVAL_ENTRYPOINT_MODULE`). Есть приватный override
  `evolver._eval_entrypoint_module` — только для тестов.
- `repair_callback` прокидывается обратно в evolution loop, чтобы на
  ошибке evaluator-а можно было репарировать `implementation.py`.
- `GpuJobPool` умеет `_kill_process_tree` через `psutil` и честно
  обрабатывает timeout.

### 4.6 Subset evaluation (Neyman + conditional Poisson)

Эта часть описана в секции 2, но по факту это центральный канонический
контур:

```
build_allocation_snapshot → sample_experiments_poisson →
    EvolverOrchestrator.enqueue(...) → mean_score
```

Поле `aggregate_score` в `OrganismEvaluationSummary` (types.py:204) —
это Horvitz-Thompson-подобная оценка среднего по **всем** экспериментам
(не только выбранным) через эффективные `q_i_cond`.

### 4.7 Resume / manifest

- `population_state.json` хранит `current_generation`, `active_organisms`,
  `best_organism_id`, `best_simple_score`, `timestamp`,
  `relationship_history`, а также **новые** поля `inflight_seed` и
  `inflight_generation`. Они позволяют корректно поднять процесс после
  прерывания на половине seed-пачки или половине поколения.
- `_build_relationship_history()` на каждом write сканирует
  `gen_*/island_*/org_*/organism.json` и строит компактную историю
  родства (organism_id, mother, father, island, operator). Это нужно
  визуализации и аудиту родословных.

### 4.8 Визуализация

`src/evolve/visualization.py` (716 строк) строит per-run overview:
score-кривые, best-lineage overlay, распределение операторов по
поколениям, sparkline-ы. Вызывается из `EvolutionLoop` на каждом
шаге; артефакты складываются в `population_root`.

---

## 5. Task-family пакеты `experiments/`

Раньше канонически был один `optimization_survey`. Сейчас — три
самостоятельных task-family, каждая со своим `_runtime`:

### 5.1 `experiments/optimization_survey/`

Исторический набор задач оптимизации: `xor_mlp`, `mnist_mlp`,
`cifar_convnet`, `poly_regression`, `sin_regression`, `two_spirals`,
`quadratic_bowl`, `rosenbrock_net`, `lora_sft`, `minigpt_wikitext2`,
`tiny_autoencoder`, `linear_denoiser`, `audio_transformer`,
`synthetic_logreg`, `matrix_factorization`, `ddpm_cifar10`,
`conv1d_classify`.

Именно здесь живёт контракт **оптимизатора**: `build_optimizer(model,
max_steps)` + `step(...)` + `zero_grad(set_to_none=True)`. Это
намеренно изолировано именно в этой task-family и не утекает в
`src/`.

### 5.2 `experiments/circle_packing_shinka/`

Новая task-family вокруг задачи упаковки кругов (родственная
Shinka-бенчмарку). Содержит `_runtime`, `unit_square_26`.
Конфиг: `conf/config_circle_packing_shinka.yaml`,
`conf/evolver/circle_packing_shinka.yaml`.

### 5.3 `experiments/awtf2025_heuristic/` (AtCoder Heuristic)

Самая свежая task-family (ветка называется
`atcoder-heuristic-experiment`). Решает задачу
`group_commands_and_wall_planning`, эвристика с двумя островами:
- `macro_partitioning.txt`
- `staged_routing_repair.txt`

Конфиг: `conf/config_awtf2025_heuristic.yaml` +
`conf/evolver/awtf2025_heuristic.yaml`. По умолчанию в нём
три локальных ollama-маршрута (nemotron-cascade-2-30b, qwen35-35b,
gemma4-31b) по одному GPU на каждый.

### 5.4 Общий контракт task-family

Каждая task-family держит:
- `_runtime/` — общий evaluator-каркас;
- один или несколько под-пакетов с `evaluate_organism(organism_dir, cfg)
  -> dict`;
- обязательное поле `score` в отчёте;
- свой `conf/experiments/<family>/` — Hydra-овые конфиги экспериментов
  и все промпты (`seed/`, `mutation/`, `crossover/`, `novelty/`,
  `implementation/`, `repair/`, `shared/project_context.txt`,
  `islands/*.txt`).

Важно: `src/` остаётся «task-blind» — он оперирует только папкой
организма, списком экспериментов и отчётами со `score`. Эта инвариант
зафиксирована в `AGENTS.md:14` и ни разу не нарушена в рабочих модулях.

---

## 6. Ручной (manual / notebook) режим

Две недавние вехи (`d0eaaf2 adding manual inferencing`,
`95ea040 Accept short preset names in manual pipeline`) дали полноценную
ручную петлю:

- `src/evolve/manual_pipeline.py` — `load_manual_pipeline_context()`
  резолвит preset по короткому имени (`awtf2025_heuristic` /
  `config_awtf2025_heuristic`), Hydra-комбинирует полный `cfg`, и
  возвращает `ManualPipelineContext` с резолвленным экспериментом и
  `PromptBundle`.
- `notebooks/manual_simple_scoring.ipynb` — тетрадь, в которой можно
  вручную:
  1. сгенерировать seed/mutation/crossover-промпт,
  2. отправить запрос через `ApiPlatformRegistry`,
  3. распарсить ответ, построить `genetic_code.md`/`implementation.py`,
  4. прогнать `evaluate_organism` без автоматической эволюции.

Это именно «ручной» sanity-путь: не замена evolution loop, а
воспроизводимая песочница для исследователя.

---

## 7. Конфиг-композиция

### 7.1 Три top-level preset-а

- `conf/config_optimization_survey.yaml`
- `conf/config_circle_packing_shinka.yaml`
- `conf/config_awtf2025_heuristic.yaml`

Каждый подключает:
- свой `evolver` (`conf/evolver/<family>.yaml`),
- нужный набор `api_platforms@api_platforms.<route_id>`,
- эксперименты (`experiments@experiments.<name>`),
- `_self_` в конце.

### 7.2 `runtime_config.ensure_root_runtime_config`

В `src/runtime_config.py:8` жёстко проверено, что в `cfg` есть
`{paths, experiments, resources, api_platforms, evolver}`. Это защищает
от случайного запуска без preset-а: сообщение прямо подсказывает
пользователю `--config-name <preset>`.

### 7.3 `conf/evolver/<family>.yaml`

Фиксированный канонический layout (пример —
`conf/evolver/awtf2025_heuristic.yaml`):

```
resume:              bool
max_generations:     int
max_retries_per_eval:int
creation: {...}
islands:  {dir, seed_organisms_per_island, max_organisms_per_island}
prompts:  {project_context, seed_*, mutation_*, crossover_*, …}
reproduction:
  offspring_per_generation
  operator_selection_strategy
  operator_weights: {within_island_crossover, inter_island_crossover, mutation}
  island_sampling: {…}
  species_sampling: {strategy, weighted_rule_lambda, *_softmax_temperature}
operators:
  mutation:  {gene_removal_probability}
  crossover: {primary_parent_gene_inheritance_probability}
phases:
  simple:       {eval_mode, timeout_sec_per_eval, experiments, allocation}
  great_filter: {enabled, interval_generations, eval_mode, timeout, top_h_per_island, experiments, allocation}
llm:
  selection_strategy
  seed
  route_weights: {...}
```

Никакого fallback на старые поля (`evolution.*`, `evaluation.*`,
`evolver.timeout_sec_per_eval`) больше нет.

---

## 8. Прогресс-логирование

Наблюдение из `user-memory` (`feedback_logging.md`) закреплено в коде:

- `src/evolve/evolution_loop.py:60` — `_announce(msg)` дублирует
  `LOGGER.info` через `print(..., file=sys.stderr, flush=True)`.
- Аналогичная функция — `src/evolve/generator.py:15`.
- `src/evolve/seed_run.py:37` — `_ensure_console_logging()` идемпотентно
  добавляет stderr-handler к root-логгеру, обходя Hydra, которая иначе
  отправила бы логи только в `outputs/.../*.log`.

Этот паттерн живёт во всех точках, где пайплайн подолгу ждёт LLM
(`design`, `implementation`, `novelty_check`, `repair`), и гарантирует,
что пользователь видит живую строку прогресса в терминале.

---

## 9. Scripts & оперирование

`scripts/` (добавлены после `10.md`):

- `run_evolution.sh` — общий раннер, завязанный на preset и env-переменные.
- `seed_population.sh` — запуск через `src.evolve.seed_run`.
- `kill_ollama.sh` — снос локальных ollama-серверов по портам 12437–12439.
- `lib_runtime.sh` — общие bash-функции (пути, логирование).

Важные env-переменные, которые уважают конфиги:
`AIFS_DATA_ROOT`, `AIFS_STATS_ROOT`, `AIFS_RUNS_ROOT`, `POP_ROOT`,
`API_PLATFORM_RUNTIME_ROOT`, `OLLAMA_MODELS`,
`OLLAMA_<ROUTE>_BASE_URL`.

---

## 10. Тестовая поверхность

Новое / существенно расширенное покрытие:

- `tests/test_neyman_allocation.py` — покрывает всю Neyman-цепочку
  (conditional Poisson, `prob_nonempty`, эффективные `q_i`).
- `tests/test_scoring.py` — Horvitz-Thompson по эффективным π_i.
- `tests/test_evolution_loop_semantics.py` — строгие ошибки при
  отсутствии канонических ключей.
- `tests/test_evolution_resume.py` — resume по manifest-у, отказ при
  отсутствующих организмах.
- `tests/test_organism_contract.py` — обязательные секции
  `CORE_GENES`/`INTERACTION_NOTES`/`COMPUTE_NOTES`, отказ на
  «тонкой» генетике.
- `tests/test_evolve_integration_fake.py` — интеграция через fake
  evaluator (override `_eval_entrypoint_module`).
- `tests/test_runtime_surface_cleanup.py` — фиксирует отсутствие
  `score`-алиасов, legacy-полей, `tournament_select`/`elite_select`.
- `tests/test_islands.py` — строгий `load_islands()`.
- `tests/test_api_platforms.py` — регистрация маршрутов, routing
  по `route_weights`.
- `tests/test_manual_pipeline.py` — purelly `load_manual_pipeline_context`
  (short-name resolution, поиск `conf/config_<name>.yaml`).
- `tests/test_awtf2025_heuristic*.py`, `tests/test_circle_packing_shinka.py`,
  `tests/test_circle_packing_prompts.py` — per-family проверки промптов и
  раннеров.
- `tests/test_hydra_compose.py` — вся Hydra-композиция, включая проверки
  ollama-маршрутов и разграничение GPU.
- `tests/test_result_schema.py` — schema отчётов evaluator-ов.

Итого в `tests/` сейчас 33 тест-файла; все они запускаются под
`pytest -q`.

---

## 11. Минимальный набор инвариантов, который обязан выдерживать репо

Этот список собран из `AGENTS.md`, `10.md` и фактического поведения
кода. Любой будущий агент должен использовать его как checklist:

1. **Single-device**. Никакого DDP / model-parallel / скрытых
   multi-GPU предположений.
2. **`src/` task-blind**. `src/` оперирует только папкой организма,
   списком экспериментов и отчётами со `score`. Контракт
   `build_optimizer(...)` живёт **только** в
   `experiments/optimization_survey/`.
3. **Каноническая файловая схема организма**: `implementation.py`,
   `genetic_code.md`, `lineage.json`, `organism.json`, `summary.json`,
   `llm_request.json`, `llm_response.json`; `results/{simple,hard}`,
   `logs/`.
4. **`population_state.json`** — единственный источник состояния
   популяции при resume.
5. **Canonical evolve = organism-first + island-aware**. `run_evolution`
   → `EvolutionLoop`, никаких альтернативных путей.
6. **Прёмпты и острова** лежат в `conf/experiments/<family>/prompts/`,
   не в глобальном `conf/prompts/`.
7. **LLM routing** идёт **только** через `api_platforms.ApiPlatformRegistry`.
   Прямые `requests.post("http://…/ollama")` из `src/` запрещены.
8. **Subset evaluation** — исключительно
   `conditional_poisson_nonempty` + Horvitz-Thompson по эффективным
   `q_i_cond`. Никакого «argmax on empty» fallback.
9. **Contract тесты** живут рядом с canonical кодом и фиксируют
   границу canonical / legacy. Ни один canonical модуль не должен
   читать artefact-и старых имён.
10. **Manual pipeline** не должен дублировать evolution loop. Он
    использует те же `PromptBundle`, `ApiPlatformRegistry`,
    `evaluate_organism`.

---

## 12. Зоны повышенного риска (не новые задачи, а зоны внимания)

Эти пункты не являются TODO, но любой рефакторинг в этих местах
заслуживает особой осторожности:

1. **`api_platforms/_core/broker.py`** — multiprocessing + socketserver
   + threading. Любая пересборка IPC требует ручной проверки
   graceful-shutdown и `_kill_process_tree` behaviour.
2. **`EvolutionLoop._announce` / stderr-flush** — если кто-то будет
   рефакторить логирование, должна остаться stderr-линия прогресса:
   Hydra подавляет `LOGGER.info`, и пользователь перестанет видеть
   происходящее (см. memory `feedback_logging.md`).
3. **Ollama GPU fallback** — `ollama serve` может показать
   `total_vram=0B` и тихо крутить модель на CPU. Любой «зависший»
   evolve-раннер в первую очередь проверяется здесь (см. memory
   `project_ollama_gpu.md`).
4. **Batch route assignment** — маленькие сид-батчи без balanced
   `_batch_route_assignments` ломают распределение по ollama-серверам.
   Рефактор `BaseLlmGenerator.sample_route_id` обязан сохранить
   двухступенчатую логику (batch-map → per-organism-hash).
5. **PlannedOrganismCreation pipeline_state** — любое изменение
   этого enum ломает resume старых `population_state.json`. Если
   добавляется новое состояние, нужна explicit миграция, а не
   silent fallback.

---

## 13. Где искать что (быстрая карта)

- **Evolution loop ядро** — `src/evolve/evolution_loop.py`
- **Neyman / HT** — `src/evolve/allocation.py`, `src/evolve/scoring.py`
- **Selection стратегии** — `src/evolve/selection.py`
- **Seed organism generator** — `src/evolve/generator.py`
- **Mutation / crossover / novelty** — `src/organisms/{mutation,crossbreeding,novelty,organism}.py`
- **Evaluator и GPU-pool** — `src/evolve/{orchestrator,gpu_pool}.py`
- **Сохранение/чтение артефактов** — `src/evolve/storage.py`
- **Islands** — `src/evolve/islands.py`
- **LLM routing** — `api_platforms/_core/{registry,broker,providers}.py`
- **Ручной режим** — `src/evolve/manual_pipeline.py`,
  `notebooks/manual_simple_scoring.ipynb`
- **Hydra top-level** — `conf/config_<family>.yaml`
- **Per-family evolver** — `conf/evolver/<family>.yaml`
- **Per-family промпты и острова** — `conf/experiments/<family>/prompts/…`
- **Инструкции / предыдущие аудиты** — `agents/1..10`
- **Репо-гайд** — `AGENTS.md` (глобально) + каждый подкаталог со своим
  `AGENTS.md`.

---

## 14. Итог

Репозиторий прошёл через последовательность аудитов (5 → 6 → 8 → 10)
и ревизию каждой зоны, которую `10.md` объявил незакрытой.

На текущий момент (`2026-04-19`) он удовлетворяет всем инвариантам:
одна каноническая evolve-петля, один canonical config schema, один
способ чтения артефактов, один путь мутации и один путь кроссовера,
изолированный LLM-слой, покрытые строгими тестами границы. На этот
canonical skeleton наращены:

- унифицированный слой `api_platforms/` с 19-ю маршрутами;
- три task-family (optimization_survey, circle_packing_shinka,
  awtf2025_heuristic) с собственными промптами и островами;
- manual/notebook-петля для ручной инференции и скоринга;
- scripts-обвязка для оперирования;
- полная визуализация прогресса и устойчивое stderr-логирование.

Этот альманах — снимок после того, как последние открытые workstream-ы
из `10.md` были закрыты и сверху добавлена платформенная и task-family
экспансия.
