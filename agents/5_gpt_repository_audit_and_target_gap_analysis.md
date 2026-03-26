# Аудит репозитория и gap-analysis относительно целевой системы эволюции оптимизаторов

## 1. Что это за документ

Это не code review в узком смысле и не просто карта файлов. Это большой аудит текущего состояния репозитория относительно той системы, которую ты описал как целевую:

- платформа для поиска новых методов оптимизации с помощью эволюции LLM-агентов;
- особь как Python-реализация оптимизатора плюс генетический код;
- явная родословная и история изменения идей;
- island model со школами/направлениями;
- раздельные simple phase и Great Filter;
- baseline-relative reward;
- fixed optimizer template;
- prompts и school descriptions как конфигурация, а не как захардкоженные runtime-assets.

Приоритет источников истины в этом документе такой:

1. Твое текущее ТЗ в сообщении.
2. Фактический код и конфиги репозитория.
3. Исторические заметки в `agents/*.md`.
4. Внешний research framing из ShinkaEvolve и AdaptEvolve.

Главный вывод заранее: в репозитории уже есть сильный каркас для baseline-relative reward, запуска экспериментов, fixed template rendering и простейшей эволюции. Но текущая система пока не соответствует твоей целевой модели как исследовательской платформе. Она ближе к гибриду из:

- benchmark/validation framework для оптимизаторов;
- single-generation LLM candidate generator;
- начатого multi-generation organism loop без island semantics и без согласованной data model.

То есть основная проблема сейчас не в том, что "ничего нет", а в том, что есть несколько частично пересекающихся архитектур, которые вместе создают ложное ощущение завершенности.

## 2. Краткий вердикт

### Что уже хорошо и реально полезно

- В проекте есть рабочий validation runtime, который умеет запускать много экспериментов через единый контракт.
- Реализован baseline-relative reward на основе train objective, а не старых статических `quality_ref`/`steps_ref`.
- Есть fixed optimizer template и structured parsing для organism-style generation.
- Есть минимальная поддержка mutation/crossbreeding на уровне `idea_dna`.
- Есть проходящий test suite: `pytest -q` сейчас зеленый, `49 passed`.

### Что архитектурно сломано или расходится с целевой системой

- Нет школ/островов как сущностей системы.
- Нет txt-конфигов с school descriptions и constant system prompt в `conf/`.
- В репозитории одновременно живут две разные эволюционные модели: `cand_*` и `org_*`.
- Родословная не дообогащается реальными score после evaluation.
- Mutation и crossover не дают LLM реально переписывать genetic code так, как требует твоя модель.
- Parent selection не соответствует заданной семантике softmax/mother-father/uniform mutation sampling.
- Great Filter реализован не так, как описан в ТЗ: он смешивает simple и hard tasks, не имеет отдельного `top-h`, и использует неправильную агрегатную семантику.
- Конфиг содержит мертвые или полумертвые knobs.
- README, runtime, prompts и tests не согласованы по optimizer contract.

### Главный практический вывод

Если продолжать наращивать функциональность поверх текущей архитектуры без рефакторинга, получится не research engine, а труднообъяснимый гибрид с высокой вероятностью semantically wrong результатов. Сначала нужно зафиксировать одну каноническую модель данных и один канонический execution path.

## 3. Каноническая модель системы, как она должна работать по ТЗ

Ниже я формализую твое описание в виде сущностей и инвариантов. Это важно, потому что дальше все расхождения будут мериться именно относительно этой модели.

### 3.1 Основные сущности

#### Особь

Одна особь должна содержать:

- код оптимизатора;
- genetic code;
- lineage/history;
- simple reward;
- при необходимости hard reward / результат Great Filter;
- принадлежность острову;
- происхождение от конкретной материнской линии.

#### Генетический код

`b)` из твоего описания не должен быть просто коротким списком traits. Это должен быть богатый текстовый набор генов, который:

- хранит, чем вдохновлен метод;
- хранит, какие идеи уже пробовали;
- позволяет удалить/изменить/добавить гены;
- достаточно информативен, чтобы по нему можно было восстановить реализацию в fixed template.

Важно: LLM должна работать не просто с "описанием идеи", а именно с генетическим кодом как с объектом эволюции.

#### Родословная

`c)` должна быть не просто журналом текстов, а последовательностью:

- ключевое изменение;
- к какому потомку это относилось;
- какой score затем получился.

То есть родословная должна быть замкнутой по смыслу: идея изменилась -> потомок был оценен -> score вписан назад в lineage.

#### Остров / школа

Остров - это отдельная ветвь поиска со своим направлением. Значит нужны:

- school prompt / school description;
- локальная популяция;
- локальный отбор;
- возможность редкого межостровного crossbreeding.

#### Simple phase

Simple phase должна:

- оценивать каждую новую особь;
- присваивать ей постоянный simple reward;
- использовать отдельную семантику allocation;
- после этого делать selection внутри каждого острова по `top-k`.

#### Great Filter

Great Filter должна:

- работать только как тяжелый селекционный барьер;
- использовать hard tasks;
- не затирать смысл simple reward;
- после фильтра оставлять `top-h` на каждом острове.

### 3.2 Канонический lifecycle одной особи

1. Особь рождается либо как seed из school prompt, либо как mutation, либо как crossover.
2. LLM получает:
   - constant system prompt,
   - genetic code,
   - lineage,
   - fixed template.
3. LLM обновляет genetic code и генерирует код оптимизатора в рамках fixed signature.
4. Новая особь проходит simple evaluation.
5. Ее simple reward сохраняется навсегда как часть состояния особи.
6. Если поколение попадает на Great Filter, выжившие проходят hard evaluation.
7. Great Filter влияет на право размножаться, но не переписывает смысл simple reward.
8. После оценки lineage обновляется записью вида "изменение -> результат".

### 3.3 Обязательные инварианты

- Initialization идет по школам, а не по одному общему seed prompt.
- Школы задаются файлами в `conf`.
- Constant system prompt живет в `conf`.
- Fixed template не дрейфует и не размножается по разным местам.
- Mother/father semantics не подменяются "dominant/non-dominant by score".
- Mutation sampling не зависит от tournament pressure, если по ТЗ оно должно быть uniform.
- Crossover sampling не зависит от tournament pressure, если по ТЗ оно должно быть softmax-by-reward.
- Lineage хранит завершенные записи с результатом, а не вечные `score=None`.
- Все публичные описания optimizer contract совпадают.

## 4. Что реально есть в репозитории сейчас

### 4.1 Три основные подсистемы

#### A. `valopt/` - validation/runtime layer

Это самая цельная часть системы. Она отвечает за:

- запуск экспериментов;
- baseline stats;
- `mode=run`, `mode=stats`, `mode=smoke`;
- импорт внешнего optimizer file;
- сериализацию канонического run result.

Ключевые файлы:

- `valopt/runner.py`
- `valopt/optimizer_api.py`
- `valopt/utils/import_utils.py`
- `valopt/utils/baselines.py`
- `src/validate/run_one.py`

#### B. `src/evolve/orchestrator.py` - single-generation candidate pipeline

Это pipeline вида:

- сгенерировать `cand_*`;
- выбрать подмножество экспериментов через allocation;
- прогнать evaluation;
- посчитать aggregate score;
- сохранить `summary.json`.

Ключевые файлы:

- `src/evolve/orchestrator.py`
- `src/evolve/generator.py`
- `src/evolve/scoring.py`
- `src/evolve/metrics_adapter.py`
- `src/evolve/storage.py`

#### C. `src/evolve/evolution_loop.py` + `src/organisms/*` - organism-style multi-generation loop

Это уже другая модель:

- `org_*` директории;
- `idea_dna.txt`;
- `evolution_log.json`;
- `organism.json`;
- mutation / crossbreeding operators.

Ключевые файлы:

- `src/evolve/evolution_loop.py`
- `src/organisms/organism.py`
- `src/organisms/mutation.py`
- `src/organisms/crossbreeding.py`
- `src/evolve/template_parser.py`

### 4.2 Главный structural smell

Сейчас проект одновременно поддерживает:

- candidate-centric evolution artifacts `cand_*`;
- organism-centric evolution artifacts `org_*`.

Это не просто разные названия папок. Это две разные модели системы.

Доказательства:

- `src/evolve/storage.py:63-99` определяет `cand_*` layout.
- `src/evolve/storage.py:128-159` определяет `org_*` layout.
- `src/evolve/run.py:29-45` по умолчанию запускает multi-generation `EvolutionLoop`, потому что `evolver.evolution` есть в дефолтном конфиге.
- `README.md:146-171` при этом документирует `cand_*` как основной output для evolve.

Практический эффект: у проекта уже нет одного канонического "evolve mode". Документация, storage, orchestration и generation semantics смотрят в разные стороны.

## 5. Что уже реализовано хорошо и что стоит сохранить

Важно отметить сильные стороны, чтобы потом не сломать их лишним рефакторингом.

### 5.1 Baseline-relative reward path

Судя по `valopt/utils/baselines.py`, `src/validate/run_one.py`, `src/evolve/metrics_adapter.py` и `src/evolve/scoring.py`, в проекте уже есть сильная база для нового reward path:

- baseline profile загружается как canonical `stats/<experiment>/baseline.json`;
- runtime умеет прокидывать `baseline_last_train_loss`;
- scoring использует `objective_last` и `first_step_at_or_below_baseline`;
- per-experiment score считается harmonic mean от quality/speed terms.

Это правильная стратегическая база. Ее не нужно выкидывать; ее нужно только встроить в правильную эволюционную модель.

### 5.2 Fixed template rendering

`src/evolve/template_parser.py:11-60` уже реализует fixed-template подход:

- template хранится отдельно;
- editable sections выделены явно;
- код собирается из sections, а не генерируется полностью как угодно.

Это очень близко к твоему желаемому invariant: сигнатура фиксирована, LLM меняет только допустимые места.

### 5.3 Общий validation scaffold

`valopt/runner.py` и `experiments/*` уже дают хороший единый рантайм для задач разной сложности. Это полезно для обеих фаз отбора.

### 5.4 Локальная test coverage

Тесты покрывают много локальных механизмов:

- allocation;
- scoring math;
- import optimizer;
- structured generation helpers;
- mutation/crossbreeding basics.

Проблема не в отсутствии тестов вообще, а в том, что они почти не страхуют product-level semantics.

## 6. Mismatch ledger: требования против текущей реализации

Это главная часть документа. Ниже каждый пункт оформлен так:

- что должно быть по ТЗ;
- что сделано сейчас;
- доказательство;
- почему это риск;
- что делать.

### 6.1 Blocker: в репозитории нет island model и school-based initialization

#### Как должно быть

Система должна:

- читать количество школ по `.txt` файлам в `conf`;
- хранить отдельное описание каждой школы;
- порождать стартовую популяцию по школам;
- затем вести отбор и размножение по островам.

#### Что есть сейчас

В `conf` нет ни одной папки с school descriptions и нет ни одного `.txt` prompt/config asset. Весь `conf` состоит только из `config.yaml`, `evolver/default.yaml` и experiment YAMLs.

Доказательства:

- список файлов в `conf` ограничен YAML-файлами: `find conf -type f` -> `conf/config.yaml`, `conf/evolver/default.yaml`, `conf/experiments/*.yaml`.
- `src/evolve/evolution_loop.py:105-127` инициализирует популяцию через `SeedOperator()` без island/school semantics.
- `src/evolve/operators.py:51-57` seed generation использует общий `seed_system.txt` и `seed_user.txt`.

#### Почему это критично

Без школ/островов система не реализует один из центральных поисковых bias-ов твоего дизайна: несколько исследовательских направлений должны конкурировать и изолированно эволюционировать. Сейчас весь поиск идет в одной общей популяции.

#### Что делать

Нужен новый конфиг-слой, например:

- `conf/prompts/system_seed.txt`
- `conf/islands/<island_name>.txt`
- `conf/evolver/islands.yaml` или блок `evolver.islands` в existing config

И нужен explicit runtime data model:

- `IslandConfig`
- `IslandState`
- `Organism.island_id`

После этого initialization должно распределять seed count по островам и строить initial organisms из island-specific text.

### 6.2 Blocker: prompts живут в коде, а не в конфиге

#### Как должно быть

Ты явно зафиксировал, что:

- constant system prompt хранится в `./conf`;
- school descriptions хранятся в `./conf`;
- шаблон должен быть фиксирован;
- prompt layer должен быть конфигурацией системы, а не зашитой частью implementation package.

#### Что есть сейчас

`OptimizerGenerator` по умолчанию берет prompts из `src/evolve/prompts`.

Доказательства:

- `src/evolve/generator.py:42-48`
- `src/organisms/mutation.py:75-76`
- `src/organisms/crossbreeding.py:88-89`

Внутри `src/evolve/prompts` уже есть дублирующиеся prompt-файлы:

- `mutation_user.txt` и `mutate_user.txt`
- `crossover_user.txt` и `crossbreed_user.txt`

Это явный признак prompt drift.

#### Почему это критично

Когда prompts лежат внутри runtime package:

- их труднее version-control'ить как исследовательские policy assets;
- сложнее сделать school-specific prompt hierarchy;
- появляется legacy duplication;
- система хуже воспроизводима как scientific engine.

#### Что делать

Вынести prompts в `conf/prompts/` и разделить на:

- global immutable system prompt;
- operator-specific user prompt templates;
- school descriptions;
- possibly mutation/crossover instruction overlays.

Код должен читать prompt paths из config, а не собирать их из `Path(__file__).resolve().parent / "prompts"`.

### 6.3 Blocker: сейчас существуют две конкурирующие эволюционные архитектуры

#### Как должно быть

Нужен один канонический evolve engine и один канонический формат артефактов.

#### Что есть сейчас

Есть:

- `cand_*` candidate pipeline через `EvolverOrchestrator`;
- `org_*` population pipeline через `EvolutionLoop`.

Доказательства:

- `src/evolve/storage.py:63-99` vs `src/evolve/storage.py:128-159`
- `src/evolve/orchestrator.py:223-260` генерирует `cand_*`
- `src/evolve/evolution_loop.py:105-183` работает с `org_*`
- `src/evolve/run.py:36-45` по умолчанию выбирает `EvolutionLoop`
- `README.md:148-160` документирует `cand_*` outputs как будто они canonical

#### Почему это критично

Сейчас невозможно однозначно ответить на вопросы:

- что такое "особь" в системе;
- где лежит ее canonical metadata;
- какой execution path считается production;
- каким путем реально обновляется evolutionary context.

Это мешает и разработке, и отладке, и анализу научных результатов.

#### Что делать

Принять одно из двух решений:

1. Полностью канонизировать organism model и встроить orchestration как evaluator-only service.
2. Полностью канонизировать candidate model и убрать organism-specific storage/model.

Для твоего ТЗ правильнее первый вариант: канонизировать organism/island model, а `EvolverOrchestrator` оставить как evaluation backend.

### 6.4 Blocker: genetic code слишком бедный и не соответствует твоей модели `b)`

#### Как должно быть

Генетический код должен быть богатым текстовым объектом, который:

- содержит rationale;
- хранит inherited ideas;
- позволяет meaningful gene-level edits;
- сам является главным носителем идеи оптимизатора.

#### Что есть сейчас

`idea_dna` хранится как список строк, который сериализуется в `idea_dna.txt` через `"; ".join(idea_dna)`.

Доказательства:

- `src/organisms/organism.py:58`
- `src/organisms/organism.py:85-90`
- `src/organisms/crossbreeding.py:31-73`
- `src/organisms/mutation.py:31-60`

По факту genetic code сейчас - это flat list коротких traits.

#### Почему это критично

Такое представление годится для toy evolutionary algorithm, но не для твоей исследовательской задачи. Из flat trait list почти невозможно:

- восстановить реальную идею optimizer;
- выразить сложное inherited reasoning;
- корректно анализировать, почему score рос или падал;
- делать качественный LLM edit в духе "удалить/изменить/добавить гены".

#### Что делать

Нужен richer schema, например:

- gene id;
- gene text;
- optional tags/category;
- provenance;
- status `kept/modified/removed/new`;
- rationale blob;
- maybe compact natural-language paragraphs split into named sections.

Минимальный pragmatic вариант:

- хранить `genetic_code.md` или `genetic_code.json`;
- разделить на `Core Principles`, `Mechanisms`, `Rejected Ideas`, `Open Hypotheses`;
- внутри sections поддерживать semicolon gene list только как derived view, а не как canonical storage.

### 6.5 Blocker: mutation не сохраняет новые или измененные гены, которые LLM предлагает

#### Как должно быть

По твоему ТЗ mutation - это не просто случайное удаление traits. После мутации LLM должна осмысленно переработать gene code:

- удалить что-то;
- изменить что-то;
- добавить что-то;
- затем сохранить уже новый genetic code.

#### Что есть сейчас

`mutation_system.txt` и `mutate_user.txt` прямо говорят модели, что она может добавить одну новую идею. Но `MutationOperator.produce()` передает в `build_organism_from_response(...)` параметр `idea_dna_override=child_dna`, то есть сохраняется только заранее вычисленный список surviving traits. Все, что LLM вернула в `## IDEA_DNA`, отбрасывается.

Доказательства:

- Prompt разрешает add-one-new-idea: `src/evolve/prompts/mutate_user.txt`
- `src/organisms/mutation.py:133-156`
- `src/organisms/organism.py:85-90`

#### Почему это критично

Это не просто архитектурная неточность, а прямой semantic bug:

- prompt говорит одно;
- persistence layer делает другое.

То есть текущая mutation pipeline обещает эволюцию genetic code, но фактически запрещает ее.

#### Что делать

Разделить два режима:

- `pre_llm_mutation` как proposal of changed genes;
- `post_llm_genetic_code` как canonical accepted DNA.

`build_organism_from_response()` не должен blindly override `IDEA_DNA`, когда оператору разрешено gene rewriting. Для mutation override допустим только как fallback safety net, но не как canonical truth.

### 6.6 High: crossover тоже не дает LLM реально переписывать gene code

#### Как должно быть

Ты описал crossover так, что:

- выбираются мать и отец;
- по вероятности наследуются гены;
- LLM получает lineage и gene set;
- затем думает, что менять в этом коде и как обновить описание гена `b)`.

#### Что есть сейчас

`CrossbreedingOperator.produce()` сначала deterministically/probabilistically строит `child_dna`, а затем снова передает его как `idea_dna_override`. То есть финальный `IDEA_DNA` из ответа LLM не является источником истины.

Доказательства:

- `src/organisms/crossbreeding.py:145-146`
- `src/organisms/crossbreeding.py:173-185`
- `src/organisms/organism.py:85-90`

#### Почему это плохо

Такой design делает crossover чем-то средним между:

- fixed trait recombination;
- code synthesis for predetermined trait set.

Но это не соответствует твоей идее, где LLM должна еще и осмысленно переработать gene description после смешения предков.

#### Что делать

Нужно разделить:

- `inherited_gene_pool`
- `proposed_gene_code`
- `accepted_gene_code`

Для crossover допустим policy "LLM не может добавлять совершенно посторонние идеи", но она все равно должна иметь право переписать textual gene code и сделать его richer/cleaner/less contradictory.

### 6.7 Blocker: lineage не получает реальные score назад

#### Как должно быть

В lineage каждая запись должна завершаться результатом оценки.

#### Что есть сейчас

При создании organism новая запись добавляется с `score=None`. После evaluation обновляются поля `simple_score`, `hard_score`, `score`, но `evolution_log` назад не переписывается.

Доказательства:

- `src/organisms/organism.py:125-133`
- `src/evolve/evolution_loop.py:227-240`

#### Почему это критично

Это ломает саму идею `c)`:

- lineage теряет обучающую ценность для следующих поколений;
- LLM не видит полную историю "что меняли -> что вышло";
- система не умеет объяснять, какие gene edits были удачными.

#### Что делать

После каждой successful evaluation нужно:

- найти последнюю незавершенную lineage entry;
- записать туда simple score;
- если есть Great Filter, записать туда hard score отдельно;
- сохранить updated lineage artifacts.

Лучше явно ввести:

- `simple_score`
- `hard_score`
- `aggregate_score`
- `phase`

внутри `LineageEntry`, а не только на уровне organism.

### 6.8 High: crossover наследует не материнскую линию, а смешанный log обоих родителей

#### Как должно быть

По твоему ТЗ у crossover есть мать и отец, а родословная хранится по материнской линии.

#### Что есть сейчас

Код использует роли `dominant` и `non_dominant`, определяя их по score, а не по maternal/paternal semantics. Потом lineage child собирается как простой merge логов обоих родителей.

Доказательства:

- `src/evolve/evolution_loop.py:157-160`
- `src/organisms/crossbreeding.py:166-169`
- `src/organisms/crossbreeding.py:171-185`

#### Почему это критично

Это противоречит твоей модели происхождения:

- мать и отец перестают быть реальными ролями;
- canonical path к особи теряется;
- lineage становится смесью двух журналов, а не историей конкретной линии.

Дополнительно здесь есть маленький code smell:

- комментарий говорит "keep unique", но код только сортирует и не делает deduplication.

#### Что делать

Нужна явная модель:

- `mother_id`
- `father_id`
- `lineage_source = mother`
- `other_parent_contribution_summary`

Если нужен полный граф предков, его надо хранить отдельно от canonical lineage.

### 6.9 Blocker: selection semantics не соответствуют ТЗ

#### Как должно быть

По твоему описанию:

- crossover parents выбираются из softmax по reward;
- mutation sampling идет равномерно;
- mother/father задаются явно;
- selection делается per-island;
- есть top-k и top-h как разные числовые параметры.

#### Что есть сейчас

Сейчас selection построен иначе:

- `elite_select` берет top-N по score;
- parents выбираются tournament selection;
- mutation и crossover оба используют selection только среди survivors;
- `mutation_rate` определяет branching, а отдельный `crossover_rate` из конфига не участвует;
- нет `top-k` / `top-h`, только один `elite_count`.

Доказательства:

- `src/evolve/selection.py:10-70`
- `src/evolve/evolution_loop.py:247-250`
- `src/evolve/evolution_loop.py:279-295`
- `conf/evolver/default.yaml:20-27`

#### Почему это критично

Это уже не "вариант реализации", а другая search dynamics:

- tournament pressure сильно меняет exploration/exploitation;
- mutation больше не uniform;
- `crossover_rate` фактически мертвый knob;
- нет per-island отбора;
- нет разделения `top-k` и `top-h`.

#### Что делать

Переписать selection layer вокруг двух явных функций:

- `sample_crossover_parents(island_population, reward_softmax_temperature, mother_bias_cfg, ...)`
- `sample_mutation_targets(island_population, uniform=True, count=...)`

И заменить `elite_count` на два разных параметра:

- `simple_top_k`
- `great_filter_top_h`

плюс island-aware state.

### 6.10 Blocker: Great Filter реализован с неправильной семантикой

#### Как должно быть

Great Filter должен:

- использовать тяжелые задачи;
- влиять только на право дальнейшего размножения;
- не затирать постоянный simple reward;
- иметь отдельный порог/размер выживших.

#### Что есть сейчас

В Great Filter код делает следующее:

- собирает `all_experiments = simple_experiments + hard_experiments`;
- переоценивает всю популяцию на объединенном наборе;
- пишет это в `hard_score`;
- затем отбирает по `hard_score` через тот же `elite_count`.

Доказательства:

- `src/evolve/evolution_loop.py:320-329`

Дополнительно:

- `org.score` обновляется как max из старого и нового score: `src/evolve/evolution_loop.py:232-234`

#### Почему это критично

Это нарушает твой design сразу в нескольких местах:

1. Great Filter у тебя должен быть hard-only, а не simple+hard aggregate.
2. Число выживших после Great Filter должно быть отдельным параметром, а не reuse `elite_count`.
3. `score = max(old_score, new_score)` ломает фазовую семантику организма: непонятно, какой именно score потом используется как basis для dominance/selection.

#### Что делать

Нужно явно разделить:

- `simple_reward`
- `hard_filter_reward`
- `reproduction_reward`

И Great Filter должен вычислять reproduction eligibility на hard tasks отдельно от постоянного simple reward.

### 6.11 High: simple/hard allocation конфиги объявлены, но по сути не подключены

#### Как должно быть

У simple phase и Great Filter должны быть свои allocation semantics.

#### Что есть сейчас

В конфиге есть:

- `evaluation.simple_allocation`
- `evaluation.hard_allocation`

Но evaluation pipeline использует глобальный `evolver.allocation`.

Доказательства:

- конфиг: `conf/evolver/default.yaml:29-57`
- `EvolverOrchestrator` читает только `cfg.evolver.get("allocation", {})`: `src/evolve/orchestrator.py:44-51`
- `EvolutionLoop._evaluate_organisms()` не переключает allocation config между simple/hard phase: `src/evolve/evolution_loop.py:199-208`

#### Почему это важно

Сейчас конфиг обещает две разные allocation политики, но кодом используется одна. Это опасный вид configuration drift: экспериментатор думает, что управляет стратегией, а реально ничего не меняется.

#### Что делать

Либо:

- удалить dead keys,

либо, что правильнее для твоего ТЗ:

- передавать separate allocation config в simple и hard phases;
- на уровне orchestrator иметь explicit parameter `allocation_cfg`.

### 6.12 Medium: `score_weights` - мертвый конфиг после reward rework

#### Что должно быть

После перехода на harmonic mean от train-loss baseline-relative terms линейные `score_weights` больше не должны быть активной частью scoring path.

#### Что есть сейчас

`conf/evolver/default.yaml` все еще содержит:

- `allocation.score_weights.quality`
- `allocation.score_weights.steps`

Но `extract_metrics(...)` делает `del scoring_cfg` и никак эти веса не использует.

Доказательства:

- `conf/evolver/default.yaml:55-57`
- `src/evolve/metrics_adapter.py:26-35`

#### Почему это важно

Это не блокер, но это прямой источник будущих ошибок настройки.

#### Что делать

Удалить `score_weights` из active config или оставить его только как legacy-commented field до cleanup migration.

### 6.13 Medium: `normalization.quality_ref` и `steps_ref` все еще висят в docs/configs и создают ложную модель scoring

#### Что должно быть

После reward rework система должна везде говорить одно и то же: baseline-driven scoring.

#### Что есть сейчас

В `conf/experiments/*.yaml` до сих пор есть `quality_ref` и `steps_ref`. README тоже документирует их как часть evolve normalization.

Доказательства:

- `README.md:169-171`
- `conf/experiments/*.yaml`
- `agents/4_train_loss_baseline_reward_followup.md` отдельно говорит, что эти ключи уже устарели логически

#### Почему это важно

Сейчас новый инженер или другой контекст будет ошибочно думать, что:

- scoring все еще идет через статические refs;
- baseline profile не является canonical source of truth.

#### Что делать

Нужен cleanup pass:

- docs cleanup;
- config cleanup или explicit deprecation note;
- test assertions, что reward действительно не зависит от `quality_ref`/`steps_ref`.

### 6.14 Blocker: optimizer contract не согласован между README, runtime, prompts и tests

#### Как должно быть

Во всем проекте должен быть один optimizer contract.

#### Что есть сейчас

README документирует старый контракт:

- `build_optimizer(cfg)`
- `initialize(named_parameters, cfg)`
- `step(weights, grads, activations)`

Доказательства:

- `README.md:126-145`

Runtime использует новый контракт:

- `build_optimizer(model, max_steps)`
- `step(weights, grads, activations, step_fn)`
- `zero_grad(...)`

Доказательства:

- `valopt/optimizer_api.py:18-36`
- `valopt/utils/import_utils.py:51-71`
- `src/validate/run_one.py:107-112`

Тест `tests/test_optimizer_generator.py` все еще считает корректным код старого формата:

- `build_optimizer(cfg)`
- `initialize(...)`
- `step(... without step_fn)`

Доказательства:

- `tests/test_optimizer_generator.py:31-50`

#### Почему это критично

Это один из самых опасных drift-ов в проекте, потому что бьет по interface contract. В такой ситуации:

- документация врет пользователю;
- часть тестов подтверждает неправильную сигнатуру;
- validator в `OptimizerGenerator._validate_code()` слишком слабый и тоже не гарантирует реальный контракт.

#### Что делать

Нужно одним проходом выровнять:

- README;
- runtime validator;
- generator validator;
- prompts;
- tests;
- example optimizers.

И ввести один canonical helper для contract validation, чтобы его не дублировать в нескольких местах.

### 6.15 High: `OptimizerGenerator._validate_code()` валидирует не тот контракт, который реально нужен runtime

#### Что должно быть

Generator должен принимать только тот optimizer code, который реально можно загрузить и выполнить в runtime.

#### Что есть сейчас

`_validate_code()` проверяет:

- есть ли `build_optimizer`;
- есть ли class с `step` и `zero_grad`.

Но он не проверяет:

- сигнатуру `build_optimizer(model, max_steps)`;
- наличие `step_fn` в `step`;
- наличие `__init__(model, max_steps)` на уровне класса;
- совместимость с actual import/runtime path.

Доказательства:

- `src/evolve/generator.py:80-109`

Для organism-template path `validate_rendered_code()` уже строже:

- требует `__init__/step/zero_grad`.

Доказательства:

- `src/evolve/template_parser.py:63-94`

#### Почему это важно

У тебя уже два разных валидатора с разной строгостью. Это еще один признак architectural split.

#### Что делать

Сделать один shared validator module и использовать его в:

- template renderer;
- candidate generator;
- import-time smoke validation.

### 6.16 High: selection и dominance опираются на агрегатный `score`, который семантически неустойчив

#### Как должно быть

Если в системе есть несколько score-типов, нужно явно знать, какой из них используется:

- для отбора;
- для crossover parent role assignment;
- для ranking;
- для reporting.

#### Что есть сейчас

`CrossbreedingOperator` выбирает `dominant` по `org.score`, а `org.score` обновляется через:

- initial simple evaluation;
- иногда hard evaluation;
- логикой "если новый score выше, положить его в `org.score`".

Доказательства:

- `src/evolve/evolution_loop.py:157-160`
- `src/evolve/evolution_loop.py:232-234`

#### Почему это важно

Это делает роль "dominant parent" неустойчивой и phase-dependent. После Great Filter организм может считаться dominant не потому, что лучше в simple phase, а потому что ему однажды повезло на другом aggregate score.

#### Что делать

Запретить use of ambiguous `score` в core logic. Должны быть отдельные поля:

- `simple_reward`
- `hard_reward`
- `selection_reward`
- `display_reward`

### 6.17 Medium: `selection_strategy`, `crossover_rate`, `novelty_filter`, `bandit_llm_selection` сейчас mostly decorative

#### Что есть

В конфиге объявлены:

- `selection_strategy`
- `crossover_rate`
- `novelty_filter.enabled`
- `bandit_llm_selection.enabled`

Но кодом:

- `selection_strategy` нигде не используется;
- `crossover_rate` не участвует в reproduction logic, потому что branching идет только через `mutation_rate`;
- novelty/bandit вообще не имеют runtime references.

Доказательства:

- `conf/evolver/default.yaml:3-4`
- `conf/evolver/default.yaml:22-23`
- `conf/evolver/default.yaml:59-63`
- поиск по коду показывает отсутствие runtime usage для novelty/bandit

#### Почему это важно

С scientific platform это плохой запах: конфиг создает видимость research features, которых на самом деле нет.

#### Что делать

Любой из этих knobs должен быть либо:

- полностью реализован,

либо:

- удален из active config,

либо:

- явно помечен как stub с runtime warning.

### 6.18 Medium: prompt layer уже содержит legacy duplication и semantic drift

#### Что есть

В prompt assets одновременно существуют:

- `crossbreed_user.txt` и `crossover_user.txt`
- `mutate_user.txt` и `mutation_user.txt`

Код при этом использует их по-разному:

- `operators.py` использует legacy prompt-only operators
- `mutation.py` и `crossbreeding.py` используют новые probabilistic operators

#### Почему это важно

Это почти гарантированно приведет к drift в behavior и к путанице при дальнейшей доработке.

#### Что делать

После канонизации одного execution model:

- удалить legacy prompts;
- оставить только canonical operator prompts;
- зафиксировать naming convention.

### 6.19 Medium: lineage prompt context искусственно обрезан и может терять полезную историю

#### Что есть

`format_evolution_log(...)` ограничивает историю пятью последними записями.

Доказательства:

- `src/organisms/organism.py:24`
- `src/organisms/organism.py:33-47`

#### Почему это может быть проблемой

Если lineage действительно должен быть главным носителем исследовательской истории, последние пять записей могут быть недостаточны. Особенно если early turning points были самыми важными.

#### Что делать

Не обязательно всегда отправлять весь lineage в prompt, но нужен richer summarization layer:

- краткий lineage summary;
- последние N raw entries;
- явные best/worst edits.

### 6.20 Medium: Great Filter survivors replenished by random seeds ломают смысл island continuity

#### Что есть

После Great Filter недостающая часть популяции добивается новыми seed organisms:

- `src/evolve/evolution_loop.py:337-340`

#### Почему это спорно

Для single-population toy loop это приемлемо, но в твоем дизайне после Great Filter логичнее:

- поддерживать island continuity;
- seed new organisms per-island policy;
- явно управлять immigration/exploration rate.

#### Что делать

После внедрения islands replenishment должен стать island-aware:

- `reseed_within_island`
- `cross_island_migration`
- `new_school_seed_budget`

## 7. Проблемы уровня документации и восприятия системы

Это отдельный важный слой, потому что у тебя уже есть несколько исторических отчетов в `agents`, и новый инженер будет опираться на них.

### 7.1 README сейчас описывает уже не тот проект, который реально живет в коде

Главные точки drift:

- optimizer contract устарел;
- evolve outputs документированы как `cand_*`, хотя дефолтный путь сейчас multi-generation organism loop;
- `normalization.quality_ref/steps_ref` описаны как будто активны;
- про islands/schools/prompt-in-conf вообще ничего нет.

### 7.2 `agents/4_train_loss_baseline_reward_followup.md` описывает сильную локальную эволюцию reward path, но может создавать ложное ощущение, что overall evolve architecture уже консистентна

На самом деле reward path действительно стал лучше, но он встроен в архитектуру, которая еще не соответствует целевой эволюционной модели.

### 7.3 `agents/3_previous_session_followup.md` полезен как walkthrough, но он описывает скорее состояние "framework + evolve scaffold", а не островную discovery platform

Это не ошибка файла, но это важно помнить при handoff.

## 8. Какие product-level риски самые опасные прямо сейчас

Ниже не просто "список багов", а риски того, что система начнет производить misleading science outcomes.

### 8.1 Ложная эволюция genetic code

Система может выглядеть как evolving gene descriptions, но реально mutation/crossover не сохраняют LLM-updated DNA как canonical truth.

### 8.2 Ложный контроль через конфиг

Экспериментатор может менять:

- `simple_allocation`
- `hard_allocation`
- `selection_strategy`
- `crossover_rate`
- `novelty_filter`
- `bandit_llm_selection`

и думать, что меняет поведение системы. Во многих случаях код этого не делает.

### 8.3 Ложная уверенность из-за тестов

Текущий test suite хороший на уровне локальных механизмов, но он почти не ловит:

- product drift;
- dead config;
- island absence;
- lineage score backfill absence;
- contract inconsistency between README/tests/runtime;
- hybrid `cand_*` vs `org_*` architecture.

### 8.4 Ложная интерпретация score

Текущий `org.score` может смешивать simple и hard phase semantics. Это опасно для scientific interpretation: можно принимать решения на основе семантически неоднородного scalar.

## 9. Что именно я бы сохранил, а что канонически удалил/заменил

### 9.1 Сохранить

- `valopt/` как experiment runtime;
- baseline-relative reward path;
- `step_fn` contract;
- fixed template rendering;
- `TrainObjectiveTracker`;
- `run_one` как subprocess evaluator;
- основные experiment modules;
- GPU pool/orchestrator как infra для evaluation tasks.

### 9.2 Канонизировать

- organism/island data model;
- one canonical prompt/config location in `conf`;
- one canonical optimizer contract;
- one canonical artifact layout.

### 9.3 Удалить или перевести в legacy

- дублирующие prompt files;
- старый README optimizer contract;
- dead config knobs без реализации;
- ambiguous `score` as core logic field;
- `cand_*` vs `org_*` dualism, если выбирается organism-first architecture.

## 10. Рекомендуемая целевая архитектура после рефакторинга

Ниже уже не диагноз, а практическое предложение, как довести систему до твоего ТЗ.

### 10.1 Канонический layout конфигов

Пример:

```text
conf/
  config.yaml
  evolver/default.yaml
  prompts/
    system_optimizer.txt
    seed_user.txt
    mutation_user.txt
    crossover_user.txt
  islands/
    momentum_school.txt
    activation_aware_school.txt
    schedule_hybrid_school.txt
```

И в `evolver/default.yaml`:

- `islands.path`
- `islands.initial_population_per_island`
- `selection.simple_top_k`
- `selection.great_filter_top_h`
- `selection.crossover_parent_sampling = softmax_reward`
- `selection.mutation_parent_sampling = uniform`
- `operators.crossover.gene_inherit_p`
- `operators.mutation.gene_delete_q`

### 10.2 Канонический organism schema

Минимально нужны:

- `organism_id`
- `island_id`
- `mother_id`
- `father_id | null`
- `genetic_code`
- `code_artifact_path`
- `lineage_entries`
- `simple_reward`
- `hard_reward`
- `selection_reward`
- `status`

А `LineageEntry`:

- `generation`
- `change_description`
- `gene_diff_summary`
- `simple_score`
- `hard_score`
- `aggregate_score`
- `operator`
- `source_parent_ids`

### 10.3 Канонический execution flow

1. Load island configs from `conf/islands/*.txt`.
2. Seed per-island population via system prompt + school description.
3. Run simple evaluation for all newborns.
4. Backfill simple reward into lineage.
5. Select `top-k` per island.
6. Generate offspring:
   - within-island crossover,
   - optional cross-island crossover,
   - mutation via uniform parent sampling.
7. Every `great_filter_interval` run hard-only filter.
8. Select `top-h` per island for reproduction rights.
9. Preserve lineage and genetic code as canonical research artifacts.

### 10.4 Канонический storage layout

Если брать organism-first model, я бы рекомендовал такой формат:

```text
populations/
  gen_0001/
    island_<name>/
      org_<id>/
        optimizer.py
        genetic_code.md
        lineage.json
        organism.json
        results/
          simple/
            <experiment>.json
          hard/
            <experiment>.json
        summary.json
```

Такой layout сразу делает явными:

- island membership;
- organism identity;
- phase-separated evaluation artifacts.

## 11. Ordered implementation roadmap

Ниже порядок, в котором это стоит делать, чтобы не множить технический долг.

### Этап 0. Freeze semantics и docs

Сначала нужно зафиксировать один canonical design document:

- что такое organism;
- что такое island;
- что такое genetic code;
- какие score поля canonical;
- какой optimizer contract canonical.

Без этого любые точечные кодовые правки будут только плодить еще один partial design.

### Этап 1. Unify evolution architecture

Первый реальный кодовый шаг:

- решить судьбу `cand_*` vs `org_*`;
- канонизировать organism-first path;
- сделать `EvolverOrchestrator` evaluator backend, а не параллельную модель сущности.

До этого не стоит добавлять islands.

### Этап 2. Fix optimizer contract everywhere

В одном проходе:

- README;
- runtime import validator;
- generator validator;
- tests;
- examples;
- prompts.

Это сравнительно локальная, но крайне важная правка.

### Этап 3. Move prompts/configurable text into `conf`

После этого:

- удалить prompt duplication;
- подключить config-driven prompt paths;
- подготовить school descriptions.

### Этап 4. Introduce island model

Добавить:

- island configs;
- per-island population state;
- per-island seeding;
- per-island selection;
- optional cross-island crossover policy.

### Этап 5. Rebuild genetic code and lineage model

Это ключевой этап для научной полезности:

- richer gene representation;
- post-LLM accepted genetic code;
- lineage score backfill;
- maternal lineage semantics.

### Этап 6. Rework selection and Great Filter

Здесь нужно:

- убрать tournament where inappropriate;
- ввести softmax parent sampling для crossover;
- uniform parent sampling для mutation;
- separate `top-k` / `top-h`;
- hard-only Great Filter.

### Этап 7. Clean dead config and docs

Удалить или deprecated:

- `quality_ref`
- `steps_ref`
- `score_weights`
- dead strategy toggles

или реализовать их полноценно.

### Этап 8. Upgrade test surface

После архитектурного выравнивания добавить tests не на локальные helper'ы, а на инварианты системы.

## 12. Какой тестовый контур нужен после рефакторинга

Сейчас тестов много, но они не страхуют главные product-level свойства. Нужны новые тесты.

### 12.1 Invariant tests

- seed initialization from multiple island txt files;
- per-island population counts;
- maternal lineage preservation after crossover;
- lineage score backfill after simple eval;
- hard filter does not overwrite simple reward;
- mutation can persist LLM-added gene;
- crossover can persist LLM-updated gene description.

### 12.2 Config truthfulness tests

- `simple_allocation` реально влияет только на simple phase;
- `hard_allocation` реально влияет только на hard phase;
- dead knobs отсутствуют или вызывают explicit warning;
- system prompt paths come from `conf`, not package internals.

### 12.3 Contract tests

- README example optimizer actually loads;
- generator validator and runtime importer accept/reject exactly the same contracts;
- generated optimizer with old `build_optimizer(cfg)` contract must fail consistently everywhere.

### 12.4 End-to-end integration tests

- multi-island seed -> simple eval -> selection -> crossover/mutation -> Great Filter;
- artifact layout is canonical and consistent;
- reload/resume works without changing semantics.

## 13. Конкретные маленькие баги и smells, которые тоже стоит исправить

Ниже менее стратегические, но реальные issues.

### 13.1 Комментарий в crossover врет про deduplication логов

`src/organisms/crossbreeding.py:166-169` говорит "keep unique", но uniqueness не реализована.

### 13.2 `EvolutionLoop._evaluate_organisms()` сам признает себя как partially integrated path

Комментарий `src/evolve/evolution_loop.py:191-196` говорит, что это "For now" и "placeholder scores", хотя это уже реальный путь исполнения. Это признак unfinished migration.

### 13.3 README описывает `cand_*` outputs как canonical, хотя default evolve path идет через `EvolutionLoop`

Это создает практическую путаницу при дебаге артефактов на диске.

### 13.4 Mixed naming `crossbreed`/`crossover`, `mutate`/`mutation`

Это кажется мелочью, но именно такие naming drifts потом мешают поддержке prompt stack.

## 14. Самые важные выводы в одной секции

Если свести весь аудит к самым важным пунктам, то их пять:

1. Система еще не реализует island/school-based research engine. Сейчас это single-population hybrid.
2. Genetic code и lineage пока слишком бедные и semantically broken относительно твоего ТЗ.
3. В проекте есть сильный evaluation/reward scaffold, который стоит сохранить.
4. Главная техническая проблема - не отсутствие функциональности как таковой, а coexistence нескольких несовместимых архитектур.
5. Следующий правильный шаг - не точечные фичи, а канонизация data model, prompt/config layer и selection semantics.

## 15. Практическая рекомендация, что делать прямо следующим ходом

Если приоритизировать ruthless pragmatically, я бы делал так:

### Шаг 1

Сначала зафиксировать one-pager canonical architecture:

- organism-first;
- island-aware;
- baseline-relative scoring;
- one optimizer contract.

### Шаг 2

Сразу после этого убрать самое опасное расхождение:

- unified optimizer contract;
- unified evolve artifact model;
- prompts from `conf`.

### Шаг 3

Потом уже вводить islands и rich genetic code.

### Шаг 4

И только затем доделывать novelty/bandit/model routing features в духе ShinkaEvolve/AdaptEvolve.

Иначе получится, что advanced research features будут навешаны на неканоническую базовую модель.

## 16. Итог

Репозиторий уже содержит полезный фундамент для optimizer evaluation и baseline-relative reward. Но как платформа для controlled evolutionary discovery новых optimizer algorithms он пока архитектурно недособран.

Главные расхождения не косметические:

- отсутствуют острова и школы;
- genetic code и lineage не дотягивают до роли, которую ты им отводишь;
- selection/Great Filter semantics отличаются от ТЗ;
- конфиг и docs местами обещают не то, что реально делает код;
- coexistence `cand_*` и `org_*` разрушает ощущение единой модели системы.

Правильная стратегия дальше - не латать это по одному месту, а провести один осознанный refactor toward canonical organism/island model, сохранив уже хороший validation and reward infrastructure.
