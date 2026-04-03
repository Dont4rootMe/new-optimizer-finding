# Текущий re-audit репозитория: island-aware optimizer discovery platform после большой волны изменений

## 0. Что это за документ

Этот файл является свежим re-audit текущего working tree репозитория.

Он не заменяет:

- `agents/6_gpt_revision_merged_comprehensive_audit_and_roadmap.md`
- `agents/7_gpt_master_execution_spec_for_optimizer_discovery_platform.md`

Его задача другая:

- посмотреть на то, что реально уже изменили в коде;
- сверить это с твоим текущим ТЗ;
- разделить изменения на `исправлено / частично исправлено / осталось открытым / регрессировало / сделано, но semantic drift остался`;
- найти новые баги и несостыковки, которые появились уже в процессе миграции;
- дать обновленный консультационный документ, по которому можно делать следующий крупный шаг разработки.

Источник истины в этом отчете такой:

1. Твое текущее сообщение с ТЗ.
2. Текущий рабочий код и конфиги в репозитории.
3. Текущий test surface.
4. Исторические документы в `agents/`.
5. Внешний research framing из [ShinkaEvolve](https://sakana.ai/shinka-evolve/) от 25 сентября 2025 года и [AdaptEvolve](https://arxiv.org/abs/2602.11931) от 12 февраля 2026 года.

Сразу зафиксирую контекст текущего состояния репозитория, который я использовал как стартовую точку:

- в репозитории появились `conf/prompts/*`;
- в репозитории появились `conf/islands/*.txt`;
- добавлен `src/evolve/islands.py`;
- добавлен `src/evolve/prompt_utils.py`;
- добавлены тесты на `resume`, `islands`, `prompt bundle`;
- `pytest -q` сейчас проходит целиком;
- текущее состояние тестов: `63 passed in 2.70s`;
- legacy prompt assets в `src/evolve/prompts/*` все еще лежат;
- legacy candidate-first path и old orchestrator pipeline все еще живы;
- часть compatibility-bridge логики уже перенесена в canonical path, но не полностью.

Это автоматически означает, что новый аудит должен смотреть не только на feature completeness, но и на migration drift.

## 1. Краткий вердикт

Короткая версия:

- репозиторий реально сильно сдвинулся в правильную сторону по сравнению с состоянием, зафиксированным в `6.md`;
- основные structural pieces из target architecture уже появились;
- это уже не тот проект, где islands, prompts in `conf`, manifest-based resume и lineage score backfill полностью отсутствовали;
- но migration еще не завершена;
- сейчас проект находится в опасной промежуточной фазе, где canonical organism-first system уже частично есть, а legacy candidate-first machinery и compatibility fallbacks все еще остаются operational;
- именно поэтому текущий репозиторий выглядит намного лучше на уровне архитектурных деклараций, README и тестов, чем на уровне полного semantic closure.

Самое важное:

- новый код исправил много P0-проблем из прошлого аудита;
- но теперь главные риски сместились из зоны `ничего не реализовано` в зону `реализовано много, но рядом остались скрытые compatibility paths, math bugs и config lies`;
- passing tests сейчас уже не означают, что system semantics совпадает с твоим ТЗ.

Мой обновленный high-level verdict:

- `island model`: реализован частично, но уже реальный;
- `prompt relocation into conf`: реализован частично, но не доведен до полного source-of-truth closure;
- `manifest-based resume`: реализован и полезен;
- `lineage score backfill`: реализован;
- `simple vs hard phase separation`: реализована частично;
- `optimizer contract unification`: в основном реализована;
- `legacy candidate path retirement`: не сделана;
- `selection semantics truthfulness`: частично реализована, но все еще содержит критические несоответствия;
- `reward semantics`: частично реализована, но содержит вероятную математическую проблему в aggregation with allocation;
- `Great Filter as genuinely harder phase`: заявлен, но не полностью подтвержден runtime semantics.

Если сжать еще сильнее:

- основная работа проделана не зря;
- migration принесла реальный прогресс;
- но система все еще не “закрыта” концептуально;
- сейчас главное не продолжать наращивать новые возможности поверх промежуточного слоя, а дочистить truthfulness и semantic consistency.

## 2. Каноническая модель системы по твоему ТЗ

Перед аудитом важно еще раз очень четко зафиксировать целевую модель.

Иначе невозможно честно оценить, что уже сделано, а что только выглядит сделанным.

### 2.1 Особь

Каждая особь должна быть не просто `optimizer.py`.

Каждая особь должна быть research artifact, состоящий минимум из:

- кода оптимизатора;
- богатого генетического кода;
- lineage-истории;
- score-истории;
- связи с island/school;
- фиксированной сигнатуры runtime-контракта.

В твоем ТЗ особь не является “просто candidate”.

Она является носителем идеи.

Именно поэтому canonical data model должен быть organism-first.

### 2.2 Генетический код

Генетический код в твоем ТЗ должен быть:

- достаточно богатым, чтобы по нему можно было восстановить реализацию в fixed template;
- представленным как набор генов, а не как одно короткое summary;
- достаточно содержательным, чтобы LLM могла думать не только о коде, но и об идее, мотивации и происхождении метода;
- mutable через crossover и mutation;
- canonical artifact на диске.

Практически это означает:

- short trait list недостаточен как final design;
- terse labels без rationale недостаточны;
- итоговый child genetic code должен быть тем, что вернула LLM, а не тем, что silently собрал pre-LLM operator;
- gene pool и final DNA не должны путаться.

### 2.3 Lineage

Lineage в твоем ТЗ:

- материнская каноническая линия;
- хранит историю изменений идеи;
- хранит ударную фразу про текущее изменение;
- после оценки дообогащается score;
- должна быть полезной и как machine artifact, и как prompt context.

Это не просто ancestry list.

Это research record.

### 2.4 Fixed template

Template должен быть фиксирован.

LLM не должна менять контракт.

Внутри фиксированного контракта она может делать что угодно.

Это значит:

- `build_optimizer(model, max_steps)` фиксирован;
- `step(weights, grads, activations, step_fn)` фиксирован;
- `zero_grad(set_to_none=True)` фиксирован;
- editable surface ограничена template parser / sections model.

### 2.5 Mutation и crossover

Mutation:

- parent sampling из равномерного распределения;
- удаление части генов по вероятности из конфига;
- затем LLM осмысленно восстанавливает coherent child;
- mutation не должна превращаться в “взяли родителя и чуть-чуть переписали код, не обновив DNA”.

Crossover:

- родители выбираются по softmax над reward;
- мать и отец различаются;
- у child есть материнская каноническая lineage;
- child получает inherited gene pool через maternal bias `p`;
- затем LLM делает final coherent child genetic code и code implementation.

### 2.6 Islands

Island в твоем ТЗ:

- отдельная школа/направление;
- задается `.txt` файлом в `conf`;
- используется при seed initialization;
- изолирует локальный генофонд;
- selection делается per-island;
- inter-island crossover разрешен, но отдельно управляется;
- island identity должна сохраняться в organism metadata.

### 2.7 Простая фаза

Simple phase должна:

- запускать простые эксперименты;
- использовать Neyman allocation;
- давать каждой новой особи `simple_reward`;
- оставлять этот reward с особью навсегда;
- после simple evaluation отбирать `top-k` per island.

### 2.8 Великий фильтр

Great Filter должен:

- запускаться на hard experiments;
- быть отдельной более строгой фазой;
- не стирать смысл `simple_reward`;
- выбирать репродуктивно значимых survivors;
- оставлять `top-h` per island.

### 2.9 Reward semantics

По твоему текущему ТЗ reward должен быть baseline-relative и опираться на:

- качество в терминах train/val objective;
- число шагов до достижения нужного loss threshold;
- F1-like aggregation между quality ratio и steps ratio.

Это важно:

- reward pipeline должен соответствовать формальной математике, а не только “приблизительно выглядеть похожим”;
- если вводится Neyman allocation, итоговая aggregation semantics должна сохранять нужное ожидание;
- если hard phase использует те же score primitives, она все равно должна быть phase-separate по смыслу.

## 3. Что реально улучшилось по сравнению с прошлым аудитом

Это очень важный раздел.

Новый аудит не должен делать вид, что в репозитории ничего не изменилось.

Изменилось много.

### 3.1 Что улучшилось структурно

Список реально полезных улучшений:

- prompt assets действительно вынесены в `conf/prompts`;
- island descriptions действительно появились в `conf/islands`;
- появился loader для islands;
- появился `PromptBundle` и config-driven prompt loading;
- появился `population_manifest.json`;
- `EvolutionLoop` стал реально organism-first и island-aware;
- `OrganismMeta` теперь хранит `island_id`, `simple_reward`, `hard_reward`, `selection_reward`;
- появился `LineageEntry` с phase-specific score fields;
- lineage теперь дообогащается после evaluation;
- organism artifacts стали полноценнее;
- `genetic_code.md` стал canonical storage artifact;
- `read_genetic_code` и `read_lineage` умеют читать новые артефакты;
- `README` заметно ближе к целевой системе;
- появились новые тесты на resume/islands/prompts;
- optimizer contract в runtime и validator действительно ужесточен.

### 3.2 Что улучшилось в data model

Самые заметные прогрессы:

- `OrganismMeta` больше не выглядит как тонкая обертка над legacy candidate;
- separation `simple_reward / hard_reward / selection_reward` появилась;
- mother/father ids присутствуют;
- generation semantics стали богаче;
- genetic code и lineage теперь лежат в organism data model явно.

### 3.3 Что улучшилось в selection/topology

Положительные изменения:

- per-island selection реально есть;
- `select_top_k_per_island` и `select_top_h_per_island` существуют;
- evolution loop действительно seed’ит популяцию по островам;
- softmax parent selection и uniform mutation sampling тоже появились в runtime;
- inter-island crossover как отдельная идея появился в коде, а не только в документации.

### 3.4 Что улучшилось в reward pipeline

Прогресс:

- baseline profiles действительно существуют как canonical dependency;
- `train_loss` теперь является центральной objective semantics;
- `first_step_at_or_below_baseline` используется;
- quality ratio и steps ratio действительно агрегируются через F1-like formula;
- baseline-relative scoring стал реальной частью evaluator path.

### 3.5 Что улучшилось в docs

Сдвиг в правильную сторону:

- README больше не описывает старый optimizer contract;
- в документации уже сказано про organism-first и island-aware evolution;
- в `agents/7.md` есть уже execution-spec;
- репозиторий на уровне narrative значительно ближе к твоему ТЗ, чем раньше.

### 3.6 Почему это все равно не закрывает migration

Потому что текущие улучшения не убрали следующие классы проблем:

- duplicate paths;
- compatibility layers that still influence runtime;
- config fields that are declared but ignored;
- tests that still validate legacy behavior more aggressively than canonical behavior;
- math/selection bugs that появились уже после рефакторинга;
- documentation claims that are slightly ahead of runtime closure.

Именно это и является главным содержанием нового re-audit.

## 4. Repo map текущего состояния

Ниже не “список файлов”, а смысловая карта того, как устроен проект сейчас.

### 4.1 Canonical-looking path

То, что сейчас похоже на целевую архитектуру:

- `conf/prompts/*`
- `conf/islands/*`
- `src/evolve/islands.py`
- `src/evolve/prompt_utils.py`
- `src/evolve/evolution_loop.py`
- `src/organisms/organism.py`
- `src/organisms/mutation.py`
- `src/organisms/crossbreeding.py`
- `src/evolve/storage.py` в части organism artifacts и manifest
- `tests/test_evolution_resume.py`
- `tests/test_prompt_bundle.py`
- `tests/test_islands.py`

### 4.2 Legacy path, который все еще жив

То, что не является canonical target state, но все еще operational:

- candidate-based generation inside `src/evolve/orchestrator.py`
- `cand_*` directory model
- `load_best_context()` over candidate summaries
- `src/evolve/prompts/optimizer_system.txt`
- `src/evolve/prompts/optimizer_user.txt`
- legacy selection helpers `tournament_select`, `elite_select`, `select_parents_for_reproduction`
- fallback logic to old config keys and old artifacts

### 4.3 Transitional path

То, что сейчас является bridge layer:

- `PromptBundle` with fallback to legacy prompt files;
- storage readers that can load new and old organism artifacts;
- `load_population()` kept as compatibility helper;
- `run.py` still capable of legacy single-generation mode;
- `generator.py`, который обслуживает и candidate, и organism generation.

### 4.4 Почему это важно

Пока проект остается в таком mixed state:

- невозможно просто смотреть на README и думать, что все уже canonical;
- passing tests тоже не гарантируют final semantic closure;
- новые баги легко возникают не в чистой canonical logic, а на стыке canonical и transitional layers.

## 5. Что уже сделано хорошо и что я бы точно сохранял

Чтобы отчет был честным, нужно зафиксировать не только проблемы.

Ниже вещи, которые выглядят как правильное направление и которые не стоит ломать без причины.

### 5.1 `conf/prompts` как новое canonical место

Это правильное направление.

Почему:

- prompt assets перестают быть спрятанными рядом с кодом;
- конфиг начинает управлять текстовыми активами;
- это согласуется с твоим ТЗ;
- это готовит основу для versioned prompt evolution.

### 5.2 `conf/islands` как research-school layer

Это тоже правильный ход.

Почему:

- острова наконец перестали быть только абстракцией в документации;
- school descriptions отделены от code logic;
- seed generation наконец может учитывать island identity.

### 5.3 `population_manifest.json`

Это очень правильное решение.

Почему:

- active population больше не должна выводиться из “сканирования текущего поколения”;
- survive-from-older-generation case реально поддерживается;
- resume semantics стали ближе к правильной evolutionary model.

### 5.4 Разделение `simple_reward / hard_reward / selection_reward`

Это один из самых полезных недавних сдвигов.

Даже при текущих caveats это уже намного лучше старой one-scalar ambiguity.

### 5.5 `genetic_code.md` и `lineage.json`

Это правильная materialization strategy.

Почему:

- идея и lineage становятся явными artifacts;
- организм перестает быть только optimizer file;
- future tooling around inspection/analysis становится легче.

### 5.6 Stricter optimizer contract

Это тоже однозначно нужно сохранять:

- `build_optimizer(model, max_steps)`
- `step(..., step_fn)`
- `zero_grad(set_to_none=True)`

Текущий runtime/validator hardening здесь правильный.

### 5.7 `train_loss`-centered baseline-relative scoring

Это соответствует твоему текущему ТЗ лучше, чем старая static normalization модель.

Да, aggregation still needs audit.

Но направление верное.

## 6. Mismatch ledger: prompts и текстовые артефакты

### 6.1 Finding: `conf/prompts` уже не единственный source of truth

Ожидание:

- prompts должны canonical храниться в `conf`;
- runtime не должен зависеть от старых prompt assets в `src/evolve/prompts`.

Что сейчас сделано:

- organism operators действительно используют `PromptBundle`, который читает `conf/prompts`;
- но `OptimizerGenerator` все еще читает `optimizer_system.txt` и `optimizer_user.txt` из `src/evolve/prompts`.

Где доказательство:

- `src/evolve/prompt_utils.py::load_prompt_bundle`
- `src/evolve/generator.py::__init__`

Почему это проблема:

- в репозитории теперь фактически две prompt systems;
- canonical organism-first path уже перешел на `conf/prompts`;
- candidate path все еще держит старый prompt source;
- итоговая правда о prompt assets оказывается размытой.

Что делать:

- либо полностью изолировать candidate path как explicit legacy mode;
- либо тоже перевести candidate prompt assets в config-driven bundle;
- либо объявить candidate path non-canonical и не позволять ему влиять на main evolution system.

Приоритет:

- высокий.

### 6.2 Finding: `PromptBundle` использует legacy fallback, что полезно для migration, но размывает closure

Ожидание:

- canonical path должен читать prompts из `conf`;
- legacy fallback должен быть временным и контролируемым.

Что сейчас сделано:

- `PromptBundle` сначала пробует config/`conf/prompts`;
- потом может упасть назад на legacy files for several keys.

Где доказательство:

- `src/evolve/prompt_utils.py::_resolve_prompt_path`

Почему это одновременно хорошо и плохо:

- хорошо, потому что migration не рвется резко;
- плохо, потому что repo все еще operationally tolerates old prompt source;
- это затрудняет момент, когда можно сказать “migration complete”.

Что делать:

- оставить fallback только как short-lived migration bridge;
- добавить явный deprecation note;
- в следующем этапе либо удалить fallback, либо ограничить его only explicit legacy mode.

Приоритет:

- средний.

### 6.3 Finding: prompt relocation сделана неполностью по semantic closure

Ожидание:

- после migration implementer и future agents должны смотреть на `conf/prompts` и считать вопрос закрытым.

Что сейчас происходит:

- для organism-first path это почти так;
- для generator/candidate path это не так.

Почему это важно:

- документация уже рассказывает историю как будто relocation завершена;
- код рассказывает историю “almost, but not quite”.

Риск:

- новые изменения могут случайно править не тот prompt layer;
- future agent может исправить `conf/prompts`, а active legacy path останется прежним.

Что делать:

- в новом roadmap explicitly разделить:
  - canonical organism prompts;
  - legacy candidate prompts;
  - migration plan for removal or quarantine.

Приоритет:

- высокий.

### 6.4 Finding: `system_project_context.txt` выглядит полезно и содержательно

Это не баг, а положительная фиксация.

Что хорошо:

- prompt реально отражает island-based program;
- contract fixed;
- reward semantics and lineage mentioned;
- gene richness explicitly requested;
- budget-awareness mentioned.

Почему важно зафиксировать:

- этот файл уже можно считать хорошей опорной точкой;
- при дальнейшей доработке его надо усиливать, а не перепридумывать с нуля.

Приоритет:

- сохранить.

### 6.5 Finding: seed/mutation/crossover prompt system texts уже соответствуют target mode лучше старых

Что улучшилось:

- structured section contract есть;
- `CORE_GENES`/`INTERACTION_NOTES`/`COMPUTE_NOTES` explicit;
- crossover explicitly просит maternal bias;
- mutation explicitly говорит, что returned DNA canonical.

Почему это хорошо:

- prompts наконец начали говорить на языке твоего ТЗ;
- это снижает drift между research concept и codegen layer.

Но caveat:

- наличие хороших новых prompts не отменяет existence старых generator prompts.

### 6.6 Finding: нет отдельной проверки богатства генетического кода на runtime level

Ожидание:

- если genetic code является canonical artifact, runtime должен уметь отвергать совсем слабые ответы;
- слишком бедный `CORE_GENES` не должен silently считаться нормой.

Что сейчас:

- prompts просят богатые гены;
- parser и builder почти ничего не валидируют по richness;
- пустой или бедный output может пройти в очень мягком режиме.

Где доказательство:

- `src/evolve/template_parser.py::parse_llm_response`
- `src/organisms/organism.py::_parse_core_genes`
- `src/organisms/organism.py::_build_genetic_code`

Почему это важно:

- canonical artifact quality сейчас largely зависит от добросовестности LLM;
- system itself не enforce’ит, что gene code действительно rich enough.

Что делать:

- ввести validation layer:
  - minimum number of genes;
  - non-empty notes;
  - no purely one-word gene list;
  - rejection or retry on malformed/too-thin DNA.

Приоритет:

- высокий.

### 6.7 Finding: old `optimizer_system.txt` / `optimizer_user.txt` now clearly represent second generation mode

Семантически это не просто “legacy files somewhere”.

Это альтернативный generation regime:

- return raw Python code;
- context from previous candidates;
- no organism DNA artifact contract;
- no lineage semantics.

Почему это критично:

- existence of these files means repository still embodies two different creative paradigms:
  - candidate-as-code;
  - organism-as-research-artifact.

Что делать:

- перестать относиться к ним как к harmless leftovers;
- в новом `8.md` их надо считать residual architecture branch.

Приоритет:

- высокий.

### 6.8 Finding: prompt tests подтверждают загрузку, но не подтверждают closure

Что покрыто:

- default conf assets load;
- explicit prompt paths load;
- compose_system_prompt works.

Что не покрыто:

- no test that canonical evolve path no longer depends on `src/evolve/prompts`;
- no test that missing configured prompt fails as desired in production mode;
- no test that fallback behavior is explicitly deprecated.

Вывод:

- tests for prompt loading are useful;
- tests for prompt ownership are missing.

Приоритет:

- средний.

## 7. Mismatch ledger: islands и population topology

### 7.1 Finding: islands реально появились и участвуют в runtime

Это большой положительный сдвиг.

Доказательства:

- `conf/islands/*.txt`
- `src/evolve/islands.py`
- `src/evolve/evolution_loop.py::_seed_initial_population`
- `src/evolve/evolution_loop.py::_group_by_island`
- `tests/test_islands.py`

Почему это важно:

- это уже не paper-only idea;
- island identity дошла до code level.

Статус:

- фиксировано с caveats.

### 7.2 Finding: inter-island crossover probability реализована с вероятностным багом

Ожидание:

- `inter_island_crossover_rate = r` должна означать реальную вероятность межостровного скрещивания порядка `r`, когда local и foreign pools доступны.

Что сейчас:

- `_select_father_pool` делает один random draw для decision “остаемся локально?”;
- потом второй независимый random draw для decision “идем в foreign pool?”;
- в typical case реальная вероятность foreign father becomes approximately `r^2`, а не `r`.

Где доказательство:

- `src/evolve/evolution_loop.py::_select_father_pool`

Почему это серьезно:

- при default `r = 0.1` фактическая cross-island probability примерно `0.01`, а не `0.1`;
- research topology становится намного более изолированной, чем думает config;
- user-level intuition про island exchange оказывается ложной.

Что делать:

- заменить double-random logic на one-shot decision:
  - sample one Bernoulli for cross-island intent;
  - if success and foreign pools exist -> choose foreign;
  - else local.

Приоритет:

- P0.

### 7.3 Finding: islands пока поддерживают только равный population size

Ожидание:

- user explicitly допускает конфигурирование числа особей каждой школы;
- default может быть equal population;
- но canonical model должна хотя бы оставлять понятный путь к unequal sizes.

Что сейчас:

- в `default.yaml` есть `organisms_per_island`;
- per-island override отсутствует.

Почему это не блокер, но gap:

- текущая реализация соответствует reasonable default;
- но полная реализация user ТЗ про “сколько особей каждой школы” пока не достигнута.

Что делать:

- либо явно задокументировать equal-per-island as current limitation;
- либо добавить optional per-island override map.

Приоритет:

- средний.

### 7.4 Finding: island seed generation в целом соответствует intended behavior

Что хорошо:

- `SeedOperator` принимает `Island`;
- seed prompt получает `island_id`, `island_name`, `island_description`;
- initial population seed’ится по островам.

Где доказательство:

- `src/evolve/operators.py::SeedOperator`
- `src/evolve/evolution_loop.py::_seed_island`

Почему это важно:

- один из самых больших structural gaps из старого аудита реально закрыт.

Статус:

- good and keep.

### 7.5 Finding: island identity корректно сохраняется в organism metadata

Что сейчас:

- `OrganismMeta.island_id` есть;
- directory layout содержит `island_<id>`;
- manifest stores `island_id`;
- selection groups by island.

Почему это хорошо:

- island model не просто decorative;
- topology materially enters storage and selection semantics.

### 7.6 Finding: migration support for pre-island organisms есть, но это временный слой

Что сделано:

- `legacy_flat_population_island()` существует;
- old artifacts can be coerced into island-aware model.

Почему это полезно:

- migration safer;
- старые population artifacts не instantly dead.

Почему это нужно держать под контролем:

- если compatibility island останется слишком надолго, она превратится в скрытый третий режим работы;
- canonical topology again размоется.

Приоритет:

- средний.

### 7.7 Finding: lineage пока не подчеркивает cross-island event как отдельный semantic fact

Ожидание:

- inter-island crossover исследовательски важен;
- lineage or summary ideally должен позволять видеть, когда отец пришел с другого острова.

Что сейчас:

- father id записывается;
- island id child inherited from mother;
- explicit cross-island annotation в lineage entry нет.

Почему это не критический bug, но потеря signal:

- для research analytics потом трудно будет отслеживать, как inter-island exchange влияло на score.

Что делать:

- добавить в lineage entry или summary derived field:
  - `cross_island: bool`
  - `father_island_id`

Приоритет:

- средний.

### 7.8 Finding: tests on islands проверяют loader, но не проверяют full island lifecycle

Что покрыто:

- multiple files load;
- empty file rejected.

Что не покрыто:

- no end-to-end test that seeding by islands works;
- no test that cross-island reproduction preserves maternal island;
- no test that inter-island probability matches config;
- no test that top-k/top-h respect islands over multiple generations.

Приоритет:

- высокий.

## 8. Mismatch ledger: organism model, genetic code и lineage

### 8.1 Finding: `OrganismMeta` стал намного ближе к target data model

Это фиксируем как положительный прогресс.

Почему:

- есть `mother_id`, `father_id`;
- есть `genetic_code_path`, `lineage_path`;
- есть три reward fields;
- есть `status`;
- есть `genetic_code` and `lineage` in-memory payloads.

Это уже не “бедный candidate wrapper”.

### 8.2 Finding: `LineageEntry` все еще хранит ambiguous `aggregate_score`

Ожидание:

- lineage должен уметь сохранять both simple and hard semantics, не уничтожая ясность.

Что сейчас:

- entry stores `simple_score`, `hard_score`, `aggregate_score`;
- after simple phase `aggregate_score` can equal simple score;
- after hard phase `aggregate_score` becomes hard score.

Почему это неоднозначно:

- слово `aggregate_score` не объясняет, что именно агрегировано;
- при наличии separate phase scores оно превращается в weak alias;
- prompt summary prints all three, но interpretive semantics остается расплывчатой.

Что делать:

- либо переименовать `aggregate_score` в `current_selection_score`;
- либо четко определить его как derived field and document that;
- либо вообще убрать его из canonical lineage and keep only phase scores.

Приоритет:

- средний.

### 8.3 Finding: lineage backfill теперь есть, и это реальное улучшение

Что сейчас:

- `update_latest_lineage_entry()` существует;
- simple phase updates selected simple experiments and simple score;
- hard phase updates selected hard experiments and hard score;
- lineages persist back to disk.

Почему это хорошо:

- одна из самых болезненных проблем старого аудита закрыта;
- lineage больше не остается endless `score=None`.

Статус:

- fixed.

### 8.4 Finding: `format_lineage_summary` все еще слишком utilitarian для LLM reasoning

Ожидание:

- lineage summary in prompt должен передавать impact history так, чтобы LLM видела смысл эволюции идеи.

Что сейчас:

- summary is flat log of last 5 entries;
- fields dumped as `gen=... | op=... | change=... | genes=... | simple=... | hard=... | aggregate=...`.

Почему это недостаточно:

- это machine-readable-ish dump, а не сильный research narrative;
- missing causal emphasis;
- no explicit pattern summary like “when we added X, score improved / worsened”.

Что делать:

- оставить current machine summary as raw mode if needed;
- но в prompt rendering добавить more interpretive short history:
  - strongest improvements;
  - strongest regressions;
  - latest maternal trajectory.

Приоритет:

- средний.

### 8.5 Finding: genetic code canonicalized as file, но богатство не enforce’ится

Ожидание:

- `genetic_code.md` должен быть rich enough by system design, not just by prompt wish.

Что сейчас:

- `CORE_GENES`, `INTERACTION_NOTES`, `COMPUTE_NOTES` are persisted;
- builder accepts very minimal content;
- missing notes are not fatal;
- no minimal richness validation.

Где доказательство:

- `src/evolve/storage.py::_render_genetic_code`
- `src/evolve/storage.py::read_genetic_code`
- `src/organisms/organism.py::_build_genetic_code`

Почему это важно:

- one of the main scientific assumptions of the platform is that gene text encodes the method idea;
- current runtime does not verify that artifact quality actually reaches this bar.

Приоритет:

- высокий.

### 8.6 Finding: `build_organism_from_response` все еще принимает legacy `IDEA_DNA` fallback

Ожидание:

- canonical prompt contract now prefers `CORE_GENES`;
- legacy compatibility may exist temporarily, but should be explicit.

Что сейчас:

- `_parse_core_genes()` falls back to parsing `IDEA_DNA` if `CORE_GENES` absent.

Почему это двояко:

- good for migration and old mock responses;
- bad for closure, because malformed or outdated response format can still silently pass.

Что делать:

- for canonical organism generation, treat `IDEA_DNA` fallback as deprecated;
- optionally permit only in explicit legacy compatibility mode.

Приоритет:

- средний.

### 8.7 Finding: malformed LLM structured response can still degrade quietly

Ожидание:

- structured response parser should reject materially incomplete artifacts.

Что сейчас:

- parser only splits sections;
- builder supplies defaults for code sections;
- partial response may still create syntactically valid but conceptually weak organism.

Почему это риск:

- low-quality LLM answer may not trigger retry;
- evolution may accept organisms whose scientific artifact is poor but code compiles.

Что делать:

- add explicit required-sections validation:
  - `CORE_GENES`
  - `CHANGE_DESCRIPTION`
  - `IMPORTS`
  - `INIT_BODY`
  - `STEP_BODY`
  - `ZERO_GRAD_BODY`
- add retry trigger on missing key sections.

Приоритет:

- высокий.

### 8.8 Finding: diff summary in mutation/crossover is still generic, not actual semantic gene diff

Ожидание:

- lineage should preserve what concretely changed.

Что сейчас:

- mutation writes summary like “kept N genes and removed M genes”;
- crossover writes summary like “inherited maternal-biased pool of N genes”.

Почему это слабее, чем нужно:

- these are operator-level statistics, not actual scientific differences;
- future prompt context does not learn enough from that text alone.

Что делать:

- build diff summary from actual old vs new genes:
  - added genes;
  - dropped genes;
  - rewritten themes;
- keep short, but concept-specific.

Приоритет:

- средний.

### 8.9 Finding: mother lineage inheritance is now implemented correctly

Что сейчас:

- mutation child gets `parent_lineage=parent.lineage`;
- crossover child gets `parent_lineage=mother.lineage`.

Почему это хорошо:

- one of your core lineage invariants is finally respected;
- maternal continuity no longer exists only in docs.

Статус:

- fixed.

### 8.10 Finding: canonical `src/organisms/*` operators still lack direct test coverage

Что сейчас:

- `tests/test_crossbreeding.py` tests only pure DNA recombination helper;
- `tests/test_mutation_dna.py` tests only DNA deletion helper;
- `tests/test_genetic_operators.py` covers legacy operators in `src/evolve/operators.py`, not canonical `src/organisms/mutation.py` and `src/organisms/crossbreeding.py`.

Почему это плохо:

- the canonical path that matters most is under-tested;
- passing tests over legacy operator layer can create false confidence.

Приоритет:

- P0/P1 boundary.

## 9. Mismatch ledger: storage, manifest и migration

### 9.1 Finding: manifest-based resume is real and valuable

Это один из сильнейших recent fixes.

Что сейчас:

- `population_manifest.json` exists;
- evolution state tracks current generation;
- resume restores active population from manifest;
- old-generation survivors can remain active.

Подтверждение:

- `src/evolve/evolution_loop.py::_restore_population_from_manifest`
- `tests/test_evolution_resume.py`

Статус:

- fixed with caveats.

### 9.2 Finding: manifest validation is still shallow

Ожидание:

- canonical manifest should be treated as critical control-plane artifact;
- invalid entries should be caught early.

Что сейчас:

- `read_population_manifest()` only validates that payload is dict and `active_organisms` is list;
- per-entry schema validation is thin.

Почему это риск:

- duplicate organism ids;
- duplicate dirs;
- missing fields;
- invalid generation numbers;
- malformed island ids.

Все это может пройти слишком далеко before failure.

Что делать:

- add explicit manifest entry validation;
- optionally introduce dataclass/schema loader.

Приоритет:

- средний.

### 9.3 Finding: storage layer still exposes too many legacy candidate helpers as first-class citizens

Ожидание:

- candidate storage can exist for legacy mode, but should be clearly quarantined.

Что сейчас:

- `candidate_dir`, `result_path`, `selection_path`, `summary_path`, `load_best_context`, legacy resume helpers and organism helpers coexist in the same module.

Почему это проблема:

- storage module still encodes dual architecture;
- future implementer can accidentally reuse wrong helper;
- organism-first migration remains semantically incomplete.

Что делать:

- either split storage into `candidate_storage` and `organism_storage`;
- or mark candidate helpers explicitly as legacy and keep organism helpers at top-level.

Приоритет:

- высокий.

### 9.4 Finding: `load_recent_experiment_scores()` mixes candidate and organism histories

Ожидание:

- once organism-first path becomes canonical, Neyman allocation history for canonical evolution should be based on canonical comparable artifacts;
- mixing legacy candidate summaries may bias allocation history.

Что сейчас:

- `load_recent_experiment_scores()` scans both:
  - `gen_*/cand_*/summary.json`
  - `gen_*/island_*/org_*/summary.json`

Почему это риск:

- candidate path may have different semantics, prompt regime, or generation regime;
- allocation history for organism evolution becomes polluted by legacy runs;
- research signal becomes harder to interpret.

Что делать:

- either separate candidate and organism histories;
- or use organism-only history in organism-first evolution;
- or add explicit config knob documenting mixed history if truly intended.

Приоритет:

- P0/P1.

### 9.5 Finding: legacy artifact fallbacks are useful, but now need a sunset plan

Что сейчас:

- `read_genetic_code()` falls back to `idea_dna.txt`;
- `read_lineage()` falls back to `evolution_log.json`;
- `resolve_generation_dir()` supports old `gen_N`;
- `read_organism_meta()` coerces legacy payloads.

Почему это нормально на migration stage:

- you do not want to lose old work.

Почему это становится проблемой later:

- if all fallbacks stay indefinitely, repo never reaches one stable canonical shape;
- bugs become harder to localize.

Что делать:

- declare sunset milestone for each fallback;
- add warnings or metrics that show when fallback is used;
- after migration cutoff, reject silent fallback in normal mode.

Приоритет:

- средний.

### 9.6 Finding: `load_population()` still exists as compatibility path

Это не bug само по себе.

Но это important architecture note.

Потому что:

- canonical restore should not depend on generation scan;
- `load_population()` now exists as fallback if manifest missing;
- this is fine short-term;
- but it must not again become main restore mechanism.

Статус:

- acceptable as temporary fallback;
- should stay clearly labeled compatibility helper.

### 9.7 Finding: manifest path stores active organisms, but no higher-level integrity checks

Что missing:

- validation that active population size matches expected island totals;
- validation that no eliminated organism remains in active set;
- validation that island balance is coherent.

Почему важно:

- with island-aware evolution, control-plane integrity matters more than before.

Что делать:

- on resume:
  - validate duplicates;
  - validate island ids exist or resolve legacy island;
  - validate paths exist;
  - warn if counts differ from config.

Приоритет:

- средний.

### 9.8 Finding: organism summary format improved, but allocation history consumers still need clarity

Что сейчас хорошо:

- `summary.json` stores `phase_results.simple` and `phase_results.hard`;
- `simple_reward`, `hard_reward`, `selection_reward` exposed.

Что still needs clarity:

- some readers still look at `experiments` top-level legacy candidate format;
- some history logic reads both legacy and organism summary forms.

Итог:

- summary migration progressed;
- consumer migration incomplete.

### 9.9 Finding: `load_best_organism_context()` exists, but candidate path still uses `load_best_context()`

Ожидание:

- canonical generation should condition on best organism context, not best candidate context.

Что сейчас:

- organism summary context loader exists;
- candidate generation path still uses `load_best_context()`.

Почему это важно:

- repo is still architecturally split in two conceptually different self-improvement loops.

Приоритет:

- высокий.

### 9.10 Finding: storage/tests do not yet prove corrupted-manifest resilience

Что не проверено:

- missing active organism path;
- duplicate manifest entries;
- stale relative paths;
- invalid island id;
- generation mismatch + empty manifest.

Приоритет:

- средний.

## 10. Mismatch ledger: evolution runtime и selection semantics

### 10.1 Finding: organism-first multi-generation loop now exists for real

Это большой зафиксированный прогресс.

Подтверждение:

- `src/evolve/evolution_loop.py`
- per-island seeding;
- reproduction loop;
- simple phase evaluation;
- optional Great Filter;
- manifest-based state.

Статус:

- fixed as architecture skeleton.

### 10.2 Finding: mutation-vs-crossover split is not actually controlled by current canonical config

Ожидание:

- operator probabilities should be configured from canonical config.

Что сейчас:

- `EvolutionLoop._mutation_probability()` looks for:
  - `evolver.operators.mutation.probability`
  - legacy `evolver.evolution.mutation_rate`
  - else default `0.5`
- but `conf/evolver/default.yaml` does not define `operators.mutation.probability`.

Почему это важно:

- current default config makes mutation/crossover mix silently hardcoded;
- user may think `operators.*` block fully controls operator behavior;
- actually one critical knob is absent.

Что делать:

- add explicit canonical knob, for example `operators.mutation.probability`;
- or explicit `operators.crossover.probability`;
- or config for offspring plan composition.

Приоритет:

- P0.

### 10.3 Finding: `parent_sampling` keys are declared in config but ignored in runtime

Ожидание:

- if config declares `operators.mutation.parent_sampling=uniform` and `operators.crossover.parent_sampling=softmax_reward`, runtime should read and obey them;
- or those keys should not exist.

Что сейчас:

- runtime hardcodes:
  - mutation parent = `uniform_select_organisms()`
  - crossover parents = `softmax_select_organisms()`
- config keys `parent_sampling` are not consulted.

Где доказательство:

- `conf/evolver/default.yaml`
- `src/evolve/evolution_loop.py::_reproduce_for_island`

Почему это плохо:

- config lies;
- reader thinks a behavior is configurable when it is not.

Что делать:

- either wire config keys into runtime;
- or remove them from canonical config and docs.

Приоритет:

- высокий.

### 10.4 Finding: father selection bug already discussed is selection-semantics critical

Повторю отдельно потому что это selection-critical P0.

Проблема:

- actual inter-island crossover rate is lower than declared.

Почему это особенно опасно:

- this is not cosmetic;
- it changes exploration topology of the whole search process.

### 10.5 Finding: `selection_reward` is better than old `score`, but still semantically overloaded

Ожидание:

- `selection_reward` should clearly mean “the scalar currently used for parent sampling”.

Что сейчас:

- it mostly does;
- but `OrganismMeta.score` property aliases to `selection_reward`;
- `best organism` logic looks at `selection_reward`;
- after hard phase, `selection_reward` becomes hard score;
- in non-hard generations, it gets reset to simple score.

Почему это acceptable but still muddy:

- semantics depend on phase timing;
- `score` alias keeps old ambiguity alive;
- downstream code and humans can still confuse historical quality with current reproductive criterion.

Что делать:

- keep `selection_reward` as explicit field;
- remove or de-emphasize generic `score` alias where possible;
- document current-phase meaning very explicitly.

Приоритет:

- средний.

### 10.6 Finding: Great Filter selection is phase-separated, but not fully hard by runtime guarantees

Ожидание:

- Great Filter should be meaningfully harder than simple phase.

Что сейчас:

- great filter uses different experiment list;
- but task `mode` for evaluation is still global `evolver.eval_mode`, defaulting to `smoke`.

Почему это mismatch:

- hard experiments can still be run in smoke mode;
- Great Filter becomes “different experiments” but not necessarily “harder evaluation regime”.

Где доказательство:

- `conf/evolver/default.yaml: eval_mode=smoke`
- `src/evolve/orchestrator.py::_build_organism_task`
- `src/evolve/evolution_loop.py::_evaluate_phase`

Что делать:

- add phase-specific eval mode or budget profile;
- or rename semantics if “Great Filter” is not actually computationally stricter.

Приоритет:

- P0/P1.

### 10.7 Finding: `run.py` still keeps legacy single-generation orchestrator as automatic fallback

Ожидание:

- canonical entrypoint should clearly pick canonical architecture.

Что сейчас:

- `run_evolution()` falls back to single-generation mode if no multigen config present.

Почему это not ideal:

- repo still has two top-level evolution stories;
- accidental config shape can send user into legacy mode.

Что делать:

- either keep explicit `run_single_generation()` as legacy command only;
- or make fallback more explicit and less automatic.

Приоритет:

- средний.

### 10.8 Finding: per-island top-k and top-h runtime exists and looks correct

Это хороший progress note.

Что сейчас:

- newborns/simple survivors selected per island;
- hard survivors selected per island;
- eliminated organisms are marked.

Статус:

- mostly fixed.

### 10.9 Finding: selection helper module still centers legacy behavior in test surface

Что сейчас:

- `selection.py` still exports tournament and elite helpers;
- tests cover them heavily;
- canonical path uses uniform/softmax/per-island helpers.

Почему это risk:

- test attention is still partly pointed at legacy behavior;
- maintenance priority may remain distorted.

Что делать:

- keep legacy tests only if legacy mode truly supported;
- otherwise isolate legacy helpers and stop treating them as central.

Приоритет:

- средний.

### 10.10 Finding: no runtime validation that `top_h_per_island <= top_k_per_island`

Ожидание:

- Great Filter should not accidentally ask for more survivors than simple phase allowed.

Что сейчас:

- config values are read;
- no explicit consistency validation found.

Почему это matters:

- misconfigured hard selection could produce semantically weird outcomes.

Что делать:

- validate config on startup.

Приоритет:

- средний.

### 10.11 Finding: no runtime use of declared `operators.crossover.parent_sampling`

Это стоит выделить отдельно от general config lie.

Потому что:

- softmax parent selection is a core scientific choice;
- if config advertises it, it should be auditable and switchable;
- otherwise it is a hidden constant.

Приоритет:

- высокий.

## 11. Mismatch ledger: evaluation, reward и Neyman allocation

### 11.1 Finding: baseline-relative F1-like experiment score реально реализован

Это важный положительный факт.

Что сейчас:

- `extract_metrics()` computes:
  - `quality_ratio`
  - `steps_ratio`
  - F1-like `exp_score`
- baseline loaded from `stats/<experiment>/baseline.json`;
- only `train_loss`, `min` direction accepted.

Это уже намного ближе к твоему описанию reward scheme.

Статус:

- fixed directionally.

### 11.2 Finding: math of subset aggregation under Neyman allocation likely не соответствует заявленному ожиданию

Ожидание:

- если используется subset of experiments from Neyman allocation, итоговый aggregate должен иметь корректную estimator semantics;
- твое ТЗ явно говорит, что матожидание должно совпадать с тем, как если бы мы все эксперименты прогнали.

Что сейчас:

- `mean_score()` aggregates selected experiments as weighted mean with weights `pi`;
- то есть essentially `sum(pi * score) / sum(pi)` over selected subset.

Почему это подозрительно:

- при sampling subset by probability `pi`, такой estimator generally не является unbiased estimate of uniform full-experiment mean;
- high-probability experiments get systematically favored twice:
  - once in sampling;
  - once in aggregation weight.

Где доказательство:

- `src/evolve/allocation.py`
- `src/evolve/scoring.py::mean_score`

Почему это важнейшая проблема:

- если цель allocation именно ускорять оценку без искажения expected mean, текущая aggregation likely ломает scientific meaning of reward;
- это не cosmetic bug, а statistical bug.

Что делать:

- отдельно formalize desired estimator;
- скорее всего нужен inverse-probability style correction or another explicit estimator;
- если biased weighted score intentional, это должно быть прямо документировано как новая semantics, а не Neyman approximation to full mean.

Приоритет:

- P0.

### 11.3 Finding: tests на allocation не доказывают правильность итогового estimator

Что тестируется:

- `pi` computation;
- fallback uniform;
- sampling without replacement;
- cost normalization.

Что не тестируется:

- unbiasedness or intended expectation of final aggregate;
- interaction of allocation + scoring together.

Вывод:

- allocation unit tests полезны;
- reward math at system level все еще практически не доказана тестами.

Приоритет:

- высокий.

### 11.4 Finding: `scoring_cfg` в runtime на самом деле является allocation cfg

Ожидание:

- scoring config and allocation config conceptually different.

Что сейчас:

- `mean_score(... scoring_cfg=request.allocation_cfg)` gets allocation cfg;
- `extract_metrics()` currently ignores `scoring_cfg` via `del scoring_cfg`.

Почему это smell:

- name implies scoring customization exists;
- runtime actually forwards allocation config into scoring function;
- future developer may think score math is configurable through wrong payload.

Что делать:

- either rename parameter;
- or create dedicated scoring config if needed;
- or remove unused parameter entirely.

Приоритет:

- средний.

### 11.5 Finding: `quality_ref` / `steps_ref` still live in experiment configs but are no longer part of canonical score

Ожидание:

- old static normalization fields should be removed or deprecated if no longer used.

Что сейчас:

- many experiment configs still contain `quality_ref` and `steps_ref`;
- tests like `test_evolve_integration_fake` still fabricate them;
- current scoring path ignores them.

Почему это problem:

- config lies again;
- docs and experiments carry semantic fossils from old scoring regime.

Что делать:

- either remove these fields from configs;
- or annotate as deprecated and ignored;
- update tests accordingly.

Приоритет:

- высокий.

### 11.6 Finding: simple vs hard phase separation implemented, but hard mode can still inherit simple eval regime

Already partly noted, but reward implications matter:

- hard experiments differ;
- but `eval_mode` remains global;
- if all phases run in `smoke`, Great Filter may not be a true hard validator.

Impact:

- `hard_reward` may be computed on not-really-hard runtime budgets;
- selection semantics can be overconfident.

Приоритет:

- высокий.

### 11.7 Finding: `simple_reward` persistence is mostly respected

Что сейчас:

- after simple phase organism gets `simple_reward`;
- later hard phase writes `hard_reward` and `selection_reward`;
- `simple_reward` itself is preserved.

Почему это good:

- one of your key invariants is respected better than before.

Но caveat:

- best-organism and selection logic use `selection_reward`, not persistent simple history;
- human interpretation still needs clarity.

Статус:

- mostly fixed.

### 11.8 Finding: missing baseline profiles currently hard-fail experiment scoring

Ожидание:

- system should have explicit policy for baseline absence.

Что сейчас:

- missing/invalid baseline marks experiment as failed.

Почему это maybe okay:

- baseline-relative score cannot be computed otherwise.

Но стоит explicit зафиксировать:

- this makes baseline availability a hard dependency for evolution;
- if one experiment lacks baseline, selected-evaluation subset may partial/fail.

Что делать:

- document baseline generation workflow as mandatory precondition;
- optionally add preflight baseline completeness check.

Приоритет:

- средний.

### 11.9 Finding: reward path currently enforces `train_loss` and `min`, which matches the present spec

Это good.

Почему стоит зафиксировать:

- there is now much less ambiguity about the optimization objective used for reward;
- this aligns runtime closer to your current reward theory.

### 11.10 Finding: no dedicated test that Great Filter leaves `simple_reward` intact while using hard-only results for selection

Это ключевой invariant.

Сейчас я не вижу явного теста, который доказывает:

- `simple_reward` survives hard phase untouched;
- `hard_reward` writes separately;
- `selection_reward` switches as intended;
- phase_results.simple and phase_results.hard both remain available.

Приоритет:

- высокий.

### 11.11 Finding: no dedicated test for `first_step_at_or_below_baseline` semantics edge cases

Краевые случаи:

- candidate never reaches baseline;
- candidate reaches baseline on first step;
- baseline last loss is extremely small;
- steps missing but objective present.

Current code:

- maps “never reached” to `steps_ratio = 0.0`.

Это может быть intentional.

Но test surface around it is thin.

Приоритет:

- средний.

### 11.12 Finding: evaluator path is materially better, but still dual-purpose

`EvolverOrchestrator` now:

- supports legacy candidate evaluation;
- supports public `evaluate_organisms()` seam.

Это удобно.

Но это also means:

- evaluator and legacy orchestrator logic are tightly co-located;
- long-term maintenance complexity stays high.

Приоритет:

- средний.

## 12. Mismatch ledger: public optimizer contract, template и docs

### 12.1 Finding: optimizer contract is mostly unified correctly

Это один из strongest completed items.

Подтверждение:

- `valopt/utils/import_utils.py`
- `src/evolve/template_parser.py`
- `README.md`
- example optimizer
- tests rejecting `build_optimizer(cfg)`

Статус:

- fixed.

### 12.2 Finding: `BuildOptimizerCallable` typing in `import_utils.py` is stale

Что сейчас:

- `BuildOptimizerCallable = Callable[[Any], OptimizerControllerProtocol]`

Почему это smell:

- actual builder contract is `(model, max_steps)`;
- this alias is outdated and misleading.

Это небольшой, но хороший показатель того, что migration cleanup еще не завершен полностью.

Приоритет:

- низкий.

### 12.3 Finding: template contract now rejects missing `step_fn`, and this is good

Это зафиксировать важно:

- hard rejection of old signature is now present in both import path and template validator;
- this closes a major previous mismatch.

Статус:

- keep.

### 12.4 Finding: README is much stronger, но местами чуть впереди runtime closure

Примеры:

- README говорит, что prompt assets live in `conf/prompts/` as canonical story;
- фактически organism path yes, candidate path still no.

- README говорит про canonical organism-first island-aware pipeline;
- фактически true as main architecture, but legacy run path still exists.

- README narrative suggests Great Filter style hard evaluation;
- current runtime keeps a global `eval_mode=smoke`.

Почему это важно:

- README mostly correct directionally;
- but some statements are “future-complete wording” rather than “strictly current runtime wording”.

Что делать:

- keep README near current truth;
- do not oversell finished migration.

Приоритет:

- средний.

### 12.5 Finding: candidate-mode integration test still encodes old config worldview

`tests/test_evolve_integration_fake.py` still uses:

- `selection_strategy`
- `eval_experiments`
- old normalization fields
- candidate directories

Почему это important:

- the test suite still treats old candidate orchestrator as a supported first-class path;
- this biases perceived product center of gravity away from organism-first.

Приоритет:

- высокий.

### 12.6 Finding: example optimizer and runtime docs are aligned

Это хороший знак.

`optimizer_guesses/examples/sgd_baseline.py` matches runtime contract.

Надо сохранить.

## 13. Mismatch ledger: tests и ложное чувство завершенности

### 13.1 Текущее тестовое состояние

Сейчас `pytest -q` проходит:

- `63 passed in 2.70s`

Это надо трактовать правильно.

Это значит:

- базовый quality floor вырос;
- runtime contract и часть migration реально покрыты;
- проект объективно здоровее, чем раньше.

Но это не значит:

- что migration complete;
- что all config knobs truthful;
- что reward math proved;
- что canonical path is the best-tested path.

### 13.2 Finding: new tests закрыли несколько старых дыр

Что реально стало лучше:

- есть test for manifest-driven resume from older generation survivors;
- есть prompt bundle tests;
- есть island loader tests;
- import contract tests strong;
- template contract tests strong;
- mutation/crossbreed pure helper tests exist.

Это надо честно засчитать как progress.

### 13.3 Finding: canonical organism operators are still under-tested relative to legacy operator layer

Повторю это на уровне test strategy:

- `src/organisms/mutation.py` and `src/organisms/crossbreeding.py` are important;
- but tests mostly cover either pure DNA helpers or legacy `src/evolve/operators.py`.

Это один из strongest test-gap findings.

### 13.4 Finding: no test currently catches inter-island probability bug

Очень важный practical point:

- current suite can stay green forever even if configured `inter_island_crossover_rate=0.1` acts like `0.01`.

Почему:

- there is no test that samples father-pool selection statistically or structurally.

### 13.5 Finding: no test proves that current config fields actually influence runtime

Examples:

- `operators.mutation.parent_sampling`
- `operators.crossover.parent_sampling`
- absence of `operators.mutation.probability`

Current suite:

- does not assert config truthfulness here.

### 13.6 Finding: no test proves Great Filter is truly phase-separated in runtime budgets

Missing:

- phase-specific eval mode/budget test;
- hard-vs-simple mode distinction test.

### 13.7 Finding: no test proves Neyman subset aggregation has intended expectation semantics

This is probably the most important mathematical gap in the entire test surface.

### 13.8 Finding: no test asserts organism-only history for canonical allocation

Missing invariant:

- canonical organism-first evolution should not silently depend on legacy candidate history unless explicitly intended.

### 13.9 Finding: no test asserts that canonical evolve path can run after removing legacy candidate prompt files

Why this matters:

- as long as such a test does not exist, repo may still have hidden dependency on residual prompt files.

### 13.10 Finding: no test checks manifest integrity edge cases

Missing:

- duplicate entries;
- broken path;
- stale island id;
- malformed entry payload.

### 13.11 Finding: no test explicitly proves `simple_reward` persistence across Great Filter

This should exist.

### 13.12 Finding: the suite still over-represents legacy candidate mode compared to target product

This is the most important meta-test finding.

Why:

- legacy mode has integration test;
- canonical organism-first path has more unit-style partial coverage, but less end-to-end coverage.

### 13.13 Test verdict

My updated verdict on test surface:

- stronger than before;
- not enough to certify semantic completion;
- still biased toward a mix of old and new architecture.

## 14. Delta-аудит относительно `6.md` и `7.md`

Ниже статусы по главным направлениям прошлого цикла.

### 14.1 Prompts moved into `conf`

Статус:

- `partially fixed`

Почему:

- new prompt bundle and conf assets added;
- old candidate prompt layer still active.

### 14.2 Islands introduced as real subsystem

Статус:

- `fixed, but with bugs/caveats`

Почему:

- islands exist and drive seeding/selection;
- inter-island probability bug remains;
- per-island size customization not yet rich.

### 14.3 Population manifest and semantically correct resume

Статус:

- `fixed, but validation still shallow`

Почему:

- manifest now drives active population restore;
- older generation survivors supported;
- integrity checks still minimal.

### 14.4 Lineage score backfill

Статус:

- `fixed`

Почему:

- simple/hard scores now backfill latest lineage entry.

### 14.5 Maternal lineage semantics

Статус:

- `fixed`

Почему:

- mutation and crossover now inherit maternal lineage path correctly.

### 14.6 Rich canonical genetic code artifact

Статус:

- `partially fixed`

Почему:

- `genetic_code.md` exists and is richer than old `idea_dna`;
- runtime still under-validates richness and still accepts legacy fallback forms.

### 14.7 Organism-first canonical architecture

Статус:

- `partially fixed`

Почему:

- organism-first loop exists and is substantial;
- candidate-first path still alive and still influences repo surface.

### 14.8 Remove duplicate architecture branches

Статус:

- `still missing`

Почему:

- candidate orchestrator, legacy prompts, candidate storage, legacy operator layer all remain.

### 14.9 Unify optimizer contract

Статус:

- `mostly fixed`

Почему:

- validator, import path, README and examples now agree far better.

### 14.10 Simple phase and Great Filter semantics

Статус:

- `partially fixed`

Почему:

- separate phase structure now exists;
- Great Filter still may run in smoke mode;
- selection semantics clearer but not fully closed.

### 14.11 Config cleanup and truthfulness

Статус:

- `partially fixed but still weak`

Почему:

- config structure improved;
- several knobs remain unused or fossilized.

### 14.12 Test-surface rebuild

Статус:

- `partially fixed`

Почему:

- more tests exist;
- canonical path still under-covered compared to what matters most.

### 14.13 Reward pipeline overhaul

Статус:

- `partially fixed`

Почему:

- baseline-relative F1 path implemented;
- subset aggregation math remains questionable;
- hard/simple budget separation incomplete.

## 15. Самые важные новые findings, которых не было в прежней картине

Это, пожалуй, центральный практический раздел нового re-audit.

### 15.1 Inter-island crossover rate bug

Это новый concrete bug.

Он не просто theoretical.

Он меняет search dynamics.

### 15.2 Mutation/crossover split not really config-controlled

Это новый config truthfulness gap.

Теперь архитектура уже есть, но один из ключевых evolutionary knobs silently defaults to `0.5`.

### 15.3 Neyman subset aggregation likely mathematically wrong

Это, возможно, самый опасный scientific bug в текущем состоянии.

Если estimator wrong, then reward ranking itself may be biased.

### 15.4 Great Filter may not actually be hard in runtime terms

Это сильная product-level semantic mismatch.

На уровне narrative hard phase есть.

На уровне runtime budget semantics это не до конца правда.

### 15.5 Canonical path still coexists with active legacy generation pipeline

В старом аудите это была mostly architectural complaint.

Сейчас это уже конкретнее:

- coexistence affects prompts;
- affects storage;
- affects allocation history;
- affects tests;
- affects docs truthfulness.

### 15.6 Tests are now strong enough to hide problems, not just expose absence

Раньше зеленые тесты почти ничего не значили.

Теперь зеленые тесты значат больше.

Именно поэтому они теперь могут создавать более опасное ложное чувство завершенности.

## 16. Что я бы канонически сохранял без рефакторинга назад

Ниже пункты, которые выглядят как правильные конечные направления.

Сохранять:

- `conf/prompts` model;
- `conf/islands` model;
- `population_manifest.json`;
- organism-first `EvolutionLoop`;
- `genetic_code.md` format as base shape;
- `lineage.json`;
- `simple_reward / hard_reward / selection_reward` split;
- strict optimizer contract;
- `train_loss` baseline-relative reward framing;
- `PromptBundle` as abstraction;
- per-island top-k and top-h helpers;
- manifest restore over generation scan.

## 17. Что я бы канонически считал legacy и перестал развивать

Считать legacy:

- raw candidate-only generation as main research path;
- `src/evolve/prompts/optimizer_system.txt`;
- `src/evolve/prompts/optimizer_user.txt`;
- `load_best_context()` as canonical prompt context source;
- `cand_*` layout as main evolution layout;
- tournament/elite reproduction planning as product-center logic;
- stale normalization fields `quality_ref` / `steps_ref` as active semantics;
- silent belief that `selection_strategy` or `parent_sampling` already управляют всем runtime behavior.

## 18. Priority matrix

### P0

- Fix inter-island crossover probability bug in father-pool selection.
- Fix or formally redefine Neyman subset aggregate estimator.
- Make mutation-vs-crossover probability explicit and config-truthful.
- Make Great Filter truly hard by phase-specific eval budget/mode, or rename semantics honestly.
- Stop legacy candidate history from contaminating canonical organism allocation history.
- Add tests for canonical organism operators and key phase invariants.

### P1

- Finish prompt source-of-truth migration or quarantine legacy prompt pipeline.
- Tighten structured LLM response validation and genetic-code richness checks.
- Deprecate/remove unused config fossils and stale experiment normalization fields.
- Reduce legacy architecture surface from core modules or isolate it clearly.
- Add stronger manifest integrity validation.

### P2

- Improve lineage summary quality for prompt reasoning.
- Add per-island population customization if still desired.
- Add stronger research analytics fields for cross-island ancestry.
- Remove weak `score` alias semantics over time.

### P3

- Clean up small type aliases and stale helper names.
- Further split storage/orchestrator modules by legacy vs canonical concerns if needed.

## 19. Concrete remediation roadmap after this audit

Ниже roadmap не такой большой, как `7.md`, но достаточный как updated post-reaudit guidance.

### Шаг 1. Закрыть statistical и topology bugs

Сначала:

- inter-island probability bug;
- Neyman aggregate math;
- explicit mutation/crossover probability config.

Почему first:

- это bugs, которые прямо меняют search dynamics and score meaning.

### Шаг 2. Довести phase semantics до честного состояния

Сделать:

- phase-specific eval regime or explicit hard-mode semantics;
- direct tests for simple reward persistence and Great Filter behavior;
- clearer selection_reward documentation/runtime naming.

### Шаг 3. Добить source-of-truth cleanup

Сделать:

- quarantine or migrate candidate prompt path;
- decide whether candidate orchestration is still supported product behavior;
- remove config lies around unused keys.

### Шаг 4. Добить canonical organism quality

Сделать:

- richer validation of genetic code;
- stronger LLM response required sections;
- real tests for `src/organisms/mutation.py` and `src/organisms/crossbreeding.py`.

### Шаг 5. Только после этого продолжать новые research features

Не стоит раньше:

- добавлять новые novelty heuristics;
- добавлять новые orchestration layers;
- сильно расширять experiment set semantics.

Сначала нужно закрыть truthfulness.

## 20. Конкретные маленькие баги и smells

Ниже flat list smaller items, которые не являются главным содержанием audit, но стоят внимания.

### 20.1

`valopt/utils/import_utils.py` содержит stale alias `BuildOptimizerCallable`, который не совпадает с real contract.

### 20.2

`tests/test_evolve_integration_fake.py` still encodes old normalization fields although reward path is baseline-relative.

### 20.3

`tests/test_genetic_operators.py` title and coverage can mislead reader into thinking canonical genetic operators are covered, when mostly legacy prompt-operator layer is being tested.

### 20.4

`src/evolve/generator.py` still has dual personality:

- legacy raw-code generation;
- structured organism generation.

Это сильный architectural smell even if not immediate bug.

### 20.5

`src/evolve/storage.py` is doing too much for both worlds at once.

### 20.6

`read_population_manifest()` validates container shape weakly.

### 20.7

`format_lineage_summary()` shows recent entries only by truncation count, without ranking by importance or improvement magnitude.

### 20.8

`format_genetic_code()` and storage format are readable, but there is no explicit maximum/quality control, so artifact drift can accumulate.

### 20.9

`README` correctly points users toward new architecture, but does not explicitly warn which legacy paths are still supported only for compatibility.

### 20.10

`run.py` auto-fallback can obscure which engine actually ran.

## 21. Обновленный интегральный вердикт по подсистемам

### 21.1 Prompts

- better than before;
- not fully closed;
- duplicate source of truth remains.

### 21.2 Islands

- real subsystem now;
- one critical probability bug;
- lifecycle testing incomplete.

### 21.3 Organism model

- substantially improved;
- lineage backfill solved;
- richness enforcement still weak.

### 21.4 Storage/resume

- manifest design is a real win;
- migration helpers too broad;
- integrity validation still shallow.

### 21.5 Selection/runtime

- mostly aligned with intended design;
- still not config-truthful enough;
- one topology bug and one semantics ambiguity remain important.

### 21.6 Reward/evaluation

- major conceptual progress;
- likely mathematical problem in aggregate estimator;
- hard-phase realism incomplete.

### 21.7 Public contract

- strongest completed area;
- only minor cleanup remains.

### 21.8 Tests

- stronger and more respectable;
- still insufficient as proof of full semantic closure.

## 22. Что особенно важно понять перед следующим шагом разработки

Сейчас проект уже не находится в фазе “architectural vacuum”.

Он находится в другой фазе.

Это важнее, чем может показаться.

Новая фаза такая:

- architecture skeleton уже есть;
- основные nouns уже materialized;
- главная опасность сместилась с `не придумали модель` на `рано решили, что модель уже доведена`.

Именно поэтому следующий шаг должен быть не “еще одна большая feature”.

Следующий шаг должен быть:

- truthfulness cleanup;
- semantic closure;
- removal or quarantine of duplicate paths;
- fixing math/selection bugs;
- upgrading tests from “что-то работает” to “target semantics proved”.

Если этого не сделать, проект окажется в самой опасной промежуточной точке:

- сверху будет казаться, что island-aware organism platform уже готова;
- снизу она все еще будет partially steered by legacy paths and hidden assumptions.

## 23. Итог

Новый re-audit показывает:

- работа последних итераций реально продвинула репозиторий вперед;
- часть старых критических замечаний больше нельзя считать open;
- но migration еще не завершена и теперь содержит уже другой класс рисков.

Самые важные итоги одной плоской матрицей:

- `prompt assets in conf`: да, но не полностью закрыто.
- `islands as real runtime entity`: да.
- `manifest-driven resume`: да.
- `lineage score backfill`: да.
- `optimizer contract unification`: почти да.
- `simple/hard phase semantics`: частично да.
- `Great Filter truly hard`: пока нет полностью.
- `selection semantics config truthfulness`: нет полностью.
- `Neyman estimator semantics`: под сильным вопросом.
- `legacy architecture retirement`: пока нет.
- `canonical path best-tested`: пока нет.

Мой главный practical recommendation after current re-audit:

не расширять систему дальше, пока не закрыты следующие пять вещей:

1. inter-island probability bug;
2. unbiased/intended aggregate math under Neyman allocation;
3. honest phase semantics for Great Filter;
4. config truthfulness for reproduction operators;
5. decisive separation of canonical organism path from legacy candidate residue.

После этого проект будет уже не просто “сильно лучше, чем раньше”.

Он начнет быть именно тем исследовательским двигателем эволюции оптимизаторов, который ты описываешь в ТЗ.
