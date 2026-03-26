# Follow-up: Train-Loss Baseline Reward Rework

## 1. Исходный контекст

До этого запроса контекст был такой:

- Нужно было переработать reward-схему для оценки оптимизаторов, которые предлагает нейросеть.
- Пользователь хотел уйти от старой схемы, где score считался через статические `quality_ref` / `steps_ref`.
- Вместо этого нужно было:
  - иметь отдельный `stat run` для каждого эксперимента;
  - сохранять из него baseline-статистики;
  - затем считать reward кандидата относительно этого baseline.

Изначально формулировка была про `val_loss`, `val_total_steps` и `F1-score` от двух отношений. В процессе обсуждения пользователь уточнил и зафиксировал более важное ограничение:

- reward должен использовать **только train objective**;
- `val_*`, accuracy, perplexity и прочее можно сохранять как дополнительную информацию, но в reward они не участвуют;
- на эксперимент теперь должен быть **один baseline JSON**;
- baseline-run пользователь хочет иметь возможность остановить руками через `Ctrl+C`, ориентируясь по логам train-loss.

Это было важным поворотом, потому что кодовая база до этого была построена вокруг primary metric, validation metrics и старой нормализации.

## 2. Что было в коде до изменений

Перед реализацией я просмотрел текущую архитектуру и зафиксировал следующее:

### 2.1 Режимы запуска

- В проекте уже были режимы `run`, `stats`, `smoke`, `evolve`.
- `mode=stats` уже существовал и писал JSON в `stats/<experiment>/baseline__<run_id>.json`.
- `mode=run` работал с внешним optimizer file.
- `src.validate.run_one` использовался subprocess-воркерами в evolve.

### 2.2 Как раньше считался reward

- Reward в evolve шёл через:
  - `src/evolve/scoring.py`
  - `src/evolve/metrics_adapter.py`
- Там score строился из:
  - `quality_ratio`
  - `steps_ratio`
  - линейной комбинации с `score_weights`
- Базой служили конфиговые `normalization.quality_ref` и `normalization.steps_ref`, а не реальные baseline-run результаты.

### 2.3 Что сохраняли train loop’ы

- Все train loop’ы возвращали `final_metrics`, `best_metrics`, `steps`, `wall_time_sec` и `extra_metrics`.
- Но:
  - почти нигде не было единого top-level контракта для train objective;
  - не было поля типа `objective_last`, `objective_best`, `first_step_at_or_below_baseline`;
  - не было общего tracker’а train objective;
  - baseline-relative скорость нельзя было посчитать без дополнительной логики;
  - `Ctrl+C` не сериализовал partial baseline-run как валидный результат.

### 2.4 Что ещё обнаружилось

- В нескольких loop’ах вообще не было нормального логирования train loss по `log_every_steps`.
- `final_metrics` и `best_metrics` часто были целиком завязаны на validation evaluation.
- Для DDPM уже использовались train-like метрики (`train_loss`, `ema_loss`), то есть экосистема уже не была единообразно `val_loss`-only.
- В репозитории уже были пользовательские незакоммиченные изменения и новые файлы; я их не трогал и не откатывал.

## 3. Как был выработан финальный план

После просмотра кода и уточнений от пользователя была зафиксирована следующая рабочая модель:

### 3.1 Новый источник reward

- Reward считается **только по train-loss**.
- Для первой версии использовать:
  - `baseline.objective_last`
  - `candidate.objective_last`
  - `candidate.first_step_at_or_below_baseline`

### 3.2 Формулы

- `quality_term = baseline.last_train_loss / candidate.last_train_loss`
- `speed_term = baseline.total_steps / candidate.first_step_at_or_below_baseline`
- Если кандидат baseline-loss не достиг:
  - `speed_term = 0`
  - `exp_score = 0`
- Финальный per-experiment score:
  - harmonic mean этих двух термов

### 3.3 Baseline storage

- Один baseline JSON на эксперимент.
- Канонический путь:
  - `stats/<experiment>/baseline.json`

### 3.4 Train loop contract

- Все loop’ы должны начать возвращать единые поля:
  - `objective_name`
  - `objective_direction`
  - `objective_last`
  - `objective_best`
  - `objective_best_step`
  - `first_step_at_or_below_baseline`

### 3.5 Interrupt model

- `mode=stats` должен переживать `KeyboardInterrupt`.
- При ручной остановке нужно всё равно сериализовать usable baseline JSON.

## 4. Что именно было реализовано

Ниже то, что было сделано фактически.

### 4.1 Добавлен общий tracker train objective

Добавлены новые utility-файлы:

- `valopt/utils/objective_tracking.py`
- `valopt/utils/baselines.py`

Что они дали:

- единый `TrainObjectiveTracker` для:
  - `last_train_loss`
  - `best_train_loss`
  - `best_train_loss_step`
  - `first_step_at_or_below_baseline`
- helper’ы для:
  - безопасного чтения objective float;
  - подмешивания `train_loss` в metrics;
  - загрузки и валидации baseline profile;
  - инъекции baseline threshold в runtime config эксперимента.

### 4.2 Расширен runtime step accounting

В `valopt/utils/optimizer_runtime.py` был добавлен `current_cycle_consumed`, чтобы:

- корректно считать step index не только для обычных outer-loop шагов;
- но и для дополнительных `step_fn` вызовов внутри optimizer controller.

Это было важно, потому что новая baseline-hit логика должна учитывать все runtime-steps, которые реально списываются бюджетом.

### 4.3 Расширена схема результата

В `valopt/schemas.py`:

- в `RunResult` добавлены поля:
  - `objective_name`
  - `objective_direction`
  - `objective_last`
  - `objective_best`
  - `objective_best_step`
  - `first_step_at_or_below_baseline`
- `status` расширен значением `interrupted`
- schema validation обновлена под новые поля

Идея была такая:

- не прятать train-objective внутрь произвольных nested dict;
- сделать его явной частью контракта результата.

### 4.4 Переписан runner под baseline-aware execution

В `valopt/runner.py` были сделаны следующие вещи:

- `mode=run` теперь пытается подхватить baseline profile эксперимента и прокинуть:
  - `runtime.baseline_path`
  - `runtime.baseline_last_train_loss`
  - `runtime.baseline_load_error`
- `mode=stats` теперь пишет canonical:
  - `stats/<experiment>/baseline.json`
- fallback payload’ы для failed/skipped сценариев тоже заполняют objective-поля
- serialization `RunResult` теперь включает новые objective-поля

Это было нужно для того, чтобы:

- baseline был единым источником истины;
- train loop’ы могли знать baseline threshold заранее;
- stats и run использовали один и тот же контракт результатов.

### 4.5 Обновлён subprocess evaluator

В `src/validate/run_one.py`:

- добавлена baseline injection в experiment runtime;
- payload теперь сохраняет наверх:
  - `objective_name`
  - `objective_direction`
  - `objective_last`
  - `objective_best`
  - `objective_best_step`
  - `first_step_at_or_below_baseline`
- legacy `raw_metric` / `final_score` сохранены для совместимости, но scoring больше на них не опирается

Это было критично для evolve worker’ов, потому что именно `run_one` формирует JSON, который потом читает scoring layer.

### 4.6 Обновлены train loop’ы

Были изменены:

- `experiments/_shared/train_supervised.py`
- `experiments/_shared/train_paramonly.py`
- `experiments/cifar_convnet/train.py`
- `experiments/audio_transformer/train.py`
- `experiments/mnist_mlp/train.py`
- `experiments/synthetic_logreg/train.py`
- `experiments/minigpt_wikitext2/train.py`
- `experiments/lora_sft/train.py`
- `experiments/ddpm_cifar10/train.py`

Что было внедрено во всех этих loop’ах:

- подключён `TrainObjectiveTracker`
- train loss обновляется на каждом учитываемом runtime-step
- если optimizer controller вызывает `step_fn`, objective tracker тоже обновляется на этих шагах
- `train_loss` теперь гарантированно приклеивается к `final_metrics`
- `train_loss` также присутствует в `best_metrics`
- на верхний уровень `result_core` добавлены:
  - `objective_name="train_loss"`
  - `objective_direction="min"`
  - `objective_last`
  - `objective_best`
  - `objective_best_step`
  - `first_step_at_or_below_baseline`
- добавлено train-loss логирование по `train.log_every_steps`
- добавлена обработка `KeyboardInterrupt`
  - статус выставляется в `interrupted`
  - уже собранные objective-метрики не теряются

Особенно важный edge case, который отдельно был закрыт:

- если run прерывается между eval-чекпоинтами, `final_metrics.train_loss` всё равно обновляется до последнего реально наблюдавшегося значения, а не остаётся старым snapshot’ом от предыдущей evaluation точки.

### 4.7 Переписан evolve scoring

Изменены:

- `src/evolve/metrics_adapter.py`
- `src/evolve/scoring.py`
- `src/evolve/orchestrator.py`

Что изменилось по сути:

- orchestrator теперь заранее грузит baseline profiles для ожидаемых экспериментов;
- старый scoring от `quality_ref` / `steps_ref` удалён из reward path;
- `extract_metrics(...)` теперь считает:
  - `quality_ratio` как baseline-relative loss term
  - `steps_ratio` как baseline-relative speed term
  - `exp_score` как harmonic mean
- если baseline отсутствует или broken:
  - scoring для эксперимента помечается как `failed`
  - в `error_msg` попадает явная причина
- если run успешен, но baseline threshold не достигнут:
  - `status` остаётся `ok`
  - `steps_ratio = 0`
  - `exp_score = 0`
- aggregate score по нескольким задачам остался weighted mean по allocation `pi`

То есть логика “как агрегировать по нескольким экспериментам” сохранилась, а логика “что является per-experiment reward” была заменена полностью.

## 5. Что было сделано в тестах

Тестовый слой тоже был обновлён под новый контракт.

### 5.1 Обновлены существующие тесты

Изменены:

- `tests/test_result_schema.py`
- `tests/test_metrics_normalization.py`
- `tests/test_scoring.py`
- `tests/test_evolve_integration_fake.py`
- `tests/fixtures/fake_eval.py`

Что там поменялось:

- schema test теперь проверяет новые поля и `status="interrupted"`
- metrics tests проверяют:
  - baseline-relative `quality_ratio`
  - baseline-relative `steps_ratio`
  - harmonic mean
  - сценарий missing baseline
- scoring tests теперь используют baseline profiles вместо `quality_ref` / `steps_ref`
- fake evaluator теперь отдаёт objective-поля, а не только legacy `final_score`
- fake evolve integration создаёт baseline.json для тестовых экспериментов и валидирует, что новый scoring реально используется

### 5.2 Добавлен новый baseline test

Добавлен файл:

- `tests/test_baselines.py`

Он проверяет:

- успешную загрузку baseline profile
- missing baseline file
- invalid baseline content

## 6. Что было проверено после реализации

### 6.1 Целевой набор тестов

Сначала был прогнан набор тестов по изменённым зонам:

- `tests/test_result_schema.py`
- `tests/test_metrics_normalization.py`
- `tests/test_scoring.py`
- `tests/test_baselines.py`
- `tests/test_evolve_integration_fake.py`
- `tests/test_run_validation_overrides.py`

На этом шаге всплыл один конфиговый баг:

- baseline injection в experiment cfg приводил к тому, что `${paths.data_root}` резолвился раньше, чем `paths` были вставлены в experiment config.

Это было исправлено перестановкой порядка инициализации:

- сначала `exp_cfg.paths`
- потом доступ к `exp_cfg.data.root`

После этого targeted test set стал зелёным.

### 6.2 Синтаксический sanity-pass

Отдельно был сделан `py_compile` для изменённых train loop’ов и orchestration-файлов, чтобы не оставить синтаксические ошибки в файлах, которые тесты могли не зацепить напрямую.

### 6.3 Полный pytest

После targeted проверки был прогнан полный `pytest -q`.

Итог:

- `49 passed in 2.36s`

## 7. Итоговое состояние после работы

К моменту этого follow-up было сделано следующее:

- train-loss стал явным first-class objective во всей системе результатов;
- появился canonical baseline profile на эксперимент;
- `mode=stats` стал пригоден для ручного прерывания без потери результата;
- `run_one` и main runner начали сериализовать objective-контракт наверх;
- evolve scoring перестал опираться на static normalization refs и перешёл на baseline-relative reward;
- тестовый контур был приведён в соответствие и целиком проходит.

## 8. Что я сознательно не делал

Есть несколько вещей, которые я намеренно не трогал:

- не откатывал и не переписывал чужие незакоммиченные изменения в репозитории;
- не чистил старые конфиговые `normalization.*` ключи физически из YAML, хотя reward их больше не использует;
- не вводил отдельный “pure optimizer.step counter”;
  - использован текущий runtime step counter проекта, который уже учитывает дополнительные `step_fn` вызовы;
- не переводил reward на `objective_best`;
  - он сохраняется, но в текущую формулу не входит.

## 9. Краткая суть одним блоком

Если совсем кратко, то до этого запроса я:

- разобрал текущую reward-архитектуру и её ограничения;
- зафиксировал новый baseline-relative train-loss reward contract;
- внедрил train-objective tracking во все нужные training loops;
- расширил result schema и runner/run_one;
- перевёл evolve scoring на baseline profile;
- добавил canonical baseline storage;
- сделал stats-run устойчивым к ручному interrupt;
- обновил и прогнал тесты до полного зелёного прогона.
