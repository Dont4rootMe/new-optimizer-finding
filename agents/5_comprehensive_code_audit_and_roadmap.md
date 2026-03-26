# 5. Comprehensive Code Audit & Implementation Roadmap

**Date:** 2026-03-26
**Scope:** Full codebase audit against specification. Every discrepancy, bug, and missing feature documented.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture: Current State vs Specification](#2-architecture-current-state-vs-specification)
3. [CRITICAL: Missing Features](#3-critical-missing-features)
4. [Specification Violations](#4-specification-violations)
5. [Bugs & Defects](#5-bugs--defects)
6. [Prompt Engineering Issues](#6-prompt-engineering-issues)
7. [Configuration Issues](#7-configuration-issues)
8. [Reward System Audit](#8-reward-system-audit)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Priority Matrix](#10-priority-matrix)

---

## 1. Executive Summary

The codebase has a solid foundation: 17 experiments, a working LLM-driven code generation pipeline, template-based optimizer rendering, Neyman allocation, baseline-relative scoring, GPU worker pool, and async orchestration. However, comparing the implementation against the specification reveals **5 critically missing features**, **8 specification violations**, **7 bugs**, and **multiple prompt engineering deficiencies**.

The single most impactful gap is the **complete absence of the island model** --- the entire population topology that should drive diversity and specialization is missing. The second most impactful is the **broken parent selection mechanism** (tournament instead of softmax over reward). Together, these mean the evolutionary dynamics are fundamentally different from what was designed.

---

## 2. Architecture: Current State vs Specification

### What exists (working)

| Component | Location | Status |
|---|---|---|
| Optimizer template with fixed signature | `src/evolve/templates/optimizer_template.txt` | OK |
| LLM code generation with retry | `src/evolve/generator.py` | OK |
| Idea DNA as semicolon-separated traits | `src/evolve/types.py:OrganismMeta.idea_dna` | Partially OK |
| Evolution log (lineage history) | `src/evolve/types.py:OrganismMeta.evolution_log` | Buggy |
| Probabilistic mutation (trait deletion) | `src/organisms/mutation.py` | OK mechanically, wrong selection |
| Probabilistic crossbreeding (trait recombination) | `src/organisms/crossbreeding.py` | OK mechanically, wrong selection & lineage |
| Template parser & validator | `src/evolve/template_parser.py` | OK |
| Neyman allocation | `src/evolve/allocation.py` | OK |
| Baseline-relative scoring (F1/harmonic mean) | `src/evolve/metrics_adapter.py` | Mostly OK |
| GPU worker pool | `src/evolve/gpu_pool.py` | OK |
| Async orchestrator | `src/evolve/orchestrator.py` | OK for candidates |
| Multi-generation evolution loop | `src/evolve/evolution_loop.py` | Structurally incomplete |
| 17 experiment configs | `conf/experiments/*.yaml` | OK |
| Train objective tracking | `valopt/utils/objective_tracking.py` | OK |
| Baseline profiles | `valopt/utils/baselines.py` | OK |

### What is missing entirely

| Component | Specification Reference | Impact |
|---|---|---|
| Island model | Spec point 4 | CRITICAL --- no population topology |
| Per-island selection | Spec point 3 | CRITICAL --- no independent evolution tracks |
| Softmax parent selection for crossbreeding | Spec point 2 (cross breading) | CRITICAL --- wrong selection pressure |
| Island-specific system prompts & seeding | Spec point 4 | CRITICAL --- no research "schools" |
| Inter-island cross-breeding | Spec point 3 | HIGH --- no cross-pollination |
| `conf/islands/*.txt` directory & loader | Spec point 4 | HIGH --- no island definitions |
| Constant system prompt in `./conf` | Spec point 2a | MEDIUM --- wrong prompt location |

---

## 3. CRITICAL: Missing Features

### 3.1 Island Model --- Completely Absent

**Spec says:**
> Остров --- это группа особей, что разделяют своей собственное и неповторимое направление --- как типо школа исследователей, которая изучает какое-то направление оптимизации. Количество островов задается числом txt файлов в специально отведенной папке в ./conf --- каждый файл txt --- это описание этой самой школы.

**Current state:** No concept of islands exists anywhere in the code. There is:
- No `conf/islands/` directory
- No island data structure
- No per-island population tracking
- No island assignment for organisms
- No island-specific prompts or seeding

**What needs to be built:**

```
conf/
  islands/
    gradient_methods.txt      # "School of gradient-based optimization"
    second_order.txt          # "School of curvature-aware methods"
    population_based.txt      # "School of population-inspired ideas"
    ...
```

New data structures needed:
- `Island` dataclass: `island_id`, `name`, `description_path`, `population: list[OrganismMeta]`
- `island_id` field on `OrganismMeta`
- Per-island population tracking in `EvolutionLoop`

New logic needed:
- Island loader: reads `conf/islands/*.txt`, creates Island objects
- Per-island seeding: for each island, generate N organisms using island-specific prompt
- Per-island selection: top-k/top-h within each island independently
- Intra-island breeding: cross-breed within an island
- Inter-island breeding: cross-breed between islands (configurable rate)

**Files to modify:**
- `src/evolve/types.py` --- add `island_id` to `OrganismMeta`
- `src/evolve/evolution_loop.py` --- complete rewrite of population management
- `src/evolve/selection.py` --- add per-island variants
- `src/evolve/operators.py` --- `SeedOperator` must accept island description
- `src/evolve/prompts/seed_system.txt` / `seed_user.txt` --- add island context
- `conf/evolver/default.yaml` --- add island config (num_per_island, inter_island_rate, etc.)

**Estimated scope:** ~500-700 lines of new/modified code.

---

### 3.2 Softmax Parent Selection for Cross-breeding

**Spec says:**
> cross breading --- мы выбираем из распределения softmax над reward этих моделей две особи --- одна получается мать другая отец

**Current state (`src/evolve/selection.py:62-74`):**
```python
if rng.random() < mutation_rate or len(survivors) < 2:
    parent = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
    plan.append(("mutation", [parent]))
else:
    parent_a = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
    parent_b = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
```

Tournament selection is used for BOTH mutation and crossover. The spec requires:
- **Crossover:** softmax(reward) sampling for both parents
- **Mutation:** uniform random sampling

**What needs to change:**

New function needed in `selection.py`:
```python
def softmax_select(population, k=1, score_key="simple_score", temperature=1.0, rng=None):
    """Sample k organisms with probability proportional to softmax(score/temperature)."""
    scores = [getattr(org, score_key) or 0.0 for org in population]
    # softmax
    max_s = max(scores)
    exp_scores = [math.exp((s - max_s) / temperature) for s in scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]
    return rng.choices(population, weights=probs, k=k)
```

New function for mutation:
```python
def uniform_select(population, k=1, rng=None):
    """Sample k organisms uniformly at random."""
    return rng.choices(population, k=k)
```

Update `select_parents_for_reproduction` to use:
- `softmax_select` for crossover parents
- `uniform_select` for mutation parents

**Config addition** (`conf/evolver/default.yaml`):
```yaml
evolution:
  softmax_temperature: 1.0  # temperature for softmax parent selection
```

---

### 3.3 Per-Island Selection (top-k and top-h)

**Spec says:**
> Далее это число приписывается особи и остается с ней навсегда. После всего этого мы ранжируем наши все особи на всех островах по отдельности и оставляем только top-k особей на каждом острове

> мы так же ранжируем все особи на каждом острове по отдельности и оставляем другое число top-h

**Current state (`src/evolve/evolution_loop.py:278-281`):**
```python
survivors = elite_select(
    self.population, elite_count, score_key="simple_score"
)
```

Single flat `elite_select` over the entire population. No per-island logic.

**What needs to change:**
- `elite_select` must operate per-island
- Two separate counts: `simple_top_k` and `great_filter_top_h`
- Current `elite_count: 5` must be split into two config values

**Config additions:**
```yaml
evolution:
  simple_top_k: 5          # survivors per island after simple evaluation
  great_filter_top_h: 3    # survivors per island after Great Filter
```

---

### 3.4 Island-Specific Seeding

**Spec says:**
> Далее мы должны иметь специальную команду --- она посмотрит на конфиг и сделает следующее --- найдет, сколько особей каждой школы для инициализации стоит придумать, далее идет в API и со специально подготовленным системным промптом плюс текстом из txt просит LLM придумать один генетический код

**Current state:** `SeedOperator` uses generic `seed_system.txt` / `seed_user.txt` with no island context.

**What needs to change:**
- `SeedOperator.build_prompts()` must accept island description text
- System prompt must include island/school context
- User prompt must reference the school's research direction
- Config must specify `organisms_per_island` for initialization

**New prompt structure:**
```
seed_system.txt (modified):
  "You are designing an optimizer in the tradition of {island_description}..."

seed_user.txt (modified):
  "Your school of thought focuses on: {island_text}
   Design a novel optimizer inspired by this direction..."
```

---

### 3.5 System Prompt Location

**Spec says:**
> a) --- константная вещь, что должна храниться в ./conf директории, как текстовый файл

**Current state:** All prompts live in `src/evolve/prompts/`. The constant system prompt that "immerses the agent into the context" should be in `./conf`.

**What needs to change:**
- Move or create the constant system prompt at `conf/system_prompt.txt`
- The per-operation prompts (mutation, crossover, seed) can stay in `src/evolve/prompts/` as they're code-coupled
- `OptimizerGenerator` should load the constant context from `conf/system_prompt.txt` and prepend it to all LLM calls

---

## 4. Specification Violations

### 4.1 Crossover Merges Both Parents' Evolution Logs

**Spec says:**
> В частности, в любом размножении у нас есть главная особь (типо мать) --- хранитель родословной, особь следующая хранит в себе этот путь со скорами.

Only the mother's (dominant parent's) lineage should be inherited.

**Current code (`src/organisms/crossbreeding.py:167-169`):**
```python
merged_log = list(dominant.evolution_log) + list(non_dominant.evolution_log)
merged_log.sort(key=lambda e: e.get("generation", 0))
```

This merges BOTH parents' logs, creating a combined history that doesn't match either parent's lineage.

**Fix:** Replace with:
```python
# Only inherit the dominant parent's (mother's) lineage
parent_evolution_log = list(dominant.evolution_log)
```

**Location:** `src/organisms/crossbreeding.py:167-169`

---

### 4.2 Great Filter Evaluates on Simple + Hard (Should Be Hard Only)

**Spec says:**
> Сложная фаза или "Великий фильтр" --- мы прогоняем каждую дожившую особь через сложный алгоритм отбора по каждому сложному алгоритму и усредняем результат

Great Filter should use ONLY `hard_experiments`.

**Current code (`src/evolve/evolution_loop.py:322-324`):**
```python
all_experiments = self.simple_experiments + self.hard_experiments
await self._evaluate_organisms(
    self.population, all_experiments, score_key="hard_score"
)
```

**Fix:** Replace `all_experiments` with `self.hard_experiments`:
```python
await self._evaluate_organisms(
    self.population, self.hard_experiments, score_key="hard_score"
)
```

**Location:** `src/evolve/evolution_loop.py:322`

---

### 4.3 Mutation Uses Tournament Selection Instead of Uniform

**Spec says:**
> mutation --- мы так же случайным образом уже из равномерного распределения семплируем несколько особей

**Current code (`src/evolve/selection.py:63-65`):**
```python
if rng.random() < mutation_rate or len(survivors) < 2:
    parent = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
```

**Fix:** Use `rng.choice(survivors)` for mutation parent selection instead of `tournament_select`.

---

### 4.4 No Distinction Between top-k (Simple) and top-h (Great Filter)

**Spec says two different counts:**
> оставляем только top-k особей на каждом острове (простая фаза)
> оставляем другое число top-h (Великий фильтр)

**Current code:** Both use single `elite_count: 5`.

**Fix:** Add `simple_top_k` and `great_filter_top_h` to config, use appropriate one for each phase.

---

### 4.5 LLM's New Ideas During Mutation Are Silently Discarded

**Spec says:**
> LLM должно подумать, что она хочет сделать сейчас, изменить описание гена b) и резюмировать изменения

The LLM should be able to modify the DNA --- add, remove, and change genes.

**Current code (`src/organisms/mutation.py:155-156`):**
```python
idea_dna_override=child_dna,  # <-- discards whatever DNA the LLM proposed
```

Meanwhile, `src/evolve/prompts/mutate_user.txt:9` tells the LLM:
> You MAY add ONE new idea to replace the removed ones. If you do, include it in the IDEA_DNA section of your response.

But the `idea_dna_override` parameter in `build_organism_from_response` (line 85-86 of `organism.py`) causes the LLM's IDEA_DNA output to be ignored:
```python
if idea_dna_override is not None:
    idea_dna = list(idea_dna_override)
```

**This is a significant design contradiction.** The LLM is promised the ability to evolve DNA, but the code overrides it.

**Fix options:**
1. **Option A (spec-aligned):** Remove `idea_dna_override` for mutation. Let the LLM's IDEA_DNA section be used. Phase 1 (probabilistic deletion) determines which genes are removed; Phase 2 (LLM) determines what replaces them. The LLM's output DNA becomes the child's DNA.
2. **Option B (hybrid):** Phase 1 removes genes, LLM sees the reduced set, LLM can add new genes. Merge: keep Phase 1 removals, add LLM's new additions.

Recommendation: **Option A** is closer to the spec where "LLM должно подумать, что она хочет сделать сейчас, изменить описание гена".

---

### 4.6 Idea DNA Is Not Rich Enough

**Spec says:**
> b) часть должна быть настолько богатой, что по ней можно восстановить идею для реализации оптимизатора из шаблона d)

**Current state:** The mock DNA looks like: `"mock SGD with cosine schedule; warmup phase"`. The seed prompt asks for:
> A semicolon-separated list of core ideas

This doesn't encourage the level of richness the spec demands. Each "gene" should be a detailed description of a principle, not a brief label.

**Fix:** Revise prompts to demand detailed genes. Example of rich DNA:
```
Adaptive per-layer learning rate: scale base_lr by inverse of gradient RMS for each parameter group,
clamped to [0.1x, 10x] base_lr to prevent instability;
Dual momentum with phase switching: use heavy momentum (0.95) during first 30% of training for
exploration, switch to light momentum (0.8) for fine-grained convergence;
Gradient noise injection with annealing: add Gaussian noise N(0, sigma/(1+t)^0.55) to gradients
to escape sharp minima, where sigma is calibrated to initial gradient norm
```

---

### 4.7 Prompts Don't Frame Genes as Individually Manipulable Units

**Spec says:**
> LLM должна воспринимать этот генетический код (буквально тексты разделенные ;) как список генов, и работать с ним он должен соответствующе

**Current prompts** describe idea_dna as "core ideas" but don't frame them with the genetic metaphor the spec demands. The LLM should understand:
- Each semicolon-separated block is a "gene"
- Genes can be independently added, removed, or modified
- The DNA is the complete genetic blueprint from which the optimizer is reconstructed
- Evolution log shows which gene changes led to score improvements/regressions

**Fix:** Revise all system prompts to include explicit genetic framing.

---

### 4.8 `score_weights` Config Is Ignored

**Config (`conf/evolver/default.yaml:55-57`):**
```yaml
score_weights:
  quality: 0.7
  steps: 0.3
```

**Current code (`src/evolve/metrics_adapter.py:104-105`):**
```python
exp_score = (2.0 * quality_ratio * steps_ratio) / max(quality_ratio + steps_ratio, eps)
```

This is a pure harmonic mean (F1-score) with equal weighting. The `score_weights` from config are passed as `scoring_cfg` to `extract_metrics` but immediately discarded (line 35: `del scoring_cfg`).

**Fix:** Either:
1. Use `score_weights` to compute weighted harmonic mean: `(1+beta^2) * q * s / (beta^2 * q + s)` where `beta = quality_weight / steps_weight`
2. Or remove `score_weights` from config to avoid confusion

The spec says "F1-score", which is standard harmonic mean. So probably remove `score_weights` config and document that F1 is used.

---

## 5. Bugs & Defects

### BUG-1: Resume Breaks Across Generations (CRITICAL)

**Location:** `src/evolve/evolution_loop.py:257-258` and `src/evolve/storage.py:147-159`

**Problem:** When the evolution loop resumes, it calls:
```python
pop_dicts = load_population(self.population_root, self.generation)
```

But `load_population` only looks in `gen_{N}/org_*`:
```python
gen_dir = root / f"gen_{generation}"
for org_path in sorted(gen_dir.glob("org_*")):
```

If an organism from generation 0 survives to generation 5 (via elite selection), its files are still stored in `gen_0/org_XXX`. When we resume at generation 5, `load_population(root, 5)` will NOT find it because it only scans `gen_5/org_*`.

**Result:** On resume, all surviving organisms from previous generations are lost. The population is rebuilt only from organisms created in the current generation.

**Fix options:**
1. **Copy surviving organisms to new generation directory** after each selection round
2. **Save population state as a manifest file** that lists all active organism paths regardless of generation
3. **Change `_save_state` to persist full organism dicts, not just IDs**

Recommendation: Option 2. Add a `population_manifest.json` in the population root:
```json
{
  "generation": 5,
  "organisms": [
    {"organism_id": "abc123", "organism_dir": "gen_0/org_abc123", "score": 0.85},
    {"organism_id": "def456", "organism_dir": "gen_5/org_def456", "score": 0.72}
  ]
}
```

---

### BUG-2: Evolution Log Scores Never Updated (HIGH)

**Location:** `src/organisms/organism.py:125-130`

**Problem:** When a new organism is created, its evolution log entry has `score=None`:
```python
new_entry = EvolutionEntry(
    generation=generation,
    change_description=change_description,
    score=None,  # <-- always None
    parent_ids=parent_ids,
)
```

Later, when the organism is evaluated and gets a score, only `OrganismMeta.score` is updated (`evolution_loop.py:232`), but the corresponding entry in `evolution_log` is never updated.

**Result:** When children inherit this evolution log, they see `score=None` for ALL entries. The LLM sees:
```
gen=0: Initial creation (score=None)
gen=1: Added momentum cycling (score=None)
gen=2: Switched to cosine annealing (score=None)
```

This defeats the purpose of the evolution log --- the LLM can't learn which changes helped or hurt.

**Spec says:**
> c) Эволюция скора предков в процессе эволюции и изменения идей --- то есть тут должен быть список --- основное нововведение/изменение парадигмы => результат

**Fix:** After evaluation, update the last evolution log entry's score:
```python
# In _evaluate_organisms, after setting org.score:
if org.evolution_log:
    org.evolution_log[-1]["score"] = org.score
```

Also update the on-disk `evolution_log.json`.

---

### BUG-3: `_save_state` Crashes on Empty or All-None-Score Population

**Location:** `src/evolve/evolution_loop.py:86-93`

**Problem:**
```python
"best_score": max(
    (org.score for org in self.population if org.score is not None),
    default=None,
),
"best_organism_id": max(
    self.population,
    key=lambda o: o.score if o.score is not None else -float("inf"),
).organism_id if self.population else None,
```

The first `max()` has `default=None`, so it's fine. But the second `max()` will crash if `self.population` is non-empty but ALL organisms have `score=None`, because `max()` will compare `None` values.

Actually wait --- it uses a `key` function that returns `-inf` for None scores, so `max()` will still work. But if ALL scores are None, it returns an organism with score None, which is fine.

**Revised severity:** LOW --- not actually a crash. The key function handles None correctly. False alarm, but the code is fragile and deserves a defensive check.

---

### BUG-4: `_evaluate_organisms` Uses Private Orchestrator Methods (MEDIUM)

**Location:** `src/evolve/evolution_loop.py:213`

```python
allocation_snapshot = orchestrator._build_candidate_allocation(org.organism_id)
...
orchestrator._register_candidate(...)
```

Using private methods (`_build_candidate_allocation`, `_register_candidate`) of `EvolverOrchestrator` creates tight coupling. If the orchestrator's internal API changes, the evolution loop breaks silently.

**Fix:** Either:
1. Make these methods public (remove underscore prefix)
2. Create a proper public interface on the orchestrator for organism evaluation
3. Add a dedicated `evaluate_organisms()` method to the orchestrator

---

### BUG-5: Organism/Candidate Duality Creates Confusion (MEDIUM)

**Problem:** Two parallel meta systems exist:
- `CandidateMeta` --- used by `EvolverOrchestrator` for single-generation runs
- `OrganismMeta` --- used by `EvolutionLoop` for multi-generation evolution

When `_evaluate_organisms` bridges between them, it treats organisms as candidates, which creates mapping issues:
- Organisms are stored in `org_*` directories, candidates in `cand_*` directories
- The orchestrator expects `cand_*` directories but receives organism dirs
- `load_best_context` searches for `cand_*/summary.json` and won't find organism summaries

**Result:** The orchestrator's context-loading for LLM prompts (`load_best_context`) only finds candidates from single-generation runs, not organisms from the evolution loop. This means the LLM gets no context about previous successful organisms during evolution.

**Fix:** Unify the meta systems. Either:
1. Make `OrganismMeta` extend `CandidateMeta`
2. Use a single meta type
3. Add organism-aware context loading to storage.py

---

### BUG-6: Mutation Prompt Promises Features Code Doesn't Deliver (HIGH)

**See Section 4.5** --- the prompt tells LLM it can add ideas, but `idea_dna_override` silently discards them.

---

### BUG-7: `score_weights` Config Is Silently Ignored (MEDIUM)

**See Section 4.8** --- config exists but code does `del scoring_cfg`.

---

## 6. Prompt Engineering Issues

### 6.1 Constant System Prompt Doesn't Exist as Described

**Spec says:**
> a) Константный промпт --- типо системный промпт, что погружает LLM просто в курс дела. Говорит что делать и как делать

The spec envisions a single, rich "constant" system prompt that fully immerses the LLM in the project context. Currently, there are 4 different system prompts:
- `seed_system.txt` --- for creating new organisms
- `mutation_system.txt` --- for mutating organisms
- `crossover_system.txt` --- for crossing organisms
- `optimizer_system.txt` --- for legacy candidate generation

These are operation-specific, not a single constant context prompt. The spec wants:
1. One constant context prompt in `./conf/system_prompt.txt` that's prepended to everything
2. Operation-specific instructions appended per-operation

**Fix:** Create `conf/system_prompt.txt` with comprehensive context about the project, optimizer design principles, the evolutionary process, etc. Then prepend it to all operation-specific prompts.

---

### 6.2 Seed Prompt Lacks Island/School Context

Current `seed_user.txt` is generic:
```
Design a novel optimizer for single-GPU PyTorch training experiments.
```

No mention of the island's research direction. When islands are implemented, this must include:
```
Your school of thought: {island_description}
Design an optimizer inspired by and aligned with this research direction.
```

---

### 6.3 Evolution Log Format in Prompts Lacks Impact

Current format (`organism.py:46`):
```
gen=0: Initial creation (score=None)
```

Should be punchier per spec:
> LLM должно подумать, что она хочет сделать сейчас, изменить описание гена b) и резюмировать изменения броской и хлесткой фразой в родословной c)

The format should be more like:
```
Generation 0 [score=0.72]: "Gradient-aware adaptive momentum with cosine phase switching"
Generation 1 [score=0.81]: "Added per-layer learning rate scaling --- 12% improvement"
Generation 2 [score=0.79]: "Replaced warmup with linear probe --- slight regression, exploring"
```

The prompt should explicitly ask the LLM to make the CHANGE_DESCRIPTION catchy and informative.

---

### 6.4 DNA Richness Not Enforced in Prompts

Current seed prompt:
```
## IDEA_DNA
A semicolon-separated list of core ideas (e.g., "adaptive gradient scaling; cosine momentum cycling")
```

This encourages terse labels. Should be:
```
## IDEA_DNA
A semicolon-separated list of genes. Each gene is a DETAILED description of one algorithmic principle:
what it does, why it helps, and how it interacts with other genes. Each gene must be rich enough that
a skilled engineer could implement it from the description alone.

Example of a single gene (note the detail level):
"Per-layer learning rate scaling via inverse gradient RMS: for each parameter group, compute the RMS
of recent gradients (EMA with decay=0.99) and scale the base learning rate inversely, clamped to
[0.1x, 10x] to prevent instability. Rationale: layers with larger gradients need smaller steps to
avoid overshooting, while layers with vanishing gradients need amplification."
```

---

### 6.5 Crossbreed Prompt Doesn't Explain What Happened

`crossbreed_user.txt` gives the LLM a TARGET IDEA DNA and asks it to implement. But it doesn't tell the LLM:
> "This is an organism after genetic crossbreeding. Genes were probabilistically selected from two parents."

The spec says:
> Мы отдаем этой особи родословную матери и подаем это все в LLM --- сообщаем ей, что это особь после слияния нескольких до этого и просим, чтобы она посмотрела на набор генов и придумала, что в этом коде нужно изменить, чтобы все заработало.

The prompt should explain that crossbreeding happened and let the LLM reason about coherence of the combined gene set.

---

## 7. Configuration Issues

### 7.1 Missing Island Configuration

Need to add:
```yaml
# conf/evolver/default.yaml
islands:
  dir: conf/islands         # directory with island .txt files
  organisms_per_island: 5   # initial population per island
  inter_island_rate: 0.1    # fraction of cross-breeding that happens between islands
```

### 7.2 Missing Separate top-k and top-h

Current:
```yaml
evolution:
  elite_count: 5
```

Need:
```yaml
evolution:
  simple_top_k: 5           # survivors per island after simple evaluation
  great_filter_top_h: 3     # survivors per island after Great Filter
```

### 7.3 Missing Softmax Temperature

Need:
```yaml
evolution:
  softmax_temperature: 1.0  # temperature for softmax parent selection
```

### 7.4 `eval_experiments` vs `simple_experiments`/`hard_experiments` Confusion

Top-level `eval_experiments: [cifar_convnet, minigpt_wikitext2]` exists alongside `evaluation.simple_experiments` and `evaluation.hard_experiments`. The orchestrator uses `eval_experiments`, the evolution loop uses `evaluation.*`. This duality is confusing.

**Fix:** Remove top-level `eval_experiments` or clarify its role. The evolution loop should exclusively use `evaluation.simple_experiments` and `evaluation.hard_experiments`.

### 7.5 `allocation` Block Duplicates `evaluation.simple_allocation`

Two allocation configs exist:
```yaml
evaluation:
  simple_allocation:
    enabled: true
    method: neyman
    sample_size: 2
    ...

allocation:
  enabled: true
  method: neyman
  sample_size: 2
  ...
```

The orchestrator uses `allocation` (top-level), the evolution loop should use `evaluation.simple_allocation`. But currently `_evaluate_organisms` creates an `EvolverOrchestrator` which reads `cfg.evolver.allocation` (top-level), ignoring the per-phase allocation configs.

---

## 8. Reward System Audit

### 8.1 Formula Verification

**Spec says:**
```
1) val_loss / last_species_loss
2) val_total_steps / first_step_when_loss_is_<=_than_val_loss
F1-score of these two
```

**Current implementation (`metrics_adapter.py:87-105`):**
```python
quality_ratio = baseline_last / max(raw_metric, eps)           # baseline / candidate
steps_ratio = baseline_total_steps / max(float(first_step), eps)  # baseline_steps / first_step
exp_score = (2.0 * quality_ratio * steps_ratio) / max(quality_ratio + steps_ratio, eps)  # F1
```

**Analysis:**
- The spec writes `val_loss / last_species_loss` (candidate/baseline), but the code computes `baseline / candidate`. These are inverses.
- The code's formulation makes both ratios "higher is better": if candidate loss < baseline loss, quality_ratio > 1 (good). If candidate reaches baseline faster, steps_ratio > 1 (good).
- The spec's formulation `val_loss / last_species_loss` would be "lower is better" for quality.
- The F1/harmonic mean only makes sense when both inputs go the same direction.
- **Conclusion:** The code's formulation is mathematically correct for producing an F1 where higher = better. The spec notation is likely informal. **No bug here**, but worth documenting.

### 8.2 Edge Case: Candidate Never Reaches Baseline

**Current (`metrics_adapter.py:93-95`):**
```python
if first_step is None or first_step <= 0:
    steps_ratio = 0.0
```

When `steps_ratio = 0`, the F1-score becomes `0.0` regardless of quality_ratio. This is by spec:
> If baseline not reached: speed_ratio = 0, exp_score = 0

**Correct.**

### 8.3 `objective_name` Must Be `train_loss`

**Current (`metrics_adapter.py:56-57`):**
```python
if status == "ok" and objective_name != "train_loss":
    status = "failed"
```

This hard-codes that only `train_loss` is a valid objective. Some experiments use `val_loss`, `val_acc`, etc. as their `primary_metric`. The scoring only works if the experiment's training loop reports `train_loss` as its objective via `TrainObjectiveTracker`.

**Potential issue:** If any experiment's training loop doesn't use `TrainObjectiveTracker` or uses a different objective name, it will always fail scoring. Need to verify all 17 experiments have been updated.

### 8.4 Baseline Validation: Hard-Coded to `min` Direction

**Current (`metrics_adapter.py:58-59`):**
```python
if status == "ok" and direction != "min":
    status = "failed"
```

This assumes all objectives are minimization (loss). But what about experiments with `primary_metric.direction: max` (like accuracy)? The scoring system only works for loss-based objectives.

**This is by design** (per agent report 4: "train-loss is now first-class top-level objective"). The objective tracking tracks train_loss regardless of the experiment's primary_metric. But this constraint should be documented.

---

## 9. Implementation Roadmap

### Phase 1: Fix Critical Bugs (estimate: 1-2 sessions)

1. **BUG-2: Evolution log scores** --- update scores after evaluation
   - Modify `_evaluate_organisms` in `evolution_loop.py`
   - Update `evolution_log.json` on disk after scoring

2. **BUG-1: Resume across generations** --- implement population manifest
   - Add `save_population_manifest()` and `load_population_manifest()` to `storage.py`
   - Update `_save_state` and `_load_state` in `evolution_loop.py`

3. **BUG-5: Organism context for LLM** --- add `load_best_organism_context()` to `storage.py`

### Phase 2: Fix Specification Violations (estimate: 1-2 sessions)

4. **4.2: Great Filter hard-only** --- one-line fix in `evolution_loop.py:322`
5. **4.1: Crossover lineage** --- change `crossbreeding.py:167` to use only dominant log
6. **4.3: Mutation uniform selection** --- update `selection.py`
7. **4.4: Separate top-k and top-h** --- add config, update selection calls
8. **4.5: Mutation DNA override** --- remove `idea_dna_override` for mutation, let LLM's DNA be used
9. **4.8: score_weights** --- either implement weighted F1 or remove config

### Phase 3: Island Model (estimate: 3-4 sessions)

10. **Create island infrastructure:**
    - `conf/islands/` directory with 3-5 initial island descriptions
    - `Island` dataclass in `types.py`
    - Island loader utility
    - `island_id` on `OrganismMeta`

11. **Per-island population management:**
    - Rewrite `EvolutionLoop.run()` to manage per-island populations
    - Per-island seeding with island-specific prompts
    - Per-island elite selection

12. **Breeding with islands:**
    - Intra-island cross-breeding (majority)
    - Inter-island cross-breeding (configurable rate)
    - Softmax selection within island for crossover parents

13. **Selection with islands:**
    - Per-island top-k after simple phase
    - Per-island top-h after Great Filter

### Phase 4: Prompt Engineering Overhaul (estimate: 1-2 sessions)

14. **Create constant system prompt** at `conf/system_prompt.txt`
15. **Revise DNA richness requirements** in all prompts
16. **Add genetic metaphor framing** to all prompts
17. **Revise evolution log format** for punchier descriptions
18. **Add island context** to seed prompts
19. **Improve crossbreed prompt** to explain what happened

### Phase 5: Configuration Cleanup (estimate: 1 session)

20. **Consolidate allocation configs** --- remove duplication
21. **Remove or clarify `eval_experiments`** vs `evaluation.*_experiments`
22. **Add all new config fields** (islands, top-k/top-h, softmax temperature)

---

## 10. Priority Matrix

| Priority | Issue | Type | Impact | Effort |
|---|---|---|---|---|
| P0 | Island model missing | Missing feature | Fundamental architecture gap | Large |
| P0 | Resume bug (BUG-1) | Bug | Data loss on restart | Medium |
| P0 | Evolution log scores (BUG-2) | Bug | LLM gets no feedback | Small |
| P1 | Softmax selection | Spec violation | Wrong evolutionary dynamics | Small |
| P1 | Crossover lineage (only mother) | Spec violation | Wrong inheritance model | Small |
| P1 | Great Filter hard-only | Spec violation | Wrong evaluation scope | Trivial |
| P1 | Mutation DNA override (BUG-6) | Bug + Spec violation | LLM creativity suppressed | Small |
| P1 | Separate top-k / top-h | Spec violation | Wrong selection pressure | Small |
| P2 | Prompt engineering overhaul | Prompts | Reduced LLM effectiveness | Medium |
| P2 | System prompt in conf | Spec violation | Wrong file organization | Small |
| P2 | Config cleanup | Config | Developer confusion | Small |
| P2 | Organism/Candidate duality (BUG-5) | Architecture | Fragile integration | Medium |
| P3 | score_weights ignored | Config bug | Misleading config | Trivial |
| P3 | Uniform mutation selection | Spec violation | Minor selection bias | Trivial |

---

## Appendix A: File Index

### Files requiring changes

| File | Changes needed |
|---|---|
| `src/evolve/evolution_loop.py` | Major rewrite: island model, per-island selection, resume fix |
| `src/evolve/selection.py` | Add softmax_select, uniform_select, per-island variants |
| `src/evolve/types.py` | Add `island_id` to OrganismMeta, add `Island` dataclass |
| `src/evolve/storage.py` | Add population manifest, organism context loading |
| `src/organisms/crossbreeding.py` | Fix lineage (line 167) |
| `src/organisms/mutation.py` | Remove idea_dna_override usage |
| `src/organisms/organism.py` | Score update in evolution log |
| `src/evolve/operators.py` | SeedOperator island-aware |
| `src/evolve/generator.py` | Load constant system prompt from conf |
| `src/evolve/metrics_adapter.py` | Remove `del scoring_cfg` or implement weighted F1 |
| `conf/evolver/default.yaml` | Add island, top-k/top-h, softmax_temperature configs |

### Files to create

| File | Purpose |
|---|---|
| `conf/system_prompt.txt` | Constant system prompt for all LLM operations |
| `conf/islands/gradient_methods.txt` | Island 1 school description |
| `conf/islands/second_order.txt` | Island 2 school description |
| `conf/islands/adaptive_methods.txt` | Island 3 school description |
| (user defines remaining islands) | |

### Files that are OK as-is

| File | Notes |
|---|---|
| `src/evolve/allocation.py` | Neyman allocation correct |
| `src/evolve/gpu_pool.py` | GPU pool working |
| `src/evolve/template_parser.py` | Template system solid |
| `src/evolve/scoring.py` | Aggregation logic correct |
| `src/evolve/orchestrator.py` | Single-gen orchestration OK |
| `valopt/utils/baselines.py` | Baseline loading correct |
| `valopt/utils/objective_tracking.py` | Objective tracking correct |
| All experiment configs | OK |
| All experiment implementations | OK |

---

*End of audit. All findings are based on code snapshot from 2026-03-26.*
