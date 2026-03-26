# 6. Revision: Merged Comprehensive Audit & Roadmap

**Date:** 2026-03-26
**Sources:** Claude Opus audit (agents/5_comprehensive_code_audit_and_roadmap.md) + GPT audit (agents/5_gpt_repository_audit_and_target_gap_analysis.md)
**Scope:** Unified canonical audit of the entire repository against the target specification for an evolutionary optimizer discovery platform.

---

## Table of Contents

1. [About This Document](#1-about-this-document)
2. [Executive Summary](#2-executive-summary)
3. [Canonical Target Model (What the System MUST Be)](#3-canonical-target-model)
4. [What Already Works Well](#4-what-already-works-well)
5. [CRITICAL: Missing Features](#5-critical-missing-features)
6. [Specification Violations](#6-specification-violations)
7. [Bugs & Defects](#7-bugs--defects)
8. [Prompt Engineering Issues](#8-prompt-engineering-issues)
9. [Configuration Issues](#9-configuration-issues)
10. [Reward System Audit](#10-reward-system-audit)
11. [Documentation & Contract Drift](#11-documentation--contract-drift)
12. [Product-Level Risks](#12-product-level-risks)
13. [Target Architecture After Refactoring](#13-target-architecture-after-refactoring)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Required Test Surface](#15-required-test-surface)
16. [Priority Matrix](#16-priority-matrix)
17. [File-by-File Change Program](#17-file-by-file-change-program)

---

## 1. About This Document

This is the **canonical handoff document** that merges findings from two independent audits:

- **Claude Opus audit** --- code-level defect analysis with specific file:line references and fix code snippets
- **GPT audit** --- architecture-level gap analysis with formal canonical model, product-level risks, and config truthfulness analysis

Both audits were conducted against the same codebase snapshot and the same specification (user's description of the evolutionary optimizer discovery platform inspired by Sakana AI's ShinkaEvolve and AdaptEvolve).

**Priority of truth sources:**

1. User's current specification (the message describing the platform)
2. Actual code and configs in the repository
3. Historical agent notes in `agents/*.md`
4. External research framing (ShinkaEvolve, AdaptEvolve papers)

**Key conclusion shared by both audits:** The repository has a strong foundation (validation runtime, baseline-relative reward, fixed template, GPU pool, 17 experiments). But it fundamentally does not yet implement the target system. The main problem is not missing features per se, but the **coexistence of several partially overlapping architectures** that together create a false sense of completeness.

---

## 2. Executive Summary

### What exists and works

| Component | Location | Status |
|---|---|---|
| Optimizer template with fixed signature | `src/evolve/templates/optimizer_template.txt` | OK |
| LLM code generation with retry | `src/evolve/generator.py` | OK |
| Idea DNA as semicolon-separated traits | `src/evolve/types.py:OrganismMeta.idea_dna` | Partial --- too shallow |
| Evolution log (lineage history) | `src/evolve/types.py:OrganismMeta.evolution_log` | Broken --- scores never backfilled |
| Probabilistic mutation (trait deletion) | `src/organisms/mutation.py` | Mechanically OK, wrong selection + LLM output discarded |
| Probabilistic crossbreeding (trait recombination) | `src/organisms/crossbreeding.py` | Mechanically OK, wrong selection + wrong lineage |
| Template parser & validator | `src/evolve/template_parser.py` | OK |
| Neyman allocation | `src/evolve/allocation.py` | OK |
| Baseline-relative scoring (F1/harmonic mean) | `src/evolve/metrics_adapter.py` | Mostly OK |
| GPU worker pool | `src/evolve/gpu_pool.py` | OK |
| Async orchestrator | `src/evolve/orchestrator.py` | OK for single-gen candidates |
| Multi-generation evolution loop | `src/evolve/evolution_loop.py` | Structurally incomplete |
| 17 experiment configs & implementations | `conf/experiments/*.yaml`, `experiments/*/` | OK |
| Train objective tracking | `valopt/utils/objective_tracking.py` | OK |
| Baseline profiles | `valopt/utils/baselines.py` | OK |
| Test suite | `tests/` | 49 tests passing, but product-level gaps |

### What is fundamentally missing or broken

| Issue | Severity | Both audits agree? |
|---|---|---|
| Island model completely absent | CRITICAL | Yes |
| Softmax parent selection for crossbreeding | CRITICAL | Yes |
| Per-island selection (top-k / top-h) | CRITICAL | Yes |
| Island-specific seeding with school prompts | CRITICAL | Yes |
| Evolution log scores never backfilled | CRITICAL BUG | Yes |
| Mutation/crossover silently discard LLM's DNA changes | CRITICAL BUG | Yes |
| `cand_*` vs `org_*` architecture duality | HIGH | Yes |
| Crossover merges both parents' lineage | HIGH | Yes |
| Great Filter uses simple+hard instead of hard-only | HIGH | Yes |
| Optimizer contract drift (README/tests vs runtime) | HIGH | GPT found, Claude confirmed |
| Resume breaks across generations | HIGH BUG | Claude found |
| Dead config knobs create false control | MEDIUM | GPT found |
| `score_weights` silently ignored | MEDIUM | Both found |
| Prompts live in code, not in conf | MEDIUM | Both found |

---

## 3. Canonical Target Model

*This section formalizes the user's specification into a reference model. All discrepancies in later sections are measured against this model.*

### 3.1 Core Entities

#### Organism (Особь)

An organism must contain:

- **Optimizer code** --- Python class implementing the fixed template signature
- **Genetic code (b)** --- rich textual description of the optimizer's principles, inspirations, and rationale. Must be detailed enough to reconstruct the optimizer from scratch. Stored as semicolon-separated "genes" that LLM treats as individually manipulable units
- **Lineage / evolution log (c)** --- chronological record of changes with scores. Follows the maternal line. Format: "key change => result score"
- **Island membership** --- which research school/direction this organism belongs to
- **Simple reward** --- permanent score from simple evaluation, stays forever
- **Hard reward** --- score from Great Filter (if applicable), used only for selection
- **Origin** --- operator (seed/mutation/crossover), parent IDs, maternal lineage

#### Genetic Code (b)

The genetic code is NOT a flat list of short labels. It must be:

- Rich enough to reconstruct the optimizer idea from the template alone
- Structured as semicolon-separated "genes", each a detailed description of one algorithmic principle
- Evolvable: LLM can add, remove, or modify individual genes
- The primary carrier of the optimizer's identity (more important than the code itself)

Example of a proper gene (the level of detail required):
```
Per-layer learning rate scaling via inverse gradient RMS: for each parameter group,
compute the RMS of recent gradients (EMA with decay=0.99) and scale the base learning
rate inversely, clamped to [0.1x, 10x] to prevent instability. Rationale: layers with
larger gradients need smaller steps to avoid overshooting, while layers with vanishing
gradients need amplification.
```

#### Lineage (c)

The lineage is a closed-loop record:

- `change_description` --- catchy, punchy phrase summarizing what changed
- `score` --- the actual evaluation result (backfilled after evaluation)
- Only the **mother's** (dominant parent's) lineage is inherited in crossover
- If full ancestry graph is needed, it should be stored separately

#### Island / School

An island is an isolated research direction:

- Defined by a `.txt` file in `conf/islands/`
- Has its own population
- Selection happens within each island independently
- Most breeding is intra-island; configurable inter-island crossover rate

#### Simple Phase

- Evaluate every new organism on simple experiments
- Use Neyman allocation to sample experiment subset
- Assign permanent `simple_reward`
- Select top-k organisms per island

#### Great Filter

- Runs every N generations
- Uses ONLY hard experiments
- Selects top-h organisms per island (h != k, separate config)
- Does NOT overwrite simple_reward
- Determines reproduction eligibility, not score identity

### 3.2 Canonical Lifecycle of One Organism

1. Born via seed (from island school prompt), mutation, or crossover
2. LLM receives: constant system prompt + genetic code + lineage + fixed template
3. LLM updates genetic code and generates optimizer code within fixed signature
4. Organism passes simple evaluation
5. Simple reward saved permanently; backfilled into lineage
6. If generation hits Great Filter, survivors pass hard evaluation
7. Great Filter affects reproduction rights but doesn't overwrite simple reward
8. After evaluation, lineage entry updated with score

### 3.3 Mandatory Invariants

- Initialization goes by schools/islands, not one generic seed prompt
- Schools defined by files in `conf/`
- Constant system prompt lives in `conf/`
- Fixed template does not drift or duplicate
- Mother/father semantics are real roles, not just "dominant/non-dominant by score"
- Mutation sampling uses uniform distribution (not tournament)
- Crossover parent sampling uses softmax over reward (not tournament)
- Lineage entries contain completed scores (not eternal `score=None`)
- All public descriptions of optimizer contract match the actual runtime contract

### 3.4 Breeding Operations

**Crossover:**
1. Select two parents from softmax(reward) distribution within island
2. First parent = mother (lineage keeper, gene probability p), second = father (probability 1-p)
3. Probabilistically recombine genes
4. Give mother's lineage to child
5. Tell LLM about crossover, ask it to reconcile genes and write coherent code
6. LLM may refine gene descriptions; its output DNA becomes canonical

**Mutation:**
1. Select parent from uniform distribution within island
2. Delete genes with probability q
3. Tell LLM what was removed; ask it to adapt, potentially add new ideas
4. LLM's output DNA becomes canonical (NOT overridden by pre-LLM state)

---

## 4. What Already Works Well

*Important to note so refactoring doesn't break these.*

### 4.1 Baseline-Relative Reward Path

`valopt/utils/baselines.py` + `src/validate/run_one.py` + `src/evolve/metrics_adapter.py` + `src/evolve/scoring.py`:

- Baseline profile loaded from canonical `stats/<experiment>/baseline.json`
- Runtime injects `baseline_last_train_loss` into experiment config
- Scoring uses `objective_last` and `first_step_at_or_below_baseline`
- Per-experiment score computed as harmonic mean (F1) of quality_ratio and steps_ratio

**Keep this.** It's the correct strategic foundation.

### 4.2 Fixed Template Rendering

`src/evolve/template_parser.py:11-60`:

- Template stored separately at `src/evolve/templates/optimizer_template.txt`
- Editable sections explicitly marked with `#===EDITABLE: NAME===`
- Code assembled from sections, not free-form generated

Very close to the spec's invariant: signature is fixed, LLM only changes allowed sections.

### 4.3 Validation Scaffold

`valopt/runner.py` + `experiments/*` provide a unified runtime for tasks of different complexity. Useful for both simple and hard evaluation phases.

### 4.4 GPU Pool & Async Orchestration

`src/evolve/gpu_pool.py` + `src/evolve/orchestrator.py`:

- Process-based worker pool with one worker per GPU
- Retry + timeout handling
- Async submit/collect pattern

This infrastructure should be retained as the evaluation backend.

### 4.5 Neyman Allocation

`src/evolve/allocation.py`:

- Per-experiment mean/std/cost normalization
- Proper Neyman allocation: pi_i proportional to std_i / sqrt(cost_i)
- Weighted sampling without replacement
- Fallback to uniform when insufficient history

Correctly implemented, no changes needed.

### 4.6 Local Test Coverage

49 passing tests covering:
- Allocation math
- Scoring math
- Optimizer import
- Structured generation helpers
- Mutation/crossbreeding basics

**Problem:** Tests cover local mechanics well but almost never test product-level semantics (see Section 15).

---

## 5. CRITICAL: Missing Features

### 5.1 Island Model --- Completely Absent

**Spec:** "Остров --- это группа особей, что разделяют своей собственное и неповторимое направление. Количество островов задается числом txt файлов в специально отведенной папке в ./conf."

**Current state:** No concept of islands exists anywhere:
- No `conf/islands/` directory
- No island data structure in `types.py`
- No per-island population tracking
- No island assignment for organisms
- No island-specific prompts

**Evidence:**
- `conf/` contains only YAML files (no `.txt` school descriptions)
- `src/evolve/evolution_loop.py:105-127` initializes population via generic `SeedOperator()` without island semantics
- `src/evolve/operators.py:51-57` seed generation uses one-size-fits-all `seed_system.txt`

**Impact:** Without islands, the system cannot implement the core search topology: multiple research directions competing and evolving independently. All search happens in one undifferentiated population.

**What needs to be built:**

New directory structure:
```
conf/
  islands/
    gradient_methods.txt      # "School of gradient-based optimization"
    second_order.txt          # "School of curvature-aware methods"
    adaptive_methods.txt      # "School of adaptive learning rate ideas"
    ...
```

New data structures in `src/evolve/types.py`:
```python
@dataclass(slots=True)
class IslandConfig:
    island_id: str
    name: str
    description: str  # loaded from .txt file
    description_path: str

@dataclass(slots=True)
class IslandState:
    island_id: str
    population: list[OrganismMeta]
```

Add to `OrganismMeta`:
```python
island_id: str  # which island this organism belongs to
```

New config in `conf/evolver/default.yaml`:
```yaml
islands:
  dir: conf/islands
  organisms_per_island: 5
  inter_island_crossover_rate: 0.1
```

**Files to modify:** `types.py`, `evolution_loop.py` (major rewrite), `selection.py`, `operators.py`, `storage.py`, `conf/evolver/default.yaml`

**Estimated scope:** ~500-700 lines of new/modified code.

---

### 5.2 Softmax Parent Selection for Crossbreeding

**Spec:** "cross breading --- мы выбираем из распределения softmax над reward этих моделей две особи"

**Current code (`src/evolve/selection.py:62-74`):**
```python
parent_a = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
parent_b = tournament_select(survivors, k=1, tournament_size=tournament_size, rng=rng)[0]
```

Tournament selection has fundamentally different search dynamics than softmax:
- Tournament creates strong exploitation pressure (best K always have high probability)
- Softmax is smoother --- weak organisms still have non-zero probability proportional to exp(score)
- The exploration/exploitation balance is completely different

**Fix --- add to `selection.py`:**
```python
import math

def softmax_select(
    population: list[OrganismMeta],
    k: int = 1,
    score_key: str = "simple_score",
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Sample k organisms with probability proportional to softmax(score/temperature)."""
    rng = rng or random.Random()
    scores = [getattr(org, score_key) or 0.0 for org in population]
    max_s = max(scores) if scores else 0.0
    exp_scores = [math.exp((s - max_s) / max(temperature, 1e-8)) for s in scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]
    return rng.choices(population, weights=probs, k=k)
```

**Config addition:**
```yaml
evolution:
  softmax_temperature: 1.0
```

---

### 5.3 Uniform Mutation Parent Selection

**Spec:** "mutation --- мы так же случайным образом уже из равномерного распределения семплируем несколько особей"

**Current:** Mutation also uses tournament selection.

**Fix --- add to `selection.py`:**
```python
def uniform_select(
    population: list[OrganismMeta],
    k: int = 1,
    rng: random.Random | None = None,
) -> list[OrganismMeta]:
    """Sample k organisms uniformly at random."""
    rng = rng or random.Random()
    return rng.choices(population, k=k)
```

**Update `select_parents_for_reproduction`** to use `softmax_select` for crossover and `uniform_select` for mutation.

---

### 5.4 Per-Island Selection with Separate top-k and top-h

**Spec:**
- Simple phase: "ранжируем наши все особи на всех островах по отдельности и оставляем только top-k"
- Great Filter: "ранжируем все особи на каждом острове по отдельности и оставляем другое число top-h"

**Current:** Single flat `elite_select` over entire population using single `elite_count: 5`.

**Fix:** Two separate config values:
```yaml
evolution:
  simple_top_k: 5      # survivors per island after simple evaluation
  great_filter_top_h: 3  # survivors per island after Great Filter
```

Selection logic must loop over islands:
```python
for island in islands:
    island_pop = [org for org in population if org.island_id == island.island_id]
    survivors = elite_select(island_pop, top_k, score_key="simple_score")
    # ... keep only survivors
```

---

### 5.5 Island-Specific Seeding with School Prompts

**Spec:** "найдет, сколько особей каждой школы для инициализации стоит придумать, далее идет в API и со специально подготовленным системным промптом плюс текстом из txt просит LLM придумать один генетический код"

**Current:** `SeedOperator` uses generic `seed_system.txt` / `seed_user.txt` with no island context.

**Fix:**
- `SeedOperator.build_prompts()` must accept island description text
- System prompt includes island/school context
- User prompt references the school's research direction

---

### 5.6 System Prompt Location

**Spec:** "a) --- константная вещь, что должна храниться в ./conf директории, как текстовый файл"

**Current:** All prompts live in `src/evolve/prompts/`.

**Fix:** Create `conf/system_prompt.txt` with comprehensive context. `OptimizerGenerator` prepends it to all LLM calls. Operation-specific prompts can stay in `src/evolve/prompts/` as they're code-coupled, but the constant context moves to `conf/`.

---

## 6. Specification Violations

### 6.1 Crossover Merges Both Parents' Evolution Logs --- Should Only Keep Mother's

**Spec:** "В частности, в любом размножении у нас есть главная особь (типо мать) --- хранитель родословной"

**Current code (`src/organisms/crossbreeding.py:167-169`):**
```python
merged_log = list(dominant.evolution_log) + list(non_dominant.evolution_log)
merged_log.sort(key=lambda e: e.get("generation", 0))
```

Merges BOTH parents' logs, creating a combined history that doesn't match either parent's lineage.

Additionally, the comment says "keep unique" but no deduplication is actually performed.

**Fix:**
```python
# Only inherit the dominant parent's (mother's) lineage
parent_evolution_log = list(dominant.evolution_log)
```

**Location:** `src/organisms/crossbreeding.py:167-169`

**Additional recommendation (from GPT):** Add explicit fields:
- `mother_id` and `father_id` (instead of just `parent_ids` list)
- `lineage_source = "mother"` to make the inheritance explicit

---

### 6.2 Great Filter Evaluates on Simple + Hard --- Should Be Hard Only

**Spec:** "Сложная фаза или 'Великий фильтр' --- мы прогоняем каждую дожившую особь через сложный алгоритм отбора по каждому сложному алгоритму и усредняем результат"

**Current code (`src/evolve/evolution_loop.py:322-324`):**
```python
all_experiments = self.simple_experiments + self.hard_experiments
await self._evaluate_organisms(
    self.population, all_experiments, score_key="hard_score"
)
```

**Fix:** One-line change:
```python
await self._evaluate_organisms(
    self.population, self.hard_experiments, score_key="hard_score"
)
```

---

### 6.3 LLM's DNA Changes Silently Discarded in Both Mutation and Crossover

**Spec:** "LLM должно подумать, что она хочет сделать сейчас, изменить описание гена b)"

**This is a critical semantic bug confirmed by both audits.**

**In mutation (`src/organisms/mutation.py:155-156`):**
```python
idea_dna_override=child_dna,  # discards LLM's IDEA_DNA output
```

Meanwhile `src/evolve/prompts/mutate_user.txt:9` tells LLM:
> You MAY add ONE new idea to replace the removed ones

The prompt promises the ability to evolve DNA, but the code overrides it.

**In crossover (`src/organisms/crossbreeding.py:173-185`):**
```python
idea_dna_override=child_dna,  # same problem
```

**The root cause** is in `src/organisms/organism.py:85-86`:
```python
if idea_dna_override is not None:
    idea_dna = list(idea_dna_override)
```

When `idea_dna_override` is provided, the LLM's `## IDEA_DNA` section output is completely ignored.

**Fix --- two-phase approach:**

Phase 1 (probabilistic): Mutation deletes traits, crossover recombines traits. This determines the *input* to the LLM --- what the LLM sees as the "current state".

Phase 2 (LLM): The LLM generates new DNA based on the input. **The LLM's output DNA becomes the canonical child DNA.**

Implementation: For mutation, remove `idea_dna_override` parameter. For crossover, pass `idea_dna_override` only as a *suggestion* to the LLM (in the prompt), not as an override of the output.

---

### 6.4 `score = max(old_score, new_score)` Makes Score Semantically Unstable

**(GPT finding, confirmed by code analysis)**

**Current code (`src/evolve/evolution_loop.py:232-234`):**
The `org.score` field is updated as `max(old, new)` across phases. This means:

- After simple evaluation: `score = simple_score`
- After Great Filter: `score = max(simple_score, hard_score)`

This creates a semantically ambiguous scalar. When crossover uses `org.score` to determine the "dominant" parent, it may compare organisms measured on different scales.

**Fix:** Never use ambiguous `score` in core logic. Use explicit fields:
- `simple_reward` --- for softmax parent selection in crossover
- `hard_reward` --- for Great Filter ranking
- `selection_reward` --- derived, explicit about what it represents

---

### 6.5 No Separate top-k and top-h Counts

**Spec:** Simple phase keeps top-k, Great Filter keeps top-h (different numbers).

**Current:** Both use single `elite_count: 5`.

**Fix:** Replace with:
```yaml
evolution:
  simple_top_k: 5
  great_filter_top_h: 3
```

---

### 6.6 Crossover Parent Roles Are Score-Based, Not Mother/Father

**Current code (`src/evolve/evolution_loop.py:157-160`):**
The "dominant" parent is determined by score comparison, not by explicit mother/father assignment.

**Spec:** Mother and father are explicit roles. Mother is the lineage keeper (gene probability p). Father provides genetic material (probability 1-p).

**Fix:** In `select_parents_for_reproduction`, explicitly designate first parent as mother (softmax-selected) and second as father. The mother's lineage is inherited. The dominant/non-dominant naming should map to mother/father, not to higher/lower score.

---

### 6.7 Mutation Uses Tournament Selection Instead of Uniform

Already covered in Section 5.3.

---

### 6.8 Genetic Code (idea_dna) Is Too Shallow

**Spec:** "b) часть должна быть настолько богатой, что по ней можно восстановить идею для реализации оптимизатора из шаблона d)"

**Current:** Mock DNA looks like: `"mock SGD with cosine schedule; warmup phase"`. This is a flat list of short labels.

**GPT's recommendation for richer schema:**
- Gene ID
- Gene text (detailed paragraph)
- Provenance (which ancestor introduced this gene)
- Status: `kept/modified/removed/new`
- Rationale blob

**Minimal pragmatic approach:** Keep semicolon-separated format but enforce richness via prompts. Each gene should be a multi-sentence description, not a label.

---

## 7. Bugs & Defects

### BUG-1: Resume Breaks Across Generations (CRITICAL)

**(Claude finding --- not in GPT report)**

**Location:** `src/evolve/evolution_loop.py` + `src/evolve/storage.py:147-159`

**Problem:** When the evolution loop resumes, it calls:
```python
pop_dicts = load_population(self.population_root, self.generation)
```

But `load_population` only scans `gen_{N}/org_*`:
```python
gen_dir = root / f"gen_{generation}"
for org_path in sorted(gen_dir.glob("org_*")):
```

If an organism from generation 0 survives to generation 5 (via elite selection), its files remain in `gen_0/org_XXX`. When resuming at generation 5, `load_population(root, 5)` **will NOT find it** because it only scans `gen_5/org_*`.

**Result:** On resume, all surviving organisms from previous generations are silently lost.

**Fix:** Implement a population manifest file:

Add `save_population_manifest()` and `load_population_manifest()` to `storage.py`:
```python
def save_population_manifest(population_root: Path, generation: int, organisms: list[dict]) -> Path:
    """Save manifest of active organisms regardless of their origin generation."""
    manifest = {
        "generation": generation,
        "organisms": [
            {
                "organism_id": org["organism_id"],
                "organism_dir": org["organism_dir"],
                "island_id": org.get("island_id"),
                "simple_score": org.get("simple_score"),
                "hard_score": org.get("hard_score"),
            }
            for org in organisms
        ]
    }
    return write_json(Path(population_root) / "population_manifest.json", manifest)
```

---

### BUG-2: Evolution Log Scores Never Updated (CRITICAL)

**(Both audits found this)**

**Location:** `src/organisms/organism.py:125-130`

When a new organism is created, its evolution log entry has `score=None`:
```python
new_entry = EvolutionEntry(
    generation=generation,
    change_description=change_description,
    score=None,  # always None at creation time
    parent_ids=parent_ids,
)
```

After evaluation, only `OrganismMeta.score`/`simple_score`/`hard_score` are updated, but the `evolution_log` entry is NEVER updated with the actual score.

**Result:** Children always see `score=None` in their inherited lineage:
```
gen=0: Initial creation (score=None)
gen=1: Added momentum cycling (score=None)
gen=2: Switched to cosine annealing (score=None)
```

This completely defeats the purpose of the lineage --- the LLM cannot learn which changes helped or hurt.

**Fix:** After evaluation, update the last evolution log entry:
```python
# In _evaluate_organisms, after setting org.simple_score:
if org.evolution_log:
    org.evolution_log[-1]["score"] = org.simple_score
    # Persist updated evolution_log.json to disk
    write_json(evolution_log_path(Path(org.organism_dir)), org.evolution_log)
```

**GPT's additional recommendation:** Add richer fields to `LineageEntry`:
```python
@dataclass
class LineageEntry:
    generation: int
    change_description: str
    gene_diff_summary: str
    simple_score: float | None
    hard_score: float | None
    operator: str
    parent_ids: list[str]
```

---

### BUG-3: `_evaluate_organisms()` Uses Private Orchestrator Methods (MEDIUM)

**Location:** `src/evolve/evolution_loop.py:213`

```python
allocation_snapshot = orchestrator._build_candidate_allocation(org.organism_id)
orchestrator._register_candidate(...)
```

Using private methods (`_build_candidate_allocation`, `_register_candidate`) of `EvolverOrchestrator` creates tight coupling. If the orchestrator's internal API changes, the evolution loop breaks silently.

**GPT note:** The `_evaluate_organisms()` method even has comments acknowledging it's a "placeholder" / "For now" solution --- a clear sign of unfinished migration.

**Fix:** Either make these methods public (remove underscore prefix) or create a dedicated `evaluate_organisms()` public method on the orchestrator.

---

### BUG-4: Organism/Candidate Duality Creates Confusion (HIGH)

**(Both audits found this --- GPT calls it "the main structural smell")**

Two parallel meta systems exist:
- `CandidateMeta` + `cand_*` directories --- used by `EvolverOrchestrator` for single-generation runs
- `OrganismMeta` + `org_*` directories --- used by `EvolutionLoop` for multi-generation evolution

**Evidence:**
- `src/evolve/storage.py:63-99` defines `cand_*` layout
- `src/evolve/storage.py:128-159` defines `org_*` layout
- `src/evolve/run.py:29-45` defaults to `EvolutionLoop`
- `README.md:146-171` documents `cand_*` as primary output

**Practical problems:**
- `load_best_context()` in `storage.py:183-200` searches for `cand_*/summary.json` --- it won't find organism summaries. So during evolution, the LLM gets no context about previous successful organisms.
- It's unclear which execution path is "production"
- New developers can't tell what an "entity" is in the system

**Fix:** Canonicalize organism-first path. Repurpose `EvolverOrchestrator` as evaluation-only backend. Remove `cand_*` as an alternative entity model.

---

### BUG-5: Crossover Comment Claims "Keep Unique" But Deduplication Not Implemented

**(GPT finding)**

**Location:** `src/organisms/crossbreeding.py:166-169`

```python
merged_log = list(dominant.evolution_log) + list(non_dominant.evolution_log)
# Sort by generation, keep unique  <-- comment says "keep unique"
merged_log.sort(key=lambda e: e.get("generation", 0))
# No deduplication actually performed!
```

The code sorts but never deduplicates. Minor bug, but compounds the lineage merge problem.

---

### BUG-6: `_save_state` Could Produce Misleading Best Score

**Location:** `src/evolve/evolution_loop.py:86-93`

```python
"best_score": max(
    (org.score for org in self.population if org.score is not None),
    default=None,
),
```

While this won't crash (due to `default=None`), the `best_score` mixes simple and hard phase scores (because `org.score` is semantically unstable --- see Section 6.4). This means the saved "best score" may not be comparable across generations.

---

## 8. Prompt Engineering Issues

### 8.1 Constant System Prompt Doesn't Exist as Described

**Spec describes four layers:**
- a) Constant system prompt (in `conf/`) --- immerses LLM in context
- b) Genetic code --- the evolvable DNA
- c) Evolution log --- lineage with scores
- d) Template --- fixed optimizer signature

**Current:** There are 4 different system prompts scattered in `src/evolve/prompts/`:
- `seed_system.txt`, `mutation_system.txt`, `crossover_system.txt`, `optimizer_system.txt`

These are operation-specific, not a single constant context prompt.

**Fix:** Create `conf/system_prompt.txt` that comprehensively describes:
- What the project is about
- What an optimizer is in this context
- How the genetic code works
- What the LLM's role is
- How to think about optimizer design principles

Then prepend this to all operation-specific prompts.

---

### 8.2 Seed Prompt Lacks Island/School Context

Current `seed_user.txt`:
```
Design a novel optimizer for single-GPU PyTorch training experiments.
```

Generic, no mention of island's research direction.

**Fix after islands are implemented:**
```
Your research school: {island_description}

Design an optimizer inspired by and aligned with this research direction.
The optimizer's genetic code (idea DNA) should reflect the school's principles.
```

---

### 8.3 Prompts Don't Frame Genes as Individually Manipulable Units

**Spec:** "LLM должна воспринимать этот генетический код (буквально тексты разделенные ;) как список генов, и работать с ним он должен соответствующе"

Current prompts describe idea_dna as "core ideas" but don't explicitly frame them as genetic units.

**Fix:** Add to all system prompts:
```
Your IDEA_DNA is a genetic code --- a semicolon-separated list of genes. Each gene
describes one algorithmic principle in detail. You should think of these as biological
genes:
- Each gene can be independently added, removed, or mutated
- The DNA is the complete blueprint from which the optimizer is reconstructed
- When evolving, consider which genes contributed to improvements and which didn't
- Every gene should be detailed enough that a skilled engineer could implement it
```

---

### 8.4 Evolution Log Format Lacks Impact

Current format (`organism.py:46`):
```
gen=0: Initial creation (score=None)
```

**Spec demands:** "резюмировать изменения броской и хлесткой фразой в родословной"

**Fix --- update prompt instructions:**
```
## CHANGE_DESCRIPTION
Write a CATCHY, MEMORABLE phrase summarizing your key change. Think of it as a
headline for a research paper. Examples:
- "Replaced warmup with gradient-adaptive scaling --- 15% convergence speedup"
- "Introduced dual-phase momentum: heavy for exploration, light for convergence"
```

**Fix --- update log format:**
```
Generation 0 [score=0.72]: "Gradient-aware adaptive momentum with cosine phase switching"
Generation 1 [score=0.81]: "Added per-layer learning rate scaling --- 12% improvement"
Generation 2 [score=0.79]: "Replaced warmup with linear probe --- slight regression"
```

---

### 8.5 DNA Richness Not Enforced

Current seed prompt:
```
## IDEA_DNA
A semicolon-separated list of core ideas (e.g., "adaptive gradient scaling; cosine momentum cycling")
```

Encourages terse labels. Should be:
```
## IDEA_DNA
A semicolon-separated list of genes. Each gene is a DETAILED description (2-4 sentences)
of one algorithmic principle: what it does, why it helps, and how it interacts with
other genes. Each gene must be rich enough that a skilled engineer could implement it
from the description alone, without seeing the code.
```

---

### 8.6 Crossbreed Prompt Doesn't Explain the Breeding Context

`crossbreed_user.txt` gives a TARGET IDEA DNA and asks the LLM to implement it, but doesn't say:
> "This organism was created by genetic crossbreeding. Genes were probabilistically selected from two parents."

The spec says the LLM should be told about the crossbreeding event and asked to make the combined gene set coherent.

---

### 8.7 Prompt Duplication Creates Drift Risk

**(GPT finding)**

In `src/evolve/prompts/` there are duplicate pairs:
- `mutation_user.txt` AND `mutate_user.txt`
- `crossover_user.txt` AND `crossbreed_user.txt`

Legacy operators (`operators.py`) use one set; probabilistic operators (`mutation.py`, `crossbreeding.py`) use the other. This will inevitably lead to behavioral drift.

**Fix:** After canonicalizing one execution model, delete legacy prompts and fix naming convention.

---

### 8.8 Lineage Context Truncated to 5 Entries

**(GPT finding)**

`src/organisms/organism.py:24`: `_MAX_EVOLUTION_LOG_IN_PROMPT = 5`

If the lineage is supposed to be the primary carrier of research history, 5 entries may lose important early turning points.

**Fix:** Implement a richer summarization layer:
- Condensed lineage summary (covering all entries)
- Last N raw entries
- Explicit best/worst edits marked

---

## 9. Configuration Issues

### 9.1 Missing Island Configuration

```yaml
# Needed in conf/evolver/default.yaml
islands:
  dir: conf/islands
  organisms_per_island: 5
  inter_island_crossover_rate: 0.1
```

---

### 9.2 Missing Separate top-k and top-h

Current: `elite_count: 5`

Need:
```yaml
evolution:
  simple_top_k: 5
  great_filter_top_h: 3
```

---

### 9.3 Missing Softmax Temperature

```yaml
evolution:
  softmax_temperature: 1.0
```

---

### 9.4 `eval_experiments` vs `evaluation.*_experiments` Confusion

Top-level `eval_experiments: [cifar_convnet, minigpt_wikitext2]` coexists with `evaluation.simple_experiments` and `evaluation.hard_experiments`.

The orchestrator uses `eval_experiments`; the evolution loop uses `evaluation.*`. This creates confusion.

**Fix:** Remove top-level `eval_experiments` or clarify its role. Evolution loop should exclusively use `evaluation.simple_experiments` and `evaluation.hard_experiments`.

---

### 9.5 `simple_allocation` / `hard_allocation` Configs Declared But Not Connected

**(Both audits found this)**

Config declares:
```yaml
evaluation:
  simple_allocation:
    enabled: true
    method: neyman
    sample_size: 2
  hard_allocation:
    enabled: false
```

But the evaluation pipeline uses only `evolver.allocation` (top-level). The per-phase allocation configs are dead.

**Evidence:**
- `EvolverOrchestrator` reads `cfg.evolver.get("allocation", {})`: `orchestrator.py:44-51`
- `_evaluate_organisms()` doesn't switch allocation config between phases

**Fix:** Pass separate allocation configs to simple and hard phases, or delete the dead keys.

---

### 9.6 `score_weights` Silently Ignored

**(Both audits found this)**

Config: `allocation.score_weights: {quality: 0.7, steps: 0.3}`

Code (`metrics_adapter.py:35`): `del scoring_cfg` --- weights are explicitly discarded.

Harmonic mean (F1) is used regardless. The config creates false impression that quality/steps weighting is customizable.

**Fix:** Either implement weighted harmonic mean or remove `score_weights` from config.

---

### 9.7 Dead Config Knobs Create False Control

**(GPT finding)**

Several config keys have no runtime implementation:

| Config Key | File | Used in Code? |
|---|---|---|
| `selection_strategy: uniform` | `default.yaml:4` | NO --- not referenced anywhere |
| `crossover_rate: 0.3` | `default.yaml:23` | NO --- branching uses only `mutation_rate` |
| `novelty_filter.enabled: false` | `default.yaml:59-60` | NO --- no runtime references |
| `bandit_llm_selection.enabled: false` | `default.yaml:62-63` | NO --- no runtime references |

**Evidence:** `grep -r` across the codebase shows no usage of `selection_strategy`, `crossover_rate` (in actual branching logic), `novelty_filter`, or `bandit_llm_selection`.

**Impact:** An experimenter could change these values thinking they're tuning the system. Nothing would change.

**Fix:** Either implement them or remove them. For a research platform, ghost controls are dangerous --- they lead to incorrect conclusions about what was tested.

---

### 9.8 Legacy `quality_ref` / `steps_ref` Still in Experiment Configs

**(GPT finding)**

After the reward rework (documented in `agents/4_train_loss_baseline_reward_followup.md`), scoring is now baseline-driven. But experiment configs still contain `normalization.quality_ref` and `normalization.steps_ref`, and the README documents them as active.

**Fix:** Cleanup pass: remove from configs, update README, add test assertion that scoring doesn't depend on these values.

---

## 10. Reward System Audit

### 10.1 Formula Verification

**Spec:**
```
1) val_loss / last_species_loss
2) val_total_steps / first_step_when_loss_is_<=_than_val_loss
F1-score of these two
```

**Implementation (`metrics_adapter.py:87-105`):**
```python
quality_ratio = baseline_last / max(raw_metric, eps)
steps_ratio = baseline_total_steps / max(float(first_step), eps)
exp_score = (2.0 * quality_ratio * steps_ratio) / max(quality_ratio + steps_ratio, eps)
```

**Analysis:**
- The spec writes `val_loss / last_species_loss` (candidate/baseline = lower is better)
- The code computes `baseline / candidate` (higher is better)
- These are inverses, but the code's formulation is correct for F1: both ratios are "higher = better", and harmonic mean produces a single "higher = better" scalar
- **No bug here** --- the spec's notation is informal, the code's math is sound

### 10.2 Edge Case: Candidate Never Reaches Baseline

```python
if first_step is None or first_step <= 0:
    steps_ratio = 0.0  # -> exp_score = 0.0
```

When candidate never reaches baseline loss, `steps_ratio = 0`, so `exp_score = 0`. Correct by spec.

### 10.3 Hard-Coded `train_loss` Constraint

```python
if status == "ok" and objective_name != "train_loss":
    status = "failed"
if status == "ok" and direction != "min":
    status = "failed"
```

Only `train_loss` (minimization) is accepted. This is by design per the reward rework, but:
- Not all experiments may report `train_loss` as their objective
- Experiments with `primary_metric.direction: max` (accuracy) will always fail scoring

**Recommendation:** This is correct but should be documented. Verify all 17 experiments report `train_loss` via `TrainObjectiveTracker`.

### 10.4 Baseline Validation

The baseline validation pipeline (`valopt/utils/baselines.py`) correctly:
- Requires `objective_name == "train_loss"` and `objective_direction == "min"`
- Validates finite `objective_last`
- Requires `steps > 0`

**This is solid.** No changes needed.

---

## 11. Documentation & Contract Drift

### 11.1 Optimizer Contract Inconsistency

**(GPT finding --- critical)**

**README** documents old contract:
- `build_optimizer(cfg)`
- `initialize(named_parameters, cfg)`
- `step(weights, grads, activations)` --- no `step_fn`!

**Runtime** uses new contract:
- `build_optimizer(model, max_steps)` --- different signature!
- `step(weights, grads, activations, step_fn)` --- has `step_fn`!
- `zero_grad(set_to_none=True)` --- not in README!

**Test** (`tests/test_optimizer_generator.py:31-50`) validates OLD contract:
- `build_optimizer(cfg)` accepted as valid
- `initialize(...)` expected
- `step(... without step_fn)` expected

**Evidence:**
- `README.md:126-145` (old contract)
- `valopt/optimizer_api.py:18-36` (new contract)
- `valopt/utils/import_utils.py:51-71` (validates new contract at import time)
- `src/validate/run_one.py:107-112` (uses new contract at runtime)

**This is one of the most dangerous drift-ons.** Documentation lies to users, tests validate the wrong interface.

**Fix:** Single pass to align: README, runtime validator, generator validator, prompts, tests, example optimizers.

---

### 11.2 Two Validators with Different Strictness

**(GPT finding)**

**`OptimizerGenerator._validate_code()`** (generator.py:80-109):
- Checks for `build_optimizer` function
- Checks for class with `step` and `zero_grad`
- Does NOT check `__init__` method
- Does NOT check `step_fn` parameter
- Does NOT check `build_optimizer(model, max_steps)` signature

**`validate_rendered_code()`** (template_parser.py:63-94):
- Checks for `build_optimizer` function
- Checks for class with `__init__`, `step`, AND `zero_grad`
- Stricter than the generator validator

**Result:** Code passing `_validate_code()` may fail `validate_rendered_code()`, creating inconsistent behavior between candidate and organism generation paths.

**Fix:** Create one shared validator module used everywhere. Should check:
- `build_optimizer(model, max_steps)` function exists with correct params
- Controller class with `__init__(self, model, max_steps)`, `step(self, weights, grads, activations, step_fn)`, `zero_grad(self, set_to_none)`

---

### 11.3 README Documents `cand_*` as Primary Output

`README.md` describes `cand_*` outputs as the canonical evolve result, but the default execution path (`src/evolve/run.py:36-45`) routes to `EvolutionLoop` which produces `org_*` outputs.

---

## 12. Product-Level Risks

**(GPT's risk analysis --- critical for a scientific platform)**

### 12.1 False Evolution of Genetic Code

The system looks like it's evolving gene descriptions, but mutation/crossover don't actually save LLM-updated DNA as canonical truth (see BUG in Section 6.3). An experimenter could run 100 generations and think the genetic code evolved, when in reality the DNA was mechanically recombined without LLM refinement.

### 12.2 False Control via Config

An experimenter can change `simple_allocation`, `hard_allocation`, `selection_strategy`, `crossover_rate`, `novelty_filter`, `bandit_llm_selection`, `score_weights` --- and think they're controlling the system. In many cases, the code ignores these values entirely (Section 9.7).

This is especially dangerous for a research platform: it leads to incorrect conclusions about what experimental conditions were tested.

### 12.3 False Confidence from Tests

The test suite passes (49 tests), giving the impression of correctness. But the tests almost never catch:
- Product drift
- Dead config
- Island absence
- Lineage score backfill absence
- Contract inconsistency between README/tests/runtime
- Hybrid `cand_*` vs `org_*` architecture

### 12.4 False Score Interpretation

`org.score` can mix simple and hard phase semantics (Section 6.4). Decisions based on this semantically heterogeneous scalar may be scientifically invalid.

### 12.5 False Research Diversity

Without islands, all organisms evolve in one undifferentiated population. There's no guarantee of exploring diverse research directions. The system may converge prematurely to a local optimum.

---

## 13. Target Architecture After Refactoring

### 13.1 Canonical Config Layout

```text
conf/
  config.yaml                    # main Hydra config
  system_prompt.txt              # constant system prompt (immersive context)
  evolver/
    default.yaml                 # evolution parameters
  prompts/
    seed_user.txt                # seed operation user prompt template
    mutation_user.txt            # mutation user prompt template
    crossover_user.txt           # crossover user prompt template
  islands/
    gradient_methods.txt         # school 1
    second_order.txt             # school 2
    adaptive_methods.txt         # school 3
  experiments/
    *.yaml                       # experiment configs
```

### 13.2 Canonical Config (evolver/default.yaml)

```yaml
enabled: true

islands:
  dir: conf/islands
  organisms_per_island: 5
  inter_island_crossover_rate: 0.1

evolution:
  max_generations: 100
  simple_top_k: 5
  great_filter_top_h: 3
  great_filter_interval: 5
  softmax_temperature: 1.0
  crossover_p: 0.7       # mother gene probability
  mutation_q: 0.2         # gene deletion probability
  mutation_rate: 0.7      # fraction of offspring from mutation vs crossover

evaluation:
  simple_experiments: [...]
  hard_experiments: [...]
  simple_allocation:
    enabled: true
    method: neyman
    sample_size: 2
  hard_allocation:
    enabled: false

llm:
  provider: chatgpt
  model: gpt-5.4-pro
  temperature: 0.5
  max_output_tokens: 8000
```

### 13.3 Canonical Organism Schema

```python
@dataclass(slots=True)
class OrganismMeta:
    organism_id: str
    island_id: str
    generation: int
    timestamp: str
    mother_id: str | None       # explicit maternal parent
    father_id: str | None       # explicit paternal parent (for crossover)
    operator: str               # "seed" | "mutation" | "crossover"
    idea_dna: list[str]         # rich semicolon-separated genes
    evolution_log: list[dict]   # lineage with backfilled scores
    model_name: str
    prompt_hash: str
    seed: int
    organism_dir: str
    optimizer_path: str
    simple_score: float | None  # permanent, from simple evaluation
    hard_score: float | None    # from Great Filter (if applicable)
    status: str                 # "pending" | "evaluated" | "eliminated"
```

```python
@dataclass(slots=True)
class LineageEntry:
    generation: int
    change_description: str     # catchy phrase
    gene_diff_summary: str      # what genes changed
    simple_score: float | None  # backfilled after evaluation
    hard_score: float | None    # backfilled after Great Filter
    operator: str
    parent_ids: list[str]
```

### 13.4 Canonical Execution Flow

1. Load island configs from `conf/islands/*.txt`
2. Load constant system prompt from `conf/system_prompt.txt`
3. Seed per-island population (N organisms per island, each with school-specific prompt)
4. For each generation:
   a. Simple evaluation for all new organisms (Neyman allocation on simple_experiments)
   b. Backfill simple_score into lineage entries
   c. Select top-k per island
   d. Generate offspring:
      - Within-island crossover (softmax parent selection)
      - Optional inter-island crossover (configurable rate)
      - Mutation (uniform parent selection)
   e. Every `great_filter_interval` generations:
      - Hard evaluation on hard_experiments only
      - Select top-h per island
      - Replenish with island-aware seeds
5. Save state for resume

### 13.5 Canonical Storage Layout

```text
populations/
  population_manifest.json           # active organisms, their dirs, scores
  evolution_state.json               # current generation, best score
  gen_0000/
    island_gradient_methods/
      org_abc123/
        optimizer.py
        idea_dna.txt
        evolution_log.json
        organism.json
        results/
          simple/
            synthetic_logreg.json
            mnist_mlp.json
          hard/
            cifar_convnet.json
        llm_request.json
        llm_response.json
  gen_0001/
    ...
```

---

## 14. Implementation Roadmap

### Phase 0: Freeze Semantics (prerequisite, no code)

Finalize one canonical design document:
- What is an organism
- What is an island
- What is genetic code
- Which score fields are canonical
- What the optimizer contract is

**Without this, code changes will keep producing partial designs.**

### Phase 1: Fix Critical Bugs (1-2 sessions)

1. **BUG-2: Evolution log scores** --- backfill scores after evaluation
   - Modify `_evaluate_organisms` in `evolution_loop.py`
   - Update `evolution_log.json` on disk after scoring

2. **BUG-1: Resume across generations** --- implement population manifest
   - Add `save_population_manifest()` and `load_population_manifest()` to `storage.py`
   - Update `_save_state` and `_load_state`

3. **Section 6.3: LLM DNA override** --- stop discarding LLM's IDEA_DNA output
   - Remove `idea_dna_override` from mutation
   - Make crossover use LLM's output DNA as canonical

### Phase 2: Unify Evolution Architecture (1-2 sessions)

4. **BUG-4: Resolve `cand_*` vs `org_*` duality**
   - Canonicalize organism-first path
   - Repurpose `EvolverOrchestrator` as evaluation-only backend
   - Add organism-aware context loading to `storage.py`

5. **Section 11.1: Fix optimizer contract everywhere**
   - README, runtime validator, generator validator, tests, examples, prompts
   - Create one shared validator module

### Phase 3: Move Prompts to Config (1 session)

6. Create `conf/system_prompt.txt`
7. Move operator prompt templates to `conf/prompts/`
8. Delete duplicate prompt files (`mutation_user.txt` vs `mutate_user.txt`, `crossover_user.txt` vs `crossbreed_user.txt`)
9. Update `OptimizerGenerator` to load prompts from config paths

### Phase 4: Fix Specification Violations (1-2 sessions)

10. **Section 6.2: Great Filter hard-only** --- one-line fix
11. **Section 6.1: Crossover lineage** --- only mother's log
12. **Section 6.6: Mother/father semantics** --- explicit roles
13. **Section 6.5: Separate top-k / top-h** --- add config, update selection
14. **Section 6.4: Remove ambiguous `score` field** --- use `simple_score` everywhere

### Phase 5: Island Model (3-4 sessions)

15. Create `conf/islands/` directory with 3-5 initial school descriptions
16. Add `IslandConfig`, `IslandState` types
17. Add `island_id` to `OrganismMeta`
18. Rewrite `EvolutionLoop.run()` for per-island population management
19. Per-island seeding with school-specific prompts
20. Per-island elite selection
21. Intra-island crossover (majority) + inter-island crossover (configurable rate)

### Phase 6: Selection Rework (1 session)

22. Implement `softmax_select` for crossover parents
23. Implement `uniform_select` for mutation parents
24. Update `select_parents_for_reproduction` to use correct selection methods
25. Add `softmax_temperature` config

### Phase 7: Prompt Engineering Overhaul (1-2 sessions)

26. Revise DNA richness requirements in all prompts
27. Add genetic metaphor framing to all prompts
28. Revise evolution log format for punchier descriptions
29. Add island context to seed prompts
30. Improve crossbreed prompt to explain what happened
31. Implement lineage summarization layer (not just last 5 entries)

### Phase 8: Configuration Cleanup (1 session)

32. Remove dead config knobs (`selection_strategy`, `novelty_filter`, `bandit_llm_selection`) or implement them
33. Remove legacy `quality_ref` / `steps_ref` from experiment configs
34. Fix `score_weights` (implement or remove)
35. Consolidate allocation configs (remove duplication)
36. Resolve `eval_experiments` vs `evaluation.*_experiments`
37. Fix `crossover_rate` (currently dead --- `mutation_rate` controls everything)

### Phase 9: Test Surface Upgrade (1-2 sessions)

See Section 15.

---

## 15. Required Test Surface

### 15.1 Invariant Tests (product-level correctness)

- Seed initialization creates organisms per island, not globally
- Per-island population counts are maintained after selection
- Maternal lineage is preserved after crossover (only mother's log inherited)
- Lineage score is backfilled after simple evaluation (no `score=None`)
- Great Filter does NOT overwrite simple_score
- Great Filter uses ONLY hard experiments
- Mutation CAN persist LLM-added genes (IDEA_DNA from LLM is canonical)
- Crossover CAN persist LLM-updated gene descriptions
- Resume correctly loads organisms from earlier generations

### 15.2 Config Truthfulness Tests

- `simple_allocation` only affects simple phase evaluation
- `hard_allocation` only affects hard phase evaluation
- Dead knobs either cause explicit warning or are absent from config
- System prompt paths come from `conf`, not package internals
- Every config key has a runtime reference or is explicitly marked as stub

### 15.3 Contract Tests

- README example optimizer actually loads and runs
- Generator validator and runtime importer accept/reject exactly the same contracts
- Optimizer with old `build_optimizer(cfg)` contract fails consistently everywhere
- Template-rendered code passes the same validator as free-form generated code

### 15.4 End-to-End Integration Tests

- Multi-island seed -> simple eval -> selection -> crossover/mutation -> Great Filter
- Artifact layout is canonical and consistent (no `cand_*` / `org_*` mixing)
- Reload/resume works without changing semantics
- Score semantics are preserved across generations (simple_score doesn't get overwritten)

### 15.5 Regression Tests for Fixed Bugs

- BUG-1: Organisms from earlier generations survive resume
- BUG-2: Evolution log entries have non-None scores after evaluation
- BUG-3: LLM's IDEA_DNA section is used (not overridden) in mutation/crossover
- BUG-4: Crossover lineage contains only maternal entries

---

## 16. Priority Matrix

| Priority | Issue | Type | Impact | Effort | Section |
|---|---|---|---|---|---|
| P0 | Island model missing | Missing feature | Fundamental architecture gap | Large | 5.1 |
| P0 | Resume bug across generations | Bug | Data loss on restart | Medium | 7.1 |
| P0 | Evolution log scores never backfilled | Bug | LLM gets no feedback | Small | 7.2 |
| P0 | LLM DNA output silently discarded | Bug + Spec violation | False evolution | Small | 6.3 |
| P0 | `cand_*` vs `org_*` architecture duality | Architecture | Confused data model | Medium | 7.4 |
| P1 | Softmax parent selection | Spec violation | Wrong search dynamics | Small | 5.2 |
| P1 | Crossover lineage (only mother) | Spec violation | Wrong inheritance | Small | 6.1 |
| P1 | Great Filter hard-only | Spec violation | Wrong evaluation scope | Trivial | 6.2 |
| P1 | Separate top-k / top-h | Spec violation | Wrong selection pressure | Small | 6.5 |
| P1 | Optimizer contract drift | Documentation | Users/tests validate wrong interface | Medium | 11.1 |
| P1 | `score = max(old, new)` ambiguity | Spec violation | Unreliable parent selection | Small | 6.4 |
| P2 | Prompt engineering overhaul | Prompts | Reduced LLM effectiveness | Medium | 8.* |
| P2 | System prompt in conf | Spec violation | Wrong file organization | Small | 5.6 |
| P2 | Two validators with different strictness | Architecture | Inconsistent validation | Small | 11.2 |
| P2 | Dead config knobs | Config | False control (science risk) | Small | 9.7 |
| P2 | Allocation config duality | Config | Dead config | Small | 9.5 |
| P3 | `score_weights` ignored | Config bug | Misleading config | Trivial | 9.6 |
| P3 | Legacy `quality_ref`/`steps_ref` | Config cleanup | Documentation debt | Trivial | 9.8 |
| P3 | Lineage truncation to 5 entries | Prompt | Possible history loss | Small | 8.8 |
| P3 | Crossover comment lies about dedup | Code smell | Minor | Trivial | 7.5 |
| P3 | Mixed naming conventions | Code smell | Maintenance friction | Trivial | 8.7 |

---

## 17. File-by-File Change Program

### Files requiring major changes

| File | Changes | Priority |
|---|---|---|
| `src/evolve/evolution_loop.py` | Complete rewrite: island model, per-island selection, resume fix, Great Filter hard-only, correct score semantics | P0 |
| `src/evolve/selection.py` | Add `softmax_select`, `uniform_select`, per-island variants, remove tournament for inappropriate uses | P1 |
| `src/evolve/types.py` | Add `island_id`, `mother_id`, `father_id` to OrganismMeta; add `IslandConfig`, `IslandState`; richer `LineageEntry` | P0 |
| `src/evolve/storage.py` | Add population manifest, organism context loading, island-aware paths | P0 |
| `src/organisms/crossbreeding.py` | Fix lineage (line 167), remove `idea_dna_override`, fix mother/father semantics | P0/P1 |
| `src/organisms/mutation.py` | Remove `idea_dna_override`, use LLM's DNA as canonical | P0 |
| `src/organisms/organism.py` | Score backfill in evolution log, richer lineage entry | P0 |

### Files requiring moderate changes

| File | Changes | Priority |
|---|---|---|
| `src/evolve/generator.py` | Load constant system prompt from conf, shared validator | P2 |
| `src/evolve/operators.py` | `SeedOperator` island-aware, remove legacy operators or mark deprecated | P1 |
| `src/evolve/orchestrator.py` | Accept separate allocation configs, public API for organism evaluation | P2 |
| `src/evolve/metrics_adapter.py` | Remove `del scoring_cfg`, either use weights or document F1 | P3 |
| `conf/evolver/default.yaml` | Add island config, top-k/top-h, softmax_temperature, remove dead knobs | P1 |
| `README.md` | Update optimizer contract, remove legacy docs, add island/school docs | P1 |

### Files to create

| File | Purpose | Priority |
|---|---|---|
| `conf/system_prompt.txt` | Constant system prompt for all LLM operations | P2 |
| `conf/islands/gradient_methods.txt` | Island 1 school description | P0 |
| `conf/islands/second_order.txt` | Island 2 school description | P0 |
| `conf/islands/adaptive_methods.txt` | Island 3 school description | P0 |
| `conf/prompts/seed_user.txt` | Moved from src (with island context) | P2 |
| `conf/prompts/mutation_user.txt` | Moved from src | P2 |
| `conf/prompts/crossover_user.txt` | Moved from src | P2 |

### Files to delete or deprecate

| File | Reason | Priority |
|---|---|---|
| `src/evolve/prompts/mutation_user.txt` | Legacy duplicate of `mutate_user.txt` | P2 |
| `src/evolve/prompts/crossover_user.txt` | Legacy duplicate of `crossbreed_user.txt` | P2 |
| `src/evolve/prompts/optimizer_system.txt` | Legacy candidate-path prompt | P2 |
| `src/evolve/prompts/optimizer_user.txt` | Legacy candidate-path prompt | P2 |

### Files that are OK as-is

| File | Notes |
|---|---|
| `src/evolve/allocation.py` | Neyman allocation correctly implemented |
| `src/evolve/gpu_pool.py` | GPU pool working |
| `src/evolve/template_parser.py` | Template system solid (keep shared validator separate) |
| `src/evolve/scoring.py` | Aggregation logic correct |
| `src/evolve/templates/optimizer_template.txt` | Fixed template --- do not change |
| `valopt/utils/baselines.py` | Baseline loading correct |
| `valopt/utils/objective_tracking.py` | Objective tracking correct |
| `valopt/runner.py` | Validation runtime solid |
| `valopt/optimizer_api.py` | New contract defined correctly |
| `valopt/utils/import_utils.py` | Import validation correct |
| `src/validate/run_one.py` | Subprocess evaluator correct |
| All experiment configs & implementations | OK |

---

## 18. Final Summary

Both audits converge on the same five conclusions:

1. **The system does not yet implement island/school-based evolutionary search.** It's a single-population hybrid. This is the single largest gap.

2. **Genetic code and lineage are semantically broken.** DNA is too shallow, LLM output is discarded, scores are never backfilled into the lineage. The evolutionary loop looks like it works but doesn't actually evolve ideas.

3. **The evaluation/reward infrastructure is strong.** Baseline-relative scoring, Neyman allocation, GPU pool, 17 experiments --- all working. Preserve this.

4. **The main technical problem is architectural ambiguity**, not missing features. Two entity models (`cand_*`/`org_*`), two validators, dead config knobs, drifted documentation --- these create a system that looks complete but isn't coherent.

5. **The correct next step is canonicalization**, not feature additions. Fix the data model, fix the bugs, fix the contracts, fix the config. Then add islands and rich genetic code on a clean foundation.

If code is built on top of the current architecture without this refactoring, the result will be a research platform that produces semantically wrong results while appearing to work correctly. This is the worst possible outcome for a scientific tool.

---

*End of merged audit. All findings backed by specific file:line references from codebase snapshot 2026-03-26.*
*Sources: Claude Opus audit + GPT audit, unified and cross-verified.*
