# optbench

Hydra-centric validation scaffold for benchmarking optimizers across modular ML training experiments.

## Key properties

- Single-GPU design (`1x NVIDIA A100` target) with explicit per-experiment compute limits.
- Unified experiment interface (`build_datamodule`, `build_model`, `train`, `evaluate`).
- Dynamic optimizer loading from external Python file (`build_optimizer(model, max_steps)` returning a controller object).
- Reproducibility defaults (seed control, deterministic toggle, resolved config snapshots).
- Safety defaults (NaN detection, optional abort-on-NaN, grad-norm logging, grad clipping).

## Project structure

- `conf/` Hydra configs and prompt assets
- `optbench/` runner, schemas, registry, utils
- `experiments/` per-experiment modules
- `optimizer_guesses/examples/` external optimizer examples
- `tests/` minimal tests

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[audio]
pip install -e .[hf]
pip install -e .[lora]
pip install -e .[evolve]
```

## Run

Smoke run for all enabled experiments:

```bash
python -m optbench.main mode=smoke
```

Baseline stats with each experiment's default optimizer:

```bash
python -m optbench.main mode=stats
```

Run with external optimizer file:

```bash
python -m optbench.main mode=run optimizer_path=optimizer_guesses/examples/sgd_baseline.py
```

Run the canonical organism-first evolution pipeline:

```bash
export OPENAI_API_KEY=...
python -m src.main mode=evolve
```

Standalone canonical evolve entrypoint:

```bash
python -m src.evolve.run mode=evolve
```

Defaults for evolve LLM:
- `evolver.llm.provider=chatgpt` (OpenAI Python SDK)
- `evolver.llm.model=gpt-5.4-pro` (latest pro-thinking default in this repo)
- `evolver.llm.reasoning_effort=xhigh`

You can also use model alias:

```bash
python -m src.main mode=evolve evolver.llm.model=latest_pro_thinking
```

Run evolve with simple-phase partial benchmark evaluation (Neyman allocation):

```bash
python -m src.main mode=evolve \
  evolver.phases.simple.allocation.enabled=true \
  evolver.phases.simple.allocation.sample_size=1
```

`mode=evolve` always runs the multi-generation organism-first `EvolutionLoop`.
The canonical loop reads only the `evolver.*` schema from `conf/evolver/default.yaml`.

## Useful Hydra overrides

Disable a specific experiment:

```bash
python -m optbench.main mode=smoke experiments.audio_transformer.enabled=false
```

Force CPU / fp32:

```bash
python -m optbench.main mode=smoke device=cpu precision=fp32
```

Override experiment budget:

```bash
python -m optbench.main mode=smoke experiments.cifar_convnet.compute.smoke_steps=50
```

Run-only validation cutoffs for candidate optimizer:

```bash
python -m optbench.main mode=run \
  optimizer_path=optimizer_guesses/examples/sgd_baseline.py \
  experiments.cifar_convnet.run_validation.max_steps=100 \
  experiments.cifar_convnet.run_validation.target_quality=0.85
```

## Notes on optional dependencies

- `audio_transformer` requires `.[audio]`.
- `minigpt_wikitext2` requires `.[hf]`.
- `lora_sft` requires `.[lora]`.
- `mode=evolve` with OpenAI provider requires `.[evolve]`.
- In `mode=smoke`, missing optional dependencies are recorded as `status="skipped"`.
- In `mode=run` / `mode=stats`, missing optional dependencies raise a clear install error.

## Optimizer contract

External optimizers must provide:

```python
def build_optimizer(model, max_steps):
    ...
```

The returned object must implement:

```python
class OptimizerController:
    def step(self, weights, grads, activations, step_fn): ...
    def zero_grad(self, set_to_none=True): ...
```

`model` is the full `torch.nn.Module`. `max_steps` is the total optimization budget for the run.
`step_fn()` is an optional extra forward-backward closure; each call consumes one optimization step from the same global budget.

## Evolve outputs

The canonical evolution pipeline is organism-first, island-aware, prompt-driven from `conf/prompts/`, and restore-driven from `population_manifest.json`. Prompt pairs are grouped by task under `conf/prompts/<task>/`, and island descriptions live in `conf/prompts/islands/`.
Canonical resume is strict: missing `population_manifest.json`, `genetic_code.md`, or `lineage.json` is treated as corruption and fails fast.

Phase defaults in the canonical path:
- simple phase: `eval_mode=smoke`
- Great Filter: `eval_mode=full`

The Great Filter evaluates only its configured hard experiment list. `simple_reward` remains persisted on the organism; `hard_reward` is stored separately; `selection_reward` switches to the hard score only for the hard-phase selection step.

Canonical organism artifacts are written to:

```text
<population_root>/
  evolution_state.json
  population_manifest.json
  gen_<G>/
    island_<island_id>/
      org_<UUID>/
        optimizer.py
        genetic_code.md
        lineage.json
        organism.json
        summary.json
        llm_request.json
        llm_response.json
        results/
          simple/<experiment>.json
          hard/<experiment>.json
        logs/
          simple_<experiment>.out
          simple_<experiment>.err
          hard_<experiment>.out
          hard_<experiment>.err
```

`summary.json` stores phase-specific evaluation results, including:
- `simple_reward`
- `hard_reward`
- `selection_reward`
- `phase_results.simple`
- `phase_results.hard`

## Single-GPU constraint

All defaults are configured for single-device execution only.
No DDP, model parallel, or multi-GPU logic is included.
