# valopt

Hydra-centric validation scaffold for benchmarking optimizers across modular ML training experiments.

## Key properties

- Single-GPU design (`1x NVIDIA A100` target) with explicit per-experiment compute limits.
- Unified experiment interface (`build_datamodule`, `build_model`, `train`, `evaluate`).
- Dynamic optimizer loading from external Python file (`build_optimizer(cfg)` returning a controller object).
- Reproducibility defaults (seed control, deterministic toggle, resolved config snapshots).
- Safety defaults (NaN detection, optional abort-on-NaN, grad-norm logging, grad clipping).

## Project structure

- `conf/` Hydra configs
- `valopt/` runner, schemas, registry, utils
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
python -m valopt.main mode=smoke
```

Baseline stats with each experiment's default optimizer:

```bash
python -m valopt.main mode=stats
```

Run with external optimizer file:

```bash
python -m valopt.main mode=run optimizer_path=optimizer_guesses/examples/sgd_baseline.py
```

Run async evolution pipeline:

```bash
export OPENAI_API_KEY=...
python -m src.main mode=evolve
```

Standalone evolve entrypoint:

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

Run evolve with partial benchmark evaluation (Neyman allocation):

```bash
python -m src.main mode=evolve \
  evolver.allocation.enabled=true \
  evolver.allocation.sample_size=1
```

## Useful Hydra overrides

Disable a specific experiment:

```bash
python -m valopt.main mode=smoke experiments.audio_transformer.enabled=false
```

Force CPU / fp32:

```bash
python -m valopt.main mode=smoke device=cpu precision=fp32
```

Override experiment budget:

```bash
python -m valopt.main mode=smoke experiments.cifar_convnet.compute.smoke_steps=50
```

Run-only validation cutoffs for candidate optimizer:

```bash
python -m valopt.main mode=run \
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
def build_optimizer(cfg):
    ...
```

The returned object must implement:

```python
class OptimizerController:
    def initialize(self, named_parameters, cfg): ...
    def step(self, weights, grads, activations): ...
    def zero_grad(self, set_to_none=True): ...
```

Framework runtime passes only `cfg["max_steps"]` during `initialize(...)`.

## Evolve outputs

The evolution pipeline writes candidates to:

```text
<population_root>/gen_<G>/cand_<UUID>/
  optimizer.py
  llm_request.json
  llm_response.json
  meta.json
  results/<experiment>.json
  logs/<experiment>.out
  logs/<experiment>.err
  summary.json
```

Generation events are appended to `index.jsonl` in `gen_<G>/`.

Each candidate `summary.json` includes:
- `selected_experiments`
- `allocation` snapshot (`pi`, `stats`, `history_window`, `sample_size`)
- per-experiment normalized fields: `raw_metric`, `quality_ratio`, `steps_ratio`, `exp_score`

Experiment-level normalization references are configured in:
- `conf/experiments/*.yaml`
  `normalization.quality_ref`, `normalization.steps_ref`, `normalization.eps`

## Single-GPU constraint

All defaults are configured for single-device execution only.
No DDP, model parallel, or multi-GPU logic is included.
