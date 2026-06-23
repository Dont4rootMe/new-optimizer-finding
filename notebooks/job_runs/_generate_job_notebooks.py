"""Generate cloud job-run notebooks for the organism-first experiments.

One .ipynb per task. Each notebook submits TWO independent cluster jobs:

  1. **Normal run**  -> ``scripts/run_evolution.sh --seed --config-name <preset>``
       starts the local Ollama instances, seeds generation 0, runs evolution.
  2. **Baseline**    -> ``scripts/run_shinka_baseline.sh --config-name baselines/<preset>``
       same Ollama lifecycle -> ShinkaEvolve over the same task evaluator.

The cluster scaffold (``client_lib.Job`` shape, ``echimbulatov | ... #ID0137
#rnd`` job description, queue ``diff``, region ``A100-MT``, ``priority_class``,
the a100.8gpu SKU formula, and the Comet creds) is copied verbatim from the
canonical job-run notebook. Only the launch command + experiment identity
change per task. The Comet api_key / workspace are identical to the canonical
notebook (and to the repo's Hydra config), and ``COMET_RUN_NAME`` is left
unset so the config's canonical per-task run-name (a system-visible
identifier) is preserved.

Regenerate with:  python notebooks/job_runs/_generate_job_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

# --- cluster constants (confirmed with the operator) ------------------------
WORK_DIR = "/home/jovyan/echimbulatov/fork_afedorov/constant_repos/new-optimizer-finding"
BASE_IMAGE = "cr.ai.cloud.ru/2754eb6e-ae19-4123-87ce-06ec3cc96500/job-latentdiffusion:flash-clear"

OUT_DIR = Path(__file__).resolve().parent


# --- per-task definitions ---------------------------------------------------
# slug          : notebook file slug + EXP_BASE used in the job description
# title         : notebook H1
# blurb         : 2-4 line task description (markdown)
# run_config    : evolution-run Hydra preset (--config-name)
# baseline_config: ShinkaEvolve baseline preset (--config-name)
# overrides     : hydra overrides appended to BOTH jobs ("" or CO_BENCH_TASK=...)
# comet_run     : canonical Comet run-name produced by the config (display only)
# bootstrap     : True -> chain scripts/bootstrap_cobench.sh first (CO-Bench data)
def _cobench(task_id: str, slug: str, title_task: str, blurb: str) -> dict:
    return {
        "slug": f"cobench-{slug}",
        "title": f"CO-Bench — {title_task}",
        "blurb": blurb,
        "run_config": "config_co-bench",
        "baseline_config": "baselines/co-bench",
        "overrides": f" experiments.co_bench.CO_BENCH_TASK={task_id}",
        "comet_run": f"cobench-{slug}",
        "bootstrap": True,
    }


TASKS: list[dict] = [
    {
        "slug": "circle-packing-shinka",
        "title": "Circle Packing — ShinkaEvolve unit-square (26 circles)",
        "blurb": (
            "Pack exactly **26 circles** in the unit square and **maximize the sum of "
            "radii**. Each organism is a `run_packing() -> (centers, radii, reported_sum)` "
            "candidate (ShinkaEvolve circle-packing benchmark)."
        ),
        "run_config": "config_circle_packing_shinka",
        "baseline_config": "baselines/circle_packing_shinka",
        "overrides": "",
        "comet_run": "circle-packing-shinka",
        "bootstrap": False,
    },
    {
        "slug": "atcoder-awtf2025",
        "title": "AtCoder AWTF 2025 Heuristic — Group Commands & Wall Planning",
        "blurb": (
            "AtCoder World Tour Finals 2025 Heuristic A. Each organism is a "
            "`solve_case(input_text) -> str` candidate, scored on the vendored "
            "`seed=0..99` corpus; the objective is to **minimize** the mean absolute "
            "score `T + 100 * sum_k Manhattan(final_k, target_k)` (the framework maximizes "
            "its negation). The corpus is vendored in-repo — no dataset bootstrap needed."
        ),
        "run_config": "config_awtf2025_heuristic",
        "baseline_config": "baselines/awtf2025_heuristic",
        "overrides": "",
        "comet_run": "awtf2025_heuristic",
        "bootstrap": False,
    },
    _cobench(
        "TSP", "tsp", "Travelling Salesman Problem (TSP)",
        "Evolve a `solve(**instance) -> {\"tour\": [...]}` algorithm; scored by CO-Bench's "
        "own per-task evaluator (higher normalized score = better).",
    ),
    _cobench(
        "BIN_PACKING_1D", "bin_packing_1d", "Bin Packing — one-dimensional",
        "Evolve a `solve(**instance)` bin-packing algorithm; scored by CO-Bench's per-task "
        "evaluator (higher normalized score = better).",
    ),
    _cobench(
        "MULTI_KNAPSACK", "multi_knapsack", "Multidimensional Knapsack Problem",
        "Evolve a `solve(**instance)` multidimensional-knapsack algorithm; scored by "
        "CO-Bench's per-task evaluator (higher normalized score = better).",
    ),
    _cobench(
        "SET_COVERING", "set_covering", "Set Covering",
        "Evolve a `solve(**instance)` set-covering algorithm; scored by CO-Bench's per-task "
        "evaluator (higher normalized score = better).",
    ),
    _cobench(
        "GRAPH_COLOURING", "graph_colouring", "Graph Colouring",
        "Evolve a `solve(**instance)` graph-colouring algorithm; scored by CO-Bench's "
        "per-task evaluator (higher normalized score = better).",
    ),
    _cobench(
        "JOB_SHOP", "job_shop", "Job Shop Scheduling",
        "Evolve a `solve(**instance)` job-shop-scheduling algorithm; scored by CO-Bench's "
        "per-task evaluator (higher normalized score = better).",
    ),
]


# --- cell builders ----------------------------------------------------------
def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def _title_md(task: dict) -> str:
    bootstrap_note = ""
    if task["bootstrap"]:
        bootstrap_note = (
            "\n\n> **CO-Bench data:** both jobs chain `scripts/bootstrap_cobench.sh` first "
            "(clone the CO-Bench checkout + download the dataset into `./data/co-bench`). "
            "It is idempotent — re-runs skip work that already exists."
        )
    return (
        f"# Job run — {task['title']}\n\n"
        f"{task['blurb']}\n\n"
        "Two **independent** cluster jobs, each with its own *submit / status / kill* cells:\n\n"
        f"1. **Normal run** — `scripts/run_evolution.sh --seed --config-name {task['run_config']}"
        f"{task['overrides']}` (starts Ollama, seeds gen 0, evolves to the configured stop criteria).\n"
        f"2. **Baseline** — `scripts/run_shinka_baseline.sh --config-name {task['baseline_config']}"
        f"{task['overrides']}` (ShinkaEvolve over the same evaluator + local Ollama).\n\n"
        f"Canonical Comet run-name (set by the repo config, **unchanged**): `{task['comet_run']}`. "
        "Run *Common config* first, then either job group."
        f"{bootstrap_note}"
    )


_COMMON_CFG = '''import client_lib

# ---------------------------------------------------------------------------
# Cluster + repo paths
# ---------------------------------------------------------------------------
WORK_DIR = "@@WORK_DIR@@"
BASE_IMAGE = "@@BASE_IMAGE@@"

# ---------------------------------------------------------------------------
# Hardware - a100.{N_GPUS}gpu SKU in the A100-MT region (verbatim from the
# canonical job-run notebook on this cluster). The organism-first evolution
# loop is a SINGLE-PROCESS orchestrator; the 8 GPUs host the local Ollama
# model instances that scripts/run_evolution.sh + scripts/lib_runtime.sh
# start (gemma 4 31B + qwen 3.5 across the 8 GPUs - see the experiment config).
# ---------------------------------------------------------------------------
N_NODES = 1
N_GPUS = 8
instance_cpu_count = 8
instance_addition = (
    f".{N_GPUS * instance_cpu_count}C.{N_GPUS * 243}G"
    if instance_cpu_count == 8 else ""
)
INSTANCE_TYPE = f"a100.{N_GPUS}gpu{instance_addition}"
REGION = "A100-MT"

_total_gpus = N_NODES * N_GPUS
print(f"Instance: {INSTANCE_TYPE}  (region={REGION})")
print(f"Total GPUs: {_total_gpus}  (= {N_NODES} nodes x {N_GPUS} GPUs)")

# ---------------------------------------------------------------------------
# Experiment identity. EXP_BASE is the human label used in the job description
# only. The system-visible run identity (the Comet run-name) is owned by the
# repo's Hydra config (cfg.comet.run_name) and left at its canonical default:
#   @@COMET_RUN@@
# ---------------------------------------------------------------------------
EXP_BASE = "@@EXP_BASE@@"
RUN_CONFIG = "@@RUN_CONFIG@@"             # normal evolution-run preset
BASELINE_CONFIG = "@@BASELINE_CONFIG@@"   # ShinkaEvolve baseline preset
OVERRIDES = "@@OVERRIDES@@"               # hydra overrides appended to BOTH jobs
NEEDS_COBENCH_BOOTSTRAP = @@BOOTSTRAP@@   # CO-Bench needs the dataset + checkout first

# ---------------------------------------------------------------------------
# Common job env - cluster scaffolding + Comet creds. Same Comet api_key and
# workspace as the canonical notebook (the repo's Hydra config bakes in the
# same api_key too); COMET_PROJECT points at this project's Comet project.
# COMET_RUN_NAME is intentionally NOT set, so the config's canonical per-task
# run-name is preserved.
# ---------------------------------------------------------------------------
common_env = {
    "PROJECT_ROOT": WORK_DIR,
    "PYTHONNOUSERSITE": 1,
    "PIP_USER": "no",
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_TIMEOUT": 23,
    "NCCL_IB_RETRY_CNT": 5,
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": 3600,
    "MLS_JOB_REGION_NAME": REGION,
    "MLS_JOB_TOTAL_GPU": _total_gpus,
    "CLEARML_CONFIG_FILE": "/home/jovyan/inkoziev/myclearml.conf",
    "COMET_API_KEY": "RrClhd4FveFQKO4qLo4jBjrKu",
    "COMET_WORKSPACE": "dont4rootme",
    "COMET_PROJECT": "new-optimizer-search",
    "COMET_MODE": "online",
    "COMET_LOGGING_CONSOLE": "true",
}'''


_RUN_SCRIPT = '''# ---------------------------------------------------------------------------
# Normal training run = the organism-first evolution loop.
# scripts/run_evolution.sh:
#   * starts/refreshes the local Ollama instances declared in the config
#     (scripts/lib_runtime.sh) on the node GPUs,
#   * with --seed, bootstraps the generation-0 population if missing,
#   * runs seeding + evolution to the configured stop criteria
#     (max_generations / max_organism_creations / per-model token budget).
# ---------------------------------------------------------------------------
_bootstrap = f"bash {WORK_DIR}/scripts/bootstrap_cobench.sh && " if NEEDS_COBENCH_BOOTSTRAP else ""
run_script = (
    f"cd {WORK_DIR} && {_bootstrap}"
    f"bash scripts/run_evolution.sh --seed --config-name {RUN_CONFIG}{OVERRIDES}"
)
EXP_NAME_RUN = f"{EXP_BASE}-run"
print(f"[{EXP_NAME_RUN}]")
print(run_script)'''


_RUN_SUBMIT = '''run_env = dict(common_env)

# `#ID0137 #rnd` are the user-quota / priority-category tags the cluster
# scheduler needs (verbatim from the canonical job-run notebook); without them
# the job lands in a default bucket with a short wall-time limit.
run_job = client_lib.Job(
    job_desc=f"echimbulatov | {EXP_NAME_RUN} #ID0137 #rnd",
    queue_name="diff",
    base_image=BASE_IMAGE,
    script=run_script,
    n_workers=N_NODES,
    instance_type=INSTANCE_TYPE,
    type="pytorch2",
    preflight_check=True,
    env_variables=run_env,
    region=REGION,
    flags={},
    priority_class="high",
)
run_job.submit()'''


_RUN_STATUS = '''while run_job.status() == "Job status=Pending":
    run_job.status()
run_job.logs()'''


_BASELINE_SCRIPT = '''# ---------------------------------------------------------------------------
# Baseline = ShinkaEvolve over the SAME task evaluator + local Ollama models.
# scripts/run_shinka_baseline.sh shares the Ollama lifecycle with the run above
# and invokes src.baselines.shinka.run (ShinkaEvolve dataclasses + our
# evaluator). Writes its per-program DB under shinka_runs/.
# ---------------------------------------------------------------------------
_bootstrap = f"bash {WORK_DIR}/scripts/bootstrap_cobench.sh && " if NEEDS_COBENCH_BOOTSTRAP else ""
baseline_script = (
    f"cd {WORK_DIR} && {_bootstrap}"
    f"bash scripts/run_shinka_baseline.sh --config-name {BASELINE_CONFIG}{OVERRIDES}"
)
EXP_NAME_BASELINE = f"{EXP_BASE}-baseline"
print(f"[{EXP_NAME_BASELINE}]")
print(baseline_script)'''


_BASELINE_SUBMIT = '''baseline_env = dict(common_env)

baseline_job = client_lib.Job(
    job_desc=f"echimbulatov | {EXP_NAME_BASELINE} #ID0137 #rnd",
    queue_name="diff",
    base_image=BASE_IMAGE,
    script=baseline_script,
    n_workers=N_NODES,
    instance_type=INSTANCE_TYPE,
    type="pytorch2",
    preflight_check=True,
    env_variables=baseline_env,
    region=REGION,
    flags={},
    priority_class="high",
)
baseline_job.submit()'''


_BASELINE_STATUS = '''while baseline_job.status() == "Job status=Pending":
    baseline_job.status()
baseline_job.logs()'''


def _common_cfg(task: dict) -> str:
    return (
        _COMMON_CFG.replace("@@WORK_DIR@@", WORK_DIR)
        .replace("@@BASE_IMAGE@@", BASE_IMAGE)
        .replace("@@EXP_BASE@@", task["slug"])
        .replace("@@RUN_CONFIG@@", task["run_config"])
        .replace("@@BASELINE_CONFIG@@", task["baseline_config"])
        .replace("@@OVERRIDES@@", task["overrides"])
        .replace("@@BOOTSTRAP@@", "True" if task["bootstrap"] else "False")
        .replace("@@COMET_RUN@@", task["comet_run"])
    )


def build_notebook(task: dict) -> dict:
    cells = [
        _md(_title_md(task)),
        _md("## Common config"),
        _code(_common_cfg(task)),
        _md("## Job 1 — Normal evolution run"),
        _code(_RUN_SCRIPT),
        _code(_RUN_SUBMIT),
        _code(_RUN_STATUS),
        _code("# run_job.kill()"),
        _md("## Job 2 — ShinkaEvolve baseline"),
        _code(_BASELINE_SCRIPT),
        _code(_BASELINE_SUBMIT),
        _code(_BASELINE_STATUS),
        _code("# baseline_job.kill()"),
    ]
    for idx, cell in enumerate(cells):
        cell["id"] = f"cell-{idx}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for task in TASKS:
        nb = build_notebook(task)
        path = OUT_DIR / f"job-run-{task['slug']}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path.name)
    print(f"Wrote {len(written)} notebooks to {OUT_DIR}:")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
