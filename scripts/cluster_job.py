#!/usr/bin/env python
"""Rank-0 launcher for cloud (pytorch2 / torchrun) jobs.

The cluster's ``pytorch2`` job type runs a job's ``script`` through
``torch.distributed.launch`` / ``torchrun`` as ``python <script>`` with one
process per GPU (e.g. 8 ranks on an a100.8gpu node). That is correct for a DDP
training script, but THIS project's evolution loop / ShinkaEvolve baseline is a
SINGLE-PROCESS orchestrator: the GPUs host local Ollama model instances, not a
data-parallel model. A bash command cannot be the ``script`` (torchrun would
try to ``python`` its first token), and we do NOT want N copies of the
orchestrator fighting over Ollama ports + the population dir.

So jobs point ``script`` at this file instead. It:
  * runs the real command on GLOBAL RANK 0 only; every other rank exits 0
    immediately (a clean exit = success for torch elastic),
  * builds the wrapper invocation from plain flags (no shell operators, no
    quoting -- avoids the nested bash/torchrun quote-eval pitfalls),
  * optionally runs it inside a named conda/micromamba env (``--env``) so the
    repo's run_evolution.sh / run_shinka_baseline.sh pick up that env's Python
    (lib_runtime.sh resolves ``python`` from PATH),
  * makes all node GPUs visible on rank 0 (``--num-gpus``) so Ollama can place
    its instances across them.

Usage (built by notebooks/job_runs/*.ipynb):
  cluster_job.py --kind evolve   --config-name config_co-bench --task TSP \
      --bootstrap --env optfind --manager conda --num-gpus 8
  cluster_job.py --kind baseline --config-name baselines/circle_packing_shinka \
      --env optfind --manager conda --num-gpus 8

Only stdlib is used, so it runs under the cluster's base Python; the heavy deps
live in the ``--env`` the wrapper runs in.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_RANK_ENV_VARS = ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "GROUP_RANK", "LOCAL_RANK")


def global_rank() -> int:
    """Best-effort global rank across torchrun / OpenMPI / generic launchers."""
    for var in _RANK_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _wrapper_argv(args: argparse.Namespace) -> list[str]:
    """Build the bash-wrapper invocation as a plain argv list (no shell)."""
    if args.kind == "evolve":
        argv = ["bash", "scripts/run_evolution.sh", "--seed", "--config-name", args.config_name]
    else:  # baseline
        argv = ["bash", "scripts/run_shinka_baseline.sh", "--config-name", args.config_name]
    if args.task:
        # The CO-Bench task selector. Built here so no '=' ever appears in the
        # job's `script` string (which torchrun splits on whitespace).
        argv.append(f"experiments.co_bench.CO_BENCH_TASK={args.task}")
    return argv


def _maybe_env_prefix(args: argparse.Namespace) -> list[str]:
    """`<manager> run -n <env>` prefix so the wrapper uses the project env."""
    if not args.env:
        return []
    return [args.manager, "run", "-n", args.env]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["evolve", "baseline"])
    parser.add_argument("--config-name", required=True, dest="config_name")
    parser.add_argument("--task", default="", help="CO-Bench CO_BENCH_TASK id (UPPER-CASE); blank for non-CO-Bench.")
    parser.add_argument("--bootstrap", action="store_true", help="Run scripts/bootstrap_cobench.sh first (CO-Bench data).")
    parser.add_argument("--env", default="", help="conda/micromamba env to run the wrapper in (blank = current PATH).")
    parser.add_argument("--manager", default="conda", choices=["conda", "micromamba", "mamba"])
    parser.add_argument("--num-gpus", type=int, default=8, dest="num_gpus",
                        help="Make GPUs 0..N-1 visible on rank 0 (0 = leave CUDA_VISIBLE_DEVICES untouched).")
    parser.add_argument("--print-only", action="store_true", help="Print the resolved commands and exit (no execution).")
    # Ignore launcher-injected args we don't care about (e.g. --local-rank).
    args, unknown = parser.parse_known_args(argv)

    rank = global_rank()
    if rank != 0:
        # Single-process orchestrator: non-primary ranks have nothing to do.
        print(f"[cluster_job] rank {rank} != 0; nothing to do, exiting 0.", flush=True)
        return 0

    if unknown:
        print(f"[cluster_job] ignoring unrecognized launcher args: {unknown}", flush=True)

    env = dict(os.environ)
    if args.num_gpus and args.num_gpus > 0:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    prefix = _maybe_env_prefix(args)
    steps: list[list[str]] = []
    if args.bootstrap:
        steps.append(prefix + ["bash", "scripts/bootstrap_cobench.sh"])
    steps.append(prefix + _wrapper_argv(args))

    print(f"[cluster_job] rank 0; repo={REPO_ROOT}", flush=True)
    print(f"[cluster_job] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(unchanged)')}", flush=True)
    for step in steps:
        print(f"[cluster_job] step: {' '.join(step)}", flush=True)
    if args.print_only:
        return 0

    for step in steps:
        result = subprocess.run(step, cwd=str(REPO_ROOT), env=env)
        if result.returncode != 0:
            print(f"[cluster_job] step failed (exit {result.returncode}): {' '.join(step)}", file=sys.stderr, flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
