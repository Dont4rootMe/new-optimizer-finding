#!/usr/bin/env python
"""Legacy rank-0 launcher for Ollama cloud (pytorch2 / torchrun) jobs.

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

New jobs should use ``scripts/cluster/`` and ``type=binary``. This entrypoint
remains available only to reproduce the historical multi-rank Ollama path:

  ollama_torchrun_job.py --kind evolve --config-name config_co-bench --task TSP \
      --bootstrap --env optfind --manager conda --num-gpus 8
  ollama_torchrun_job.py --kind baseline --config-name baselines/circle_packing_shinka \
      --env optfind --manager conda --num-gpus 8

Only stdlib is used, so it runs under the cluster's base Python; the heavy deps
live in the ``--env`` the wrapper runs in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_RANK_ENV_VARS = ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "GROUP_RANK", "LOCAL_RANK")

# Official static linux build (bin/ollama + bundled GPU libs). Current releases
# ship .tar.zst (zstd); we decompress it with python `zstandard` (a conda
# dependency, so importable under the base conda python) with shell fallbacks.
# `latest` so the models' minimum-Ollama-version requirement is satisfied.
# Override with --ollama-url (a .tgz URL also works -- extraction auto-detects).
OLLAMA_URL = "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"


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


def _env_present(args: argparse.Namespace) -> bool:
    """True if the requested env already exists (prefix path or named env)."""
    if args.env_path:
        return os.path.exists(os.path.join(args.env_path, "conda-meta")) or os.path.exists(
            os.path.join(args.env_path, "bin", "python")
        )
    if args.env:
        try:
            out = subprocess.run([args.manager, "env", "list"], capture_output=True, text=True)
            return any(
                line.split() and line.split()[0] == args.env for line in out.stdout.splitlines()
            )
        except Exception:  # noqa: BLE001
            return False
    return True  # no env requested -> nothing to ensure


def _ensure_env(args: argparse.Namespace, env: dict) -> int:
    """Create the project env if it is missing (idempotent, lock-guarded).

    On a fresh job container the shared-FS prefix env may not exist yet. Build
    it once with scripts/create_env.sh; concurrent jobs coordinate via an
    mkdir lock next to the prefix so only one creates it and the rest wait.
    Returns 0 on success (env present), non-zero to abort the job.
    """
    if _env_present(args):
        return 0
    create = ["bash", "scripts/create_env.sh", "--manager", args.manager,
              "--extras", args.env_extras, "-y"]
    if args.env_path:
        create += ["--prefix", args.env_path]
    elif args.env:
        create += ["--name", args.env]
    else:
        return 0

    # Named envs have no shared lock target; just build it.
    if not args.env_path:
        print(f"[cluster_job] env missing; creating: {' '.join(create)}", flush=True)
        return subprocess.run(create, cwd=str(REPO_ROOT), env=env).returncode

    lock = args.env_path + ".creating.lock"
    try:
        os.makedirs(args.env_path.rsplit("/", 1)[0], exist_ok=True)
    except OSError:
        pass
    try:
        os.mkdir(lock)  # atomic: only one creator wins
        owns_lock = True
    except FileExistsError:
        owns_lock = False

    if owns_lock:
        print(f"[cluster_job] env missing; creating prefix env: {' '.join(create)}", flush=True)
        try:
            rc = subprocess.run(create, cwd=str(REPO_ROOT), env=env).returncode
        finally:
            try:
                os.rmdir(lock)
            except OSError:
                pass
        if rc != 0:
            return rc
        if not _env_present(args):
            print("[cluster_job] create_env.sh finished but env still not present.", file=sys.stderr, flush=True)
            return 1
        return 0

    # Another job is building it: wait for the env to appear.
    print(f"[cluster_job] another job is creating {args.env_path}; waiting...", flush=True)
    deadline = time.time() + 3600
    while time.time() < deadline:
        if _env_present(args):
            print("[cluster_job] env became available; continuing.", flush=True)
            return 0
        time.sleep(15)
    print(f"[cluster_job] timed out waiting for {args.env_path}. If a previous build "
          f"crashed, remove {lock} and retry, or create the env manually with "
          "scripts/create_env.sh.", file=sys.stderr, flush=True)
    return 1


def _ollama_present(ollama_dir: str) -> bool:
    return os.path.exists(os.path.join(ollama_dir, "bin", "ollama"))


def _extract_archive(archive_path: str, dest_dir: str, url: str) -> int:
    """Extract a .tar.zst (zstd) or .tgz/.tar.gz (gzip) tarball into dest_dir."""
    import tarfile

    low = url.lower()
    if low.endswith((".tar.zst", ".tzst", ".zst")):
        # Prefer python zstandard (a conda dependency -> usually importable).
        try:
            import zstandard  # type: ignore

            dctx = zstandard.ZstdDecompressor()
            with open(archive_path, "rb") as fh, dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:  # streaming, non-seekable
                    tf.extractall(dest_dir)
            return 0
        except ImportError:
            pass
        # Shell fallbacks if zstandard is unavailable.
        for cmd in (
            ["tar", "--use-compress-program=unzstd", "-xf", archive_path, "-C", dest_dir],
            ["tar", "--zstd", "-xf", archive_path, "-C", dest_dir],
        ):
            try:
                if subprocess.run(cmd).returncode == 0:
                    return 0
            except FileNotFoundError:
                continue
        try:
            decomp = subprocess.Popen(["zstd", "-dc", archive_path], stdout=subprocess.PIPE)
            rc = subprocess.run(["tar", "-x", "-C", dest_dir], stdin=decomp.stdout).returncode
            decomp.wait()
            if rc == 0:
                return 0
        except FileNotFoundError:
            pass
        print("[cluster_job] cannot decompress .tar.zst (need python zstandard, or "
              "zstd / GNU tar --zstd on PATH).", file=sys.stderr, flush=True)
        return 1
    # gzip tarball
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    return 0


def _download_and_extract_ollama(ollama_dir: str, url: str) -> int:
    import tempfile
    import urllib.request

    os.makedirs(ollama_dir, exist_ok=True)
    suffix = ".tar.zst" if ".zst" in url.lower() else ".tgz"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(url, tmp_path)
        rc = _extract_archive(tmp_path, ollama_dir, url)  # trusted source; layout is bin/ + lib/
        if rc != 0:
            return rc
        os.chmod(os.path.join(ollama_dir, "bin", "ollama"), 0o755)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[cluster_job] ollama download/extract failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _ensure_ollama(args: argparse.Namespace, env: dict) -> int:
    """Put the `ollama` CLI on PATH (the run/baseline wrappers require it).

    If it is already on PATH (image provides it) or already cached at
    --ollama-dir, just reuse it; otherwise download the official static linux
    build once to the shared filesystem (lock-guarded), and prepend its bin dir
    to the subprocess PATH so `conda run` -> the wrapper can find it.
    """
    if args.no_ensure_ollama:
        return 0
    ollama_dir = args.ollama_dir or os.path.join(os.path.dirname(str(REPO_ROOT)), ".ollama-dist")
    ollama_url = args.ollama_url or OLLAMA_URL
    bin_dir = os.path.join(ollama_dir, "bin")

    if _ollama_present(ollama_dir):
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        print(f"[cluster_job] using cached ollama at {bin_dir}", flush=True)
        return 0
    if shutil.which("ollama"):
        print("[cluster_job] ollama already on PATH; using it.", flush=True)
        return 0

    lock = ollama_dir + ".creating.lock"
    try:
        os.makedirs(os.path.dirname(ollama_dir), exist_ok=True)
    except OSError:
        pass
    try:
        os.mkdir(lock)
        owns_lock = True
    except FileExistsError:
        owns_lock = False

    if owns_lock:
        print(f"[cluster_job] ollama missing; downloading {ollama_url} -> {ollama_dir}", flush=True)
        try:
            rc = _download_and_extract_ollama(ollama_dir, ollama_url)
        finally:
            try:
                os.rmdir(lock)
            except OSError:
                pass
        if rc != 0 or not _ollama_present(ollama_dir):
            return rc or 1
    else:
        print(f"[cluster_job] another job is downloading ollama to {ollama_dir}; waiting...", flush=True)
        deadline = time.time() + 1800
        while time.time() < deadline:
            if _ollama_present(ollama_dir):
                break
            time.sleep(10)
        if not _ollama_present(ollama_dir):
            print(f"[cluster_job] timed out waiting for ollama. Remove {lock} and retry, "
                  "or install ollama into the image.", file=sys.stderr, flush=True)
            return 1

    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    print(f"[cluster_job] ollama ready at {bin_dir}", flush=True)
    return 0


def _maybe_env_prefix(args: argparse.Namespace) -> list[str]:
    """`<manager> run -p/-n <env>` prefix so the wrapper uses the project env.

    Prefer --env-path (a prefix env at an absolute path on the shared
    filesystem, which survives into a fresh job container) over --env (a named
    env, which usually does NOT).
    """
    if args.env_path:
        return [args.manager, "run", "-p", args.env_path]
    if args.env:
        return [args.manager, "run", "-n", args.env]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["evolve", "baseline"])
    parser.add_argument("--config-name", required=True, dest="config_name")
    parser.add_argument("--task", default="", help="CO-Bench CO_BENCH_TASK id (UPPER-CASE); blank for non-CO-Bench.")
    parser.add_argument("--bootstrap", action="store_true", help="Run scripts/bootstrap_cobench.sh first (CO-Bench data).")
    parser.add_argument("--env", default="", help="Named conda/micromamba env to run the wrapper in (blank = current PATH).")
    parser.add_argument("--env-path", default="", dest="env_path",
                        help="Prefix env at an absolute path (preferred for cluster jobs); wins over --env.")
    parser.add_argument("--manager", default="conda", choices=["conda", "micromamba", "mamba"])
    parser.add_argument("--env-extras", default="evolve,shinka_baseline,co_bench", dest="env_extras",
                        help="pip extras for the env when auto-creating it (superset that serves run + baseline + co_bench).")
    parser.add_argument("--no-ensure-env", action="store_true", dest="no_ensure_env",
                        help="Do NOT auto-create the env if missing (fail fast instead).")
    parser.add_argument("--ollama-dir", default="", dest="ollama_dir",
                        help="Where to cache the ollama binary (default: <repo_parent>/.ollama-dist).")
    parser.add_argument("--ollama-url", default="", dest="ollama_url",
                        help="Override the ollama archive URL (default: official latest; legacy/non-reproducible).")
    parser.add_argument("--no-ensure-ollama", action="store_true", dest="no_ensure_ollama",
                        help="Do NOT auto-provision the ollama CLI (assume it is on PATH).")
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

    # Self-bootstrap the project env if it is missing (unless disabled).
    if not args.no_ensure_env:
        rc = _ensure_env(args, env)
        if rc != 0:
            return rc

    # Ensure the `ollama` CLI is on PATH (the wrappers require it).
    rc = _ensure_ollama(args, env)
    if rc != 0:
        return rc

    for step in steps:
        result = subprocess.run(step, cwd=str(REPO_ROOT), env=env)
        if result.returncode != 0:
            print(f"[cluster_job] step failed (exit {result.returncode}): {' '.join(step)}", file=sys.stderr, flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
