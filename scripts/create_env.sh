#!/usr/bin/env bash
set -euo pipefail

# Create a conda/micromamba environment for the organism-framework project and
# pip-install this repo (editable) plus the requested optional extras into it.
#
# The project is a normal pip package (pyproject.toml), so the env manager only
# provides the Python interpreter; dependencies are installed with pip inside
# the new env via `<manager> run -n/-p <env> ...` (no activation needed).
#
# Two env styles:
#   --name NAME     a *named* env (lives in the manager's envs dir).
#   --prefix PATH   a *prefix* env at an absolute path. Use this for cloud jobs:
#                   a job container starts fresh from the base image, so named
#                   envs created interactively are NOT visible to it; a prefix
#                   env on a SHARED/persistent path (e.g. next to this repo) is.

show_help() {
  cat <<'EOF'
Create a conda / micromamba env for this project and install it (editable).

Usage:
  ./scripts/create_env.sh <env-name> [options]
  ./scripts/create_env.sh --name <env-name> [options]
  ./scripts/create_env.sh --prefix /abs/path/to/env [options]

Env location (choose exactly one):
  <env-name> | -n, --name NAME   Named env (in the manager's envs dir).
  -P, --prefix PATH              Prefix env at an absolute PATH. Preferred for
                                 cluster jobs -- put it on a shared/persistent
                                 filesystem so job containers can see it.

Options:
  -m, --manager {conda|micromamba|mamba}
                       Env manager to use. Default: auto-detect
                       (micromamba > mamba > conda).
  -p, --python VERSION Python version for the env. Default: 3.11
                       (project requires >=3.10).
  -e, --extras LIST    Comma-separated pip extras to install. Default: evolve.
                       Known: audio, hf, lora, evolve, shinka_baseline, co_bench.
                       Special values:
                         all   -> every extra above
                         none  -> base install only (no extras)
  -c, --channel CHAN   Conda channel for the python/pip packages.
                       Default: conda-forge.
  -f, --force          If the env already exists, delete and recreate it from
                       scratch. Without --force an existing env is REUSED and
                       only missing / outdated packages are installed into it.
  -y, --yes            Non-interactive (assume yes for the manager prompts).
  -h, --help           Show this help.

Examples:
  ./scripts/create_env.sh optfind
  ./scripts/create_env.sh --name optfind --manager micromamba
  ./scripts/create_env.sh optfind -p 3.11 -e evolve,co_bench
  ./scripts/create_env.sh optfind --extras all --force
  # Cluster jobs: prefix env on the shared FS next to the repo (one-time):
  ./scripts/create_env.sh --prefix /home/.../constant_repos/.conda-envs/optfind \
      --manager conda --extras evolve,shinka_baseline,co_bench

Notes:
  - If the env already exists it is reused: the script skips creation and just
    runs the editable install, so pip pulls in only the missing/updated
    packages. Use --force to wipe and rebuild. (When reusing, --python is
    ignored -- the existing env keeps its interpreter.)
  - torch/torchvision come from pip's default index (a CUDA build on Linux, a
    CPU build on macOS); pass a custom index by editing the pip step if you
    need a specific CUDA wheel.
  - The `evolve` extra (default) pulls the LLM/eval/viz stack the evolution
    loop and baselines use (openai, anthropic, comet_ml, plotly, psutil, ...).
  - `shinka_baseline` adds shinka-evolve; `co_bench` adds the dataset/solver
    deps -- the CO-Bench checkout + dataset still need scripts/bootstrap_cobench.sh.
EOF
}

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
ENV_NAME=""
PREFIX=""
MANAGER=""
PY_VERSION="3.11"
EXTRAS="evolve"
CHANNEL="conda-forge"
FORCE=0
ASSUME_YES=0

ALL_EXTRAS="audio,hf,lora,evolve,shinka_baseline,co_bench"

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) show_help; exit 0 ;;
    -n|--name) ENV_NAME="${2:-}"; shift 2 ;;
    --name=*) ENV_NAME="${1#*=}"; shift ;;
    -P|--prefix) PREFIX="${2:-}"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    -m|--manager) MANAGER="${2:-}"; shift 2 ;;
    --manager=*) MANAGER="${1#*=}"; shift ;;
    -p|--python) PY_VERSION="${2:-}"; shift 2 ;;
    --python=*) PY_VERSION="${1#*=}"; shift ;;
    -e|--extras) EXTRAS="${2:-}"; shift 2 ;;
    --extras=*) EXTRAS="${1#*=}"; shift ;;
    -c|--channel) CHANNEL="${2:-}"; shift 2 ;;
    --channel=*) CHANNEL="${1#*=}"; shift ;;
    -f|--force) FORCE=1; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    --) shift; break ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      show_help >&2
      exit 2
      ;;
    *)
      if [[ -z "$ENV_NAME" && -z "$PREFIX" ]]; then
        ENV_NAME="$1"; shift
      else
        echo "Error: unexpected positional argument '$1' (env already set)." >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -n "$ENV_NAME" && -n "$PREFIX" ]]; then
  echo "Error: pass either a name or --prefix, not both." >&2
  exit 2
fi
if [[ -z "$ENV_NAME" && -z "$PREFIX" ]]; then
  echo "Error: an environment is required (positional/--name, or --prefix PATH)." >&2
  show_help >&2
  exit 2
fi

# Target args shared by every manager call: `-n NAME` or `-p PATH`.
if [[ -n "$PREFIX" ]]; then
  # Make the prefix absolute without requiring it (or its parent) to exist yet.
  case "$PREFIX" in
    /*) : ;;
    *)  PREFIX="$(pwd)/$PREFIX" ;;
  esac
  MGR_TARGET=( -p "$PREFIX" )
  ENV_REF="$PREFIX"
else
  MGR_TARGET=( -n "$ENV_NAME" )
  ENV_REF="$ENV_NAME"
fi

# --------------------------------------------------------------------------
# Resolve repo root (this script lives in <root>/scripts)
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
  echo "Error: pyproject.toml not found at ${ROOT_DIR}; run this script from inside the repo." >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Pick the env manager
# --------------------------------------------------------------------------
if [[ -n "$MANAGER" ]]; then
  case "$MANAGER" in
    conda|micromamba|mamba) ;;
    *) echo "Error: --manager must be one of conda|micromamba|mamba (got '$MANAGER')." >&2; exit 2 ;;
  esac
  if ! command -v "$MANAGER" >/dev/null 2>&1; then
    echo "Error: requested manager '$MANAGER' is not on PATH." >&2
    exit 1
  fi
else
  for candidate in micromamba mamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      MANAGER="$candidate"
      break
    fi
  done
  if [[ -z "$MANAGER" ]]; then
    echo "Error: no env manager found on PATH (looked for micromamba, mamba, conda)." >&2
    echo "Install one, or pass --manager explicitly." >&2
    exit 1
  fi
fi

YES_FLAG=""
if [[ "$ASSUME_YES" -eq 1 ]]; then
  YES_FLAG="-y"
fi

# --------------------------------------------------------------------------
# Normalise extras -> pip target like ".[evolve,co_bench]" (or "." for none)
# --------------------------------------------------------------------------
extras_norm="$(printf '%s' "$EXTRAS" | tr 'A-Z' 'a-z' | tr -d '[:space:]')"
case "$extras_norm" in
  all) extras_norm="$ALL_EXTRAS" ;;
  none|"") extras_norm="" ;;
esac

if [[ -n "$extras_norm" ]]; then
  PIP_TARGET=".[${extras_norm}]"
else
  PIP_TARGET="."
fi

# --------------------------------------------------------------------------
# Does the env already exist?
# --------------------------------------------------------------------------
env_exists() {
  if [[ -n "$PREFIX" ]]; then
    [[ -e "$PREFIX/conda-meta" || -x "$PREFIX/bin/python" ]]
  else
    "$MANAGER" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"
  fi
}

echo "==> manager : $MANAGER"
echo "==> env     : $ENV_REF"
echo "==> python  : $PY_VERSION"
echo "==> channel : $CHANNEL"
echo "==> extras  : ${extras_norm:-<none>}"
echo "==> install : pip install -e \"${PIP_TARGET}\"  (from ${ROOT_DIR})"

REUSE_ENV=0
if env_exists; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "==> env '$ENV_REF' exists; removing (--force)..."
    "$MANAGER" env remove "${MGR_TARGET[@]}" ${YES_FLAG} \
      || "$MANAGER" remove "${MGR_TARGET[@]}" --all ${YES_FLAG}
  else
    REUSE_ENV=1
    echo "==> env '$ENV_REF' already exists; reusing it -- pip will install only"
    echo "    missing / outdated packages (pass --force to delete and recreate)."
    echo "    note: --python ${PY_VERSION} is ignored for an existing env."
  fi
fi

# --------------------------------------------------------------------------
# Create the env (unless reusing) + pip install (editable) into it
# --------------------------------------------------------------------------
if [[ "$REUSE_ENV" -eq 1 ]]; then
  echo "==> reusing existing env '$ENV_REF'."
else
  if [[ -n "$PREFIX" ]]; then
    mkdir -p "$(dirname "$PREFIX")"
  fi
  echo "==> creating env '$ENV_REF' (python=$PY_VERSION) with $MANAGER ..."
  "$MANAGER" create ${YES_FLAG} "${MGR_TARGET[@]}" -c "$CHANNEL" "python=${PY_VERSION}" pip
fi

echo "==> upgrading pip tooling in '$ENV_REF' ..."
"$MANAGER" run "${MGR_TARGET[@]}" python -m pip install --upgrade pip setuptools wheel

echo "==> installing this project (editable) into '$ENV_REF' ..."
# Run from the repo root so the `.[extras]` editable target resolves here.
( cd "${ROOT_DIR}" && "$MANAGER" run "${MGR_TARGET[@]}" python -m pip install -e "${PIP_TARGET}" )

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
echo
echo "Done. Environment '$ENV_REF' is ready."
echo "  python: $("$MANAGER" run "${MGR_TARGET[@]}" python --version 2>&1)"
echo
echo "Activate it with:"
echo "  $MANAGER activate $ENV_REF"
if [[ -n "$PREFIX" ]]; then
  echo "Legacy Ollama launcher: scripts/legacy/ollama_torchrun_job.py --env-path $PREFIX"
fi
if [[ ",$extras_norm," == *",co_bench,"* ]]; then
  echo
  echo "co_bench extra installed -- also fetch the CO-Bench checkout + dataset:"
  echo "  $MANAGER run ${MGR_TARGET[*]} bash ${ROOT_DIR}/scripts/bootstrap_cobench.sh"
fi
