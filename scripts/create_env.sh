#!/usr/bin/env bash
set -euo pipefail

# Create a conda/micromamba environment for the organism-framework project and
# pip-install this repo (editable) plus the requested optional extras into it.
#
# The project is a normal pip package (pyproject.toml), so the env manager only
# provides the Python interpreter; dependencies are installed with pip inside
# the new env via `<manager> run -n <name> ...` (no activation needed).

show_help() {
  cat <<'EOF'
Create a conda / micromamba env for this project and install it (editable).

Usage:
  ./scripts/create_env.sh <env-name> [options]
  ./scripts/create_env.sh --name <env-name> [options]

Required:
  <env-name> | -n, --name NAME   Name of the environment to create.

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
  -f, --force          If the env already exists, remove and recreate it.
  -y, --yes            Non-interactive (assume yes for the manager prompts).
  -h, --help           Show this help.

Examples:
  ./scripts/create_env.sh optfind
  ./scripts/create_env.sh --name optfind --manager micromamba
  ./scripts/create_env.sh optfind -p 3.11 -e evolve,co_bench
  ./scripts/create_env.sh optfind --extras all --force
  ./scripts/create_env.sh optfind --extras none      # base deps only

Notes:
  - torch/torchvision come from pip's default index (a CUDA build on Linux, a
    CPU build on macOS); pass a custom index by editing the pip step if you
    need a specific CUDA wheel.
  - The `evolve` extra (default) pulls the LLM/eval/viz stack the evolution
    loop and baselines use (openai, anthropic, comet_ml, plotly, psutil, ...).
  - `shinka_baseline` adds shinka-evolve; `co_bench` adds the dataset/solver
    deps -- the CO-Bench checkout + dataset still need scripts/bootstrap_cobench.sh.
  - After it finishes, activate with:  <manager> activate <env-name>
EOF
}

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
ENV_NAME=""
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
      if [[ -z "$ENV_NAME" ]]; then
        ENV_NAME="$1"; shift
      else
        echo "Error: unexpected positional argument '$1' (env name already set to '$ENV_NAME')." >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$ENV_NAME" ]]; then
  echo "Error: an environment name is required (positional or --name)." >&2
  show_help >&2
  exit 2
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
# Handle an existing env of the same name
# --------------------------------------------------------------------------
env_exists() {
  "$MANAGER" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

echo "==> manager : $MANAGER"
echo "==> env name: $ENV_NAME"
echo "==> python  : $PY_VERSION"
echo "==> channel : $CHANNEL"
echo "==> extras  : ${extras_norm:-<none>}"
echo "==> install : pip install -e \"${PIP_TARGET}\"  (from ${ROOT_DIR})"

if env_exists; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "==> env '$ENV_NAME' exists; removing (--force)..."
    "$MANAGER" env remove -n "$ENV_NAME" ${YES_FLAG} || "$MANAGER" remove -n "$ENV_NAME" --all ${YES_FLAG}
  else
    echo "Error: env '$ENV_NAME' already exists. Re-run with --force to recreate it." >&2
    exit 1
  fi
fi

# --------------------------------------------------------------------------
# Create the env + pip install (editable) into it
# --------------------------------------------------------------------------
echo "==> creating env '$ENV_NAME' (python=$PY_VERSION) with $MANAGER ..."
"$MANAGER" create ${YES_FLAG} -n "$ENV_NAME" -c "$CHANNEL" "python=${PY_VERSION}" pip

echo "==> upgrading pip tooling in '$ENV_NAME' ..."
"$MANAGER" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

echo "==> installing this project (editable) into '$ENV_NAME' ..."
# Run from the repo root so the `.[extras]` editable target resolves here.
( cd "${ROOT_DIR}" && "$MANAGER" run -n "$ENV_NAME" python -m pip install -e "${PIP_TARGET}" )

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
echo
echo "Done. Environment '$ENV_NAME' is ready."
echo "  python: $("$MANAGER" run -n "$ENV_NAME" python --version 2>&1)"
echo
echo "Activate it with:"
echo "  $MANAGER activate $ENV_NAME"
if [[ ",$extras_norm," == *",co_bench,"* ]]; then
  echo
  echo "co_bench extra installed -- also fetch the CO-Bench checkout + dataset:"
  echo "  $MANAGER run -n $ENV_NAME bash ${ROOT_DIR}/scripts/bootstrap_cobench.sh"
fi
