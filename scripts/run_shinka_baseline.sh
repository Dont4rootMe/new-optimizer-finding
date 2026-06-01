#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run a ShinkaEvolve baseline using one of the per-experiment configs in conf/baselines/.

Usage:
  ./scripts/run_shinka_baseline.sh --config-name <baseline-preset> [HYDRA_OVERRIDES...]

What it does:
  - composes the chosen baseline Hydra config (which inherits from the main experiment config
    and adds a `shinka_evolve:` block on top)
  - starts/refreshes any Ollama instances declared under api_platforms (same lifecycle as
    scripts/run_evolution.sh — uses scripts/lib_runtime.sh)
  - invokes src.baselines.shinka.run, which constructs ShinkaEvolve dataclasses and runs
    the evolutionary loop with our task evaluator and our local Ollama models

Required config keys (under top-level `shinka_evolve:` in the baseline preset):
  - initial_program_path     path to a seed program with EVOLVE-BLOCK markers
  - evaluate_program_path    path to a python file exposing `def main(program_path, results_dir)`
  - results_dir              where ShinkaEvolve writes its DB / per-program snapshots
  - num_generations          how many generations to run
  - num_islands              ShinkaEvolve island count
  - max_parallel_jobs        parallel evaluator workers

Common examples:
  ./scripts/run_shinka_baseline.sh --config-name baselines/awtf2025_heuristic
  ./scripts/run_shinka_baseline.sh --config-name baselines/circle_packing_shinka
  ./scripts/run_shinka_baseline.sh --config-name baselines/circle_packing_shinka shinka_evolve.num_generations=5 shinka_evolve.num_islands=1

Notes:
  - all trailing arguments are forwarded to Hydra as overrides
  - Ollama lifecycle is identical to run_evolution.sh — same lib_runtime.sh helpers
  - if shinka-evolve is not installed, install with: pip install -e ".[shinka_baseline]"
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "help" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib_runtime.sh"
require_python_bin || exit 1

forward_args=("$@")

has_config_name=0
for arg in "${forward_args[@]}"; do
  case "$arg" in
    --config-name|--config-name=*)
      has_config_name=1
      break
      ;;
  esac
done

if [[ "$has_config_name" -ne 1 ]]; then
  echo "Error: this script requires an explicit --config-name <baseline-preset>." >&2
  show_help >&2
  exit 2
fi

arm_ollama_cleanup_trap "$ROOT_DIR" "${forward_args[@]}"
kill_ollama_runtime "$ROOT_DIR" "${forward_args[@]}"
ensure_ollama_runtime "$ROOT_DIR" "${forward_args[@]}"

"${PYTHON_BIN}" -m src.baselines.shinka.run --project-root "${ROOT_DIR}" "${forward_args[@]}"
