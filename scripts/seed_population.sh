#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Seed island populations and stop after generation 0 simple scoring.

Usage:
  ./scripts/seed_population.sh --config-name <preset> [HYDRA_OVERRIDES...]

What it does:
  - runs the dedicated seed-only pipeline
  - creates the initial organisms on every island
  - scores them with the usual simple evaluation phase
  - finalizes generation 0 and exits

Common examples:
  ./scripts/seed_population.sh --config-name config_optimization_survey
  ./scripts/seed_population.sh --config-name config_circle_packing_shinka
  ./scripts/seed_population.sh --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/seed_population.sh --config-name config_optimization_survey api_platforms@api_platforms.gpt_5_4=gpt_5_4 evolver.llm.route_weights='{gpt_5_4: 1.0}'

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - use this script before the main evolution script
  - if the selected config uses local Ollama routes, this wrapper auto-starts Ollama and pulls missing models
  - this wrapper prints script help on --help or -h
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "help" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib_runtime.sh"

has_config_name=0
for arg in "$@"; do
  case "$arg" in
    --config-name|--config-name=*)
      has_config_name=1
      break
      ;;
  esac
done

if [[ "$has_config_name" -ne 1 ]]; then
  echo "Error: this script requires an explicit --config-name <preset>." >&2
  show_help >&2
  exit 2
fi

arm_ollama_cleanup_trap "$ROOT_DIR" "$@"
kill_ollama_runtime "$ROOT_DIR" "$@"
ensure_ollama_runtime "$ROOT_DIR" "$@"

python -m src.evolve.seed_run "$@"
