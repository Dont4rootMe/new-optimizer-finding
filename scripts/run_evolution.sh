#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run the main evolution loop starting from an already seeded population.

Usage:
  ./scripts/run_evolution.sh --config-name <preset> [HYDRA_OVERRIDES...]

What it does:
  - starts the canonical evolution loop for generations >= 1
  - expects an existing seeded population in paths.population_root
  - continues creating, scoring, and selecting organisms until max_generations

Common examples:
  ./scripts/run_evolution.sh --config-name config_optimization_survey
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka_ollama_dual
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_evolution.sh --config-name config_optimization_survey evolver.max_generations=50 evolver.max_proposal_jobs=8

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - run ./scripts/seed_population.sh first for a fresh population root
  - this wrapper forces mode=evolve
  - this wrapper prints script help on --help or -h
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "help" ]]; then
  show_help
  exit 0
fi

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

python -m src.main mode=evolve "$@"
