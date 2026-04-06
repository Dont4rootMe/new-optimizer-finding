#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Seed island populations and stop after generation 0 simple scoring.

Usage:
  ./scripts/seed_population.sh [HYDRA_OVERRIDES...]

What it does:
  - runs the dedicated seed-only pipeline
  - creates the initial organisms on every island
  - scores them with the usual simple evaluation phase
  - finalizes generation 0 and exits

Common examples:
  ./scripts/seed_population.sh
  ./scripts/seed_population.sh --config-name config_circle_packing_shinka
  ./scripts/seed_population.sh --config-name config_circle_packing_shinka_ollama_dual
  ./scripts/seed_population.sh --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/seed_population.sh api_platforms@api_platforms.gpt_5_4=gpt_5_4 evolver.llm.route_weights='{gpt_5_4: 1.0}'

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - use this script before the main evolution script
  - this wrapper prints script help on --help or -h
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "help" ]]; then
  show_help
  exit 0
fi

python -m src.evolve.seed_run "$@"
