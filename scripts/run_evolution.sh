#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run the main evolution loop starting from an already seeded population.

Usage:
  ./scripts/run_evolution.sh [--seed] --config-name <preset> [HYDRA_OVERRIDES...]

What it does:
  - starts the canonical evolution loop for generations >= 1
  - expects an existing seeded population in paths.population_root
  - continues creating, scoring, and selecting organisms until max_generations
  - with --seed, auto-runs generation-0 seeding if paths.population_root/gen_0000 is missing

Common examples:
  ./scripts/run_evolution.sh --config-name config_optimization_survey
  ./scripts/run_evolution.sh --seed --config-name config_optimization_survey
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka_ollama_dual
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_evolution.sh --config-name config_optimization_survey evolver.max_generations=50 evolver.max_proposal_jobs=8

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - use --seed to bootstrap a fresh population root automatically
  - this wrapper forces mode=evolve
  - this wrapper prints script help on --help or -h
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${1:-}" == "help" ]]; then
  show_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

auto_seed=0
forward_args=()
for arg in "$@"; do
  case "$arg" in
    --seed)
      auto_seed=1
      ;;
    *)
      forward_args+=("$arg")
      ;;
  esac
done

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
  echo "Error: this script requires an explicit --config-name <preset>." >&2
  show_help >&2
  exit 2
fi

if [[ "$auto_seed" -eq 1 ]]; then
  population_root="$(python - "$ROOT_DIR" "${forward_args[@]}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> int:
    root_dir = Path(sys.argv[1]).resolve()
    cli_args = sys.argv[2:]

    config_name = None
    overrides: list[str] = []
    idx = 0
    while idx < len(cli_args):
        arg = cli_args[idx]
        if arg == "--config-name":
            if idx + 1 >= len(cli_args):
                raise SystemExit("--config-name requires a preset name")
            config_name = cli_args[idx + 1]
            idx += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            idx += 1
            continue
        overrides.append(arg)
        idx += 1

    if not config_name:
        raise SystemExit("missing --config-name")

    with initialize_config_dir(config_dir=str(root_dir / "conf"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    print(Path(str(cfg.paths.population_root)).expanduser().resolve())
    return 0


raise SystemExit(main())
PY
)"

  if [[ ! -d "${population_root}/gen_0000" ]]; then
    echo "Generation 0 population is missing in ${population_root}; running seed_population.sh first."
    "${SCRIPT_DIR}/seed_population.sh" "${forward_args[@]}"
  fi
fi

python -m src.main "${forward_args[@]}" mode=evolve
