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
    or max_organism_creations, whichever enabled limit is reached first
  - with --seed, auto-runs generation-0 seeding if paths.population_root/gen_0000 is missing

Common examples:
  ./scripts/run_evolution.sh --config-name config_optimization_survey
  ./scripts/run_evolution.sh --seed --config-name config_optimization_survey
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka
  ./scripts/run_evolution.sh --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_evolution.sh --seed --config-name config_circle_packing_shinka evolver.max_generations=false evolver.max_organism_creations=200
  ./scripts/run_evolution.sh --config-name config_optimization_survey evolver.max_generations=50 evolver.creation.max_parallel_organisms=8

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - use --seed to bootstrap a fresh population root automatically
  - if the selected config uses local Ollama routes, this wrapper auto-starts Ollama and pulls missing models
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
source "${SCRIPT_DIR}/lib_runtime.sh"
require_python_bin || exit 1

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

arm_ollama_cleanup_trap "$ROOT_DIR" "${forward_args[@]}"
kill_ollama_runtime "$ROOT_DIR" "${forward_args[@]}"

inspect_population_state() {
  "${PYTHON_BIN}" - "$ROOT_DIR" "__codex_inspect_population__" "${forward_args[@]}" <<'PY'
from __future__ import annotations

import json
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
        if arg.startswith("__codex_inspect_"):
            idx += 1
            continue
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
    population_root = Path(str(cfg.paths.population_root)).expanduser().resolve()
    state_path = population_root / "population_state.json"

    status = "ready"
    message = ""
    if not state_path.exists():
        if population_root.exists() and any(population_root.iterdir()):
            status = "stale_missing_state"
            message = (
                "population_state.json is required for seeded-only evolution. "
                "Run scripts/seed_population.sh first."
            )
        else:
            status = "missing_state"
            message = (
                "population_state.json is required for seeded-only evolution. "
                "Run scripts/seed_population.sh first."
            )
    else:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        inflight_seed = payload.get("inflight_seed")
        active_organisms = payload.get("active_organisms")
        if isinstance(inflight_seed, dict):
            status = "inflight_seed"
            message = (
                "population_state.json contains inflight_seed. "
                "Continue seeding with scripts/seed_population.sh before running evolution."
            )
        elif not isinstance(active_organisms, list) or len(active_organisms) == 0:
            status = "empty_population"
            message = (
                "population_state.json contains no active organisms. "
                "Run scripts/seed_population.sh to initialize the population first."
            )

    print(population_root)
    print(status)
    print(message)
    return 0


raise SystemExit(main())
PY
}

prompt_seed_now() {
  local reply
  if ! read -r -p "Run seed_population.sh now for this run? [Y/n] " reply; then
    echo "No interactive input available to confirm reseeding." >&2
    return 1
  fi
  case "$reply" in
    ""|[Yy]|[Yy][Ee][Ss])
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "$auto_seed" -eq 1 ]]; then
  mapfile -t seed_inspect < <(inspect_population_state)
  population_root="${seed_inspect[0]}"
  population_status="${seed_inspect[1]:-ready}"
  population_message="${seed_inspect[2]:-}"

  case "$population_status" in
    ready)
      ;;
    missing_state)
      echo "Generation 0 population is missing in ${population_root}; running seed_population.sh first."
      "${SCRIPT_DIR}/seed_population.sh" "${forward_args[@]}"
      ;;
    inflight_seed)
      echo "${population_message}"
      "${SCRIPT_DIR}/seed_population.sh" "${forward_args[@]}"
      ;;
    empty_population|stale_missing_state)
      echo "${population_message}"
      if ! prompt_seed_now; then
        exit 1
      fi
      if [[ -e "${population_root}" ]]; then
        backup_root="${population_root}.stale.$(date +%Y%m%d-%H%M%S)"
        echo "Backing up stale population root to ${backup_root} before reseeding."
        mv "${population_root}" "${backup_root}"
      fi
      "${SCRIPT_DIR}/seed_population.sh" "${forward_args[@]}"
      ;;
    *)
      echo "Unexpected seed inspection status '${population_status}' for ${population_root}." >&2
      exit 1
      ;;
  esac

  mapfile -t post_seed_inspect < <(inspect_population_state)
  post_seed_status="${post_seed_inspect[1]:-ready}"
  post_seed_message="${post_seed_inspect[2]:-}"
  if [[ "$post_seed_status" != "ready" ]]; then
    echo "Seed run did not produce an active generation-0 population." >&2
    if [[ -n "$post_seed_message" ]]; then
      echo "$post_seed_message" >&2
    fi
    echo "Check the seed logs above. For Ollama routes, verify OLLAMA_BASE_URL and that the Ollama server is reachable from this node." >&2
    exit 1
  fi
fi

ensure_ollama_runtime "$ROOT_DIR" "${forward_args[@]}"

"${PYTHON_BIN}" -m src.main "${forward_args[@]}" mode=evolve
