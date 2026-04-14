#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Stop local Ollama servers configured by a Hydra preset.

Usage:
  ./scripts/kill_ollama.sh --config-name <preset> [HYDRA_OVERRIDES...]

What it does:
  - resolves local Ollama routes from the selected config
  - stops configured local Ollama servers before a rerun or cleanup
  - removes managed pid files for those servers

Common examples:
  ./scripts/kill_ollama.sh --config-name config_circle_packing_shinka
  ./scripts/kill_ollama.sh --config-name config_circle_packing_shinka paths.ollama_cache_root=/tmp/ollama_cache

Notes:
  - all trailing arguments are passed directly to Hydra as overrides
  - only local Ollama routes are affected
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

kill_ollama_runtime "$ROOT_DIR" "$@"
