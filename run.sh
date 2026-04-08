#!/bin/sh
set -eu

show_help() {
  cat <<'EOF'
Run the Circle Packing Shinka preset with a single entrypoint.

Usage:
  ./scripts/run_circle_packing.sh [--seed-only|--evolve-only] [HYDRA_OVERRIDES...]

What it does:
  - always uses --config-name config_circle_packing_shinka
  - if population_state.json is missing, seeds the population and then runs evolution
  - if population_state.json already exists, resumes evolution directly

Common examples:
  ./scripts/run_circle_packing.sh
  ./scripts/run_circle_packing.sh evolver.max_generations=20
  ./scripts/run_circle_packing.sh paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_circle_packing.sh --seed-only paths.population_root=/tmp/circle_pack_pop
  ./scripts/run_circle_packing.sh --evolve-only paths.population_root=/tmp/circle_pack_pop

Notes:
  - trailing key=value arguments are passed directly to Hydra
  - population root is inferred from paths.population_root=..., then POP_ROOT, then ./populations
  - for custom presets, use the generic seed/evolve entrypoints directly
EOF
}

resolve_dir() {
  path=$1

  case "$path" in
    "~")
      path=$HOME
      ;;
    "~/"*)
      path=$HOME/${path#~/}
      ;;
  esac

  case "$path" in
    /*)
      ;;
    *)
      path=$PWD/$path
      ;;
  esac

  resolved=$(CDPATH= cd "$path" 2>/dev/null && pwd -P) || resolved=$path
  printf '%s\n' "$resolved"
}

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)

if [ -d "$script_dir/src" ] && [ -d "$script_dir/conf" ] && [ -f "$script_dir/pyproject.toml" ]; then
  repo_root=$script_dir
elif [ -d "$script_dir/../src" ] && [ -d "$script_dir/../conf" ] && [ -f "$script_dir/../pyproject.toml" ]; then
  repo_root=$(CDPATH= cd "$script_dir/.." && pwd)
else
  echo "Error: could not locate repository root from $0" >&2
  exit 2
fi

cd "$repo_root"

mode="auto"
mode_is_explicit=
population_root=${POP_ROOT:-./populations}
tmp_args=${TMPDIR:-/tmp}/run_circle_packing_args.$$

trap 'rm -f "$tmp_args"' EXIT HUP INT TERM
: > "$tmp_args"

while [ "$#" -gt 0 ]; do
  arg=$1
  shift
  case "$arg" in
    --help|-h|help)
      show_help
      exit 0
      ;;
    --seed-only)
      if [ -n "$mode_is_explicit" ] && [ "$mode" != "seed" ]; then
        echo "Error: choose only one of --seed-only or --evolve-only." >&2
        exit 2
      fi
      mode="seed"
      mode_is_explicit=1
      ;;
    --evolve-only)
      if [ -n "$mode_is_explicit" ] && [ "$mode" != "evolve" ]; then
        echo "Error: choose only one of --seed-only or --evolve-only." >&2
        exit 2
      fi
      mode="evolve"
      mode_is_explicit=1
      ;;
    --config-name|--config-name=*)
      echo "Error: run_circle_packing.sh is pinned to --config-name config_circle_packing_shinka." >&2
      echo "Use python -m src.evolve.seed_run or python -m src.evolve.run for other presets." >&2
      exit 2
      ;;
    *)
      printf '%s\n' "$arg" >> "$tmp_args"
      case "$arg" in
        paths.population_root=*)
          population_root=${arg#paths.population_root=}
          ;;
      esac
      ;;
  esac
done

set --
while IFS= read -r arg || [ -n "$arg" ]; do
  set -- "$@" "$arg"
done < "$tmp_args"

resolved_population_root=$(resolve_dir "$population_root")
population_state_path=$resolved_population_root/population_state.json

if [ "$mode" = "auto" ]; then
  if [ -f "$population_state_path" ]; then
    mode="evolve"
  else
    mode="seed_then_evolve"
  fi
fi

run_seed() {
  echo "Circle packing: seeding population in $resolved_population_root" >&2
  python -m src.evolve.seed_run --config-name config_circle_packing_shinka "$@"
}

run_evolve() {
  echo "Circle packing: resuming evolution from $resolved_population_root" >&2
  python -m src.evolve.run --config-name config_circle_packing_shinka "$@"
}

if [ "$mode" = "seed" ]; then
  run_seed "$@"
  exit 0
fi

if [ "$mode" = "evolve" ]; then
  run_evolve "$@"
  exit 0
fi

echo "Circle packing: no population_state.json in $resolved_population_root; running seed then evolve" >&2
run_seed "$@"
run_evolve "$@"
