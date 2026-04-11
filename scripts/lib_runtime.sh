#!/usr/bin/env bash

resolve_ollama_requirements() {
  local root_dir="$1"
  shift
  python - "$root_dir" "__codex_inspect_ollama__" "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from api_platforms._core.discovery import load_route_configs


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
    runtime_root = Path(str(cfg.paths.api_platform_runtime_root)).expanduser().resolve()
    print(f"API_PLATFORM_RUNTIME_ROOT={runtime_root}")
    for route_id, route_cfg in sorted(load_route_configs(cfg).items()):
        if route_cfg.backend != "ollama":
            continue
        base_url = str(route_cfg.base_url or "http://127.0.0.1:11434/api").rstrip("/")
        gpu_ranks = ",".join(str(rank) for rank in route_cfg.gpu_ranks)
        print(f"OLLAMA_ROUTE={route_id}|{base_url}|{route_cfg.provider_model_id}|{gpu_ranks}")
    return 0


raise SystemExit(main())
PY
}

_normalize_ollama_base_url() {
  local base_url="${1:-http://127.0.0.1:11434/api}"
  base_url="${base_url%/}"
  base_url="${base_url/localhost/127.0.0.1}"
  printf '%s\n' "$base_url"
}

_ollama_tags_url() {
  local base_url
  base_url="$(_normalize_ollama_base_url "$1")"
  if [[ "$base_url" == */api/chat ]]; then
    printf '%s\n' "${base_url%/chat}/tags"
    return
  fi
  if [[ "$base_url" == */api ]]; then
    printf '%s\n' "${base_url}/tags"
    return
  fi
  printf '%s\n' "${base_url}/api/tags"
}

_ollama_pull_url() {
  local base_url
  base_url="$(_normalize_ollama_base_url "$1")"
  if [[ "$base_url" == */api/chat ]]; then
    printf '%s\n' "${base_url%/chat}/pull"
    return
  fi
  if [[ "$base_url" == */api ]]; then
    printf '%s\n' "${base_url}/pull"
    return
  fi
  printf '%s\n' "${base_url}/api/pull"
}

_ollama_origin() {
  local base_url
  base_url="$(_normalize_ollama_base_url "$1")"
  base_url="${base_url#http://}"
  base_url="${base_url#https://}"
  base_url="${base_url%%/*}"
  printf '%s\n' "$base_url"
}

_ollama_is_local() {
  local origin host
  origin="$(_ollama_origin "$1")"
  host="${origin%%:*}"
  case "$host" in
    127.0.0.1|localhost)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_ollama_healthcheck() {
  curl -fsS --max-time 2 "$(_ollama_tags_url "$1")" >/dev/null 2>&1
}

_ollama_model_present() {
  local base_url="$1"
  local model="$2"
  curl -fsS --max-time 5 "$(_ollama_tags_url "$base_url")" | python -c '
from __future__ import annotations

import json
import sys

model = sys.argv[1]
payload = json.load(sys.stdin)
names = {
    str(entry.get("name", "")).strip()
    for entry in payload.get("models", [])
    if isinstance(entry, dict)
}
raise SystemExit(0 if model in names else 1)
' "$model"
}

_stream_ollama_pull_progress() {
  local model="$1"
  python -c '
from __future__ import annotations

import json
import sys

model = sys.argv[1]
last_status = None
last_progress_key = None

def _format_bytes(value: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    unit = 0
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        print(f"  {line}", flush=True)
        continue

    error = payload.get("error")
    if error:
        print(f"  error while pulling {model}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    status = str(payload.get("status", "")).strip()
    completed = payload.get("completed")
    total = payload.get("total")

    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
        percent = max(0, min(100, int((float(completed) / float(total)) * 100)))
        progress_key = (status, percent)
        if progress_key != last_progress_key:
            print(
                f"  {status or 'pulling'}: {percent:3d}% "
                f"({_format_bytes(completed)} / {_format_bytes(total)})",
                flush=True,
            )
            last_progress_key = progress_key
            last_status = status
        continue

    if status and status != last_status:
        print(f"  {status}", flush=True)
        last_status = status
        last_progress_key = None
' "$model"
}

_start_local_ollama() {
  local base_url="$1"
  local runtime_root="$2"
  local gpu_rank="${3:-}"
  local origin log_dir log_name stdout_log stderr_log
  local -a launch_env=()

  if ! command -v ollama >/dev/null 2>&1; then
    echo "Error: config requires a local Ollama route, but the 'ollama' CLI is not installed or not on PATH." >&2
    return 1
  fi

  origin="$(_ollama_origin "$base_url")"
  launch_env=(env "OLLAMA_HOST=${origin}")
  if [[ -n "$gpu_rank" ]]; then
    launch_env+=("CUDA_VISIBLE_DEVICES=${gpu_rank}")
  fi
  log_dir="${runtime_root}/ollama"
  mkdir -p "$log_dir"
  log_name="${origin//[:\/]/_}"
  stdout_log="${log_dir}/serve.${log_name}.stdout.log"
  stderr_log="${log_dir}/serve.${log_name}.stderr.log"

  if [[ -n "$gpu_rank" ]]; then
    echo "Starting local Ollama server for ${base_url} on gpu:${gpu_rank}."
  else
    echo "Starting local Ollama server for ${base_url}."
  fi
  nohup "${launch_env[@]}" ollama serve >"${stdout_log}" 2>"${stderr_log}" </dev/null &

  local attempt
  for attempt in $(seq 1 60); do
    if _ollama_healthcheck "$base_url"; then
      return 0
    fi
    sleep 1
  done

  echo "Error: local Ollama server did not become ready at ${base_url}." >&2
  echo "See logs: ${stdout_log} and ${stderr_log}" >&2
  return 1
}

_ensure_ollama_server() {
  local base_url="$1"
  local runtime_root="$2"
  local gpu_rank="${3:-}"
  if _ollama_healthcheck "$base_url"; then
    return 0
  fi

  if _ollama_is_local "$base_url"; then
    _start_local_ollama "$base_url" "$runtime_root" "$gpu_rank" || return 1
    return 0
  fi

  echo "Error: Ollama route ${base_url} is unreachable and is not local, so the scripts cannot auto-start it." >&2
  return 1
}

_ensure_ollama_model() {
  local base_url="$1"
  local model="$2"
  if _ollama_model_present "$base_url" "$model"; then
    return 0
  fi

  echo "Pulling Ollama model ${model} from ${base_url}."
  curl -fsS --no-buffer --max-time 0 \
    -H 'Content-Type: application/json' \
    -X POST \
    -d "{\"model\":\"${model}\"}" \
    "$(_ollama_pull_url "$base_url")" | _stream_ollama_pull_progress "$model"
}

ensure_ollama_runtime() {
  local root_dir="$1"
  shift

  local runtime_root=""
  local -a ollama_routes=()
  while IFS= read -r line; do
    case "$line" in
      API_PLATFORM_RUNTIME_ROOT=*)
        runtime_root="${line#*=}"
        ;;
      OLLAMA_ROUTE=*)
        ollama_routes+=("${line#*=}")
        ;;
    esac
  done < <(resolve_ollama_requirements "$root_dir" "$@")

  if [[ "${#ollama_routes[@]}" -eq 0 ]]; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "Error: config requires Ollama routes, but 'curl' is not installed or not on PATH." >&2
    return 1
  fi

  if [[ -z "$runtime_root" ]]; then
    runtime_root="${root_dir}/.api_platform_runtime"
  fi

  declare -A server_gpu_by_url=()
  declare -A server_url_by_gpu=()
  local route route_id base_url model gpu_ranks_csv gpu_rank
  for route in "${ollama_routes[@]}"; do
    IFS='|' read -r route_id base_url model gpu_ranks_csv <<<"$route"
    base_url="$(_normalize_ollama_base_url "$base_url")"
    gpu_rank=""
    if [[ -n "$gpu_ranks_csv" ]]; then
      if [[ "$gpu_ranks_csv" == *,* ]]; then
        echo "Error: local Ollama route ${route_id} must define at most one gpu_ranks entry, got '${gpu_ranks_csv}'." >&2
        return 1
      fi
      gpu_rank="$gpu_ranks_csv"
    fi
    if ! _ollama_is_local "$base_url"; then
      continue
    fi
    if [[ -n "${server_gpu_by_url[$base_url]+x}" ]]; then
      if [[ "${server_gpu_by_url[$base_url]}" != "$gpu_rank" ]]; then
        echo "Error: local Ollama base_url ${base_url} is assigned conflicting gpu_ranks (${server_gpu_by_url[$base_url]} vs ${gpu_rank})." >&2
        echo "Use distinct base_url values for routes that must stay on different GPUs." >&2
        return 1
      fi
    else
      server_gpu_by_url["$base_url"]="$gpu_rank"
    fi
    if [[ -n "$gpu_rank" ]]; then
      if [[ -n "${server_url_by_gpu[$gpu_rank]+x}" && "${server_url_by_gpu[$gpu_rank]}" != "$base_url" ]]; then
        echo "Error: local Ollama gpu:${gpu_rank} is assigned to multiple base_url values (${server_url_by_gpu[$gpu_rank]} and ${base_url})." >&2
        echo "Each local Ollama GPU should correspond to exactly one local service base_url." >&2
        return 1
      fi
      server_url_by_gpu["$gpu_rank"]="$base_url"
    fi
  done

  declare -A started_base_urls=()
  for route in "${ollama_routes[@]}"; do
    IFS='|' read -r route_id base_url model gpu_ranks_csv <<<"$route"
    base_url="$(_normalize_ollama_base_url "$base_url")"
    gpu_rank="${server_gpu_by_url[$base_url]-}"
    if [[ -z "${started_base_urls[$base_url]+x}" ]]; then
      _ensure_ollama_server "$base_url" "$runtime_root" "$gpu_rank" || return 1
      started_base_urls["$base_url"]=1
    fi
    _ensure_ollama_model "$base_url" "$model" || {
      echo "Error: failed to prepare Ollama model ${model} for route ${route_id}." >&2
      return 1
    }
  done
}
