#!/usr/bin/env bash

# CUDA forward-compatibility packages bundle a user-mode libcuda.  On managed
# clusters that library can be older than the host driver mounted by the
# scheduler, which makes an otherwise compatible CUDA runtime fail with error
# 803.  Keep the scheduler's native driver libraries and remove only path
# components named `compat` (empty components are also unsafe because they
# make the dynamic loader search the current working directory).

sanitize_cuda_driver_path() {
  local original_path="${LD_LIBRARY_PATH-}"
  local cleaned_path=""
  local segment
  local removed_count=0
  local -a path_segments=()

  IFS=':' read -r -a path_segments <<< "$original_path"
  for segment in "${path_segments[@]}"; do
    if [[ -z "$segment" ]]; then
      continue
    fi
    case "/${segment#/}/" in
      */compat/*)
        removed_count=$((removed_count + 1))
        ;;
      *)
        cleaned_path="${cleaned_path:+${cleaned_path}:}${segment}"
        ;;
    esac
  done

  export LD_LIBRARY_PATH="$cleaned_path"
  if [[ "$removed_count" -gt 0 ]]; then
    echo "[cuda-driver-env] removed ${removed_count} CUDA compat path(s); using scheduler-mounted libcuda" >&2
  fi
}
