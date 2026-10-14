#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deploy_root="${repo_root}/gear_sonic_deploy"
build_dir="${deploy_root}/build"
binary="${deploy_root}/target/release/h2_static_hold_diag"

if [[ ! -x "${binary}" ]]; then
  echo "[INFO] h2_static_hold_diag not found; building it first..."
  if ! cmake --build "${build_dir}" --target h2_static_hold_diag -j"$(nproc)"; then
    echo "[INFO] Build target was missing; reconfiguring CMake and retrying..."
    cmake -S "${deploy_root}" -B "${build_dir}"
    cmake --build "${build_dir}" --target h2_static_hold_diag -j"$(nproc)"
  fi
fi

exec "${binary}" "$@"
