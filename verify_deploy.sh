#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "Checking native deployment prerequisites (no robot connection will be made)."
for command_name in cmake g++ make; do
  command -v "$command_name" >/dev/null || { echo "Missing command: $command_name" >&2; exit 1; }
done
[[ -n "${TensorRT_ROOT:-}" ]] || { echo "Set TensorRT_ROOT to a TensorRT 10.x installation." >&2; exit 1; }
[[ -f "$TensorRT_ROOT/include/NvInfer.h" ]] || { echo "TensorRT headers not found under $TensorRT_ROOT" >&2; exit 1; }
[[ -f "$ROOT/gear_sonic_deploy/thirdparty/unitree_sdk2/CMakeLists.txt" ]]
[[ -f "$ROOT/gear_sonic_deploy/policy/release/observation_config_h2_native.yaml" ]]
echo "Native prerequisites: PASS"
