#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED_ROOT="${SEED_ROOT:-/mnt/data/seed}"
OUT_ROOT="${H2LORA_IDLE_OUT:-/mnt/data/seed/h2lora_motion_lib/idle_debug}"
LIMIT="${H2LORA_IDLE_LIMIT:-8}"
WORKERS="${H2LORA_DATA_WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv-h2lora/bin/python}"
FPS="${H2LORA_DATA_FPS:-50}"
SMPL_MODE="${H2LORA_SMPL_MODE:-soma-bvh}"
SMPL_ROOT_MODE="${H2LORA_SMPL_ROOT_MODE:-soma-bvh-compatible}"
SMPL_BVH_ROOT="${H2LORA_SMPL_BVH_ROOT:-${SEED_ROOT}/soma_uniform/bvh}"

"${PYTHON_BIN}" gear_sonic/data_process/build_h2lora_seed_motion_lib.py \
  --seed-root "${SEED_ROOT}" \
  --output-root "${OUT_ROOT}" \
  --keyword idle \
  --limit "${LIMIT}" \
  --fps-source 120 \
  --fps "${FPS}" \
  --num-workers "${WORKERS}" \
  --smpl-mode "${SMPL_MODE}" \
  --smpl-bvh-root "${SMPL_BVH_ROOT}" \
  --smpl-root-mode "${SMPL_ROOT_MODE}" \
  --skip-existing

echo "H2LORA_MOTION_FILE=${OUT_ROOT}/robot"
echo "H2LORA_SMPL_MOTION_FILE=${OUT_ROOT}/smpl"
