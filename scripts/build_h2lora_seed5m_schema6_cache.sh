#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv-h2lora/bin/python}"
SEED_ROOT="${H2LORA_SEED_ROOT:-/mnt/data/seed}"
SOURCE_OUT="${H2LORA_SEED5M_OUT:-/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6}"
BALANCED_OUT="${H2LORA_BALANCED_OUT:-/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6}"
TARGET_FRAMES="${H2LORA_TARGET_FRAMES:-5000000}"
SOURCE_RANDOM_SEED="${H2LORA_SOURCE_RANDOM_SEED:-20260602}"
BALANCED_RANDOM_SEED="${H2LORA_BALANCED_RANDOM_SEED:-20260603}"
WORKERS="${H2LORA_DATA_WORKERS:-16}"
FPS_SOURCE="${H2LORA_FPS_SOURCE:-120}"
FPS="${H2LORA_FPS:-50}"
SMPL_BVH_ROOT="${H2LORA_SMPL_BVH_ROOT:-${SEED_ROOT}/soma_uniform/bvh}"
BUCKET_QUOTAS="${H2LORA_BUCKET_QUOTAS:-locomotion=0.60,dynamic=0.20,general=0.15,static=0.05}"
REQUIRED_KEYS="${H2LORA_BALANCED_REQUIRED_KEYS:-walk_forward_loop_002__A036,jog_forward_loop_003__A029_M,kozak_dance_002__A354,high_jump_R_002__A350_M}"

if [[ "${H2LORA_OVERWRITE_SOURCE:-false}" == "true" && -d "${SOURCE_OUT}" ]]; then
  rm -rf "${SOURCE_OUT}"
fi

echo "[h2lora-cache] source_out=${SOURCE_OUT}"
echo "[h2lora-cache] balanced_out=${BALANCED_OUT}"
echo "[h2lora-cache] fps=${FPS_SOURCE}->${FPS} target_frames=${TARGET_FRAMES} workers=${WORKERS}"
echo "[h2lora-cache] smpl_mode=soma-bvh smpl_root_mode=soma-bvh-compatible"

"${PYTHON_BIN}" gear_sonic/data_process/build_h2lora_seed_motion_lib.py \
  --seed-root "${SEED_ROOT}" \
  --output-root "${SOURCE_OUT}" \
  --fps-source "${FPS_SOURCE}" \
  --fps "${FPS}" \
  --selection-mode random \
  --target-frames "${TARGET_FRAMES}" \
  --random-seed "${SOURCE_RANDOM_SEED}" \
  --num-workers "${WORKERS}" \
  --smpl-mode soma-bvh \
  --smpl-bvh-root "${SMPL_BVH_ROOT}" \
  --smpl-root-mode soma-bvh-compatible \
  --skip-existing

BALANCED_ARGS=()
if [[ "${H2LORA_OVERWRITE_BALANCED:-false}" == "true" ]]; then
  BALANCED_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" scripts/build_h2lora_balanced_from_motionlib.py \
  --source-root "${SOURCE_OUT}" \
  --output-root "${BALANCED_OUT}" \
  --target-frames "${TARGET_FRAMES}" \
  --random-seed "${BALANCED_RANDOM_SEED}" \
  --bucket-quotas "${BUCKET_QUOTAS}" \
  --required-keys "${REQUIRED_KEYS}" \
  --require-fps "${FPS}" \
  --require-smpl-schema-version 6 \
  --require-smpl-source soma_bvh_fk_name_mapped_to_smpl24 \
  --require-smpl-root-mode soma-bvh-compatible \
  "${BALANCED_ARGS[@]}"

echo "[h2lora-cache] done"
echo "[h2lora-cache] robot_motion=${BALANCED_OUT}/robot"
echo "[h2lora-cache] smpl_motion=${BALANCED_OUT}/smpl"
