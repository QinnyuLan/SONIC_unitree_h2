#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="${H2LORA_RUN_DIR:-}"
RUN_GLOB="${H2LORA_RUN_GLOB:-}"
EXP_NAME="${H2LORA_EXP_NAME:-h2lora}"
MOTION_FILE="${H2LORA_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/robot}"
SMPL_MOTION_FILE="${H2LORA_SMPL_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/smpl}"
OUTPUT_ROOT="${H2LORA_VIZ_OUTPUT_ROOT:-visualizations/h2lora_auto}"
INTERVAL="${H2LORA_VIZ_INTERVAL:-2500}"
START_STEP="${H2LORA_VIZ_START_STEP:-${INTERVAL}}"
MAX_STEP="${H2LORA_VIZ_MAX_STEP:-50000}"
NUM_STEPS="${H2LORA_VIZ_NUM_STEPS:-371}"
FPS="${H2LORA_VIZ_FPS:-50}"
ISAAC_PYTHON="${H2LORA_VIZ_ISAAC_PYTHON:-/home/sky/software/isaac-sim/python.sh}"
UV_PYTHON="${H2LORA_VIZ_UV_PYTHON:-.venv-h2lora/bin/python}"
SLEEP_SECONDS="${H2LORA_VIZ_POLL_SECONDS:-30}"
RUN_STAMP="${H2LORA_VIZ_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_MOTIONS="walk_forward_stop_001__A039:g1,walk_forward_stop_001__A039:smpl,jog_forward_loop_002__A037_M:g1,turn_walk_360_001__A149_M:g1,jog_backward_loop_003__A034:g1,jog_sideway_045_loop_002__A025:g1,Relaxed_throw_ball_001__A057_M:g1,neutral_stoop_down_R_007__A100_M:g1,hurry_idle_loop_004__A146_M:g1"
MOTIONS="${H2LORA_VIZ_MOTIONS:-${DEFAULT_MOTIONS}}"
MOTIONS_FILE="${H2LORA_VIZ_MOTIONS_FILE:-}"

if [[ -z "${RUN_DIR}" && -z "${RUN_GLOB}" ]]; then
  echo "[h2lora-viz-watch] H2LORA_RUN_DIR or H2LORA_RUN_GLOB is required" >&2
  exit 1
fi
if [[ ! -d "${MOTION_FILE}" ]]; then
  echo "[h2lora-viz-watch] Missing robot motion dir: ${MOTION_FILE}" >&2
  exit 1
fi
if [[ ! -d "${SMPL_MOTION_FILE}" ]]; then
  echo "[h2lora-viz-watch] Missing SMPL motion dir: ${SMPL_MOTION_FILE}" >&2
  exit 1
fi

resolve_run_dir() {
  if [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
    printf '%s\n' "${RUN_DIR}"
    return 0
  fi
  if [[ -z "${RUN_GLOB}" ]]; then
    return 1
  fi
  shopt -s nullglob
  local matches=(${RUN_GLOB})
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    return 1
  fi
  ls -td "${matches[@]}" 2>/dev/null | head -1
}

echo "[h2lora-viz-watch] started_at=$(date --iso-8601=seconds)"
echo "[h2lora-viz-watch] exp=${EXP_NAME}"
echo "[h2lora-viz-watch] run_glob=${RUN_GLOB}"
echo "[h2lora-viz-watch] output_root=${OUTPUT_ROOT}"
echo "[h2lora-viz-watch] interval=${INTERVAL} start_step=${START_STEP} max_step=${MAX_STEP} motions=${MOTIONS}"
echo "[h2lora-viz-watch] motions_file=${MOTIONS_FILE}"

until RESOLVED_RUN_DIR="$(resolve_run_dir)"; do
  echo "[h2lora-viz-watch] waiting for run dir..."
  sleep "${SLEEP_SECONDS}"
done
RUN_DIR="${RESOLVED_RUN_DIR}"
echo "[h2lora-viz-watch] run_dir=${RUN_DIR}"

load_motion_items() {
  local motions_text="${MOTIONS}"
  if [[ -n "${MOTIONS_FILE}" && -f "${MOTIONS_FILE}" ]]; then
    motions_text="$(
      sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "${MOTIONS_FILE}" \
        | tr '\n' ',' \
        | sed 's/,$//'
    )"
  fi
  IFS=',' read -r -a MOTION_ITEMS <<<"${motions_text}"
  echo "[h2lora-viz-watch] active_motions=${motions_text}"
}

for (( STEP=START_STEP; STEP<=MAX_STEP; STEP+=INTERVAL )); do
  CHECKPOINT="${RUN_DIR}/model_step_$(printf '%06d' "${STEP}").pt"
  while [[ ! -f "${CHECKPOINT}" ]]; do
    if [[ -f "${RUN_DIR}/last.pt" ]] && ! pgrep -af "train_agent_trl.py.*${EXP_NAME}" >/dev/null 2>&1; then
      echo "[h2lora-viz-watch] training appears stopped before checkpoint ${CHECKPOINT}; exiting"
      exit 0
    fi
    echo "[h2lora-viz-watch] waiting for ${CHECKPOINT}"
    sleep "${SLEEP_SECONDS}"
  done

  load_motion_items
  for ITEM in "${MOTION_ITEMS[@]}"; do
    MOTION_KEY="${ITEM%%:*}"
    ENCODER="${ITEM#*:}"
    if [[ "${ENCODER}" == "${ITEM}" ]]; then
      ENCODER="g1"
    fi
    SAFE_MOTION="${MOTION_KEY//[^A-Za-z0-9_.-]/_}"
    OUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}_${RUN_STAMP}/step_$(printf '%06d' "${STEP}")/${SAFE_MOTION}_${ENCODER}"
    VIDEO="${OUT_DIR}/ref_tracking.mp4"
    TRAJ="${OUT_DIR}/trajectory_ref_tracking.npz"
    if [[ -f "${VIDEO}" ]]; then
      echo "[h2lora-viz-watch] skip existing ${VIDEO}"
      continue
    fi

    mkdir -p "${OUT_DIR}"
    echo "[h2lora-viz-watch] collect step=${STEP} motion=${MOTION_KEY} encoder=${ENCODER}"
    set +e
    "${ISAAC_PYTHON}" scripts/visualize_h2lora_ref_tracking.py \
      --checkpoint "${CHECKPOINT}" \
      --motion-file "${MOTION_FILE}" \
      --smpl-motion-file "${SMPL_MOTION_FILE}" \
      --motion-key "${MOTION_KEY}" \
      --use-encoder "${ENCODER}" \
      --num-steps "${NUM_STEPS}" \
      --fps "${FPS}" \
      --action-mode mean \
      --collect-only \
      --output-dir "${OUT_DIR}" \
      --label "step ${STEP} ${MOTION_KEY} ${ENCODER}"
    COLLECT_STATUS=$?
    set -e
    if [[ "${COLLECT_STATUS}" -ne 0 || ! -f "${TRAJ}" ]]; then
      echo "[h2lora-viz-watch] collect failed step=${STEP} motion=${MOTION_KEY} encoder=${ENCODER} status=${COLLECT_STATUS}"
      continue
    fi

    echo "[h2lora-viz-watch] render step=${STEP} motion=${MOTION_KEY} encoder=${ENCODER}"
    set +e
    "${UV_PYTHON}" scripts/visualize_h2lora_ref_tracking.py \
      --render-only \
      --trajectory "${TRAJ}" \
      --output-dir "${OUT_DIR}" \
      --fps "${FPS}" \
      --label "step ${STEP} ${MOTION_KEY} ${ENCODER}"
    RENDER_STATUS=$?
    set -e
    if [[ "${RENDER_STATUS}" -ne 0 ]]; then
      echo "[h2lora-viz-watch] render failed step=${STEP} motion=${MOTION_KEY} encoder=${ENCODER} status=${RENDER_STATUS}"
    else
      echo "[h2lora-viz-watch] video=${VIDEO}"
    fi
  done
done

echo "[h2lora-viz-watch] finished_at=$(date --iso-8601=seconds)"
