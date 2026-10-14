#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_DIR="${SWANLAB_ACTIVE_RUN_DIR:?SWANLAB_ACTIVE_RUN_DIR is required}"
RUN_ID="${SWANLAB_ACTIVE_RUN_ID:?SWANLAB_ACTIVE_RUN_ID is required}"
PROJECT="${SWANLAB_ACTIVE_PROJECT:-sonic-h2lora}"
WORKSPACE="${SWANLAB_ACTIVE_WORKSPACE:-}"
PYTHON_BIN="${SWANLAB_ACTIVE_PYTHON:-.venv-h2lora/bin/python}"
INTERVAL_SECONDS="${SWANLAB_ACTIVE_SYNC_INTERVAL:-600}"
MIN_GROWTH_BYTES="${SWANLAB_ACTIVE_SYNC_MIN_GROWTH_BYTES:-1048576}"
LOG_PREFIX="${SWANLAB_ACTIVE_LOG_PREFIX:-/tmp/h2lora_longtrain/swanlab_active_sync_${RUN_ID}}"
TRAIN_PID="${SWANLAB_ACTIVE_TRAIN_PID:-}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[swanlab-active-watch] Missing run dir: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/backup.swanlab" ]]; then
  echo "[swanlab-active-watch] Missing backup file: ${RUN_DIR}/backup.swanlab" >&2
  exit 1
fi

last_synced_size=0
echo "[swanlab-active-watch] started_at=$(date --iso-8601=seconds)"
echo "[swanlab-active-watch] run_dir=${RUN_DIR}"
echo "[swanlab-active-watch] run_id=${RUN_ID}"
echo "[swanlab-active-watch] project=${PROJECT}"
echo "[swanlab-active-watch] interval=${INTERVAL_SECONDS}s min_growth=${MIN_GROWTH_BYTES}B"

while true; do
  if [[ -n "${TRAIN_PID}" ]] && ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "[swanlab-active-watch] train_pid=${TRAIN_PID} ended; doing one final active sync then exiting"
    force_sync=true
  else
    force_sync=false
  fi

  backup_size=$(stat -c '%s' "${RUN_DIR}/backup.swanlab")
  growth=$(( backup_size - last_synced_size ))
  if [[ "${force_sync}" == "true" || "${last_synced_size}" -eq 0 || "${growth}" -ge "${MIN_GROWTH_BYTES}" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    sync_log="${LOG_PREFIX}_${stamp}.log"
    echo "[swanlab-active-watch] sync start size=${backup_size} growth=${growth} log=${sync_log}"
    set +e
    if [[ -n "${WORKSPACE}" ]]; then
      "${PYTHON_BIN}" scripts/swanlab_safe_active_sync.py \
        --run-dir "${RUN_DIR}" \
        --run-id "${RUN_ID}" \
        --project "${PROJECT}" \
        --workspace "${WORKSPACE}" \
        >"${sync_log}" 2>&1
    else
      "${PYTHON_BIN}" scripts/swanlab_safe_active_sync.py \
        --run-dir "${RUN_DIR}" \
        --run-id "${RUN_ID}" \
        --project "${PROJECT}" \
        >"${sync_log}" 2>&1
    fi
    status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
      last_synced_size="${backup_size}"
      echo "[swanlab-active-watch] sync ok size=${backup_size}"
    else
      echo "[swanlab-active-watch] sync failed status=${status} log=${sync_log}"
      tail -n 40 "${sync_log}" || true
    fi
  else
    echo "[swanlab-active-watch] skip size=${backup_size} growth=${growth}"
  fi

  if [[ "${force_sync}" == "true" ]]; then
    echo "[swanlab-active-watch] finished_at=$(date --iso-8601=seconds)"
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
