#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXP="manager/universal_token/all_modes/sonic_h2lora"
MOTION_FILE="${H2LORA_MOTION_FILE:?Set H2LORA_MOTION_FILE to the retargeted H2 headless MotionLib PKL or directory.}"
SMPL_MOTION_FILE="${H2LORA_SMPL_MOTION_FILE:-dummy}"
GPU_LIMIT_GB="${H2LORA_GPU_LIMIT_GB:-31.0}"
PYTHON_BIN="${H2LORA_TRAIN_PYTHON:-/home/sky/software/isaac-sim/python.sh}"

best=0
for envs in 1024 2048 3072 4096; do
  echo "== probing num_envs=${envs} =="
  if SWANLAB_MODE=disabled WANDB_MODE=disabled "${PYTHON_BIN}" gear_sonic/train_agent_trl.py \
    +exp="${EXP}" \
    num_envs="${envs}" \
    headless=True \
    algo.config.num_learning_iterations=2 \
    manager_env.commands.motion.motion_lib_cfg.motion_file="${MOTION_FILE}" \
    manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="${SMPL_MOTION_FILE}" \
    manager_env.commands.motion.motion_lib_cfg.cache_all_motions_to_ddr=false \
    +manager_env.commands.motion.motion_lib_cfg.max_unique_motions=2; then
    echo "num_envs=${envs} finished"
    best="${envs}"
  else
    echo "num_envs=${envs} failed; stopping probe"
    break
  fi
done

echo "recommended_num_envs=${best}, gpu_limit_gb=${GPU_LIMIT_GB}"
