#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${H2LORA_EXP_VAR:-h2lora_seed5m50hz_release_lora_10k_6144env}"
LOG_DIR="${H2LORA_LOG_DIR:-/tmp/h2lora_longtrain}"
mkdir -p "${LOG_DIR}"

ROBOT_MOTION="${H2LORA_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6/robot}"
SMPL_MOTION="${H2LORA_SMPL_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6/smpl}"
PYTHON_BIN="${H2LORA_TRAIN_PYTHON:-/home/sky/software/isaac-sim/python.sh}"
CHECKPOINT="${H2LORA_CHECKPOINT:-}"
RESUME_FLAG="${H2LORA_RESUME:-false}"

TRAIN_LOG="${LOG_DIR}/${EXP_NAME}_${RUN_STAMP}.log"
GPU_LOG="${LOG_DIR}/${EXP_NAME}_${RUN_STAMP}_gpu.csv"
PID_FILE="${LOG_DIR}/${EXP_NAME}_${RUN_STAMP}.pid"
GPU_PID_FILE="${LOG_DIR}/${EXP_NAME}_${RUN_STAMP}_gpu.pid"
TRAIN_CMD="${LOG_DIR}/${EXP_NAME}_${RUN_STAMP}_train_cmd.sh"

if [[ ! -d "${ROBOT_MOTION}" ]]; then
  echo "Missing robot motion directory: ${ROBOT_MOTION}" >&2
  exit 1
fi
if [[ ! -d "${SMPL_MOTION}" ]]; then
  echo "Missing SMPL motion directory: ${SMPL_MOTION}" >&2
  exit 1
fi
if [[ ! -f "sonic_release/last.pt" ]]; then
  echo "Missing Sonic release checkpoint: sonic_release/last.pt" >&2
  exit 1
fi
if [[ -n "${CHECKPOINT}" && ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cat >"${TRAIN_CMD}" <<EOF
#!/usr/bin/env bash
set -u
cd "$(pwd)"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA}"
export ACCEPT_EULA="${ACCEPT_EULA}"
export SWANLAB_MODE="${SWANLAB_MODE}"
export WANDB_MODE="${WANDB_MODE}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED}"
echo "[h2lora-run] started_at=\$(date --iso-8601=seconds)"
echo "[h2lora-run] python=${PYTHON_BIN}"
echo "[h2lora-run] pretrained=sonic_release/last.pt"
echo "[h2lora-run] checkpoint=${CHECKPOINT}"
echo "[h2lora-run] resume=${RESUME_FLAG}"
echo "[h2lora-run] robot_motion=${ROBOT_MOTION}"
echo "[h2lora-run] smpl_motion=${SMPL_MOTION}"
EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(checkpoint="${CHECKPOINT}" +resume="${RESUME_FLAG}")
else
  EXTRA_ARGS+=(+resume=false)
fi
set +e
"${PYTHON_BIN}" gear_sonic/train_agent_trl.py \\
  +exp=manager/universal_token/all_modes/sonic_h2lora \\
  exp_var="${EXP_NAME}" \\
  num_envs="${H2LORA_NUM_ENVS:-6144}" \\
  headless=True \\
  use_wandb=false \\
  use_swanlab="${H2LORA_USE_SWANLAB:-true}" \\
  "\${EXTRA_ARGS[@]}" \\
  algo.config.num_learning_iterations="${H2LORA_MAX_ITERATIONS:-10000}" \\
  algo.trl.num_total_batches="${H2LORA_MAX_ITERATIONS:-10000}" \\
  algo.trl.total_episodes=null \\
  algo.config.num_steps_per_env="${H2LORA_NUM_STEPS_PER_ENV:-24}" \\
  algo.config.num_mini_batches="${H2LORA_NUM_MINI_BATCHES:-4}" \\
  algo.config.num_learning_epochs="${H2LORA_NUM_LEARNING_EPOCHS:-5}" \\
  algo.config.actor_learning_rate="${H2LORA_ACTOR_LR:-2.5e-6}" \\
  algo.config.critic_learning_rate="${H2LORA_CRITIC_LR:-5.0e-6}" \\
  algo.config.max_grad_norm="${H2LORA_MAX_GRAD_NORM:-0.03}" \\
  algo.config.std_clamp_min="${H2LORA_STD_CLAMP_MIN:-0.001}" \\
  algo.config.std_clamp_max="${H2LORA_STD_CLAMP_MAX:-0.38}" \\
  algo.config.action_mean_clip="${H2LORA_ACTION_MEAN_CLIP:-12.0}" \\
  manager_env.rewards.anti_shake_ang_vel.weight="${H2LORA_ANTI_SHAKE_WEIGHT:--1.5e-2}" \\
  manager_env.rewards.body_ang_vel_l2.weight="${H2LORA_BODY_ANG_VEL_WEIGHT:--2.0e-4}" \\
  manager_env.rewards.feet_acc.weight="${H2LORA_FEET_ACC_WEIGHT:--3.5e-6}" \\
  callbacks.model_save.save_frequency="${H2LORA_SAVE_FREQUENCY:-1000}" \\
  +callbacks.model_save.save_last_frequency="${H2LORA_SAVE_LAST_FREQUENCY:-20}" \\
  manager_env.commands.motion.motion_lib_cfg.motion_file="${ROBOT_MOTION}" \\
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="${SMPL_MOTION}" \\
  manager_env.commands.motion.motion_lib_cfg.cache_all_motions_to_ddr=true \\
  manager_env.commands.motion.motion_lib_cfg.max_cpu_workers="${H2LORA_MAX_CPU_WORKERS:-16}" \\
  manager_env.commands.motion.motion_lib_cfg.ddr_cache_max_bytes="${H2LORA_DDR_CACHE_MAX_BYTES:-49392123904}" \\
  swanlab.project="${H2LORA_SWANLAB_PROJECT:-sonic-h2lora}" \\
  swanlab.group="${H2LORA_SWANLAB_GROUP:-h2lora-seed5m-50hz-release}" \\
  swanlab.job_type=train
STATUS=\$?
echo "[h2lora-run] exit_status=\${STATUS}"
echo "[h2lora-run] finished_at=\$(date --iso-8601=seconds)"
exit "\${STATUS}"
EOF
chmod +x "${TRAIN_CMD}"

setsid "${TRAIN_CMD}" >"${TRAIN_LOG}" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "${TRAIN_PID}" >"${PID_FILE}"

cat >"${GPU_LOG}" <<'EOF'
timestamp,index,memory_used_mb,memory_total_mb,utilization_gpu_pct,utilization_memory_pct,power_w,power_limit_w,temperature_c
EOF

setsid bash -c "
  while kill -0 '${TRAIN_PID}' 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu \
      --format=csv,noheader,nounits >>'${GPU_LOG}' 2>/dev/null || true
    sleep '${H2LORA_GPU_MONITOR_INTERVAL:-5}'
  done
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu \
    --format=csv,noheader,nounits >>'${GPU_LOG}' 2>/dev/null || true
" >/dev/null 2>&1 &
GPU_PID=$!
echo "${GPU_PID}" >"${GPU_PID_FILE}"

echo "train_pid=${TRAIN_PID}"
echo "gpu_pid=${GPU_PID}"
echo "train_log=${TRAIN_LOG}"
echo "gpu_log=${GPU_LOG}"
echo "pid_file=${PID_FILE}"
echo "gpu_pid_file=${GPU_PID_FILE}"
echo "train_cmd=${TRAIN_CMD}"
