#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${H2LORA_EXP_VAR:-h2lora_locomotion_tiny_mixed_encoder_overfit_2048env}"
LOG_DIR="${H2LORA_LOG_DIR:-/tmp/h2lora_longtrain}"
mkdir -p "${LOG_DIR}"

ROBOT_MOTION="${H2LORA_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/locomotion_tiny_mixed_encoder_50hz/robot}"
SMPL_MOTION="${H2LORA_SMPL_MOTION_FILE:-/mnt/data/seed/h2lora_motion_lib/locomotion_tiny_mixed_encoder_50hz/smpl}"
PYTHON_BIN="${H2LORA_TRAIN_PYTHON:-/home/sky/software/isaac-sim/python.sh}"

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
echo "[h2lora-run] robot_motion=${ROBOT_MOTION}"
echo "[h2lora-run] smpl_motion=${SMPL_MOTION}"
set +e
"${PYTHON_BIN}" gear_sonic/train_agent_trl.py \\
  +exp=manager/universal_token/all_modes/sonic_h2lora \\
  exp_var="${EXP_NAME}" \\
  num_envs="${H2LORA_NUM_ENVS:-2048}" \\
  headless=True \\
  use_wandb=false \\
  use_swanlab="${H2LORA_USE_SWANLAB:-true}" \\
  +resume=false \\
  algo.config.num_learning_iterations="${H2LORA_MAX_ITERATIONS:-3000}" \\
  algo.trl.num_total_batches="${H2LORA_MAX_ITERATIONS:-3000}" \\
  algo.trl.total_episodes=null \\
  algo.config.num_steps_per_env="${H2LORA_NUM_STEPS_PER_ENV:-24}" \\
  algo.config.num_mini_batches="${H2LORA_NUM_MINI_BATCHES:-4}" \\
  algo.config.num_learning_epochs="${H2LORA_NUM_LEARNING_EPOCHS:-5}" \\
  algo.config.actor_learning_rate="${H2LORA_ACTOR_LR:-5.0e-6}" \\
  algo.config.critic_learning_rate="${H2LORA_CRITIC_LR:-1.0e-5}" \\
  algo.config.max_grad_norm="${H2LORA_MAX_GRAD_NORM:-0.05}" \\
  algo.config.std_clamp_min="${H2LORA_STD_CLAMP_MIN:-0.001}" \\
  algo.config.std_clamp_max="${H2LORA_STD_CLAMP_MAX:-0.5}" \\
  algo.config.action_mean_clip="${H2LORA_ACTION_MEAN_CLIP:-20.0}" \\
  manager_env.commands.motion.encoder_sample_probs.g1="${H2LORA_ENCODER_PROB_G1:-1.0}" \\
  manager_env.commands.motion.encoder_sample_probs.teleop="${H2LORA_ENCODER_PROB_TELEOP:-1.0}" \\
  manager_env.commands.motion.encoder_sample_probs.smpl="${H2LORA_ENCODER_PROB_SMPL:-1.0}" \\
  manager_env.commands.motion.teleop_sample_prob_when_smpl="${H2LORA_TELEOP_WHEN_SMPL:-0.5}" \\
  manager_env.commands.motion.motion_lib_cfg.motion_file="${ROBOT_MOTION}" \\
  manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="${SMPL_MOTION}" \\
  manager_env.commands.motion.motion_lib_cfg.cache_all_motions_to_ddr=true \\
  manager_env.commands.motion.motion_lib_cfg.max_cpu_workers="${H2LORA_MAX_CPU_WORKERS:-8}" \\
  manager_env.commands.motion.motion_lib_cfg.ddr_cache_max_bytes="${H2LORA_DDR_CACHE_MAX_BYTES:-17179869184}" \\
  callbacks.model_save.save_frequency="${H2LORA_SAVE_FREQUENCY:-500}" \\
  +callbacks.model_save.save_last_frequency="${H2LORA_SAVE_LAST_FREQUENCY:-20}" \\
  swanlab.project="${H2LORA_SWANLAB_PROJECT:-sonic-h2lora}" \\
  swanlab.group="${H2LORA_SWANLAB_GROUP:-h2lora-locomotion-tiny-mixed-encoder}" \\
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
