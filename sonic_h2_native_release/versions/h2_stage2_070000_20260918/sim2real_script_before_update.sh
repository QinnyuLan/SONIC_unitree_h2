#!/usr/bin/env bash

# ==================== 手动修改这里 ====================
# Kp = 原始 Kp × scale；只修改 Kp，不修改 Kd。
# scale 必须为有限正数；修改后需要重新启动本脚本。
LEFT_ANKLE_KP=1.4    # 左脚踝 pitch：原始 Kp = 40
RIGHT_ANKLE_KP=1.6   # 右脚踝 pitch：原始 Kp = 40
LEFT_KNEE_KP=1.3     # 左膝 pitch：原始 Kp = 200
RIGHT_KNEE_KP=1.3    # 右膝 pitch：原始 Kp = 200

# 模型路径相对于 gear_sonic_deploy，也可以填写绝对路径。
MODEL=../sonic_h2_native_release/experiments/gpu3_20260907_stage2_step009450
# MOTION=squat_001__A359
# reference/example_h2/deploy
NETWORK_INTERFACE=eno1
DDS_DOMAIN=0
# ====================================================

# 从任意工作目录运行，都切换到 gear_sonic_deploy。
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd -- "$H2_DEPLOY_DIR" || exit 1
# setup_env.sh 会读取尚未定义的环境变量，因此在启用 nounset 前加载。
source scripts/setup_env.sh || exit 1
set -eu

export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN"
unset GEAR_SONIC_AUTO_START_AFTER_SEC GEAR_SONIC_AUTO_PLAY_AFTER_SEC
unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC GEAR_SONIC_AUTO_STOP_AFTER_SEC
unset GEAR_SONIC_SKIP_MOTION_SWITCHER
# export GEAR_SONIC_INITIAL_MOTION_NAME="$MOTION"

# printf '模型：%s\n动作：%s\n' "$MODEL" "$MOTION"
# printf 'Kp scale：左脚踝=%s，右脚踝=%s，左膝=%s，右膝=%s\n' \
#   "$LEFT_ANKLE_KP" "$RIGHT_ANKLE_KP" "$LEFT_KNEE_KP" "$RIGHT_KNEE_KP"
# printf '操作：y → 等待 Init Done → ] → 稳定后 t 播放。\n'
# printf 'o 进入阻尼并退出姿态保持，请先确保机器人有支撑。\n'

exec ./deploy.sh \
  --robot-profile h2_native \
  --encoder-file "$MODEL/model_encoder_full.onnx" \
  --decoder-file "$MODEL/model_decoder.onnx" \
  --obs-config policy/release/observation_config_h2_native.yaml \
  --motion-data reference/example_g1_explore \
  --no-planner --input-type keyboard --output-type zmq \
  --hand-backend none \
  --h2-left-ankle-pitch-kp-scale "$LEFT_ANKLE_KP" \
  --h2-right-ankle-pitch-kp-scale "$RIGHT_ANKLE_KP" \
  --h2-left-knee-pitch-kp-scale "$LEFT_KNEE_KP" \
  --h2-right-knee-pitch-kp-scale "$RIGHT_KNEE_KP" \
  --enable-csv-logs \
  --logs-dir "logs/h2_four_kp_$(date +%Y%m%d_%H%M%S)" \
  "$NETWORK_INTERFACE"
