#!/usr/bin/env bash
# Two terminals: this script deploy / this script pico. No CSV recording.
set -euo pipefail
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ROOT="$(cd -- "$H2_DEPLOY_DIR/.." && pwd)"
H2_ROLE="${1:-}"
if [[ "$H2_ROLE" != deploy && "$H2_ROLE" != pico ]] || (( $# > 2 )) || \
   [[ -n "${2:-}" && "${2:-}" != --print-command ]]; then
  echo "Usage: $0 {deploy|pico} [--print-command]" >&2
  exit 2
fi
cd -- "$H2_DEPLOY_DIR"
# Gains, motion set and network use the user's current real-robot settings.
source "${H2_DEPLOY_CONFIG:-$H2_DEPLOY_DIR/config/h2_deploy.env}"
# Released model and head settings for VR teleoperation.
source "$H2_DEPLOY_DIR/config/h2_head_adjust.env"
export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN"
unset GEAR_SONIC_AUTO_START_AFTER_SEC GEAR_SONIC_AUTO_PLAY_AFTER_SEC
unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC GEAR_SONIC_AUTO_STOP_AFTER_SEC
unset GEAR_SONIC_SKIP_MOTION_SWITCHER GEAR_SONIC_INITIAL_MOTION_NAME
if [[ "$H2_ROLE" == deploy ]]; then
  if [[ ! -f planner/target_vel/V2/planner_sonic.onnx ]]; then
    echo "G1 planner is missing. See $H2_ROOT/docs/PLANNER.md" >&2
    echo "Download: python3 $H2_ROOT/download_from_hf.py --planner-only" >&2
    exit 1
  fi
  for H2_MODEL_FILE in "$MODEL/model_encoder_full.onnx" "$MODEL/model_decoder.onnx"; do
    [[ -f "$H2_MODEL_FILE" ]] || { echo "Missing model: $H2_MODEL_FILE" >&2; exit 1; }
  done
  [[ -d "$MOTION_DATA" ]] || { echo "Missing motion directory: $MOTION_DATA" >&2; exit 1; }
  H2_COMMAND=(./deploy.sh --robot-profile h2_native
    --encoder-file "$MODEL/model_encoder_full.onnx"
    --decoder-file "$MODEL/model_decoder.onnx"
    --obs-config policy/release/observation_config_h2_native.yaml
    --motion-data "$MOTION_DATA"
    --planner planner/target_vel/V2/planner_sonic.onnx
    --input-type zmq_manager --output-type zmq
    --h2-vr-head-control --hand-backend "$HAND_BACKEND"
    --h2-left-ankle-pitch-kp-scale "$LEFT_ANKLE_KP"
    --h2-right-ankle-pitch-kp-scale "$RIGHT_ANKLE_KP"
    --h2-left-ankle-roll-kp-scale "${LEFT_ANKLE_ROLL_KP:-1.0}"
    --h2-right-ankle-roll-kp-scale "${RIGHT_ANKLE_ROLL_KP:-1.0}"
    --h2-left-knee-pitch-kp-scale "$LEFT_KNEE_KP"
    --h2-right-knee-pitch-kp-scale "$RIGHT_KNEE_KP"
    "$NETWORK_INTERFACE")
  printf '模型：%s\n手部：%s\nCSV 日志：关闭\n' "$MODEL" "$HAND_BACKEND"
else
  cd -- "$H2_ROOT"
  H2_COMMAND=("$H2_ROOT/.venv_teleop/bin/python"
    "$H2_ROOT/gear_sonic/scripts/pico_manager_thread_server.py"
    --manager --vr3pt-robot-profile g1 --smpl-wrist-source g1
    --head-control "$HEAD_CONTROL" --head-yaw-deg "$HEAD_YAW_DEG"
    --head-pitch-deg "$HEAD_PITCH_DEG" --head-pitch-rate-deg-s "$HEAD_PITCH_RATE_DEG_S"
    --head-stick-deadzone "$HEAD_STICK_DEADZONE")
  printf '头部模式：%s；head yaw：%s°；head pitch：%s°\n' \
    "$HEAD_CONTROL" "$HEAD_YAW_DEG" "$HEAD_PITCH_DEG"
fi
if [[ "${2:-}" == --print-command ]]; then
  printf '%q ' "${H2_COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ "$H2_ROLE" == deploy ]]; then
  set +u
  source scripts/setup_env.sh
  set -u
fi
exec "${H2_COMMAND[@]}"
