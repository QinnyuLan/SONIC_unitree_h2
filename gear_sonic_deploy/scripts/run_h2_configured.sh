#!/usr/bin/env bash
# sim2real launcher: use run_h2_sim2real_four_kp.sh.
set -e
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_CONFIG_FILE="${H2_DEPLOY_CONFIG:-$H2_DEPLOY_DIR/config/h2_deploy.env}"
H2_TARGET="${1:-}"
if [[ "$H2_TARGET" != real ]]; then
  echo "Usage: $0 real [--no-logs] [--print-command]" >&2
  exit 2
fi
shift
H2_PRINT_ONLY=""
H2_RECORD_LOGS=true
for H2_OPTION in "$@"; do
  case "$H2_OPTION" in
    --print-command) H2_PRINT_ONLY=--print-command ;;
    --no-logs) H2_RECORD_LOGS=false ;;
    *) echo "Unknown option: $H2_OPTION (supported: --no-logs, --print-command)" >&2; exit 2 ;;
  esac
done
cd -- "$H2_DEPLOY_DIR"
source "$H2_CONFIG_FILE"
set -u

for H2_MODEL_FILE in "$MODEL/model_encoder_full.onnx" "$MODEL/model_decoder.onnx"; do
  [[ -f "$H2_MODEL_FILE" ]] || { echo "Missing model: $H2_MODEL_FILE" >&2; exit 1; }
done
[[ -d "$MOTION_DATA" ]] || { echo "Missing motion directory: $MOTION_DATA" >&2; exit 1; }
if [[ -n "$INITIAL_MOTION" && ! -d "$MOTION_DATA/$INITIAL_MOTION" ]]; then
  echo "Motion not found in selected collection: $INITIAL_MOTION" >&2
  exit 1
fi

H2_INTERFACE="$NETWORK_INTERFACE"
export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN"
unset GEAR_SONIC_AUTO_START_AFTER_SEC GEAR_SONIC_AUTO_PLAY_AFTER_SEC
unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC GEAR_SONIC_AUTO_STOP_AFTER_SEC
unset GEAR_SONIC_SKIP_MOTION_SWITCHER GEAR_SONIC_INITIAL_MOTION_NAME
if [[ -n "$INITIAL_MOTION" ]]; then
  export GEAR_SONIC_INITIAL_MOTION_NAME="$INITIAL_MOTION"
fi

H2_COMMAND=(./deploy.sh
  --robot-profile h2_native
  --encoder-file "$MODEL/model_encoder_full.onnx"
  --decoder-file "$MODEL/model_decoder.onnx"
  --obs-config policy/release/observation_config_h2_native.yaml
  --motion-data "$MOTION_DATA"
  --no-planner --input-type keyboard --output-type zmq --hand-backend none
  --h2-left-ankle-pitch-kp-scale "$LEFT_ANKLE_KP"
  --h2-right-ankle-pitch-kp-scale "$RIGHT_ANKLE_KP"
  --h2-left-ankle-roll-kp-scale "${LEFT_ANKLE_ROLL_KP:-1.0}"
  --h2-right-ankle-roll-kp-scale "${RIGHT_ANKLE_ROLL_KP:-1.0}"
  --h2-left-knee-pitch-kp-scale "$LEFT_KNEE_KP"
  --h2-right-knee-pitch-kp-scale "$RIGHT_KNEE_KP")
if [[ "$H2_RECORD_LOGS" == true ]]; then
  H2_COMMAND+=(--enable-csv-logs --logs-dir "$LOG_ROOT/h2_${H2_TARGET}_$(date +%Y%m%d_%H%M%S)")
fi
H2_COMMAND+=("$H2_INTERFACE")
printf '配置：%s\n模型：%s\n动作集合：%s\nDDS domain：%s\n' \
  "$H2_CONFIG_FILE" "$MODEL" "$MOTION_DATA" "$DDS_DOMAIN"
printf 'Kp scale：左脚踝=%s 右脚踝=%s 左膝=%s 右膝=%s\n' \
  "$LEFT_ANKLE_KP" "$RIGHT_ANKLE_KP" "$LEFT_KNEE_KP" "$RIGHT_KNEE_KP"
printf '脚踝 roll Kp scale：左=%s 右=%s\n' "${LEFT_ANKLE_ROLL_KP:-1.0}" "${RIGHT_ANKLE_ROLL_KP:-1.0}"
printf 'CSV 日志：%s\n' "$H2_RECORD_LOGS"
if [[ "$H2_PRINT_ONLY" == --print-command ]]; then
  printf '%q ' "${H2_COMMAND[@]}"
  printf '\n'
  exit 0
fi
# setup_env.sh reads optional unset variables; restore nounset afterward.
set +u
source scripts/setup_env.sh
set -u
exec "${H2_COMMAND[@]}"
