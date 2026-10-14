#!/usr/bin/env bash
set -euo pipefail

# sim2sim 独立参数；不读取实机 h2_deploy.env。
# 使用正式发布模型。
MODEL=../models/released
MOTION_DATA=reference/example  # 常规 SONIC demo，G1 输入 selector 3 → H2 本体
INITIAL_MOTION=""
DDS_DOMAIN=0
LOG_ROOT=logs
# 仿真使用 nominal Kp/Kd，四个关节的 Kp scale 均为 1.0。

if (( $# > 1 )) || [[ -n "${1:-}" && "${1:-}" != --print-command ]]; then
  echo "Usage: $0 [--print-command]" >&2
  exit 2
fi
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$H2_DEPLOY_DIR"
for H2_MODEL_FILE in "$MODEL/model_encoder_full.onnx" "$MODEL/model_decoder.onnx"; do
  [[ -f "$H2_MODEL_FILE" ]] || { echo "Missing model: $H2_MODEL_FILE" >&2; exit 1; }
done
[[ -d "$MOTION_DATA" ]] || { echo "Missing motion directory: $MOTION_DATA" >&2; exit 1; }
if [[ -n "$INITIAL_MOTION" && ! -d "$MOTION_DATA/$INITIAL_MOTION" ]]; then
  echo "Motion not found in selected collection: $INITIAL_MOTION" >&2
  exit 1
fi
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
  --enable-csv-logs --logs-dir "$LOG_ROOT/h2_sim_$(date +%Y%m%d_%H%M%S)"
  sim)
printf '模型：%s\n动作集合：%s\nDDS domain：%s\nKp scale：1.0（nominal）\n' \
  "$MODEL" "$MOTION_DATA" "$DDS_DOMAIN"
if [[ "${1:-}" == --print-command ]]; then
  printf '%q ' "${H2_COMMAND[@]}"
  printf '\n'
  exit 0
fi
set +u
source scripts/setup_env.sh
set -u
exec "${H2_COMMAND[@]}"
