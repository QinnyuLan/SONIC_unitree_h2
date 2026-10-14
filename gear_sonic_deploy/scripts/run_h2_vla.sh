#!/usr/bin/env bash
# No automatic start/play: the operator issues stand/run in the live terminal.
set -euo pipefail
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ROOT="$(cd -- "$H2_DEPLOY_DIR/.." && pwd)"
H2_ROLE="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$H2_DEPLOY_DIR"
source "${H2_DEPLOY_CONFIG:-$H2_DEPLOY_DIR/config/h2_deploy.env}"
source "${H2_VLA_CONFIG:-$H2_DEPLOY_DIR/config/h2_vla.env}"
export PYTHONPATH="$H2_ROOT:$H2_ROOT/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1
case "$H2_ROLE" in
  server)
    H2_COMMAND=("$H2_ROOT/.venv_n1d7/bin/python" -u "$H2_ROOT/gear_sonic/scripts/run_h2_n1d7_server.py"
      --model "$VLA_MODEL" --port "$VLA_POLICY_PORT") ;;
  check|replay|shadow|live)
    H2_COMMAND=("$H2_ROOT/.venv_data_collection/bin/python" -u "$H2_ROOT/gear_sonic/scripts/run_h2_vla_inference.py"
      --mode "$H2_ROLE" --host "$VLA_POLICY_HOST" --port "$VLA_POLICY_PORT"
      --interface "$NETWORK_INTERFACE" --domain "$DDS_DOMAIN" --rgb-port "$VLA_RGB_PORT"
      --sonic-model "$VLA_SONIC_MODEL" --fixture "$VLA_MODEL/replay_fixture.npz"
      --inference-interval "$VLA_INFERENCE_INTERVAL") ;;
  deploy)
    export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN" GEAR_SONIC_H2_VLA=1
    export GEAR_SONIC_INITIAL_MOTION_NAME="$VLA_STAND_MOTION"
    unset GEAR_SONIC_AUTO_START_AFTER_SEC GEAR_SONIC_AUTO_PLAY_AFTER_SEC
    unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC GEAR_SONIC_AUTO_STOP_AFTER_SEC GEAR_SONIC_SKIP_MOTION_SWITCHER
    H2_COMMAND=(./deploy.sh --robot-profile h2_native
      --encoder-file "$VLA_SONIC_MODEL/model_encoder_full.onnx"
      --decoder-file "$VLA_SONIC_MODEL/model_decoder.onnx"
      --obs-config policy/release/observation_config_h2_native.yaml
      --motion-data "$VLA_STAND_REFERENCE" --planner planner/target_vel/V2/planner_sonic.onnx
      --input-type zmq_manager --output-type zmq --hand-backend inspire --h2-vr-head-control
      --h2-left-ankle-pitch-kp-scale "$LEFT_ANKLE_KP"
      --h2-right-ankle-pitch-kp-scale "$RIGHT_ANKLE_KP"
      --h2-left-ankle-roll-kp-scale "${LEFT_ANKLE_ROLL_KP:-1.0}"
      --h2-right-ankle-roll-kp-scale "${RIGHT_ANKLE_ROLL_KP:-1.0}"
      --h2-left-knee-pitch-kp-scale "$LEFT_KNEE_KP"
      --h2-right-knee-pitch-kp-scale "$RIGHT_KNEE_KP"
      --enable-csv-logs --logs-dir "$LOG_ROOT/h2_vla_controller_$(date +%Y%m%d_%H%M%S)"
      "$NETWORK_INTERFACE") ;;
  *) echo "Usage: $0 {check|server|replay|deploy|shadow|live} [--print-command | extra options]" >&2; exit 2 ;;
esac
if [[ "${1:-}" == --print-command ]]; then
  printf '%q ' "${H2_COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ "$H2_ROLE" == deploy ]]; then
  "$H2_ROOT/.venv_data_collection/bin/python" "$H2_ROOT/gear_sonic/scripts/run_h2_vla_inference.py" --mode check --sonic-model "$VLA_SONIC_MODEL"
  # Rebuilt binary must contain the receiver-side watchdog before any DDS opens.
  if ! strings target/release/g1_deploy_onnx_ref | grep -F 'h2_vla_watchdog' > /dev/null; then
    echo 'Rebuild the C++ controller with the H2 VLA adaptation first.' >&2
    exit 1
  fi
  set +u
  source scripts/setup_env.sh
  set -u
fi
exec "${H2_COMMAND[@]}" "$@"
