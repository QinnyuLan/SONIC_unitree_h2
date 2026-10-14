#!/usr/bin/env bash
set -euo pipefail
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ROOT="$(cd -- "$H2_DEPLOY_DIR/.." && pwd)"
H2_ROLE="${1:-}"
[[ $# -gt 0 ]] && shift
case "$H2_ROLE" in
  deploy|pico) exec bash "$H2_DEPLOY_DIR/scripts/run_h2_teleop_head_adjust.sh" "$H2_ROLE" "$@" ;;
  record|check) ;;
  *) echo "Usage: $0 {deploy|pico|record|check} [recorder options / --print-command]" >&2; exit 2 ;;
esac
cd "$H2_DEPLOY_DIR"
H2_REAL_CONFIG="${H2_DEPLOY_CONFIG:-$H2_DEPLOY_DIR/config/h2_deploy.env}"
H2_RECORD_CONFIG="${H2_COLLECTION_CONFIG:-$H2_DEPLOY_DIR/config/h2_data_collection.env}"
source "$H2_REAL_CONFIG"
source "$H2_DEPLOY_DIR/config/h2_head_adjust.env"
source "$H2_RECORD_CONFIG"
[[ "$HAND_BACKEND" == inspire ]] || { echo 'Collection requires HAND_BACKEND=inspire' >&2; exit 2; }
H2_COMMAND=("$H2_ROOT/.venv_data_collection/bin/python" "$H2_ROOT/gear_sonic/scripts/collect_h2_vla.py"
  --root "$DATASET_ROOT" --task "$TASK_PROMPT" --model "$MODEL"
  --interface "$NETWORK_INTERFACE" --domain "$DDS_DOMAIN"
  --fps "$RECORD_FPS" --rgb-port "$RGB_PORT" --width "$RGB_WIDTH" --height "$RGB_HEIGHT"
  --min-free-gb "$MIN_FREE_GB" --teleop-address "$TELEOP_ADDRESS" --state-address "$STATE_ADDRESS"
  --config "$H2_REAL_CONFIG" --config "$H2_DEPLOY_DIR/config/h2_head_adjust.env" --config "$H2_RECORD_CONFIG")
[[ -n "$CAMERA_CALIBRATION" ]] && H2_COMMAND+=(--calibration "$CAMERA_CALIBRATION")
[[ "$HEADLESS" == 1 ]] && H2_COMMAND+=(--headless)
[[ "$H2_ROLE" == check ]] && H2_COMMAND+=(--check)
if [[ "${1:-}" == --print-command ]]; then
  printf '%q ' "${H2_COMMAND[@]}"
  printf '\n'
  exit 0
fi
[[ -x "${H2_COMMAND[0]}" ]] || { echo "Run: bash $H2_ROOT/install_scripts/install_h2_data_collection.sh" >&2; exit 1; }
exec "${H2_COMMAND[@]}" "$@"
