#!/usr/bin/env bash
# Run the configured controller with CSV logging, then analyze its own log directory.
set -euo pipefail
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ROOT="$(dirname -- "$H2_DEPLOY_DIR")"
cd -- "$H2_DEPLOY_DIR"
if [[ "${1:-}" == --print-command && $# == 1 ]]; then
  exec bash "$H2_DEPLOY_DIR/scripts/run_h2_sim2real_four_kp.sh" --print-command
elif (( $# != 0 )); then
  echo "Usage: $0 [--print-command]" >&2
  exit 2
fi
H2_CONFIG_FILE="${H2_DEPLOY_CONFIG:-$H2_DEPLOY_DIR/config/h2_deploy.env}"
source "$H2_CONFIG_FILE"
# Preflight plotting dependencies before starting robot control.
"$H2_ROOT/.venv_teleop/bin/python" -c 'import numpy, matplotlib'
mkdir -p -- "$LOG_ROOT"
H2_CAPTURE_DIR="$(mktemp -d "$LOG_ROOT/h2_ankle_capture_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
H2_CAPTURE_DIR="$(cd -- "$H2_CAPTURE_DIR" && pwd)"
cp -- "$H2_CONFIG_FILE" "$H2_CAPTURE_DIR/deploy_config.env"
printf '\nLOG_ROOT=%q\n' "$H2_CAPTURE_DIR" >> "$H2_CAPTURE_DIR/deploy_config.env"
printf '本次记录目录：%s\n正常按 o 退出控制后自动分析；Ctrl+C 停止控制后也会尝试分析。\n' "$H2_CAPTURE_DIR"

# The foreground controller receives terminal SIGINT; let it stop and flush logs.
trap ':' INT
set +e
H2_DEPLOY_CONFIG="$H2_CAPTURE_DIR/deploy_config.env" bash "$H2_DEPLOY_DIR/scripts/run_h2_sim2real_four_kp.sh"
H2_CONTROL_STATUS=$?
set -e
trap - INT
shopt -s nullglob
H2_CAPTURE_LOGS=("$H2_CAPTURE_DIR"/h2_real_*)
if (( ${#H2_CAPTURE_LOGS[@]} == 1 )) && [[ -f "${H2_CAPTURE_LOGS[0]}/q.csv" ]]; then
  "$H2_ROOT/.venv_teleop/bin/python" "$H2_ROOT/scripts/analyze_h2_ankle_heat.py" "${H2_CAPTURE_LOGS[0]}"
else
  echo "未找到完整控制日志，请检查：$H2_CAPTURE_DIR" >&2
  exit 1
fi
exit "$H2_CONTROL_STATUS"
