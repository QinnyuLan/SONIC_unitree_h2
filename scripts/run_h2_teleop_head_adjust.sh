#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROLE="${1:-}"
[[ $# -gt 0 ]] && shift
cd "$ROOT"
source "$ROOT/config/h2_head_adjust.env"
case "$ROLE" in
  deploy) exec bash "$ROOT/scripts/run_runtime.sh" real "$@" ;;
  bridge) exec "$ROOT/.venv_teleop/bin/python" -m interface.teleop.bridge "$@" ;;
  planner) exec bash "$ROOT/scripts/run_planner.sh" --publish-only "$@" ;;
  pico)
    COMMAND=("$ROOT/.venv_teleop/bin/python" -m interface.teleop.pico
      --manager --vr3pt-robot-profile g1 --smpl-wrist-source g1
      --head-control "$HEAD_CONTROL" --head-yaw-deg "$HEAD_YAW_DEG"
      --head-pitch-deg "$HEAD_PITCH_DEG" --head-pitch-rate-deg-s "$HEAD_PITCH_RATE_DEG_S"
      --head-stick-deadzone "$HEAD_STICK_DEADZONE")
    if [[ "${1:-}" == --print-command ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
    exec "${COMMAND[@]}" "$@" ;;
  *) echo "Usage: $0 {pico|bridge|planner|deploy} [options]" >&2; exit 2 ;;
esac
