#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
if [[ "$MODE" != sim && "$MODE" != real && "$MODE" != dry-run ]]; then
  echo "Usage: $0 {sim|real|dry-run} [--print-command] [runtime options]" >&2
  exit 2
fi
shift
PRINT_ONLY=false
EXTRA=()
for ARG in "$@"; do
  if [[ "$ARG" == --print-command ]]; then PRINT_ONLY=true; else EXTRA+=("$ARG"); fi
done
cd "$ROOT"
COMMAND=("$ROOT/build/runtime/bin/sonic_h2_runtime" --model-dir "$ROOT/models/released")
if [[ "$MODE" == real ]]; then
  source "${H2_DEPLOY_CONFIG:-$ROOT/config/h2_deploy.env}"
  COMMAND+=(--network "$NETWORK_INTERFACE" --domain "$DDS_DOMAIN" --hand-backend "$HAND_BACKEND"
    --left-ankle-kp "$LEFT_ANKLE_KP" --right-ankle-kp "$RIGHT_ANKLE_KP"
    --left-ankle-roll-kp "$LEFT_ANKLE_ROLL_KP" --right-ankle-roll-kp "$RIGHT_ANKLE_ROLL_KP"
    --left-knee-kp "$LEFT_KNEE_KP" --right-knee-kp "$RIGHT_KNEE_KP")
elif [[ "$MODE" == sim ]]; then
  COMMAND+=(--sim --network lo --domain "${DDS_DOMAIN:-0}")
else
  COMMAND+=(--dry-run)
fi
COMMAND+=("${EXTRA[@]}")
if "$PRINT_ONLY"; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
[[ -x "${COMMAND[0]}" ]] || { echo "Build first: bash scripts/build_controller.sh" >&2; exit 1; }
source "$ROOT/scripts/setup_env.sh"
if [[ "$MODE" == real ]]; then
  echo "H2 will move to its default standing pose and accept commands on $NETWORK_INTERFACE (DDS $DDS_DOMAIN)."
  read -r -p "Type RUN to start robot control: " RESPONSE
  [[ "$RESPONSE" == RUN ]] || exit 1
fi
exec "${COMMAND[@]}"
