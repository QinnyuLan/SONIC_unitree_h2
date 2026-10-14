#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/gear_sonic_deploy/scripts/setup_env.sh"
"$ROOT/verify_deploy.sh"
cmake -S "$ROOT/gear_sonic_deploy" -B "$ROOT/gear_sonic_deploy/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/gear_sonic_deploy/build" --parallel "${BUILD_JOBS:-4}"
