#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/gear_sonic_deploy/scripts/setup_env.sh"
exec "$ROOT/gear_sonic_deploy/target/release/sonic_g1_planner" "$@"
