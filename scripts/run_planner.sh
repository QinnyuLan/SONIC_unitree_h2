#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/gear_sonic_deploy/scripts/setup_env.sh"
PYTHON="${PLANNER_PYTHON:-$ROOT/.venv_planner/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Install the planner environment: ./install.sh planner" >&2; exit 1; }
exec "$PYTHON" -m gear_sonic.sources.planner "$@"
