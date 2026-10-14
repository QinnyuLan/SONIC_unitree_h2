#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/setup_env.sh"
PYTHON="${PLANNER_PYTHON:-$ROOT/.venv_planner/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Install the planner environment: ./install.sh planner" >&2; exit 1; }
exec "$PYTHON" -m interface.planner.service "$@"
