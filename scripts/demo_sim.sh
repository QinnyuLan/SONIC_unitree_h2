#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv_sim/bin/python ]]; then
  echo "Install the simulation environment first: ./install.sh sim2sim" >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv_sim/bin/python tools/demos/sim2sim.py "$@"
