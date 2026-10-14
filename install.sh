#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_KIND="${1:-sim2sim}"
case "$ENV_KIND" in
  inputs) ENV_DIR="$ROOT/.venv_inputs"; EXTRA=inputs ;;
  sim2sim) ENV_DIR="$ROOT/.venv_sim"; EXTRA=sim ;;
  teleop) ENV_DIR="$ROOT/.venv_teleop"; EXTRA=teleop ;;
  *) echo "Usage: $0 {inputs|sim2sim|teleop}" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR" --python 3.10 --prompt "sonic_h2_${ENV_KIND}"
fi
# Teleoperation uses CPU pose mathematics; model inference runs in the C++ controller.
uv pip install --python "$ENV_DIR/bin/python" --torch-backend=cpu \
  -c "$ROOT/requirements/$EXTRA.lock" -e "$ROOT/gear_sonic[$EXTRA]"
if [[ "$ENV_KIND" == sim2sim ]]; then
  uv pip install --python "$ENV_DIR/bin/python" -c "$ROOT/requirements/sim.lock" \
    -e "$ROOT/external_dependencies/unitree_sdk2_python"
elif [[ "$ENV_KIND" == teleop ]]; then
  uv pip install --python "$ENV_DIR/bin/python" 'cmake>=3.14,<4' 'pybind11==3.0.4' 'setuptools<82' wheel
  uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -e "$ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
fi
uv pip check --python "$ENV_DIR/bin/python"
echo "Installed $ENV_KIND in $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
