#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_KIND="${1:-sim2sim}"
case "$ENV_KIND" in
  planner) ENV_DIR="$ROOT/.venv_planner"; EXTRA=planner ;;
  inputs) ENV_DIR="$ROOT/.venv_inputs"; EXTRA=inputs ;;
  sim2sim) ENV_DIR="$ROOT/.venv_sim"; EXTRA=sim ;;
  teleop) ENV_DIR="$ROOT/.venv_teleop"; EXTRA=teleop ;;
  *) echo "Usage: $0 {inputs|planner|sim2sim|teleop}" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR" --python 3.10 --prompt "sonic_h2_${ENV_KIND}"
fi
# Teleoperation uses CPU pose mathematics; model inference uses TensorRT.
uv pip install --python "$ENV_DIR/bin/python" --torch-backend=cpu \
  -c "$ROOT/requirements/$EXTRA.lock" -e "$ROOT[$EXTRA]"
if [[ "$ENV_KIND" == planner ]]; then
  # Use the Python bindings shipped with the same TensorRT SDK as the C++ runtime.
  source "$ROOT/scripts/setup_env.sh"
  if [[ -n "${TENSORRT_WHEEL:-}" ]]; then
    TRT_WHEELS=("$TENSORRT_WHEEL")
  else
    shopt -s nullglob
    TRT_WHEELS=("$TensorRT_ROOT"/python/tensorrt-10.13.3.9-cp310-none-linux_"$(uname -m)".whl)
  fi
  if [[ ${#TRT_WHEELS[@]} != 1 || ! -f "${TRT_WHEELS[0]}" ]]; then
    echo "TensorRT Python wheel not found. Set TensorRT_ROOT to the SDK or TENSORRT_WHEEL to its matching Python 3.10 wheel." >&2
    exit 1
  fi
  uv pip install --python "$ENV_DIR/bin/python" -c "$ROOT/requirements/planner.lock" "${TRT_WHEELS[0]}"
  "$ENV_DIR/bin/python" -c 'import tensorrt; from cuda.bindings import runtime; print("TensorRT", tensorrt.__version__)'
elif [[ "$ENV_KIND" == sim2sim ]]; then
  uv pip install --python "$ENV_DIR/bin/python" -c "$ROOT/requirements/sim.lock" \
    -e "$ROOT/external_dependencies/unitree_sdk2_python"
elif [[ "$ENV_KIND" == teleop ]]; then
  uv pip install --python "$ENV_DIR/bin/python" 'cmake>=3.14,<4' 'pybind11==3.0.4' 'setuptools<82' wheel
  uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -e "$ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
fi
uv pip check --python "$ENV_DIR/bin/python"
echo "Installed $ENV_KIND in $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
