#!/usr/bin/env bash
set -euo pipefail

# Install one reproducible Python environment. Native C++ deployment dependencies
# (CUDA/TensorRT/ONNX Runtime/ZeroMQ) are checked separately by verify_deploy.sh.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_KIND="${1:-sim2sim}"
case "$ENV_KIND" in
  sim2sim) ENV_DIR="$ROOT/.venv_sim"; EXTRA='sim' ;;
  teleop) ENV_DIR="$ROOT/.venv_teleop"; EXTRA='teleop' ;;
  data_collection) ENV_DIR="$ROOT/.venv_data_collection"; EXTRA='data_collection' ;;
  inference) ENV_DIR="$ROOT/.venv_inference"; EXTRA='inference' ;;
  *) echo "Usage: $0 {sim2sim|teleop|data_collection|inference}" >&2; exit 2 ;;
esac
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
uv venv "$ENV_DIR" --python 3.10 --prompt "sonic_h2_${ENV_KIND}"
uv pip install --python "$ENV_DIR/bin/python" -e "$ROOT/gear_sonic[$EXTRA]"
if [[ "$ENV_KIND" != inference ]]; then
  uv pip install --python "$ENV_DIR/bin/python" -e "$ROOT/external_dependencies/unitree_sdk2_python"
fi
if [[ "$ENV_KIND" == teleop || "$ENV_KIND" == data_collection ]]; then
  uv pip install --python "$ENV_DIR/bin/python" cmake pybind11 setuptools
  uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -e "$ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
fi
echo "Installed $ENV_KIND in $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
