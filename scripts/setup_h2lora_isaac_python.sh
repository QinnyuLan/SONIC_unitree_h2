#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/sky/software/isaac-sim}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/sky/workspace/basic/IsaacLab-2.3.2}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT}/python.sh"
ISAAC_PYTHON_BIN="${ISAAC_SIM_ROOT}/kit/python/bin/python3"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Missing Isaac Sim python.sh: ${ISAAC_PYTHON}" >&2
  exit 1
fi

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic isaaclab_contrib; do
  if [[ ! -d "${ISAACLAB_ROOT}/source/${pkg}" ]]; then
    echo "Missing IsaacLab source package: ${ISAACLAB_ROOT}/source/${pkg}" >&2
    exit 1
  fi
done

"${ISAAC_PYTHON}" -m pip install --no-deps \
  -e "${ISAACLAB_ROOT}/source/isaaclab" \
  -e "${ISAACLAB_ROOT}/source/isaaclab_assets" \
  -e "${ISAACLAB_ROOT}/source/isaaclab_tasks" \
  -e "${ISAACLAB_ROOT}/source/isaaclab_rl" \
  -e "${ISAACLAB_ROOT}/source/isaaclab_mimic" \
  -e "${ISAACLAB_ROOT}/source/isaaclab_contrib"

URDF_CONVERTER="${ISAACLAB_ROOT}/source/isaaclab/isaaclab/sim/converters/urdf_converter.py"
"${ISAAC_PYTHON}" - "${URDF_CONVERTER}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "        import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)\n"
new = (
    "        if hasattr(import_config, \"set_merge_fixed_ignore_inertia\"):\n"
    "            import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)\n"
)
if new not in text:
    if old not in text:
        raise SystemExit(f"Cannot apply URDF importer compatibility patch to {path}")
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
else:
    print(f"patch already present in {path}")
PY

exclude_file="$(mktemp)"
trap 'rm -f "${exclude_file}"' EXIT
cat >"${exclude_file}" <<'EOF'
torch
triton
sympy
networkx
nvidia-cublas-cu12
nvidia-cuda-cupti-cu12
nvidia-cuda-nvrtc-cu12
nvidia-cuda-runtime-cu12
nvidia-cudnn-cu12
nvidia-cufft-cu12
nvidia-cufile-cu12
nvidia-curand-cu12
nvidia-cusolver-cu12
nvidia-cusparse-cu12
nvidia-cusparselt-cu12
nvidia-nccl-cu12
nvidia-nvjitlink-cu12
nvidia-nvtx-cu12
EOF

uv pip install --python "${ISAAC_PYTHON_BIN}" --excludes "${exclude_file}" \
  numpy==1.26.4 \
  scipy==1.15.3 \
  pandas==2.3.3 \
  hydra-core==1.3.2 \
  wandb==0.27.0 \
  trl==0.28.0 \
  transformers==5.9.0 \
  accelerate==1.13.0 \
  tensorboard==2.20.0 \
  swanlab==0.7.19 \
  gymnasium==1.3.0 \
  lazy-loader==0.5 \
  prettytable==3.17.0 \
  h5py==3.16.0 \
  rich==13.9.4 \
  smplx==0.1.28 \
  easydict==1.13 \
  loguru==0.7.3 \
  joblib==1.5.3 \
  einops==0.8.2 \
  datasets==4.8.5 \
  flatdict==4.0.0 \
  toml==0.10.2 \
  warp-lang \
  tensordict \
  vector-quantize-pytorch \
  pytest

"${ISAAC_PYTHON}" - <<'PY'
import importlib

for name in [
    "isaaclab",
    "hydra",
    "omegaconf",
    "trl",
    "transformers",
    "accelerate",
    "swanlab",
    "tensordict",
    "vector_quantize_pytorch",
]:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'ok')}")

from isaaclab.app import AppLauncher

print(f"AppLauncher={AppLauncher.__name__}")
PY
