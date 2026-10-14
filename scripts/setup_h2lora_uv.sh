#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv venv .venv-h2lora --python 3.10
source .venv-h2lora/bin/activate
uv pip install -e "gear_sonic[training]" swanlab pytest imageio-ffmpeg

python - <<'PY'
import torch
import swanlab

print(f"torch={torch.__version__}")
print(f"swanlab={getattr(swanlab, '__version__', 'unknown')}")
print("isaaclab=not checked in .venv-h2lora")
PY

echo "Use scripts/setup_h2lora_isaac_python.sh for IsaacLab/Isaac Sim training."
