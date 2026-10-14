#!/usr/bin/env bash
# Reuse the verified teleop SDK/OpenCV GStreamer runtime; don't replace its cv2 wheel.
set -euo pipefail
H2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$H2_ROOT"
[[ -x .venv_teleop/bin/python ]] || { echo 'Missing .venv_teleop' >&2; exit 1; }
if [[ ! -x .venv_data_collection/bin/python ]]; then
  uv venv .venv_data_collection --python .venv_teleop/bin/python
fi
.venv_teleop/bin/python - <<'PY'
from pathlib import Path
import site
p = next(Path('.venv_data_collection/lib').glob('python*/site-packages'))/'teleop_runtime.pth'
p.write_text('\n'.join(site.getsitepackages())+'\n')
PY
uv pip install --python .venv_data_collection/bin/python pyarrow==19.0.1 av==14.2.0
.venv_data_collection/bin/python - <<'PY'
import av, cv2, pyarrow, zmq, msgpack
assert any('GStreamer:' in line and 'YES' in line for line in cv2.getBuildInformation().splitlines())
av.codec.Codec('libx264', 'w')
print('H2 collection runtime ready; no robot/camera connection was opened.')
PY
