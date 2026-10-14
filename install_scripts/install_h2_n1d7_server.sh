#!/usr/bin/env bash
set -euo pipefail
H2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_UV="${H2_UV:-$HOME/.local/bin/uv}"
"$H2_UV" python install 3.12
if [[ ! -x "$H2_ROOT/.venv_n1d7/bin/python" ]]; then
  "$H2_UV" venv --python 3.12 "$H2_ROOT/.venv_n1d7"
fi
"$H2_UV" pip install --python "$H2_ROOT/.venv_n1d7/bin/python" \
  torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$H2_UV" pip install --python "$H2_ROOT/.venv_n1d7/bin/python" \
  -r "$H2_ROOT/install_scripts/requirements_h2_n1d7.txt"
echo 'N1.7 server runtime installed. H2 sensor client continues using .venv_data_collection.'
