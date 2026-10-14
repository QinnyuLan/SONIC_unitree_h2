#!/usr/bin/env bash
# CPU/loopback checks only. Does not create a DDS participant or motor publisher.
set -euo pipefail
H2_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ROOT="$(cd -- "$H2_DEPLOY_DIR/.." && pwd)"
H2_TEST_DIR="$(mktemp -d /tmp/h2_vla_checks.XXXXXX)"
trap 'rm -rf "$H2_TEST_DIR"' EXIT
cd "$H2_ROOT"
export PYTHONPATH="$H2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$H2_ROOT/.venv_data_collection/bin/python" -m pytest gear_sonic/tests/test_h2_vla_inference.py -q
bash "$H2_DEPLOY_DIR/scripts/run_h2_vla.sh" check
cd "$H2_DEPLOY_DIR"
c++ -std=c++20 -O3 -ffast-math \
  src/g1/g1_deploy_onnx_ref/tools/test_h2_vla_watchdog.cpp -o "$H2_TEST_DIR/watchdog"
"$H2_TEST_DIR/watchdog"
c++ -std=c++20 -O2 -ffast-math -I/usr/include/eigen3 -Isrc \
  -Isrc/g1/g1_deploy_onnx_ref/include -Ithirdparty/unitree_sdk2/include \
  -Ithirdparty/unitree_sdk2/thirdparty/include -Ithirdparty/unitree_sdk2/thirdparty/include/ddscxx \
  -I/opt/onnxruntime/include -I/usr/local/cuda/targets/x86_64-linux/include \
  src/g1/g1_deploy_onnx_ref/tools/h2_vla_receiver_probe.cpp -lzmq -lpthread \
  -o "$H2_TEST_DIR/receiver"
"$H2_ROOT/.venv_data_collection/bin/python" "$H2_ROOT/gear_sonic/tests/test_h2_vla_receiver.py" --binary "$H2_TEST_DIR/receiver"
