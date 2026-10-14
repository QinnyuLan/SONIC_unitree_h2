#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/setup_env.sh"
"$ROOT/tools/validation/check_native.sh"
cmake -S "$ROOT/src/runtime" -B "$ROOT/build/runtime" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/build/runtime" --parallel "${BUILD_JOBS:-4}"
