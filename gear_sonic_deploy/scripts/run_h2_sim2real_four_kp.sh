#!/usr/bin/env bash
# 参数集中到 gear_sonic_deploy/config/h2_deploy.env；修改后重启生效。
set -euo pipefail
H2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$H2_SCRIPT_DIR/run_h2_configured.sh" real "$@"
