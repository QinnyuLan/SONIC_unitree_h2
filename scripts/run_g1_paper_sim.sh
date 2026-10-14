#!/usr/bin/env bash
set -euo pipefail

# ==================== 录屏构图：修改这里 ====================
CAMERA_MODE=fixed      # fixed：固定镜头；follow：跟随机器人
CAMERA_DISTANCE=3.2    # 米；跳跃/大幅位移动作可增大到 5～6
CAMERA_AZIMUTH=135
CAMERA_ELEVATION=-15
LOOKAT_X=0.0
LOOKAT_Y=0.0
LOOKAT_Z=0.65          # G1 固定镜头观察中心高度（米）
DDS_DOMAIN=0
# ==========================================================

G1_PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$G1_PAPER_ROOT"
export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --wbc-version sonic_model12 \
  --interface sim \
  --visual-style paper \
  --presentation-camera "$CAMERA_MODE" \
  --presentation-distance "$CAMERA_DISTANCE" \
  --presentation-azimuth "$CAMERA_AZIMUTH" \
  --presentation-elevation "$CAMERA_ELEVATION" \
  --presentation-lookat "$LOOKAT_X" "$LOOKAT_Y" "$LOOKAT_Z" \
  "$@"
