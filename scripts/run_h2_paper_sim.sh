#!/usr/bin/env bash
set -euo pipefail

# ==================== 录屏构图：修改这里 ====================
CAMERA_MODE=fixed      # fixed：固定镜头（论文对比）；follow：跟随机器人
CAMERA_DISTANCE=4.5    # 米；跳跃/大幅位移动作可以增大到 6～8
CAMERA_AZIMUTH=135     # 水平方向，单位度
CAMERA_ELEVATION=-15   # 俯视角，单位度；越负越俯视
LOOKAT_X=0.0
LOOKAT_Y=0.0
LOOKAT_Z=0.85          # 固定镜头的观察中心高度，单位米
DDS_DOMAIN=${DDS_DOMAIN:-0}          # 与 run_runtime.sh sim 中的 DDS_DOMAIN 一致
# ==========================================================

H2_PAPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$H2_PAPER_ROOT"
export GEAR_SONIC_DOMAIN_ID="$DDS_DOMAIN"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec .venv_sim/bin/python -m sim \
  --wbc-version h2_native_unitree_ros_loop \
  --interface sim \
  --visual-style paper \
  --presentation-camera "$CAMERA_MODE" \
  --presentation-distance "$CAMERA_DISTANCE" \
  --presentation-azimuth "$CAMERA_AZIMUTH" \
  --presentation-elevation "$CAMERA_ELEVATION" \
  --presentation-lookat "$LOOKAT_X" "$LOOKAT_Y" "$LOOKAT_Z" \
  "$@"
