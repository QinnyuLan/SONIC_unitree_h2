# 静态 SMPL mode 2 双臂跟踪实验

日期：2026-07-24

## 结论摘要

当前链路中曾存在一个确定的 G1 MuJoCo 执行错误，并非单纯的模型能力问题：

- G1 43-DoF MJCF 的 actuator 顺序为：左侧 body、左手、右臂、右手。
- `motor_effort_limit_list` 沿用这个 actuator 顺序。
- H2 仿真重构后，代码曾误把列表前 29 项解释成连续的 29 个 body joint。
- 因此右肩和右肘被错误限制为 `2.45/0.7 Nm`，而不是 `25 Nm`，表现为右臂无法举起。

修复后，G1 左右臂不再出现单侧失效；但双肘仍明显弯曲。剩余问题属于官方 G1 预训练 policy 对这两个静态 SMPL mode-2 reference 的跟踪能力，而不是单侧关节映射或 reference 构造错误。

## Reference 契约

当前必须使用 generator contract v6。旧 v1-v5 结果不再用于评价当前 H2LoRA
在线局部运动学链路。

v5 的两个 reference 均为 300 帧、50 Hz：

- `arms_overhead_straight`：双臂平行向上，目标方向为根坐标系 `+Z`。
- `arms_forward_straight`：双臂平行向前，目标方向为根坐标系 `+X`。

生成器使用工程的 SMPL FK 计算 24 个 joint，磁盘保存 runtime raw joints 和匹配的
root quaternion。部署端按训练契约用 inverse-root 得到 local joints。G1 的 6 个 wrist
reference 使用原始 `PoseStreamer` 公式；H2LoRA 将 SMPL 的左右手腕局部旋转直接
分解到 H2 的 roll-pitch-yaw 关节，并只把这 6 个 wrist joint 写入 mode-2 的
`W(60)`。该转换是训练时离线 SOMA 全身优化标签的无状态局部近似，不宣称逐帧等价。

生成结果中，两侧肘内角均为 180 度，肩到腕方向与目标完全一致；每帧相同，不包含
全身 retargeting、插值或网络延迟。

## 模型与输入

- G1：`policy/release/model_encoder.onnx`、`model_decoder.onnx` 和 `observation_config.yaml`。
- H2 LoRA：`sonic_h2lora_release/exported/model_encoder.onnx`、`model_decoder_latest.onnx` 和 `observation_config_h2_lora.yaml`。
- 两者均使用 encoder mode 2，必须打印 `Switched to mode 'smpl' (ID=2)`，不能发生 fallback。

## G1 执行层 A/B

使用同一 G1 encoder/decoder、同一 deploy、同一段真实连续 `dance_in_da_party_001__A464` SMPL reference，在第 320 帧比较：

| 仿真路径 | 左臂方向误差 | 右臂方向误差 |
| --- | ---: | ---: |
| 修复前当前 MuJoCo | 3.1 度 | 60.6 度 |
| 原版 commit `2a5e6a1` MuJoCo | 29.2 度 | 16.3 度 |
| 修复后当前 MuJoCo | 24.7 度 | 20.5 度 |

这个 A/B 排除了静态跳变、合成骨架和 policy 文件差异。修复后左右误差恢复到原版同一量级。

## 静态 Reference 结果

只统计 `motion_playing=1`、`encoder_mode=2` 的最后 3 秒。日志中的实测 `q` 经对应 MJCF FK 后评估：

| 模型 | 动作 | 左/右肘内角 | 左/右方向误差 | 基座倾斜均值/最大值 |
| --- | --- | ---: | ---: | ---: |
| G1 修复前 | 过顶 | 127.4 / 141.4 度 | 24.2 / 85.3 度 | 0.5 / 1.0 度 |
| G1 修复前 | 前举 | 153.0 / 138.5 度 | 18.5 / 77.4 度 | 2.5 / 4.9 度 |
| G1 修复后 | 过顶 | 121.5 / 124.2 度 | 28.9 / 27.0 度 | 2.9 / 4.4 度 |
| G1 修复后 | 前举 | 121.4 / 136.1 度 | 23.3 / 16.9 度 | 4.9 / 10.9 度 |
| H2 LoRA | 过顶 | 待同条件量化 | 待同条件量化 | 待同条件量化 |
| H2 LoRA | 前举 | 待同条件量化 | 待同条件量化 | 待同条件量化 |

修复后 G1 没有倒地，且左右方向误差对称性恢复。双肘距离 reference 的 180 度仍有约 44 到 59 度误差，所以静态 SMPL mode 2 不能让该预训练模型可靠地产生“手臂完全打直”的姿态。

## 复现

生成 reference：

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_teleop/bin/activate
python scripts/make_static_smpl_arm_references.py \
  --robot-profile g1 \
  --output-dir gear_sonic_deploy/reference/smpl_static_arm_test_g1
python scripts/make_static_smpl_arm_references.py \
  --robot-profile h2_lora \
  --output-dir gear_sonic_deploy/reference/smpl_static_arm_test_h2
```

终端 1 启动 G1 仿真：

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py \
  --wbc-version sonic_model12 \
  --no-data-collection
```

终端 2 启动 G1 deploy：

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
unset GEAR_SONIC_AUTO_START_AFTER_SEC
unset GEAR_SONIC_AUTO_PLAY_AFTER_SEC
unset GEAR_SONIC_AUTO_NEXT_AFTER_SEC
unset GEAR_SONIC_AUTO_STOP_AFTER_SEC
export GEAR_SONIC_DOMAIN_ID=0
./deploy.sh \
  --robot g1 \
  --input-type keyboard \
  --no-planner \
  --motion-data reference/smpl_static_arm_test_g1 \
  sim
```

输入 `Y`，等待 `Init Done`，按一次 `]` 启动 policy。初始为过顶动作，按 `N` 切换前举，按 `P` 返回过顶，按 `O` 退出。静态目测无需按 `T`；需要按上述统计口径采集 `motion_playing=1` 日志时，对每个动作按一次 `T`。

H2 不再使用相同目录：将终端 1 的 `--wbc-version` 改为 `h2_lora`，终端 2 的
`--robot` 改为 `h2_lora`，并使用
`--motion-data reference/smpl_static_arm_test_h2`。两者的 `S/R` 相同，但 wrist
reference 分别匹配各自训练契约。G1 与 H2 应顺序测试，不能同时占用 domain 0。
