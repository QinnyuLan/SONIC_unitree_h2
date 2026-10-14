# SONIC-H2 工作总结

本文把工作区中已经完成并纳入发布包的工作按可复现流程归档。逐项实验数据和诊断细节仍保留在 `docs/report/`。

## 1. 控制与部署主线

- 完成 G1/H2 Whole Body Control 的 Python 仿真侧、Unitree DDS 桥接、ZMQ 输入输出和 C++ TensorRT 部署侧整合。
- 固化 H2 Native 31 body-joint contract、观测配置、PD/action scale、关节顺序和 selector/reference 路由。
- 保留 ONNX 导出、模型配置、运行契约、SHA256 和离线 smoke/validation 记录；部署端在目标机按 TensorRT 版本重建 engine。
- 实机启动配置集中到 `gear_sonic_deploy/config/h2_deploy.env`，包括动作集、DDS domain、网卡、四个关节组的 Kp scale 和日志目录。

## 2. sim2sim

- MuJoCo H2 场景、闭环踝/膝/腰传动、实时循环、presentation camera 和 DDS loopback 已整理到 `gear_sonic/`。
- 发布入口是 `gear_sonic_deploy/scripts/run_h2_sim2sim.sh`，采用 `models/released` 配对模型、`reference/example` 和 nominal Kp/Kd。
- 相关纸面场景、实时率、深蹲对比和脚踝仿真诊断报告包括：`h2_paper_sim2sim.md`、`g1_paper_sim2sim.md`、`h2_squat_sim_real_diagnostics.md`、`h2_ankle_*` 系列。

## 3. sim2real

- C++ 控制端支持 H2 Native encoder/decoder、ZMQ/Unitree 输出、planner、CSV 日志和可独立缩放的踝 pitch/roll、膝 Kp。
- 当前发布配置使用 `models/released`，与仿真和 VR 遥操作共用同一直接微调模型。历史报告仅用于实验追溯。
- 目标机需要 CUDA、TensorRT、ONNX Runtime、ZeroMQ、DDS 和正确的 Unitree 网络接口；这些条件由 `verify_deploy.sh` 提前检查。

## 4. VR 三点遥操作

- PICO headset/controller/ankle tracker 数据通过 XRoboToolkit 接入，Python manager 负责 ZMQ 发布、三点运动学、SMPL 手臂 retarget、腕部和头部控制。
- `run_h2_teleop_head_adjust.sh` 提供 `deploy` 与 `pico` 两个终端入口，当前 VR 配置使用 `models/released`、SONIC planner、固定头部 pitch 40° 和 Inspire hand backend。
- PICO 安装、配对、tracker 校准和网络说明在 `docs/source/getting_started/vr_teleop_setup.md`；头部调角实验记录在 `h2_head_pitch_adjustment.md`。

## 5. 机器人模型、执行器和诊断

- 纳入 H2 Unitree ROS/MuJoCo 资产、H2 ankle system identification、动态响应、热记录、backlash、range、loaded actuator 和 squat diagnostics 工具。
- `scripts/` 中的分析脚本均保留为离线工具；默认不会连接 DDS 或发出电机命令，现场采集脚本需由操作员单独启动。
- H2 Inspire hand 的 DDS/串口桥、头部控制、VLA 采集和 camera bridge 代码也随发布包保留，便于从 sim2sim 继续到现场集成。

## 6. 数据、VLA 和辅助工作

- 保留 H2 VLA 数据采集/检查、LeRobot writer、相机服务、GR00T inference client 及其安装脚本。
- 保留 SMPL/VR 静态手臂追踪、关节 frame calibration、wrist kinematics、运动参考导出和 planner ONNX 文档。
- 原始历史日志、虚拟环境、构建目录和缓存没有进入发布包；可复现实验依赖输入资源、模型和代码都已保留。

## 7. 复现建议

先执行 `python3 tools/verify_release.py`，然后只安装需要的环境：sim2sim 用 `./install.sh sim2sim`，VR 用 `./install.sh teleop`。完成回环仿真和原生部署检查后，再由现场操作员核对急停、网卡、DDS domain、动作和增益，执行 sim2real 或 VR 控制。
