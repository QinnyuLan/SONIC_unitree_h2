# H2 第二阶段部署接口核对（2026-09-18）

`runtime_contract.json` 保存完整关节名顺序及对应的 q0、action scale、Kp/Kd、输入切片。它依据原始第二阶段配置和代码生成，并读取 step 30000 checkpoint 验证两个 adapter 的 H2 action scale 与指定的 official H2 profile 一致。未连接真机。

- H2-only：`model_encoder.onnx` (1×640 → 1×64) + `model_decoder.onnx` (1×994 → 1×29)。输入 dtype=float32，部署图当前 batch 固定为 1。
- 640 = 全部 10 帧绝对 H2 q（290）+ 全部 10 帧 qdot（290）+ 全部 10 帧相对根部朝向（60）；各组时间从当前到未来 0.9 秒，间隔 0.1 秒。不能按每帧 64 维交错拼接。
- 本模型是 full-root-relative v0：参考朝向乘当前 pelvis **完整**姿态的逆，再取旋转矩阵前两列，逐行展平。不是 v1.1 的 heading-only。
- 994 = token 64 + proprio 930；proprio 的块依次为角速度 30、q−q0 290、qdot 290、已裁剪的上一步 normalized action 290、重力方向 30。每块 10 帧由旧到新，50 Hz；保留全部重力历史。
- ONNX 输出是 normalized action：q_target = q0 + scale × clip(action, −6, 6)。H2 adapter 已折叠，不再添加 G1/H2 转换。头部 pitch/yaw 在 29 维输出之外。
- full encoder 的第 0 维是标量 selector：0=H2、1=teleop、2=SMPL、3=G1；[1:4] 是旧布局保留槽，不是 selector。完整切片见 JSON。
- Kp/Kd 为训练时 H2 **输出关节坐标**的 nominal PD 数值。实际 H2 闭链的电机接口转换仍须由真机 runtime 正确实现。训练随机化后的 env0 q0 不可当作真机 nominal q0。
- 本工程现存 `gear_sonic_deploy/src/g1` 是 G1 runtime，含 G1 硬编码比例和排列；不能仅替换此 H2 ONNX 就认定完成 H2 部署。外部实际 H2 runtime 未提供，本审查不确认它的 SDK 电机顺序、IMU 坐标与闭链执行器兼容性。

生成命令：`CUDA_VISIBLE_DEVICES='' .venv_train/bin/python logs_eval/stage2_selection_20260918/build_runtime_contract.py`。

历史 step 9450、step 30000、新迁移 step 6000 的 manifest 均记录相同接口及 CPU ONNX 等价校验通过；未发现需要因此重导的既有导出 bug。是否选择某个 checkpoint 由同口径 rollout 结果决定。
