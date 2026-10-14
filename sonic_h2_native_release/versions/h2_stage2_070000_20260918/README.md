# H2 第二阶段 70,000 轮部署包

推荐 checkpoint：原始实验 `20260906_204409/stage2/model_step_070000.pt`。
累计训练轮次：第一阶段2,000 + 第二阶段70,000。

## 模型

- `model_encoder.onnx`：H2动作参考，float32 `[1,640] → [1,64]`。
- `model_encoder_full.onnx`：完整encoder，float32 `[1,1751] → [1,64]`；标量selector 0=H2、1=teleop、2=SMPL、3=G1。SMPL输入用selector=2。
- `model_decoder.onnx`：上述任一token + H2自身观测，float32 `[1,994] → [1,29]`，输出H2 normalized action。
- `runtime_contract.json`：完整输入布局、29个输出关节顺序、默认姿态、动作缩放、PD参数及仿真交叉校验。
- `model_config.yaml`：训练配置追溯，不是外部真机runtime的启动配置。

## 接口必须匹配

策略50 Hz。H2 encoder 640维按全部10帧q、全部10帧qdot、全部10帧朝向分块，不能按每帧交错。参考朝向采用当前pelvis完整姿态的逆，即 `sonic_full_root_relative_v0`，不是heading-only。

930维自身观测按角速度30、q−q0 290、qdot290、上一步裁剪后normalized action290、重力30分块；各块10帧历史从旧到新。输出29维对应JSON中的policy_joint_order，执行 `q_target = q0 + scale * clip(action, -6, 6)`。LoRA和H2输入/输出转换已合并进图，不能重复添加G1/H2桥接。头部两个关节不在29维输出内。

Full encoder第0维是单个selector，之后[1:4]是旧三维保留槽，robot command从第4维起。具体切片见JSON。

本包PD对应H2输出关节坐标，闭链内部电机坐标转换由真实H2 runtime处理。仓库自带G1 runtime有G1硬编码映射，不能仅替换ONNX即视为完成H2适配。用户外部H2 runtime未提供，未连接真机验证。

## 本次筛选结果

固定256动作、每条最多20秒、确定性动作，无训练随机扰动。H2：249完成/0失败/7截断；SMPL：248完成/1失败/7截断，失败动作为 `reaching_down_R_003__A408_M`（4.50秒，末端位置终止）。这是仿真筛选结果，不是实机成功率。

70k相比73150整体跟踪接近，关节误差较低；动作变化指标H2低3.60%、SMPL低3.70%。详见 `selection_report.md` 和各 `eval_*.json`。

## 校验

在此目录执行 `sha256sum -c SHA256SUMS`。`deployment_manifest.json`保留导出时原始绝对路径作溯源，模型文件使用本包同名文件即可。

checkpoint SHA256：`64334c3824e1747ce5aafaddcd878c063c11882200d57adba2f7714f0793ab03`。
