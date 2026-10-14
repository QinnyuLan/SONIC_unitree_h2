# Sonic G1 -> Unitree H2LoRA Any2Any 迁移实现报告

生成日期：2026-05-31
最近更新：2026-06-22
论文依据：[Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking, arXiv:2605.23733](https://arxiv.org/abs/2605.23733)
实现范围：当前工程中新增/修改的 `h2lora` 训练链路，以及后续补充的 H2LoRA deploy/sim2sim/sim2real 工程接入。deploy/sim2sim/sim2real 不是 Any2Any 论文方法本身，而是为了验证和使用 H2LoRA policy 所需的工程链路；仍不包含尚未完成的 retargeting 工具本身。

## 1. 结论概览

当前实现按 Any2Any 的核心范式组织为两段：

1. **Kinematic Alignment**：让 H2 以 G1 SONIC release 的 29DoF body 语义接入 policy。实现包括 observation layout 对齐、29DoF joint order 对齐、hip pitch/roll 局部线性映射、action 反向映射，以及 H2 head 两个 DoF 的 policy 外默认指令。
2. **Dynamics Adaptation**：先 strict load `sonic_release/last.pt`，冻结预训练主体参数，只在 dynamics-sensitive 路径插入 repo-native LoRA；当前实现对应论文 S7 的思想，但落到本工程的 SONIC MLP 结构时，具体注入点是 `g1_dyn` decoder 的 proprioception slice、后续 hidden/output linear，以及 critic hidden backbone。

实现不是一个通用 Any2Any 框架，而是一个 **G1 SONIC release -> Unitree H2 headless body** 的工程化实例。它保留了 H2 31DoF 仿真本体，但对 policy 暴露和训练的动作/本体观测始终固定为 29DoF；`head_pitch_joint/head_yaw_joint` 作为 policy 外 direct command 保持默认角。

当前已经解决 IsaacLab import 阻塞，并完成 Isaac Sim 内的 strict checkpoint load、LoRA 注入、DDR cache 训练 smoke、H2 head 物理冻结验证、ref/tracking 双机器人视频可视化，以及 `/mnt/data/seed` 的 SMPL 分支 schema6 修复。2026-06-04 复查确认：旧 schema2/3/4/5 会让 SMPL 分支在 G1 release 本体上也基本不走；schema6 改为从 `soma_uniform/bvh` 做 FK 并按 SONIC 的 22 body + 2 thumb 顺序 name-map 后，同一条 walking motion 可以被 `sonic_release/last.pt` 的 G1 SMPL encoder 正常执行。H2LoRA 旧 checkpoint 在 schema6 输入下也能跟随同一条 SMPL walking reference。

因此当前结论是 **SMPL 数据链路的主要错误已经定位并修复，本地 SEED -> SONIC SMPL token 链路已用 G1 release/H2LoRA smoke 验证闭合；但旧 1M/5M 训练是在错误 SMPL schema 上完成的，不能作为修复后训练效果结论。论文级 Any2Any 复现仍需要用 schema6 重建数据并重新训练验证收敛**。

## 2. 论文方法要点与代码对应

### 2.1 预训练 WBT policy 的复用

论文设定是给定 source embodiment 的已训练 WBT specialist，通过少量 target 数据和计算迁移到目标 humanoid。论文明确强调直接 full fine-tuning 容易破坏 source prior，Any2Any 冻结大部分预训练参数，只学习小的 target-specific residual。

当前代码对应：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml`
  - `pretrained_model.path: sonic_release/last.pt`
  - `strict: true`
  - `module_mapping.policy: policy_state_dict`
  - `module_mapping.value_model: value_state_dict`
- `gear_sonic/train_agent_trl.py`
  - `materialize_lazy_params(policy, env)` 后加载 checkpoint。
  - 支持 `state_dict_key: checkpoint`，即将顶层 checkpoint 当作容器读取，再按 `policy_state_dict/value_state_dict` 加载。
  - `_assert_h2lora_release_shapes()` 强制 H2LoRA policy interface 保持 release G1 形状：actor obs 930、critic obs 1645、action dim 29。
- `gear_sonic/trl/utils/lora.py`
  - `apply_sonic_h2lora()` 先 `_freeze_module(policy)` 和 `_freeze_module(value_model)`，再只打开 LoRA 参数。

实现状态：与论文“复用 source WBT prior + 冻结大部分参数”的方向一致。不同点是论文在实验中包含多种 source/target 和多种 backbone，本实现只面向 `sonic_release/last.pt` 的 G1->H2LoRA。

### 2.2 Observation Layout Alignment

论文的 Level 1 alignment 要求将 target 机器人的观测项按 source policy 期望的语义布局排列，包括 base states、reference motion、proprioception、action history 等。

当前代码对应：

- `gear_sonic/config/manager_env/observations/policy/local_dir_hist_h2lora.yaml`
  - policy 观测项的最终 IsaacLab `PolicyCfg` concatenation 顺序为
    `base_ang_vel -> joint_pos_h2lora -> joint_vel_h2lora -> actions_h2lora -> gravity_dir`。
  - history 长度沿用 release 配置，最终 actor obs 维度为 `10*(3+3+29+29)+10*29 = 930`。
- `gear_sonic/config/manager_env/observations/critic/privileged_mf_hist_h2lora.yaml`
  - critic 观测保留 release 的 privileged/multi-future 组织方式，但 joint/action 项走 H2LoRA 29DoF 语义。
- `gear_sonic/train_agent_trl.py`
  - `_assert_h2lora_release_shapes()` 对 actor/critic/action shape 做启动时 fail-fast。
- `gear_sonic/envs/wrapper/manager_env_wrapper.py`
  - `_expected_actor_obs_dim_h2lora()` 对 wrapper 输出的 actor obs 再做 930 维检查。
- `gear_sonic/utils/mujoco_sim/base_sim.py` 与 `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`
  - deploy sim2sim 中主 IMU `gyroscope` 使用 MuJoCo root body-local angular velocity，对齐 IsaacLab `mdp.base_ang_vel == asset.data.root_ang_vel_b`；odometry 仍保留 MuJoCo/world 表达。

实现状态：与论文 Level 1 完全同向。当前没有实现“自动从 source config 推导 layout”的通用机制，而是以 `sonic_h2lora_release/config.yaml` 和 ONNX 输入契约为准写死 h2lora observation YAML 和 shape/layout guard。这样做的原因是 source checkpoint 的 shape 与 term 顺序都是硬约束，提前失败比训练或部署中 silent mismatch 更安全。

### 2.3 Joint-Level Kinematic Alignment：稀疏对应/重排

论文的 Level 2 alignment 先通过 joint correspondence/scattering matrix 将 target joint variables 映射到 source joint semantic order。论文允许 target 无对应 joint 被 mask/pad、冗余 joint 被丢弃。

当前代码对应：

- `gear_sonic/envs/env_utils/joint_utils.py`
  - `H2LORA_BODY_JOINTS_IN_G1_ORDER` 定义 H2 headless 29DoF，在 G1 policy 语义顺序下排列。
  - `H2_HEAD_JOINTS = ["head_pitch_joint", "head_yaw_joint"]` 单独列出。
  - `H2LORA_FULL_H2_ISAACLAB_DOF_INDICES` 将 full H2 31DoF IsaacLab 顺序裁剪/重排为 headless 29DoF G1 policy 顺序，同时处理 H2 ankle roll/pitch 与 G1 policy 顺序差异。
  - `get_h2lora_body_joint_indices()` 和 `get_h2_head_joint_indices()` 按 name-based mapping 取仿真里的 joint index。
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`
  - 新增 robot type `h2lora`，复用 `H2_CFG/H2_ACTION_SCALE/H2_ISAACLAB_TO_MUJOCO_MAPPING`，但设置 `self.use_h2lora_alignment = robot.type == "h2lora"`。
- `gear_sonic/envs/manager_env/mdp/commands.py`
  - 当 `use_h2lora_alignment` 且 motion lib 是 29DoF、robot 是 31DoF 时，body joints 使用 `get_h2lora_body_joint_indices()`，extra joints 使用 head indices。

实现状态：对应论文的 sparse scattering/permutation 思路，但不是通用矩阵类实现，而是 name-based index list。这样设计是因为当前目标只有 G1->H2，name-based 更可审计，也避免手写矩阵和仿真 joint order 漂移。

### 2.4 Hip Decoupling / 局部线性轴映射

论文在 joint-level alignment 中给出 hip decoupling，用一个局部线性变换处理倾斜 hip axis 的 source/target 表达差异。用户计划中要求使用：

- target -> source: `source = pinv(B_g1) @ B_h2 @ target`
- source -> target: `target = pinv(B_h2) @ B_g1 @ source`

当前代码对应：

- `gear_sonic/envs/env_utils/h2lora_alignment.py`
  - `parse_mjcf_joint_axes()` 和 `parse_urdf_joint_axes()` 从 MJCF/URDF 零姿态解析 joint axis。
  - `hip_alignment_matrices()` 对每侧 hip pitch/roll 构造 `source_basis` 和 `target_basis`，并计算：
    - `target_to_source = torch.linalg.pinv(source_basis) @ target_basis`
    - `source_to_target = torch.linalg.pinv(target_basis) @ source_basis`
  - `h2_to_g1_semantic()` 对 H2 29D physical joint values 的左右 hip pair 应用 `target_to_source`。
  - `g1_to_h2_physical()` 对 G1 semantic action/reference values 的左右 hip pair 应用 `source_to_target`。
  - `assert_axis_audit()` 检查 G1/H2 hip pitch/head axes 的预期性质。

实现状态：这部分是当前实现中最接近论文数学表达的部分。差异是论文写法使用通用 `D_r`/`Phi_r` 矩阵描述，本实现只对左右 hip pitch/roll 2D 子空间应用局部矩阵，其他 joint 由 identity/name order 承担。这是因为当前已知的非平行轴 gap 集中在 hip pitch/roll；扩大到全身矩阵会增加未验证的耦合风险。

需要特别注明：真实 G1/H2 hip pitch/roll basis 不在同一个 3D 平面内时，`pinv(B_g1) @ B_h2` 与 `pinv(B_h2) @ B_g1` 的组合不是严格 identity，而是一个投影/最小二乘近似。当前单测因此不再要求 `G1 -> H2 -> G1` 误差为 `1e-6`，而是验证代码精确实现论文式两个 `pinv` 矩阵组合。这个信息损失是当前迁移风险之一，不应在报告中写成无损 roundtrip。

### 2.5 Parallel Joint Coupling

论文还提出对 closed-chain 或 parallel mechanism 加入 `J_r`，例如并联 ankle 或 closed-loop waist，使 alignment 不只是 permutation。

当前代码状态：

- 没有实现通用 closed-chain Jacobian correction `J_r`。
- 当前只处理 H2 ankle **顺序差异**：`H2LORA_BODY_JOINTS_IN_G1_ORDER` 与 `H2LORA_FULL_H2_ISAACLAB_DOF_INDICES` 把 ankle pitch/roll 放回 G1 policy 期望位置。
- 没有对 H2 ankle 的机械耦合、传动比、并联约束做 Jacobian 级变换。

设计原因：

- 当前 H2 资产在工程内以 actuated joint state/action 的 IsaacLab 表示进入 policy wrapper；已有资料和代码没有提供可直接验证的 ankle closed-chain Jacobian。
- 为了避免想当然引入错误动力学映射，本实现只做已能从 joint names/order 和 axis audit 验证的部分。

风险：

- 如果 H2 ankle 实际控制接口与 kinematic joint 并非一一等效，仅重排可能不足；这会表现为 ankle tracking/reward 低、足端接触不稳或 action residual 学习负担过大。后续需要结合 retarget 输出、URDF/MJCF actuator 定义和 sim rollout 做专项 audit。

### 2.6 Reference Motion / Command 的语义与物理双表示

论文要求 target 机器人在 source-aligned semantic space 中向 frozen source policy 提供 reference/obs，但执行和物理 reward 仍落到 target embodiment。

当前代码对应：

- `gear_sonic/envs/manager_env/mdp/commands.py`
  - `joint_pos` / `joint_vel` / `joint_pos_multi_future` / `joint_vel_multi_future` 在 `use_h2lora_alignment` 时把 MotionLib 的 H2 physical 29D 转为 G1 semantic 29D。
  - 新增 `joint_pos_physical` / `joint_vel_physical`，用于 reset/物理侧。
  - reset 时若 H2LoRA，使用 `joint_pos_physical/joint_vel_physical` 填入 H2 body joints，head 填默认值。
  - H2LoRA metrics 中默认 `error_joint_pos/error_joint_vel` 现在按 H2 physical reference 与 H2 physical robot state 计算；同时额外记录 `error_joint_pos_semantic/error_joint_vel_semantic`、`error_hip_*_physical` 和 `error_ankle_*_physical`，避免把 G1 semantic 与 H2 physical 直接相减造成误判。
- `gear_sonic/envs/manager_env/mdp/observations.py`
  - `joint_pos_wo_hand()` 和 `joint_vel_wo_hand()` 在 H2LoRA 下从 H2 body joint indices 取物理状态，减 default 后映射到 G1 semantic。
  - `last_action_wo_hand()` 优先返回 `_last_policy_action_h2lora`，即原始 29D policy action。

实现状态：符合论文“observation/reference/action spaces aligned before PEFT”的核心要求。一个工程性细节是 `last_action` 保存原始 29D policy action，而不是从 H2 31D sim action 反推；这样避免 H2 scale 和 head direct command 污染 policy action history。

### 2.7 Action Space Alignment

论文要求 policy 在 source-aligned action convention 下输出，再映射回 target robot control convention 执行。

当前代码对应：

- `gear_sonic/envs/wrapper/manager_env_wrapper.py`
  - `_expand_h2lora_policy_action()` 要求 policy action 最后一维为 29。
  - 先使用 G1 release action scale 将 raw action 还原到 G1 joint delta 空间：`g1_delta = policy_actions * g1_scale`。
  - 再用 `h2lora_alignment.g1_to_h2_physical(g1_delta)` 转到 H2 physical headless 29D delta。
  - 再除以 H2 action scale 得到 H2 action manager 接收的 normalized body action。
  - 最后创建 31D sim action，将 29D body 写入 H2 body indices，将 head 两维写成默认角对应的 direct command。
- `step()` 中在 action clipping 前保存 raw 29D policy action 到 `_last_policy_action_h2lora`。
- action clipping 时保留 head direct command 不被 clip 改写。

实现状态：对应论文 action alignment。与论文无关但必要的工程设计是显式处理 G1/H2 action scale，因为当前工程中的 policy action 是 normalized offset，不是裸 joint angle。

本轮实际训练中修正了一个关键实现细节：`_g1_body_action_scale()` 依赖 `re.fullmatch()` 匹配 G1 action scale 正则，wrapper 现在显式 import `re`，避免静默 fallback 到全 1 scale；`_action_scale_tensor()` 也按 `scale[..., idx]` 处理 IsaacLab action term 中 `(num_envs, num_joints)` 形状的 per-env scale。修正后 8 env DDR smoke 的第一轮 `error_joint_pos` 为 `0.0550`，明显低于修正前同类训练中的 `0.33+` 量级。

### 2.8 H2 Head 冻结

论文没有专门讨论“目标机器人多出来的 head DoF 由 policy 外控制”的实现细节；这是本任务约束。

当前代码对应：

- H2 仿真仍是 31DoF，但 `policy_action_dim: 29`。
- `direct_joint_defaults` 在 `sonic_h2lora.yaml` 中设置：
  - `head_pitch_joint: 0.0`
  - `head_yaw_joint: 0.0`
- `commands.py` 中 extra joints 使用 head indices，reset 时 head joint pos/vel 填默认值/零速度，reset noise 不作用于 head。
- `manager_env_wrapper.py` 每步将 head action 计算为 `(head_default - current_default) / head_scale`，使 action manager 发送默认角目标。
- `manager_env_wrapper.py` 在 `reset()` 和每个 `env.step()` 后调用 `_lock_h2lora_head_state()`，把 head joint state 直接重写为默认角/零速度。增加这一步是因为仅发送默认 action target 时，隐式执行器和物理扰动仍可能让 head 状态偏离；最新 head-lock 训练日志中 `Env/step/h2lora/error_head_default_pos=0.0000`，说明当前代码满足“训练时 head 冻结默认姿态”的约束。
- `modular_tracking_env_cfg.py` 在 `robot.type == "h2lora"` 时把 startup `add_joint_default_pos` 的 `joint_names` 从 `[".*"]` 改为 29 个 body joints，避免 head default position randomization。当前日志中的 debug print 发生在该替换之前，所以 debug 行里仍可能看到 `joint_names=['.*']`，但 EventManager 构建前已被替换。

设计原因：用户明确要求不改模型，policy 只控制 29DoF body；head 类似灵巧手 direct command，不进入 policy 学习。

风险：如果 H2 硬件或仿真默认 head 姿态不是 0/0，需要只改 `direct_joint_defaults`，否则 reset/reference/action 三处会保持 0/0。

## 3. LoRA 动力学适配实现细节

### 3.1 论文 S7 对应关系

论文 LoRA scope 消融中 S7 为最佳设置：actor backbone、actor proprioception input projection、actor output projection、critic backbone 加 LoRA；reference input、critic input/output 等保持冻结。论文解释为 reference stream 更偏 embodiment-agnostic，而 proprioception/action/critic backbone 更直接承载 target dynamics residual。

当前代码对应：

- `gear_sonic/trl/utils/lora.py`
  - `LoRALinear` 将原始 `nn.Linear` 包成 frozen base + trainable low-rank residual。
  - `lora_b` 零初始化，因此插入后初始函数等于 base network。
  - `rank` 默认 32，`alpha` 默认等于 rank，`dropout` 默认 0。
- `apply_sonic_h2lora()` 的具体注入：
  - actor 只要求存在 `policy.actor_module.decoders["g1_dyn"].module`。
  - `g1_dyn` 第一个 Linear 只对 proprioception slice 加 LoRA：`input_slice=(token_dim, first_child.in_features)`；token/reference slice 不参与 LoRA。
  - `g1_dyn` 后续 Linear 全部加 LoRA，覆盖 hidden backbone 和 output layer。
  - encoders、FSQ、`g1_kin` decoder、base weights 全部冻结。
  - critic 的 `critic_module.module` 中只对 `[1:-1]` Linear 加 LoRA，input projection 和 value output projection 冻结。
  - trainable ratio 超过 `max_trainable_ratio` 会报错；配置默认 8%。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml`
  - `rank: 32`
  - `alpha: 32`
  - `dropout: 0.0`
  - `freeze_noise_std: true`

实现状态：与论文 S7 的模块选择原则一致，但不是逐层复刻论文中所有 backbone 类型。论文含 Transformer/MLP/OliWBT/Sonic 多种结构，本工程将 S7 映射到当前 SONIC MLP release 的 `g1_dyn` decoder 和 critic MLP。

未由论文明确给出的参数：

- 论文明确支持 rank=32 的选择，但没有在当前可读正文中给出本工程所需的 `alpha/dropout` 细节。
- 当前采用 LoRA 标准缩放 `alpha/r = 1`，即 `alpha=32, rank=32`，`dropout=0.0`。原因是 WBT PPO 是闭环控制，dropout 会给 action 分布引入额外训练噪声；初始等价 base policy 更利于稳定迁移。

### 3.2 Optimizer 只训练 LoRA

当前代码对应：

- `gear_sonic/trl/trainer/ppo_trainer.py`
  - `create_optimizer()` 只收集 `requires_grad=True` 的参数。
  - actor/critic 分开 param group，critic lr 可通过 `critic_learning_rate` 独立设置。
  - 自适应 KL 调整学习率时，通过 `lr_scale_to_actor` 保持 critic/actor lr 比例。
- `gear_sonic/trl/modules/actor_critic_modules.py`
  - 当 `freeze_noise_std: true`，`std` 或 `log_std` 不训练。

实现状态：符合“冻结 base，只优化 LoRA residual”的要求。需要注意，optimizer 是否只含 LoRA 取决于 `apply_sonic_h2lora()` 在 trainer 创建前成功执行；当前 `train_agent_trl.py` 的调用顺序满足这一点。

## 4. MotionLib / 数据链路 / DDR cache

### 4.1 Retargeting 输入契约

论文实验使用 AMASS，经 GMR/retargeting 到 target humanoid。当前任务中 retargeting 尚未完成，因此实现只定义 MotionLib PKL 输入契约和训练侧处理。

当前代码对应：

- `gear_sonic/utils/motion_lib/motion_lib_base.py`
  - 当 `h2lora_headless_dof: true`：
    - PKL 必须包含 `dof` 或 `dof_pos`。
    - 29DoF PKL 必须包含 `joint_names` 或 `dof_names`，且顺序必须等于 `H2LORA_BODY_JOINTS_IN_G1_ORDER`。
    - 31DoF PKL 若提供 joint names，其集合必须等于 29 body joints + 2 head joints。
    - DoF 只能是 29 或 31。
  - load 后如果是 31DoF，会先按 `h2lora_full_dof_order` 处理 full-body 顺序；当前 seed 数据链路输出 `mujoco`/MJCF 顺序，所以 loader 先用 `mujoco_to_isaaclab_dof` 转为 H2 IsaacLab 顺序，再用 `H2LORA_FULL_H2_ISAACLAB_DOF_INDICES` 裁剪/重排为 29DoF。
  - SMPL directory mode 现在用 basename 建索引，支持 `smpl/220705/xxx.pkl` 这样的 session 子目录；加载单条 `{key: entry}` PKL 时会先 unwrap。
  - 当 SMPL PKL fps 与 MotionLib `target_fps` 不一致时，`pose_aa`、`smpl_joints`、`transl` 都同步插值到 target fps，避免 robot FK 从 30Hz 变 50Hz 后 SMPL 分支帧数不一致。
- `commands.py`
  - 启用 H2LoRA 时强制 `motion_lib_cfg.disable_dof_reorder = True` 和 `h2lora_headless_dof = True`，避免普通 H2 reorder 与 H2LoRA headless reorder 叠加。

实现状态：这是论文之外的工程性契约。原因是目前 retargeting 还未完成，训练链路必须先把可接受的 PKL schema 锁死，否则 joint order 错一位会让 LoRA 学到错误的残差。31DoF 输入保留 head 是为了让 H2 MJCF FK 使用完整运动学树；policy/reward 所需的 headless 29DoF 在 MotionLib load 后统一裁剪。

### 4.2 `/mnt/data/seed` 数据链路实现

用户补充的数据根目录为 `/mnt/data/seed`：

- `h2/csv`：H2 retarget CSV，实测 header 为 root 平移、root 欧拉角和 31 个 H2 `_dof` 列；单位与原工程 Bones-SEED converter 一致，root 平移为 cm、角度为 degree。
- `smpl-x/uniform`：文件名虽为 SMPL-X uniform npz，但实测 `joint_names` 是 SOMA-X/BVH 77 关节顺序，例如 `Hips, Spine1, Spine2, Chest, Neck1, ... LeftLeg ... RightToeEnd`；`frame_time=0.008333...`，即约 120Hz；`unit=meters`，`rotation_repr=rotvec`，`poses` 形状为 `(T, 77, 3)`。因此这里的 `poses[:, :24]` 不能当作 SONIC SMPL24 pose 使用。
- `soma_uniform/bvh`：原始 mocap BVH。schema6 开始直接消费 BVH：通过 `smpl-x/uniform` 中的 `source_bvh_relpath` 找到同名 BVH，做 SOMA FK，再 name-map 到 SONIC SMPL encoder 需要的 24 个 body/thumb joints。

新增代码：

- `gear_sonic/data_process/build_h2lora_seed_motion_lib.py`
  - 以 basename 匹配 `h2/csv/**/*.csv` 与 `smpl-x/uniform/**/*.npz`。
  - 支持 `--keyword idle`、`--include-regex`、`--limit`、`--num-workers`、`--skip-existing`。
  - 默认 `--fps-source 120 --fps 50`；当 `fps_source % fps == 0` 时使用 stride downsample，当 `120Hz -> 50Hz` 这种非整数比例时使用时间轴线性重采样，root quaternion 使用 Slerp。原因是直接 `stride=2` 会把 120Hz 数据实际变成 60Hz，却在 PKL 中标成 50Hz。
  - H2 robot PKL 输出到 `output_root/robot/<session>/<key>.pkl`，字段包括 `root_trans_offset`、`pose_aa`、`dof`、`root_rot`、`fps`、`joint_names`、`dof_order: mujoco`、`body_order: h2_mjcf`。
  - H2 `pose_aa` 不是 headless 29 body，而是完整 H2 MJCF body order，当前实测 shape 为 `(T, 32, 3)`；`dof` 为 31D H2 MJCF actuator order，包含 head。这样 FK 使用完整 H2 树，之后 loader 再转成 policy 29D。
  - SMPL PKL 输出到 `output_root/smpl/<session>/<key>.pkl`，字段包括 `pose_aa` `(T, 72)`、`smpl_joints` `(T, 24, 3)`、`transl` `(T, 3)`、`fps` 和 schema metadata。
  - 当前代码版本为 schema6，默认 `--smpl-mode soma-bvh`、`--smpl-root-mode soma-bvh-compatible`。`smpl_joints` 从 BVH FK 的 SOMA skeleton positions 生成，先减 `Hips` 得到 pelvis-centered body-local vectors，再从 Y-up 转为 Z-up；不携带 root translation。
  - schema6 的 24 个 joint 不是随意的标准 SMPL24，而是对齐 `compute_human_joints(..., use_thumb_joints=True)` 的输出：前 22 个 body joints，最后两个是 `LeftHandThumb1/RightHandThumb1`。映射表写在 `SMPL24_FROM_SOMA_NAMES`，并写入每个 PKL 的 `smpl_joints_source_names` metadata。
  - schema6 的 `pose_aa` 只保存 root rotvec，非 root 23 个关节置零，因为本地 `smpl-x/uniform poses` 是 SOMA-X 77 关节顺序而不是 SONIC 可用的 SMPL24 body pose。root rotvec 经过 `_soma_bvh_root_quat_to_smpl_pose_rotvec()` 编码，使 runtime 的 `smpl_root_ytoz_up + remove_smpl_base_rot` 与原 SOMA/BVH 链路的 `smpl_root_ytoz_up + remove_bvh_base_rot` 旋转等价。
  - `transl` 保存 SOMA/BVH `Hips` 的 Y-up world translation，仅作为 MotionLib 字段保留；当前 SMPL tokenizer observation 不直接消费该平移。
  - `--skip-existing` 现在同时检查 robot PKL 的 fps/joint order，以及 SMPL PKL 的 schema version、fps、`smpl_joints_source`、`smpl_root_mode`，避免 30Hz 或旧 schema 文件被误跳过。
  - 提供 `--smpl-mode human-fk`、`--smpl-mode smplx-fk` 和 `--smpl-mode zeros` 作为诊断 fallback；全量训练应使用 schema6 默认的 `soma-bvh`，旧 `smplx-fk`/`human-fk` 训练结果不再作为 SMPL 分支有效性结论。
- `scripts/build_h2lora_idle_debug_data.sh`
  - 默认生成极小 idle 子集到 `/mnt/data/seed/h2lora_motion_lib/idle_debug`。
  - 输出可直接用于训练的环境变量：
    - `H2LORA_MOTION_FILE=/mnt/data/seed/h2lora_motion_lib/idle_debug/robot`
    - `H2LORA_SMPL_MOTION_FILE=/mnt/data/seed/h2lora_motion_lib/idle_debug/smpl`
- `gear_sonic/trl/utils/smplx/body_model/smplx_lite.py`
  - 修正原来硬编码的 `groot/rl/...` 回归器路径，改为相对当前文件加载 `smplx2smpl_sparse.pt` 与 `smpl_neutral_J_regressor.pt`。
  - 原先 forward 每次写 `data/human_joints_info.pkl`，现在只在 `GEAR_SONIC_DUMP_SMPLX_JOINTS=1` 时 dump，避免数据预处理污染工作区。

这部分不是论文直接规定的实现细节，而是把论文中的 target motion dataset / retarget result 落到本工程 MotionLib schema 的桥接层。schema6 修复后，训练数据默认按 SONIC `target_fps=50` 生成；早期 30Hz/`smplx-fk` idle 和 5M 实验属于旧数据链路诊断记录，不能代表当前 SMPL 链路。

### 4.3 DDR cache

论文强调 Any2Any 降低数据/计算成本，但没有规定本工程的 DDR cache 机制。DDR cache 是按用户资源约束新增的训练效率优化。

当前代码对应：

- `motion_lib_base.py`
  - directory 模式下，当 `cache_all_motions_to_ddr: true`，会按 `ddr_cache_max_bytes` 预加载 MotionLib PKL 到内存。
  - 默认上限来自 config，当前 `sonic_h2lora.yaml` 设置为 `47244640256` bytes，约 44GiB。
  - 加载统计写入 `ddr_cache_stats`。
  - 已预加载的数据通过 `curr_file["data"]` 直接使用，避免重复 `joblib.load`。
- `sonic_h2lora.yaml`
  - `max_cpu_workers: 16`
  - `cache_all_motions_to_ddr: true`
  - `ddr_cache_max_bytes: 47244640256`
- `train_agent_trl.py`
  - 若 SwanLab 启用，会在 step 0 记录 cache enabled/count/GB/maxGB。

实现状态：完全是工程扩展，不是论文方法的一部分。设计原因是单卡 RTX 5090 + 48GB DDR 约束下，重复 joblib/disk IO 会拖低 GPU 利用率。

## 5. 训练配置与日志

### 5.1 H2LoRA 实验入口

当前主入口：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml`

关键配置：

- `num_envs: 2048`
- `num_steps_per_env: 24`
- `num_mini_batches: 4`
- `robot.type: h2lora`
- `policy_action_dim: 29`
- `use_h2lora_bridge: true`
- `motion_lib_num_dof: 29`
- `use_h2lora_alignment: true`
- `asset.assetFileName: h2.xml`
- SwanLab project: `sonic-h2lora`

论文训练 procedure 表述为保持 source training setup 的 action/observation/reward/PPO/domain randomization 等尽量一致，仅引入 kinematic alignment 和 dynamic adaptation。当前配置继承原 SONIC manager env、PPO、actor/critic、tokenizer、reward/event/termination 配置，并新增 H2LoRA 特定 observation/action bridge/LoRA/cache/SwanLab。

需要注明的差异：

- 当前配置里 `manager_env.rewards.feet_acc.weight: -2.5e-6` 是工程配置选择，不是论文直接给出的 Any2Any 必要项。
- 论文实验资源提到多卡 A100/H100；当前配置按用户单 RTX 5090 约束给出 2048 env 和 capacity probe。

### 5.2 SwanLab

用户要求按 `~/tools/skills/swanlab-training-skill/` 使用 SwanLab。当前实现没有读取该目录内额外脚本，而是在工程内直接接入 SwanLab。

当前代码对应：

- `gear_sonic/train_agent_trl.py`
  - rank 0 初始化 `swanlab.init()`。
  - `SWANLAB_MODE` 环境变量优先，默认 `cloud`。
  - 记录 config、tags、group、job_type。
  - step 0 记录 LoRA 参数量和 cache stats。
- `gear_sonic/trl/callbacks/swanlab_callback.py`
  - rank 0 `on_log()` 过滤数值型 logs 并写入 SwanLab。
  - `on_train_end()` 调用 `swanlab.finish()`。
  - 当 `SWANLAB_MODE=disabled/off/none`，或 callback 被 Hydra defaults 保留但 `use_swanlab=false` 未初始化 run 时，callback 会跳过 `log/finish`，避免诊断训练因 `swanlab.init()` 未调用而中断。
- `gear_sonic/trl/trainer/ppo_trainer.py`
  - PPO log 中增加 GPU memory allocated/reserved/max allocated。

实现状态：这是用户要求的工程扩展，不是论文方法内容。

### 5.3 uv 环境与容量探测

当前脚本：

- `scripts/setup_h2lora_uv.sh`
  - `uv venv .venv-h2lora --python 3.10`
  - `uv pip install -e "gear_sonic[training]" swanlab pytest imageio-ffmpeg`
  - 验证 `torch/swanlab` import。
  - 该环境用于数据预处理、轻量 pytest、MuJoCo 离屏视频渲染等非 IsaacSim 工具脚本；不再要求在 Python 3.10 venv 中 import `isaaclab`。
- `scripts/setup_h2lora_isaac_python.sh`
  - 使用本机 Isaac Sim Python：`/home/sky/software/isaac-sim/python.sh`，实际 Python 为 3.11.13。
  - 默认 IsaacLab 源码为 `/home/sky/workspace/basic/IsaacLab-2.3.2`，用 `pip install --no-deps -e` 安装 `isaaclab/isaaclab_assets/isaaclab_tasks/isaaclab_rl/isaaclab_mimic/isaaclab_contrib`。
  - 用 `uv pip install --python /home/sky/software/isaac-sim/kit/python/bin/python3 --excludes ...` 安装训练依赖，并排除 `torch/triton/nvidia-*`，避免覆盖 Isaac Sim 自带的 CUDA/Torch 栈。
  - 对本地 IsaacLab `urdf_converter.py` 做幂等兼容补丁：当本机 URDF importer 没有 `set_merge_fixed_ignore_inertia()` 时跳过该调用。这个补丁是为当前 Isaac Sim/IsaacLab 组合服务的，不属于仓库内 h2lora 方法。
- `scripts/h2lora_capacity_probe.sh`
  - 要求设置 `H2LORA_MOTION_FILE`。
  - 支持 `H2LORA_SMPL_MOTION_FILE`，默认 `dummy`。
  - 依次尝试 `1024/2048/3072/4096` env，运行 2 个 PPO iteration。
  - 默认训练解释器为 `/home/sky/software/isaac-sim/python.sh`，可用 `H2LORA_TRAIN_PYTHON` 覆盖。
  - probe 中关闭 DDR cache 以专注显存/仿真容量。
  - 脚本只根据训练进程是否成功选择最大 env 数；显存峰值需要看 PPO/SwanLab 中记录的 GPU memory 指标，而不是脚本结束后另开 Python 进程读取。
- `scripts/visualize_h2lora_ref_tracking.py`
  - `--collect-only` 在 Isaac Sim Python 中启动 H2LoRA env、strict load 指定 checkpoint、套用 LoRA wrapper，并从 1 个 env 采集 ref/tracking rollout 到 `trajectory_ref_tracking.npz`。
  - `--render-only` 在 `.venv-h2lora` 中用 MuJoCo 复刻部署侧 sim 可视化思路：同一场景中放两台 H2 MJCF 机器人，蓝色半透明表示 reference，橙色表示 tracking，按每帧 root pose + 31DoF MuJoCo joint state 写入 qpos，并保存 `ref_tracking.mp4` 和 `preview.png`。
  - 默认输出目录为 `visualizations/h2lora/<实验名称>_<YYYYmmdd_HHMMSS>`；也可用 `--output-dir` 指定精确目录。目录内固定保存 `trajectory_ref_tracking.npz`、`collect_meta.json`、`ref_tracking.mp4`、`preview.png`、`render_meta.json`。
  - 采集默认关闭 domain randomization events 和 early termination，只用于生成连续 tracking 可视化；若需要完全复现训练 termination，可加 `--keep-events --keep-terminations`。
- `gear_sonic/pyproject.toml`
  - `training` extra 增加 `tensordict` 和 `vector-quantize-pytorch`。
  - `open3d` 没有加入 training extra；`torch_humanoid_batch.py` 中 `open3d` 已改为可选 import。训练/FK 不需要 mesh 时可以没有 open3d，调用 `mesh_fk()` 时才会显式报错。

实现状态：这些脚本是资源约束下的工程配套，不属于论文。

### 5.4 Encoder sampling：从 SMPL-only 回到 release mixed/legacy

历史背景：用户曾提出 H2 与 G1 构型差距较大，encoder 和 FSQ 都冻结，因此训练初期先只开 SMPL 分支 encoder 输入。这个策略在旧 schema2/3/4 SMPL 数据链路下表现为“输入 SMPL 就原地站/倒地”。2026-06-04 复查后确认主因不是 SMPL encoder 天然不足，而是本地 SEED `smpl-x/uniform` 被误当成 SMPL24 使用。schema6 修复后，当前代码恢复 release 风格的 mixed encoder sampling。

当前代码对应：

- `sonic_h2lora.yaml`
  - `manager_env.commands.motion.encoder_sample_probs` 为 `g1: 1.0, teleop: 1.0, smpl: 1.0`。
  - `manager_env.commands.motion.optimize_encoders_ratio_for_CHIP: false`。
  - `manager_env.commands.motion.teleop_sample_prob_when_smpl: 0.5`。
  - `algo.config.actor.backbone.optimize_encoders_ratio_for_CHIP: false`。
- `scripts/run_h2lora_seed5m_balanced_mixed_50k.sh`
  - 默认 `H2LORA_OPTIMIZE_ENCODERS_RATIO_FOR_CHIP=false`，避免命令行覆盖回旧 one-hot 行为。
- `commands.py`
  - 修复当 `encoder_sample_probs_no_smpl` 总和为 0 时的除零问题；若某条 motion 没有 SMPL 数据，fallback 到 G1 encoder，避免 NaN 概率。

需要特别说明：代码仍然实例化 G1/teleop/SMPL 三个 encoder，以保证 `sonic_release/last.pt` strict load 的 state_dict key 和 shape 不变。`optimize_encoders_ratio_for_CHIP=false` 会恢复 release legacy multi-hot 行为：SMPL-native episode 会同时激活 G1 encoder，并可按概率激活 teleop encoder，用于 latent alignment。旧 checkpoint 保存的 config 仍可能显示 `optimize_encoders_ratio_for_CHIP=true`；该值只代表旧 run，不代表当前默认训练配置。

## 6. 新本体 `h2lora` 与原 `h2` 的隔离性

用户要求“按照新的本体 h2lora 接入，不要动原本 h2 本体”。

当前代码状态：

- `modular_tracking_env_cfg.py` 中新增 `"h2lora"` key，复用 H2 资产和动力学参数。
- 原 `"h2"` key 保持存在，未被替换。
- H2LoRA 特定行为由以下开关触发：
  - `robot.type == "h2lora"`
  - `use_h2lora_alignment`
  - `use_h2lora_bridge`
  - `h2lora_headless_dof`
- 原 H2 若不设置这些 flag，不会走 H2LoRA 的 29DoF policy bridge。

实现状态：符合隔离要求。需要注意的是部分公共文件被修改以支持新分支，但运行时行为通过 robot type/config flag 分流。

## 7. 测试与验证状态

已完成的静态/轻量验证：

- `python3 -m py_compile` 对本轮新增/修改 Python 文件通过。
- `bash -n scripts/*.sh` 通过。
- `.venv-h2lora` 已安装 `pytest/tensordict/vector-quantize-pytorch` 等轻量测试依赖；`pytest gear_sonic/tests/test_h2lora_alignment.py gear_sonic/tests/test_lora_adaptation.py -q` 结果为 `6 passed`。
- Isaac Sim Python 训练环境已验证以下 import：
  - `isaaclab=0.54.3`
  - `hydra=1.3.2`
  - `trl=0.28.0`
  - `transformers=5.9.0`
  - `accelerate=1.13.0`
  - `swanlab=0.7.19`
  - `tensordict=0.12.4`
  - `vector_quantize_pytorch=ok`
- 已生成极小 idle 数据集：
  - 历史说明：这是 schema6 修复前的旧 30Hz/`smplx-fk` smoke 记录，只说明当时 MotionLib loader 能跑通，不代表当前默认数据链路。
  - 命令：`H2LORA_IDLE_LIMIT=2 H2LORA_DATA_WORKERS=1 PYTHON_BIN=.venv-h2lora/bin/python bash scripts/build_h2lora_idle_debug_data.sh`
  - 输出：`/mnt/data/seed/h2lora_motion_lib/idle_debug`
  - manifest 中两条 motion：`Idle_Left_001__A017`、`Idle_Left_001__A017_M`
  - robot PKL 实测：`pose_aa (2297, 32, 3)`、`dof (2297, 31)`、`fps 30`、`dof_order mujoco`
  - SMPL PKL 实测：`pose_aa (2297, 72)`、`smpl_joints (2297, 24, 3)`、`transl (2297, 3)`、`fps 30`
- 已做 MotionLib CPU loader smoke：
  - 使用 `MotionLibRobot` 加载 `Idle_Left_001__A017`。
  - 注意：`open3d` 现在是可选依赖；没有 open3d 时 `Humanoid_Batch.load_mesh()` 会跳过 mesh map，普通 MotionLib FK/训练不受影响。只有调用 `mesh_fk()` 时才需要安装 open3d。
  - DDR cache 预加载 1 个 PKL，`loaded_bytes=1899545`。
  - H2 FK + 30Hz->50Hz 插值后总帧数为 3827。
  - `dof_pos/dof_vel` 均为 `(3827, 29)`，说明 31D H2 MJCF 输入已按 `mujoco_to_isaaclab_dof` + headless index 转为 policy 29D。
  - `body_pos_w_full/body_quat_w_full` 为 `(3827, 32, 3/4)`。
  - `smpl_pose/smpl_joints/smpl_transl` 为 `(3827, 72)`、`(3827, 24, 3)`、`(3827, 3)`，说明 SMPL 分支与 robot FK 帧数同步。
  - `motion_has_smpl == [True]`，说明 SMPL-only encoder sampling 有真实 SMPL 数据可用。

已完成的 IsaacLab / PPO 验证：

- 解决 `.venv-h2lora` 中 `import isaaclab` 失败的问题：仿真训练改用 Isaac Sim Python 3.11，而 `.venv-h2lora` 保留给数据处理和轻量测试。
- 启动时完成 `sonic_release/last.pt` strict load、H2LoRA shape guard 和 TRL checkpoint compatibility shim。
- LoRA 注入统计：`trainable=866272/38240091 (2.265%)`，模块数 `12`；optimizer 只包含 `866272` 个可训练参数，其中 actor `489440`、critic `376832`。
- 1M 级 idle 数据集与长训：
  - 历史说明：这是 schema6 修复前的旧 `fps=30`/`smpl_mode=smplx-fk` 训练，不能作为当前 schema6 SMPL 分支收敛结论。
  - 数据集目录：`/mnt/data/seed/h2lora_motion_lib/idle_1m`。
  - 构建方式：从 `/mnt/data/seed/h2/csv` 与 `/mnt/data/seed/smpl-x/uniform` 中按 basename 配对，筛选 `idle`，选择 3693 条动作；`fps_source=120`、`fps=30`、`smpl_mode=smplx-fk`。
  - 校验结果：robot/smpl 各 `3693` 个 PKL，robot/smpl 总帧数均为 `1,002,885` 帧（30Hz），单条 robot 动作最短 `82` 帧、最长 `2733` 帧，磁盘体积约 `697M`。
  - 已启动 100k PPO iteration 长训：`exp_var=h2lora_idle1m_100k_4096env`，`num_envs=4096`、`num_steps_per_env=24`、`num_mini_batches=4`、`num_learning_epochs=5`、`cache_all_motions_to_ddr=true`、`max_cpu_workers=16`、`ddr_cache_max_bytes=47244640256`。
  - 注意：这里的 `100k` 按当前 TRL/PPO 配置语义是 `100000` 个 PPO iteration，不是 100k environment steps；每 iteration 为 `4096 * 24 = 98,304` env steps。
  - 训练进程：wrapper PID `1315940`、Python PID `1315954`；日志 `/tmp/h2lora_longtrain/h2lora_idle1m_100k_4096env_20260601_010526.log`，GPU 采样日志 `/tmp/h2lora_longtrain/h2lora_idle1m_100k_4096env_20260601_010526_gpu.log`。
  - 训练输出目录：`logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_idle1m_100k_4096env-20260601_010528`。
  - SwanLab cloud 已开启并成功上传，run URL：`https://swanlab.cn/@sky931/sonic-h2lora/runs/uqaoca08wn1hjtgn4y6ie`。
  - 初始化日志显示 H2LoRA bridge 生效：`policy_action_dim=29`、`sim_joints=31`、`head_defaults=[0.0, 0.0]`；MotionLib DDR cache 预载 `3693/3693` files，日志统计 `0.85 GB`。
  - 第 40 iteration 已保存 `last.pt`；后续 `last.pt` 每 20 iteration 覆盖保存，完整 checkpoint 每 1000 iteration 保存一次。
  - 前 40+ iteration 性能：约 `27.2k steps/s`，单 iteration 约 `3.6s`，启动阶段 ETA 约 `3.6e5s`（约 4.2 天）。GPU 采样显示显存峰值约 `14.6GB`，稳定约 `11.2GB`；GPU 利用率约 `76%~84%`，功耗采样约 `175W~224W`。
  - 第 41 iteration 训练指标：reward `-1.25804`，episode `error_joint_pos_physical=0.4325`、`error_joint_pos_semantic=0.4289`、`error_hip_pos_physical=0.2925`、`error_ankle_pos_physical=0.2459`、`termination_foot_pos_exceeded=0.9365`；step `error_joint_pos_physical=0.2609`、`error_head_default_pos=0.0000`。
  - 当前结论：4096 env / 4 mini-batch 已能稳定启动并上传 SwanLab，训练链路和 checkpoint 链路打通；前几十轮仍主要表现为足端 termination 高，尚不能声称收敛。
- DDR smoke：
  - 命令见 `/tmp/h2lora_ddr_smoke_8env_1iter.log`。
  - `num_envs=8`、`num_steps_per_env=4`、`cache_all_motions_to_ddr=true`。
  - Motion DDR cache：`2/2 files, 0.00 GB`。
  - policy obs `(930,)`、critic obs `(1645,)`、action manager shape `31`。
  - 第 1 iteration：reward `0.10661`，mean length `2.00000`，`error_joint_pos=0.0550`，`error_body_pos=0.0458`，`foot_pos_xyz=0.0938`，total timesteps `32`。
- 默认 strict idle 小数据训练：
  - 最新 head-lock + 诊断指标版本日志见 `/tmp/h2lora_idle_strict_headlock_16env_50iter.log`。
  - `num_envs=16`、`num_steps_per_env=16`、`num_learning_iterations=50`、DDR cache 开启、`SWANLAB_MODE=disabled`。
  - 该 run 没有上传 SwanLab；`config.yaml` 中 `use_swanlab: false`，日志中未出现 `swanlab.init/log`，这是为了本地 debug 避免云端噪声。工程代码支持 SwanLab，但本报告这次 idle/head-lock 验证结果只在本地日志和 checkpoint 中。
  - head 冻结验证：所有已抽查 iteration 中 `Env/step/h2lora/error_head_default_pos=0.0000`。
  - 第 1 iteration：reward `-0.24733`，mean length `5.92105`，episode `error_joint_pos_physical=0.3266`，`error_joint_pos_semantic=0.3244`，`error_hip_pos_physical=0.2730`，`error_ankle_pos_physical=0.2107`，`foot_pos_xyz=0.4688`，`termination_foot_pos_exceeded=0.7236`。
  - 第 10 iteration：reward `-1.19006`，mean length `4.13000`，episode `error_joint_pos_physical=0.4705`，`error_joint_pos_semantic=0.4669`，`error_hip_pos_physical=0.2485`，`error_ankle_pos_physical=0.3057`，`foot_pos_xyz=0.8203`，`termination_foot_pos_exceeded=0.8656`。
  - 第 25 iteration：reward `-1.24246`，mean length `3.94000`，episode `error_joint_pos_physical=0.4768`，`error_joint_pos_semantic=0.4731`，`error_hip_pos_physical=0.2839`，`error_ankle_pos_physical=0.2943`，`foot_pos_xyz=0.8750`，`termination_foot_pos_exceeded=0.8672`。
  - 第 50 iteration：reward `-1.01048`，mean length `4.03000`，episode `error_joint_pos_physical=0.4254`，`error_joint_pos_semantic=0.4226`，`error_hip_pos_physical=0.2633`，`error_ankle_pos_physical=0.2585`，`foot_pos_xyz=0.7070`，`termination_foot_pos_exceeded=0.7208`。
  - 结论：默认 strict 配置能稳定训练且 head 已严格冻结，但没有在 12,800 timesteps 内收敛；episode 仍主要被足端位置 termination 截短。
- 诊断指标实现：
  - `gear_sonic/envs/manager_env/mdp/terminations.py` 在 `exceeded_body_pos()` 遇到左右脚踝 body 时记录 `termination_foot_pos_max/mean/exceeded`；`exceeded_body_height()` 遇到左右脚踝和双腕组合时记录 `termination_ee_body_height_*`。
  - `gear_sonic/envs/wrapper/manager_env_wrapper.py` 每 step 追加 `Env/step/Metrics/motion/*` 和 `Env/step/h2lora/*`，包括 physical/semantic joint error、hip/ankle error、feet body error、head default error。
  - 需要注意：strict 配置下，`Env/Metrics/motion/*` 来自 episode/termination 统计，更能反映失败帧；`Env/step/*` 是 wrapper 在 `env.step()` 返回后读取的当前状态，部分 env 可能已被自动 reset，因此适合做运行时 smoke 和 relaxed 诊断，但不能单独解释 strict termination rate。
- 诊断性 relaxed run：
  - 尝试关闭 early termination 和域随机化，只保留 timeout；日志见 `/tmp/h2lora_idle_overfit_relaxed_16env_40iter.log`。
  - 该配置导致 episode 汇总指标长期为 `reward=0`、`mean length=nan`、tracking metrics 为 0，说明当前 PPO 日志统计依赖 episode reset 汇总，不能用该 run 判断学习信号；进程已停止，不作为收敛证据。
- 本次 ref/tracking 可视化：
  - 输出目录：`visualizations/h2lora/h2lora_idle_strict_headlock_viz_20260601_000700`。
  - 视频：`ref_tracking.mp4`，160 frames，30 FPS，1280x720。
  - 轨迹：`trajectory_ref_tracking.npz`，checkpoint 为 `logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora-20260531_232745/last.pt`，motion key 为 `Idle_Left_001__A017`。
  - 为了生成连续视频，采集时 `disable_terminations=true`、`disable_events=true`；因此该视频用于观察 policy tracking 漂移，不用于替代 strict episode termination 统计。
  - 可视化指标：`mean_root_error_m=0.8105`、`max_root_error_m=0.9664`、`mean_joint_mae_rad=1.0488`、`max_joint_mae_rad=1.2222`、`done_count=0`。
  - head 可视化校验：ref/tracking 的 head 两维在整段轨迹中 `mean/max error = 0.0/0.0`，符合 H2 head policy 外冻结约束。
  - 首帧误差很小（root `0.0m`、joint MAE `0.0024rad`），末帧漂移到 root `0.9537m`、joint MAE `1.2222rad`；这与当前 idle strict 训练尚未收敛的结论一致。

已新增测试文件：

- `gear_sonic/tests/test_h2lora_alignment.py`
  - axis audit。
  - 验证 G1/H2 hip 映射等于论文式 `pinv` 投影组合；非 hip joints 保持 identity。
  - H2 body/head joint indices 检查。
  - full H2 31D -> headless 29D indices 检查。
- `gear_sonic/tests/test_lora_adaptation.py`
  - LoRALinear shape/base freeze。
  - `apply_sonic_h2lora()` 只让 LoRA 参数可训练，`std` 冻结，proprioception slice 被包裹。

## 8. 与论文不完全一致或论文未规定的点

1. **不是通用 Any2Any 自动映射框架**
   论文描述的是跨多个 humanoid 的 general paradigm；当前实现是 G1 SONIC release 到 Unitree H2LoRA 的特化实现。

2. **Parallel Joint Coupling 未实现**
   论文写了 `J_r` closed-chain correction；当前只做 ankle order 修正，没有 closed-chain Jacobian。原因是缺少可靠的 H2 并联机构映射数据，贸然实现会引入不可审计误差。

3. **LoRA alpha/dropout 是工程选择**
   当前可读论文正文强调 LoRA/rank 和 S7 scope，但没有给出本工程所需的 alpha/dropout 细节。实现采用 `alpha=rank=32`、`dropout=0`。

4. **H2 head direct command 是任务约束**
   论文没有针对 target 多余 head DoF 的直接默认指令方案；这是用户要求“不改模型、policy 只控制 29DoF body”的工程设计。当前实现不仅发送默认 head action target，还在 reset/step 后重写 head state，以满足“冻结默认姿态”的训练约束。

5. **MotionLib schema/cache 是工程扩展**
   论文说明 motion dataset/retargeting，但没有规定本工程的 PKL metadata 校验、DDR cache、worker 限制。

6. **SwanLab 是工程记录系统**
   论文无 SwanLab；这是用户要求的训练管理实现。

7. **capacity probe 不是论文实验协议**
   probe 是为单 RTX 5090 寻找可承受 env 数；论文实验使用更强多卡资源。

8. **尚未验证真实 sim 收敛**
   目前已完成 IsaacLab rollout、strict checkpoint load、LoRA 注入、head lock 验证、诊断指标和 idle 小数据训练测试；但默认 strict 配置在 16 env / 50 iteration / 12,800 timesteps 内没有收敛。因此不能把当前结果写成论文级复现成功，只能写成训练链路打通且暴露出足端/构型适配问题。

9. **Encoder sampling 回到 release mixed/legacy**
   曾经尝试过 SMPL-only 训练策略，但 schema6 复查后确认旧异常主要来自数据契约错误。当前默认恢复 release mixed encoder sampling，并关闭 `optimize_encoders_ratio_for_CHIP` 以保留 legacy multi-hot alignment 行为。这个选择是工程诊断后的策略调整，不是论文直接规定。

10. **seed 数据链路消费 paired H2/SMPL-X/BVH**
    H2 robot branch 仍使用 retarget 后的 `h2/csv`；SMPL encoder branch 现在必须通过 `smpl-x/uniform` 的 `source_bvh_relpath` 读取 `soma_uniform/bvh` 做 FK/name mapping。原因是本地 `smpl-x/uniform poses` 实测为 SOMA-X 77 关节顺序，不能直接作为 SONIC SMPL24 body pose。

11. **Hip alignment 不是无损互逆**
    论文的 `pinv` 形式本身是局部线性最小二乘投影；在当前 G1/H2 hip axis 上，target->source 与 source->target 的组合不是 identity。代码按论文公式实现，但这会给 source semantic action/reference 带来投影误差，需要在训练结果中单独观察 hip/feet 相关指标。

12. **Isaac/TRL 兼容 shim 是工程修复**
    `train_agent_trl.py` 增加了 TRL checkpoint symbol shim；`scripts/setup_h2lora_isaac_python.sh` 会给本地 IsaacLab URDF converter 加兼容 patch。这些都不是论文方法，而是当前本机依赖版本组合下复现训练所需的工程补丁。

13. **双机器人视频可视化是工程诊断工具**
    论文没有规定 MuJoCo 双机器人视频输出。本实现仿照部署侧 sim 可视化，把 ref/tracking 同场景叠加成两种颜色，目的是快速观察 H2LoRA tracking 漂移、head lock 和 joint/root error，而不是论文方法的一部分。

## 9. 当前最需要后续核对的问题

1. H2 ankle 是否存在必须处理的 parallel/closed-chain Jacobian；如果有，需要补 `J_r`，不能只靠顺序重排。
2. Hip `pinv` 投影的误差是否是当前 foot termination 的主要来源；需要按 joint/body 分解 tracking error，而不是只看全局 `error_joint_pos`。
3. `head_pitch_joint/head_yaw_joint` 已在当前仿真训练中锁定到 0/0，并由 `error_head_default_pos=0.0000` 验证；仍需和真实 H2 硬件默认姿态核对，如果硬件默认不同，应只改 `direct_joint_defaults`。
4. 默认 strict 训练下 idle 小数据没有收敛，是否需要给 H2LoRA 增加课程：先放宽 foot/ee/root termination 或降低 exploration std，再逐步恢复原 SONIC strict 配置。
5. DDR cache 44GiB 是否会与系统/IsaacSim 内存竞争；如果真实数据集很大，可能需要调低 `ddr_cache_max_bytes` 或分层 GPU working set。
6. SMPL 链路的主要错误已经修复：旧 schema2/3/4/5 都不能驱动 G1 release 的 SMPL locomotion，schema6 使用 SOMA/BVH FK name-map 后已经通过 G1 release + H2LoRA smoke。后续风险从“链路是否完全错”转为“schema6 是否覆盖所有动作类别、旧训练是否需要全部重建、官方 `data/smpl_filtered` 是否存在额外细节差异”。
7. 诊断性 relaxed run 关闭 early termination 后 episode 日志为 0/nan；当前已补 `Env/step/*` rollout-step tracking metrics，但 strict 配置下仍应优先看 episode/termination 统计，因为 step 指标可能在自动 reset 后读取。
8. 当前可视化 rollout 为连续展示而关闭 terminations/events；若后续要对比训练真实失败点，应使用 `--keep-terminations --keep-events` 另存一组短视频。

## 10. 2026-06-01 3M 50Hz 续训实验状态更新

本节记录当前代码实际执行状态，避免后续把旧的 30Hz/1M idle 结论误读为本轮 3M 50Hz 训练结论。

### 10.1 数据链路修正

- `gear_sonic/data_process/build_h2lora_seed_motion_lib.py`
  - `downsample_array()` 已修正为时间轴重采样：当 `fps_source % fps_target != 0` 时使用线性插值，而不是旧的 `int(fps_source / fps_target)` stride。原因是 `120Hz -> 50Hz` 不能用 stride=2，否则实际数据是 60Hz 却在 MotionLib 中标成 50Hz。
  - 新增 `resample_quat_xyzw()`，root quaternion 在非整数频率转换时使用 `scipy.spatial.transform.Slerp` 并归一化。原因是 quaternion 不能直接线性插值，否则 root rotation 可能不是单位四元数。
- 新构建的数据集：
  - 输出目录：`/mnt/data/seed/h2lora_motion_lib/idle_3m_50hz`
  - robot motion：`/mnt/data/seed/h2lora_motion_lib/idle_3m_50hz/robot`
  - SMPL motion：`/mnt/data/seed/h2lora_motion_lib/idle_3m_50hz/smpl`
  - `num_motions=5907`，`fps=50`，总 robot 帧数 `3,002,741`。
  - 完整性检查结果：robot/smpl PKL 均为 5907 个；robot dof shape 均为 31D；抽查 root quaternion norm 无异常；抽查 robot/smpl frame count 对齐。

### 10.2 NaN 防线修正

- `gear_sonic/trl/modules/actor_critic_modules.py`
  - `Actor.forward()` 对 action mean 做 `nan_to_num` 和 `action_mean_clip`。
  - `Actor.update_distribution()` 对 `std` 和 `mean` 做 finite 检查、clamp，并使用 `Normal(mean, std.expand_as(mean))`，避免旧逻辑里 `(mean * 0 + std)` 被 mean NaN 污染。
  - `Actor.get_std` 对原始 `std` parameterization 增加 in-place finite 修复，避免 checkpoint 或 optimizer step 后出现非正/非 finite std。
- `gear_sonic/trl/trainer/ppo_trainer.py`
  - PPO loss 对 `new_logprobs/sigma/mu/entropy/value/old_mu/old_sigma/returns/advantages` 做 finite gate；非 finite microbatch 会跳过 optimizer step。
  - `logprobs_diff` clamp 到 `[-20, 20]` 后再 `exp`，降低 ratio overflow 风险。
  - `_compute_returns()` 对 returns/advantages 做 `nan_to_num`，并在 advantage normalization 后再次清理。
  - `_gradient_clipping()` 现在除了逐参数 grad NaN/Inf 外，还检查 clip 后全局 grad norm 是否 finite；若不是 finite，跳过 optimizer step。
- `gear_sonic/utils/running_mean_std.py`
  - running mean/var 同步和 forward 中增加 finite/正数保护；遇到非 finite batch stats 时不更新 running stats。

这些修正是工程稳定性补丁，不属于论文 Any2Any 方法本身；目的是保证“坏 batch 跳过，不污染 LoRA checkpoint”。

### 10.3 抗身体晃动与诊断指标

- 新增 reward term：
  - `gear_sonic/config/manager_env/rewards/terms/body_ang_vel_l2.yaml`
  - `gear_sonic.envs.manager_env.mdp.rewards.body_ang_vel_l2()`
  - 默认挂到 `base_5point_local_feet_acc.yaml`，H2LoRA 当前训练中覆盖为 `-2.0e-4`。
- 当前 3M 续训启动时的晃动相关权重：
  - `anti_shake_ang_vel.weight=-1.5e-2`
  - `body_ang_vel_l2.weight=-2.0e-4`
  - `feet_acc.weight=-3.5e-6`
  - `max_grad_norm=0.03`
  - `std_clamp_max=0.38`
  - `action_mean_clip=12.0`
- `ManagerEnvWrapper._append_h2lora_step_diagnostics()` 增加 step-level 指标：
  - `step/h2lora/policy_action_rate_l2`
  - `step/h2lora/policy_action_abs_mean`
  - `step/h2lora/robot_joint_vel_abs_mean`
  - hip/ankle/joint/feet/root/body angular velocity tracking 相关指标。
  - 2026-06-02 修正：之前这里对 `command.body_indexes` 直接调用 `.index()`，而运行时它是 `torch.Tensor`，会触发 `index() missing required argument 'indices'` 并跳过整段 body angular velocity 诊断。当前已先转为 Python list，并额外记录 `root/torso/wrist/ankle` 的 `ref_ang_vel_norm`，用于直接区分“reference 本身在抖”还是“robot tracking 在抖”。

这些指标是工程诊断扩展，论文没有规定；原因是用户关心“身体频繁晃动”，只看 reward 总值无法区分是 action 抖、关节速度、root/torso angular velocity 还是 foot termination 主导。

### 10.4 SwanLab/GPU 记录

- `gear_sonic/trl/callbacks/swanlab_callback.py`
  - SwanLab callback 现在能记录 torch/numpy scalar，不再只记录 Python `int/float`。
- `gear_sonic/trl/trainer/ppo_trainer.py`
  - 每个 PPO iteration 增加 `nvidia-smi` 采样指标：`gpu/utilization_pct`、`gpu/memory_utilization_pct`、`gpu/nvidia_smi_memory_used_gb`、`gpu/power_w`、`gpu/power_limit_w`、`gpu/temperature_c`。
- 外部监控：
  - `scripts/run_h2lora_idle3m_resume10k.sh` 会生成同名 `*_gpu.csv`，每 5 秒记录显存、利用率、功耗、温度。

### 10.5 当前 10K 续训 run

- 启动脚本：`scripts/run_h2lora_idle3m_resume10k.sh`
- 使用 checkpoint：
  - `logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_idle1m_100k_4096env-20260601_010528/model_step_008000.pt`
  - 已检查 `policy_state_dict/value_state_dict/optimizer_state_dict` 均无 non-finite tensor，`global_step=8000`。
  - 没有使用该 run 的 `last.pt`，因为旧 `last.pt` 已知 value running stats 曾被 NaN 污染。
- 启动参数：
  - `num_envs=6144`
  - `num_steps_per_env=24`
  - `num_mini_batches=4`
  - `num_learning_epochs=5`
  - `num_learning_iterations=10000`
  - `cache_all_motions_to_ddr=true`
  - `max_cpu_workers=16`
  - `ddr_cache_max_bytes=49392123904`
  - `resume=false`：只加载权重，不恢复旧 optimizer/trainer/env state。
- 训练日志：
  - `/tmp/h2lora_longtrain/h2lora_idle3m50hz_resume8k_10k_6144env_20260601_161807.log`
- GPU 采样：
  - `/tmp/h2lora_longtrain/h2lora_idle3m50hz_resume8k_10k_6144env_20260601_161807_gpu.csv`
- 输出目录：
  - `logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_idle3m50hz_resume8k_10k_6144env-20260601_161809`
- SwanLab run：
  - `https://swanlab.cn/@sky931/sonic-h2lora/runs/dfl3b7g05195g0obdr1sv`

截至 2026-06-01 16:30，训练已跑到至少 iteration 153，未出现 `NaN/Inf/Invalid PPO/CUDA out of memory/Traceback`。最新日志显示 total timesteps `22,560,768`，iteration time 约 `4.34s`，吞吐约 `34k steps/s`；GPU 采样中利用率多次达到 `90%+`，功耗采样峰值约 `490W`，显存峰值约 `20GB / 32.6GB`。这说明 6144 env 在当前 5090 上比旧 4096 env 更接近资源上限，同时仍保留了显存余量。

当前早期训练指标不能视为收敛结论，只能说明链路稳定启动：

- iteration 9：`Mean rewards=8.143`，`error_body_ang_vel=1.741`，`error_joint_pos_physical=0.165`，`foot_pos_xyz termination=0.466`。
- iteration 38：`Mean rewards=6.906`，`error_body_ang_vel=2.051`，`error_joint_pos_physical=0.172`，`foot_pos_xyz termination=0.622`。
- iteration 153：`Mean rewards=6.104`，仍处于早期波动区间；step-level 指标为 `error_joint_pos_physical=0.1585`、`error_hip_pos_physical=0.1191`、`error_ankle_pos_physical=0.0876`、`policy_action_rate_l2=0.5528`、`policy_action_abs_mean=1.4678`、`robot_joint_vel_abs_mean=0.9554`；episode-level `error_body_ang_vel=2.171`、`termination_foot_pos_exceeded=0.679`。

需要继续等到 1000/5000/10000 checkpoint 后，再做同一 ref motion 的视频对比和阶段性指标分析。

### 10.6 当前 last.pt 可视化晃动原因分析

本节对应 2026-06-01 中断后的当前 checkpoint 可视化：

- checkpoint：`logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_idle3m50hz_resume8k_10k_6144env-20260601_161809/last.pt`
- motion：`Idle_Left_001__A017`
- 输出目录：`visualizations/h2lora/h2lora_idle3m_lastpt_20260601_233836`
- 视频：`ref_tracking.mp4`
- 轨迹：`trajectory_ref_tracking.npz`
- 帧数/频率：`3827` frames，`50 FPS`，约 `76.54s`

首先核对 H2 DoF 顺序：运行时 `robot.joint_names` 与 `H2LORA_FULL_H2_ISAACLAB_DOF_INDICES` 一致，body 29D 选出来的顺序正好是 `H2LORA_BODY_JOINTS_IN_G1_ORDER`；`head_pitch/head_yaw` 在 runtime indices `[11, 16]`，被排除在 policy 29D 外。因此这次上半身晃动不应归因于 H2LoRA body DoF 顺序错位。可视化脚本也已补充保存 `ref_dof_isaac/tracking_dof_isaac/robot_joint_names/body_joint_indices`，后续视频可以直接审计 IsaacLab 与 MuJoCo 两侧的 joint order。

对同一条 `trajectory_ref_tracking.npz` 做 FK/速度统计后，reference 本身非常静，而 tracking 产生了明显高频动态：

| 指标 | reference | tracking | 结论 |
| --- | ---: | ---: | --- |
| root linear velocity RMS | `0.0073 m/s` | `0.0773 m/s` | tracking 约 10.6 倍 |
| root angular velocity RMS | `0.0193 rad/s` | `0.2988 rad/s` | tracking 约 15.5 倍 |
| torso angular velocity RMS | `0.0272 rad/s` | `0.3619 rad/s` | tracking 约 13.3 倍 |
| left wrist angular velocity RMS | `0.1000 rad/s` | `0.5043 rad/s` | tracking 约 5.0 倍 |
| right wrist angular velocity RMS | `0.0548 rad/s` | `0.5318 rad/s` | tracking 约 9.7 倍 |
| left ankle angular velocity RMS | `0.0182 rad/s` | `0.4454 rad/s` | tracking 约 24.4 倍 |
| right ankle angular velocity RMS | `0.0364 rad/s` | `0.3819 rad/s` | tracking 约 10.5 倍 |
| torso position error mean/max | - | `0.0724m / 0.1208m` | 上身跟踪仍有持续误差 |

关节速度/加速度也显示同一趋势：

| joint group | ref vel RMS | tracking vel RMS | ref acc RMS | tracking acc RMS |
| --- | ---: | ---: | ---: | ---: |
| waist | `0.0138` | `0.2245` | `0.5138` | `9.7637` |
| arms | `0.0265` | `0.2691` | `0.8827` | `9.3417` |
| ankles | `0.0200` | `0.3325` | `1.5021` | `16.4844` |
| all 31 DoF | `0.0255` | `0.2669` | `1.5142` | `10.6276` |

所以当前判断是：**至少在这条 idle motion 上，晃动主因不是 H2 reference 数据本身“很差/很抖”。更可能是未收敛的 policy 在 H2 动力学、足端约束、上身 tracking reward 和动作平滑之间找到了高频补偿策略。** 这和 3M 训练日志里 `error_body_ang_vel` 持续较高、`foot_pos_xyz termination` 高、`policy_action_rate_l2` 和 `robot_joint_vel_abs_mean` 偏高的现象一致。

当前 reward 侧也解释了为什么这种策略能存活一段时间：

- `anti_shake_ang_vel` 当前有 `threshold=1.0 rad/s` 死区，而可视化里 torso RMS 约 `0.36 rad/s`，大量中等幅度晃动不会被该项惩罚。
- `body_ang_vel_l2` 只盯 `torso_link`，权重为 `-2.0e-4`，相对 tracking 正奖励较弱。
- `tracking_vr_5point_local` 权重为 `2.0`，会鼓励 torso/wrist 点贴近 ref，但不直接约束上身角速度平滑。
- `foot_pos_xyz` early termination 很高，policy 很可能优先用髋/踝/腰的快速补偿争取站住和贴脚，副作用就是上半身连带摇摆。

因此下一轮不应先重做 retarget 数据，而应优先做更可控的 ablation：

1. 保持同一条 `Idle_Left_001__A017` 和同一 checkpoint，开启新增 `ref_ang_vel_norm` 指标，确认 SwanLab 中 ref/robot 角速度差距。
2. 跑一个短 curriculum run：保留 tracking reward，临时放宽 foot/ee early termination 或延后启用 adaptive sampling，让 episode 不被过早截短。
3. 把 `anti_shake_ang_vel.threshold` 从 `1.0` 降到 `0.25~0.4`，并把 body list 扩到 `torso_link/left_wrist_yaw_link/right_wrist_yaw_link`；同时适度提高 `body_ang_vel_l2` 到 `-1e-3` 量级做对照。
4. 降低 `std_clamp_max` 或提高 action-rate/关节速度平滑惩罚，观察 `policy_action_rate_l2`、`robot_joint_vel_abs_mean`、`root/torso/wrist robot_ang_vel_norm` 是否下降，且 `error_joint_pos_physical` 不明显恶化。

### 10.7 5M seed 随机完整片段 LoRA run

本节对应 2026-06-02 用户要求的“从完整 seed 数据集中均匀随机采样约 5M 帧，并从 Sonic release 重新训练 LoRA”的实验。

重要更正：本节记录的是 schema6 修复前的历史实验。当时 SMPL 分支仍使用旧 `smplx-fk`/错误 SMPL24 假设，因此本节中关于 SMPL encoder tracking 异常、原地站立、倒地等结论只能作为旧数据链路失败诊断，不能代表当前 schema6 修复后的训练效果。

数据集构造脚本为 `gear_sonic/data_process/build_h2lora_seed_motion_lib.py`。本次新增并使用：

- `--selection-mode random`
- `--target-frames 5000000`
- `--random-seed 20260602`

实现语义是先在完整 paired seed motion 池上做 motion 级随机 shuffle，然后按完整 motion 累加到目标帧数。也就是说这里不是截断 motion 片段，不是只采某些类别，而是从完整 seed 成对数据中随机选择完整 motion；最终帧数允许略超过目标值。

本次构造命令的关键参数：

- seed root：`/mnt/data/seed`
- 输出：`/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz`
- 原始频率：`120Hz`
- 训练频率：`50Hz`
- worker：`16`
- SMPL 分支：`smplx-fk`。这是 schema6 修复前的旧设置；当前全量训练应改用默认 `soma-bvh`。

落盘统计：

- 完整 paired motion pool：`142220`
- 选中 motion 数：`13889`
- 目标帧数：`5000000`
- 实际/估算选中帧数：`5000227`
- robot pkl 数：`13889`
- smpl pkl 数：`13889`
- 磁盘占用：约 `3.5GB`
- 全量 robot pkl 读取校验：`31DoF`、`50Hz`、总帧数 `5000227`，bad count `0`
- SMPL pkl 抽样校验：`128` 个样本，bad count `0`

类别 motion 数前十为：

| 类别 | motion 数 |
| --- | ---: |
| jog | `1309` |
| walk | `1053` |
| dance | `904` |
| injured | `861` |
| jump | `763` |
| idle | `447` |
| turn | `442` |
| medium | `432` |
| itching | `341` |
| dancing | `335` |

训练脚本为 `scripts/run_h2lora_seed5m_from_release.sh`。注意这个脚本没有传入 `checkpoint=...`，而是依赖配置中的：

- `checkpoint: null`
- `resume: false`
- `algo.config.pretrained_model.path: sonic_release/last.pt`
- `algo.config.pretrained_model.state_dict_key: checkpoint`

因此本次训练是从 Sonic release 重新 strict load，再应用 H2LoRA；不是从 3M/idle 旧 H2LoRA checkpoint 续训。这里的“LoRA 全量微调”指 H2LoRA 配置中定义的 LoRA target modules 全部参与训练，仍然不是全参数微调。

训练并行度沿用上一轮高效配置：

- `num_envs=6144`
- `num_steps_per_env=24`
- `num_mini_batches=4`
- `num_learning_epochs=5`
- `num_learning_iterations=10000`
- `actor_learning_rate=2.5e-6`
- `critic_learning_rate=5.0e-6`
- `max_grad_norm=0.03`
- `std_clamp_max=0.38`
- `action_mean_clip=12.0`
- `cache_all_motions_to_ddr=true`
- `max_cpu_workers=16`
- `ddr_cache_max_bytes=49392123904`

本次 run 信息：

- train pid：`275393`
- Python pid：`275403`
- train log：`/tmp/h2lora_longtrain/h2lora_seed5m50hz_release_lora_10k_6144env_20260602_013259.log`
- GPU log：`/tmp/h2lora_longtrain/h2lora_seed5m50hz_release_lora_10k_6144env_20260602_013259_gpu.csv`
- 输出目录：`logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_seed5m50hz_release_lora_10k_6144env-20260602_013303`
- SwanLab run：`https://swanlab.cn/@sky931/sonic-h2lora/runs/c5phhn70u8gu7qz1iuc7e`

初始化和 LoRA 检查：

- H2LoRA bridge 日志：`policy_action_dim=29, sim_joints=31, head_defaults=[0.0, 0.0]`
- LoRA 日志：`rank=32, alpha=32.0, dropout=0.0`
- trainable params：`866272 / 38240091 = 2.265%`
- actor trainable：`489440`
- critic trainable：`376832`
- `error_head_default_pos` 在早期训练中保持 `0.0000`
- 日志中未发现 `Traceback/RuntimeError/NaN/Inf`

截至 2026-06-02 01:43 左右，训练已跑到至少 iteration `109`，`last.pt` 已正常写出。吞吐和资源情况：

- iteration `99`：约 `32395 steps/s`
- iteration `109`：约 `30705 steps/s`
- GPU 采样全局峰值：util `91%`，显存 `19389MB`，功耗 `503.71W`
- 最近 60 条 GPU 采样均值：util `63.12%`，显存 `14020.53MB`，功耗 `277.27W`

早期训练趋势只能视为“链路健康并出现初步下降”，不能视为收敛：

| 指标 | 初始附近 | iteration 99/109 附近 | 趋势 |
| --- | ---: | ---: | --- |
| `Env/step/h2lora/error_joint_pos_physical` | `0.1698` 左右 | `0.1511~0.1526` | 下降 |
| `Env/step/h2lora/error_joint_vel_physical` | `3.13` 左右 | `2.96~2.98` | 下降 |
| `Env/step/h2lora/policy_action_abs_mean` | `2.27` 左右 | `2.11~2.13` | 下降 |
| `Env/step/h2lora/policy_action_rate_l2` | `1.14` 左右 | `1.02~1.03` | 下降 |
| `root_robot_ang_vel_norm` | `3.30` 左右 | `2.96~2.99` | 下降 |
| `torso_link/robot_ang_vel_norm` | `3.55` 左右 | `3.36` 左右 | 小幅下降 |
| `left_wrist_yaw_link/robot_ang_vel_norm` | `9.8` 左右 | `8.7` 左右 | 下降 |
| `right_wrist_yaw_link/robot_ang_vel_norm` | `8.0` 左右 | `7.1` 左右 | 下降 |
| `left_ankle_roll_link/robot_ang_vel_norm` | `6.3` 左右 | `5.4` 左右 | 下降 |
| `right_ankle_roll_link/robot_ang_vel_norm` | `6.6` 左右 | `6.7` 左右 | 未明显下降 |

同时仍有明显问题：

- episode-level `foot_pos_xyz` termination 仍在 `0.88~0.89`，说明大量 episode 仍因足端跟踪/接触相关条件失败。
- `Mean length` 仍只有 `4~5` 左右，episode 偏短。
- `error_body_ang_vel` 仍约 `6.2`，上身/身体角速度问题没有解决。
- `Mean rewards` 早期波动很大，不应用作收敛结论。

因此本次实验当前结论是：5M 数据链路、SwanLab、release strict load、LoRA-only 参数冻结、6144 env 并行度和 checkpoint 写出均已验证；早期指标比初始状态有下降，但身体晃动和 foot termination 仍是主要风险，需要等 1000/5000/10000 checkpoint 后再做同一 ref motion 的可视化对比。

## 11. 建议的下一步验证顺序

1. 用 `scripts/setup_h2lora_uv.sh` 建数据/测试环境；用 `scripts/setup_h2lora_isaac_python.sh` 建 Isaac Sim 训练环境。
2. 做一个 H2LoRA curriculum 配置：先降低/关闭 foot_pos_xyz、ee_body_pos、anchor_ori_full early termination，或使用更短固定片段 timeout 产生有效 episode summary，再逐步恢复 strict 配置。
3. 对比三组 idle 训练：base frozen/no LoRA、LoRA strict、LoRA curriculum；如果 curriculum 仍不降 error，优先检查数据坐标、SMPL token、hip 投影和 ankle coupling。
4. 每个关键 checkpoint 用 `scripts/visualize_h2lora_ref_tracking.py` 生成同名 `实验名称_时间` 视频，并对比 root/joint MAE、head error 和肉眼漂移。
5. 在 idle 能稳定下降后再跑 128/1024 env 2 iteration，确认 SwanLab logs、GPU memory、cache stats。
6. 最后运行 `scripts/h2lora_capacity_probe.sh`，选择 32GB 显存下稳定的最大 `num_envs`。

## 12. 2026-06-03 Locomotion 失效诊断

本节不是论文中的固定实现细节，而是针对 5M run 可视化中“多数动作站在原地、上半身晃动”的排障实验。目标是把问题拆成四个部分：数据分布、resume 权重来源、G1 token/reference 是否退化、H2 action adapter/动力学是否能复现 locomotion。

首先复查 encoder 结论：不能把主因简单归结为 `SMPL-only encoder 输入不足以表达 locomotion/root displacement`。H2LoRA 配置中的 SMPL 分支输入为 `smpl_joints_multi_future_local_nonflat`、`smpl_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_smpl`，G1 分支输入为 `command_multi_future_nonflat`、`motion_anchor_ori_b_mf_nonflat`，二者都进入 universal-token encoder。当前更值得注意的实现差异是 `optimize_encoders_ratio_for_CHIP: true`：在 `gear_sonic/envs/manager_env/mdp/commands.py` 中，它会跳过原始 legacy 行为里“SMPL-native episode 同时激活 G1 encoder”的 OR 逻辑。这可能改变 SMPL/G1 latent 对齐方式，但不是本节诊断中已经证明的主因。

5M 数据分布复查结果如下，统计对象为 `/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz/robot`：

| 统计项 | 数值 |
| --- | ---: |
| motion 数 | `13889` |
| 总帧数 | `5000227` |
| 时长（50Hz） | `27.78h` |
| locomotion 关键词帧占比 | `37.14%` |
| static/idle 关键词帧占比 | `8.24%` |
| root xy 位移 `<0.20m` 帧占比 | `52.46%` |
| root xy 位移 `>=2.0m` 帧占比 | `30.49%` |

因此 5M 数据不是 idle-only，也不是只有站立；但随机完整采样里小位移/近静态 motion 占比确实偏高，足以强化站立局部最优。

resume 权重来源复查结论：本次 5M run 不是从旧 idle checkpoint 继续。初始启动脚本 `/tmp/h2lora_longtrain/h2lora_seed5m50hz_release_lora_10k_6144env_20260602_013259_train_cmd.sh` 中 `checkpoint` 为空、`resume=false`，只通过配置 strict load `sonic_release/last.pt`。后续 resume 脚本使用的是同一个 5M run 目录下的 `last.pt`，不是 idle run。LoRA 权重距离也支持这个结论：5M step1000 与 5M step10000 的距离明显不同于 idle step8000/idle last 的距离。

为隔离 H2 侧问题，新增了诊断脚本 `scripts/visualize_g1_release_on_h2adapted_ref.py`。该脚本把 H2 MotionLib 中的 31DoF H2 motion 截取 headless 29DoF 后，通过 `h2lora_alignment.h2_to_g1_semantic` 转成 G1 semantic joint ref，再在 G1 环境中加载 `sonic_release/last.pt` 做 rollout 和双机器人可视化。对 `walk_forward_loop_002__A036` 强制使用 G1 encoder 的结果为：

| 实验 | ref xy | tracking xy | mean root err | joint MAE | action abs mean | encoder |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| G1 release on H2->G1 ref | `8.188m` | `5.618m` | `1.247m` | `0.094rad` | `0.738` | `[1,0,0]` |

视频路径：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/g1_release_h2adapted/g1_release_h2adapted_walk_g1encoder_probe_20260603_131840/g1_release_h2adapted_ref_tracking.mp4`。

该结果说明：同一条 H2-derived reference 转到 G1 semantic 后并不是静态 token，release G1 policy 也会输出有效 locomotion action。它跟踪不完美，但足以排除“reference/token 完全退化导致只能站立”的解释。

随后扩展 `scripts/visualize_h2lora_ref_tracking.py`，新增 `--action-source`：

- `policy`：原始 H2LoRA checkpoint policy。
- `pd-ref`：不加载 policy，直接把当前 MotionLib 的 H2 physical joint target 转成 IsaacLab `JointPositionAction` 的 sim action。
- `external-g1-actions`：读取另一个 rollout 的 `policy_action_used` 29D G1 action，经当前 H2LoRA bridge 映射到 H2 31D sim action。

对同一条 `walk_forward_loop_002__A036` 的结果如下：

| 实验 | ref xy | tracking xy | mean root err | joint MAE | sim action abs mean | reward mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2LoRA 10k policy mean | `5.775m` | `0.711m` | `2.271m` | `0.230rad` | `0.482` | `0.053` |
| H2 direct PD ref | `8.188m` | `1.077m` | `3.090m` | `0.121rad` | `0.181` | `0.006` |
| G1 release action -> H2 bridge | `8.188m` | `1.244m` | `3.025m` | `0.146rad` | `0.228` | `0.009` |

视频路径：

- H2 direct PD ref：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2_pd_ref_walk_probe_20260603_140815/ref_tracking.mp4`
- G1 release action -> H2 bridge：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2_g1release_actions_bridge_walk_probe_20260603_140856/ref_tracking.mp4`
- H2LoRA policy 对照：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2lora_debug_action_mode_20260603_124816/walk_mean/ref_tracking.mp4`

这个结果把问题定位得更窄：H2LoRA policy 不是唯一原因。即使不经过 policy、直接用 H2 reference joint target 做 PD replay，H2 也只产生约 `1.08m` 位移；即使把 release G1 在同一 converted ref 上输出的 locomotion action 通过当前 H2 action adapter 打到 H2，也只产生约 `1.24m` 位移。相比之下，G1 本体上同一 converted ref 可以走约 `5.62m`。

需要注意，PD replay 不是最终 policy 的等价替代，因为 root 不能被直接驱动，纯关节目标未必能稳定复现所有 mocap locomotion。但 walk ref 在 H2 上几乎无法产生对应前进位移，说明当前 H2 retarget/reference、H2 PD/dynamics、H2/G1 action adapter 至少有一个环节不能支持 locomotion。结合 G1 对照，当前更可能的问题优先级为：

1. H2 retargeted joint trajectory 的动态可行性不足，尤其是足端接触、root 速度与 ankle/hip 配合。
2. H2 PD/action scale 与 reference 幅度不匹配，导致关节角 tracking 尚可但无法产生推进。
3. G1->H2 action adapter 在 locomotion 的 hip/ankle/waist 方向和尺度上不够等价。
4. 5M 数据小位移占比偏高，训练目标容易接受站立局部最优；但这不是唯一原因。
5. `optimize_encoders_ratio_for_CHIP=True` 可能影响 SMPL/G1 token 对齐，建议做 ablation。

下一步优先实验建议：

1. 做 H2 reference replay 细分：关节 PD target、root 固定/非固定、只下肢、只 hip/ankle，并输出 foot contact、root velocity、CoM/feet 相对位姿。
2. 逐关节检查 H2 action scale 和 soft limit，重点看 ankle pitch/roll、hip pitch/roll/yaw、waist yaw/pitch/roll。
3. 针对 locomotion motion 构造小规模 balanced subset，临时提高 locomotion/root velocity reward 或降低 static 数据比例，验证是否能突破站立局部最优。
4. 对 `optimize_encoders_ratio_for_CHIP=false` 做短训 ablation，使 SMPL-native episode 恢复 release 中的 G1 encoder OR 行为。
5. 如果 reference replay 仍无法走，应优先修 retargeting/PD/reference 可行性，而不是继续加长同一训练。

## 13. 2026-06-03 Tiny Locomotion Mixed-Encoder Overfit

本节是上一节之后的修正实验。用户指出直接 PD replay 和 open-loop G1 action bridge 都不能证明 H2 locomotion reference 不可执行；同时，既然 H2 reference 经 adapter 转成 G1 reference 后能被 G1 release policy 执行，就没有必要坚持只用 SMPL 分支。基于这个判断，本实验恢复 release 风格的 encoder 采样比例，并用极少量 locomotion 数据做 overfit 验证。

代码变更：

- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml`
  - `encoder_sample_probs.g1: 1.0`
  - `encoder_sample_probs.teleop: 1.0`
  - `encoder_sample_probs.smpl: 1.0`
  - `teleop_sample_prob_when_smpl: 0.5`
  - `optimize_encoders_ratio_for_CHIP` 仍保持 `true`，本轮不做该 ablation。
- `scripts/run_h2lora_locomotion_tiny_mixed_encoder.sh`
  - 新增 tiny locomotion overfit 启动脚本。
- `scripts/visualize_h2lora_ref_tracking.py`
  - 新增 `--use-encoder g1|teleop|smpl`，用于同一 checkpoint 强制指定 encoder 分支做可视化诊断。

小数据集为 `/mnt/data/seed/h2lora_motion_lib/locomotion_tiny_mixed_encoder_50hz`，从已构造的 5M MotionLib 中复制 paired robot/smpl pkl，不重新 retarget。数据规模：

| motion key | frames | root xy |
| --- | ---: | ---: |
| `walk_forward_loop_002__A036` | `450` | `9.06m` |
| `walk_forward_loop_003__A032_M` | `483` | `9.82m` |
| `jog_forward_loop_003__A029_M` | `442` | `11.68m` |
| `jog_forward_loop_001__A029_M` | `468` | `11.18m` |
| `walk_ff_loop_360_002__A094` | `379` | `6.92m` |
| `jog_ff_loop_225_001__A093` | `198` | `7.87m` |
| `walk_ff_loop_270_002__A093` | `406` | `8.08m` |
| `jog_ff_loop_360_002__A095` | `125` | `5.26m` |

总计 `8` 条 motion、`2951` 帧、`50Hz`。训练脚本：

`scripts/run_h2lora_locomotion_tiny_mixed_encoder.sh`

训练配置：

- `num_envs=2048`
- `num_steps_per_env=24`
- `num_mini_batches=4`
- `num_learning_epochs=5`
- `actor_lr=5e-6`
- `critic_lr=1e-5`
- `save_frequency=500`
- `pretrained=sonic_release/last.pt`
- `resume=false`
- SwanLab run: `https://swanlab.cn/@sky931/sonic-h2lora/runs/fxl041hr9kc2abr3cx03f`

训练到 `model_step_001000.pt` 后手动停止，保留 `model_step_000500.pt`、`model_step_001000.pt` 和 `last.pt`。停止时日志中存在一个 Isaac `carb.tasking.Mutex` assertion，这是手动停止/Isaac 退出阶段产生的错误，不影响已经保存的 checkpoint 与可视化轨迹。停止后确认无训练/可视化进程残留。

训练趋势非常明确：

| 指标 | 初期附近 | step 1000 附近 |
| --- | ---: | ---: |
| `adp_samp/failure_rate_mean` | `9.7` 左右 | `2.64` 左右 |
| `Mean rewards` | 负值/接近 `-0.2` | `>1.0` |
| `Mean length` | `4~5` | `17+` |
| `error_joint_pos_physical` | `0.29` 左右 | `0.16~0.17` |
| `error_joint_vel_physical` | `4.2` 左右 | `1.5~2.1` |
| `root_error_ang_vel_norm` | `2.9` 左右 | `1.6` 左右 |
| `foot_pos_xyz` episode termination | `0.76` 左右 | `0.60` 左右 |

最关键的是可视化结果。用 `walk_forward_loop_002__A036`，rollout 340 帧，disable terminations/events，统计如下：

| checkpoint / encoder | ref xy | tracking xy | mean root err | joint MAE | reward mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5M step10000, SMPL-only | `5.775m` | `0.711m` | `2.271m` | `0.230rad` | `0.053` |
| tiny step500, sampled teleop | `8.188m` | `6.400m` | `0.822m` | `0.119rad` | `0.098` |
| tiny step1000, forced G1 | `8.188m` | `6.670m` | `0.741m` | `0.141rad` | `0.103` |
| tiny step1000, forced teleop | `8.188m` | `6.571m` | `0.806m` | `0.134rad` | `0.102` |
| tiny step1000, forced SMPL | `8.188m` | `1.223m` | `3.620m` | `0.443rad` | `-0.037` |

视频路径：

- step500 sampled teleop：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2lora_locomotion_tiny_mixed_step500_walk_20260603_145422/ref_tracking.mp4`
- step1000 forced G1：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2lora_locomotion_tiny_mixed_step1000_g1_walk_20260603_151442/ref_tracking.mp4`
- step1000 forced teleop：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2lora_locomotion_tiny_mixed_step1000_teleop_walk_20260603_151505/ref_tracking.mp4`
- step1000 forced SMPL：`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora/h2lora_locomotion_tiny_mixed_step1000_smpl_walk_20260603_151528/ref_tracking.mp4`

本实验给出的结论比上一节更强：

1. H2LoRA 不是结构性地只能站立；少量 locomotion 数据 + release 比例 encoder 采样下，G1/teleop 分支可以学出明显前进。
2. 之前 5M 训练“几乎所有动作站立”的重要原因之一是 SMPL-only 训练/评估。强制 SMPL 分支在 tiny step1000 仍只走 `1.22m`，而同 checkpoint 的 G1/teleop 分支可以走 `6.5m+`。
3. 这也修正了“SMPL 输入不足以表达 locomotion”的表述：SMPL 输入本身包含 root/pose 信息，但在当前 frozen encoder + FSQ + H2LoRA LoRA-only 适配设置下，SMPL token 到 H2 locomotion action 的路径没有学好；相比之下，G1/teleop token 与 release policy 的原始动作先验更匹配。
4. 数据分布和 termination 仍然重要，但不是唯一主因。tiny locomotion 数据消除了大部分 static/idle 干扰后，G1/teleop 能走，而 SMPL 仍失败，说明 encoder 分支差异是当前必须正视的问题。

后续建议：

1. 全量/大数据训练不要再使用 SMPL-only；默认采用 release 比例 `g1=1, teleop=1, smpl=1`。
2. 如果目标最终需要 SMPL-only 输入，应该单独设计 SMPL 分支适配策略，例如提高 SMPL 分支 LoRA/adapter 容量、解冻或微调 SMPL encoder 的小模块、加入 G1-SMPL token alignment loss、或先用 G1/teleop 分支学稳定 H2 locomotion 后再蒸馏到 SMPL 分支。
3. 下一轮全量训练建议增加按 encoder 分支拆分的 eval：每个 checkpoint 固定同一 motion，分别强制 g1/teleop/smpl，避免 aggregate 指标掩盖 SMPL 分支 collapse。

## 14. 2026-06-03 Balanced 5M Mixed-Encoder 50k Fresh Run

本节对应用户要求：在确认 tiny locomotion mixed-encoder 能学会前进后，恢复原始 SONIC release 风格的 encoder 采样比例，在 5M 规模数据上从 `sonic_release/last.pt` 重新开始 LoRA 训练，最大训练规模改为 `50k` iteration，并每隔 `2.5k` checkpoint 自动导出典型动作可视化视频。

### 14.1 Balanced 5M 数据集

原始 `/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz` 是从完整 SEED 中均匀随机抽取全 clip 得到的 5M 级 MotionLib，但上一轮分析显示小位移/root displacement 很小的片段占比偏高，容易与 H2LoRA 的站立局部最优叠加。为了在不重新做 H2 retarget 和 SMPL-X FK 的情况下优化训练分布，本次新增：

- `scripts/build_h2lora_balanced_from_motionlib.py`

这个脚本以已经验证过的 50Hz paired MotionLib 为输入，只做 bucket 级有放回重采样和 key 重命名：

- 输入：`/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz`
- 输出：`/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz`
- target：`5,000,000` frames
- quotas：`locomotion=0.60,dynamic=0.20,general=0.15,static=0.05`
- required keys：`walk_forward_loop_002__A036`、`jog_forward_loop_003__A029_M`、`kozak_dance_002__A354`、`high_jump_R_002__A350_M`

这里选择“有放回重采样”而不是重新 retarget 原始数据，原因是当前问题主要是训练采样分布和 encoder 分支，而不是 MotionLib 契约本身；复用已验证的 50Hz PKL 能避免重新引入 retarget/SMPL-FK 变量。重复片段会被写成 `__balNNN` 后缀的新 key，并在 pkl entry 中记录 `balanced_source_key`、`balanced_copy_index`、`balanced_bucket`，因此 MotionLib 加载时不会因为 key collision 覆盖数据。

实际构造结果：

| bucket | output motions | frames | frame pct | unique source motions |
| --- | ---: | ---: | ---: | ---: |
| `locomotion` | `8527` | `2,999,762` | `59.99%` | `5421` |
| `dynamic` | `3277` | `1,000,029` | `20.00%` | `2777` |
| `general` | `1870` | `750,038` | `15.00%` | `98` |
| `static` | `648` | `250,208` | `5.00%` | `648` |

总计：

- output motions：`14,322`
- unique source motions：`8,944`
- duplicate occurrences：`5,378`
- total frames：`5,000,037`
- robot pkl：`14,322`
- smpl pkl：`14,322`
- 磁盘大小：约 `3.5G`

典型动作校验：

| key | frames | dof | fps | bucket |
| --- | ---: | ---: | ---: | --- |
| `walk_forward_loop_002__A036` | `450` | `31` | `50` | `locomotion` |
| `jog_forward_loop_003__A029_M` | `442` | `31` | `50` | `locomotion` |
| `kozak_dance_002__A354` | `244` | `31` | `50` | `dynamic` |
| `high_jump_R_002__A350_M` | `151` | `31` | `50` | `dynamic` |

注意：数据 PKL 仍然保存 H2 full 31DoF；训练加载时仍由 H2LoRA MotionLib loader 按 `h2lora_headless_dof=true` 裁成 29DoF body，head 两个自由度不进入 policy。

### 14.2 训练配置

新增启动脚本：

- `scripts/run_h2lora_seed5m_balanced_mixed_50k.sh`

关键配置：

- fresh start：不传入旧 `checkpoint`，`+resume=false`
- strict load base：仍通过 `sonic_h2lora.yaml` 的 `pretrained_model.path=sonic_release/last.pt` 加载 release G1 checkpoint
- LoRA：rank `32`、alpha `32`、dropout `0`
- encoder sampling：`g1=1.0`、`teleop=1.0`、`smpl=1.0`
- `teleop_sample_prob_when_smpl=0.5`
- `optimize_encoders_ratio_for_CHIP=true`。这是旧 run 的配置；当前默认已改为 `false`，以恢复 release legacy multi-hot encoder alignment。
- `num_envs=6144`
- `num_steps_per_env=24`
- `num_mini_batches=4`
- `num_learning_epochs=5`
- `actor_lr=2.5e-6`
- `critic_lr=5.0e-6`
- `max_grad_norm=0.03`
- `std_clamp_max=0.38`
- `action_mean_clip=12.0`
- `max_iterations=50000`
- `save_frequency=2500`
- DDR cache：`cache_all_motions_to_ddr=true`、`max_cpu_workers=16`、`ddr_cache_max_bytes=49392123904`
- SwanLab group：`h2lora-seed5m-balanced-mixed-50k`

这次仍是 LoRA-only 训练，不做全参数微调；“从 0 开始”在本文中指 LoRA 适配权重从 release base 重新初始化，而不是从旧 idle/5M checkpoint resume。

### 14.3 每 2.5k 自动可视化

新增 checkpoint watcher：

- `scripts/watch_h2lora_checkpoints_visualize.sh`

启动脚本会在训练进程外额外启动一个 watcher。它会等待当前 run 目录下出现 `model_step_002500.pt`、`model_step_005000.pt`、...、`model_step_050000.pt`，并为每个 checkpoint 导出固定探针视频。

本次根据 `kozak_dance_002__A354` 明显不适合作为当前 H2LoRA 收敛探针的问题，更新了自动可视化链路：

- `scripts/watch_h2lora_checkpoints_visualize.sh`
  - 新增 `H2LORA_VIZ_START_STEP`，可以从指定 checkpoint 开始等待，避免替换 watcher 后重复跑旧 checkpoint。
  - 新增 `H2LORA_VIZ_MOTIONS_FILE`，每个 checkpoint 触发前都会重新读取 motion list；因此可以在不中断训练、不重启 watcher 的情况下修改后续可视化动作。
  - 默认 `H2LORA_VIZ_NUM_STEPS` 从 `360` 改为 `371`，覆盖当前典型动作列表里最长的 `walk_forward_stop_001__A039` 371 帧。
- `scripts/h2lora_typical_viz_motions.txt`
  - 新增典型动作列表文件，一行一个 `motion_key[:encoder]`。
- `scripts/run_h2lora_seed5m_balanced_mixed_50k.sh`
  - 后续新启动的 auto-viz 默认使用 `scripts/h2lora_typical_viz_motions.txt`，不再内置旧的 `kozak_dance` 探针。

当前典型动作列表：

- `walk_forward_stop_001__A039:g1`
- `walk_forward_stop_001__A039:smpl`
- `jog_forward_loop_002__A037_M:g1`
- `turn_walk_360_001__A149_M:g1`
- `jog_backward_loop_003__A034:g1`
- `jog_sideway_045_loop_002__A025:g1`
- `Relaxed_throw_ball_001__A057_M:g1`
- `neutral_stoop_down_R_007__A100_M:g1`
- `hurry_idle_loop_004__A146_M:g1`

这些动作均从当前 balanced 5M MotionLib 的实际 PKL 中确认存在，且都有 paired SMPL 数据。帧数在 `341~371` 之间，50Hz 下约 `6.8~7.4s`，适合用 `371` 帧窗口播放一个相对完整的动作，而不是像早先 8 帧/过短视频那样无法判断行为。

注意：`kozak_dance_002__A354` 仍然记录在 14.1 的 required keys 中，那是 balanced 数据集构造时的真实输入；但它已经从当前自动评估视频队列中移除。这里区分“训练数据中是否存在”和“是否适合作为固定收敛探针”两个问题。

每个视频流程分两段：

1. Isaac Sim Python 执行 `scripts/visualize_h2lora_ref_tracking.py --collect-only`，采集当前 checkpoint 的 1-env rollout。
2. `.venv-h2lora/bin/python` 执行同一脚本 `--render-only`，生成双 H2 机器人 mp4：蓝色半透明为 reference，橙色为 tracking。

输出命名：

`visualizations/h2lora_auto/<exp_name>_<run_timestamp>/step_<step>/<motion_key>_<encoder>/ref_tracking.mp4`

这个机制是工程评估补充，不属于 Any2Any 论文方法。加入它的原因是当前问题多次表现为 aggregate reward 还在下降/上升但肉眼 tracking 模式完全不同，必须固定 motion、固定 encoder 分支、固定 checkpoint 间隔做可视化对比。

### 14.4 实际启动状态

已于 `2026-06-03 15:35:58 +08:00` 启动训练：

- exp：`h2lora_seed5m_balanced_mixed_release_lora_50k_6144env`
- train PID：`22624`
- Isaac Python PID：`22639`
- GPU monitor PID：`22626`
- old auto-viz watcher PID：`22633`，已停止并替换
- current typical auto-viz watcher PID：`538152`
- train log：`/tmp/h2lora_longtrain/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_153558.log`
- GPU log：`/tmp/h2lora_longtrain/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_153558_gpu.csv`
- old auto-viz log：`/tmp/h2lora_longtrain/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_153558_viz.log`
- current typical auto-viz log：`/tmp/h2lora_longtrain/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_153558_viz_typical_20260603_205712_typical.log`
- output dir：`logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_seed5m_balanced_mixed_release_lora_50k_6144env-20260603_153600`
- SwanLab run：`https://swanlab.cn/@sky931/sonic-h2lora/runs/e93dr284jga8cf9e5g57j`

启动日志确认：

- `checkpoint=<empty; fresh LoRA from release>`
- `resume=false`
- saved `config.yaml` 中 `checkpoint: null`
- saved `config.yaml` 中 `resume: false`
- `motion_file=/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz/robot`
- `smpl_motion_file=/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz/smpl`
- `Loaded 14322 motion files`
- `H2LoRA bridge enabled: policy_action_dim=29, sim_joints=31, head_defaults=[0.0, 0.0]`
- `Applied H2LoRA: rank=32, alpha=32.0, dropout=0.0, trainable=866272/38240091 (2.265%), modules=12`
- optimizer trainable params：actor `489440`、critic `376832`

早期健康检查：

- 已进入 PPO training loop，抽查到 `Learning iteration 17`
- `Computation` 约 `35k steps/s`
- `Mean rewards` 从初期负值进入 `0.06` 附近波动
- `Env/step/h2lora/error_head_default_pos=0.0000`
- 未发现 NaN/inf/训练退出
- GPU 采样中显存峰值约 `19.9GB`，GPU util 最高约 `92%`，瞬时功耗最高约 `474.6W`

`model_step_002500.pt` 已保存并完成一批新的典型动作可视化。输出目录：

`visualizations/h2lora_auto/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_205712_typical/step_002500/`

这批 step 2500 是手动补跑，使用 `H2LORA_VIZ_NUM_STEPS=370`，50Hz 下约 `7.4s`；当前后台 watcher 已改为 `371` 帧，用于后续 checkpoint。

step 2500 视频摘要：

| motion / encoder | mean root error | mean joint MAE | done count |
| --- | ---: | ---: | ---: |
| `walk_forward_stop_001__A039:g1` | `0.6419` | `0.1489` | `0` |
| `walk_forward_stop_001__A039:smpl` | `2.2826` | `0.3825` | `0` |
| `jog_forward_loop_002__A037_M:g1` | `1.2967` | `0.1822` | `0` |
| `turn_walk_360_001__A149_M:g1` | `0.3544` | `0.2013` | `0` |
| `jog_backward_loop_003__A034:g1` | `0.5615` | `0.1867` | `0` |
| `jog_sideway_045_loop_002__A025:g1` | `2.1320` | `0.1558` | `0` |
| `Relaxed_throw_ball_001__A057_M:g1` | `0.6325` | `0.2023` | `0` |
| `neutral_stoop_down_R_007__A100_M:g1` | `0.6978` | `0.2558` | `0` |
| `hurry_idle_loop_004__A146_M:g1` | `0.0519` | `0.1419` | `0` |

这些数值来自 `collect_meta.json`，可视化采集时关闭了训练 early termination/events，因此 `done count=0` 表示 rollout 没被可视化端中断，不等同于训练中的 episode termination rate。当前 step 2500 仍属于早期阶段：idle/root 静态类最容易跟上，SMPL 分支的同一 walk 明显比 G1 分支差，侧移和慢跑 root error 仍大，说明 mixed-encoder 大训练还没有完全摆脱 locomotion 局部最优。

当前 typical auto-viz watcher 已重新挂起，正在等待：

`model_step_005000.pt`

后续输出仍写入：

`visualizations/h2lora_auto/h2lora_seed5m_balanced_mixed_release_lora_50k_6144env_20260603_205712_typical/step_<step>/`

### 14.5 SwanLab 断网后的 active-run 安全补传

训练中途网络短暂中断后，网页端 SwanLab 指标停止更新，但本地 `backup.swanlab` 仍持续增长。复查 SwanLab CLI 后发现，普通：

`swanlab sync swanlog/run-...`

不适合直接用于仍在训练的 active run。原因是 active run 的 `backup.swanlab` 尚无 footer，SwanLab CLI sync 在结尾会把没有 footer 的实验状态更新为 `CRASHED`。因此本次新增：

- `scripts/swanlab_safe_active_sync.py`
  - 对单个 active run 做安全补传。
  - 传入同一个 cloud run id。
  - monkeypatch 掉 `update_experiment_state`，只上传指标/日志，不修改云端实验状态。
- `scripts/watch_swanlab_active_sync.sh`
  - 周期性调用上面的安全补传脚本。
  - 默认每 `600s` 检查一次。
  - 默认本地 `backup.swanlab` 比上次同步增长超过 `1MiB` 才补传。
  - 可绑定训练 PID，训练结束后做最后一次 active-sync 并退出。

当前 run 已启动该 watcher：

- SwanLab run：`https://swanlab.cn/@sky931/sonic-h2lora/runs/e93dr284jga8cf9e5g57j`
- local run dir：`/home/sky/workspace/GR00T-WholeBodyControl/swanlog/run-20260603_153603-e93dr284jga8cf9e5g57j`
- active-sync watcher PID：`196935`
- watcher log：`/tmp/h2lora_longtrain/swanlab_active_watch_e93dr284jga8cf9e5g57j_20260603_173458.log`
- per-sync log prefix：`/tmp/h2lora_longtrain/swanlab_active_sync_e93dr284jga8cf9e5g57j_*.log`

第一轮自动补传结果：

- backup size：`129,482,536` bytes
- 上传条目：`73,066/73,066`
- 状态：`Sync completed`
- 训练进程未受影响

注意：训练正常结束后，仍建议再做一次普通 `swanlab sync --id e93dr284jga8cf9e5g57j ...` 作为最终收尾；那时 backup 已有 footer，普通 sync 才能正确写入 `FINISHED` 状态。

## 15. 2026-06-04 SMPL 链路复查与 schema6 修复

用户指出最新 SMPL 分支可视化仍明显像“原地保持平衡”，且 frozen SMPL encoder/FSQ 不应天然失效。本节记录本次继续排查后的真实结论：旧 schema2/3/4/5 的确没有复现 SONIC release 的 SMPL 输入分布；根因是本地 `/mnt/data/seed/smpl-x/uniform` 的 `poses` 不是 SMPL24/SMPL-X body pose 顺序，而是 SOMA-X/BVH 77 关节顺序。schema6 改为直接消费 paired `soma_uniform/bvh`，从 BVH FK/name mapping 生成 SONIC SMPL encoder 需要的 joints/root 后，G1 release 的 SMPL tracking 已恢复 locomotion，H2LoRA 旧 checkpoint 也能跟随同一条 SMPL walk。

### 15.1 停止状态

已停止：

- H2LoRA 训练进程和 Isaac Python 子进程。
- SwanLab active-sync watcher。
- checkpoint auto-viz watcher。
- 本次用于复查的可视化 collect/render 进程。

复查 `nvidia-smi --query-compute-apps` 只剩 RustDesk；`ps` 中不再有 `h2lora`/`swanlab`/`visualize`/`train_agent`/`isaac-sim` 相关残留进程。本节之后没有重启训练，也没有重新挂起自动同步或自动可视化。

### 15.2 原始 SONIC SMPL 链路事实

`sonic_release/config.yaml` 和当前 resolved config 中的 SMPL 相关事实：

- `motion_lib_cfg.target_fps = 50`。
- `motion_lib_cfg.smpl_y_up = true`。
- SMPL tokenizer 输入包括 `smpl_joints_multi_future_local_nonflat`、`smpl_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_smpl`。
- release encoder sampling 是 `g1=1.0, teleop=1.0, smpl=1.0`。

`gear_sonic/envs/manager_env/mdp/observations.py` 中 `smpl_joints_multi_future_local()` 的实际实现是：读取 `command.smpl_joints_multi_future`，再用 `command.smpl_root_quat_w_multi_future` 做 `quat_inv(root_quat) * ref_joints`。它不减 root translation，因此离线 PKL 的 `smpl_joints` 本身必须已经是 pelvis/root-centered 的 joint vectors。

`gear_sonic/envs/manager_env/mdp/commands.py` 中 SMPL root quaternion 的 runtime 路径是：`pose_aa[..., :3] -> angle_axis_to_quaternion -> smpl_root_ytoz_up -> remove_smpl_base_rot`。实时 teleop 路径 `gear_sonic/scripts/pico_manager_thread_server.py::process_smpl_joints()` 也会先 `smpl_root_ytoz_up`，再 `compute_human_joints()`，最后 `remove_smpl_base_rot` 并把 joints 转成 root-local。

`gear_sonic/trl/utils/torch_transform.py::compute_human_joints()` 的输出不是严格标准 SMPL24：它输出前 22 个 SMPL body joints，并在 `use_thumb_joints=True` 时追加 SMPL indices `39, 54`，即左右 thumb joint。这就是 schema6 name-map 最后两个 joint 使用 `LeftHandThumb1/RightHandThumb1` 的原因。

### 15.3 根因

`/mnt/data/seed/smpl-x/uniform/220727/walk_forward_stop_001__A039.npz` 的实测 metadata：

- `joint_names` 前若干项为 `Hips, Spine1, Spine2, Chest, Neck1, Neck2, Head, HeadEnd, Jaw, LeftEye, RightEye, LeftShoulder, ...`。
- `source_bvh_relpath = 220727/walk_forward_stop_001__A039.bvh`。
- `poses.shape = (889, 77, 3)`。
- `transl.shape = (889, 3)`。
- `frame_time = 0.008333...`，约 120Hz。
- `unit = meters`，`rotation_repr = rotvec`。

这说明 `smpl-x/uniform` 文件名容易误导：这里的 `poses[:, :24]` 不是 SONIC SMPL24 body pose。旧 builder 把 `poses[:, :24]` 或 `poses[:, 1:22]` 当作 SMPL body pose，等价于把 neck/head/eye/hand/finger 等 SOMA-X 关节错误塞进 release 的 SMPL encoder 分支，导致 SMPL token 与预训练分布严重错位。

### 15.4 schema6 当前代码状态

当前代码状态：

- `gear_sonic/data_process/build_h2lora_seed_motion_lib.py`
  - `H2LORA_SEED_SMPL_SCHEMA_VERSION = 6`。
  - 默认 `--fps-source 120 --fps 50`、`--smpl-mode soma-bvh`、`--smpl-root-mode soma-bvh-compatible`。
  - 通过 `source_bvh_relpath` 在 `/mnt/data/seed/soma_uniform/bvh` 中找到 paired BVH。
  - 调用 `gear_sonic.data_process.extract_soma_joints_from_bvh.parse_bvh/compute_fk_selected` 做 BVH FK。
  - 用 `SMPL24_FROM_SOMA_NAMES` name-map 到 SONIC 需要的 22 body + 2 thumb joint 顺序。
  - `smpl_joints` 是 Z-up、pelvis-centered、world/root-oriented body vectors，不带 translation。
  - `pose_aa` 只写 root rotvec，非 root 23 个 joints 为 0，因为本地 SEED `poses` 不是 SMPL24 body pose。
  - `_soma_bvh_root_quat_to_smpl_pose_rotvec()` 使 schema6 写入的 SMPL root 在 runtime `smpl_root_ytoz_up + remove_smpl_base_rot` 后，与原 SOMA/BVH root 的 `smpl_root_ytoz_up + remove_bvh_base_rot` 旋转等价。
  - `--skip-existing` 同时检查 robot fps/joint order 与 SMPL schema/fps/source/root mode，避免旧 30Hz 或旧 schema 被误复用。
- `scripts/build_h2lora_idle_debug_data.sh`
  - 默认改为 50Hz、`soma-bvh`、`soma-bvh-compatible`。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2lora.yaml`
  - `optimize_encoders_ratio_for_CHIP=false`，恢复 release legacy multi-hot encoder behavior。
- `scripts/run_h2lora_seed5m_balanced_mixed_50k.sh`
  - 默认 `H2LORA_OPTIMIZE_ENCODERS_RATIO_FOR_CHIP=false`，不再把 YAML 覆盖回旧 one-hot 行为。

### 15.5 schema6 单条数据 smoke

用 `walk_forward_stop_001__A039` 构建 schema6 50Hz 数据：

`/tmp/h2lora_schema6_50hz_probe2`

关键输出：

- manifest：`smpl_schema_version = 6`、`smpl_mode = soma-bvh`、`smpl_root_mode = soma-bvh-compatible`、`fps = 50`、`selected_total_frames = 371`。
- 完整 paired motion pool：`142220`。
- robot：`fps 50`、`dof (371, 31)`、`pose_aa (371, 32, 3)`、root XY displacement `3.5048m`。
- SMPL：`fps 50`、`pose_aa (371, 72)`、`smpl_joints (371, 24, 3)`、`transl (371, 3)`。
- SMPL metadata：`schema_version = 6`、`smpl_joints_source = soma_bvh_fk_name_mapped_to_smpl24`、`smpl_root_mode = soma-bvh-compatible`。
- `pose_aa[:, 3:]` max abs：`0.0`。
- `smpl_joints[:, 0]` max abs：`0.0`。
- `smpl_joints` mean/std/min/max：`-0.0465 / 0.2871 / -0.9910 / 0.5892`。
- SMPL transl XZ displacement：`3.3376m`。
- 第二次同命令加 `--skip-existing` 后结果为 `converted=0, skipped=1`。

### 15.6 G1 release 正证实验

为了排除“H2LoRA/Unitree H2 迁移导致 SMPL 走不起来”，用同一条 `walk_forward_stop_001__A039` 的 G1 source ref 和原始 `sonic_release/last.pt` 跑 G1 release，本体不经过 H2 adapter，强制使用 schema6 SMPL motion：

`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/g1_release_source_ref/schema6_somabvh_walk_probe_20260604_g1release_smpl_20260604_213909/g1_release_source_ref_tracking.mp4`

结果：

| run | ref XY | tracking XY | mean root err | max root err | mean joint MAE | mean action abs | done | encoder index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| G1 release + schema4 SMPL | `2.5846m` | `0.0285m` | `1.7446m` | - | `0.2488` | `1.043` | - | `[1,1,1]` |
| G1 release + schema5 SMPL | `2.5846m` | `0.0431m` | `1.7266m` | - | - | - | - | `[1,1,1]` |
| G1 release + schema6 SMPL | `2.5846m` | `2.7547m` | `0.1142m` | `0.2000m` | `0.1239` | `0.8073` | `0` | `[1,1,1]` |

这个结果是关键正证：同样是 G1 release 本体和 release checkpoint，schema4/5 不走，schema6 可以走完整 walking displacement。因此旧 SMPL 分支“原地站立”的主因是数据契约错，不是 frozen SMPL encoder/FSQ 天然无法表达 locomotion。

### 15.7 H2LoRA 旧 checkpoint smoke

使用旧 5M run 的最新 checkpoint：

`/home/sky/workspace/GR00T-WholeBodyControl/logs_rl/sonic-h2lora/manager/universal_token/all_modes/sonic_h2lora_h2lora_seed5m_balanced_mixed_release_lora_50k_6144env-20260603_153600/last.pt`

强制 SMPL encoder，输入 schema6 同一条 `walk_forward_stop_001__A039`：

`/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora_manual/schema6_somabvh_walk_probe_20260604_h2lora_smpl_20260604_214111/ref_tracking.mp4`

结果：

| run | ref XY | tracking XY | mean root err | max root err | mean joint MAE | mean policy action abs | mean sim action abs | done | encoder index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H2LoRA old ckpt + schema3 SMPL | `3.5048m` | `1.3801m` | `3.6067m` | - | `0.4877` | - | - | - | `[0,0,1]` |
| H2LoRA old ckpt + schema4 SMPL | `3.5048m` | `0.3995m` | `2.4836m` | - | `0.3434` | - | - | - | `[0,0,1]` |
| H2LoRA old ckpt + schema6 SMPL | `3.5048m` | `3.4709m` | `0.2879m` | `0.4238m` | `0.1968` | `1.6670` | `0.5148` | `0` | `[0,0,1]` |

这个 smoke 不能证明旧 checkpoint 已经是最终好模型，因为该 checkpoint 本身是旧 schema/旧 CHIP 配置训练出来的；但它证明 schema6 SMPL token 至少能进入 H2LoRA policy 并触发 locomotion，不再是“任意 SMPL 输入都原地站”。需要注意 action clipping 仍存在：旧 H2LoRA collect 中 policy action max 到 `12.0`、sim action max 到 `4.0`，后续训练仍要观察 action saturation、hip/ankle error 和 torso shake。

### 15.8 测试覆盖

当前通过的轻量测试：

`.venv-h2lora/bin/python -m pytest gear_sonic/tests/test_h2lora_seed_smpl.py gear_sonic/tests/test_h2lora_alignment.py gear_sonic/tests/test_lora_adaptation.py -q`

结果：`14 passed`。

其中 `gear_sonic/tests/test_h2lora_seed_smpl.py` 覆盖：

- runtime `smpl_root_ytoz_up` root conversion。
- `human-fk`/`smplx-fk` fallback 均不携带 translation，pelvis-centered。
- schema6 metadata 与 pelvis-centered contract。
- `soma-bvh` 模式会忽略错误的 77-joint `poses` body 部分，使用 BVH FK name-map，非 root `pose_aa` 为 0。
- `soma-bvh-compatible` root 写入后，与 BVH runtime root 旋转等价。
- `--skip-existing` 只接受当前 schema。

### 15.9 仍需关注的问题

1. 官方 release 训练配置里的 SMPL 路径是 `data/motion_lib_bones_seed/1215_bones_seed_filtered/`，README/HF downloader 也提到 `data/smpl_filtered/`。本机当前仍没有官方 SMPL PKL，HF API/下载曾长时间无响应。因此 schema6 是通过本地 SEED + G1 release 行为验证闭合，不是逐字段对照官方 PKL 后的完全同源证明。
2. 旧 1M idle、3M、5M balanced/mixed 训练都基于错误 SMPL schema 或旧 `optimize_encoders_ratio_for_CHIP=true` 配置，不能作为 schema6 之后的收敛结论。
3. schema6 的 `pose_aa` 非 root body pose 置零。这符合“本地 SEED 文件不是 SMPL24 pose”的现实约束，也足以驱动 release SMPL encoder 的 tested walking clip；但对全动作类别是否足够，需要重新训练和多类别可视化验证。
4. H2 action adapter、hip `pinv` 投影、ankle coupling、foot termination/curriculum、torso shake 仍是 H2LoRA 动力学适配问题；SMPL 数据链路修复不等于这些问题自动解决。

### 15.10 当前建议

1. 立即用 schema6 默认参数重建目标规模数据集：`--fps-source 120 --fps 50 --smpl-mode soma-bvh --smpl-root-mode soma-bvh-compatible`。
2. 从 `sonic_release/last.pt` 重新开始 LoRA 训练，不沿用旧 idle/5M checkpoint 作为最终 run 的初始化。
3. 使用 release mixed encoder sampling：`g1=1, teleop=1, smpl=1`、`teleop_sample_prob_when_smpl=0.5`、`optimize_encoders_ratio_for_CHIP=false`。
4. 先做小 motion overfit 和固定 walking/jog/side-step/idle probe，每个 checkpoint 分别强制 `g1/teleop/smpl` encoder 可视化，再启动 5M/50k 长训。
5. SwanLab 中继续分 encoder 记录 root displacement、root error、joint MAE、hip/ankle error、termination、action saturation、torso/wrist angular velocity，避免 aggregate reward 掩盖 SMPL 分支或 dynamics 分支问题。

## 16. 2026-06-04 schema6 5M cache 重建准备与可视化 fps 修复

用户反馈 schema6 smoke 的 tracking 看起来是对的，但视频像慢动作。复查 `ffprobe` 后确认：

- `/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora_manual/h2lora_seed5m_last_schema2_compare_20260604_183813/*/ref_tracking.mp4`
  - 容器帧率：`50/1`
  - 平均帧率：`50/1`
  - 帧数：`371`
  - 时长：`7.42s`
  - 结论：这个旧 schema2 对比目录本身不是 30fps 慢放。
- `/home/sky/workspace/GR00T-WholeBodyControl/visualizations/h2lora_manual/schema6_somabvh_walk_probe_20260604_h2lora_smpl_20260604_214111/ref_tracking.mp4`
  - 容器帧率：`30/1`
  - 帧数：`371`
  - 时长：`12.366667s`
  - 结论：schema6 H2LoRA smoke 是手动 render 时沿用了 `scripts/visualize_h2lora_ref_tracking.py` 的旧默认 `--fps=30`，因此同一条 50Hz 轨迹被播放成了 30fps。
- `/home/sky/workspace/GR00T-WholeBodyControl/visualizations/g1_release_source_ref/schema6_somabvh_walk_probe_20260604_g1release_smpl_20260604_213909/g1_release_source_ref_tracking.mp4`
  - 容器帧率：`30/1`
  - 帧数：`371`
  - 时长：`12.366667s`
  - 结论：这条 G1 schema6 smoke 也是 render 参数导致的慢放。

代码修复：

- `scripts/visualize_h2lora_ref_tracking.py`
  - `--fps` 默认从 `30.0` 改为 `50.0`。
  - 新增 `_trajectory_fps()`，`--fps 0` 时使用 trajectory `.npz` 内保存的 fps。
  - `render_meta.json` 新增 `trajectory_fps`、`duration_s`、`trajectory_duration_s`，用于快速判断视频播放速度是否和训练/数据 fps 一致。
- `scripts/visualize_g1_release_on_source_ref.py`
  - 复用 `_trajectory_fps()`，metadata 同步记录 trajectory fps 和 video fps。
- `scripts/visualize_g1_release_on_h2adapted_ref.py`
  - 同步上述 fps metadata，默认仍为 50fps。

这部分不是论文 Any2Any 的算法细节，而是实验可解释性修复。论文评估和本文工程诊断都依赖视频观察，如果 50Hz motion 以 30fps 播放，会把步频、恢复速度、躯干晃动频率全部误读为“动作慢/控制迟缓”。

### 16.1 新 schema6 5M cache 构建入口

新增：

- `scripts/build_h2lora_seed5m_schema6_cache.sh`

默认输出：

- source random 5M：`/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6`
- balanced mixed 5M：`/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6`

默认参数：

- `fps_source=120`
- `fps=50`
- `target_frames=5_000_000`
- source random seed：`20260602`
- balanced random seed：`20260603`
- `num_workers=16`
- `smpl_mode=soma-bvh`
- `smpl_root_mode=soma-bvh-compatible`
- `smpl_bvh_root=/mnt/data/seed/soma_uniform/bvh`

设计原因：

- 不覆盖旧 `/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz` 和 `/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz`，因为旧目录是 `smplx-fk`/错误 SMPL schema 的历史诊断证据。
- 新目录用 `_schema6` 后缀，训练脚本、watcher、G1 adapter 默认都切到新目录，避免误用旧数据。
- 构建链路仍保持“全 clip 随机采样 -> balanced resampling”的原有工程策略，区别只在 SMPL 分支数据契约修复和 schema 强校验。

### 16.2 balanced cache schema 校验

`scripts/build_h2lora_balanced_from_motionlib.py` 新增源数据校验：

- robot/SMPL key 必须成对。
- robot fps 和 SMPL fps 必须一致，默认要求 `50`。
- `root_trans_offset` shape 必须为 `(T, 3)`。
- `smpl_joints` shape 必须为 `(T, 24, 3)`。
- `pose_aa` shape 必须为 `(T, 72)`。
- robot/SMPL frame count 必须一致。
- 默认要求 `schema_version=6`。
- 默认要求 `smpl_joints_source=soma_bvh_fk_name_mapped_to_smpl24`。
- 默认要求 `smpl_root_mode=soma-bvh-compatible`。

这同样不是论文要求的算法细节，而是为了确保实验真正使用修复后的 SMPL token 输入分布。旧 5M manifest 复查显示：

- `/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz`
  - `fps=50`
  - `smpl_mode=smplx-fk`
  - `smpl_schema_version=null`
  - `selected_total_frames=5000227`
  - `num_motions=13889`
- `/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz`
  - `source_root=/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz`
  - `selected_total_frames=5000037`
  - `num_output_motions=14322`
  - `num_unique_source_motions=8944`

因此旧 5M/balanced 5M 不能继续作为当前 schema6 训练数据。新的 balanced 脚本会在源数据不是 schema6 时直接报错，而不是静默复制。

### 16.3 继续训练入口

已更新：

- `scripts/run_h2lora_seed5m_balanced_mixed_50k.sh`
  - 默认 `H2LORA_MOTION_FILE=/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/robot`
  - 默认 `H2LORA_SMPL_MOTION_FILE=/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6/smpl`
  - 新增 `H2LORA_CHECKPOINT` 和 `H2LORA_RESUME` 支持。
- `scripts/run_h2lora_seed5m_from_release.sh`
  - 默认 source 5M 数据切到 `/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6`。
- `scripts/watch_h2lora_checkpoints_visualize.sh`
  - 默认可视化数据切到 balanced schema6 路径。
- `scripts/visualize_g1_release_on_h2adapted_ref.py`
  - 默认 H2 source 数据切到 source schema6 路径。

继续上次权重的建议区分两种场景：

- 如果要严格恢复旧 run 的 optimizer/scheduler：`H2LORA_CHECKPOINT=/path/to/last.pt H2LORA_RESUME=true`。
- 如果只想用旧权重初始化，在 schema6 cache 上开一个新 run：`H2LORA_CHECKPOINT=/path/to/last.pt H2LORA_RESUME=false`。

由于旧 checkpoint 是在旧 SMPL schema 上训练出的权重，当前更推荐第二种：只加载权重，重新记录一个 schema6 run，避免旧 optimizer state 与新数据分布强绑定。

### 16.4 schema6 5M cache 实际构建结果

构建命令：

`PYTHON_BIN=.venv-h2lora/bin/python H2LORA_DATA_WORKERS=16 bash scripts/build_h2lora_seed5m_schema6_cache.sh`

source random 5M：

- 路径：`/mnt/data/seed/h2lora_motion_lib/seed_5m_50hz_schema6`
- robot 文件数：`13889`
- SMPL 文件数：`13889`
- 目录大小：`2.8G`
- `fps=50`
- `smpl_schema_version=6`
- `smpl_mode=soma-bvh`
- `smpl_root_mode=soma-bvh-compatible`
- `selected_total_frames=5000227`
- `num_motions=13889`
- `total_paired_motions=142220`

balanced mixed 5M：

- 路径：`/mnt/data/seed/h2lora_motion_lib/seed_5m_balanced_mixed_50hz_schema6`
- robot 文件数：`14322`
- SMPL 文件数：`14322`
- 目录大小：`2.9G`
- `selected_total_frames=5000037`
- `num_output_motions=14322`
- `num_unique_source_motions=8944`
- `duplicate_occurrences=5378`
- `required_fps=50`
- `required_smpl_schema_version=6`
- `required_smpl_source=soma_bvh_fk_name_mapped_to_smpl24`
- `required_smpl_root_mode=soma-bvh-compatible`
- source manifest：`schema_version=6`、`smpl_mode=soma-bvh`、`smpl_root_mode=soma-bvh-compatible`

balanced bucket 分布：

| bucket | motions | frames | unique sources | frame pct |
| --- | ---: | ---: | ---: | ---: |
| locomotion | `8527` | `2999762` | `5421` | `59.9957%` |
| dynamic | `3277` | `1000029` | `2777` | `20.0004%` |
| general | `1870` | `750038` | `98` | `15.0007%` |
| static | `648` | `250208` | `648` | `5.0032%` |

随机抽样校验 8 条 balanced PKL，robot/SMPL 均满足：

- robot fps = SMPL fps = `50`
- SMPL `schema_version=6`
- SMPL `smpl_joints_source=soma_bvh_fk_name_mapped_to_smpl24`
- SMPL `smpl_root_mode=soma-bvh-compatible`
- `smpl_joints` shape = `(T, 24, 3)`
- `pose_aa` shape = `(T, 72)`
- robot/SMPL frame count 一致

因此本轮继续训练的数据准备状态为：schema6 5M source 和 balanced cache 均已构建完成，默认训练脚本已指向 balanced schema6 路径。

## 17. H2LoRA deploy/sim2sim 与 G1 sim2sim 一致性核对

本节记录 2026-06-19 针对部署侧 MuJoCo sim2sim 的核对结果，重点对应用户反馈的“输入 `y` 后、按 `]`/`9` 后机器人像膝盖没力一样跪倒”。

### 17.1 状态机行为

G1 和 H2LoRA 共用 `g1_deploy_onnx_ref.cpp` 的状态机：

- `y` 是 shell/deploy.sh 的启动确认，不是 policy control start。
- 程序启动后先进入 `INIT`，`InitControl()` 会以 50Hz 发送默认站姿 PD 命令，约 3 秒内从当前电机状态插值到默认角。
- `]` 之后才会从 `WAIT_FOR_CONTROL` 进入 `CONTROL`，开始采集 observation、跑 encoder/decoder、发送 policy action。

因此 `y` 后机器人有轻微归位动作是正常的；`y` 本身不应该启动 policy。若在 `]` 前就有明显高频大幅抖动，应优先检查 lowcmd 默认姿态、MuJoCo 初始姿态和 DDS 是否有旧进程残留。

### 17.2 29D/31D adapter 一致性

已用脚本从 C++ header 和 Python 训练侧常量做逐项核对：

- C++ `h2lora_body_to_h2_mujoco` 展开的 joint name 序列与 Python `H2LORA_BODY_JOINTS_IN_G1_ORDER` 完全一致。
- C++ `h2lora_target_to_source_hip` / `h2lora_source_to_target_hip` 与 Python `h2lora_alignment.hip_alignment_matrices()` 数值一致，最大差异为 `1.1e-16` 量级。
- H2LoRA MJCF 的 31 个 body/head joint 顺序与 31 个 actuator 顺序完全一致。
- 默认姿态检查：H2LoRA YAML `DEFAULT_DOF_ANGLES` 与 deploy `h2_default_angles` 逐项一致。

结论：当前没有发现 29D semantic 到 H2 31D physical 的训推 adapter 索引或 hip 投影不一致。

### 17.3 发现并修复的 deploy observation 顺序错误

上一轮 deploy 核对中定位到的关键问题之一是 H2LoRA deploy 的 decoder observation 拼接顺序与训练配置不一致。

训练侧 `sonic_h2lora_release/config.yaml` 中 actor policy obs 的有效拼接顺序以 ONNX export 后的 decoder 输入契约为准。虽然部分 IsaacLab YAML 默认项中曾把 `gravity_dir` 写在前面，但实际 `PolicyCfg`/ONNX proprioception slice 的顺序是：

1. `base_ang_vel`
2. `joint_pos`
3. `joint_vel`
4. `actions`
5. `gravity_dir`

每个 proprioception term 的 history length 都是 10，总维度 `30 + 30 + 290 + 290 + 290 = 930`。ONNX decoder 输入维度为 `994 = 64 token_state + 930 actor_obs`。

因此 H2LoRA deploy YAML 的正式顺序必须为：

1. `token_state`
2. `his_base_angular_velocity_10frame_step1`
3. `his_body_joint_positions_10frame_step1`
4. `his_body_joint_velocities_10frame_step1`
5. `his_last_actions_10frame_step1`
6. `his_gravity_dir_10frame_step1`

此前排障中出现过把 `his_gravity_dir_10frame_step1` 放到 token 后面的 deploy YAML。那会导致 930D proprioception slice 整段错位。由于维度仍然等于 994，旧逻辑只做维度检查无法发现该错误；policy 实际看到的是错误语义的输入，因此默认站立 reference 也会输出较大的非站立动作，视觉上表现为“膝盖像没力气、努力稳定但跪倒”。

已修复：

- `gear_sonic_deploy/policy/release/observation_config_h2_lora.yaml`
  - 保持为 `token_state -> base_ang_vel -> joint_pos -> joint_vel -> last_action -> gravity`。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  - 新增 H2LoRA 专属 observation layout fail-fast 校验。
  - H2LoRA 下必须严格满足 `64 + 930 = 994`，并逐项检查 observation 名称和维度；如果后续 YAML 顺序再次偏离训练配置，程序会在启动阶段直接报错。

### 17.4 H2LoRA 专属 sim 配置

为了避免 H2LoRA 修复影响 G1 baseline，以下行为均改为 H2LoRA YAML 显式开启：

- `APPLY_DEFAULT_POSE_ON_RESET: True`
  - 只在 `h2_lora_sonic.yaml` 开启；避免 MuJoCo H2 从 XML 零姿态开始。
- LowCmd wait gate 已删除
  - 当前 H2LoRA MuJoCo 与 G1 一样始终推进物理；没有 fresh LowCmd 时，显式 torque PD 自然输出零/上一状态，不再有 H2 专属“等待第一条 LowCmd 再 step”的配置项。
  - 这样做是为了避免 sim 端出现 G1 没有的隐藏状态机，使 `9`、deploy 停止、按键切换等操作语义一致。
- `USE_LOCAL_BASE_GYRO: True`
  - H2LoRA 主 IMU gyro 使用 MuJoCo root body-local angular velocity，对齐 IsaacLab `base_ang_vel`/`root_ang_vel_b`。
  - G1 默认仍走原 `floating_base_vel[3:6]`，不静默改变 G1 sim2sim。
- `OVERRIDE_MUJOCO_JOINT_FORCE_LIMITS: True`
  - 不修改 MJCF/URDF 资产文件，只在 MuJoCo model 加载后把 `jnt_actfrcrange` 覆盖成训练侧 H2LoRA actuator effort limit。
  - 这个修复用于对齐训练侧力矩上限，尤其 H2 ankle roll/pitch；不是通过改资产“增强机器人”。
- `OVERRIDE_MUJOCO_DOF_PASSIVE: True`
  - 不落盘修改 MJCF，只在 runtime `MjModel` 上覆盖 31 个 body/head joint 的 `dof_armature/dof_damping/dof_frictionloss`。
  - 当前 H2LoRA sim2sim 使用按关节配置的 `MUJOCO_DOF_ARMATURE` 列表，并统一设置 `damping=0.05`、`frictionloss=0.2`，用于让 H2 的显式 torque PD 回路更接近训练侧 actuator 配置。
- `MUJOCO_INTEGRATOR/SOLVER/ITERATIONS`
  - H2LoRA 使用 `implicitfast + Newton + 50 iterations + 10 ls_iterations`。
  - 这是 deploy MuJoCo runtime 的数值设置，不属于 Any2Any 论文方法，也不改变训练或导出模型。
- `ADD_DEFAULT_MUJOCO_SCENE` / `MJCF_MESHDIR`
  - 在运行时临时 scene XML 中补 floor/light/camera/meshdir；原始 H2 MJCF 文件不落盘修改。
- MuJoCo low-level PD
  - 当前固定为 Unitree/G1 风格显式 torque PD：`tau + kp*(q_des-q) + kd*(dq_des-dq)`。
  - 曾短暂尝试通过 Python 全局 lowcmd kp/kd scale 或 split default-pose attractor 改善默认站立，但这些开关会改变 `CONTROL` 下 policy action target 的闭环动力学，因此 cleanup 后已删除，不作为正式 sim2sim 路径。
- H2LoRA INIT/WAIT holding gain
  - `g1_deploy_onnx_ref` 中新增 `ProfileInitKp/ProfileInitKd`，只在 `InitControl()` 默认 standing command 中把 H2LoRA kp/kd 放大为 `32.0/5.7` 倍。
  - `CreatePolicyCommand()` 进入 `CONTROL` 后仍使用原始 `h2_kps/h2_kds`，不放大 policy 动作目标。
- `DEFAULT_BASE_POS` / `ELASTIC_BAND_POINT`
  - `DEFAULT_BASE_POS` 改为 `[0, 0, 1.04]`，与 IsaacLab H2 init root height 对齐。
  - 当前不新增任何“未收到 LowCmd 前默认站姿控制”逻辑；H2 第一终端启动时和 G1 一样只依赖 MuJoCo `ElasticBand` 外力支撑。
  - `ELASTIC_BAND_POINT=[0,0,1.30]`，对应 H2 默认站姿下 `torso_link` 高度约 `1.09m`，与 G1 吊带相对 attached body 的高度差约 `0.21m` 对齐。
  - `ELASTIC_BAND_KP_POS/KD_POS/KP_ANG/KD_ANG=10000/1000/1000/10`，恢复为 G1 sim2sim 的同量级弹性支撑，而不是之前较弱的 translational-only sling。

### 17.5 当前验证

已运行：

- `.venv-h2lora/bin/python -m compileall -q gear_sonic/utils/mujoco_sim gear_sonic/scripts/run_sim_loop.py`
- `.venv-h2lora/bin/pytest gear_sonic/tests/test_h2lora_alignment.py -q`
  - 结果：`4 passed`
- `cd gear_sonic_deploy && source scripts/setup_env.sh && just build`
  - 结果：`g1_deploy_onnx_ref` 构建成功。
- `.venv-h2lora/bin/python -m py_compile gear_sonic/utils/mujoco_sim/base_sim.py gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py gear_sonic/scripts/run_sim_loop.py`
  - 结果：通过。
- MuJoCo XML 加载检查：
  - H2 MJCF runtime model: `nq=38`、`nv=37`、`nu=31`、`njnt=32`。
  - 31 个 H2 body/head joint 的 MuJoCo joint 顺序与 actuator/control 顺序逐项一致。

默认站立 reference 的 headless 短测结果：

| case | first severe tilt | first tilt > 0.3 rad | first 10s action norm mean/max | note |
| --- | ---: | ---: | ---: | --- |
| 修复前，proprioception 顺序错误 | `5.16s` 超过 `0.6 rad` | `5.04s` | `8.18 / 21.39` | 默认站立也很快跪倒 |
| 修复后，顺序与训练一致 | 55s 内未超过 `0.6 rad` | `28.38s` | `3.66 / 6.16` | 默认站立显著稳定 |

直接运行 `g1_deploy_onnx_ref` 并在 `Init Done` 后发送 `]` 的 CONTROL smoke：

- 日志目录：`gear_sonic_deploy/logs/h2_lora_default_stand_direct_control_20260619_131208`
- runtime policy observation order：
  - `token_state@0`
  - `his_base_angular_velocity_10frame_step1@64`
  - `his_body_joint_positions_10frame_step1@94`
  - `his_body_joint_velocities_10frame_step1@384`
  - `his_last_actions_10frame_step1@674`
  - `his_gravity_dir_10frame_step1@964`
- CONTROL 时长约 `32.08s`，`motion_playing=0`，即默认站立 reference 停在第 0 帧。
- action norm：mean `3.76`、max `6.41`、p95 `5.68`。
- base tilt：max `0.244 rad`、p95 `0.208 rad`、结束时 `0.114 rad`。

### 17.6 2026-06-19 追加：H2 MuJoCo 显式 torque PD 校准

用户继续反馈默认站立在 MuJoCo sim2sim 中仍表现为膝盖/脚踝力气不够。为排除 policy 和 action adapter 的影响，本次直接在 `DefaultEnv` 中构造无 policy 的默认 standing lowcmd，走同一条 `lowcmd -> compute_body_torques() -> mj_data.ctrl -> mujoco.mj_step()` 路径。

诊断结论：

- H2LoRA C++ deploy 的 `h2_default_angles/h2_kps/h2_kds` 与 Python YAML 的 `DEFAULT_MOTOR_ANGLES/MOTOR_KP/MOTOR_KD` 数值一致，默认角最大差异为 `0`，kp/kd 最大差异为浮点舍入量级。
- 使用原始 release `kp/kd`、H2 MJCF 和外部 Python 显式 torque PD 时，无 policy 默认站立约 `1.3s` 就会倒；这说明“按 `y` 后还没按 `]` 就站不住”的问题不能只从 decoder/action/ref 去解释。
- 旧 `ELASTIC_BAND_POINT=[0,0,1.08]` 低于 H2 默认站姿的 `torso_link` 高度，悬挂阶段会产生向下拉力；因此“悬挂着也在抖/放下后更快跪”会被这个配置放大。
- 当前没有修改 H2 MJCF/URDF 资产文件。修复全部发生在 H2LoRA runtime config 和 Python MuJoCo wrapper 中，目的是让部署 sim 的低层动力学更接近训练侧 implicit actuator 的效果。

新增实现：

- `gear_sonic/utils/mujoco_sim/base_sim.py`
  - 新增 `_apply_mujoco_runtime_options()`：从 YAML 设置 integrator/solver/iterations。
  - 新增 `_override_mujoco_dof_passive()`：runtime 覆盖 dof passive 参数。
  - `compute_body_torques()` 固定为 Unitree/G1 风格显式 torque PD，不再暴露 H2 专属 PD scale/split 调试开关。
- `gear_sonic/utils/mujoco_sim/wbc_configs/h2_lora_sonic.yaml`
  - `DEFAULT_BASE_POS=[0,0,1.04]`
  - `ELASTIC_BAND_POINT=[0,0,1.30]`
  - `ELASTIC_BAND_KP_POS/KD_POS/KP_ANG/KD_ANG=10000/1000/1000/10`
  - `MUJOCO_INTEGRATOR=implicitfast`
  - `MUJOCO_SOLVER=Newton`
  - `MUJOCO_ITERATIONS=50`
  - `MUJOCO_LS_ITERATIONS=10`
  - `MUJOCO_DOF_ARMATURE` 为 31 维按关节列表
  - `MUJOCO_DOF_DAMPING=0.05`
  - `MUJOCO_DOF_FRICTIONLOSS=0.2`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`
  - 新增 `ProfileInitKp/ProfileInitKd`。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  - `InitControl()` 使用 `ProfileInitKp/ProfileInitKd`。
  - `CreatePolicyCommand()` 保持使用原始 `h2_kps/h2_kds`。

无 policy 默认站立 smoke：

| case | result |
| --- | --- |
| release gain，Python 全局 scale=1，无悬挂带 | 约 `1.055s` 到 root height `0.5m`，说明原始显式 torque PD 默认站立不够 |
| INIT/WAIT 等效 gain `32.0/5.7`，无悬挂带，10s | 未倒；最终 root height `0.9842m`；最大 base tilt `0.0298rad`；最大 torque limit ratio `0.602` |
| INIT/WAIT 等效 gain `32.0/5.7`，悬挂带开启，10s | 未倒；最终 root height `0.9903m`；最大 base tilt `0.0076rad`；最大 torque limit ratio `0.317` |

补充说明：

- 这个校准不是 Any2Any 论文方法的一部分，也不是 LoRA 训练链路的一部分；它只服务于 H2LoRA 的 MuJoCo sim2sim 可视化/调试。
- 现阶段不需要修改 MJCF 资产文件，因为在不改资产的前提下，同一条低层 torque path 已经能稳定 hold 默认站立。
- 如果后续仍出现按 `]` 后倒地，应优先区分是 `WAIT_FOR_CONTROL` 默认 holding 失败，还是 `CONTROL` 下 ONNX action/ref tracking 失败。前者看低层 PD/物理/DDS；后者看 observation、token、reference、action adapter 和 policy 本身。特别注意：当前正式链路不再提供 Python 全局 PD scale/split 开关来“救”CONTROL。

当前总括结论：H2LoRA deploy/sim2sim 已修复两类互相叠加的问题。第一类是 decoder observation layout 与训练 layout 不一致，会让 policy 看到错位 proprioception；第二类是 H2 MJCF 在 Python MuJoCo 显式 torque PD 下的低层 holding 能力不足，并被错误悬挂点放大。二者都不是必须修改机器人资产文件才能解释的问题；资产修改暂时不是首选。

### 17.7 2026-06-22 追加：H2 sim2sim 操作逻辑对齐 G1 与 cleanup

本次根据“h2 sim2sim 的操作逻辑需要和 g1 sim2sim 一样”的要求，调整的是默认 sim/deploy 操作语义，不改变 ONNX 模型、policy 前后处理或机器人资产文件。

当前默认入口：

```bash
# Terminal 1，仓库根目录
.venv-h2lora/bin/python gear_sonic/scripts/run_sim_loop.py --wbc-version h2_lora

# Terminal 2，gear_sonic_deploy 目录
./deploy.sh --robot-profile h2_lora --input-type keyboard sim
```

当前默认状态/按键语义：

- `y`：确认 deploy 启动，进入 INIT/WAIT_FOR_CONTROL 默认站姿 ramp；不启动 policy tracking。
- `]`：进入 CONTROL；与 G1 一样立即采集 observation、跑 encoder/decoder，并发布 policy LowCmd。H2LoRA 默认不再用 direct default-pose hold 绕过 policy。
- MuJoCo `9`：与 G1 一样直接切换/释放 elastic support。H2LoRA 默认不再等待第一条 LowCmd 才推进物理，因此只开 MuJoCo 端时 `9` 也会生效。
- `t/T`：与 G1 一样只控制 reference frame 是否播放/暂停；H2LoRA 默认不再在 play-start 时写 MuJoCo reset file，也不等待 reset ack/warmup。
- `n/p`：切换动作；默认不通过 H2 reset-gate 强制把 MuJoCo root/joints 改到新 reference。

对应实现：

- `gear_sonic/utils/mujoco_sim/base_sim.py`
  - 删除 H2LoRA sim reset file/ack、low-level CSV logger、等待 LowCmd 冻结、position-actuator 分支和自动释放弹力带逻辑。
  - `sim_step()` 只保留 G1 风格主链路：发布 LowState、处理 joystick、应用 elastic band、计算显式 torque PD、裁剪 torque、`mj_step()`。
- `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`
  - `ElasticBand` 恢复为直接开关：MuJoCo `9` 只切换 enable，不再做渐释放、force scale 或环境变量调参。
  - 弹簧公式使用 G1/unitree_mujoco 风格的向量 PD。
- `gear_sonic/scripts/run_sim_loop.py`
  - 只保留 `GEAR_SONIC_DOMAIN_ID` 和 `GEAR_SONIC_MUJOCO_ENABLE_ELASTIC` 这类通用入口；删除 H2 专属 scene/meshdir/wait gate 调试覆盖。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  - 删除 H2LoRA play-start reset gate、reference/default direct hold、paused zero-action、external policy action replay、policy zero-group 等诊断分支。
  - H2LoRA 仍保留正式能力：profile 选择、H2 reference manifest 校验、encoder/decoder ONNX、observation layout fail-fast、G1/H2 action adapter、DDS/MuJoCo joint order映射。
- `gear_sonic_deploy/deploy.sh`
  - H2LoRA 默认 motion data 仍为 `reference/example_h2/deploy`，继续校验 `h2_lora_reference_manifest.json`，避免误用 G1 example reference。

被移除的内容包括固定日志 GUI wrapper、deploy/mujoco analyzer、golden/replay/onestep 诊断脚本、sim2sim audit 临时报告和 probe/golden/default-stand reference 目录。保留 `scripts/export_h2lora_deploy_reference.py`，因为它是从 H2 MotionLib 生成 deploy reference 的正式数据链路，不属于 sim2sim 临时调试。

### 17.8 2026-06-22 追加：H2LoRA sim2real 接入与 sim2sim 一致性复查

本次参考了官方 `unitreerobotics/unitree_sdk2` 与 `unitree_sdk2_python` 的 H2 low-level ankle swing 示例。官方 H2 示例使用：

- `unitree_hg::msg::dds_::LowCmd_` / `LowState_`
- `rt/lowcmd` / `rt/lowstate`
- `rt/secondary_imu`
- `MotionSwitcherClient`
- 500Hz LowCmd 写线程
- LowState CRC 校验
- `mode_pr` 和从 LowState 回传的 `mode_machine`
- H2 DDS motor order：腿、腰、双臂、head，共 31 电机

因此 H2LoRA sim2real 不需要新建一套真实机器人控制协议；当前实现继续复用 G1 deploy 可执行程序和状态机，只通过 `--robot-profile h2_lora` 切换 profile。

当前真机入口：

```bash
cd /home/sky/workspace/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --robot-profile h2_lora real
```

也可以显式指定网卡：

```bash
./deploy.sh --robot-profile h2_lora enP8p1s0
```

真机与 G1 保持一致的部分：

- `deploy.sh` 默认 `real` 自动寻找 `192.168.123.*` 网卡；也支持直接传网卡名。
- `y` 只确认 deploy 启动；进入 INIT/WAIT_FOR_CONTROL 默认姿态 ramp。
- `]` 进入 CONTROL，开始按 50Hz 控制循环采集 observation、跑 encoder/decoder、写 policy command。
- LowCmd writer 以 500Hz 发布 `rt/lowcmd`。
- 真机模式启用 CRC 校验。
- 真机模式不跳过 `MotionSwitcherClient`，会像 G1 一样尝试释放高层运动服务。
- `t/T`、`n/p` 等 motion playback 操作仍走 G1 的 input/state 逻辑。

H2LoRA profile 的差异：

- policy interface 仍是 29D body action，不控制 head。
- LowCmd command motor count 是 31；head pitch/yaw 两个 DDS motor 每帧写 H2 默认角度和 H2 gain。
- H2 reference 必须来自 `reference/example_h2/deploy` 这类带 `h2_lora_reference_manifest.json` 的 H2 MotionLib export，不能复用 G1 `reference/example`。
- action 后处理仍走训练一致的 H2LoRA kinematic alignment：G1 semantic action -> H2 physical body action -> H2 DDS 31 motor command。
- 日志顺序需要区分：`lowcmd_command.csv` 与 `motor_temperature/motor_error/motor_torque` 是 H2 DDS order；`q.csv/dq.csv` 会从 29D policy-space state 重建为本地 H2 MuJoCo/training order；`action.csv` 保持 29D policy action。

本次补齐的 sim2real 安全/观测实现：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/error_monitor.hpp`
  - 原实现只接受 `std::array<uint32_t, G1_NUM_MOTOR>`，H2LoRA 路径在 `LowStateHandler` 中被直接跳过。
  - 现在改为 profile-aware，支持最多 `MAX_NUM_MOTOR=31`，并接收 DDS-order joint name 表。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`
  - 新增 `G1_JOINT_NAMES_DDS`，避免温度/错误报告在多个位置复制 G1 29 joint name。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
  - `LowStateHandler` 现在对 G1 扫描 29 个 motorstate，对 H2LoRA 扫描 31 个 motorstate。
  - 高温 hysteresis 从 29D 扩展到 31D，H2LoRA 会覆盖 head 两个电机。
  - F 键温度报告从 G1-only 改为 profile-aware；H2LoRA 会打印 31 个电机并播报 top-3 hottest motors。
  - `motor_temperature/motor_error/motor_torque` 日志按当前 profile 截断：G1 为 29/58，H2LoRA 为 31/62。
  - policy observation 的 `body_q/body_dq/last_action` 仍保持 29D，不改变模型输入契约。

sim2sim 与 sim2real 一致性复查：

一致部分：

- 两者都通过同一个 `g1_deploy_onnx_ref` 程序构造 LowCmd。
- 两者都使用 `--robot-profile h2_lora` 的同一套 observation layout 校验、encoder/decoder ONNX、action adapter 和 H2 DDS motor order。
- 两者 LowCmd writer 都按 `mode_pr/mode_machine + motor_cmd[i].mode/q/dq/kp/kd/tau + crc` 组包。
- Python MuJoCo bridge 在 sim 侧把官方 H2 DDS order 映射回本地 H2 MJCF order；真实 H2 则直接消费 DDS order。
- sim 与 real 都保持 29D policy body action，head 两电机 direct default command。

预期差异：

- sim 模式由 `deploy.sh sim` 自动追加 `--disable-crc-check`，因为 MuJoCo bridge 发布的 LowState 不需要真实硬件 CRC 校验；real 模式不追加，CRC 必须启用。
- sim 模式跳过 `MotionSwitcherClient`，因为 MuJoCo 没有真实高层运动服务；real 模式保留该 release step，和 G1 sim2real 一致。
- sim 模式有 MuJoCo 窗口、elastic band、接触/solver/runtime passive 参数；real 模式没有这些仿真专属机制。
- sim 模式需要单独运行 `gear_sonic/scripts/run_sim_loop.py --wbc-version h2_lora`；real 模式只运行 deploy 并连接 H2 网卡。

验证：

- `cd gear_sonic_deploy && source scripts/setup_env.sh && just build` 通过。
- `git diff --check` 通过。
