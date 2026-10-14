# H2LoRA 训练指标说明

本文档解释当前 H2LoRA / SONIC G1 到 Unitree H2 迁移训练在 SwanLab 中记录的主要指标。内容对应当前代码实现，不是通用 PPO 教科书式说明；主要来源是 `gear_sonic/trl/trainer/ppo_trainer.py`、`gear_sonic/envs/wrapper/manager_env_wrapper.py`、`gear_sonic/envs/manager_env/mdp/commands.py`、`gear_sonic/envs/manager_env/mdp/rewards.py` 和 `gear_sonic/envs/manager_env/mdp/terminations.py`。

## 1. 先看哪几组曲线

如果只想快速判断训练是否健康，建议先看这几组：

1. `objective/rewards` 或 SwanLab 展示名 `Mean rewards`：整体 episode return，长期上升通常是好事。
2. `objective/length` 或 `Mean length`：episode 平均长度，越长说明越少提前终止。
3. `Env/step/h2lora/error_joint_pos_physical`、`Env/step/h2lora/error_hip_pos_physical`、`Env/step/h2lora/error_ankle_pos_physical`：关节 tracking 误差，越低越好。
4. `Env/Metrics/motion/error_body_ang_vel`、`Env/step/h2lora/root_error_ang_vel_norm`、`Env/step/h2lora/torso_link/robot_ang_vel_norm`：身体晃动和角速度 tracking，越低越稳。
5. `Env/Episode_Termination/foot_pos_xyz`、`Env/Metrics/motion/termination_foot_pos_exceeded`：足端漂移/足端提前终止，越低越好。
6. `policy/approxkl_avg`、`policy/clipfrac_avg`、`val/ratio`、`Policy/mean_noise_std`：PPO 更新是否过猛。
7. `debug/invalid_update_count`、`debug/invalid_grad_count`：数值稳定性，理想情况下保持 0。
8. `gpu/utilization_pct`、`gpu/power_w`、`fps`：资源利用率和吞吐。

注意：代码里会同时写入带 `Env/` 前缀和不带 `Env/` 前缀的 env 指标。SwanLab 里如果看到两条名字类似的曲线，优先看 `Env/...` 版本；无前缀版本多半是同一份 env log 的重复入口。

## 2. 训练进度与吞吐

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `it` | PPO iteration 序号。当前每个 iteration 收集 `num_envs * num_steps_per_env` 个环境步。3M 续训中是 `6144 * 24 = 147456` env steps/iteration。 | 单调增加。 |
| `tot_timesteps` | 累计环境步数。 | 衡量真实采样规模，比 iteration 更直观。 |
| `episode` 或 `Total episodes` | trainer 内部累计 episode 计数。这里按 batch 规模累加，不完全等于真实完成 episode 数。 | 主要用于进度，不建议拿它判断学习质量。 |
| `fps` 或终端里的 `Computation` | 每秒处理的环境步数，包含采样和学习。 | 越高说明吞吐越好；但如果 reward 恶化，不能只追 fps。 |
| `collection_time` | rollout 采样耗时。 | 太高通常是仿真/环境瓶颈。 |
| `learn_time` | PPO 反向传播和 optimizer step 耗时。 | 太高通常是模型/mini-batch/GPU 训练瓶颈。 |
| `tot_time` | 累计训练耗时，秒。 | 仅进度统计。 |
| `batch_idx`、`num_total_batches` | 当前第几个 batch / 总 batch 数。 | 用来估算 ETA。 |
| `eps` | 代码按 `state.episode / elapsed_time` 计算的 episode/s。 | 这个值受 trainer 的 episode 计数定义影响，参考意义弱于 `fps`。 |

本轮 3M 续训主要看 `fps`、`collection_time`、`learn_time`。如果 `collection_time` 比 `learn_time` 大很多，说明主要卡 Isaac Sim 环境；如果 `learn_time` 大很多，说明 GPU 训练部分更重。

## 3. PPO 学习指标

| 指标 | 含义 | 趋势判断 |
| --- | --- | --- |
| `objective/rewards` / `Mean rewards` | 最近完成 episode 的平均累计 reward。 | 长期上升通常好；短期震荡正常。idle 数据中如果长期下降，说明 tracking 或 termination 仍有问题。 |
| `objective/length` / `Mean length` | 最近 episode 平均长度。 | 越长越好；如果很低，通常是很快触发 termination，训练信号会差。 |
| `objective/entropy` / `loss/entropy_avg` / `Mean entropy` | policy action distribution 的 entropy。 | 过高说明动作太随机，过低说明探索不足或 std 被压太小。本轮冻结/限制 std 后应相对稳定。 |
| `Policy/mean_noise_std` / `Mean action noise std` | policy 动作噪声标准差均值。 | H2LoRA 当前冻结/限制 std，目标是避免 std 爆炸或变成 NaN。突然变大或非 finite 是危险信号。 |
| `policy/approxkl_avg` | 新旧 policy 的近似 KL。 | 越大说明一次更新改动越猛。太大容易发散；接近 0 且 reward 不变则可能学习太慢。 |
| `policy/clipfrac_avg` | PPO ratio 被 clip 的比例。 | 高说明更新经常越界，可能学习率过大或 advantage 波动大；长期接近 0 可能更新太保守。 |
| `loss/policy_avg` | PPO policy loss 均值。 | 绝对值不如趋势有意义；突然尖峰常对应 bad batch 或 reward/advantage 异常。 |
| `loss/value_avg` | critic value loss。 | 应逐步稳定或下降；持续升高说明 critic 预测 return 困难，可能 termination 太混乱或 reward 尺度不稳。 |
| `loss/weighted_ppo_loss_avg` | policy/value/entropy 等组合后的总 PPO loss。 | 用于发现整体训练是否出现尖峰；不建议单独判断好坏。 |
| `val/ratio` | PPO probability ratio 均值，即新 policy 概率 / 旧 policy 概率。 | 理想附近是 1。明显偏离 1 说明更新过大。 |
| `val/ratio_var` | ratio 方差。 | 越大说明不同样本更新幅度差异大，可能不稳定。 |
| `val/clipfrac_avg` | value clipping 触发比例。 | 高说明 critic 更新被大量裁剪，可能 value 学得不稳。 |
| `val/advantage_mean` | mini-batch advantage 均值。 | GAE/归一化后应接近 0。明显漂移需要查 return 或 normalization。 |
| `val/advantage_std` | mini-batch advantage 标准差。 | 应在合理范围内；过大代表训练信号噪声大。 |
| `lr` | 当前 trainer 学习率字段。 | 当前 actor/critic 实际可有不同 lr group，因此这个字段只作参考。 |

H2LoRA 当前是 LoRA 适配，不做全参数微调，所以 PPO 指标要结合 tracking 指标一起看。reward 上升但 `error_body_ang_vel` 同时升高，可能是 policy 找到了“能活但晃”的策略，不算理想。

## 4. 数值稳定性指标

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `debug/invalid_update_count` | 检测到 PPO 输入/输出或 loss 非 finite 后跳过 optimizer step 的次数。 | 理想为 0。偶发增加说明 NaN 防线生效，但仍要追来源。连续增加代表训练已经不健康。 |
| `debug/invalid_grad_count` | 检测到 grad 或 grad norm 非 finite 后跳过 optimizer step 的次数。 | 理想为 0。若增加，通常需要降学习率、收紧 action/std 或检查 reward/obs。 |
| 日志中的 `Invalid PPO update detected` | 对应 invalid update skip。 | 出现一次就应留意；密集出现需要暂停分析。 |
| 日志中的 `NaN in gradient! Skipped!!!!` | grad 非 finite 或 global grad norm 非 finite。 | 这是红灯。 |

这两个 debug 指标是本轮为修复 NaN 加的工程防线，不属于 Any2Any 论文方法本身。它们的目标是防止一个坏 batch 污染 LoRA 权重。

## 5. Reward 分项指标

这些指标在 SwanLab 中通常以 `Env/Episode_Reward/...` 出现，表示 episode 维度累计/平均后的 reward term。正权重 tracking 项越高越好，负权重 penalty 项越接近 0 越好。注意它们已经乘过配置里的 reward weight，所以数值正负取决于 term 权重。

| 指标 | 当前含义 | 趋势判断 |
| --- | --- | --- |
| `Env/Episode_Reward/tracking_anchor_pos` | root/anchor 位置 tracking reward，Gaussian kernel。 | 越高越好。低说明根部位置漂。 |
| `Env/Episode_Reward/tracking_anchor_ori` | root/anchor 朝向 tracking reward。 | 越高越好。低且 `anchor_ori_full` termination 高时，说明身体朝向控制差。 |
| `Env/Episode_Reward/tracking_relative_body_pos` | 各 tracked body 相对 anchor 的位置 tracking reward。 | 越高越好。反映整体姿态位置是否贴合 ref。 |
| `Env/Episode_Reward/tracking_relative_body_ori` | 各 tracked body 相对 anchor 的姿态 tracking reward。 | 越高越好。 |
| `Env/Episode_Reward/tracking_body_linvel` | tracked body 线速度 tracking reward。 | 越高越好。 |
| `Env/Episode_Reward/tracking_body_angvel` | tracked body 角速度 tracking reward。 | 越高越好；但要和晃动 penalty 一起看。 |
| `Env/Episode_Reward/tracking_vr_5point_local` | 5 点局部 tracking reward。H2LoRA 当前配置中 reward point 主要是 torso、左右 wrist。 | 越高越好。对上身/手腕/torso 跟踪很敏感。 |
| `Env/Episode_Reward/action_rate_l2` | action 变化惩罚。 | 是负项，越接近 0 越平滑；更负说明动作抖。 |
| `Env/Episode_Reward/joint_limit` | 关节限位惩罚。 | 越接近 0 越好；明显为负说明关节经常撞限位。 |
| `Env/Episode_Reward/undesired_contacts` | 非期望接触惩罚。 | 越接近 0 越好；更负表示身体/手臂等不该接触的部位接触环境。 |
| `Env/Episode_Reward/anti_shake_ang_vel` | 超过阈值的 selected body 角速度惩罚。当前用于抑制 torso/wrist 晃动。 | 越接近 0 越稳；更负说明晃动或快速甩动严重。 |
| `Env/Episode_Reward/body_ang_vel_l2` | selected body 角速度 L2 惩罚。当前主要盯 torso。 | 越接近 0 越好。 |
| `Env/Episode_Reward/feet_acc` | 脚踝相关 joint acceleration 惩罚。 | 越接近 0 越平滑；更负说明脚踝抖动/高频动作。 |

看 reward 分项时，建议不要只看总 reward。比如 `tracking_body_angvel` 可能变好，但 `anti_shake_ang_vel` 也变得更负，表示模型为了追角速度开始抖；这不是我们想要的稳定 tracking。

## 6. Motion tracking 误差指标

这些指标来自 `TrackingCommand._update_metrics()`，通常以 `Env/Metrics/motion/...` 记录。多数是物理单位的误差，越低越好。

| 指标 | 单位/含义 | 怎么看 |
| --- | --- | --- |
| `Env/Metrics/motion/error_anchor_pos` | anchor/root 位置误差，单位 m。 | 越低越好；高说明根部漂移。 |
| `Env/Metrics/motion/error_anchor_rot` | anchor/root 姿态误差，单位 rad。 | 越低越好；高时容易触发 `anchor_ori_full`。 |
| `Env/Metrics/motion/error_anchor_lin_vel` | anchor/root 线速度误差，单位 m/s。 | 越低越好。 |
| `Env/Metrics/motion/error_anchor_ang_vel` | anchor/root 角速度误差，单位 rad/s。 | 越低越好；高通常对应身体转动/晃动。 |
| `Env/Metrics/motion/error_body_pos` | tracked body 平均位置误差，单位 m。 | 越低越好。全身姿态位置核心指标。 |
| `Env/Metrics/motion/error_body_rot` | tracked body 平均姿态误差，单位 rad。 | 越低越好。 |
| `Env/Metrics/motion/error_body_lin_vel` | tracked body 平均线速度误差。 | 越低越好。 |
| `Env/Metrics/motion/error_body_ang_vel` | tracked body 平均角速度误差。 | 越低越好；本次判断“身体频繁晃动”的核心指标之一。 |
| `Env/Metrics/motion/error_joint_pos` | joint position 平均绝对误差。H2LoRA 下等同 physical joint space。 | 越低越好。 |
| `Env/Metrics/motion/error_joint_vel` | joint velocity 平均绝对误差。 | 越低越好。 |
| `Env/Metrics/motion/error_joint_pos_physical` | H2 physical 29DoF body joint position 误差。 | H2LoRA 最重要的关节 tracking 指标，越低越好。 |
| `Env/Metrics/motion/error_joint_vel_physical` | H2 physical joint velocity 误差。 | 越低越好；高说明动态 tracking 差。 |
| `Env/Metrics/motion/error_joint_pos_semantic` | 把 H2 状态映射回 G1 semantic space 后的 joint position 误差。 | 用于看 Any2Any 语义对齐后的误差，越低越好。 |
| `Env/Metrics/motion/error_joint_vel_semantic` | G1 semantic space joint velocity 误差。 | 越低越好。 |
| `Env/Metrics/motion/error_hip_pos_physical` | hip 相关关节 physical position 误差。 | 越低越好；高时优先怀疑 hip 局部线性映射或动作尺度。 |
| `Env/Metrics/motion/error_hip_vel_physical` | hip 相关关节 velocity 误差。 | 越低越好。 |
| `Env/Metrics/motion/error_ankle_pos_physical` | ankle 相关关节 position 误差。 | 越低越好；高时优先怀疑 ankle 顺序/coupling/足端 termination。 |
| `Env/Metrics/motion/error_ankle_vel_physical` | ankle 相关关节 velocity 误差。 | 越低越好；高时通常伴随脚踝抖动。 |

`Env/Metrics/motion/...` 是 episode/log 窗口聚合后的指标；如果 episode 很短，它会被频繁失败片段主导。因此要和下一节的 step-level 指标一起看。

## 7. Step-level 指标

`Env/step/...` 是每个 rollout step 通过 `to_log` 累积平均出来的诊断项，不只统计完成 episode。它对早期训练更有用，因为 early termination 很频繁时，episode summary 容易偏。

| 指标 | 含义 | 趋势判断 |
| --- | --- | --- |
| `Env/step/Metrics/motion/error_anchor_pos` | step 平均 anchor 位置误差。 | 越低越好。 |
| `Env/step/Metrics/motion/error_body_pos` | step 平均 body 位置误差。 | 越低越好。 |
| `Env/step/Metrics/motion/error_joint_pos` | step 平均 joint position 误差。 | 越低越好。 |
| `Env/step/Metrics/motion/error_joint_vel` | step 平均 joint velocity 误差。 | 越低越好。 |
| `Env/step/Metrics/motion/termination_foot_pos_max` | step 中足端位置误差最大值。 | 越低越好；接近或超过 threshold 时会频繁终止。 |
| `Env/step/Metrics/motion/termination_foot_pos_mean` | 左右足端位置误差均值。 | 越低越好。 |
| `Env/step/Metrics/motion/termination_foot_pos_exceeded` | step 中足端误差是否超过阈值的比例。 | 越接近 0 越好。 |
| `Env/step/Metrics/motion/termination_ee_body_height_max` | 足/腕等末端高度误差最大值。 | 越低越好。 |
| `Env/step/Metrics/motion/termination_ee_body_height_mean` | 足/腕等末端高度误差均值。 | 越低越好。 |
| `Env/step/Metrics/motion/termination_ee_body_height_exceeded` | 末端高度误差超过阈值的比例。 | 越接近 0 越好。 |

如果 `Env/step/...` 在下降，但 `Env/Metrics/...` 或 `Episode_Termination/...` 不下降，说明模型在普通 step 上变好，但仍被少数困难片段或 termination 门槛卡住。这个时候要看 adaptive sampling 是否过度集中到困难 bin。

## 8. H2LoRA 专项指标

这些是本次为了 G1 policy 迁移 H2 新加的诊断项，代码在 `ManagerEnvWrapper._append_h2lora_step_diagnostics()`。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `Env/step/h2lora/policy_action_rate_l2` | 当前 29D policy raw action 与上一步 raw action 的均方变化。 | 越低越平滑。下降通常表示动作不再高频抖。过低但 tracking 差可能说明 policy 太保守。 |
| `Env/step/h2lora/policy_action_abs_mean` | 29D policy raw action 绝对值均值。 | 用于看动作幅度。突然变大可能是发散或 action scale 不合适。 |
| `Env/step/h2lora/robot_joint_vel_abs_mean` | H2 body joint velocity 绝对值均值。 | 越低越稳，但不能低到无法 tracking。晃动时这个通常升高。 |
| `Env/step/h2lora/error_joint_pos_physical` | H2 physical body joint position step 误差。 | 主要 tracking 指标，越低越好。 |
| `Env/step/h2lora/error_joint_vel_physical` | H2 physical body joint velocity step 误差。 | 越低越好。 |
| `Env/step/h2lora/error_joint_pos_semantic` | H2 状态映射回 G1 semantic space 后的 step joint position 误差。 | 用于看语义对齐是否变好，越低越好。 |
| `Env/step/h2lora/error_hip_pos_physical` | hip 子集 physical position step 误差。 | 越低越好；本项目里 hip 是局部线性映射重点。 |
| `Env/step/h2lora/error_hip_vel_physical` | hip 子集 velocity step 误差。 | 越低越好。 |
| `Env/step/h2lora/error_ankle_pos_physical` | ankle 子集 position step 误差。 | 越低越好；用于检查 H2 ankle 与 G1 映射差距。 |
| `Env/step/h2lora/error_ankle_vel_physical` | ankle 子集 velocity step 误差。 | 越低越好；高时常伴随足端抖动。 |
| `Env/step/h2lora/error_feet_pos` | 左右 foot body position tracking error。 | 越低越好；和 `foot_pos_xyz` termination 直接相关。 |
| `Env/step/h2lora/torso_link/robot_ang_vel_norm` | torso 实际角速度范数。 | 越低越稳；idle 数据里尤其应该低。 |
| `Env/step/h2lora/torso_link/ref_ang_vel_norm` | reference torso 角速度范数。 | 用来判断 ref 本身是否在动/抖；idle 上应接近 0。 |
| `Env/step/h2lora/torso_link/error_ang_vel_norm` | torso 角速度 tracking 误差。 | 越低越好；高就是身体晃动核心信号。 |
| `Env/step/h2lora/left_wrist_yaw_link/robot_ang_vel_norm` / `right_wrist_yaw_link/robot_ang_vel_norm` | 左右腕末端实际角速度范数。 | 高说明上肢末端快速甩动，常和视频里的上半身晃动同时出现。 |
| `Env/step/h2lora/left_wrist_yaw_link/ref_ang_vel_norm` / `right_wrist_yaw_link/ref_ang_vel_norm` | 左右腕 reference 角速度范数。 | 如果 ref 很低但 robot 很高，说明晃动主要来自 policy/control，而不是数据本身。 |
| `Env/step/h2lora/left_wrist_yaw_link/error_ang_vel_norm` / `right_wrist_yaw_link/error_ang_vel_norm` | 左右腕角速度 tracking 误差。 | 越低越好；高时需要加强上肢平滑或调低对应 tracking/探索冲突。 |
| `Env/step/h2lora/left_ankle_roll_link/robot_ang_vel_norm` | 左脚踝 link 实际角速度范数。 | 高说明脚部旋转剧烈，可能对应足端不稳。 |
| `Env/step/h2lora/right_ankle_roll_link/robot_ang_vel_norm` | 右脚踝 link 实际角速度范数。 | 同上。 |
| `Env/step/h2lora/left_ankle_roll_link/ref_ang_vel_norm` / `right_ankle_roll_link/ref_ang_vel_norm` | 左右脚踝 reference 角速度范数。 | 用来区分 ref 里脚是否真的在快速旋转。 |
| `Env/step/h2lora/left_ankle_roll_link/error_ang_vel_norm` | 左脚踝角速度误差。 | 越低越好。 |
| `Env/step/h2lora/right_ankle_roll_link/error_ang_vel_norm` | 右脚踝角速度误差。 | 越低越好。 |
| `Env/step/h2lora/root_robot_ang_vel_norm` | root 实际角速度范数。 | idle 上应逐步变低；高表示整体身体旋转/摇晃。 |
| `Env/step/h2lora/root_ref_ang_vel_norm` | reference root 角速度范数。 | 判断 ref 根部本身是否在转动；idle 上应接近 0。 |
| `Env/step/h2lora/root_error_ang_vel_norm` | root 角速度 tracking 误差。 | 越低越好。 |
| `Env/step/h2lora/error_head_default_pos` | H2 head 两个自由度与默认角度的平均绝对误差。 | 应始终接近 0。非 0 说明 head lock/direct command 出问题。 |

本次你关心“身体频繁晃动”，优先组合看：

1. `Env/step/h2lora/policy_action_rate_l2`
2. `Env/step/h2lora/robot_joint_vel_abs_mean`
3. `Env/step/h2lora/root_robot_ang_vel_norm`
4. `Env/step/h2lora/torso_link/robot_ang_vel_norm`
5. `Env/Metrics/motion/error_body_ang_vel`
6. `Env/Episode_Reward/anti_shake_ang_vel`
7. `Env/Episode_Reward/body_ang_vel_l2`

如果这些下降，同时 `error_joint_pos_physical` 没有恶化，说明 anti-shake 调参有效。如果晃动指标下降但 tracking error 大幅上升，说明抑制过强，policy 变“僵”。

## 9. Termination 指标

| 指标 | 含义 | 趋势判断 |
| --- | --- | --- |
| `Env/Episode_Termination/time_out` | motion 正常播放到片段结束而终止的比例/平均值。 | 越高通常越好，表示不是因为失败提前结束。 |
| `Env/Episode_Termination/anchor_pos` | root/anchor 高度或位置相关 termination。当前 strict 配置阈值较紧。 | 越低越好。 |
| `Env/Episode_Termination/anchor_ori_full` | anchor/root 朝向误差超过阈值。 | 越低越好；高时说明整体朝向/姿态失控。 |
| `Env/Episode_Termination/ee_body_pos` | 末端 body 高度误差过大，包含脚/腕。 | 越低越好。 |
| `Env/Episode_Termination/foot_pos_xyz` | 左右脚位置误差过大。当前 strict 配置中 threshold 是 0.2m。 | 越低越好；这是当前 H2LoRA idle 训练最需要盯的失败来源之一。 |
| `Env/Metrics/motion/termination_foot_pos_max` | 足端误差最大值，单位 m。 | 低于 threshold 较安全。 |
| `Env/Metrics/motion/termination_foot_pos_mean` | 足端误差均值，单位 m。 | 越低越好。 |
| `Env/Metrics/motion/termination_foot_pos_exceeded` | 足端超过阈值比例。 | 越接近 0 越好。 |
| `Env/Metrics/motion/termination_ee_body_height_max` | 末端高度误差最大值。 | 越低越好。 |
| `Env/Metrics/motion/termination_ee_body_height_mean` | 末端高度误差均值。 | 越低越好。 |
| `Env/Metrics/motion/termination_ee_body_height_exceeded` | 末端高度超过阈值比例。 | 越接近 0 越好。 |

如果 `Mean length` 低，同时某个 `Episode_Termination/...` 高，训练主要问题不是 PPO 本身，而是环境很快失败，导致有效学习片段短。

## 10. Adaptive sampling 指标

MotionLib 开启 adaptive sampling 后，会把更容易失败的 motion/time bin 抽得更频繁。这能加速学习难点，但也可能让训练过早集中在少数特别难的片段。

| 指标 | 含义 | 趋势判断 |
| --- | --- | --- |
| `Env/adp_samp/num_episodes_min` | 每个采样 bin 被采到的 episode 次数最小值。 | 太低说明覆盖还不均。 |
| `Env/adp_samp/num_episodes_max` | 某个 bin 被采到的 episode 次数最大值。 | 很高说明采样集中。 |
| `Env/adp_samp/num_episodes_mean` | bin 平均采样次数。 | 随训练增加。 |
| `Env/adp_samp/num_failures_min/max/mean` | 每个 bin 的失败次数统计。 | max 很高说明存在困难片段。 |
| `Env/adp_samp/failure_rate_min/max/mean` | 每个 bin 的失败率统计。 | mean 下降是好事；max 长期很高说明有卡点。 |
| `Env/adp_samp/prob_max` | 当前最大采样概率。 | 越高说明越集中到少数 bin。 |
| `Env/adp_samp/prob_min` | 当前最小采样概率。 | 接近 0 说明某些 bin 很少被采。 |
| `Env/adp_samp/prob_mean` | 平均采样概率，约等于 `1 / bin 数`。 | 主要用于 sanity check。 |
| `Env/adp_samp/prob_max_over_uniform` | 最大采样概率相对均匀采样的倍数。 | 过高说明采样高度集中；当前配置允许较大集中度。 |
| `Env/adp_samp/effective_num_bins` | 按概率计算的有效 bin 数。 | 越高覆盖越广，越低越集中。 |
| `Env/adp_samp/num_concentrated_bins` | 采样概率超过均匀 10 倍的 bin 数。 | 高说明许多困难片段被放大。 |
| `Env/adp_samp/episodes_max_over_mean` | 最大 episode 数 / 平均 episode 数。 | 高说明采样历史不均。 |

如果 `foot_pos_xyz` termination 很高，同时 `prob_max_over_uniform` 和 `num_concentrated_bins` 也升高，说明 adaptive sampling 正在抓足端失败片段。这不一定坏，但需要用视频确认这些片段是否真实可学，还是数据/retarget/映射有系统性错误。

## 11. Cache、LoRA 与初始化指标

这些一般只在 SwanLab step 0 或启动日志中出现。

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `cache/enabled` | MotionLib DDR cache 是否开启。 | 1 表示开启。 |
| `cache/loaded_count` | 已加载进 DDR cache 的 motion 数。 | 应接近数据集 motion 数；3M idle 是 5907。 |
| `cache/loaded_gb` | DDR cache 实际加载大小，GB。 | 用于确认没有频繁磁盘 IO。 |
| `cache/max_gb` | DDR cache 上限，GB。 | 本轮约 49.39GB。 |
| `lora/trainable_params` | LoRA 可训练参数数量。 | 应显著小于总参数。 |
| `lora/total_params` | 模型总参数数量。 | 参考值。 |
| `lora/trainable_ratio` | 可训练参数比例。 | 当前配置要求低于 `0.08`，即 8%。 |
| 启动日志 `Optimizer trainable params` | optimizer 实际收到的可训练参数数，以及 actor/critic 拆分。 | 用于确认没有把冻结参数加进 optimizer。 |

如果 `lora/trainable_ratio` 异常偏高，说明 LoRA/freeze 范围可能错了；如果为 0，说明没有任何参数在训练。

## 12. GPU 与资源指标

| 指标 | 含义 | 怎么看 |
| --- | --- | --- |
| `gpu/memory_allocated_gb` | PyTorch 当前 allocated 显存。 | 比 `nvidia-smi` used 低是正常的。 |
| `gpu/memory_reserved_gb` | PyTorch caching allocator reserved 显存。 | 会高于 allocated。 |
| `gpu/max_memory_allocated_gb` | PyTorch 运行以来最大 allocated 显存。 | 用于判断峰值。 |
| `gpu/utilization_pct` | `nvidia-smi` GPU 利用率。 | 越高说明 GPU 忙；训练中 80%+ 较好。 |
| `gpu/memory_utilization_pct` | `nvidia-smi` memory controller 利用率。 | 高说明显存带宽压力大。 |
| `gpu/nvidia_smi_memory_used_gb` | `nvidia-smi` 看到的总显存占用。 | 包含 PyTorch reserved、Isaac Sim、图形/驱动上下文等。 |
| `gpu/nvidia_smi_memory_total_gb` | GPU 总显存。 | 5090 当前约 32.6GB。 |
| `gpu/power_w` | GPU 实时功耗。 | 高说明负载吃上去了；会随采样/学习阶段波动。 |
| `gpu/power_limit_w` | GPU 功耗上限。 | 用于判断还有没有功耗余量。 |
| `gpu/temperature_c` | GPU 温度。 | 过高可能降频。 |

外部脚本还会写同名 `*_gpu.csv`，它比 SwanLab 的 iteration 级采样更密，适合看采样/学习阶段的周期性波动。

## 13. 当前 H2LoRA 训练的阅读模板

每次看一个阶段 checkpoint 前，建议按下面顺序读：

1. 数值是否安全：`debug/invalid_update_count=0`、`debug/invalid_grad_count=0`，日志无 NaN/OOM。
2. 是否活得更久：`objective/length` 上升，`time_out` 上升，提前 termination 下降。
3. 是否 tracking 变好：`error_joint_pos_physical`、`error_body_pos`、`error_feet_pos` 下降。
4. 是否不晃：`policy_action_rate_l2`、`robot_joint_vel_abs_mean`、`root_robot_ang_vel_norm`、`torso_link/robot_ang_vel_norm` 下降或稳定。
5. 是否足端仍是主失败源：`foot_pos_xyz`、`termination_foot_pos_exceeded` 是否下降。
6. 是否 PPO 更新正常：`approxkl` 不爆，`ratio` 接近 1，`clipfrac` 不长期过高。
7. 是否 adaptive sampling 过于集中：`prob_max_over_uniform`、`effective_num_bins`、`num_concentrated_bins`。
8. 最后看视频：同一 ref motion 下比较不同 checkpoint 的 ref/tracking 差异。

当前 3M 续训早期的现象是：数值稳定和 GPU 吞吐已经正常，但 episode-level `foot_pos_xyz` 和 `error_body_ang_vel` 仍需要继续观察。也就是说，现在还不能只看 reward 就判断成功，必须等 1000/5000/10000 checkpoint 后结合视频和分项曲线判断。

## 14. 常见诊断组合

### Reward 上升，但机器人仍晃

重点看：

1. `Env/step/h2lora/root_robot_ang_vel_norm`
2. `Env/step/h2lora/torso_link/robot_ang_vel_norm`
3. `Env/step/h2lora/policy_action_rate_l2`
4. `Env/Episode_Reward/anti_shake_ang_vel`
5. `Env/Episode_Reward/body_ang_vel_l2`

如果这些没降，说明 reward 上升来自其他 tracking 项，抗晃动还没学到。

### Joint tracking 变好，但 episode length 不涨

重点看：

1. `Env/Episode_Termination/foot_pos_xyz`
2. `Env/Episode_Termination/ee_body_pos`
3. `Env/Episode_Termination/anchor_ori_full`
4. `Env/step/Metrics/motion/termination_foot_pos_max`

这通常说明某个 termination 仍然太容易触发，或者某些困难 motion 片段系统性失败。

### Semantic error 低，physical error 高

重点看：

1. `Env/step/h2lora/error_joint_pos_semantic`
2. `Env/step/h2lora/error_joint_pos_physical`
3. `Env/step/h2lora/error_hip_pos_physical`
4. `Env/step/h2lora/error_ankle_pos_physical`

这种情况说明 G1 semantic space 看似对齐，但 H2 physical 执行仍有 gap，优先检查 hip/ankle 映射、action scale 和 retargeted H2 数据。

### GPU 利用率低

重点看：

1. `collection_time` vs `learn_time`
2. `fps`
3. `gpu/utilization_pct`
4. `gpu/power_w`
5. `cache/loaded_gb`

如果 collection 慢，可能是 env 数不足、Isaac Sim 或 CPU/DDR cache 瓶颈；如果 learn 慢，可能是 batch/mini-batch 或模型训练瓶颈。

### NaN 防线被触发

重点看：

1. `debug/invalid_update_count`
2. `debug/invalid_grad_count`
3. `Policy/mean_noise_std`
4. `policy/approxkl_avg`
5. `val/ratio_var`
6. `loss/value_avg`

如果 debug 指标连续增加，应优先降低 actor/critic lr、收紧 `max_grad_norm`、检查 obs/reward 是否有非 finite，并确认没有加载被 NaN 污染的 checkpoint。
