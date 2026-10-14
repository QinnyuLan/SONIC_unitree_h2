# H2 脚踝连杆虚位 MuJoCo 仿真

该仿真基于已经验证的 H2-native official `H2_loop.xml` sim2sim 链路，但不会修改
Unitree 原始模型、原 H2-native YAML、策略模型或部署配置。

## 磨损模型

- 左脚踝 roll 输出侧虚位：`-0.6 deg` 到 `+0.6 deg`。
- 右脚踝 roll 输出侧虚位：`-0.8 deg` 到 `+0.8 deg`。
- 左脚踝 pitch 输出侧虚位：`-0.2 deg` 到 `+0.2 deg`。
- 腰部 roll 连杆输出虚位：`-1.0 deg` 到 `+1.0 deg`。
- 腰部 pitch 连杆输出虚位：`-1.0 deg` 到 `+1.0 deg`。
- 髋部/骨盆 IMU pitch 报告偏置：`-1.0 deg`。IMU 少报 `1 deg` 前倾，促使闭环
  将真实 pelvis/重心向前补偿约 `1 deg`。
- 原 `left/right_ankle_roll_joint` 仍代表电机和编码器侧，并继续接收原控制力矩。
- 左脚踝原 pitch 闭链输出和连杆连接点保持不变，虚位位于闭链输出与实际脚部之间。
- 在原输出轴与实际脚部之间增加无执行器、无弹簧、无粘性阻尼的有限位被动轴。
- 被动轴在间隙内由脚部接触力和动力学自由运动，到端点后由 MuJoCo joint limit 承载。
- 腰部原 roll/pitch 输出关节、双连杆和 equality 约束保持不变；实际 torso 位于闭链
  输出后的两个无驱动虚位轴上，因此上半身可在 roll/pitch 方向分别自由摆动 `+/-1 deg`。
- IMU 零偏只作用于 LowState 发布的 pelvis quaternion；MuJoCo 真实姿态、odometry、
  torso IMU、关节状态和脚踝机械结构均不旋转。
- 策略、DDS 和部署端仍使用原来的 31 个电机关节，不会观察到额外的被动轴。

运行时从 official MJCF 生成临时副本并加入磨损轴；源文件不会被写入。参数位于：

`gear_sonic/utils/mujoco_sim/wbc_configs/h2_native_sonic_unitree_ros_loop_ankle_backlash.yaml`

## 运行

终端 1 启动带磨损的 MuJoCo：

```bash
cd /home/qingyul/workspace/SONIC-H2

.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py \
  --wbc-version h2_native_unitree_ros_loop_ankle_backlash \
  --no-data-collection
```

终端 2 使用原 H2-native 部署链路：

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh

./deploy.sh \
  --robot-profile h2_native \
  --input-type keyboard \
  --no-planner \
  --motion-data reference/example_h2/deploy \
  sim
```

deploy 显示 `Init Done` 后，按 `]` 进入 policy control，按 `T` 播放当前动作，按
`N`/`P` 切换动作，按 `O` 正常停止。MuJoCo 侧可用鼠标靠近脚踝观察接触载荷下的
roll/pitch 晃动。IMU 的 `-1 deg` 报告偏置不会直接改变模型几何形状，而是通过策略
的反向补偿体现为真实重心/机身前倾；各虚位角仍较小，建议放大观察。
