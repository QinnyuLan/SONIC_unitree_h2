# H2 + DFX/Inspire 遥操作 VLA 采集

本采集器只订阅 DDS、ZMQ 和 RGB 视频，不发布机器人或手部控制命令。控制沿用已经配置的 **GPU3 Stage2 5k、head pitch=40°、yaw=0°、Inspire 后端**。实机 Kp 仍由 `gear_sonic_deploy/config/h2_deploy.env` 控制，模型及头部由 `h2_head_adjust.env` 覆盖。

## 一次性配置

本机已建立 `.venv_data_collection` 并安装 PyArrow / PyAV。它复用 `.venv_teleop` 的 SDK、ZMQ 和支持 GStreamer 的 OpenCV；不安装会覆盖该 OpenCV 的 pip opencv 包。如需重建：

```bash
cd /home/qingyul/workspace/SONIC-H2
bash install_scripts/install_h2_data_collection.sh
```

编辑 `/home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/config/h2_data_collection.env`：

```bash
TASK_PROMPT='将桌面上的红色方块拿起，放入左侧盒子'
DATASET_ROOT=../datasets/h2_teleop
RECORD_FPS=50
RGB_WIDTH=1920
RGB_HEIGHT=1080
MIN_FREE_GB=2
```

任务描述必须准确对应本次演示。默认 `CHANGE_ME` 会阻止启动。长时间采集建议把 `DATASET_ROOT` 改到空间充足的数据盘；2026-09-19 检查时系统盘仅约 8 GB 可用。低于保留空间时当前 episode 标为 incomplete，采集不删除旧数据，也不会停止机器人。

如有相机内外参 JSON，将 `CAMERA_CALIBRATION` 设为该文件；采集器原样保存并标记 `provided_unverified`。没有文件就明确记录为未提供，**不会生成假的标定**。默认保存原始分辨率；若修改尺寸，训练时需按源/输出尺寸调整内参。

离线检查（不会连接机器人、PICO 或相机）：

```bash
bash gear_sonic_deploy/scripts/run_h2_data_collection.sh check
```

## 三个终端

先按已有的 H2 第一视角链路启用 PC1 的 `stereo_patch_pc1` 相机应用，将 original RGB 的 RTP/H264 流发到本机机器人网卡 IP（当前 `eno1` 的 `192.168.123.222`）**UDP 5002**。沿用已验证的相机配置；本采集器不远程启停 PC1 服务。

**关闭单独的 `h2_stereo_viewer.py` / 视频桥中对 UDP 5002 的接收进程**，由采集器接收、显示并录制。采集器启动时检查该端口是否被占用。PICO 和 XRoboToolkit 的追踪服务继续按原有方式连接。

终端 1，H2 控制（5k + 固定头部 + Inspire）：

```bash
cd /home/qingyul/workspace/SONIC-H2
bash gear_sonic_deploy/scripts/run_h2_data_collection.sh deploy
```

终端 2，PICO 遥操作：

```bash
cd /home/qingyul/workspace/SONIC-H2
bash gear_sonic_deploy/scripts/run_h2_data_collection.sh pico
```

终端 3，采集及第一视角预览：

```bash
cd /home/qingyul/workspace/SONIC-H2
bash gear_sonic_deploy/scripts/run_h2_data_collection.sh record
```

本次增加了 manager 的录制事件累计计数，**请重启旧的 PICO manager 进程**。它保留原来的布尔字段，兼容原官方 exporter；新采集器拒绝没有累计计数的旧 manager，防止错过一个按钮包就永久错乱。

## 操作与录制状态

1. 在 deploy 终端按 `y`，等待 InitDone。
2. PICO 按 `ABXY` 完成标定并进入 Planner；左摇杆按下进入 VR 三点模式，按现有流程对齐手臂；需要全身 POSE 时使用 `AX`。头部目标固定 pitch 40° / yaw 0°。
3. 双手 Trigger 各松开一次完成使能，然后用 Trigger 控制对应手开合。不是用摇杆控制手。采集器必须看到左右手真实位置、正常 lost 标志和真实 DDS 指令才能 READY。
4. 预览窗口和终端显示 **READY** 才开始。**左 Grip+A** 开始，窗口显示红色 **REC**，终端每秒显示帧数与时间。录制的是无文字叠加的第一视角画面。
5. 完成本次任务后，可先在采集窗口按 `s` 标记成功，`f` 标记失败，`u` 取消结果标记。这里只改标签，不停止录制。没有标记时 outcome 为 **unlabeled**，正常保存不自动等同于成功。
6. 再按 **左 Grip+A**：结束并保存，终端出现 SAVING → SAVED → READY。
7. 如果这条不需要，**录制过程中按左 Grip+B**：结束并标记 DISCARDED。为了可审计，文件保留，但训练必须排除；不能先 A 保存后再 B 丢弃上一条。
8. 窗口 `q` 或终端 Ctrl+C 退出采集器；若当时仍在 REC，该条标为 INCOMPLETE。**退出采集器不停止机器人**；机器人停止仍沿用 PICO ABXY / deploy `o` 的原流程。

WAIT 会列出缺失或过期的数据源。录制期间遇到手部离线、必要消息超时、模型不匹配、坏 CRC、多 DDS 发布者、写盘失败、严重采样卡顿或空间不足，该段自动标为 incomplete。正常网络恢复后可重新开始；不可恢复的采集线程/写盘错误需要先解决原因并重启采集器。不会把缺失的关键状态补零伪装成有效演示。

## 实际记录内容与结构

```text
datasets/h2_teleop/episode_YYYYMMDD_HHMMSS_xxxxxxxxx/
  metadata.json       任务、状态/结果、模型 SHA256、实际部署配置、源配置、标定及质量信息
  schema.json         字段顺序、单位、同步方式与不可用数据说明
  frames.parquet      每个采样时刻的状态、动作、IMU、token、遥操作输入与时间戳
  head_rgb.mp4        第一视角 RGB，H264，默认 1920×1080 / 50 fps 时间轴
  raw_zmq.msgpack     原始 pose/planner/head/manager/command/debug/config 接收消息流
```

异常断电或写盘错误可能留下 `.partial` 目录，它永远不是有效训练 episode。普通丢弃与可正常关闭的不完整片段保存到普通 episode 目录，须依据 metadata 的 status 过滤。

| 数据 | 字段与语义 |
|---|---|
| 身体+双手实测位置 | `observation.state` 共 43 维：H2 DDS 顺序 31 + 左手 6 + 右手 6。身体单位 rad，手单位为 0闭合～1张开的归一化执行器位置 |
| 实际发送目标 | `action` 同样 43 维，从 `rt/lowcmd` 和 `rt/inspire/cmd` 读取绝对位置目标；不拿旧 G1 debug 的变换后数值做 H2 标签 |
| 头部 | 上述两列中索引 29=head pitch、30=head yaw，实测值与目标值分别保存；不会把固定 40°冒充实测值 |
| 身体动力学/电机 | 实测 dq、tau_est、双温度传感器、motor_error、目标 dq/tau_ff/Kp/Kd/mode、mode_pr、tick |
| 手部 | 左右手 6 个真实执行器的位置、DDS 目标、lost。DDS 总线原来右手在前，落盘统一左手在前，并明确记录顺序 |
| 感知 | 身体 IMU 的 wxyz 四元数、角速度、加速度；secondary IMU 若有则保留 JSON 及独立时间戳 |
| SONIC 高层动作 | 64 维 `action.motion_token`、policy index |
| 遥操作 | `input.pose.json` 完整 SMPL 窗口；`input.planner.json` 完整 planner/VR 三点姿态；head、manager、模式/Trigger 等。原始接收字节也保留在 sidecar |
| 时间与质量 | 每源 arrival monotonic/wall ns、DDS source ns（如有）、接收序号、发布者、age_ms/fresh；每帧 host 时间、nominal timestamp 和 camera_frame_id |
| 任务与结果 | 每 episode 的自然语言任务，status=saved/discarded/incomplete；outcome=success/failure/unlabeled |
| 复现信息 | encoder/decoder 路径与 SHA256，deployment manifest（如有），部署时发布的 robot_config，头部/增益/采集配置文件内容 |

注意：

- 50 Hz 是统一数据时间轴。相机源约 15 Hz，所以有些行复用同一幅图，`camera_frame_id` 可识别；不会宣称采到了 50 Hz 独立相机画面。视频和表格行一一对应。
- 相机时间是本机解码完成后的到达时间，**不是曝光硬件时间戳，也不是硬件同步**。网络/编码带来的延迟尚未标定；DDS 不同机器的 source 时间也不能假定同一时钟。需要严格视觉动作同步时仍需做现场延迟测量或升级相机时间戳链路。
- 不会记录当前链路没有提供的深度、触觉/接触力、手部电流/温度/力矩、世界坐标真值。tau_est 是驱动估计值；手指目标是发给桥的 DDS 命令，不是串口执行确认。RGB + 本体状态 + 动作 + 语言可以用于常规 VLA，要求上述额外模态的训练任务仍需补传感器链路。
- 当前产物是有明确 schema 的 **`h2_vla_raw_v1` 原始数据集**，不是直接兼容官方 G1 feature schema 的 LeRobot 数据集。训练前需按 H2 43 维状态/动作或 64 维 motion token 目标建立对应的数据转换；无需重新采集这些已保存的字段。原始 sidecar 用于保留未展平的遥操作信息。

## 采集后检查

对整个目录检查视频帧数、Parquet 行数、时间轴、必需字段的有效性和原始消息可解码性：

```bash
cd /home/qingyul/workspace/SONIC-H2
.venv_data_collection/bin/python gear_sonic/scripts/inspect_h2_vla.py datasets/h2_teleop
```

也可以传单个 episode 目录。只有 `training_eligible: true` 的片段才是这份检查认为完整的正常保存片段；成功/失败筛选另看 outcome。该检查不能证明任务语义正确或现场同步延迟为零。

## 已验证与待现场确认

已做离线依赖检查、实际 Parquet/MP4/msgpack 写入回读、H2/手部索引与单位映射、PICO 计数事件丢包/重启、缺失/过期/坏数据、正常/丢弃/失败保存测试，以及头部控制回归。没有启动实机控制、发送手部命令或读取现场相机做硬件测试。首次现场使用先采一条短 episode 并执行上述检查，再开始批量采集。

## `writer queue full` 排查与 2026-09-19 修复

这表示写入队列满，不能再保证本条 episode 的原始消息完整，不能当作正常保存的训练样本。`WAIT frames=0` 是结束当前片段后回到等待状态，帧数计数已复位，并非先前保存的 episode 全部变成零帧。

2026-09-19 的 `episode_20260919_014040_166053520` 保存了 389 行，标记为 incomplete。该段约 7.8 秒内收到 1434 条 manager_state，异常突发填满 256 项的共用队列；当时剩余磁盘空间为 57.7 GB。代码中 planner 遇到未更新的 XRT 时间戳会立即返回，但外围 manager 仍无上限发布状态，是这次突发的来源。

已修复：manager_state 常规心跳最多 50 Hz，录制/丢弃按钮边沿立即发送；写入队列异常按 episode 隔离并去重，结束这条不完整片段后可重新开始，下一条不会继承旧错误。每条新增 writer_queue 容量/最高占用记录。没有通过丢掉必要消息并把片段标成成功来掩盖队列溢出。

正在运行的 Python 进程不会自动加载修改。先按原流程结束本轮机器人遥操作，再退出、重启 PICO 和采集器两个终端；deploy 参数及 PD 配置不需要修改。旧 incomplete 片段保留原标记，不能因修复而改成 saved。

修复经过离线消息突发、按钮保留、原始消息/视频帧两种队列失败后重新采集并保存的回归测试；没有替用户停止或重启正在运行的实机进程。
