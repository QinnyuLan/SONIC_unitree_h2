# H2 5k 遥操作头部取景调角

2026-09-19 后续更新：用户已选定固定 pitch **40°**、yaw **0°**。
`h2_head_adjust.env` 当前使用 `HEAD_CONTROL=fixed`，原来的 deploy/pico 两个
启动指令继续适用；右摇杆及头显不再改变头部目标。改回 `right-stick` 才启用
下文调角流程。PICO 启动策略后打印 `[HeadFixed] pitch_target=+40.00 deg ...`。
固定模式仍保留原有新鲜帧检查和部署端消息超时处理。

同时检查灵巧手：启动入口明确传递 `--hand-backend inspire`。手部使用
食指 Trigger 扳机控制开合（不是摇杆，也不是侧面 Grip）：先完全松开以使能，
再按下关闭、松开打开；当前为整手开合映射。控制端成功使能会打印
`[InspireHands] Left/Right Trigger input armed`。
00:27 左右只读 DDS 检查收到连续 `rt/inspire/state`，但双手均为 OFFLINE、
12 个 q 为 NaN，lost 约 2082–2091；此时控制端会阻止手命令发布。
这说明当时反馈链路仍在发布，而双手设备反馈无效，不能据此断言此前整段
遥操作的设备状态。需检查 PC2 串口/供电与手部连接，再用只读监视器确认 ONLINE。

随后在 PC2 只读确认：`h2-inspire-hands.service` 自 00:10:06 起 active，
使用 right if01→ttyUSB1、left if02→ttyUSB2；FTDI 的四个串口均存在。
服务日志左右均持续报告 `position read failed: response timeout or wrong length`。
目前证据指向串口到手设备的响应异常；需要检查手部供电、RS-485 线缆/接口，
不能仅凭超时区分供电、接线、地址/波特率或设备故障。本次没有重启服务或发送手命令。

```bash
cd /home/qingyul/workspace/SONIC-H2
.venv_teleop/bin/python gear_sonic_deploy/scripts/h2_inspire_hand_dds_demo.py \
  --network eno1 --monitor-seconds 5
```

固定头部和现有头部/摇杆/Planner 相关 16 项测试通过；未发送实机运动或手部开合命令。

2026-09-19。用于选择固定的头部相机 pitch。专用启动脚本加载 GPU3 Stage2
5k（5000 轮），使用当前实机增益、`inspire` 手部后端，不开启 CSV 或数据集录制。

## 启动

完整第一视角流程继承会话
`codex://threads/019f8e43-59cf-7ad0-91ab-ef5ad1c55bed` 的最终结果：使用
`stereo patch pc1` 的 UDP/5002 原图，在电脑 OpenCV 窗口查看。历史实测
1920×1080、约 15 FPS；左右目 UDP/5004、5006 为各 544×448、约 10 FPS。
这不是把视频显示到 PICO 内的 Remote Vision 流程。

### 先确认相机画面

在 Unitree App 中关闭 `video_hub`、开启 `stereo patch pc1`，推流目标设为
工作站 `eno1` IPv4。2026-09-19 本机检查值是 `192.168.123.222`。
PC1 不对开发者开放，不需要 SSH 登录 PC1。

相机终端：

```bash
cd /home/qingyul/workspace/SONIC-H2
.venv_teleop/bin/python gear_sonic_deploy/scripts/h2_stereo_viewer.py \
  --rgb-mode original --no-depth
```

先看到连续更新的第一视角，再启动下面的身体控制终端。可以拖动窗口边缘放大；
显示窗口初始缩小不改变接收帧的分辨率。若需要左右目和深度检查，替换参数为
`--rgb-mode all --depth`，不要同时启动两个抢占同一 UDP 接收端口的查看器。

PICO 的 XRoboToolkit `PC Service` 地址应填工作站的 Wi-Fi IP；2026-09-19
检查值为 `192.168.56.232`。这与机器人相机的有线推流目标地址不同。
确认 PICO 显示 WORKING，Head/Controller 启用，Data/Control 为 Send，
Full body 跟踪已校准。IP 变更后以 `ip -4 -brief addr` 为准。

当日检查确认 OpenCV 4.13.0 / GStreamer YES、XRoboToolkit PC 服务正在运行；
但有界相机接收测试没有解码到首帧，因此本次未确认机器人相机当前正在出图。
历史成功结果不能替代当前画面检查。黑屏时先检查 App 服务开关和推流目标，
不要通过启动身体控制来尝试恢复相机。

### 再启动控制与 PICO

先停止旧的部署控制器和 PICO Manager。DFX/Inspire 手服务沿用 PC2 上的
`h2-inspire-hands.service`。两个终端：

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh deploy
```

```bash
bash /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy/scripts/run_h2_teleop_head_adjust.sh pico
```

部署端按 `y` 确认，等 `Init Done`。按原有校准姿势站好，松开双手 Trigger，
在 PICO 按 `A+B+X+Y` 校准并启动身体策略，进入 Planner idle。
可以先留在此模式调头，无需切换到 POSE 或 VR3PT。

右摇杆向前（SDK Y 正）增加 pitch、向后减小。H2 head pitch 为 +Y 轴，
正角是低头，负角是抬头。推满默认 5°/s，轻推更慢，死区 0.15。
松开后保持当前 pitch；头显转动不改变头部目标。head yaw 默认始终为 0°。
这是头部关节相对躯干的角度，不是相机相对世界的角度。

右摇杆左右仍控制身体 heading；左摇杆仍控制身体移动。
需要手臂遥操作时，对齐机器人手臂后按左摇杆进入 VR3PT；`A+X` 可进入 POSE。
停止用 PICO `A+B+X+Y` 或部署端 `o`。

调角时把电脑的相机窗口和 PICO 终端并排：观察桌面、操作区域、双手是否进入画面，
轻推右摇杆，松开等画面稳定，再读取 `[hold]` 的 pitch 目标值。机器人躯干姿态会
改变最终视角，选择角度时使用准备采集的站姿/操作高度。
结束时先停止机器人控制，再关闭 PICO Manager；相机窗口 `q`/Esc 只退出查看器，
不会停止机器人。此次调角没有启动 LeRobot exporter，不录制训练 episode。

## 终端输出和复用角度

PICO 终端首次发送、调整过程中（至多 4 次/秒）和松开后输出：

```text
[HeadAdjust] pitch_target=+15.00 deg (+0.261799 rad), yaw_fixed=+0.00 deg [hold]
```

这是发送的目标关节角，非编码器实测角。保持目标的重复发布不受打印节流影响。
身体模式切换/校准不清零所选角度；进程重启使用配置中的初始值。

修改 `gear_sonic_deploy/config/h2_head_adjust.env`：

```bash
HEAD_YAW_DEG=0.0
HEAD_PITCH_DEG=15.0         # 把实际选中的目标角度填在这里
HEAD_PITCH_RATE_DEG_S=5.0
HEAD_STICK_DEADZONE=0.15
HAND_BACKEND=inspire
```

该文件显式选择 5k 模型，覆盖通用 `h2_deploy.env` 中的 MODEL；其余身体
Kp、动作集、网络参数读取通用实机配置。两个入口支持 `--print-command`，仅打印命令。

pitch 限位与部署端一致：−0.5236～0.83775 rad（约 −30～+48°）。
仅接收新鲜追踪帧，追踪停止时不会持续积分旧摇杆输入。头部消息超时继续沿用
C++ 现有的回参考目标逻辑；因此“保持”适用于正常通信期间，不是断流锁定。

## 验证

15 项 Python 测试通过：新摇杆调角单元/发布测试、现有头部姿态及消息编码测试、
Planner 速度测试。覆盖正反调节、固定 yaw、死区保持、限位、非有限输入、
暂停时间截断、重复/过期样本不发布、恢复后无积累跳变、松开打印和校准保留角度。
启动脚本语法、两个入口 dry run 和 Python 编译检查通过。未启动实机。
