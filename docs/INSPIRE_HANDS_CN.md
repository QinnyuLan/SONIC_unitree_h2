# H2 机器人端 Inspire 灵巧手服务

[English](INSPIRE_HANDS.md) · [部署指南](DEPLOYMENT_CN.md) · [工具目录](../tools/README.md)

[`tools/robot_services/inspire_hands/`](../tools/robot_services/inspire_hands/) 中的服务运行在连接灵巧手串口的 H2 PC2 上，将工作站发来的 DDS 指令转换为左右手各自的 RS-485 指令，并发布实测位置。这是 Inspire RH56 后端；Unitree Dex3 使用另一套 DDS 话题，不经过此串口服务。

```text
Python 遥操作／手部指令
  → 工作站 C++ policy runtime（HAND_BACKEND=inspire）
  → 机器人网络上的 DDS
  → H2 PC2：inspire_hand_service
  → 左右手独立 RS-485 串口
```

PC2 仅自动启动手部服务。身体 policy、VR 等进程仍由操作者单独启动。手部服务不需要 policy 权重、CUDA、TensorRT 或 Python 环境。

DDS 使用 `rt/inspire/cmd`（`unitree_go::msg::dds_::MotorCmds_`）和 `rt/inspire/state`（`MotorStates_`）。每条消息均为 12 项，前 6 项右手、后 6 项左手；单侧顺序为 `[pinky, ring, middle, index, thumb_bend, thumb_rotate]`。位置归一化到 `[0, 1]`，`0` 全闭、`1` 全开；状态 `lost>0` 表示测量缺失或过期。runtime 会把上游的 7 维手部表示转换为这里每侧 6 项的协议。

## 传输与编译

在开发工作站的仓库根目录打包源码，不包含模型和 Python 环境：

```bash
mkdir -p outputs
tar -czf outputs/inspire-hands-source.tar.gz \
  tools/robot_services/inspire_hands \
  src/runtime/third_party/unitree_sdk2 \
  tests/tools/robot_services/inspire_hands \
  licenses/inspire_hands docs/INSPIRE_HANDS.md docs/INSPIRE_HANDS_CN.md
scp outputs/inspire-hands-source.tar.gz unitree@192.168.123.162:/home/unitree/
```

`192.168.123.162` 是原部署记录中的 PC2 地址，请按实际机器人修改地址和账户。在 PC2 上执行：

```bash
mkdir -p ~/SONIC_unitree_h2
tar -xzf ~/inspire-hands-source.tar.gz -C ~/SONIC_unitree_h2
cd ~/SONIC_unitree_h2
sudo apt-get update
sudo apt-get install build-essential cmake libboost-program-options-dev libeigen3-dev
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/inspire_hands -j2
```

应在 PC2 上按其架构和系统库编译，不要将工作站的 x86_64 二进制直接复制到 ARM64 机器人。默认使用仓库中随附的 Unitree SDK，包含这两种架构，无需联网下载。也可在配置时加 `-DUNITREE_SDK_ROOT=/unitree/opt` 使用机载已安装的 SDK；切换 SDK 时使用新的构建目录。如果 PC2 无外网，应提前准备与其 Ubuntu 版本和架构匹配的上述安装包及依赖，或使用已安装的编译环境。

产物是 `build/inspire_hands/bin/inspire_hand_service` 和 `inspire_hand_client`。后者默认只读取 DDS 状态，只有显式指定 `--position` 或 `--cycle` 才发送动作。

## 串口映射与首次检查

探测或替换服务前，停止手部指令发送端和旧手部进程。探测只读取位置，不初始化 DDS、不发送位置目标，但仍需独占串口。设置时保持手部静止并留出活动空间。

在 PC2 上查看旧服务、串口和网卡：

```bash
systemctl cat h2-inspire-hands.service
sudo systemctl stop h2-inspire-hands.service
pgrep -af 'example_h2_native485|inspire_hand_service|inspire_g1|inspire_h1'
ls -l /dev/serial/by-id/
ip -br link
```

首次安装时可能还没有这个服务；旧的手动启动进程需要单独停止。原 G1 接线方式的 FTDI 转接器使用 **右手 `if01`、左手 `if02`，两侧设备 ID 均为 1**，波特率 115200、8N1。请在当前机器人上核对实际接线，使用稳定的 `/dev/serial/by-id/` 路径，避免 `/dev/ttyUSB*` 编号变化造成左右手交换。原生接线也可使用 `/dev/ttyUN0`、`/dev/ttyUN1`，但同样需要先确认。

安装二进制，并仅在配置不存在时创建配置文件：

```bash
sudo cmake --install build/inspire_hands --prefix /opt/SONIC_unitree_h2/inspire_hands
sudo install -d -m 0755 /etc/sonic
if [ ! -e /etc/sonic/inspire-hands.env ]; then
  sudo install -m 0644 tools/robot_services/inspire_hands/systemd/inspire-hands.env \
    /etc/sonic/inspire-hands.env
fi
sudoedit /etc/sonic/inspire-hands.env
```

把 `NETWORK_INTERFACE` 设置为 PC2 连接机器人网络的网卡，通常为 `eth0`；`DDS_DOMAIN` 与 runtime 保持一致，通常为 `0`。**左右两项 `REPLACE_WITH_*` 必须替换为本机串口路径。** 工作站可使用 `eno1`，两台机器的网卡名称无需相同。配置保持简单的 shell 变量赋值格式。安装时会将所选 SDK 的 DDS 动态库放到服务旁的 `lib/`，因此 systemd 启动无需加载交互式 shell 或设置 `LD_LIBRARY_PATH`。

以服务用户及 `dialout` 组执行只读探测：

```bash
set -a
source /etc/sonic/inspire-hands.env
set +a
sudo -u unitree -g dialout /opt/SONIC_unitree_h2/inspire_hands/bin/inspire_hand_service \
  --probe-only --hands both --require-both \
  --right-port "$RIGHT_PORT" --left-port "$LEFT_PORT" \
  --right-id "$RIGHT_ID" --left-id "$LEFT_ID"
```

双手均应报告 `ONLINE`。`--require-both` 仅用于这次探测验收，常驻服务允许手部暂时缺失。串口打开失败时检查设备权限和占用进程；systemd 单元通过 `SupplementaryGroups=dialout` 提供串口访问权限，无需以 root 运行整个服务。

## 安装服务与开机自启动

替换已有安装前保留旧服务文件：

```bash
if [ -f /etc/systemd/system/h2-inspire-hands.service ]; then
  sudo cp -an /etc/systemd/system/h2-inspire-hands.service \
    /etc/systemd/system/h2-inspire-hands.service.before-sonic
fi
sudo install -m 0644 tools/robot_services/inspire_hands/systemd/h2-inspire-hands.service \
  /etc/systemd/system/h2-inspire-hands.service
sudo systemd-analyze verify /etc/systemd/system/h2-inspire-hands.service
sudo systemctl daemon-reload
sudo systemctl enable --now h2-inspire-hands.service
systemctl is-enabled h2-inspire-hands.service
systemctl status h2-inspire-hands.service --no-pager
journalctl -u h2-inspire-hands.service -b -n 50 -o cat
```

服务文件默认使用 `unitree:unitree`；如果机载账户不同，需修改安装后的 `User` 和 `Group`。服务进入 `active` 表示 DDS 和工作线程已启动，不代表双手都在线，还需只读查看状态：

```bash
/opt/SONIC_unitree_h2/inspire_hands/bin/inspire_hand_client \
  --network "$NETWORK_INTERFACE" --domain "$DDS_DOMAIN" --duration-ms 5000
```

双手应为 `lost=0`，位置值应有限。需要测试张手时，先停止其他 `rt/inspire/cmd` 发布者，再给上述客户端增加 `--position 1.0`。`0` 表示全闭，`1` 表示全开；显式位置测试会同时控制双手。

工作站的 [`config/h2_deploy.env`](../config/h2_deploy.env) 设置 `HAND_BACKEND=inspire`，DDS domain 与 PC2 一致，然后按 [Sim2Real](DEPLOYMENT_CN.md#sim2real) 启动。runtime 控手前先退出测试发送端。同一对灵巧手只保留一个指令发布者和一个串口服务。

## 看门狗与故障恢复

| 情况 | 默认行为 |
|---|---|
| PC2 开机 | 已启用的服务随 `multi-user.target` 启动，排在 network-online target 之后 |
| 进程退出或崩溃 | systemd 等待 2 秒后重启 |
| DDS 主循环卡住 | 心跳停止，10 秒 watchdog 超时后杀死并重启进程 |
| 单侧串口工作线程卡住 | 超过该线程正常周期预算及 2 秒余量后停止心跳，随后由 systemd 超时重启 |
| USB 未接入或被拔掉 | DDS 持续运行，每侧独立每秒尝试重开一次串口 |
| 串口存在但手部无应答 | 连续 3 次失败后该侧降为 2 Hz 查询，读成功后恢复正常 50 Hz 循环 |
| DDS 指令中断 | 不再写入新的位置目标；排队超过 500 ms 的指令被丢弃 |
| 操作者执行 `systemctl stop` | 保持停止，`Restart=always` 不会抵消手动停止操作 |

服务从主循环发出 `READY=1`、`WATCHDOG=1`，同时检查左右串口线程是否还在推进。USB 缺失属于可恢复设备状态，不会触发进程重启循环。状态正常以 20 Hz 发布，超过 250 ms 未更新的实测值会标记丢失。这是进程看门狗和设备恢复机制，不会重启卡死的 PC2 操作系统，也不会切断电机电源。机制定义见 systemd 的 [服务与看门狗配置](https://www.freedesktop.org/software/systemd/man/systemd.service.html) 和 [通知协议](https://www.freedesktop.org/software/systemd/man/sd_notify.html)。

服务启动、恢复和退出都不会自行生成开合姿态。USB 丢失会清除排队指令，失联期间到达的指令不会缓存后重放。恢复后服务接收新指令，runtime 的 Inspire 适配器另有“先松开 Trigger，再允许闭合”的限制。通信中断时，如果手部仍供电，可能继续保持最后目标。

日常查看看门狗状态和日志：

```bash
systemctl show h2-inspire-hands.service \
  -p ActiveState -p MainPID -p NRestarts -p WatchdogUSec -p WatchdogTimestamp
journalctl -u h2-inspire-hands.service -f
```

进行主动看门狗验收前，停止所有手部指令发送端并清空手部活动区域。下面的操作会暂停手部服务、短暂中断通信；预期在约一个 10 秒看门狗周期加 2 秒重启等待内，`MainPID` 改变、`NRestarts` 增加：

```bash
sudo systemctl kill --kill-who=main --signal=SIGSTOP h2-inspire-hands.service
systemctl show h2-inspire-hands.service -p MainPID -p NRestarts
```

等待看门狗周期后再次查看，日志应出现 watchdog 超时及重新启动记录。计划内重启 PC2 后，再检查 `is-enabled`、`is-active` 和双手 `lost=0`。仓库的本地验证不会更新或重启机器人。

## 维护与回退

修改 `/etc/sonic/inspire-hands.env` 后重启服务；修改服务单元后先执行 `daemon-reload`。执行 `sudo systemctl disable --now h2-inspire-hands.service` 可同时停止服务并取消开机启动。排障先查看日志、网卡/domain、串口权限和 `by-id` 链接；服务为 `active` 但 `lost>0` 时，应检查设备和串口，而非重复启动另一份服务。

更新二进制前，停止发送端和服务，将 `/opt/SONIC_unitree_h2/inspire_hands/` 与 `/etc/sonic/inspire-hands.env` 复制到带版本标记的备份目录。安装新版本后重新执行探测和只读 DDS 检查。如需退回迁移前的旧服务，停止新服务，用 `h2-inspire-hands.service.before-sonic` 恢复服务文件，执行 `daemon-reload` 后再启动。新部署验收完成前保留旧 `/home/unitree/dfx_inspire_service/`，因为旧服务文件可能仍引用该目录。

## 本机验证

开发 Linux 主机需 Python 3.10 或更新版本：

```bash
cmake -S tools/robot_services/inspire_hands -B build/inspire_hands -DBUILD_TESTING=ON
cmake --build build/inspire_hands -j2
ctest --test-dir build/inspire_hands --output-on-failure
python3 tools/validation/check_inspire_watchdog.py
```

CTest 使用伪终端和 loopback DDS domain 218，覆盖分段串口应答、设备缺失、重连与退避、DDS 到串口指令、失联指令不重放、systemd 通知及注入的串口线程卡死。最后一条命令还需要运行中的 systemd **用户级**管理器：它创建仅连接 loopback、串口路径不存在的临时服务，验证看门狗重启、崩溃重启和手动停止，结束后删除临时服务，不安装系统级服务。结果写入 `outputs/validation/inspire_watchdog.json`。这些验证不能替代实际机器人上的验收。
