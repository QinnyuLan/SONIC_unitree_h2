# Runtime 与指令协议

[部署指南](DEPLOYMENT_CN.md) · [完整字段及关节顺序](ARCHITECTURE.md)

## 模块边界

- `src/runtime/`：默认站立、参考采样、encoder / decoder 推理、指令仲裁和 H2 DDS 输出。启动不依赖动作文件或 planner。
- `src/interface/replay.py`：识别离线 G1、原生 H2 和 SMPL 动作，包括已生成的 text2motion 结果，输出对应类型的 10 帧参考。
- `src/interface/text2motion/echo_g.py`：使用独立 PyTorch 环境离线生成语音及 G1 手势，输出带校验值的标准回放包，不连接 runtime。通用 replay 调度身体参考与独立音频播放器，见 [text2motion](TEXT2MOTION_CN.md)。
- `src/interface/planner/service.py`：Python planner 服务；`src/interface/planner/` 完成轨迹处理，`src/interface/planner/engine.py` 直接调用 TensorRT Python API，输出相同的稀疏 10 帧。
- `src/interface/teleop/bridge.py`：把 PICO 跟踪数据转换为 SMPL / VR 身体参考，并独立发送头部与手部指令。可订阅 planner 的下肢参考。
- `src/interface/commands/`：轻量 Python 发送接口，无机器人 SDK 或模型依赖。仅预留 64 维 token 输入，不包含 VLA 或 GR00T 实现。

输入端与 planner 全部使用 Python，policy runtime 保留 C++。encoder、decoder、planner 推理均使用 TensorRT。部署不依赖 ONNX Runtime；模型接口和元数据通过发布清单与 SHA256 绑定校验，加载引擎时再检查张量接口。两端使用相同的 TensorRT 缓存格式和 FP32 构建设置（保留 TensorRT 默认的 TF32 许可，不启用 FP16）。缓存按模型、GPU、精度和 TensorRT 版本校验，不兼容时在本机重建。

C++ 执行代码、TensorRT 封装、Unitree SDK 和 CMake 构建入口统一位于 `src/runtime/`。测试所需的动作读取和运动学代码放在 `src/runtime/tests/`，不参与正式 runtime 的编译。模型、回放示例、部署配置分别位于 `models/`、`motions/`、`config/`；编译产物为 `build/runtime/bin/sonic_h2_runtime`。目录说明见[工程结构](LAYOUT.md)。

runtime 用三秒过渡到默认站姿，随后以 50 Hz 运行身体策略、500 Hz 发布 DDS 指令。无身体指令或指令超时后选择站立参考。机器人状态丢失、推理异常、控制线程超时或 `stop` 会结束控制并进入阻尼；停止后需重启 runtime。

仿真器实现独立位于 `src/sim/`，通过 DDS 与 runtime 通信，不导入上游指令模块。仿真与遥操作使用的机器人资源统一位于 `assets/`。Python 安装入口为根目录 `pyproject.toml`。

## 独立通道

| 默认端口 | 接收内容 |
|---|---|
| 5558 | 身体：G1、H2、SMPL、VR、token、stand、stop |
| 5559 | runtime 发布原生 H2 状态及当前指令来源 |
| 5560 | 独立 planner 接收行走模式、方向、速度、高度 |
| 5561 | planner 发布稀疏 G1 参考 |
| 5562 | 头部：pitch / yaw |
| 5563 | 灵巧手：left / right |
| 5556 | PICO 原有二进制跟踪数据，由适配端接收 |

新通道使用 ZeroMQ 单帧 JSON。输入为 runtime / planner 绑定的 PULL，发送端连接 PUSH；状态与 planner 输出为 PUB/SUB。默认仅监听本机，跨机器时显式设置地址并同步系统时间。

每个包包含 schema、kind、source、session、递增 seq、Unix 毫秒时间戳、ttl_ms 和 data。默认有效期 250 ms，可设为 50–1000 ms，传输延迟计入有效期。格式、维度、有限数值和四元数均在执行前验证。

每一路有独立的来源占用规则。当前来源未释放或超时前，其他来源需显式指定 `takeover`。旧来源退出不会清除新来源指令。身体 `stand` 只释放身体；头部、手部不会随身体切换被重置。发送成功仅表示本地已入队，实际激活来源应从 5559 状态反馈确认。

## 参考数据

G1 回放、planner 和 G1 格式的离线 text2motion 统一发送 `joint_pos[10,29]`、`joint_vel[10,29]`、`root_quat[10,4]`，时间偏移为 0.0–0.9 秒，每 0.1 秒一帧。关节使用原生 G1 semantic 顺序，角度 rad、速度 rad/s，不在 runtime 内投影为 H2。

H2 回放读取原生 31 维 MuJoCo 关节文件，排除头部并重排为 29 维原生身体策略顺序，发送 `h2` 类型。字段与时间间隔和 G1 一致，但使用原生 H2 物理角度，不能将两种输入混用；不进行 G1/H2 运动学投影。

VR 使用左腕、右腕、头部三点的 `positions[3,3]` 与 `orientations[3,4]`，坐标为标定后的机器人局部坐标；`lower_body` 携带相同的 G1 稀疏参考。SMPL 使用间隔 20 ms 的 10 帧，包含 `joints[10,24,3]`、`root_quat[10,4]` 和 `wrist_joints[10,6]`。SMPL joints 为以根为中心、世界轴表达的向量，runtime 转为 SMPL 根坐标。腕部顺序为左 roll、右 roll、左 pitch、右 pitch、左 yaw、右 yaw。所有四元数均为 **wxyz**。

runtime 根据包的时间推进参考，角度/速度线性插值，四元数使用 SLERP。来源、session 或身体类型切换时对齐 heading；同一连续输入的后续包保留转向变化。超出参考末帧时暂时保持末值，直到指令到期回到站立。

发布模型接口保持为 full encoder 1751 → 64、decoder 994 → 29。selector 0/1/2/3 分别表示 H2 原生轨迹或默认站立、VR、SMPL、G1。decoder 的 10 帧历史按从旧到新排列。

## 头部与灵巧手

头部数据为 `{"pitch": 0.3, "yaw": -0.2}`，直接写入 H2 DDS 的 29、30 号关节，不进入 encoder 或 decoder。pitch 限位为 `[-0.5236, 0.83775]`，yaw 为 `[-1.7453, 1.7453]`，最大目标速度 2 rad/s；指令到期按同一限速回中。

手部数据为 `left` 和/或 `right` 的 7 维 Dex3 兼容目标，Inspire 后端将其转换为闭合度。缺失的一侧不复用旧指令；超时停止发布，不额外生成开合动作。Inspire 保留先松开再允许闭合的逻辑。

字段示例、Python 调用和完整关节顺序见 [英文协议](ARCHITECTURE.md)。
