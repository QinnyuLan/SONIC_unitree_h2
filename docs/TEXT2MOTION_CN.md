# 离线 Text-to-Motion：ECHO-G

[English](TEXT2MOTION.md) · [部署指南](DEPLOYMENT_CN.md)

ECHO-G 作为本工程“先生成、后回放”的 text2motion 范例：输入机器人**要说的话**，生成对应语音和伴随手势，再用统一的 motion replay 展示。它不负责理解“向前走两步”等动作规划指令。

生成端是独立 Python 进程，调用固定版本的官方 PyTorch 实现，只输出文件，不启动机器人或播放声音。replay 将生成动作作为 G1 稀疏 10 帧发送给 H2 runtime；policy 和 planner 仍使用 TensorRT。

## 直接展示已有示例

仓库提供中英文两组短示例，无需安装 ECHO-G 或下载生成模型：

```bash
./install.sh inputs
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_zh --inspect
```

按[部署指南](DEPLOYMENT_CN.md)启动仿真器和 runtime，等待 runtime 进入站立策略，再执行：

```bash
# 中文：动作与语音一起播放，语音从发送端电脑输出。
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_zh

# 英文：只回放动作。
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_en --mute
```

本地音频默认使用 `aplay`，通过 `sudo apt install alsa-utils` 安装。`--inspect` 只校验文件；`--mute` 静音。先停止其他身体指令来源，或用 `--takeover` 接管。回放结束后身体通道释放，runtime 回到站立。

## 安装并生成新内容

```bash
./install.sh text2motion
sudo apt install curl ffmpeg alsa-utils
.venv_text2motion/bin/python -m interface.text2motion.echo_g download
```

生成环境为 `.venv_text2motion`，与 replay、planner、仿真和遥操作环境独立。安装器使用固定依赖和 CPU 版 PyTorch，并初始化固定版本的官方 ECHO-G submodule。模型下载到 `models/text2motion_echo_g/`，约 11 GB，包括 ECHO-G checkpoint、Qwen3.5-4B 和 wav2vec2。下载会校验全部 15 个运行文件的 SHA256；权重不提交 Git。

```bash
.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --text "大家好，欢迎来到我们的实验室。" --language zh \
  --output outputs/text2motion/greeting_zh

.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --text "Hello. Welcome to our lab." --language en \
  --output outputs/text2motion/greeting_en

.venv_inputs/bin/python -m interface.replay outputs/text2motion/greeting_zh --inspect
.venv_inputs/bin/python -m interface.replay outputs/text2motion/greeting_zh
```

也可使用 `--text-file` 读取 UTF-8 文件。默认中文声音为 `zh-CN-XiaoxiaoNeural`，英文为 `en-US-JennyNeural`，可用 `--voice` 覆盖。每句话使用新的空输出目录，`--seed` 默认为 0。生成采用官方 EMA 模型、8 步 Euler、guidance scale 1；不引入 ECHO-G ONNX 导出或 ONNX Runtime。

默认使用 CPU。准备了兼容 CUDA 版 PyTorch 的独立环境后可选 `--device cuda`；本工程默认安装的 CPU 环境不能直接启用 CUDA。

## 已有语音与离线运行

“离线”表示整句准备完毕后再回放。默认 Edge TTS 需要网络，并返回真实词时间戳；每句最多 20 秒、256 个文本 token，超限直接报错。

如果使用自己的 TTS，可以提供 16 kHz、单声道、PCM16 WAV 和对应真实词时间戳，每个 JSONL 文件只放一句：

```json
{"stem":"speech","audio_path":"speech.wav","transcript":"Hello there","words":[{"text":"Hello","start":0.10,"end":0.50},{"text":"there","start":0.55,"end":0.95}]}
```

```bash
.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --manifest path/to/utterance.jsonl --language en \
  --output outputs/text2motion/existing_speech
```

模型下载完成后，此入口不需要 TTS 网络服务。特征提取直接复用官方流程，保留文本层选择、词时间对齐和来源校验。不要将字数均分作为词时间戳。

也支持已生成的官方条件文件：

```bash
.venv_text2motion/bin/python -m interface.text2motion.echo_g download --checkpoint-only
.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --conditions path/to/speech.pt --audio path/to/speech.wav \
  --output outputs/text2motion/from_conditions
```

条件文件必须带有官方来源信息和对应 WAV 的 SHA256。此模式只加载 ECHO-G checkpoint，不加载 Qwen 或 wav2vec2。

## 回放产物与时间对齐

输出目录中的三个文件可以一起复制到另一台机器：

- `motion.npz`：50 Hz G1 原生关节、速度、根四元数和输入类型。
- `speech.wav`：生成动作时使用的同一份音频。
- `replay.json`：相对路径、文件校验值及生成参数。

`generation/` 保存特征和原始推理记录，回放不依赖这些中间文件。replay 在发送前验证语音和动作校验值，同长度的另一段音频也不能混用。

ECHO-G 的 30 Hz、39 维物理动作经过关节重排、6D 根旋转转四元数与 50 Hz 重采样，作为标准 G1 输入发送，使用 H2 encoder 的 selector 3。不进行 G1→H2 关节投影；头部和灵巧手仍由独立通道控制。

有声音时保持原始帧率、速度和单次播放；使用 `--mute` 后才可调整 `--speed`、`--fps` 或开启 `--loop`。`--audio-delay 秒数` 可按实测延迟推迟语音。动作和音频共用发送端时钟，但声卡缓冲、网络和策略跟踪存在延迟；发送成功也不等于 runtime 已接受或执行。当前逐句播放，不做跨句动作平滑拼接。

## Unitree 内置扬声器

原生构建会生成独立的音频工具，不将语音逻辑放入 policy 控制循环：

```bash
bash scripts/build_controller.sh
build/runtime/bin/unitree_audio_player --check-only \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

# 只读查询服务，将 eth0 替换为机器人网卡。
build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --probe \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_zh --audio-ready \
  --audio-command 'build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --play --wait-for-go'
```

replay 先等待播放器 READY，确认动作连接可以发送，再按回放时钟发送 PLAY。连接失败不会开始音频；退出时停止对应音频流并释放身体通道。工具复用 C++ Unitree SDK，不需要 Python DDS 环境。

公开 SDK 的语音客户端位于 `g1` 命名空间。已在 loopback 模拟服务中验证 PCM 内容、播放时序与停止目标，尚未验证目标 H2 固件的 PCM 兼容性、真实声音延迟或闭环动作跟踪。

编译后可用 `.venv_sim/bin/python tools/check_audio_replay.py` 复现 loopback 模拟验证，包括正常播放和中断清理；不会启动 policy、连接实机或播放声音。

中英文新文本已完成 CPU 全流程验证；仓库内示例已通过最小 replay 环境检查。上游版本、文件格式及许可链接见[英文说明](TEXT2MOTION.md)和[许可索引](../licenses/README.md)。
