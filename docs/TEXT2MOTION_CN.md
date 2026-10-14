# 离线 Text-to-Motion：ECHO-G

[English](TEXT2MOTION.md) · [部署指南](DEPLOYMENT_CN.md)

ECHO-G 作为本工程“先生成、后回放”的 text2motion 范例：输入机器人**要说的话**，生成对应语音和伴随手势，再用统一的 motion replay 展示。它不负责理解“向前走两步”等动作规划指令。

生成端是独立 Python 进程，使用本地精简的 PyTorch 推理代码直接加载权重，只输出文件，不启动机器人或播放声音。replay 将生成动作作为 G1 稀疏 10 帧发送给 H2 runtime；policy 和 planner 仍使用 TensorRT。

## 直接展示已有示例

仓库提供中英文两套完整的 H2 自我介绍 demo，包含 G1 动作、配套语音、讲稿和字幕。回放只需 inputs 环境及 H2 runtime，无需配置 ECHO-G、下载生成模型或连接语音服务。

| 语言 | 目录 | 时长 | 讲稿 |
|---|---|---:|---|
| 中文站定版 | `h2_trace_intro_zh` | 73.04 秒 | [中文讲稿](../motions/example_text2motion_replay/h2_trace_intro_zh/script.txt) |
| 英文站定版 | `h2_trace_intro_en` | 78.48 秒 | [英文讲稿](../motions/example_text2motion_replay/h2_trace_intro_en/script.txt) |

另提供 `h2_trace_intro_full_body_zh`、`h2_trace_intro_full_body_en` 全身对比版，时长及语音与对应站定版完全相同。站定版保留 ECHO-G 生成的腰部和手臂手势，将 12 个腿部参考关节替换为标准 G1 站姿，并使用直立根姿态；H2 的所有关节仍由 policy 控制。全身版保留各段生成的腿部和根姿态变化，可在仿真中对比下肢运动和身体摆动；两种模式均不锁定 H2 的世界坐标位置。

两份讲稿均介绍 H2、TRACE 迁移、动作跟踪、VR 遥操作和 ECHO-G。每份包含七段语音动作，段间有一秒静音和动作平滑过渡。原来的简短问候示例 `echo_g_zh`、`echo_g_en` 也保留。

```bash
./install.sh inputs
sudo apt install alsa-utils
```

先检查文件，不启动播放：

```bash
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/h2_trace_intro_zh --inspect
```

按[部署指南](DEPLOYMENT_CN.md)启动仿真器和 runtime，等待 runtime 进入站立策略，再执行：

```bash
# 终端 1/2 启动仿真器和 runtime 并开启 policy 后，在终端 3 执行。
# 中文完整介绍，语音从发送端电脑输出。
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/h2_trace_intro_zh

# 英文完整介绍，同样同时播放动作和语音。
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/h2_trace_intro_en

# 全身对比版，在仿真中查看。
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/h2_trace_intro_full_body_zh
```

本地音频默认使用 `aplay`，通过 `sudo apt install alsa-utils` 安装。`--inspect` 只校验文件；`--mute` 静音。先停止其他身体指令来源，或用 `--takeover` 接管。回放结束后身体通道释放，runtime 回到站立。

## 安装并生成新内容

```bash
./install.sh text2motion
sudo apt install curl ffmpeg alsa-utils
.venv_text2motion/bin/python -m text2motion_offline.echo_g download
```

生成环境为 `.venv_text2motion`，与 replay、planner、仿真和遥操作环境独立。安装器使用固定依赖和 CPU 版 PyTorch，不安装 ECHO-G 包或拉取其源码仓库。模型下载到 `models/text2motion_echo_g/`，约 11 GB，包括 ECHO-G checkpoint、Qwen3.5-4B 和 wav2vec2。下载会校验全部 15 个运行文件的 SHA256；权重不提交 Git。

```bash
.venv_text2motion/bin/python -m text2motion_offline.echo_g generate \
  --text "大家好，欢迎来到我们的实验室。" --language zh \
  --output outputs/text2motion/greeting_zh

.venv_text2motion/bin/python -m text2motion_offline.echo_g generate \
  --text "Hello. Welcome to our lab." --language en \
  --output outputs/text2motion/greeting_en

.venv_inputs/bin/python -m interface.motion_replay outputs/text2motion/greeting_zh --inspect
.venv_inputs/bin/python -m interface.motion_replay outputs/text2motion/greeting_zh
```

也可使用 `--text-file` 读取 UTF-8 文件。默认中文声音为 `zh-CN-XiaoxiaoNeural`，英文为 `en-US-JennyNeural`，可用 `--voice` 覆盖。每句话使用新的空输出目录，`--seed` 默认为 0。生成采用官方 EMA 模型、8 步 Euler、guidance scale 1；不引入 ECHO-G ONNX 导出或 ONNX Runtime。

推理与合成代码集中在 `tools/text2motion_offline/echo_g/`：

| 文件 | 职责 |
|---|---|
| `cli.py` | 下载、语音合成、生成入口和 G1 回放导出 |
| `features.py` | WAV 读取、Qwen/wav2vec2 特征与词时间对齐 |
| `model.py` | audio+text 网络定义、EMA 权重加载与 8 步 Euler 推理 |
| `sequence.py` | 按段生成、段间动作过渡与字幕导出 |

权重保存参数和模型配置，因此保留匹配的网络定义；不包含训练、评估、数据集或其他模型分支。WAV 使用标准库读取，移除了 librosa、soundfile 及其附带依赖。

默认使用 CPU。准备了兼容 CUDA 版 PyTorch 的独立环境后可选 `--device cuda`；本工程默认安装的 CPU 环境不能直接启用 CUDA。

## 生成完整讲稿

`generate-script` 按空行切分讲稿，逐段生成后合成为一个完整回放包。每段仍需满足 20 秒、256 token 的限制，超限会报错。

```bash
.venv_text2motion/bin/python -m text2motion_offline.echo_g generate-script \
  --text-file motions/example_text2motion_replay/h2_trace_intro_zh/script.txt \
  --language zh --seed 0 --pause 1.0 --body-mode standing \
  --output outputs/text2motion/h2_trace_intro_zh

.venv_text2motion/bin/python -m text2motion_offline.echo_g generate-script \
  --text-file motions/example_text2motion_replay/h2_trace_intro_en/script.txt \
  --language en --seed 0 --pause 1.0 --body-mode standing \
  --output outputs/text2motion/h2_trace_intro_en
```

`--body-mode standing` 使用上述站定讲话参考；`--body-mode full-body`（默认）保留生成的全身参考。仓库中的成对版本使用相同的源动作、音频和时间轴，清单记录了模式和站立参考关节值。

每段的 seed 从 `--seed` 依次递增，内置 demo 使用 0–6。合成时保留各段原始音频，只在段尾补齐到 50 Hz 帧边界，并插入 `--pause` 指定的静音。关节采用五次 smoothstep 在两端角度之间插值，过渡两端斜率为零，避免外推导致关节角度过冲，根旋转先对齐朝向再平滑插值，最后重算整段关节速度；语音不重叠、不变速。`replay.json` 记录段落起点、seed 和源文件校验值，额外导出 `script.txt`、`subtitles.srt`。TTS 服务更新可能改变重新生成的音频；复现仓库演示时直接使用已提交的素材。

## 已有语音与离线运行

“离线”表示整句准备完毕后再回放。默认 Edge TTS 需要网络，并返回真实词时间戳；每句最多 20 秒、256 个文本 token，超限直接报错。

如果使用自己的 TTS，可以提供 16 kHz、单声道、PCM16 WAV 和对应真实词时间戳，每个 JSONL 文件只放一句：

```json
{"stem":"speech","audio_path":"speech.wav","transcript":"Hello there","words":[{"text":"Hello","start":0.10,"end":0.50},{"text":"there","start":0.55,"end":0.95}]}
```

```bash
.venv_text2motion/bin/python -m text2motion_offline.echo_g generate \
  --manifest path/to/utterance.jsonl --language en \
  --output outputs/text2motion/existing_speech
```

模型下载完成后，此入口不需要 TTS 网络服务。精简后的特征提取保留发布模型要求的文本层选择、词时间对齐、FP16 存储边界和来源校验。不要将字数均分作为词时间戳。

也支持已生成的官方条件文件：

```bash
.venv_text2motion/bin/python -m text2motion_offline.echo_g download --checkpoint-only
.venv_text2motion/bin/python -m text2motion_offline.echo_g generate \
  --conditions path/to/speech.pt --audio path/to/speech.wav \
  --output outputs/text2motion/from_conditions
```

条件文件必须带有官方来源信息和对应 WAV 的 SHA256。此模式只加载 ECHO-G checkpoint，不加载 Qwen 或 wav2vec2。

## 回放产物与时间对齐

输出目录中的三个文件可以一起复制到另一台机器：

- `motion.npz`：50 Hz G1 原生关节、速度、根四元数和输入类型。
- `speech.wav`：生成动作时使用的同一份音频。
- `replay.json`：相对路径、文件校验值及生成参数。

完整讲稿还包含 `script.txt` 和 `subtitles.srt`。`generation/` 保存特征和原始推理记录，回放不依赖这些中间文件。replay 在发送前验证语音和动作校验值，同长度的另一段音频也不能混用。

ECHO-G 的 30 Hz、39 维物理动作经过关节重排、6D 根旋转转四元数与 50 Hz 重采样，作为标准 G1 输入发送，使用 H2 encoder 的 selector 3。不进行 G1→H2 关节投影；头部和灵巧手仍由独立通道控制。

有声音时保持原始帧率、速度和单次播放；使用 `--mute` 后才可调整 `--speed`、`--fps` 或开启 `--loop`。`--audio-delay 秒数` 可按实测延迟推迟语音。动作和音频共用发送端时钟，但声卡缓冲、网络和策略跟踪存在延迟；发送成功也不等于 runtime 已接受或执行。`generate` 输出单句，`generate-script` 在回放前完成段间过渡，回放时不运行生成模型。

## Unitree 内置扬声器

原生构建会生成独立的音频工具，不将语音逻辑放入 policy 控制循环：

```bash
bash scripts/build_controller.sh
build/runtime/bin/unitree_audio_player --check-only \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

# 只读查询服务，将 eth0 替换为机器人网卡。
build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --probe \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/echo_g_zh --audio-ready \
  --audio-command 'build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --play --wait-for-go'
```

replay 先等待播放器 READY，确认动作连接可以发送，再按回放时钟发送 PLAY。连接失败不会开始音频；退出时停止对应音频流并释放身体通道。工具复用 C++ Unitree SDK，不需要 Python DDS 环境。

公开 SDK 的语音客户端位于 `g1` 命名空间。已在 loopback 模拟服务中验证 PCM 内容、播放时序与停止目标，尚未验证目标 H2 固件的 PCM 兼容性、真实声音延迟或实机动作跟踪。

编译后可用 `.venv_sim/bin/python tools/validation/check_audio_replay.py` 复现 loopback 模拟验证，包括正常播放和中断清理；不会启动 policy、连接实机或播放声音。

原来的中英文简短问候示例已完成 CPU 全流程验证。移除上游包后重新提取特征并生成动作，与原实现的特征张量和最终动作数组逐元素完全一致；额外的 seed 7 采样也完全一致。仓库内示例已通过最小 replay 环境检查。完整自我介绍的仿真验证单独记录在[复现说明](REPRODUCIBILITY.md#complete-h2-introduction-demos)。上游版本、文件格式及许可链接见[英文说明](TEXT2MOTION.md)和[许可索引](../licenses/README.md)。

## 完整 demo 仿真验证

四套完整自我介绍均在 H2 policy 闭环仿真中完成回放，没有跌倒，结束后回到站立。中英文站定版的最大水平位移分别为 0.20/0.18 米，全身版为 0.28/0.34 米。两种参考均不锁定世界坐标位置。详见[测量值与验证范围](REPRODUCIBILITY.md#complete-h2-introduction-demos)。测试使用本地仿真和静音音频解码，未连接真实机器人或扬声器。
