# Offline Text-to-Motion: ECHO-G

[中文](TEXT2MOTION_CN.md) · [Deployment](DEPLOYMENT.md) · [Motion formats](../motions/README.md)

ECHO-G is the speech-and-gesture example for this repository's offline text-to-motion interface. Input text is **what the robot should say**; the model generates accompanying G1 gestures. It is not an action-command planner.

```text
text → speech + word timestamps → ECHO-G → G1 motion + matching WAV
                                                 ↓
                                           motion replay
                                                 ↓
                                  H2 policy runtime + audio player
```

Generation is a separate Python process using the pinned upstream PyTorch implementation. It never starts the policy, publishes robot commands or plays sound. Replay uses the same ten-frame G1 interface as other offline motions. Policy and planner inference continue to use TensorRT.

## Replay the included examples

The Chinese and English examples under `motions/example_text2motion_replay/` contain the generated motion, its matching WAV and a `replay.json` manifest. They can be replayed without installing ECHO-G or downloading its models.

```bash
./install.sh inputs
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_en --inspect
```

Start the simulator and policy runtime as described in [deployment](DEPLOYMENT.md#sim2sim). Wait for the runtime to enter its standing policy, then run:

```bash
# Sound plays on the computer running replay; requires alsa-utils/aplay.
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_en

# Chinese example, motion only.
.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_zh --mute
```

Audio is enabled when a bundle contains a WAV. `--inspect` checks files without connecting or playing; `--mute` disables sound. Stop other body producers first, or use `--takeover` to replace their input. Replay completion releases the body channel to standing.

## Install the optional generator

```bash
./install.sh text2motion
sudo apt install curl ffmpeg alsa-utils
.venv_text2motion/bin/python -m interface.text2motion.echo_g download
```

The installer creates `.venv_text2motion`, installs the pinned CPU PyTorch environment, and initializes the unmodified ECHO-G submodule at commit `60b85fb932b124b1a2ddd0868f4917a73c383901`. It does not install benchmark or visualization extras. This profile is independent of the runtime, planner, simulator and replay environments.

The download command fetches the audio+text EMA checkpoint and the pinned Qwen3.5-4B / wav2vec2 feature encoders into `models/text2motion_echo_g/`, approximately 11 GB in total. It verifies SHA256 for all 15 runtime files. Model files are ignored by Git. Use `download --verify-only` to check local files, or `download --checkpoint-only` when supplying existing conditions. See [model details](../models/text2motion_echo_g/README.md).

CPU generation is the tested default. `--device cuda` is available in an environment with a compatible CUDA-enabled PyTorch installation; installing the CPU profile does not enable it. The offline generator does not use an ONNX export or ONNX Runtime.

## Generate a speech/gesture bundle

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

Use `--text-file` for UTF-8 input. The default voices are `zh-CN-XiaoxiaoNeural` and `en-US-JennyNeural`; `--voice` overrides them. Each request needs a new or empty output directory. `--seed` controls sampling and defaults to 0. Generation uses the official EMA checkpoint, eight Euler steps and guidance scale 1.

The optional Edge TTS adapter uses a network service and requests actual word boundaries. Each utterance must fit within 20 seconds and 256 text tokens. Longer input is rejected rather than truncated. Speech is prepared completely before motion generation and playback; this is not streaming speech generation.

To use existing speech, provide a JSONL file containing one utterance with timings measured from that same WAV:

```json
{"stem":"speech","audio_path":"speech.wav","transcript":"Hello there","words":[{"text":"Hello","start":0.10,"end":0.50},{"text":"there","start":0.55,"end":0.95}]}
```

```bash
.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --manifest path/to/utterance.jsonl --language en \
  --output outputs/text2motion/existing_speech
```

The WAV must be 16 kHz mono PCM16. With model files already downloaded, this path works without the TTS service. Text and audio feature extraction call the official upstream functions, retaining the required layer, time alignment, FP16 feature-storage boundary and provenance checks. Do not estimate timestamps by dividing words evenly.

For previously generated upstream conditions:

```bash
.venv_text2motion/bin/python -m interface.text2motion.echo_g generate \
  --conditions path/to/speech.pt --audio path/to/speech.wav \
  --output outputs/text2motion/from_conditions
```

The condition file must contain upstream provenance and the SHA256 of the supplied WAV. This path needs the ECHO-G checkpoint but does not load Qwen or wav2vec2.

## Output contract

The portable playback bundle consists of:

```text
replay.json       # schema sonic.replay.v1; file paths, hashes, generation metadata
motion.npz        # G1 joint_pos, joint_vel, root_quat, fps, reference_type
speech.wav        # the exact speech used to generate the motion
```

Additional extraction and inference records stay under `generation/`; they are not needed for playback. Copy the three bundle files together to the replay computer. Paths inside `replay.json` are relative, and replay validates both motion and audio hashes before sending commands.

ECHO-G produces physical 39D motion at 30 Hz. The adapter reorders its 29 canonical G1 joints into SONIC semantic order, reconstructs root quaternions from the two-row 6D rotation representation, and resamples to 50 Hz. Velocities are time derivatives, with zero terminal velocity. The remaining root-velocity/yaw channels are not motor targets. Runtime receives `kind: g1`, selecting encoder branch 3; no G1-to-H2 joint projection is applied.

Audio playback uses the original frame rate and speed. `--speed`, `--fps` overrides and `--loop` require `--mute`. `--audio-delay SECONDS` delays audio relative to motion to compensate a measured playback/tracking offset. A successful motion send means it was queued, not that the policy accepted or tracked it. Playback shares a sender clock; speaker buffering and robot tracking latency are not hardware synchronized. Clips are independent utterances without cross-utterance transition blending.

## Unitree built-in speaker

The native build also produces the standalone `unitree_audio_player`. It reuses the C++ Unitree SDK and has no policy or model dependency.

```bash
bash scripts/build_controller.sh
build/runtime/bin/unitree_audio_player --check-only \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

# Read-only voice-service check. Replace eth0 with the robot network interface.
build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --probe \
  motions/example_text2motion_replay/echo_g_zh/speech.wav

.venv_inputs/bin/python -m interface.replay \
  motions/example_text2motion_replay/echo_g_zh --audio-ready \
  --audio-command 'build/runtime/bin/unitree_audio_player --interface eth0 --domain 0 --play --wait-for-go'
```

`--audio-ready` waits for the player's READY signal before queuing motion, then signals PLAY using the replay clock. Failure to initialize the speaker or connect the motion transport prevents audio start. Interrupted playback stops the matching audio stream and releases the body channel.

The public SDK names its voice client `g1`; this adapter uses that PCM service. Loopback mock-service tests passed, but compatibility with the target H2 firmware and real speaker/motion latency still require hardware validation. Default `aplay` output is for local simulation presentation.

## Validation and upstream

New Chinese and English text both completed TTS, CPU feature extraction and official PyTorch generation. The included bundles pass the minimal replay environment's loading and checksum checks. CPU tests cover joint order, rotation convention, resampling, audio binding, portable bundle paths and player startup/cleanup. A loopback mock Unitree voice service checks the exact PCM payload, normal completion and interrupted playback without a physical device. Reproduce it with `.venv_sim/bin/python tools/check_audio_replay.py` after the native build; see [validation records](REPRODUCIBILITY.md#offline-echo-g-speechgesture-validation). No closed-loop policy tracking or hardware playback was performed for these examples.

- [ECHO-G source](https://github.com/NJU-SOARLAB/ECHO-G/tree/60b85fb932b124b1a2ddd0868f4917a73c383901)
- [Official inference format](../third_party/echo_g/docs/INFERENCE.md)
- [Official model weights](https://huggingface.co/gaopusen/ECHO-G/tree/v0.2.0)
- [Upstream license and notices](../licenses/README.md)
