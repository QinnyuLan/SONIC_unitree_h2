# Python Command Interfaces

These modules produce commands for the independent C++ policy runtime. Replay accepts G1, native H2 and SMPL clips. G1 replay and planner output the same sparse ten-frame reference; teleoperation sends SMPL/VR references, with separate head and hand channels.

| Module | Purpose |
|---|---|
| `interface.commands` | Validated protocol and `RuntimeClient` |
| [`interface.motion_replay`](#motion-replay) | Replay G1/H2/SMPL clips and motion/audio bundles |
| `interface.planner.service` | TensorRT Python planner and trajectory publisher |
| `interface.planner.client` | Locomotion requests |
| `interface.teleop.pico` | PICO tracking, calibration and pose processing |
| `interface.teleop.bridge` | Tracking-to-runtime adapter |
| `interface.peripherals.client` | Independent head pitch/yaw and hand targets |
| `interface.peripherals.serial_bridge` | Optional PICO-to-Inspire serial bridge |
| `interface.peripherals.serial_command` | Explicit Inspire serial commands |

`kinematics/` and `math/` support pose processing. `paths.py` locates the checkout's shared assets and planner weight. The package has no dependency on the simulator implementation.

## Motion replay

`interface.motion_replay` reads G1, native H2 or SMPL trajectories and sends timed references to the independent policy runtime. A motion/audio bundle can also start a separate local or Unitree audio player.

| File | Responsibility |
|---|---|
| `motion_replay/motion.py` | CSV/NPZ loading, input-type detection, joint ordering and reference sampling |
| `motion_replay/cli.py` | Playback timing, runtime connection, command publishing and cleanup |
| `motion_replay/keyboard.py` | Collection browser and manual playback controls |
| `motion_replay/playback.py` | Bundle paths/checksums and the audio-player lifecycle |
| `motion_replay/__main__.py` | `python -m interface.motion_replay` entry point |

From the repository root:

```bash
./install.sh inputs
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_g1_replay/walking_quip_360_R_002__A428 --inspect
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/echo_g_zh
```

Start the simulator or robot runtime before playback. `--inspect` only validates files. Use `--mute` for motion-only playback; see [motion formats](../../motions/README.md) and [audio setup](../../docs/TEXT2MOTION.md).

The complete H2 introductions are `motions/example_text2motion_replay/h2_trace_intro_zh` (Chinese, 73 s) and `h2_trace_intro_en` (English, 78 s). Pass either directory to the same replay command. These use standing leg references with generated waist/arm gestures; `h2_trace_intro_full_body_zh` / `h2_trace_intro_full_body_en` retain the full-body references for simulation comparison. Their motion, audio, scripts and subtitles are included; playback does not load ECHO-G or call a speech service.

The shared Python motion API is `from interface.motion_replay import Motion, SmplMotion`.

### Keyboard playback

With the simulator and policy runtime running in terminals 1 and 2, start the browser in terminal 3:

```bash
.venv_inputs/bin/python -m interface.motion_replay --keyboard
```

| Key | Action |
|---|---|
| `1` / `2` / `3` / `4` | Select `example_g1_replay` / `example_g1_explore_replay` / `example_h2_replay` / `example_smpl_replay` |
| Up/Down or `k`/`j` | Select the previous/next clip; stop the current playback |
| Enter or `r` | Play the selected clip from the beginning |
| Space | Pause/resume; pause releases the body channel and preserves the playback position |
| `n` / `p` | Switch to and play the next/previous clip |
| `x` | Stop and rewind; return the runtime to standing |
| `l` | Toggle looping; use only for clips with a continuous loop boundary |
| `q` / Esc / Ctrl+C | Exit replay and release body control |

Selecting a collection does not play it automatically. Playback waits for fresh runtime state in the `running` phase. After the runtime prints `[READY]`, press **]** or **s** in its terminal to enable policy control. **o** in that terminal stops with damping and requires a runtime restart. A state or command-connection loss pauses playback; reconnecting does not resume it. Pausing returns the robot to its default standing reference rather than holding an intermediate motion pose. Space resumes from the saved reference time; Enter starts from the beginning. The runtime status and current input owner are shown above the collection list.

Use `--endpoint` and `--state` for custom runtime sockets, or `--motions-root` for another directory containing the four named collections. `--speed` sets playback speed. Speech/audio bundles continue to use the direct motion-path CLI above; the keyboard browser handles the four typed motion collections.

## Planner and teleoperation

Install the planner or teleoperation extras with `./install.sh planner` or `./install.sh teleop`. Start the policy runtime before sending commands; see [deployment](../../docs/DEPLOYMENT.md) and [message definitions](../../docs/ARCHITECTURE.md).

### Planner keyboard control

Keep the simulator and runtime in terminals 1 and 2. Exit replay, then start terminal 3 with:

```bash
bash scripts/run_planner.sh --keyboard
```

The keyboard panel sends locomotion requests to a dedicated Python/TensorRT worker. It remains responsive during engine preparation and stops only its own worker on exit. The simulator and runtime remain independent. No requests are sent at startup; default selected mode is slow walk, with speed 0.3 m/s.

| Key | Action |
|---|---|
| `0` / `1` / `2` / `3` | Select idle / slow walk / walk / run |
| `4`–`9` | Select the corresponding mode shown in the panel |
| `[` / `]` | Select the previous/next mode across the planner's 0–26 modes |
| Hold `w` / `s` | Move forward/backward relative to the selected heading |
| Hold `a` / `d` | Strafe left/right relative to the selected heading |
| `q` / `e` | Rotate the selected heading left/right by 10 degrees |
| `+` / `-` | Increase/decrease speed by 0.1 m/s, within 0–7.5 m/s |
| Enter | Activate/deactivate the selected mode, including stationary modes |
| Space or `x` | Clear keyboard requests; release body control and return to standing |
| Esc / Ctrl+C | Exit the panel and planner worker |

Mode and speed selection alone do not activate a stopped planner. Movement and heading keys activate requests once both the planner and runtime are ready. Terminal input has no key-release events: movement expires 0.35 seconds after the last repeated direction key, so keep holding the key to continue. With no movement, modes 1–3 send idle requests at zero speed; Space releases the planner's body input entirely. Losing runtime state clears keyboard intent and requires a new keypress after reconnection. Available planner modes describe the upstream generator; their H2 tracking performance still requires simulation validation.

`--model`, `--device`, `--runtime`, `--state`, `--input-bind`, `--output-bind` and `--publish-only` remain available. `--startup-timeout` bounds engine startup in keyboard mode (default 600 seconds). Worker output goes to `outputs/planner/keyboard-*.log`. Run `bash scripts/run_planner.sh` without `--keyboard` for the standalone service used by programmatic clients and teleoperation.

`terminal.py` provides the shared terminal display; `commands/status.py` reads runtime status. Neither starts DDS nor imports the simulator.

## Offline generation

The offline ECHO-G generator lives under [`tools/text2motion_offline/echo_g/`](../../tools/text2motion_offline/echo_g/). Install it with `./install.sh text2motion`; see [text-to-motion](../../docs/TEXT2MOTION.md). Its outputs replay in the minimal input environment without loading generation models.
