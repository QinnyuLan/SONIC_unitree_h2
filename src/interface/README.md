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

The shared Python motion API is `from interface.motion_replay import Motion, SmplMotion`.

## Planner and teleoperation

Install the planner or teleoperation extras with `./install.sh planner` or `./install.sh teleop`. Start the policy runtime before sending commands; see [deployment](../../docs/DEPLOYMENT.md) and [message definitions](../../docs/ARCHITECTURE.md).

## Offline generation

The offline ECHO-G generator lives under [`tools/text2motion_offline/echo_g/`](../../tools/text2motion_offline/echo_g/). Install it with `./install.sh text2motion`; see [text-to-motion](../../docs/TEXT2MOTION.md). Its outputs replay in the minimal input environment without loading generation models.
