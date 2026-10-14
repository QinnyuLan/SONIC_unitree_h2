# Motion Replay

Replay reads G1, H2 or SMPL trajectories and sends timed references to the independent policy runtime. A motion/audio bundle can also start a separate local or Unitree audio player.

| File | Responsibility |
|---|---|
| `motion.py` | CSV/NPZ loading, input-type detection, joint ordering and reference sampling |
| `cli.py` | Playback timing, runtime connection, command publishing and cleanup |
| `playback.py` | Bundle paths/checksums and the audio-player lifecycle |
| `__main__.py` | `python -m interface.motion_replay` entry point |

```bash
./install.sh inputs
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_g1_replay/walking_quip_360_R_002__A428 --inspect
.venv_inputs/bin/python -m interface.motion_replay \
  motions/example_text2motion_replay/echo_g_zh
```

Start the simulator or robot runtime before playback. `--inspect` only validates files. Use `--mute` for motion-only playback; see [motion formats](../../../motions/README.md) and [audio setup](../../../docs/TEXT2MOTION.md).

Offline generation lives in [`tools/text2motion_offline/echo_g/`](../../../tools/text2motion_offline/echo_g/). Replay imports no generation model. The shared Python motion API is `from interface.motion_replay import Motion, SmplMotion`.
