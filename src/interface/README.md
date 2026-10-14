# Python Command Interfaces

These modules produce commands for the independent C++ policy runtime. Replay accepts G1, native H2 and SMPL clips. G1 replay and planner output the same sparse ten-frame reference; teleoperation sends SMPL/VR references, with separate head and hand channels.

| Module | Purpose |
|---|---|
| `interface.commands` | Validated protocol and `RuntimeClient` |
| `interface.replay` | Replay G1/H2/SMPL CSV/NPZ clips, including offline text-to-motion results |
| `interface.planner.service` | TensorRT Python planner and trajectory publisher |
| `interface.planner.client` | Locomotion requests |
| `interface.teleop.pico` | PICO tracking, calibration and pose processing |
| `interface.teleop.bridge` | Tracking-to-runtime adapter |
| `interface.peripherals.client` | Independent head pitch/yaw and hand targets |
| `interface.peripherals.serial_bridge` | Optional PICO-to-Inspire serial bridge |
| `interface.peripherals.serial_command` | Explicit Inspire serial commands |

From the repository root:

```bash
./install.sh inputs
.venv_inputs/bin/python -m interface.replay \
  motions/example_g1_input/walking_quip_360_R_002__A428 --inspect
```

Install the planner or teleoperation extras with `./install.sh planner` or `./install.sh teleop`. Start the policy runtime before sending commands; see [deployment](../../docs/DEPLOYMENT.md) and [message definitions](../../docs/ARCHITECTURE.md).

`kinematics/` and `math/` support pose processing. `paths.py` locates the checkout's shared assets and planner weight. The package has no dependency on the simulator implementation.
