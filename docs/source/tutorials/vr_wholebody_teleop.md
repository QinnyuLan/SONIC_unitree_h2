# PICO VR Whole-body Teleop 

Full whole-body teleoperation using PICO VR headset and controllers. To teleop, use the option  `--input-type zmq_manager` during deployment. The `zmq_manager` input type switches between a **planner mode** (locomotion commands via ZMQ) and a **streamed motion mode** (full-body SMPL poses from PICO).

```{admonition} Safety Warning
:class: danger
Whole-body teleoperation involves fast, agile motions. **Always** maintain a clear safety zone and keep a safety operator at the keyboard ready to trigger an emergency stop (**`O`** in the C++ terminal, or **A+B+X+Y** on the PICO controllers).

You **must wear tight-fitting pants or leggings** to guarantee line-of-sight for the foot trackers — loose or baggy clothing can make tracking fail unpredictably and may result in dangerous motion.
```

```{video} ../_static/teleop/teleop_session_overview.mp4
:width: 100%
```
*Video: End-to-end teleoperation walkthrough — PICO calibration, policy engagement, and the robot balancing independently. See the [Teleoperation Guide](../user_guide/teleoperation.md) for detailed best practices.*

## Prerequisites

1. **Completed the [Quick Start](../getting_started/quickstart.md)** — you can run the sim2sim loop.
2. For live hardware, **complete the [VR Teleop Setup](../getting_started/vr_teleop_setup.md)**.
   The three-point offline mode below only needs the teleop Python dependencies; it does not
   require PICO hardware, the XRoboToolkit extension, or the PC robotics service.

## Offline Three-point VR Replay

The offline backend accepts only recorded headset, left-controller, and right-controller
poses plus recorded controller inputs. It contains no generated motion and does not convert
SMPL motion into VR data. Direct three-point calibration, the manager state machine, hand
control, planner message construction, ZMQ transport, C++ deployment, and simulation remain
in the loop.

An official XRoboToolkit `RobotDataRecorder` directory can be replayed directly when it
contains `head.txt` and `controller.txt`:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --offline-vr-file /path/to/xrobotoolkit_recording \
  --offline-vr-fps 50
```

The recorder is part of the official
[XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/tree/main/RoboticsService/SDKDemo/CppSrc/RobotDataRecorder).
Its files retain the actual PICO poses and button/trigger/grip/joystick values. To exercise
the complete manager flow, the recording itself must contain the controller sequence used
to start the policy and enter `PLANNER_VR_3PT`; offline replay does not inject these events.

Normalized NPZ recordings are also accepted. They require three `[T, 7]` arrays in raw XRT
`[x, y, z, qx, qy, qz, qw]` order:

- `left_controller_pose`
- `right_controller_pose`
- `headset_pose`

Optional per-frame fields are `timestamps_ns`, `left_axis`, `right_axis`, `a_button`,
`b_button`, `x_button`, `y_button`, both menu/axis-click buttons, triggers, and grips. Missing
controller fields remain neutral. No recording is bundled with this repository.

Three tracked points cannot reconstruct the 24-joint body stream required by `POSE` mode.
Consequently, direct offline replay supports planner and `PLANNER_VR_3PT` modes and rejects
attempts to enter `POSE` instead of fabricating body joints.

---

## Step-by-Step: Teleop in SIM

Run **three terminals** to teleoperate the simulated robot.

### Terminal 1 — Launch virtual robot in MuJoCo Simulator

From the **repo root**:

```bash
# Run once, or re-run after updating the repository:
bash install_scripts/install_pico.sh

source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

### Terminal 2 — C++ Deployment

From `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

For H2, enable direct headset control of the two head motors explicitly:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --robot h2_lora --input-type zmq_manager \
  --h2-vr-head-control sim
```

In H2 simulation, `deploy.sh` loads the released G1 planner by default. The
planner is initialized from H2 LowState through the target-to-source map
`Phi`; each generated G1 reference is projected through `Phi+` to H2 and back
through `Phi` to the 29-dimensional H2LoRA encoder space. Decoder actions use
the existing `Phi+` output map before the combined 31-motor command is sent.
Do not add `--no-planner` to this teleoperation command. Real H2 deployment
keeps the planner disabled unless `--planner planner/target_vel/V2/planner_sonic.onnx`
is supplied explicitly.

The PICO process measures the headset orientation relative to the operator's
upper torso and publishes its calibrated `[pitch, yaw]` values on the independent
`head_pose` ZMQ topic. The torso frame uses the tracked upper-spine, neck, and
collar positions. Turning or bending the torso together with the head therefore
does not move the robot's head; only neck motion relative to the torso does.
The deployment process does not publish a second robot command: it replaces the
H2 profile's two default-zero head targets after the 29-dimensional policy body
action is mapped, then sends one combined 31-motor `LowCmd`. Commands older than
250 ms are rejected and the head returns smoothly to zero. Calibration occurs
with the existing **A+B+X+Y** startup combination and requires full-body tracking.

```{note}
The `--zmq-host` flag defaults to `localhost`, which is correct when both C++ deployment scripts and teleop scripts (Terminal 3) run on the same machine. If the teleop script runs on a different machine, pass `--zmq-host <IP-of-teleop-machine>`.
```

### Terminal 3 — PICO Teleop Streamer

From the **repo root**:

```bash
source .venv_teleop/bin/activate

# G1 with full visualization:
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vis_vr3pt --vis_smpl

# H2-native full release: use the native G1/teleop/SMPL reference geometry.
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --vr3pt-robot-profile g1 \
    --smpl-wrist-source g1

# Legacy H2LoRA only: use the local SMPL-to-H2 wrist projection for W(60).
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
    --smpl-wrist-source h2-kinematic \
    --vis_vr3pt --vis_smpl
```

`POSE` sends ten consecutive SMPL frames using protocol v3 and encoder mode 2. The
`smpl_root_quat` field is sent explicitly with every window. The H2LoRA online path should
select `--smpl-wrist-source h2-kinematic`. It projects the two SMPL local wrist rotations
directly onto H2's roll-pitch-yaw chains and applies the training MJCF joint limits. This is a
low-latency local approximation of the offline whole-body SOMA labels used for training, not
an exact reproduction. The selected path never falls back to the G1 wrist formula.

Before switching modes, confirm Terminal 3 printed:

```text
[PoseLoop] Pico SMPL streaming enabled (pose protocol v3, encoder mode 2, wrist=h2-kinematic)
```

For the H2-native full release, both reference options above must be `g1`.
`PLANNER` uses the frozen native G1 encoder (selector 3), and Left Stick Click
enters `PLANNER_VR_3PT` using the frozen teleop encoder (selector 1). VR targets
use G1 FK/calibration and are passed directly to that encoder; planner references
are not retargeted to H2. The decoder and body robot profile remain H2-native.
`--vr3pt-robot-profile h2-native` is an explicit alternate calibration path and
is not used by this release workflow. `--smpl-wrist-source` affects POSE mode,
not the planner VR 3-point mode.

To compare the legacy planner reference mapping on the non-v1.1 H2-native
release, add `--planner-g1-to-h2` to `deploy.sh` with the planner enabled. This
opt-in mode converts the sampled future G1 q/qdot to H2 reference coordinates
and uses selector 0. It uses the existing linear hip mapping, not full-body IK.
VR 3-point mode still uses selector 1; returning to normal planner restores
selector 0. The Pico command and the H2 decoder do not change. Remove this flag
to return to the native G1 planner branch (selector 3).

The first **A+B+X+Y** startup calibration also prints:

```text
[PoseLoop] SMPL balance neutral calibrated (joints 1-11, max residual=0.000000 deg)
```

This calibration removes the PICO standing-pose bias from the SMPL hips, legs, feet, and
three spine joints. It is captured once at policy startup and remains fixed while switching
between `PLANNER` and `POSE`; root heading and arm motion are not zeroed by it.

After entering `POSE`, Terminal 2 should report streamed motion and encoder mode `2`. It
must not report a missing `smpl_root_quat` field or an encoder fallback.

When you turn on the visualization, wait for a window to pop up showing a Unitree G1 mesh with all joints at the default angles. If no window shows up, double-check the PICO's XRoboToolKit IP configuration in the [VR Teleop Setup](../getting_started/vr_teleop_setup.md).

### Your First Teleop Session

1. **Assume the calibration pose** — stand upright, feet together, upper arms at your sides, forearms bent 90° forward (L-shape at each elbow), palms inward. See [Calibration Pose](#calibration-pose) for details.
2. Press **A + B + X + Y** simultaneously to engage the control policy and run the initial full calibration (`CALIB_FULL`).
3. Align your arms with the robot's current pose, then press **A + X** to enter full-body SMPL teleop (**POSE** mode). Move your arms and legs — the robot follows.
4. Press **A + X** again to fall back to **PLANNER** (idle) mode.
5. Press **A + B + X + Y** again to stop the robot.

<figure style="margin: 1em 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/teleop_calib/basic_flow_web.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Basic whole-body teleop workflow: calibration pose → engage → POSE mode → PLANNER idle → stop.</figcaption>
</figure>

---

(pico-controls)=

## Complete PICO Controls

### Modes & Calibration

The system has **4 operating modes** and **2 calibration types**.

**Modes:**

| Mode | Encoder | Description |
|---|---|---|
| **OFF** | -- | Policy not running. Stand in [calibration pose](#calibration-pose), then press **A+B+X+Y** to start policy. |
| **POSE** | SMPL / mode 2 | Whole-body teleop using consecutive SMPL observations streamed from PICO. |
| **PLANNER** | G1 | Locomotion planner active; upper body controller by planner. Joysticks control direction and heading in walking and running modes. |
| **PLANNER_FROZEN_UPPER** | G1 | Planner locomotion; upper body frozen at last POSE snapshot. |
| **VR_3PT** | TELEOP | Planner locomotion; upper body follows VR 3-point tracking (head + 2 hands). Depends on non-IK-based VR 3-point calibration. |

**Calibration types** (non-IK workflow for minimal latency):

| Type | What Is Calibrated | When Triggered |
|---|---|---|
| **CALIB_FULL** | SMPL legs + spine as the balance-neutral pose; head + both wrists against the **all-zero** 3-point reference. | Once on first **A+B+X+Y** (startup). |
| **CALIB** | Both wrists only against the **current robot pose**. | Each switch into VR_3PT via **Left Stick Click**. |

### State Machine

There are 4 modes and 2 control chains. Each chain forms a triangle: **A+X** (or **B+Y**) returns to POSE from *both* the planner node and its VR_3PT sub-mode.

```text
  ┌──────────────────────────────────────┐
  │  A+B+X+Y (any mode) ──► OFF          │
  └──────────────────────────────────────┘

  Startup:
    OFF ──(A+B+X+Y)──► PLANNER ──(A+X)──► POSE
          CALIB_FULL

  Chain 1 — G1 encoder listens to planner-generated full-body motion: PLANNER

    PLANNER ─── L-Stick (+CALIB) ──► VR_3PT
     ▲    │ ◄─────── L-Stick ─────────  │
     │    │                             │
     │A+X │A+X                    A+X   │
     │    ▼                             ▼
     └─ POSE ◄──────────────────────────┘

  Chain 2 — G1 encoder listens to planner-generated lower-body motion: PLANNER_FROZEN_UPPER

    PLANNER_FROZEN_UPPER ── L-Stick (+CALIB) ──► VR_3PT
         ▲     │ ◄──────── L-Stick ──────────     │
         │     │                                  │
         │B+Y  │B+Y                          B+Y  │
         │     ▼                                  ▼
         └── POSE ◄───────────────────────────────┘
```

```{admonition} DANGER — Mode-Switching Safety
:class: danger
**Before switching into POSE or VR_3PT**, always align your body with the robot's current pose first!!

- **POSE:** The robot instantly snaps to your physical pose. A large mismatch causes sudden, aggressive motion.
- **VR_3PT:** A misaligned calibration produces erratic, dangerous motion as soon as you move your arms. Check more info at section [per-switch-calib](#per-switch-calib)
```

(calibration-pose)=

### VR_3PT Calibration Hint

The `VR_3PT` mode depends on accurate calibration. Two calibration events occur:

#### One-time `CALIB_FULL` (SMPL balance + head + wrists)

Before pressing **A + B + X + Y** for the first time, you **must** stand in the robot's **all-zero reference pose**. The system captures that PICO frame for the 3-point reference, the SMPL balance-neutral reference, and the neutral head-to-torso orientation.

For `POSE`, the captured hip, knee, ankle, foot, and spine rotations become the fixed
balance-neutral reference. Later motion is sent relative to this frame, which removes a
constant forward/backward lean from the SMPL observation without changing root heading or
arm rotations. Do not bend your knees or lean during this startup capture.

**The reference pose:**
1. **Stand upright**, feet together, looking straight forward.
2. **Upper arms** hang straight down, close to your torso.
3. **Forearms** bent 90° forward (L-shape at each elbow), palms facing inward.

After calibration, robot head pitch/yaw follows the headset relative to this
tracked torso frame, not relative to the room. A three-point-only offline
recording has no torso measurement, so independent head control remains disabled
instead of falling back to global headset angles.

```{tip}
Launch the teleop script with `--vis_vr3pt` to see the robot's reference pose in a visualization window. Match your body to it before pressing the start combo.
```

(per-switch-calib)=
#### Per-switch `CALIB` (wrists only)

Each time you enter `VR_3PT` via **Left Stick Click**, the system re-calibrates both wrists against the robot's **current** pose. Always align your arms with the robot before clicking.

Below is an example of **bad calibration practice** — transitioning into VR_3PT without aligning your arms to the robot's current pose. The robot may not jump immediately, but will exhibit erratic and dangerous motion as soon as you move.

<figure style="margin: 1em 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/teleop_calib/BAD_Calib_VR3PT_web.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Bad practice: entering VR_3PT without arm alignment causes erratic, unsafe motion.</figcaption>
</figure>




```{admonition} DANGER — Mode-Switching Safety
:class: danger
**Before switching into POSE or VR_3PT**, always align your body with the robot's current pose first.

**Recovery from bad VR_3PT calibration:**
1. Freeze the upper body — switch back via **Left Stick Click**.
2. Re-align your arms with the robot's current (possibly distorted) pose.
3. Switch to **POSE** mode (**A+X**) to reset.
```

Below is the **recovery procedure** — if you accidentally enter a badly calibrated VR_3PT state, freeze the upper body (Left Stick Click back), then switch to POSE mode (A+X) to reset safely.

<figure style="margin: 1em 0;">
<video width="100%" autoplay loop muted playsinline style="border-radius: 8px;">
  <source src="../_static/teleop_calib/BAD_Calib_VR3PT_recover_web.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Recovery from bad VR_3PT calibration: freeze upper body → re-align → switch to POSE mode.</figcaption>
</figure>

### Quick-Start Cheatsheet

| Action | Button | Notes |
|---|---|---|
| **Start / Stop policy** | **A+B+X+Y** | First press: engage + CALIB_FULL. Again: emergency stop → OFF. |
| **Toggle POSE** | **A+X** | Switches between PLANNER ↔ POSE. OR from VR_3PT (entered via PLANNER) → POSE. |
| **Toggle PLANNER_FROZEN_UPPER** | **B+Y** | Switches between POSE ↔ PLANNER_FROZEN_UPPER. OR from VR_3PT (entered via PLANNER_FROZEN_UPPER) → POSE. |
| **Toggle VR_3PT** | **Left Stick Click** | From any Planner mode → VR_3PT (triggers CALIB). Click again to return. |
| **Hand grasp** | **Trigger** (per hand) | Controls the corresponding hand's grasp. |

### Inspire RH56 DDS Hand Backend (H2 PC2)

For fixed head yaw and right-stick pitch adjustment on the H2 5k release, see
[H2 head pitch adjustment](../../report/h2_head_pitch_adjustment.md).
The opt-in manager flag is `--head-control right-stick`; the normal headset
tracking mode remains the default. This adjusts a joint target and prints its
angle; it does not report measured motor position.

The Inspire backend keeps serial communication on H2 PC2. The PC2 service converts the
two RS-485 hands to the Unitree DDS topics `rt/inspire/cmd` and `rt/inspire/state`; the
deployment process on the teleoperation host publishes hand targets over `eno1`. The
configured G1-wiring port map uses stable FTDI `by-id` interface `if01` for the right
hand and `if02` for the left hand. Both hands use device ID `1`.

```{admonition} Safety Warning
:class: danger
Keep both hands fully open and clear of people and objects when starting. There must be
exactly one publisher on `rt/inspire/cmd`. Stop every old `hand_example` and `inspire_g1`
process before starting the deployment backend; otherwise competing commands can make the
hands open and close unexpectedly.
```

On **H2 PC2** (`192.168.123.162`), the native RS-485 service is managed by
systemd. Do not start a second `example_h2_native485` process manually:

```bash
ssh unitree@192.168.123.162

systemctl is-enabled h2-inspire-hands.service
systemctl is-active h2-inspire-hands.service
journalctl -u h2-inspire-hands.service -b -n 30 -o cat
```

The service uses stable FTDI `by-id` links for right interface `if01` and left
interface `if02`. The daemon and DDS endpoint stay alive if either USB port is
missing; DDS state continues at 20 Hz with that side marked `lost > 0`. Each
side retries its USB port independently at 1 Hz, in process, so intentionally
removing the hand adapter does not create a systemd restart loop. The production
service therefore does not use `--require-both`; systemd handles initial startup
and actual process crashes only.

While a hand is online, its RS-485 worker polls at 50 Hz. After consecutive
protocol failures, that side backs off to 2 Hz. A successful read immediately
returns it to 50 Hz without interrupting DDS or the other hand. Commands received
before USB loss are discarded and are never replayed on reconnection. The service
never sends a startup, recovery, or shutdown pose.

It is valid to unplug the hand FTDI adapter and remove the hands for high-dynamic
body tests. This only makes the hand channels unavailable; VR input, body policy,
and DDS remain active. However, unplugging USB is not a motor-safe command: if a
hand controller is powered separately, it may keep its last target. Release both
Triggers and disconnect/remove the hands while the robot and hands are stationary
and clear, before starting the dynamic test. After reconnecting, wait for hand
state recovery and release each Trigger once to re-arm it before commanding
closure.

On the **deployment host**, select the Inspire backend instead of Dex3 when
launching H2:

```bash
cd /home/qingyul/workspace/SONIC-H2/gear_sonic_deploy
source scripts/setup_env.sh
export GEAR_SONIC_DOMAIN_ID=0

./deploy.sh \
  --robot-profile h2_native \
  --input-type zmq_manager \
  --output-type zmq \
  --planner planner/target_vel/V2/planner_sonic.onnx \
  --h2-vr-head-control \
  --hand-backend inspire \
  eno1
```

On the teleoperation host, start the Pico manager in a second terminal:

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager \
  --vr3pt-robot-profile g1 \
  --smpl-wrist-source g1
```

The left and right controller Triggers open and close their corresponding hands. A hand
does not accept closure commands until its Trigger has first been observed released. The
backend publishes only after receiving a fresh, valid 12-axis state for both hands. If VR
hand input or Inspire state becomes stale, it stops publishing; it does not issue an
automatic open/close target and never falls back to the Dex3 backend. Release the Trigger
again after input recovers or USB reconnects to re-arm that hand; pre-disconnect commands
are not replayed.

The current VR source is binary Trigger control, so all six actuators on one Inspire hand
open or close together. This backend does not claim independent finger tracking; that would
require a separate native six-axis hand input in the teleoperation protocol.

`h2_inspire_hand_dds_demo.py` remains available as a standalone DDS monitor and bench-test
client. Monitoring is read-only and can run alongside the PC2 service:

```bash
cd /home/qingyul/workspace/SONIC-H2
source .venv_teleop/bin/activate
python gear_sonic_deploy/scripts/h2_inspire_hand_dds_demo.py \
  --network eno1 \
  --monitor-seconds 0
```

Stop the deployment backend before using the demo to command a hand, so the demo remains
the only `rt/inspire/cmd` publisher. `1.0` is fully open; this example requests 20% closure
on the right hand while holding the left hand at its measured state:

```bash
python gear_sonic_deploy/scripts/h2_inspire_hand_dds_demo.py \
  --network eno1 \
  --side right \
  --position 0.8
```

### Joystick Controls (Planner Modes)

Active in **PLANNER**, **PLANNER_FROZEN_UPPER**, and **VR_3PT**:

| Input | Function |
|---|---|
| **Left Stick** | Move direction (forward / backward / strafe) |
| **Right Stick (horizontal)** | Yaw / heading (continuous accumulation) |
| **A + B** | Next locomotion mode |
| **X + Y** | Previous locomotion mode |

**Locomotion modes** (cycled via A+B / X+Y):

| ID | Mode |
|---|---|
| 0 | Idle [DEFAULT] |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5 | Kneel (two legs) |
| 6 | Kneel |
| 7 | Lying face-down |
| 8 | Crawling |
| 9–16 | Boxing variants (idle, walk, punches, hooks) |
| 17 | Forward Jump |
| 18 | Stealth Walk |
| 19 | Injured Walk |

With the default speed scale of 0.60, Slow Walk (mode 1) requests up to
0.45 m/s at full stick. WALK still requests up to 0.60 m/s and the overall
speed cap remains 0.70 m/s. These are planner commands, not measured robot speeds.

---

## Emergency Stop

| Method | Action |
|---|---|
| **PICO controllers** | Press **A+B+X+Y** simultaneously → OFF |
| **Keyboard** (C++ terminal) | Press **`O`** for immediate stop |

---

## Step-by-Step: Teleop on Real Robot

```{admonition} Safety Warning
:class: danger
Only proceed once you can smoothly control the robot in simulation and are comfortable with emergency stops. **Terminate any running `run_sim_loop.py` process first** — simultaneous sim and real instances will conflict.
```

The real-robot workflow uses **two terminals** (no MuJoCo simulator).

### Terminal 1 — C++ Deployment (Real Robot)

From `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh

# 'real' auto-detects the robot network interface (192.168.123.x).
# If auto-detection fails, pass the G1's IP directly:
#   ./deploy.sh --input-type zmq_manager <G1-IP>
./deploy.sh --input-type zmq_manager real

# Wait until you see "Init done"
```

```{note}
If the teleop script (Terminal 2) runs on a different machine, add `--zmq-host <IP-of-teleop-machine>` so the C++ side knows where the ZMQ publisher is.
```

### Terminal 2 — PICO Teleop Streamer

From the **repo root**:

```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager

# If running offboard with a display, add visualization:
#   --vis_vr3pt --vis_smpl
```

```{note}
Update the IP in the PICO's XRoboToolKit app to match this machine before starting.
```

Follow the same start sequence: calibration pose → **A+B+X+Y** → **A+X** for POSE mode. See [Complete PICO Controls](#pico-controls) for all available commands.
