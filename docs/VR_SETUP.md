# PICO VR Setup

[Deployment guide](DEPLOYMENT.md#vr-teleoperation) · [中文部署指南](DEPLOYMENT_CN.md)

## Devices and service

Use a PICO 4/4 Pro headset, two controllers, and two ankle motion trackers. Connect the headset and workstation to the same low-latency network.

1. Install the matching Ubuntu package from [XRoboToolkit PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0). Start the PC Service before connecting the headset.
2. Install the PICO application from [XRoboToolkit PICO](https://github.com/XR-Robotics/XRoboToolkit-PICO/releases) on the headset.
3. In PICO's Motion Tracker application, pair both ankle trackers. Follow its calibration prompts: stand upright with controllers at your sides, then look down until the headset recognizes the trackers.
4. Open XRoboToolkit on the headset, enter the workstation's IP address, and connect. Confirm the status is `WORKING`. Enable `Head`, `Controller`, data sending, and `Full body` tracking.

## Python environment

```bash
./install.sh teleop
.venv_teleop/bin/python -m interface.teleop.pico --help
```

The installer builds the included XRoboToolkit binding for the local Python environment. The PC Service and PICO application remain separate device software.

## H2 configuration

Set the robot network interface and DDS domain in [`h2_deploy.env`](../config/h2_deploy.env). Set the hand backend in the same file, and head angles in [`h2_head_adjust.env`](../config/h2_head_adjust.env). The default head pitch is 40 degrees with fixed yaw; `right-stick` mode enables pitch adjustment. Use `HAND_BACKEND=none` when no Inspire hand service is connected.

The default VR mapper uses G1 kinematics, matching the shared SONIC planner. The G1 URDF and meshes included in the repository are required for that mapping. The controller still drives H2.

Start the runtime, PICO tracking process and teleoperation adapter separately. Add the independent [planner](PLANNER.md) service for joystick locomotion, as shown in the [deployment guide](DEPLOYMENT.md#vr-teleoperation). Use the PICO manager controls for tracking calibration. First-time device calibration and robot enabling require the operator at the robot.

For direct serial Inspire hands, run `.venv_teleop/bin/python -m interface.peripherals.serial_bridge --help` or `.venv_teleop/bin/python -m interface.peripherals.serial_command --help` for device and communication options.
