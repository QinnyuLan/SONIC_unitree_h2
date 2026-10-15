# Sim2Real Safety and Validation

[Hardware startup](DEPLOYMENT.md#sim2real) · [中文](SIM2REAL_SAFETY_CN.md)

## Risks and staged validation

> [!WARNING]
> Real-robot testing can cause falls, collisions, equipment damage and injury. Before attempting any motion on hardware, verify stable tracking of that motion in simulation with the intended policy, reference motion and control settings. Successful simulation is a prerequisite, not a guarantee of stable hardware behavior.

Unitree H2 can exhibit a substantial simulation-to-reality gap. Pay particular attention to damping and mechanical backlash in the linkage-driven joints, especially the ankles, as well as IMU calibration and joint zero-position offsets. These differences can affect posture estimation and joint targets, alter the robot's posture and center-of-mass position, and reduce its balance margin.

Validate on hardware in the following order, advancing only after the current stage is consistently stable:

1. **Default standing:** establish stable balance in the default pose before sending motion commands.
2. **Low-difficulty motions:** verify stable tracking of slow motions with small amplitudes and limited balance demands.
3. **Progressively harder motions:** increase motion speed, amplitude and coordination demands gradually, confirming stable tracking at each step.

If standing or tracking becomes unstable, stop the test and resolve the mechanical, calibration or control issue before progressing. Passing a simpler motion does not validate a more demanding one; each candidate motion must first pass simulation.

## Before testing highly dynamic motions

- Reduce and characterize the remaining simulation-to-reality mismatch as far as practicable. Check linkage damping and backlash, particularly at the ankles, and verify IMU and joint zero-position calibration. Do not proceed with unresolved mechanical looseness or calibration errors.
- Use a clear, controlled test area with suitable footing and sufficient space for the full motion and a possible fall. Keep people outside the motion and fall zone, and have appropriate robot support and an accessible emergency stop ready.
- Fully charge the robot battery before testing and confirm normal power-system status.
- Remove dexterous hands and other fragile, nonessential attachments before testing. Disable their command backends; for the hands, set `HAND_BACKEND=none`. After removal, check that the simulation's mass and inertia settings represent the tested hardware, then repeat standing and low-difficulty validation.

These conditions apply in addition to the staged validation above. Resolve any unmet condition before attempting highly dynamic motions.
