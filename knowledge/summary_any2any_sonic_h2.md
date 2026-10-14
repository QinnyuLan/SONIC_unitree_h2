# Any2Any / SONIC G1 -> Unitree H2 Notes

Paper: https://arxiv.org/abs/2605.23733

## Paper Method

Any2Any transfers a pretrained humanoid whole-body tracker to a new embodiment in two stages:

1. Kinematic alignment.
   - Align global observation layout so semantically equivalent terms occupy the same positions.
   - Align joint-level variables with a forward target-to-source map Phi and an inverse source-to-target map Phi+.
   - Apply the mapping to reference motion observations, proprioceptive joint observations, action history, and policy actions.
   - The paper additionally discusses hip-axis decoupling and closed-chain Jacobian corrections where needed.

2. Dynamic adaptation.
   - Keep the source policy prior mostly frozen.
   - Insert PEFT, especially LoRA, into dynamics-sensitive modules.
   - For SONIC transfer experiments, the paper states LoRA is inserted into the actor dynamics decoder and critic, while FSQ and other pretrained components remain frozen.
   - The best ablation adapts the dynamics-aware path: actor backbone, actor proprioception input projection, actor output projection, and critic backbone, while preserving reference-motion encoders more strongly.

## Current Project Findings

The SONIC training stack is in `gear_sonic/`:

- Training entry: `gear_sonic/train_agent_trl.py`.
- Release training config: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`.
- Existing H2 draft config: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`.
- Robot configs: `gear_sonic/envs/manager_env/robots/g1.py` and `gear_sonic/envs/manager_env/robots/h2.py`.
- Environment robot dispatch: `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`.
- Action/observation mapping and body-only helpers: `gear_sonic/envs/env_utils/joint_utils.py`, `gear_sonic/envs/manager_env/mdp/observations.py`, `gear_sonic/envs/wrapper/manager_env_wrapper.py`.
- Policy architecture: `gear_sonic/trl/modules/universal_token_modules.py`.

The repo already has pretrained SONIC assets:

- `sonic_release/last.pt`
- `sonic_release/config.yaml`

There is already an H2 config, but it models H2 as 31 DoF with `head_pitch_joint` and `head_yaw_joint` present. The requested transfer requires freezing those two joints at default pose so the policy-facing and action-facing body space is 29 DoF.

## Required H2-29DoF Alignment

Use G1 29 DoF as the source semantic space:

```text
left_hip_pitch, right_hip_pitch, waist_yaw,
left_hip_roll, right_hip_roll, waist_roll,
left_hip_yaw, right_hip_yaw, waist_pitch,
left_knee, right_knee,
left_shoulder_pitch, right_shoulder_pitch,
left_ankle_pitch, right_ankle_pitch,
left_shoulder_roll, right_shoulder_roll,
left_ankle_roll, right_ankle_roll,
left_shoulder_yaw, right_shoulder_yaw,
left_elbow, right_elbow,
left_wrist_roll, right_wrist_roll,
left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

H2 has the same named 29 body joints plus 2 head joints. The simplest compatible design is:

- Keep the physical H2 articulation at 31 joints if IsaacLab asset conversion needs it.
- Exclude `head_pitch_joint` and `head_yaw_joint` from policy-facing joint observations, reference joint observations, action history, action outputs, metrics, and reset noise.
- Write fixed default targets or default joint state for the head joints.
- Build H2 body-joint indices by H2 joint names, not the existing G1-only utility cache.
- Ensure all 29-D policy vectors use the G1 semantic order before entering the pretrained SONIC network.
- Convert the 29-D G1 semantic action output back to H2 actuator order before applying to the environment.

This satisfies the paper's Phi/Phi+ alignment with a mostly identity name-based correspondence. Extra hip-axis or ankle-axis corrections should be validated from joint axes and initial roll/pitch frames; H2 and G1 both show nontrivial leg/shoulder frame rotations in MJCF, so pure name permutation is the first implementation, but sim validation should check sign/direction consistency joint by joint.

## Implementation Gaps

1. `joint_utils.py` is G1-specific. It must be generalized to per-robot body joint sets or given an explicit `policy_body_joint_names`.
2. `h2.py` currently includes head joints in H2 mapping and actuators. Need a frozen-head H2 robot entry, e.g. `h2_29dof_head_fixed`.
3. `sonic_h2.yaml` currently describes H2 as 31 DoF. It should use 29 policy/action DoF, `motion_lib_num_dof: 29`, and a motion/reference path compatible with H2-29.
4. The current code supports pretrained checkpoint loading and freeze flags for encoders/decoders, but it does not expose the paper's full LoRA insertion scheme for this SONIC MLP stack. Strict Any2Any replication needs a LoRA wrapper for selected `nn.Linear` layers or a PEFT integration.
5. Before real hardware deployment, the C++/ONNX deploy side under `gear_sonic_deploy/` is still G1-oriented and would need H2 SDK/joint-order support separately from simulation training.

## Feasibility

Feasible in simulation with moderate code changes:

- Kinematic alignment and H2-29DoF freezing are straightforward and should be implemented first.
- Full fine-tuning from `sonic_release/last.pt` after alignment is immediately closer to supported code than LoRA.
- Strict paper-style Any2Any requires adding LoRA and freezing rules, then training only LoRA parameters in actor `g1_dyn` decoder and critic, with encoders/FSQ frozen.

