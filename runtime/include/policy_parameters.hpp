/**
 * @file policy_parameters.hpp
 * @brief Motor constants, PID gains, joint mappings, action scales, and default
 *        standing angles for the G1 29-DOF policy.
 *
 * ## Joint Ordering
 *
 * Two ordering conventions coexist in the codebase:
 *  - **MuJoCo order** – used by the simulator, reference motions, and output
 *    interfaces.  Joint indices follow the URDF kinematic tree.
 *  - **IsaacLab order** – used internally by the RL policy and the robot SDK.
 *    Interleaves left/right joints differently.
 *
 * The arrays `isaaclab_to_mujoco` and `mujoco_to_isaaclab` provide the
 * remapping between the two orderings.
 *
 * ## PID Gain Computation
 *
 * Stiffness (Kp) and damping (Kd) values are computed from motor armature
 * constants using a second-order critically-damped model:
 *   - stiffness = armature × ω²   (ω = 10 Hz × 2π)
 *   - damping   = 2 × ζ × armature × ω   (ζ = 2.0)
 *
 * ## Action Scaling
 *
 * Policy actions are scaled by:
 *   action_scale = 0.25 × effort_limit / stiffness
 *
 * The final joint target is: target = action × action_scale + default_angle.
 */

#ifndef POLICY_PARAMETERS_HPP
#define POLICY_PARAMETERS_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "robot_parameters.hpp"

const double ONE_DEGREE = 0.0174533;  ///< One degree in radians.

// Motor armature constants (used for PID gain computation)
const double ARMATURE_5020 = 0.003609725;
const double ARMATURE_7520_14 = 0.010177520;
const double ARMATURE_7520_22 = 0.025101925;
const double ARMATURE_4010 = 0.00425;

// Control parameters for G1 PID gain computation
const double NATURAL_FREQ = 10 * 2.0 * 3.1415926535; // 10Hz
const double DAMPING_RATIO = 2;

// Computed stiffness values: stiffness = armature * natural_freq^2
const double STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;
const double STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;

// Computed damping values: damping = 2.0 * damping_ratio * armature * natural_freq
const double DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;
const double DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;
const double DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;
const double DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

// H2LoRA was trained/exported with the H2 actuator profile defaulting to
// 20 Hz.  Keep these constants separate so the original G1 deployment profile
// remains on its 10 Hz gains and action scale.
const double H2_NATURAL_FREQ = 20 * 2.0 * 3.1415926535; // 20Hz
const double H2_STIFFNESS_5020 = ARMATURE_5020 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_7520_14 = ARMATURE_7520_14 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_7520_22 = ARMATURE_7520_22 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_STIFFNESS_4010 = ARMATURE_4010 * H2_NATURAL_FREQ * H2_NATURAL_FREQ;
const double H2_DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * H2_NATURAL_FREQ;
const double H2_DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * H2_NATURAL_FREQ;
const double H2_DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * H2_NATURAL_FREQ;
const double H2_DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * H2_NATURAL_FREQ;

// H2-native officialh2 checkpoints use Unitree RL MjLab's nominal H2 actuator groups.
// These fixed gains do not use the legacy natural-frequency calculation.
constexpr const char* H2_NATIVE_ACTUATOR_PROFILE = "unitree_rl_mjlab_h2";
constexpr const char* H2_NATIVE_ACTUATOR_REFERENCE_COMMIT =
    "1425b15f73bd4095f0df53709d7c389c3eb9e790";

// Effort limits for different motor types (used for action scale computation)
const double EFFORT_LIMIT_5020 = 25.0;    // 5020 motor type
const double EFFORT_LIMIT_7520_14 = 88.0; // 7520_14 motor type
const double EFFORT_LIMIT_7520_22 = 139.0; // 7520_22 motor type
const double EFFORT_LIMIT_4010 = 5.0;     // 4010 motor type


// VR5Point index (isaaclab index) left wrist, right wrist, pelvs, left ankle, right ankle
const std::array<int, 5> vr_5point_index = {28, 29, 0, 18, 19};

// Joint mapping arrays
// VR3Point index (isaaclab index) left wrist, right wrist, torso,
const std::array<int, 3> vr_3point_index = {28, 29, 9};

// Upper body joint index (mujoco order)
const std::vector<int> upper_body_joint_mujoco_order_in_isaaclab_index = { 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
const std::vector<int> upper_body_joint_mujoco_order_in_mujoco_index = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

// upper body joint index (isaaclab order)
const std::vector<int> upper_body_joint_isaaclab_order_in_isaaclab_index = { 2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
const std::vector<int> upper_body_joint_isaaclab_order_in_mujoco_index = { 12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

// wrist joint index (mujoco order)
const std::vector<int> wrist_joint_mujoco_order_in_isaaclab_index = {23, 25, 27, 24, 26, 28};
const std::vector<int> wrist_joint_mujoco_order_in_mujoco_index = {19, 20, 21, 26, 27, 28};

// wrist joint index (isaaclab order)
const std::vector<int> wrist_joint_isaaclab_order_in_isaaclab_index = {23, 24, 25, 26, 27, 28};
const std::vector<int> wrist_joint_isaaclab_order_in_mujoco_index = {19, 26, 20, 27, 21, 28};

// lower body joint index (mujoco order)
const std::vector<int> lower_body_joint_mujoco_order_in_isaaclab_index = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18};
const std::vector<int> lower_body_joint_mujoco_order_in_mujoco_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// lower body joint index (isaaclab order)
const std::vector<int> lower_body_joint_isaaclab_order_in_isaaclab_index = {0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18};
const std::vector<int> lower_body_joint_isaaclab_order_in_mujoco_index = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

// Joint mapping arrays (mujoco order in isaaclab index)
const std::array<int, 29> isaaclab_to_mujoco = {0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                                11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
// Joint mapping arrays  (isaaclab order in mujoco index)
const std::array<int, 29> mujoco_to_isaaclab = {0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
                                                16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

const std::array<const char*, G1_NUM_MOTOR> g1_joint_names_mujoco = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
};

// Action scaling parameters
// Computed using: action_scale = 0.25 * effort_limit / stiffness
// Based on actuator configurations from IsaacLab G1_CYLINDER_CFG
const std::array<double, 29> g1_action_scale = {
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // left_hip_pitch_joint （old is 7520_14 new is 7520_22）
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // left_hip_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14, // left_hip_yaw_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // left_knee_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_ankle_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_ankle_roll_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // right_hip_pitch_joint (old is 7520_14 new is 7520_22）
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // right_hip_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14, // right_hip_yaw_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22, // right_knee_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_ankle_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_ankle_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14, // waist_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // waist_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // waist_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_shoulder_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_shoulder_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_shoulder_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_elbow_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // left_wrist_roll_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010, // left_wrist_pitch_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010, // left_wrist_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_shoulder_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_shoulder_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_shoulder_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_elbow_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020, // right_wrist_roll_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010, // right_wrist_pitch_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010, // right_wrist_yaw_joint
};

// PID control gains - Position gains (Kp)
// These values are computed based on the stiffness constants above
const std::array<float, 29> kps = {
    STIFFNESS_7520_22, // left_hip_pitch_joint (old is 7520_14 new is 7520_22）
    STIFFNESS_7520_22, // left_hip_roll_joint
    STIFFNESS_7520_14, // left_hip_yaw_joint
    STIFFNESS_7520_22, // left_knee_joint
    2.0 * STIFFNESS_5020, // left_ankle_pitch_joint
    2.0 * STIFFNESS_5020, // left_ankle_roll_joint
    STIFFNESS_7520_22, // right_hip_pitch_joint (old is 7520_14 new is 7520_22）
    STIFFNESS_7520_22, // right_hip_roll_joint
    STIFFNESS_7520_14, // right_hip_yaw_joint
    STIFFNESS_7520_22, // right_knee_joint
    2.0 * STIFFNESS_5020, // right_ankle_pitch_joint
    2.0 * STIFFNESS_5020, // right_ankle_roll_joint
    STIFFNESS_7520_14, // waist_yaw_joint
    2.0 * STIFFNESS_5020, // waist_roll_joint
    2.0 * STIFFNESS_5020, // waist_pitch_joint
    STIFFNESS_5020, // left_shoulder_pitch_joint
    STIFFNESS_5020, // left_shoulder_roll_joint
    STIFFNESS_5020, // left_shoulder_yaw_joint
    STIFFNESS_5020, // left_elbow_joint
    STIFFNESS_5020, // left_wrist_roll_joint
    STIFFNESS_4010, // left_wrist_pitch_joint
    STIFFNESS_4010, // left_wrist_yaw_joint
    STIFFNESS_5020, // right_shoulder_pitch_joint
    STIFFNESS_5020, // right_shoulder_roll_joint
    STIFFNESS_5020, // right_shoulder_yaw_joint
    STIFFNESS_5020, // right_elbow_joint
    STIFFNESS_5020, // right_wrist_roll_joint
    STIFFNESS_4010, // right_wrist_pitch_joint
    STIFFNESS_4010, // right_wrist_yaw_joint
};

// PID control gains - Derivative gains (Kd)
// These values are computed based on the damping constants above
const std::array<float, 29> kds = {
    DAMPING_7520_22, // left_hip_pitch_joint (old is 7520_14 new is 7520_22）
    DAMPING_7520_22, // left_hip_roll_joint
    DAMPING_7520_14, // left_hip_yaw_joint
    DAMPING_7520_22, // left_knee_joint
    2.0 * DAMPING_5020, // left_ankle_pitch_joint
    2.0 * DAMPING_5020, // left_ankle_roll_joint
    DAMPING_7520_22, // right_hip_pitch_joint (old is 7520_14 new is 7520_22）
    DAMPING_7520_22, // right_hip_roll_joint
    DAMPING_7520_14, // right_hip_yaw_joint
    DAMPING_7520_22, // right_knee_joint
    2.0 * DAMPING_5020, // right_ankle_pitch_joint
    2.0 * DAMPING_5020, // right_ankle_roll_joint
    DAMPING_7520_14, // waist_yaw_joint
    2.0 * DAMPING_5020, // waist_roll_joint
    2.0 * DAMPING_5020, // waist_pitch_joint
    DAMPING_5020, // left_shoulder_pitch_joint
    DAMPING_5020, // left_shoulder_roll_joint
    DAMPING_5020, // left_shoulder_yaw_joint
    DAMPING_5020, // left_elbow_joint
    DAMPING_5020, // left_wrist_roll_joint
    DAMPING_4010, // left_wrist_pitch_joint
    DAMPING_4010, // left_wrist_yaw_joint
    DAMPING_5020, // right_shoulder_pitch_joint
    DAMPING_5020, // right_shoulder_roll_joint
    DAMPING_5020, // right_shoulder_yaw_joint
    DAMPING_5020, // right_elbow_joint
    DAMPING_5020, // right_wrist_roll_joint
    DAMPING_4010, // right_wrist_pitch_joint
    DAMPING_4010, // right_wrist_yaw_joint
};

// Default joint angles (standing pose)
const std::array<double, 29> default_angles = {
    -0.312, // left_hip_pitch_joint
    0.0, // left_hip_roll_joint
    0.0, // left_hip_yaw_joint
    0.669, // left_knee_joint
    -0.363, // left_ankle_pitch_joint
    0.0, // left_ankle_roll_joint
    -0.312, // right_hip_pitch_joint
    0.0, // right_hip_roll_joint
    0.0, // right_hip_yaw_joint
    0.669, // right_knee_joint
    -0.363, // right_ankle_pitch_joint
    0.0, // right_ankle_roll_joint
    0.0, // waist_yaw_joint
    0.0, // waist_roll_joint
    0.0, // waist_pitch_joint
    0.2, // left_shoulder_pitch_joint
    0.2, // left_shoulder_roll_joint
    0.0, // left_shoulder_yaw_joint
    0.6, // left_elbow_joint
    0.0, // left_wrist_roll_joint
    0.0, // left_wrist_pitch_joint
    0.0, // left_wrist_yaw_joint
    0.2, // right_shoulder_pitch_joint
    -0.2, // right_shoulder_roll_joint
    0.0, // right_shoulder_yaw_joint
    0.6, // right_elbow_joint
    0.0, // right_wrist_roll_joint
    0.0, // right_wrist_pitch_joint
    0.0 // right_wrist_yaw_joint
};

// ---------------------------------------------------------------------------
// H2Lora deploy profile
// ---------------------------------------------------------------------------
// H2Lora keeps the released SONIC policy interface at 29 G1-semantic body
// joints and expands it to H2's 31 physical motors at the deploy boundary.
// The two head joints default to h2_default_angles and may be overwritten by
// the independent VR head command before the combined LowCmd is published.

inline bool IsH2LoraProfile(const std::string& profile) {
    return profile == "h2_lora" || profile == "h2lora" || profile == "h2";
}

inline bool IsH2NativeV11Profile(const std::string& profile) {
    return profile == "h2_native_v1_1" || profile == "h2-native-v1-1";
}

inline bool IsH2NativeProfile(const std::string& profile) {
    return profile == "h2_native" || profile == "h2-native" ||
           IsH2NativeV11Profile(profile);
}

inline bool IsH2Profile(const std::string& profile) {
    return IsH2LoraProfile(profile) || IsH2NativeProfile(profile);
}

// Local H2 MuJoCo/URDF/training order.  This is the order used by IsaacLab
// training assets and by gear_sonic/data/assets/robot_description/mjcf/h2.xml.
const std::array<const char*, H2_NUM_MOTOR> h2_joint_names_mujoco = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
};

// Official Unitree H2 HG/DDS motor order, as used by unitree_mujoco's
// h2_mujoco.xml actuator/sensor order: legs + waist + arms + head.
const std::array<const char*, H2_NUM_MOTOR> h2_joint_names_dds = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "head_pitch_joint", "head_yaw_joint",
};

// H2 local MuJoCo/training order.  Defaults mirror H2_CFG.init_state.
const std::array<double, H2_NUM_MOTOR> h2_default_angles = {
    -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
    -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
    0.0, 0.0, 0.0,
    0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
};

const std::array<float, H2_NUM_MOTOR> h2_kps = {
    H2_STIFFNESS_7520_22, H2_STIFFNESS_7520_22, H2_STIFFNESS_7520_14, H2_STIFFNESS_7520_22,
    2.0 * H2_STIFFNESS_5020, 2.0 * H2_STIFFNESS_5020,
    H2_STIFFNESS_7520_22, H2_STIFFNESS_7520_22, H2_STIFFNESS_7520_14, H2_STIFFNESS_7520_22,
    2.0 * H2_STIFFNESS_5020, 2.0 * H2_STIFFNESS_5020,
    H2_STIFFNESS_7520_14, 2.0 * H2_STIFFNESS_5020, 2.0 * H2_STIFFNESS_5020,
    2.0 * H2_STIFFNESS_5020, 2.0 * H2_STIFFNESS_5020,
    H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020,
    H2_STIFFNESS_4010, H2_STIFFNESS_4010,
    H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020, H2_STIFFNESS_5020,
    H2_STIFFNESS_4010, H2_STIFFNESS_4010,
};

const std::array<float, H2_NUM_MOTOR> h2_kds = {
    H2_DAMPING_7520_22, H2_DAMPING_7520_22, H2_DAMPING_7520_14, H2_DAMPING_7520_22,
    2.0 * H2_DAMPING_5020, 2.0 * H2_DAMPING_5020,
    H2_DAMPING_7520_22, H2_DAMPING_7520_22, H2_DAMPING_7520_14, H2_DAMPING_7520_22,
    2.0 * H2_DAMPING_5020, 2.0 * H2_DAMPING_5020,
    H2_DAMPING_7520_14, 2.0 * H2_DAMPING_5020, 2.0 * H2_DAMPING_5020,
    2.0 * H2_DAMPING_5020, 2.0 * H2_DAMPING_5020,
    H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020,
    H2_DAMPING_4010, H2_DAMPING_4010,
    H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020, H2_DAMPING_5020,
    H2_DAMPING_4010, H2_DAMPING_4010,
};

const std::array<double, H2_NUM_MOTOR> h2_action_scale = {
    0.25 * 417.0 / H2_STIFFNESS_7520_22, 0.25 * 417.0 / H2_STIFFNESS_7520_22,
    0.25 * 264.0 / H2_STIFFNESS_7520_14, 0.25 * 417.0 / H2_STIFFNESS_7520_22,
    0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020), 0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020),
    0.25 * 417.0 / H2_STIFFNESS_7520_22, 0.25 * 417.0 / H2_STIFFNESS_7520_22,
    0.25 * 264.0 / H2_STIFFNESS_7520_14, 0.25 * 417.0 / H2_STIFFNESS_7520_22,
    0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020), 0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020),
    0.25 * 264.0 / H2_STIFFNESS_7520_14,
    0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020), 0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020),
    0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020), 0.25 * 150.0 / (2.0 * H2_STIFFNESS_5020),
    0.25 * 75.0 / H2_STIFFNESS_5020, 0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 75.0 / H2_STIFFNESS_5020, 0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 15.0 / H2_STIFFNESS_4010, 0.25 * 15.0 / H2_STIFFNESS_4010,
    0.25 * 75.0 / H2_STIFFNESS_5020, 0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 75.0 / H2_STIFFNESS_5020, 0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 75.0 / H2_STIFFNESS_5020,
    0.25 * 15.0 / H2_STIFFNESS_4010, 0.25 * 15.0 / H2_STIFFNESS_4010,
};

// Official H2 DDS index -> local H2 MuJoCo/training index.
// The only structural difference is that official DDS places the two head
// motors after both arms, while the local URDF/MJCF places head before arms.
const std::array<int, H2_NUM_MOTOR> h2_dds_to_local_mujoco = {
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14,
    17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30,
    15, 16,
};

// Local H2 MuJoCo/training index -> official H2 DDS index.
const std::array<int, H2_NUM_MOTOR> h2_local_mujoco_to_dds = {
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14,
    29, 30,
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};

inline int H2DdsToLocalMujoco(int dds_index) {
    return h2_dds_to_local_mujoco.at(dds_index);
}

inline int H2LocalMujocoToDds(int local_index) {
    return h2_local_mujoco_to_dds.at(local_index);
}

inline double H2DefaultAngleDds(int dds_index) {
    return h2_default_angles.at(H2DdsToLocalMujoco(dds_index));
}

inline float H2KpDds(int dds_index) {
    return h2_kps.at(H2DdsToLocalMujoco(dds_index));
}

inline float H2KdDds(int dds_index) {
    return h2_kds.at(H2DdsToLocalMujoco(dds_index));
}

// H2 local MuJoCo/training order. Head gains conservatively extend the
// official 29-DoF profile to the local 31-DoF asset.
const std::array<float, H2_NUM_MOTOR> h2_native_kps = {
    200.0F, 200.0F, 200.0F, 200.0F, 40.0F, 40.0F,
    200.0F, 200.0F, 200.0F, 200.0F, 40.0F, 40.0F,
    150.0F, 150.0F, 150.0F, 40.0F, 40.0F,
    40.0F, 40.0F, 40.0F, 40.0F, 40.0F, 20.0F, 20.0F,
    40.0F, 40.0F, 40.0F, 40.0F, 40.0F, 20.0F, 20.0F,
};

const std::array<float, H2_NUM_MOTOR> h2_native_kds = {
    4.0F, 4.0F, 4.0F, 4.0F, 2.0F, 2.0F,
    4.0F, 4.0F, 4.0F, 4.0F, 2.0F, 2.0F,
    3.0F, 3.0F, 3.0F, 2.0F, 2.0F,
    2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 1.0F, 1.0F,
    2.0F, 2.0F, 2.0F, 2.0F, 2.0F, 1.0F, 1.0F,
};

inline float H2NativeKpDds(int dds_index) {
    return h2_native_kps.at(H2DdsToLocalMujoco(dds_index));
}

inline float H2NativeKdDds(int dds_index) {
    return h2_native_kds.at(H2DdsToLocalMujoco(dds_index));
}

struct H2GainScales {
    float waist_pitch_kp = 1.0F;
    float waist_roll_kp = 1.0F;
    float ankle_pitch_kp = 1.0F;
    float ankle_roll_kp = 1.0F;
    float ankle_pitch_kd = 1.0F;
    // Per-side ankle values replace the paired scale, independent of CLI order.
    std::optional<float> left_ankle_pitch_kp;
    std::optional<float> right_ankle_pitch_kp;
    std::optional<float> left_ankle_roll_kp;
    std::optional<float> right_ankle_roll_kp;
    float left_knee_pitch_kp = 1.0F;
    float right_knee_pitch_kp = 1.0F;

    static void ValidateValue(float value) {
        // std::isfinite can be optimized away by the release -ffast-math build.
        static_assert(sizeof(float) == sizeof(uint32_t) &&
                      std::numeric_limits<float>::is_iec559);
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U || value <= 0.0F ||
            value > std::numeric_limits<float>::max() / 200.0F) {
            throw std::invalid_argument(
                "Gain scale must be finite, positive, and keep the scaled gain representable");
        }
    }

    static float ParseValue(const std::string& text) {
        size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("Gain scale must be a single numeric value");
        }
        ValidateValue(value);
        return value;
    }

    void Validate() const {
        for (float value : {waist_pitch_kp, waist_roll_kp, ankle_pitch_kp,
                            ankle_roll_kp, ankle_pitch_kd,
                            left_ankle_pitch_kp.value_or(1.0F),
                            right_ankle_pitch_kp.value_or(1.0F),
                            left_ankle_roll_kp.value_or(1.0F),
                            right_ankle_roll_kp.value_or(1.0F),
                            left_knee_pitch_kp, right_knee_pitch_kp}) {
            ValidateValue(value);
        }
    }

    bool IsDefault() const {
        return waist_pitch_kp == 1.0F && waist_roll_kp == 1.0F &&
               ankle_pitch_kp == 1.0F && ankle_roll_kp == 1.0F &&
               ankle_pitch_kd == 1.0F &&
               left_ankle_pitch_kp.value_or(1.0F) == 1.0F &&
               right_ankle_pitch_kp.value_or(1.0F) == 1.0F &&
               left_ankle_roll_kp.value_or(1.0F) == 1.0F &&
               right_ankle_roll_kp.value_or(1.0F) == 1.0F &&
               left_knee_pitch_kp == 1.0F && right_knee_pitch_kp == 1.0F;
    }

    float KpScaleForDdsIndex(int dds_index) const {
        // Select output-coordinate joints, not closed-loop motor-side joints.
        switch (H2DdsToLocalMujoco(dds_index)) {
            case 14: return waist_pitch_kp;
            case 13: return waist_roll_kp;
            case 5: return left_ankle_pitch_kp.value_or(ankle_pitch_kp);
            case 11: return right_ankle_pitch_kp.value_or(ankle_pitch_kp);
            case 3: return left_knee_pitch_kp;
            case 9: return right_knee_pitch_kp;
            case 4: return left_ankle_roll_kp.value_or(ankle_roll_kp);
            case 10: return right_ankle_roll_kp.value_or(ankle_roll_kp);
            default: return 1.0F;
        }
    }

    float ApplyKp(int dds_index, float nominal_kp) const {
        return nominal_kp * KpScaleForDdsIndex(dds_index);
    }

    float ApplyKd(int dds_index, float nominal_kd, bool damping_only = false) const {
        const int local_index = H2DdsToLocalMujoco(dds_index);
        // Preserve explicit damping-only commands, including shutdown Kd=8.
        if (!damping_only && (local_index == 5 || local_index == 11)) {
            return nominal_kd * ankle_pitch_kd;
        }
        return nominal_kd;
    }
};

// H2Lora 29D G1-semantic body index -> H2 local MuJoCo/training motor index.
const std::array<int, POLICY_ACTION_DIM> h2lora_body_to_h2_mujoco = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    17, 24, 5, 11, 18, 25, 4, 10, 19, 26,
    20, 27, 21, 28, 22, 29, 23, 30,
};

// H2Lora 29D G1-semantic body index -> official H2 DDS motor index.
const std::array<int, POLICY_ACTION_DIM> h2lora_body_to_h2_dds = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9,
    15, 22, 5, 11, 16, 23, 4, 10, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28,
};

// H2-native ONNX contract order. It happens to match the released 29D G1
// semantic order, but it is intentionally represented as a native mapping so
// no H2/G1 kinematic transform is applied at runtime.
const std::array<int, POLICY_ACTION_DIM> h2_native_policy_to_h2_mujoco =
    h2lora_body_to_h2_mujoco;
const std::array<int, POLICY_ACTION_DIM> h2_native_policy_to_h2_dds =
    h2lora_body_to_h2_dds;

const std::array<double, POLICY_ACTION_DIM> h2_native_default_angles_policy = {
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2, 0.0,
    0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
};

const std::array<double, POLICY_ACTION_DIM> h2_native_action_scale_policy = {
    0.45, 0.45, 0.2,
    0.45, 0.45, 0.2,
    0.45, 0.45, 0.2,
    0.45, 0.45, 0.3375,
    0.3375, 0.4125, 0.4125,
    0.3375, 0.3375, 0.11875,
    0.11875, 0.3375, 0.3375,
    0.3375, 0.3375, 0.3375,
    0.3375, 0.3125, 0.3125,
    0.3125, 0.3125,
};

const std::array<int, 2> h2_head_mujoco_indices = {15, 16};
const std::array<int, 2> h2_head_dds_indices = {29, 30};

// Existing G1 action scales are stored in G1 MuJoCo order.  The policy output
// and H2Lora semantic bridge use the G1 IsaacLab/body order.  Despite the
// historical variable name, mujoco_to_isaaclab is indexed by policy/IsaacLab
// semantic index and returns the corresponding G1 MuJoCo index.
inline double G1ActionScaleSemantic(int semantic_index) {
    return g1_action_scale.at(mujoco_to_isaaclab.at(semantic_index));
}

inline double ProfileDefaultAngle(const std::string& profile, int motor_index) {
    return IsH2Profile(profile) ? H2DefaultAngleDds(motor_index) : default_angles.at(motor_index);
}

inline float ProfileKp(const std::string& profile, int motor_index) {
    if (IsH2NativeProfile(profile)) return H2NativeKpDds(motor_index);
    return IsH2LoraProfile(profile) ? H2KpDds(motor_index) : kps.at(motor_index);
}

inline float ProfileKd(const std::string& profile, int motor_index) {
    if (IsH2NativeProfile(profile)) return H2NativeKdDds(motor_index);
    return IsH2LoraProfile(profile) ? H2KdDds(motor_index) : kds.at(motor_index);
}

inline float ProfileInitKp(const std::string& profile, int motor_index) {
    return ProfileKp(profile, motor_index);
}

inline float ProfileInitKd(const std::string& profile, int motor_index) {
    return ProfileKd(profile, motor_index);
}

inline int ProfileMotorCount(const std::string& profile) {
    return IsH2Profile(profile) ? H2_NUM_MOTOR : G1_NUM_MOTOR;
}

// Matrices copied from gear_sonic.envs.env_utils.h2lora_alignment.
// target_to_source: H2 physical hip pair -> G1 semantic hip pair.
// source_to_target: G1 semantic hip pair -> H2 physical hip pair.
const std::array<std::array<double, 4>, 2> h2lora_target_to_source_hip = {{
    {0.8660245537757874, 0.0, -0.08700508624315262, 0.9847439527511597},
    {0.8660245537757874, 0.0, 0.08700508624315262, 0.9847439527511597},
}};

const std::array<std::array<double, 4>, 2> h2lora_source_to_target_hip = {{
    {0.8660245537757874, -0.08700508624315262, 0.0, 0.9847439527511597},
    {0.8660245537757874, 0.08700508624315262, 0.0, 0.9847439527511597},
}};

inline void ApplyHipPairMatrix(
    std::array<double, POLICY_ACTION_DIM>& values,
    int pitch_index,
    int roll_index,
    const std::array<double, 4>& matrix) {
    const double pitch = values.at(pitch_index);
    const double roll = values.at(roll_index);
    values.at(pitch_index) = matrix[0] * pitch + matrix[1] * roll;
    values.at(roll_index) = matrix[2] * pitch + matrix[3] * roll;
}

inline void ApplyInverseHipPairMatrix(
    std::array<double, POLICY_ACTION_DIM>& values,
    int pitch_index,
    int roll_index,
    const std::array<double, 4>& matrix) {
    const double determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2];
    const double pitch = values.at(pitch_index);
    const double roll = values.at(roll_index);
    values.at(pitch_index) = (matrix[3] * pitch - matrix[1] * roll) / determinant;
    values.at(roll_index) = (-matrix[2] * pitch + matrix[0] * roll) / determinant;
}

inline std::array<double, POLICY_ACTION_DIM> H2PhysicalToG1Semantic(
    std::array<double, POLICY_ACTION_DIM> values) {
    ApplyHipPairMatrix(values, 0, 3, h2lora_target_to_source_hip[0]);
    ApplyHipPairMatrix(values, 1, 4, h2lora_target_to_source_hip[1]);
    return values;
}

inline std::array<double, POLICY_ACTION_DIM> G1SemanticToH2Physical(
    std::array<double, POLICY_ACTION_DIM> values) {
    ApplyHipPairMatrix(values, 0, 3, h2lora_source_to_target_hip[0]);
    ApplyHipPairMatrix(values, 1, 4, h2lora_source_to_target_hip[1]);
    return values;
}

inline std::array<double, POLICY_ACTION_DIM> G1SemanticToMujocoOrder(
    const std::array<double, POLICY_ACTION_DIM>& values) {
    std::array<double, POLICY_ACTION_DIM> mujoco_values = {};
    for (int semantic_index = 0; semantic_index < POLICY_ACTION_DIM; ++semantic_index) {
        mujoco_values.at(mujoco_to_isaaclab.at(semantic_index)) = values.at(semantic_index);
    }
    return mujoco_values;
}

inline std::array<double, POLICY_ACTION_DIM> G1MujocoToSemanticOrder(
    const std::array<double, POLICY_ACTION_DIM>& values) {
    std::array<double, POLICY_ACTION_DIM> semantic_values = {};
    for (int semantic_index = 0; semantic_index < POLICY_ACTION_DIM; ++semantic_index) {
        semantic_values.at(semantic_index) = values.at(mujoco_to_isaaclab.at(semantic_index));
    }
    return semantic_values;
}

// H2LoRA robot-reference encoders were trained on Phi(q_h2).  Transform a G1
// planner reference through Phi+ and back through Phi so the online reference
// follows the same target-to-source path as the training data.
inline std::array<double, POLICY_ACTION_DIM> G1PlannerReferenceToH2LoraSemantic(
    const std::array<double, POLICY_ACTION_DIM>& g1_semantic_values) {
    return H2PhysicalToG1Semantic(G1SemanticToH2Physical(g1_semantic_values));
}

// Legacy projection for the explicit --planner-g1-to-h2 comparison mode.
// The default full-release planner uses native G1 references without this map.
inline std::array<double, POLICY_ACTION_DIM> G1PlannerReferenceToH2NativePolicy(
    const std::array<double, POLICY_ACTION_DIM>& g1_semantic_values) {
    return G1SemanticToH2Physical(g1_semantic_values);
}

inline std::array<double, POLICY_ACTION_DIM> H2LoraSemanticToG1PlannerReference(
    std::array<double, POLICY_ACTION_DIM> values) {
    // Undo Phi(Phi+) exactly when idle readaptation needs to update the raw
    // planner sequence from a target expressed in encoder semantic space.
    ApplyInverseHipPairMatrix(values, 0, 3, h2lora_target_to_source_hip[0]);
    ApplyInverseHipPairMatrix(values, 1, 4, h2lora_target_to_source_hip[1]);
    ApplyInverseHipPairMatrix(values, 0, 3, h2lora_source_to_target_hip[0]);
    ApplyInverseHipPairMatrix(values, 1, 4, h2lora_source_to_target_hip[1]);
    return values;
}

// The G1 planner consumes absolute joint positions in G1 MuJoCo order.  H2
// LowState is first gathered into the headless H2 body order used by the
// policy bridge, projected into G1 semantic space, then reordered for qpos.
inline std::array<double, POLICY_ACTION_DIM> H2PhysicalToG1PlannerMujoco(
    const std::array<double, POLICY_ACTION_DIM>& h2_physical_values) {
    return G1SemanticToMujocoOrder(H2PhysicalToG1Semantic(h2_physical_values));
}

#endif // POLICY_PARAMETERS_HPP
