#include <gtest/gtest.h>

#include "../include/policy_parameters.hpp"

TEST(H2NativePolicy, ProfileSelectionIsIndependentFromLegacyLora) {
  EXPECT_TRUE(IsH2NativeProfile("h2_native"));
  EXPECT_FALSE(IsH2LoraProfile("h2_native"));
  EXPECT_TRUE(IsH2Profile("h2_native"));
  EXPECT_EQ(ProfileMotorCount("h2_native"), H2_NUM_MOTOR);
  EXPECT_TRUE(IsH2NativeV11Profile("h2_native_v1_1"));
  EXPECT_TRUE(IsH2NativeProfile("h2_native_v1_1"));
  EXPECT_TRUE(IsH2Profile("h2_native_v1_1"));
  EXPECT_FALSE(IsH2LoraProfile("h2_native_v1_1"));
  EXPECT_EQ(ProfileMotorCount("h2_native_v1_1"), H2_NUM_MOTOR);
}

TEST(H2NativePolicy, PolicyMappingIsNameConsistent) {
  for (int policy_index = 0; policy_index < POLICY_ACTION_DIM; ++policy_index) {
    const int local_index = h2_native_policy_to_h2_mujoco[policy_index];
    const int dds_index = h2_native_policy_to_h2_dds[policy_index];
    EXPECT_STREQ(h2_joint_names_mujoco[local_index], h2_joint_names_dds[dds_index]);
    EXPECT_DOUBLE_EQ(
        h2_native_default_angles_policy[policy_index],
        h2_default_angles[local_index]);
    EXPECT_GT(h2_native_action_scale_policy[policy_index], 0.0);
  }
}

TEST(H2NativePolicy, UsesOfficialH2GainsAndActionScale) {
  EXPECT_STREQ(H2_NATIVE_ACTUATOR_PROFILE, "unitree_rl_mjlab_h2");
  EXPECT_FLOAT_EQ(H2NativeKpDds(0), 200.0F);
  EXPECT_FLOAT_EQ(H2NativeKdDds(0), 4.0F);
  EXPECT_FLOAT_EQ(ProfileKp("h2_native", 0), H2NativeKpDds(0));
  EXPECT_FLOAT_EQ(ProfileKd("h2_native", 0), H2NativeKdDds(0));
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[0], 0.45);
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[2], 0.2);
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[13], 0.4125);
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[17], 0.11875);
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[23], 0.3375);
  EXPECT_DOUBLE_EQ(h2_native_action_scale_policy[25], 0.3125);
}

TEST(H2NativeKpScales, DefaultsPreserveEveryJointAndDampingCommand) {
  const H2GainScales scales;
  EXPECT_TRUE(scales.IsDefault());
  EXPECT_NO_THROW(scales.Validate());
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, H2NativeKpDds(i)), H2NativeKpDds(i));
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, 0.0F), 0.0F);
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, H2NativeKdDds(i)), H2NativeKdDds(i));
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, 8.0F, true), 8.0F);
  }
}

TEST(H2NativeKpScales, FourIndependentScalesSelectOnlyNamedOutputJoints) {
  const H2GainScales scales{0.8F, 0.9F, 1.1F, 1.2F};
  EXPECT_FALSE(scales.IsDefault());
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    const std::string name = h2_joint_names_dds[i];
    float expected_scale = 1.0F;
    if (name == "waist_pitch_joint") expected_scale = 0.8F;
    else if (name == "waist_roll_joint") expected_scale = 0.9F;
    else if (name == "left_ankle_pitch_joint" || name == "right_ankle_pitch_joint") {
      expected_scale = 1.1F;
    } else if (name == "left_ankle_roll_joint" || name == "right_ankle_roll_joint") {
      expected_scale = 1.2F;
    }
    EXPECT_FLOAT_EQ(scales.KpScaleForDdsIndex(i), expected_scale) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, H2NativeKpDds(i)),
                    H2NativeKpDds(i) * expected_scale) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, 0.0F), 0.0F) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, H2NativeKdDds(i)), H2NativeKdDds(i)) << name;
  }
  EXPECT_FLOAT_EQ(scales.ApplyKp(14, 150.0F), 120.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKp(13, 150.0F), 135.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKp(5, 40.0F), 44.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKp(4, 40.0F), 48.0F);
  EXPECT_THROW(scales.KpScaleForDdsIndex(-1), std::out_of_range);
  EXPECT_THROW(scales.KpScaleForDdsIndex(H2_NUM_MOTOR), std::out_of_range);
}

TEST(H2NativeGainScales, AnklePitchKdIsIndependentAndLeavesShutdownUnchanged) {
  H2GainScales scales;
  scales.ankle_pitch_kd = 1.2F;
  EXPECT_FALSE(scales.IsDefault());
  EXPECT_NO_THROW(scales.Validate());
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    const std::string name = h2_joint_names_dds[i];
    const bool ankle_pitch = name == "left_ankle_pitch_joint" ||
                             name == "right_ankle_pitch_joint";
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, H2NativeKdDds(i)),
                    H2NativeKdDds(i) * (ankle_pitch ? 1.2F : 1.0F)) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, H2NativeKpDds(i)), H2NativeKpDds(i)) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, 8.0F, true), 8.0F) << name;
  }
  EXPECT_FLOAT_EQ(scales.ApplyKd(5, 2.0F), 2.4F);
  EXPECT_FLOAT_EQ(scales.ApplyKd(11, 2.0F), 2.4F);
  scales.ankle_pitch_kp = 1.3F;
  EXPECT_FLOAT_EQ(scales.ApplyKp(5, 40.0F), 52.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKd(5, 2.0F), 2.4F);
  EXPECT_FLOAT_EQ(scales.ApplyKd(5, 8.0F, true), 8.0F);
  EXPECT_THROW(scales.ApplyKd(-1, 2.0F), std::out_of_range);
  EXPECT_THROW(scales.ApplyKd(H2_NUM_MOTOR, 2.0F), std::out_of_range);
}

TEST(H2NativeKpScales, IndependentAnklesAndKneesOverridePairedScaleWithoutCompounding) {
  H2GainScales scales;
  scales.ankle_pitch_kp = 1.5F;
  scales.left_ankle_pitch_kp = 1.0F;
  EXPECT_FLOAT_EQ(scales.ApplyKp(5, 40.0F), 40.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKp(11, 40.0F), 60.0F);
  scales.left_ankle_pitch_kp = 1.1F;
  scales.right_ankle_pitch_kp = 1.2F;
  scales.left_knee_pitch_kp = 0.9F;
  scales.right_knee_pitch_kp = 1.3F;
  EXPECT_NO_THROW(scales.Validate());
  EXPECT_FALSE(scales.IsDefault());
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    const std::string name = h2_joint_names_dds[i];
    float expected = H2NativeKpDds(i);
    if (name == "left_ankle_pitch_joint") expected = 44.0F;
    if (name == "right_ankle_pitch_joint") expected = 48.0F;
    if (name == "left_knee_joint") expected = 180.0F;
    if (name == "right_knee_joint") expected = 260.0F;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, H2NativeKpDds(i)), expected) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, H2NativeKdDds(i)), H2NativeKdDds(i)) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, 0.0F), 0.0F);
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, 8.0F, true), 8.0F);
  }
}

TEST(H2NativeKpScales, NewJointScalesRejectInvalidValues) {
  for (float bad : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::max()}) {
    for (int joint = 0; joint < 6; ++joint) {
      H2GainScales scales;
      if (joint == 0) scales.left_ankle_pitch_kp = bad;
      if (joint == 1) scales.right_ankle_pitch_kp = bad;
      if (joint == 2) scales.left_knee_pitch_kp = bad;
      if (joint == 3) scales.right_knee_pitch_kp = bad;
      if (joint == 4) scales.left_ankle_roll_kp = bad;
      if (joint == 5) scales.right_ankle_roll_kp = bad;
      EXPECT_THROW(scales.Validate(), std::invalid_argument);
    }
  }
}

TEST(H2NativeKpScales, IndependentAnkleRollOverridesPairedScale) {
  H2GainScales scales;
  scales.ankle_roll_kp = 1.5F;
  scales.left_ankle_roll_kp = 0.7F;
  EXPECT_FLOAT_EQ(scales.ApplyKp(4, 40.0F), 28.0F);
  EXPECT_FLOAT_EQ(scales.ApplyKp(10, 40.0F), 60.0F);
  scales.right_ankle_roll_kp = 0.8F;
  EXPECT_NO_THROW(scales.Validate());
  EXPECT_FALSE(scales.IsDefault());
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    const std::string name = h2_joint_names_dds[i];
    float expected = H2NativeKpDds(i);
    if (name == "left_ankle_roll_joint") expected = 28.0F;
    if (name == "right_ankle_roll_joint") expected = 32.0F;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, H2NativeKpDds(i)), expected) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, H2NativeKdDds(i)), H2NativeKdDds(i)) << name;
    EXPECT_FLOAT_EQ(scales.ApplyKp(i, 0.0F), 0.0F);
    EXPECT_FLOAT_EQ(scales.ApplyKd(i, 8.0F, true), 8.0F);
  }
  scales = {};
  scales.left_ankle_roll_kp = 0.7F;
  EXPECT_FALSE(scales.IsDefault());
  scales = {};
  scales.right_ankle_roll_kp = 0.7F;
  EXPECT_FALSE(scales.IsDefault());
  scales.ankle_roll_kp = 1.5F;
  scales.right_ankle_roll_kp = 1.0F;
  EXPECT_FLOAT_EQ(scales.ApplyKp(10, 40.0F), 40.0F);
}

TEST(H2NativeKpScales, RejectsInvalidValuesBeforeControlStarts) {
  EXPECT_FLOAT_EQ(H2GainScales::ParseValue("1"), 1.0F);
  EXPECT_FLOAT_EQ(H2GainScales::ParseValue("0.8"), 0.8F);
  EXPECT_FLOAT_EQ(H2GainScales::ParseValue("1e-1"), 0.1F);
  for (const char* value : {"", "0", "-0", "-1", "nan", "inf", "-inf",
                            "1oops", "1 2", "--no-planner", "1e100", "1e-100", "1e37"}) {
    EXPECT_THROW(H2GainScales::ParseValue(value), std::exception) << value;
  }
  H2GainScales scales;
  scales.waist_pitch_kp = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(scales.Validate(), std::invalid_argument);
  scales = {};
  scales.waist_roll_kp = -1.0F;
  EXPECT_THROW(scales.Validate(), std::invalid_argument);
  scales = {};
  scales.ankle_pitch_kp = 0.0F;
  EXPECT_THROW(scales.Validate(), std::invalid_argument);
  scales = {};
  scales.ankle_roll_kp = std::numeric_limits<float>::infinity();
  EXPECT_THROW(scales.Validate(), std::invalid_argument);
  for (float value : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    scales = {};
    scales.ankle_pitch_kd = value;
    EXPECT_THROW(scales.Validate(), std::invalid_argument);
  }
}

TEST(H2NativePolicy, PlannerReferenceMapsOnceIntoH2PhysicalSpace) {
  std::array<double, POLICY_ACTION_DIM> g1_semantic = {};
  g1_semantic.at(0) = 0.4;
  g1_semantic.at(3) = -0.2;
  g1_semantic.at(1) = -0.3;
  g1_semantic.at(4) = 0.1;
  g1_semantic.at(21) = 0.7;

  const auto h2_native =
      G1PlannerReferenceToH2NativePolicy(g1_semantic);
  const auto h2_lora =
      G1PlannerReferenceToH2LoraSemantic(g1_semantic);

  EXPECT_NEAR(h2_native.at(0), 0.36381084, 1e-7);
  EXPECT_NEAR(h2_native.at(3), -0.19694879, 1e-7);
  EXPECT_NEAR(h2_native.at(1), -0.25110686, 1e-7);
  EXPECT_NEAR(h2_native.at(4), 0.09847440, 1e-7);
  EXPECT_DOUBLE_EQ(h2_native.at(21), 0.7);
  EXPECT_NE(h2_native.at(0), h2_lora.at(0));
  EXPECT_NE(h2_native.at(3), h2_lora.at(3));
}

TEST(H2NativePolicy, ProjectedPlannerVelocityMatchesPositionDerivative) {
  std::array<double, POLICY_ACTION_DIM> positions = {};
  std::array<double, POLICY_ACTION_DIM> velocities = {};
  std::array<double, POLICY_ACTION_DIM> next_positions = {};
  constexpr double dt = 0.02;
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    positions[i] = 0.03 * (i - 14);
    velocities[i] = 0.02 * (8 - i);
    next_positions[i] = positions[i] + dt * velocities[i];
  }
  const auto projected = G1PlannerReferenceToH2NativePolicy(positions);
  const auto projected_next = G1PlannerReferenceToH2NativePolicy(next_positions);
  const auto projected_velocity = G1PlannerReferenceToH2NativePolicy(velocities);
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    EXPECT_NEAR((projected_next[i] - projected[i]) / dt,
                projected_velocity[i], 1e-12);
  }
}
