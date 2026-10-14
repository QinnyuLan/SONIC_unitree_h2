#include "gtest/gtest.h"

#include "h2_head_control.hpp"

TEST(H2HeadControl, ClampsToUrdfJointLimits) {
  const auto clamped = h2_head_control::Clamp(HeadPoseCommand{2.0, -3.0});
  EXPECT_DOUBLE_EQ(clamped[0], h2_head_control::kPitchUpper);
  EXPECT_DOUBLE_EQ(clamped[1], h2_head_control::kYawLower);
}

TEST(H2HeadControl, SlewLimitsEachControlTick) {
  const auto result = h2_head_control::Slew({0.0, 0.0}, {0.5, -0.5}, 0.02);
  EXPECT_NEAR(result[0], 0.04, 1e-12);
  EXPECT_NEAR(result[1], -0.04, 1e-12);
}

TEST(H2HeadControl, SlewSettlesAtDesiredTarget) {
  const auto result = h2_head_control::Slew({0.1, -0.1}, {0.11, -0.12}, 0.02);
  EXPECT_DOUBLE_EQ(result[0], 0.11);
  EXPECT_DOUBLE_EQ(result[1], -0.12);
}

TEST(H2HeadControl, MergeReplacesOnlyPolicyHeadTargets) {
  MotorCommand policy_command;
  for (size_t i = 0; i < policy_command.q_target.size(); ++i) {
    policy_command.q_target[i] = static_cast<float>(i + 1);
  }
  const auto original = policy_command.q_target;

  h2_head_control::MergeTargets(policy_command, {0.3, -0.7}, {29, 30});

  for (size_t i = 0; i < 29; ++i) {
    EXPECT_FLOAT_EQ(policy_command.q_target[i], original[i]);
  }
  EXPECT_FLOAT_EQ(policy_command.q_target[29], 0.3F);
  EXPECT_FLOAT_EQ(policy_command.q_target[30], -0.7F);
}
