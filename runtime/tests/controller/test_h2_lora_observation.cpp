#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "h2_lora_observation.hpp"

namespace {

constexpr std::size_t kTokenDimension = 64;
constexpr std::size_t kAngularVelocityHistoryDimension = 30;
constexpr std::size_t kJointPositionHistoryDimension = 290;
constexpr std::size_t kJointVelocityHistoryDimension = 290;
constexpr std::size_t kActionHistoryDimension = 290;
constexpr std::size_t kGravityHistoryDimension = 30;
constexpr std::size_t kActionDimension = 29;
constexpr std::size_t kActionHistoryOffset =
    kTokenDimension + kAngularVelocityHistoryDimension +
    kJointPositionHistoryDimension + kJointVelocityHistoryDimension;
constexpr std::size_t kGravityHistoryOffset =
    kActionHistoryOffset + kActionHistoryDimension;
constexpr std::size_t kObservationDimension =
    kGravityHistoryOffset + kGravityHistoryDimension;

static_assert(kActionHistoryOffset == 674);
static_assert(kGravityHistoryOffset == 964);
static_assert(kObservationDimension == 994);

}  // namespace

TEST(H2LoraObservation, ReplacesNewestActionFrameAndPreservesGravity) {
  std::vector<double> observations(kObservationDimension);
  for (std::size_t i = 0; i < observations.size(); ++i) {
    observations.at(i) = static_cast<double>(i) + 0.25;
  }
  const auto original = observations;

  std::array<double, kActionDimension> latest_action = {};
  for (std::size_t i = 0; i < latest_action.size(); ++i) {
    latest_action.at(i) = -1000.0 - static_cast<double>(i);
  }

  h2_lora_observation::ReplaceNewestHistoryFrame(
      observations,
      kActionHistoryOffset,
      kActionHistoryDimension,
      latest_action);

  const std::size_t latest_action_offset =
      kGravityHistoryOffset - kActionDimension;
  EXPECT_EQ(latest_action_offset, 935);
  for (std::size_t i = 0; i < latest_action.size(); ++i) {
    EXPECT_DOUBLE_EQ(
        observations.at(latest_action_offset + i), latest_action.at(i));
  }
  EXPECT_DOUBLE_EQ(
      observations.at(latest_action_offset - 1),
      original.at(latest_action_offset - 1));
  for (std::size_t i = kGravityHistoryOffset;
       i < kObservationDimension;
       ++i) {
    EXPECT_DOUBLE_EQ(observations.at(i), original.at(i));
  }
}

TEST(H2LoraObservation, RejectsHistorySmallerThanOneFrame) {
  std::vector<double> observations(100, 0.0);
  std::array<double, kActionDimension> latest_action = {};

  EXPECT_THROW(
      h2_lora_observation::ReplaceNewestHistoryFrame(
          observations, 10, kActionDimension - 1, latest_action),
      std::invalid_argument);
}

TEST(H2LoraObservation, RejectsHistoryOutsideInputBuffer) {
  std::vector<double> observations(100, 0.0);
  std::array<double, kActionDimension> latest_action = {};

  EXPECT_THROW(
      h2_lora_observation::ReplaceNewestHistoryFrame(
          observations, 80, kActionDimension, latest_action),
      std::out_of_range);
}

TEST(H2LoraObservation, AlignsSmplRootHeadingBeforeBodyDifference) {
  constexpr double kSqrtHalf = 0.7071067811865476;
  const std::array<double, 4> robot_base = {1.0, 0.0, 0.0, 0.0};
  const std::array<double, 4> reference_initial_heading = {
      kSqrtHalf, 0.0, 0.0, -kSqrtHalf};
  const std::array<double, 4> apply_delta_heading = {
      kSqrtHalf, 0.0, 0.0, kSqrtHalf};

  const auto difference =
      h2_lora_observation::HeadingAlignedSmplRootDifference(
          robot_base, apply_delta_heading, reference_initial_heading);

  EXPECT_NEAR(difference[0], 1.0, 1e-12);
  EXPECT_NEAR(difference[1], 0.0, 1e-12);
  EXPECT_NEAR(difference[2], 0.0, 1e-12);
  EXPECT_NEAR(difference[3], 0.0, 1e-12);
}

TEST(H2LoraObservation, ComputesDifferenceFromNonIdentityRobotBase) {
  constexpr double kSqrtHalf = 0.7071067811865476;
  const std::array<double, 4> robot_base = {
      kSqrtHalf, 0.0, 0.0, kSqrtHalf};
  const std::array<double, 4> identity = {1.0, 0.0, 0.0, 0.0};

  const auto difference =
      h2_lora_observation::HeadingAlignedSmplRootDifference(
          robot_base, identity, identity);

  EXPECT_NEAR(difference[0], kSqrtHalf, 1e-12);
  EXPECT_NEAR(difference[1], 0.0, 1e-12);
  EXPECT_NEAR(difference[2], 0.0, 1e-12);
  EXPECT_NEAR(difference[3], -kSqrtHalf, 1e-12);
}

TEST(H2LoraObservation, V11HeadingExtractionIgnoresRobotTilt) {
  constexpr double kSqrtHalf = 0.7071067811865476;
  constexpr double kCosQuarter = 0.9800665778412416;
  constexpr double kSinQuarter = 0.1986693307950612;
  const std::array<double, 4> yaw = {kSqrtHalf, 0.0, 0.0, kSqrtHalf};
  const std::array<double, 4> pitch = {
      kCosQuarter, 0.0, kSinQuarter, 0.0};
  const auto tilted_robot = quat_mul_d(yaw, pitch);

  const auto heading =
      h2_lora_observation::SonicHeadingQuaternion(tilted_robot);
  EXPECT_NEAR(heading[0], yaw[0], 1e-12);
  EXPECT_NEAR(heading[1], 0.0, 1e-12);
  EXPECT_NEAR(heading[2], 0.0, 1e-12);
  EXPECT_NEAR(heading[3], yaw[3], 1e-12);

  const auto difference =
      h2_lora_observation::HeadingCanonicalizedSmplRootDifference(
          tilted_robot, {1.0, 0.0, 0.0, 0.0}, yaw);
  EXPECT_NEAR(difference[0], 1.0, 1e-12);
  EXPECT_NEAR(difference[1], 0.0, 1e-12);
  EXPECT_NEAR(difference[2], 0.0, 1e-12);
  EXPECT_NEAR(difference[3], 0.0, 1e-12);
}
