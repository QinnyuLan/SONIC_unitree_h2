#include <array>

#include <gtest/gtest.h>

#include "observation_config.hpp"

TEST(H2NativeConfig, SeparatesPolicyAndEncoderObservations) {
  const auto config = ObservationConfigParser::ParseFullConfig(
      "runtime/tests/fixtures/observation_config_h2_native.yaml");

  ASSERT_EQ(config.observations.size(), 6u);
  ASSERT_EQ(config.encoder.encoder_observations.size(), 10u);

  const std::array<size_t, 6> policy_dimensions = {64, 30, 290, 290, 290, 30};
  size_t policy_dimension = 0;
  for (const size_t dimension : policy_dimensions) policy_dimension += dimension;
  EXPECT_EQ(policy_dimension, 994u);

  const std::array<size_t, 10> encoder_dimensions = {
      4, 580, 6, 60, 240, 9, 12, 720, 60, 60};
  size_t encoder_dimension = 0;
  for (const size_t dimension : encoder_dimensions) encoder_dimension += dimension;
  EXPECT_EQ(encoder_dimension, 1751u);
  ASSERT_EQ(config.encoder.encoder_modes.size(), 4u);
  EXPECT_EQ(config.encoder.encoder_modes[0].mode_id, 0);
  EXPECT_EQ(config.encoder.encoder_modes[1].mode_id, 1);
  EXPECT_EQ(config.encoder.encoder_modes[2].mode_id, 2);
  EXPECT_EQ(config.encoder.encoder_modes[3].mode_id, 3);
  EXPECT_EQ(config.encoder.encoder_modes[3].name, "g1");
}

TEST(H2NativeConfig, H2OnlyUsesCompact640DLayout) {
  const auto config = ObservationConfigParser::ParseFullConfig(
      "runtime/tests/fixtures/observation_config_h2_compact.yaml");

  ASSERT_EQ(config.observations.size(), 6u);
  ASSERT_EQ(config.encoder.encoder_observations.size(), 2u);
  EXPECT_EQ(config.encoder.encoder_observations[0].name, "h2_native_reference");
  EXPECT_EQ(
      config.encoder.encoder_observations[1].name,
      "motion_anchor_ori_b_mf_nonflat");
  ASSERT_EQ(config.encoder.encoder_modes.size(), 1u);
  EXPECT_EQ(config.encoder.encoder_modes[0].mode_id, 0);
  EXPECT_EQ(config.encoder.encoder_modes[0].required_observations.size(), 2u);
  EXPECT_EQ(580u + 60u, 640u);
}
