#include <gtest/gtest.h>

#include "policy_parameters.hpp"

TEST(H2LoraPlannerAdapter, SemanticAndMujocoOrderRoundTrip) {
  std::array<double, POLICY_ACTION_DIM> semantic = {};
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    semantic.at(i) = static_cast<double>(i) + 0.25;
  }

  const auto mujoco = G1SemanticToMujocoOrder(semantic);
  const auto recovered = G1MujocoToSemanticOrder(mujoco);
  EXPECT_EQ(recovered, semantic);
}

TEST(H2LoraPlannerAdapter, H2DefaultPoseMapsToG1PlannerContext) {
  std::array<double, POLICY_ACTION_DIM> h2_physical = {};
  for (int semantic_index = 0; semantic_index < POLICY_ACTION_DIM; ++semantic_index) {
    h2_physical.at(semantic_index) =
        h2_default_angles.at(h2lora_body_to_h2_mujoco.at(semantic_index));
  }

  const auto planner_mujoco = H2PhysicalToG1PlannerMujoco(h2_physical);
  EXPECT_NEAR(planner_mujoco.at(0), -0.27019966, 1e-7);
  EXPECT_NEAR(planner_mujoco.at(1), 0.02714559, 1e-7);
  EXPECT_NEAR(planner_mujoco.at(6), -0.27019966, 1e-7);
  EXPECT_NEAR(planner_mujoco.at(7), -0.02714559, 1e-7);
  EXPECT_NEAR(planner_mujoco.at(3), 0.669, 1e-9);
  EXPECT_NEAR(planner_mujoco.at(9), 0.669, 1e-9);
}

TEST(H2LoraPlannerAdapter, PolicyActionMapsBackToH2PhysicalSpace) {
  std::array<double, POLICY_ACTION_DIM> g1_semantic = {};
  g1_semantic.at(0) = 0.4;
  g1_semantic.at(3) = -0.2;
  g1_semantic.at(1) = -0.3;
  g1_semantic.at(4) = 0.1;
  g1_semantic.at(21) = 0.7;

  const auto h2_physical = G1SemanticToH2Physical(g1_semantic);
  EXPECT_NEAR(h2_physical.at(0), 0.36381084, 1e-7);
  EXPECT_NEAR(h2_physical.at(3), -0.19694879, 1e-7);
  EXPECT_NEAR(h2_physical.at(1), -0.25110686, 1e-7);
  EXPECT_NEAR(h2_physical.at(4), 0.09847440, 1e-7);
  EXPECT_DOUBLE_EQ(h2_physical.at(21), 0.7);
}

TEST(H2LoraPlannerAdapter, PlannerReferenceUsesTargetRoundTrip) {
  const auto g1_default_semantic = G1MujocoToSemanticOrder(default_angles);
  const auto projected =
      G1PlannerReferenceToH2LoraSemantic(g1_default_semantic);

  EXPECT_NEAR(projected.at(0), -0.23399954, 1e-7);
  EXPECT_NEAR(projected.at(3), 0.02350874, 1e-7);
  EXPECT_NEAR(projected.at(1), -0.23399954, 1e-7);
  EXPECT_NEAR(projected.at(4), -0.02350874, 1e-7);
  EXPECT_DOUBLE_EQ(projected.at(9), g1_default_semantic.at(9));
  EXPECT_DOUBLE_EQ(projected.at(21), g1_default_semantic.at(21));
}

TEST(H2LoraPlannerAdapter, ProjectedReferenceHasExactRawInverse) {
  std::array<double, POLICY_ACTION_DIM> raw_reference = {};
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    raw_reference.at(i) = 0.03 * static_cast<double>(i - 14);
  }

  const auto projected =
      G1PlannerReferenceToH2LoraSemantic(raw_reference);
  const auto recovered =
      H2LoraSemanticToG1PlannerReference(projected);
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    EXPECT_NEAR(recovered.at(i), raw_reference.at(i), 1e-12);
  }
}

TEST(H2LoraPlannerAdapter, IdleBlendConvergesInProjectedPolicySpace) {
  std::array<double, POLICY_ACTION_DIM> raw_reference = {};
  std::array<double, POLICY_ACTION_DIM> actual_semantic = {};
  raw_reference.at(0) = -0.4;
  raw_reference.at(3) = 0.2;
  actual_semantic.at(0) = -0.15;
  actual_semantic.at(3) = -0.05;

  const auto projected_before =
      G1PlannerReferenceToH2LoraSemantic(raw_reference);
  const auto raw_adaptation_target =
      H2LoraSemanticToG1PlannerReference(actual_semantic);
  std::array<double, POLICY_ACTION_DIM> blended_raw = {};
  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    blended_raw.at(i) =
        0.98 * raw_reference.at(i) + 0.02 * raw_adaptation_target.at(i);
  }
  const auto projected_after =
      G1PlannerReferenceToH2LoraSemantic(blended_raw);

  for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
    const double expected =
        0.98 * projected_before.at(i) + 0.02 * actual_semantic.at(i);
    EXPECT_NEAR(projected_after.at(i), expected, 1e-12);
  }
}
