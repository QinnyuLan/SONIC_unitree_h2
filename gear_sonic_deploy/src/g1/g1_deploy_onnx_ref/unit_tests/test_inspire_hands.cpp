#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "hand_backend.hpp"
#include "inspire_hands.hpp"
#include "input_interface/input_interface.hpp"

using inspire_hand_control::CombinedTarget;
using inspire_hand_control::Dex3Target;
using inspire_hand_control::InputGate;

namespace {

class HandAvailabilityInput final : public InputInterface {
 public:
  void update() override {}

  void handle_input(
      MotionDataReader&,
      std::shared_ptr<const MotionSequence>&,
      int&,
      OperatorState&,
      bool&,
      DataBuffer<HeadingState>&,
      bool,
      PlannerState&,
      DataBuffer<MovementState>&,
      std::mutex&,
      bool&) override {}

  void SetPayload(bool has_left, bool has_right) {
    if (has_left) {
      std::array<double, 7> left{};
      left.fill(0.25);
      left_hand_joint_.SetData(left);
    }
    if (has_right) {
      std::array<double, 7> right{};
      right.fill(-0.25);
      right_hand_joint_.SetData(right);
    }
    SetHandJointAvailability(has_left, has_right);
  }
};

}  // namespace

TEST(HandBackend, ParsesEverySupportedBackend) {
  EXPECT_EQ(static_cast<int>(ParseHandBackend("none")),
            static_cast<int>(HandBackendKind::NONE));
  EXPECT_EQ(static_cast<int>(ParseHandBackend("dex3")),
            static_cast<int>(HandBackendKind::DEX3));
  EXPECT_EQ(static_cast<int>(ParseHandBackend("inspire")),
            static_cast<int>(HandBackendKind::INSPIRE));
  EXPECT_THROW(ParseHandBackend("serial"), std::invalid_argument);
}

TEST(HandInputAvailability, MissingSideDoesNotReuseCachedTarget) {
  HandAvailabilityInput input;
  input.SetPayload(true, true);
  EXPECT_TRUE(input.GetHandPose(true).first);
  EXPECT_TRUE(input.GetHandPose(false).first);

  input.SetPayload(true, false);
  EXPECT_TRUE(input.GetHandPose(true).first);
  EXPECT_FALSE(input.GetHandPose(false).first);
  EXPECT_TRUE(input.HasHandJoints());

  input.SetPayload(false, false);
  EXPECT_FALSE(input.GetHandPose(true).first);
  EXPECT_FALSE(input.GetHandPose(false).first);
  EXPECT_FALSE(input.HasHandJoints());
}

TEST(InspireHands, MapsCurrentVrOpenAndClosedTargets) {
  const Dex3Target open{};
  const auto left_open =
      inspire_hand_control::ClosureFromDex3Target(open, true);
  const auto right_open =
      inspire_hand_control::ClosureFromDex3Target(open, false);
  ASSERT_TRUE(left_open.has_value());
  ASSERT_TRUE(right_open.has_value());
  EXPECT_DOUBLE_EQ(*left_open, 0.0);
  EXPECT_DOUBLE_EQ(*right_open, 0.0);

  const auto left_closed = inspire_hand_control::ClosureFromDex3Target(
      inspire_hand_control::kLeftClosedReference, true);
  const auto right_closed = inspire_hand_control::ClosureFromDex3Target(
      inspire_hand_control::kRightClosedReference, false);
  ASSERT_TRUE(left_closed.has_value());
  ASSERT_TRUE(right_closed.has_value());
  EXPECT_DOUBLE_EQ(*left_closed, 1.0);
  EXPECT_DOUBLE_EQ(*right_closed, 1.0);

  const auto limited = inspire_hand_control::UniformTarget(1.0, 0.6);
  for (double q : limited) EXPECT_DOUBLE_EQ(q, 0.4);
}

TEST(InspireHands, ProjectsIntermediateDex3TargetsContinuously) {
  Dex3Target half = inspire_hand_control::kLeftClosedReference;
  for (double& q : half) q *= 0.5;
  const auto closure =
      inspire_hand_control::ClosureFromDex3Target(half, true);
  ASSERT_TRUE(closure.has_value());
  EXPECT_NEAR(*closure, 0.5, 1e-12);
  const auto target = inspire_hand_control::UniformTarget(*closure, 1.0);
  for (double q : target) EXPECT_NEAR(q, 0.5, 1e-12);
}

TEST(InspireHands, RequiresReleaseBeforePressAndRearmsAfterReset) {
  InputGate gate;
  const Dex3Target open{};
  const auto& closed = inspire_hand_control::kRightClosedReference;

  gate.Update(false, open, true, closed, 1.0, true);
  EXPECT_FALSE(gate.Snapshot().CanPublish());
  EXPECT_FALSE(gate.Snapshot().right_armed);

  gate.Update(false, open, true, open, 1.0, true);
  EXPECT_TRUE(gate.Snapshot().CanPublish());
  EXPECT_TRUE(gate.Snapshot().right_armed);
  for (double q : gate.Snapshot().right) EXPECT_DOUBLE_EQ(q, 1.0);

  gate.Update(false, open, true, closed, 1.0, true);
  for (double q : gate.Snapshot().right) EXPECT_NEAR(q, 0.0, 1e-12);

  gate.Reset();
  gate.Update(false, open, true, closed, 1.0, true);
  EXPECT_FALSE(gate.Snapshot().CanPublish());
  gate.Update(false, open, true, open, 1.0, true);
  EXPECT_TRUE(gate.Snapshot().CanPublish());
}

TEST(InspireHands, RejectsInvalidOrStaleInput) {
  InputGate gate;
  Dex3Target open{};
  gate.Update(true, open, true, open, 1.0, true);
  EXPECT_TRUE(gate.Snapshot().CanPublish());

  gate.Update(true, open, true, open, 1.0, false);
  EXPECT_FALSE(gate.Snapshot().CanPublish());

  open[3] = std::numeric_limits<double>::quiet_NaN();
  gate.Update(true, open, false, {}, 1.0, true);
  EXPECT_FALSE(gate.Snapshot().CanPublish());

  open.fill(std::numeric_limits<double>::max());
  gate.Update(true, open, false, {}, 1.0, true);
  EXPECT_FALSE(gate.Snapshot().CanPublish());
}

TEST(InspireHands, ComposesRightThenLeftAndHoldsUnarmedSide) {
  inspire_hand_control::GateSnapshot gate;
  gate.input_active = true;
  gate.left_armed = true;
  gate.left = {0.11, 0.12, 0.13, 0.14, 0.15, 0.16};
  CombinedTarget measured = {
      0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
      0.91, 0.92, 0.93, 0.94, 0.95, 0.96};

  const auto target = inspire_hand_control::ComposeTarget(gate, measured);
  for (std::size_t i = 0; i < inspire_hand_control::kHandDof; ++i) {
    EXPECT_DOUBLE_EQ(target[i], measured[i]);
    EXPECT_DOUBLE_EQ(target[inspire_hand_control::kLeftOffset + i], gate.left[i]);
  }

  const auto command = inspire_hand_control::BuildCommand(target);
  ASSERT_EQ(command.cmds().size(), inspire_hand_control::kCommandDof);
  for (std::size_t i = 0; i < target.size(); ++i) {
    EXPECT_FLOAT_EQ(command.cmds()[i].q(), static_cast<float>(target[i]));
  }
}

TEST(InspireHands, RequiresTwelveFiniteOnlineStateValues) {
  unitree_go::msg::dds_::MotorStates_ state;
  state.states().resize(inspire_hand_control::kCommandDof);
  for (std::size_t i = 0; i < state.states().size(); ++i) {
    state.states()[i].q(static_cast<float>(0.1 + 0.01 * i));
    state.states()[i].lost(0);
  }
  CombinedTarget positions{};
  EXPECT_TRUE(inspire_hand_control::DecodeUsableState(state, positions));

  state.states()[5].lost(1);
  EXPECT_FALSE(inspire_hand_control::DecodeUsableState(state, positions));
  state.states()[5].lost(0);
  state.states()[5].q(std::numeric_limits<float>::quiet_NaN());
  EXPECT_FALSE(inspire_hand_control::DecodeUsableState(state, positions));
  state.states().resize(6);
  EXPECT_FALSE(inspire_hand_control::DecodeUsableState(state, positions));
}

TEST(InspireHands, IgnoresNonFiniteCloseRatio) {
  InspireHands hands;
  hands.SetMaxCloseRatio(0.6);
  EXPECT_DOUBLE_EQ(hands.GetMaxCloseRatio(), 0.6);
  hands.SetMaxCloseRatio(std::numeric_limits<double>::quiet_NaN());
  EXPECT_DOUBLE_EQ(hands.GetMaxCloseRatio(), 0.6);
}
