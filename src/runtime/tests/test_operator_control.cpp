#include "runtime_control.hpp"
#include "standing_pose.hpp"
#include "observations.hpp"
#include "terminal_keyboard.hpp"
#include <gtest/gtest.h>
#include <pty.h>

using namespace sonic_runtime;

TEST(RuntimeControl, InitializationDoesNotArmFromAnEarlyKeyOrElapsedTime) {
  RuntimeControl control;
  const auto now = Clock::now();
  control.Keys("s");
  control.BeginInitialization(now);
  control.Keys("]");
  control.Update(now + std::chrono::seconds(2));
  EXPECT_EQ(control.Phase(), ControlPhase::Initializing);
  control.Update(now + std::chrono::seconds(3));
  EXPECT_EQ(control.Phase(), ControlPhase::Ready);
  control.Update(now + std::chrono::hours(1));
  EXPECT_FALSE(control.Running());
  EXPECT_DOUBLE_EQ(control.Ramp(now + std::chrono::hours(1)), 1);
}

TEST(RuntimeControl, BothStartKeysWorkOnlyAfterReadyAndStopAlwaysLatches) {
  const auto now = Clock::now();
  for (const auto* key : {"s", "]"}) {
    RuntimeControl control;
    control.BeginInitialization(now);
    control.Update(now + std::chrono::seconds(3));
    control.Keys(key);
    ASSERT_TRUE(control.Running());
    control.Keys("o");
    control.Keys(key);
    control.BeginInitialization(now);
    control.Update(now + std::chrono::hours(1));
    EXPECT_EQ(control.Phase(), ControlPhase::Stopped);
  }
}

TEST(RuntimeControl, StopWinsInEveryPhaseAndWithinAKeyBatch) {
  RuntimeControl waiting;
  waiting.Keys("o]s");
  EXPECT_EQ(waiting.Phase(), ControlPhase::Stopped);
  RuntimeControl ready;
  const auto now = Clock::now();
  RuntimeControl initializing;
  initializing.BeginInitialization(now);
  initializing.Keys("o");
  initializing.Update(now + std::chrono::seconds(3));
  EXPECT_EQ(initializing.Phase(), ControlPhase::Stopped);
  ready.BeginInitialization(now);
  ready.Update(now + std::chrono::seconds(3));
  ready.Keys("s]o");
  EXPECT_EQ(ready.Phase(), ControlPhase::Stopped);
}

TEST(StandingPose, UprightReferenceDoesNotChangeDecoderTrainingOffsets) {
  const auto pose = LoadStandingPose("config/runtime/standing_pose.json");
  ASSERT_LT(pose[3], H2DefaultAngleDds(3));
  ASSERT_GT(pose[3], 0);  // Retain knee flexion instead of locking the knees.
  EXPECT_EQ(pose[3], pose[9]);
  EXPECT_EQ(pose[15], pose[22]);
  EXPECT_EQ(pose[16], -pose[23]);
  EXPECT_EQ(pose[18], pose[25]);
  EXPECT_NE(pose[18], H2DefaultAngleDds(18));
  Observations observations(StandingBodyReference(pose));
  RobotState state; state.q = pose;
  const auto encoded = observations.Encoder(nullptr, state);
  const auto decoded = observations.Decoder({}, state, {});
  for (int j = 0; j < 29; ++j) {
    EXPECT_FLOAT_EQ(encoded[4 + j], pose[h2_native_policy_to_h2_dds[j]]);
    EXPECT_FLOAT_EQ(decoded[94 + j], pose[h2_native_policy_to_h2_dds[j]] - h2_native_default_angles_policy[j]);
  }
}

TEST(TerminalKeyboard, SingleKeysNeedNoEnterAndSettingsAreRestored) {
  int master = -1, slave = -1;
  ASSERT_EQ(::openpty(&master, &slave, nullptr, nullptr, nullptr), 0);
  termios before{}, during{}, after{};
  ASSERT_EQ(::tcgetattr(slave, &before), 0);
  {
    TerminalKeyboard keyboard(slave);
    ASSERT_EQ(::tcgetattr(slave, &during), 0);
    EXPECT_EQ(during.c_lflag & (ICANON | ECHO), 0);
    EXPECT_EQ(during.c_lflag & ISIG, before.c_lflag & ISIG);
    ASSERT_EQ(::write(master, "]", 1), 1);
    pollfd descriptor{slave, POLLIN, 0};
    ASSERT_EQ(::poll(&descriptor, 1, 1000), 1);
    EXPECT_EQ(keyboard.Poll(), "]");
    ASSERT_EQ(::write(master, "\033[1;2Ss", 7), 7);
    ASSERT_EQ(::poll(&descriptor, 1, 1000), 1);
    EXPECT_EQ(keyboard.Poll(), "s");
  }
  ASSERT_EQ(::tcgetattr(slave, &after), 0);
  EXPECT_EQ(after.c_lflag, before.c_lflag);
  EXPECT_EQ(after.c_cc[VMIN], before.c_cc[VMIN]);
  EXPECT_EQ(after.c_cc[VTIME], before.c_cc[VTIME]);
  ::close(master); ::close(slave);
}
