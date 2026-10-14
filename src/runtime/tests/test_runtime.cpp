#include <gtest/gtest.h>
#include "observations.hpp"
#include "transport.hpp"
using namespace sonic_runtime;
namespace {
Json Message(std::string kind = "g1", std::string source = "a", int seq = 1) {
  Json data = Json::object();
  if (kind == "g1" || kind == "h2") {
    data["joint_pos"] = std::vector<std::array<double, 29>>(10);
    data["joint_vel"] = std::vector<std::array<double, 29>>(10);
    data["root_quat"] = std::vector<Quat>(10, {1, 0, 0, 0});
    for (int f = 0; f < 10; ++f)
      for (int j = 0; j < 29; ++j) { data["joint_pos"][f][j] = f + .01 * j; data["joint_vel"][f][j] = 10; }
  }
  return {{"schema", "sonic.h2.command.v1"}, {"kind", kind}, {"source", source},
      {"session", "test"}, {"seq", seq}, {"sent_at_ms", 1000}, {"ttl_ms", 250}, {"data", data}};
}
}
TEST(Command, RejectsBadShapeQuaternionAndPeripheralPiggyback) {
  auto m = Message(); m["data"]["joint_pos"].erase(9);
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
  m = Message(); m["data"]["root_quat"][0] = {0, 0, 0, 0};
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
  m = Message(); m["head"] = {0, 0};
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
}
TEST(Command, RejectsNonFiniteAndFractionalTTL) {
  auto m = Message(); m["data"]["joint_pos"][0][0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
  m = Message(); m["ttl_ms"] = 200.5;
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
  m = Message("planner"); m["data"]["mode"] = uint64_t{1} << 32;
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
}
TEST(Inbox, ExpirationUsesOriginalSendTime) {
  CommandInbox box; const auto now = Clock::now();
  EXPECT_TRUE(box.Accept(ParseCommand(Message()), now, 1200));
  EXPECT_NE(box.Active(now + std::chrono::milliseconds(49)), nullptr);
  EXPECT_EQ(box.Active(now + std::chrono::milliseconds(50)), nullptr);
  EXPECT_FALSE(box.Accept(ParseCommand(Message()), now, 1250));
}
TEST(Inbox, TakeoverOldShutdownAndOutOfOrder) {
  CommandInbox box; const auto now = Clock::now();
  EXPECT_TRUE(box.Accept(ParseCommand(Message()), now, 1000));
  EXPECT_FALSE(box.Accept(ParseCommand(Message("g1", "b")), now, 1000));
  auto m = Message("g1", "b"); m["takeover"] = true;
  EXPECT_TRUE(box.Accept(ParseCommand(m), now, 1000));
  EXPECT_FALSE(box.Accept(ParseCommand(Message("stand", "a", 2)), now, 1000));
  EXPECT_EQ(box.Active(now)->source, "b");
  EXPECT_FALSE(box.Accept(ParseCommand(m), now, 1000));
}
TEST(Inbox, StopLatchesAndSessionKeysDoNotCollide) {
  CommandInbox box; auto now = Clock::now();
  auto a = Message("g1", "a:b"); a["session"] = "c";
  EXPECT_TRUE(box.Accept(ParseCommand(a), now, 1000));
  auto b = Message("g1", "a"); b["session"] = "b:c"; b["takeover"] = true;
  EXPECT_TRUE(box.Accept(ParseCommand(b), now, 1000));
  EXPECT_TRUE(box.Accept(ParseCommand(Message("stop", "emergency")), now, 1000));
  EXPECT_TRUE(box.Stopped()); EXPECT_EQ(box.Active(now), nullptr);
}
TEST(Observations, StandingNeedsNoMotionFileAndExcludesHead) {
  Observations observations; RobotState state;
  auto encoded = observations.Encoder(nullptr, state);
  EXPECT_EQ(encoded[0], 0); EXPECT_EQ(encoded[1], 1);
  for (int f = 0; f < 10; ++f)
    for (int j = 0; j < 29; ++j) EXPECT_FLOAT_EQ(encoded[4 + f * 29 + j], h2_native_default_angles_policy[j]);
  state.q[29] = .7; state.q[30] = -.8;
  EXPECT_EQ(observations.Encoder(nullptr, state), encoded);
}
TEST(Observations, SparseG1UsesNativeOrderAndPhysicalTime) {
  Observations observations; auto c = ParseCommand(Message()); RobotState state;
  auto encoded = observations.Encoder(&c, state, .05);
  EXPECT_EQ(encoded[0], 3); EXPECT_EQ(encoded[1], 0);
  EXPECT_NEAR(encoded[4 + 7], .57, 1e-6);
  EXPECT_NEAR(encoded[4 + 8 * 29], 8.5, 1e-6);
  EXPECT_NEAR(encoded[4 + 9 * 29], 9, 1e-6);
  EXPECT_EQ(encoded[294], 10);
  EXPECT_EQ(encoded[590], 1); EXPECT_EQ(encoded[593], 1);
}
TEST(Observations, NativeH2TracksBodyInsteadOfStandingWithoutG1Projection) {
  Observations observations; RobotState state;
  auto c = ParseCommand(Message("h2"));
  const auto encoded = observations.Encoder(&c, state, .05);
  EXPECT_EQ(encoded[0], 0); EXPECT_EQ(encoded[1], 1);
  EXPECT_EQ(encoded[2], 0); EXPECT_EQ(encoded[3], 0);
  for (int f = 0; f < 10; ++f)
    for (int j = 0; j < 29; ++j) {
      EXPECT_NEAR(encoded[4 + f * 29 + j], std::min(f + .5, 9.) + .01 * j, 1e-6);
      EXPECT_FLOAT_EQ(encoded[294 + f * 29 + j], 10);
    }
  EXPECT_EQ(encoded[911], 0);
  for (auto& root : c.data["root_quat"]) root = euler_z_to_quat_d(.6);
  EXPECT_NEAR(observations.Encoder(&c, state)[590], std::cos(.6), 1e-6);
  auto standing = observations.Encoder(nullptr, state);
  EXPECT_FLOAT_EQ(standing[4], h2_native_default_angles_policy[0]);
  EXPECT_EQ(standing[294], 0);
}
TEST(Command, H2WireAcceptsOnlyBodyPolicyJoints) {
  auto m = Message("h2");
  EXPECT_NO_THROW(ParseCommand(m));
  m["data"]["joint_pos"] = std::vector<std::array<double, 31>>(10);
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
  m = Message("h2"); m["data"]["joint_vel"][2][5] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(ParseCommand(m), std::invalid_argument);
}
TEST(Observations, KeepsHeadingAcrossPacketsAndResetsOnStand) {
  Observations observations; RobotState state;
  auto c = ParseCommand(Message()); observations.Encoder(&c, state);
  for (auto& root : c.data["root_quat"]) root = euler_z_to_quat_d(.5);
  auto encoded = observations.Encoder(&c, state);
  EXPECT_NEAR(encoded[590], std::cos(.5), 1e-6);
  encoded = observations.Encoder(nullptr, state);
  EXPECT_EQ(encoded[0], 0); EXPECT_EQ(encoded[590], 1);
}
TEST(Observations, VrUsesLegOrderAndClearsInactiveSlices) {
  auto m = Message("vr");
  m["data"] = {{"lower_body", Message()["data"]},
      {"positions", std::vector<Vec3>(3, {1, 2, 3})},
      {"orientations", std::vector<Quat>(3, {1, 0, 0, 0})}};
  auto c = ParseCommand(m); Observations observations; RobotState state;
  auto encoded = observations.Encoder(&c, state);
  EXPECT_EQ(encoded[0], 1); EXPECT_EQ(encoded[2], 1);
  EXPECT_NEAR(encoded[650 + 1], .03, 1e-6);
  EXPECT_EQ(encoded[890], 1); EXPECT_EQ(encoded[899], 1);
  EXPECT_EQ(encoded[4], 0); EXPECT_EQ(encoded[911], 0);
  encoded = observations.Encoder(nullptr, state);
  EXPECT_EQ(encoded[890], 0); EXPECT_EQ(encoded[899], 0);
}
TEST(Observations, SmplRotationAndWristSamplingMatchContract) {
  auto m = Message("smpl");
  m["data"] = {{"joints", std::vector<std::vector<Vec3>>(10, std::vector<Vec3>(24, {1, 0, 0}))},
    {"root_quat", std::vector<Quat>(10, euler_z_to_quat_d(M_PI / 2))},
    {"wrist_joints", std::vector<std::array<double, 6>>(10)}};
  m["data"]["wrist_joints"][1][0] = .2;
  auto c = ParseCommand(m); Observations observations; RobotState state;
  auto encoded = observations.Encoder(&c, state, .01);
  EXPECT_EQ(encoded[0], 2); EXPECT_EQ(encoded[3], 1);
  EXPECT_NEAR(encoded[911], 0, 1e-6); EXPECT_NEAR(encoded[912], -1, 1e-6);
  EXPECT_NEAR(encoded[1691], .1, 1e-6);
}
TEST(Observations, DecoderHistoryIsOldestFirstWithInitialPadding) {
  Observations observations; RobotState state;
  for (int j = 0; j < 31; ++j) state.q[j] = H2DefaultAngleDds(j);
  state.angular_velocity[0] = 1;
  auto encoded = observations.Decoder({}, state, {});
  EXPECT_EQ(encoded[64], 1); EXPECT_EQ(encoded[91], 1); EXPECT_EQ(encoded[94], 0);
  state.angular_velocity[0] = 2;
  std::array<double, 29> action{}; action[0] = .5;
  encoded = observations.Decoder({}, state, action);
  EXPECT_EQ(encoded[64], 1); EXPECT_EQ(encoded[91], 2);
  EXPECT_EQ(encoded[674 + 9 * 29], .5); EXPECT_EQ(encoded[993], -1);
}
TEST(Transport, BodyAndHeadHaveIndependentOwnershipAndExpiry) {
  zmq::context_t context(1);
  Receiver body(context, "inproc://body", {"g1", "stand"});
  Receiver head(context, "inproc://head", {"head", "stand"});
  zmq::socket_t body_client(context, zmq::socket_type::push), head_client(context, zmq::socket_type::push);
  body_client.connect("inproc://body"); head_client.connect("inproc://head");
  auto m = Message(); m["sent_at_ms"] = WallMilliseconds();
  body_client.send(zmq::buffer(m.dump()), zmq::send_flags::none);
  auto h = Message("head", "head_controller"); h["sent_at_ms"] = WallMilliseconds(); h["data"] = {{"pitch", .3}, {"yaw", -.2}};
  head_client.send(zmq::buffer(h.dump()), zmq::send_flags::none);
  body.Poll(); head.Poll();
  ASSERT_NE(body.inbox.Active(Clock::now()), nullptr);
  ASSERT_NE(head.inbox.Active(Clock::now()), nullptr);
  EXPECT_EQ(body.inbox.Active(Clock::now())->kind, "g1");
  EXPECT_EQ(head.inbox.Active(Clock::now())->data["pitch"], .3);
  body_client.send(zmq::buffer(h.dump()), zmq::send_flags::none);
  body.Poll(); EXPECT_EQ(body.invalid, 1);
}
