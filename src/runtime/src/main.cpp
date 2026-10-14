// H2 policy runtime. Motion generators are independent network clients.
#include "observations.hpp"
#include "transport.hpp"
#include "control_policy.hpp"
#include "encoder.hpp"
#include "h2_native_contract.hpp"
#include "hand_backend.hpp"
#include "h2_head_control.hpp"
#include "dex3_hands.hpp"
#include "inspire_hands.hpp"
#include "utils.hpp"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <atomic>
#include <csignal>
#include <filesystem>
#include <thread>
#include <iostream>

using namespace sonic_runtime;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Signal(int) { interrupted = 1; }
struct Options {
  std::string network = "lo", model = "models/sonic_h2_released";
  std::string body = "tcp://127.0.0.1:5558", state = "tcp://127.0.0.1:5559";
  std::string head = "tcp://127.0.0.1:5562", hands = "tcp://127.0.0.1:5563";
  int domain = 0;
  bool sim = false, dry_run = false;
  double seconds = 0;
  HandBackendKind hand_backend = HandBackendKind::NONE;
  H2GainScales gains;
};
Options Parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "SONIC_unitree_h2 policy runtime\n"
        "  --model-dir PATH       released encoder/decoder directory\n"
        "  --network IFACE        DDS interface (default lo)\n"
        "  --domain N             DDS domain\n"
        "  --sim                  simulator CRC and motion-switcher settings\n"
        "  --dry-run              synthetic state; no DDS or motor commands\n"
        "  --seconds N            stop after N seconds (0: until interrupted)\n"
        "  --command-bind ENDPOINT  body input (default tcp://127.0.0.1:5558)\n"
        "  --state-bind ENDPOINT    JSON state (default tcp://127.0.0.1:5559)\n"
        "  --head-bind ENDPOINT     pitch/yaw (default tcp://127.0.0.1:5562)\n"
        "  --hands-bind ENDPOINT    hands (default tcp://127.0.0.1:5563)\n"
        "  --hand-backend none|inspire|dex3\n"
        "  --left-ankle-kp S --right-ankle-kp S\n"
        "  --left-ankle-roll-kp S --right-ankle-roll-kp S\n"
        "  --left-knee-kp S --right-knee-kp S\n";
      std::exit(0);
    }
    if (key == "--sim") { o.sim = true; continue; }
    if (key == "--dry-run") { o.dry_run = true; continue; }
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[i];
    if (key == "--network") o.network = value;
    else if (key == "--model-dir") o.model = value;
    else if (key == "--domain") o.domain = std::stoi(value);
    else if (key == "--seconds") o.seconds = std::stod(value);
    else if (key == "--command-bind") o.body = value;
    else if (key == "--state-bind") o.state = value;
    else if (key == "--head-bind") o.head = value;
    else if (key == "--hands-bind") o.hands = value;
    else if (key == "--hand-backend") o.hand_backend = ParseHandBackend(value);
    else if (key == "--left-ankle-kp") o.gains.left_ankle_pitch_kp = std::stof(value);
    else if (key == "--right-ankle-kp") o.gains.right_ankle_pitch_kp = std::stof(value);
    else if (key == "--left-ankle-roll-kp") o.gains.left_ankle_roll_kp = std::stof(value);
    else if (key == "--right-ankle-roll-kp") o.gains.right_ankle_roll_kp = std::stof(value);
    else if (key == "--left-knee-kp") o.gains.left_knee_pitch_kp = std::stof(value);
    else if (key == "--right-knee-kp") o.gains.right_knee_pitch_kp = std::stof(value);
    else throw std::invalid_argument("unknown option " + key);
  }
  o.gains.Validate();
  Require(o.domain >= 0 && o.domain <= 232, "invalid DDS domain");
  Require(std::isfinite(o.seconds) && o.seconds >= 0, "invalid duration");
  Require(o.dry_run || o.sim || o.network != "lo", "real hardware requires --network");
  return o;
}

class Runtime {
 public:
  explicit Runtime(Options options) : options_(std::move(options)), context_(1),
      body_(context_, options_.body, {"g1", "h2", "smpl", "vr", "token", "stand", "stop"}),
      head_(context_, options_.head, {"head", "stand"}),
      hands_(context_, options_.hands, {"hands", "stand"}),
      state_socket_(context_, zmq::socket_type::pub) {
    state_socket_.set(zmq::sockopt::linger, 0);
    state_socket_.set(zmq::sockopt::sndhwm, 2);
    state_socket_.bind(options_.state);
    const auto dir = std::filesystem::path(options_.model);
    const auto encoder_path = (dir / "model_encoder_full.onnx").string();
    const auto decoder_path = (dir / "model_decoder.onnx").string();
    h2_native_contract::ValidateDeploymentModels(encoder_path, decoder_path,
                                                "sonic_full_root_relative_v0");
    Require(encoder_.Initialize(encoder_path, false) && encoder_.CaptureGraph(), "encoder initialization failed");
    Require(policy_.Initialize(decoder_path, false) && policy_.CaptureGraph(), "decoder initialization failed");
    Require(encoder_.GetInputDimension() == 1751 && encoder_.GetTokenDimension() == 64 &&
            policy_.GetInputDimension() == 994, "model shape mismatch");
    if (!options_.dry_run) InitializeDDS();
  }
  ~Runtime() { Shutdown(); }
  void Run() {
    const auto started = Clock::now();
    auto next = started;
    while (!interrupted && !stopped_ &&
           (options_.seconds == 0 || std::chrono::duration<double>(Clock::now() - started).count() < options_.seconds)) {
      try { Tick(); }
      catch (const std::exception& e) { std::cerr << "Runtime stopped: " << e.what() << '\n'; stopped_ = true; failed_ = true; }
      next += std::chrono::milliseconds(20);
      if (next < Clock::now()) next = Clock::now();
      std::this_thread::sleep_until(next);
    }
    Shutdown();
    if (failed_) throw std::runtime_error("policy control failed");
  }
 private:
  void InitializeDDS() {
    ChannelFactory::Instance()->Init(options_.domain, options_.network);
    if (!options_.sim) {
      unitree::robot::b2::MotionSwitcherClient switcher;
      switcher.SetTimeout(5.0f); switcher.Init();
      std::string form, mode;
      Require(switcher.CheckMode(form, mode) == 0, "cannot inspect robot motion service");
      if (!mode.empty()) {
        Require(switcher.ReleaseMode() == 0, "cannot release robot motion service");
        Require(switcher.CheckMode(form, mode) == 0 && mode.empty(), "motion service is still active");
      }
    }
    publisher_ = std::make_shared<ChannelPublisher<LowCmd_>>(HG_CMD_TOPIC);
    publisher_->InitChannel();
    subscriber_ = std::make_shared<ChannelSubscriber<LowState_>>(HG_STATE_TOPIC);
    subscriber_->InitChannel([this](const void* message) {
      auto state = *static_cast<const LowState_*>(message);
      if (!options_.sim && state.crc() != Crc32Core(reinterpret_cast<uint32_t*>(&state), (sizeof(state) >> 2) - 1)) return;
      low_state_.SetData(state);
    }, 1);
    if (options_.hand_backend == HandBackendKind::DEX3) dex3_.initialize("");
    if (options_.hand_backend == HandBackendKind::INSPIRE) inspire_.initialize("");
    writer_ = std::jthread([this](std::stop_token stop) {
      auto next = Clock::now();
      while (!stop.stop_requested()) {
        Write();
        next += std::chrono::milliseconds(2);
        std::this_thread::sleep_until(next);
      }
    });
  }
  void Tick() {
    body_.Poll(); head_.Poll(); hands_.Poll();
    if (body_.inbox.Stopped()) { stopped_ = true; return; }
    const auto now = Clock::now();
    RobotState robot;
    if (options_.dry_run) {
      for (int j = 0; j < 31; ++j) robot.q[j] = H2DefaultAngleDds(j);
    } else {
      const auto state = low_state_.GetDataWithTime();
      if (!state.data) return;
      Require(now - state.timestamp < std::chrono::milliseconds(500), "robot state timed out");
      robot.root = float_to_double<4>(state.data->imu_state().quaternion());
      robot.angular_velocity = float_to_double<3>(state.data->imu_state().gyroscope());
      mode_machine_ = state.data->mode_machine();
      for (int j = 0; j < 31; ++j) {
        robot.q[j] = state.data->motor_state()[j].q();
        robot.dq[j] = state.data->motor_state()[j].dq();
        Require(std::isfinite(robot.q[j]) && std::isfinite(robot.dq[j]), "non-finite robot state");
        Require(options_.sim || std::abs(robot.dq[j]) <= 35, "joint velocity exceeds 35 rad/s");
      }
    }
    double norm = 0;
    for (double v : robot.root) norm += v * v;
    Require(std::isfinite(norm) && norm > .1, "invalid robot quaternion");
    for (auto& v : robot.root) v /= std::sqrt(norm);
    for (double v : robot.angular_velocity) Require(std::isfinite(v), "invalid gyroscope");
    if (!initialized_) {
      initial_q_ = robot.q;
      head_targets_ = {robot.q[h2_head_dds_indices[0]], robot.q[h2_head_dds_indices[1]]};
      initialized_ = true; ramp_start_ = now;
    }
    const double ramp = options_.dry_run ? 1.0 : std::min(1.0, std::chrono::duration<double>(now - ramp_start_).count() / 3.0);
    MotorCommand motor{};
    const auto* command = body_.inbox.Active(now);
    if (ramp < 1) {
      for (int j = 0; j < 31; ++j) {
        motor.q_target[j] = initial_q_[j] + ramp * (H2DefaultAngleDds(j) - initial_q_[j]);
        motor.kp[j] = ProfileInitKp("h2_native", j);
        motor.kd[j] = ProfileInitKd("h2_native", j);
      }
    } else {
      std::array<float, 64> tokens{};
      if (command && command->kind == "token") {
        observations_.ResetReference();
        tokens = command->data["tokens"].get<std::array<float, 64>>();
      } else {
        const double age = command ? std::max(0.0, (WallMilliseconds() - command->sent_at_ms) * .001) : 0;
        const auto input = observations_.Encoder(command, robot, age);
        std::copy(input.begin(), input.end(), encoder_.GetInputBuffer().begin());
        Require(encoder_.Encode(), "encoder inference failed");
        std::copy_n(encoder_.GetTokenBuffer().begin(), 64, tokens.begin());
      }
      for (float v : tokens) Require(std::isfinite(v), "non-finite token");
      const auto input = observations_.Decoder(tokens, robot, last_action_);
      std::copy(input.begin(), input.end(), policy_.GetInputBuffer().begin());
      Require(policy_.Infer(), "decoder inference failed");
      for (size_t j = 0; j < 29; ++j) {
        const double action = policy_.GetActionBuffer()[j];
        Require(std::isfinite(action), "non-finite policy action");
        last_action_[j] = std::clamp(action, -6.0, 6.0);
        motor.q_target[h2_native_policy_to_h2_dds[j]] = h2_native_default_angles_policy[j] +
            h2_native_action_scale_policy[j] * last_action_[j];
      }
      for (int j = 0; j < 31; ++j) { motor.kp[j] = H2NativeKpDds(j); motor.kd[j] = H2NativeKdDds(j); }
    }
    // Independent head commands bypass both encoder and decoder.
    const auto* head = head_.inbox.Active(now);
    const std::array<double, 2> desired = head ? std::array<double, 2>{
        std::clamp(head->data["pitch"].get<double>(), h2_head_control::kPitchLower, h2_head_control::kPitchUpper),
        std::clamp(head->data["yaw"].get<double>(), h2_head_control::kYawLower, h2_head_control::kYawUpper)} : std::array<double, 2>{0, 0};
    head_targets_ = h2_head_control::Slew(head_targets_, desired, .02);
    for (int j = 0; j < 2; ++j) {
      motor.q_target[h2_head_dds_indices[j]] = head_targets_[j];
    }
    motor_command_.SetData(motor);
    UpdateHands(hands_.inbox.Active(now));
    Json status = {{"schema", "sonic.h2.state.v1"}, {"sent_at_ms", WallMilliseconds()},
        {"phase", ramp < 1 ? "initializing" : "running"},
        {"body_kind", command ? command->kind : "stand"}, {"source", command ? command->source : ""},
        {"root_quat", robot.root}, {"joint_pos", robot.q}, {"joint_vel", robot.dq},
        {"head_active", head != nullptr}, {"head_target", head_targets_},
        {"hands_active", hands_.inbox.Active(now) != nullptr},
        {"rejected", body_.inbox.Rejected() + body_.invalid},
        {"last_rejection", body_.last_error.empty() ? body_.inbox.LastRejection() : body_.last_error}};
    state_socket_.send(zmq::buffer(status.dump()), zmq::send_flags::dontwait);
  }
  void UpdateHands(const Command* command) {
    if (options_.dry_run) return;
    const bool left = command && command->data.contains("left");
    const bool right = command && command->data.contains("right");
    const auto lq = left ? command->data["left"].get<std::array<double, 7>>() : std::array<double, 7>{};
    const auto rq = right ? command->data["right"].get<std::array<double, 7>>() : std::array<double, 7>{};
    // Serialize access with the writer; expire publishing without synthesizing a grip.
    std::lock_guard<std::mutex> lock(hand_mutex_);
    hand_fresh_ = command != nullptr;
    left_hand_fresh_ = left; right_hand_fresh_ = right;
    if (options_.hand_backend == HandBackendKind::DEX3) {
      if (left) dex3_.setAllJointsCommand(true, lq);
      if (right) dex3_.setAllJointsCommand(false, rq);
    } else if (options_.hand_backend == HandBackendKind::INSPIRE) inspire_.UpdateTargets(left, lq, right, rq, command != nullptr);
  }
  void Write() {
    const auto current = motor_command_.GetDataWithTime();
    if (!current.data || !publisher_) return;  // Never publish before the first measured pose.
    LowCmd_ out;
    out.mode_pr() = 0; out.mode_machine() = mode_machine_;
    if (Clock::now() - current.timestamp > std::chrono::milliseconds(100)) stopped_ = true;
    const bool damp = stopped_;
    for (int j = 0; j < 31; ++j) {
      auto& m = out.motor_cmd()[j];
      m.mode() = 1; m.tau() = 0; m.dq() = 0;
      m.q() = damp ? 0 : current.data->q_target[j];
      m.kp() = damp ? 0 : options_.gains.ApplyKp(j, current.data->kp[j]);
      m.kd() = damp ? 8 : options_.gains.ApplyKd(j, current.data->kd[j], false);
    }
    out.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&out), (sizeof(out) >> 2) - 1);
    publisher_->Write(out);
    std::lock_guard<std::mutex> lock(hand_mutex_);
    if (!damp && hand_fresh_) {
      if (options_.hand_backend == HandBackendKind::DEX3) dex3_.writeOnce(left_hand_fresh_, right_hand_fresh_);
      if (options_.hand_backend == HandBackendKind::INSPIRE) inspire_.writeOnce();
    }
  }
  void Shutdown() {
    stopped_ = true;
    if (writer_.joinable()) { writer_.request_stop(); writer_.join(); Write(); }
  }
  Options options_;
  zmq::context_t context_;
  Receiver body_, head_, hands_;
  zmq::socket_t state_socket_;
  EncoderEngine encoder_;
  PolicyEngine policy_;
  Observations observations_;
  DataBuffer<LowState_> low_state_;
  DataBuffer<MotorCommand> motor_command_;
  ChannelPublisherPtr<LowCmd_> publisher_;
  ChannelSubscriberPtr<LowState_> subscriber_;
  Dex3Hands dex3_;
  InspireHands inspire_;
  std::mutex hand_mutex_;
  bool hand_fresh_ = false, left_hand_fresh_ = false, right_hand_fresh_ = false, initialized_ = false;
  std::atomic<bool> stopped_{false};
  bool failed_ = false;
  std::atomic<uint8_t> mode_machine_{0};
  std::array<double, 31> initial_q_{};
  std::array<double, 29> last_action_{};
  std::array<double, 2> head_targets_{};
  Clock::time_point ramp_start_;
  std::jthread writer_;
};
}  // namespace
int main(int argc, char** argv) {
  std::signal(SIGINT, Signal); std::signal(SIGTERM, Signal);
  try { Runtime runtime(Parse(argc, argv)); runtime.Run(); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
