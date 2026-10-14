#ifndef INSPIRE_HANDS_HPP
#define INSPIRE_HANDS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

namespace inspire_hand_control {

constexpr std::size_t kHandDof = 6;
constexpr std::size_t kCommandDof = 12;
constexpr std::size_t kRightOffset = 0;
constexpr std::size_t kLeftOffset = kHandDof;
constexpr double kReleaseThreshold = 0.05;

using Dex3Target = std::array<double, 7>;
using InspireTarget = std::array<double, kHandDof>;
using CombinedTarget = std::array<double, kCommandDof>;

// These are the exact 7-D targets produced by the current VR Trigger path.
constexpr Dex3Target kLeftClosedReference = {
    0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5};
constexpr Dex3Target kRightClosedReference = {
    0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5};

inline bool IsFinite(double value) {
  constexpr std::uint64_t kExponentMask = UINT64_C(0x7ff0000000000000);
  return (std::bit_cast<std::uint64_t>(value) & kExponentMask) !=
         kExponentMask;
}

inline bool IsFinite(float value) {
  constexpr std::uint32_t kExponentMask = UINT32_C(0x7f800000);
  return (std::bit_cast<std::uint32_t>(value) & kExponentMask) !=
         kExponentMask;
}

inline std::optional<double> ClosureFromDex3Target(
    const Dex3Target& target,
    bool is_left) {
  const auto& reference = is_left ? kLeftClosedReference
                                  : kRightClosedReference;
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t i = 0; i < target.size(); ++i) {
    if (!IsFinite(target[i])) return std::nullopt;
    numerator += target[i] * reference[i];
    denominator += reference[i] * reference[i];
  }
  if (!IsFinite(numerator) || denominator <= 0.0) return std::nullopt;
  const double closure = numerator / denominator;
  if (!IsFinite(closure)) return std::nullopt;
  return std::clamp(closure, 0.0, 1.0);
}

inline InspireTarget UniformTarget(double closure, double max_close_ratio) {
  const double bounded_closure = std::clamp(closure, 0.0, 1.0);
  const double bounded_ratio = std::clamp(max_close_ratio, 0.2, 1.0);
  InspireTarget result{};
  result.fill(1.0 - bounded_ratio * bounded_closure);
  return result;
}

inline unitree_go::msg::dds_::MotorCmds_ BuildCommand(
    const CombinedTarget& target) {
  unitree_go::msg::dds_::MotorCmds_ command;
  command.cmds().resize(kCommandDof);
  for (std::size_t i = 0; i < target.size(); ++i) {
    command.cmds()[i].mode(0);
    command.cmds()[i].q(static_cast<float>(target[i]));
    command.cmds()[i].dq(0.0F);
    command.cmds()[i].tau(0.0F);
    command.cmds()[i].kp(0.0F);
    command.cmds()[i].kd(0.0F);
  }
  return command;
}

struct GateSnapshot {
  InspireTarget left{};
  InspireTarget right{};
  bool left_armed = false;
  bool right_armed = false;
  bool input_active = false;
  std::uint64_t generation = 0;

  bool CanPublish() const {
    return input_active && (left_armed || right_armed);
  }
};

class InputGate {
 public:
  void Update(
      bool has_left,
      const Dex3Target& left,
      bool has_right,
      const Dex3Target& right,
      double max_close_ratio,
      bool input_fresh) {
    if (!input_fresh || (!has_left && !has_right)) {
      Reset();
      return;
    }

    const bool was_publishable = state_.CanPublish();
    state_.input_active = true;
    UpdateSide(has_left, left, true, max_close_ratio);
    UpdateSide(has_right, right, false, max_close_ratio);
    if (!was_publishable && state_.CanPublish()) ++state_.generation;
  }

  void Reset() {
    if (state_.input_active || state_.left_armed || state_.right_armed) {
      ++state_.generation;
    }
    state_.input_active = false;
    state_.left_armed = false;
    state_.right_armed = false;
  }

  GateSnapshot Snapshot() const { return state_; }

 private:
  void UpdateSide(
      bool present,
      const Dex3Target& target,
      bool is_left,
      double max_close_ratio) {
    bool& armed = is_left ? state_.left_armed : state_.right_armed;
    InspireTarget& output = is_left ? state_.left : state_.right;
    if (!present) {
      armed = false;
      return;
    }

    const auto closure = ClosureFromDex3Target(target, is_left);
    if (!closure.has_value()) {
      armed = false;
      return;
    }
    if (!armed && *closure > kReleaseThreshold) return;
    armed = true;
    output = UniformTarget(*closure, max_close_ratio);
  }

  GateSnapshot state_{};
};

inline bool DecodeUsableState(
    const unitree_go::msg::dds_::MotorStates_& message,
    CombinedTarget& positions) {
  if (message.states().size() != kCommandDof) return false;
  for (std::size_t i = 0; i < kCommandDof; ++i) {
    const float raw_q = message.states()[i].q();
    const double q = static_cast<double>(raw_q);
    if (!IsFinite(raw_q) || q < 0.0 || q > 1.0 ||
        message.states()[i].lost() != 0) {
      return false;
    }
    positions[i] = q;
  }
  return true;
}

inline CombinedTarget ComposeTarget(
    const GateSnapshot& gate,
    const CombinedTarget& measured) {
  CombinedTarget result = measured;
  if (gate.right_armed) {
    std::copy(gate.right.begin(), gate.right.end(),
              result.begin() + kRightOffset);
  }
  if (gate.left_armed) {
    std::copy(gate.left.begin(), gate.left.end(),
              result.begin() + kLeftOffset);
  }
  return result;
}

}  // namespace inspire_hand_control

class InspireHands {
 public:
  using Clock = std::chrono::steady_clock;

  InspireHands() = default;

  ~InspireHands() {
    if (subscriber_) subscriber_->CloseChannel();
    if (publisher_) publisher_->CloseChannel();
  }

  void initialize(const std::string& network_interface) {
    if (!network_interface.empty()) {
      unitree::robot::ChannelFactory::Instance()->Init(
          0, network_interface.c_str());
    }

    publisher_ = std::make_shared<unitree::robot::ChannelPublisher<
        unitree_go::msg::dds_::MotorCmds_>>("rt/inspire/cmd");
    subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<
        unitree_go::msg::dds_::MotorStates_>>("rt/inspire/state");
    publisher_->InitChannel();
    subscriber_->InitChannel(
        [this](const void* message) {
          const auto* incoming =
              static_cast<const unitree_go::msg::dds_::MotorStates_*>(message);
          state_buffer_.SetData(*incoming);
        },
        1);
    next_publish_time_ = Clock::now();
  }

  void SetMaxCloseRatio(double ratio) {
    if (!inspire_hand_control::IsFinite(ratio)) return;
    max_close_ratio_.store(std::clamp(ratio, 0.2, 1.0));
  }

  double GetMaxCloseRatio() const { return max_close_ratio_.load(); }

  void UpdateTargets(
      bool has_left,
      const inspire_hand_control::Dex3Target& left,
      bool has_right,
      const inspire_hand_control::Dex3Target& right,
      bool input_fresh) {
    inspire_hand_control::CombinedTarget measured{};
    const bool state_ready = ReadUsableState(Clock::now(), measured);
    std::lock_guard<std::mutex> lock(gate_mutex_);
    const auto before = gate_.Snapshot();
    gate_.Update(
        has_left, left, has_right, right, max_close_ratio_.load(),
        input_fresh && state_ready);
    const auto after = gate_.Snapshot();
    if (!before.left_armed && after.left_armed) {
      std::cout << "[InspireHands] Left Trigger input armed" << std::endl;
    }
    if (!before.right_armed && after.right_armed) {
      std::cout << "[InspireHands] Right Trigger input armed" << std::endl;
    }
    if (before.CanPublish() && !after.CanPublish()) {
      std::cout << "[InspireHands] Input or state stale; command publishing stopped"
                << std::endl;
    }
  }

  void ResetInput() {
    std::lock_guard<std::mutex> lock(gate_mutex_);
    gate_.Reset();
  }

  void writeOnce() {
    const auto now = Clock::now();
    if (now < next_publish_time_) return;
    next_publish_time_ = now + kPublishPeriod;

    // Keep ResetInput mutually exclusive with the complete DDS write. Once a
    // reset returns, no command captured before that reset can still publish.
    std::lock_guard<std::mutex> lock(gate_mutex_);
    const inspire_hand_control::GateSnapshot gate = gate_.Snapshot();
    if (!gate.CanPublish() || !publisher_) return;

    inspire_hand_control::CombinedTarget measured{};
    if (!ReadUsableState(now, measured)) {
      std::cout << "[InspireHands] State lost; command publishing stopped"
                << std::endl;
      gate_.Reset();
      return;
    }

    auto desired = inspire_hand_control::ComposeTarget(gate, measured);
    if (!has_last_output_ || gate.generation != last_generation_) {
      last_output_ = measured;
      has_last_output_ = true;
      last_generation_ = gate.generation;
    }
    for (std::size_t i = 0; i < desired.size(); ++i) {
      const double delta = std::clamp(
          desired[i] - last_output_[i], -kMaxPositionStep, kMaxPositionStep);
      last_output_[i] = std::clamp(last_output_[i] + delta, 0.0, 1.0);
    }
    publisher_->Write(inspire_hand_control::BuildCommand(last_output_));
  }

  TimestampedData<unitree_go::msg::dds_::MotorStates_> getStateWithTime() const {
    return state_buffer_.GetDataWithTime();
  }

 private:
  static constexpr std::chrono::milliseconds kPublishPeriod{50};
  static constexpr std::chrono::milliseconds kStateTimeout{500};
  static constexpr double kMaxPositionStep = 0.2;

  bool ReadUsableState(
      Clock::time_point now,
      inspire_hand_control::CombinedTarget& positions) const {
    const auto state = state_buffer_.GetDataWithTime();
    return state.data && now - state.timestamp <= kStateTimeout &&
           inspire_hand_control::DecodeUsableState(*state.data, positions);
  }

  // Declared first so it outlives the DDS subscriber callback during teardown.
  DataBuffer<unitree_go::msg::dds_::MotorStates_> state_buffer_;
  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::MotorCmds_>
      publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::MotorStates_>
      subscriber_;

  mutable std::mutex gate_mutex_;
  inspire_hand_control::InputGate gate_;
  std::atomic<double> max_close_ratio_{1.0};

  Clock::time_point next_publish_time_{};
  inspire_hand_control::CombinedTarget last_output_{};
  std::uint64_t last_generation_ = 0;
  bool has_last_output_ = false;
};

#endif  // INSPIRE_HANDS_HPP
