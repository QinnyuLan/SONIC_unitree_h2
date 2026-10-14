#pragma once
#include "command.hpp"
#include <algorithm>
#include <string_view>

namespace sonic_runtime {
enum class ControlPhase { WaitingForState, Initializing, Ready, Running, Stopped };

class RuntimeControl {
 public:
  void BeginInitialization(Clock::time_point now) {
    if (phase_ != ControlPhase::WaitingForState) return;
    initialized_at_ = now;
    phase_ = ControlPhase::Initializing;
  }
  double Ramp(Clock::time_point now) const {
    return std::clamp(std::chrono::duration<double>(now - initialized_at_).count() / 3.0, 0.0, 1.0);
  }
  void Update(Clock::time_point now) {
    if (phase_ == ControlPhase::Initializing && Ramp(now) >= 1) phase_ = ControlPhase::Ready;
  }
  // A stop wins over a start in the same input batch and cannot be undone.
  void Keys(std::string_view keys) {
    if (keys.find_first_of("oO") != std::string_view::npos) phase_ = ControlPhase::Stopped;
    else if (phase_ == ControlPhase::Ready && keys.find_first_of("]s") != std::string_view::npos)
      phase_ = ControlPhase::Running;
  }
  void Stop() { phase_ = ControlPhase::Stopped; }
  ControlPhase Phase() const { return phase_; }
  bool Running() const { return phase_ == ControlPhase::Running; }
  const char* Name() const {
    switch (phase_) {
      case ControlPhase::WaitingForState: return "waiting_for_state";
      case ControlPhase::Initializing: return "initializing";
      case ControlPhase::Ready: return "ready";
      case ControlPhase::Running: return "running";
      case ControlPhase::Stopped: return "stopped";
    }
    return "stopped";
  }
 private:
  ControlPhase phase_ = ControlPhase::WaitingForState;
  Clock::time_point initialized_at_{};
};
}  // namespace sonic_runtime
