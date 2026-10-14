#pragma once

#include <algorithm>
#include <array>

struct HeadPoseCommand { double pitch = 0; double yaw = 0; };
#include "robot_parameters.hpp"

namespace h2_head_control {

constexpr double kPitchLower = -0.5236;
constexpr double kPitchUpper = 0.83775;
constexpr double kYawLower = -1.7453;
constexpr double kYawUpper = 1.7453;
constexpr double kMaxVelocity = 2.0;

inline std::array<double, 2> Clamp(const HeadPoseCommand& command) {
  return {
      std::clamp(command.pitch, kPitchLower, kPitchUpper),
      std::clamp(command.yaw, kYawLower, kYawUpper),
  };
}

inline std::array<double, 2> Clamp(const std::array<double, 2>& targets) {
  return {
      std::clamp(targets[0], kPitchLower, kPitchUpper),
      std::clamp(targets[1], kYawLower, kYawUpper),
  };
}

inline std::array<double, 2> Slew(
    const std::array<double, 2>& current,
    const std::array<double, 2>& desired,
    double dt,
    double max_velocity = kMaxVelocity) {
  const double max_step = std::max(0.0, dt * max_velocity);
  std::array<double, 2> result{};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = current[i] + std::clamp(desired[i] - current[i], -max_step, max_step);
  }
  return result;
}

inline void MergeTargets(
    MotorCommand& policy_command,
    const std::array<double, 2>& head_targets,
    const std::array<int, 2>& head_dds_indices) {
  for (size_t i = 0; i < head_targets.size(); ++i) {
    policy_command.q_target.at(head_dds_indices[i]) =
        static_cast<float>(head_targets[i]);
  }
}

}  // namespace h2_head_control
