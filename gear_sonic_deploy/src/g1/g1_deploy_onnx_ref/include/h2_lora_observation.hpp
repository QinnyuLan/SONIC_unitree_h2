#ifndef H2_LORA_OBSERVATION_HPP
#define H2_LORA_OBSERVATION_HPP

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "math_utils.hpp"

namespace h2_lora_observation {

inline void ReplaceNewestHistoryFrame(
    std::span<double> observations,
    std::size_t history_offset,
    std::size_t history_dimension,
    std::span<const double> newest_frame) {
  if (history_dimension < newest_frame.size()) {
    throw std::invalid_argument(
        "Observation history is smaller than its newest frame");
  }
  if (history_offset > observations.size() ||
      history_dimension > observations.size() - history_offset) {
    throw std::out_of_range("Observation history is outside the input buffer");
  }

  const std::size_t newest_frame_offset =
      history_offset + history_dimension - newest_frame.size();
  std::copy(
      newest_frame.begin(),
      newest_frame.end(),
      observations.begin() + newest_frame_offset);
}

inline std::array<double, 4> HeadingAlignedSmplRootDifference(
    const std::array<double, 4>& robot_base_quat,
    const std::array<double, 4>& apply_delta_heading,
    const std::array<double, 4>& smpl_root_quat) {
  const auto aligned_smpl_root =
      quat_mul_d(apply_delta_heading, smpl_root_quat);
  return quat_mul_d(quat_conjugate_d(robot_base_quat), aligned_smpl_root);
}

// Matches torch_transform.get_heading_q used by SONIC v1.1: retain only the
// quaternion's yaw twist (w/z) and normalize. This is intentionally distinct
// from projecting the rotated x-axis, which differs for compound roll/pitch.
inline std::array<double, 4> SonicHeadingQuaternion(
    const std::array<double, 4>& quat) {
  return quat_unit_d({quat[0], 0.0, 0.0, quat[3]});
}

inline std::array<double, 4> HeadingCanonicalizedSmplRootDifference(
    const std::array<double, 4>& robot_base_quat,
    const std::array<double, 4>& apply_delta_heading,
    const std::array<double, 4>& smpl_root_quat) {
  const auto aligned_smpl_root =
      quat_mul_d(apply_delta_heading, smpl_root_quat);
  const auto robot_heading = SonicHeadingQuaternion(robot_base_quat);
  return quat_mul_d(quat_conjugate_d(robot_heading), aligned_smpl_root);
}

}  // namespace h2_lora_observation

#endif  // H2_LORA_OBSERVATION_HPP
