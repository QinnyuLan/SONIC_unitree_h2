#pragma once
#include "command.hpp"
#include "policy_parameters.hpp"
#include <fstream>

namespace sonic_runtime {
// Deployment references are independent of the trained decoder's joint offsets.
inline std::array<double, 31> LoadStandingPose(const std::string& path) {
  std::ifstream file(path);
  Require(file.good(), "cannot read standing pose: " + path);
  const auto config = Json::parse(file);
  Require(config.at("schema") == "sonic.h2.standing_pose.v1", "invalid standing pose schema");
  const auto& leg = config.at("leg_joint_angles_rad");
  Require(leg.is_object() && leg.size() == 3, "standing pose requires hip_pitch, knee and ankle_pitch");
  for (const auto* name : {"hip_pitch", "knee", "ankle_pitch"}) {
    const auto& value = leg.at(name);
    Require(value.is_number() && std::isfinite(value.get<double>()) &&
            std::abs(value.get<double>()) <= 3.141592653589793,
            std::string("invalid standing angle in radians: ") + name);
  }
  std::array<double, 31> pose{};
  for (int j = 0; j < 31; ++j) pose[j] = H2DefaultAngleDds(j);
  for (int side : {0, 6}) {
    pose[side] = leg.at("hip_pitch");
    pose[side + 3] = leg.at("knee");
    pose[side + 5] = leg.at("ankle_pitch");
  }
  return pose;
}

inline std::array<double, 29> StandingBodyReference(const std::array<double, 31>& pose) {
  std::array<double, 29> reference{};
  for (size_t j = 0; j < reference.size(); ++j) reference[j] = pose[h2_native_policy_to_h2_dds[j]];
  return reference;
}
}  // namespace sonic_runtime
