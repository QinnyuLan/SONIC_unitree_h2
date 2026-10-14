#pragma once
#include "command.hpp"
#include "math_utils.hpp"
#include "policy_parameters.hpp"
#include <deque>

namespace sonic_runtime {
using Quat = std::array<double, 4>;
using Vec3 = std::array<double, 3>;

struct RobotState {
  Quat root{1, 0, 0, 0};
  Vec3 angular_velocity{};
  std::array<double, 31> q{}, dq{};  // Unitree H2 DDS order, including head.
};

inline double Sample(const Json& frames, double frame, size_t column) {
  frame = std::clamp(frame, 0.0, 9.0);
  const size_t lo = static_cast<size_t>(frame), hi = std::min(lo + 1, size_t{9});
  return (1 - (frame - lo)) * frames[lo][column].get<double>() +
         (frame - lo) * frames[hi][column].get<double>();
}
inline Quat SampleQuat(const Json& frames, double frame) {
  frame = std::clamp(frame, 0.0, 9.0);
  const size_t lo = static_cast<size_t>(frame), hi = std::min(lo + 1, size_t{9});
  return quat_slerp_d(frames[lo].get<Quat>(), frames[hi].get<Quat>(), frame - lo);
}
template<class Buffer>
inline void PutRotation(Buffer& buffer, size_t offset, const Quat& q) {
  const auto rotation = quat_to_rotation_matrix_d(q);
  for (size_t row = 0; row < 3; ++row)
    for (size_t col = 0; col < 2; ++col) buffer[offset + row * 2 + col] = rotation[row][col];
}

// Fixed release contract: full encoder 1751 -> 64, decoder 994 -> 29.
// Joint references carry either native G1 semantic joints or native H2 body
// policy joints. There is no motion library or trajectory generator here.
class Observations {
 public:
  explicit Observations(std::array<double, 29> standing_reference = h2_native_default_angles_policy)
      : standing_reference_(standing_reference) {}
  void ResetReference() { heading_valid_ = false; }

  std::array<float, 1751> Encoder(const Command* command, const RobotState& state,
                                double age_seconds = 0) {
    const std::string kind = command ? command->kind : "stand";
    const auto identity = command
        ? std::make_tuple(command->source, command->session, kind)
        : std::make_tuple(std::string{}, std::string{}, kind);
    const Json* reference = command ? &command->data : nullptr;
    if (kind == "vr") reference = &command->data.at("lower_body");
    const double spacing = kind == "smpl" ? .02 : .1;
    const double cursor = std::max(0.0, age_seconds) / spacing;
    if (!heading_valid_ || identity != identity_) {
      const Quat root = reference && reference->contains("root_quat")
          ? SampleQuat(reference->at("root_quat"), cursor) : Quat{1, 0, 0, 0};
      alignment_ = quat_mul_d(calc_heading_quat_d(state.root), calc_heading_quat_inv_d(root));
      identity_ = identity;
      heading_valid_ = true;
    }
    const int mode = kind == "g1" ? 3 : kind == "vr" ? 1 : kind == "smpl" ? 2 : 0;
    const bool standing = kind == "stand";
    std::array<float, 1751> out{};
    out[0] = mode;
    if (mode < 3) out[1 + mode] = 1;
    auto relative = [&](const Quat& root) {
      return quat_mul_d(quat_conjugate_d(state.root), quat_mul_d(alignment_, root));
    };
    if (mode == 0 || mode == 3) {
      for (size_t f = 0; f < 10; ++f) {
        for (size_t j = 0; j < 29; ++j) {
          out[4 + f * 29 + j] = standing ? standing_reference_[j]
              : Sample(reference->at("joint_pos"), cursor + f, j);
          out[294 + f * 29 + j] = standing ? 0
              : Sample(reference->at("joint_vel"), cursor + f, j);
        }
        PutRotation(out, 590 + f * 6, relative(standing ? Quat{1, 0, 0, 0}
            : SampleQuat(reference->at("root_quat"), cursor + f)));
      }
    } else if (mode == 1) {
      PutRotation(out, 584, relative(SampleQuat(reference->at("root_quat"), cursor)));
      for (size_t f = 0; f < 10; ++f)
        for (size_t j = 0; j < 12; ++j) {
          const auto index = lower_body_joint_mujoco_order_in_isaaclab_index[j];
          out[650 + f * 12 + j] = Sample(reference->at("joint_pos"), cursor + f, index);
          out[770 + f * 12 + j] = Sample(reference->at("joint_vel"), cursor + f, index);
        }
      for (size_t point = 0; point < 3; ++point) {
        for (size_t j = 0; j < 3; ++j)
          out[890 + point * 3 + j] = command->data["positions"][point][j].get<double>();
        for (size_t j = 0; j < 4; ++j)
          out[899 + point * 4 + j] = command->data["orientations"][point][j].get<double>();
      }
    } else {
      for (size_t f = 0; f < 10; ++f) {
        const double frame = std::clamp(cursor + f, 0.0, 9.0);
        const size_t lo = static_cast<size_t>(frame), hi = std::min(lo + 1, size_t{9});
        const Quat root = SampleQuat(reference->at("root_quat"), frame);
        for (size_t j = 0; j < 24; ++j) {
          Vec3 point{};
          for (size_t k = 0; k < 3; ++k)
            point[k] = (1 - (frame - lo)) * (*reference)["joints"][lo][j][k].get<double>() +
                (frame - lo) * (*reference)["joints"][hi][j][k].get<double>();
          point = quat_rotate_d(quat_conjugate_d(root), point);
          for (size_t k = 0; k < 3; ++k) out[911 + f * 72 + j * 3 + k] = point[k];
        }
        PutRotation(out, 1631 + f * 6, relative(root));
        for (size_t j = 0; j < 6; ++j)
          out[1691 + f * 6 + j] = Sample(reference->at("wrist_joints"), frame, j);
      }
    }
    return out;
  }

  std::array<float, 994> Decoder(const std::array<float, 64>& tokens,
                               const RobotState& state,
                               const std::array<double, 29>& previous_action) {
    History frame;
    frame.angular = state.angular_velocity;
    frame.gravity = quat_rotate_d(quat_conjugate_d(state.root), {0., 0., -1.});
    frame.action = previous_action;
    for (size_t j = 0; j < 29; ++j) {
      const auto dds = h2_native_policy_to_h2_dds[j];
      frame.q[j] = state.q[dds] - h2_native_default_angles_policy[j];
      frame.dq[j] = state.dq[dds];
    }
    if (history_.empty()) history_.assign(10, frame);
    else { history_.pop_front(); history_.push_back(frame); }
    std::array<float, 994> out{};
    std::copy(tokens.begin(), tokens.end(), out.begin());
    for (size_t f = 0; f < 10; ++f) {
      const auto& h = history_[f];
      std::copy(h.angular.begin(), h.angular.end(), out.begin() + 64 + f * 3);
      std::copy(h.q.begin(), h.q.end(), out.begin() + 94 + f * 29);
      std::copy(h.dq.begin(), h.dq.end(), out.begin() + 384 + f * 29);
      std::copy(h.action.begin(), h.action.end(), out.begin() + 674 + f * 29);
      std::copy(h.gravity.begin(), h.gravity.end(), out.begin() + 964 + f * 3);
    }
    return out;
  }
 private:
  std::array<double, 29> standing_reference_;
  struct History { Vec3 angular{}, gravity{}; std::array<double, 29> q{}, dq{}, action{}; };
  std::deque<History> history_;
  bool heading_valid_ = false;
  Quat alignment_{1, 0, 0, 0};
  std::tuple<std::string, std::string, std::string> identity_;
};
}  // namespace sonic_runtime
