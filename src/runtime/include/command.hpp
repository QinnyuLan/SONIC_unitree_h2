#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace sonic_runtime {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

inline void Shape(const Json& node, std::initializer_list<size_t> dimensions,
                  const std::string& name) {
  const std::vector<size_t> dims(dimensions);
  const auto visit = [&](const auto& self, const Json& value, size_t depth) -> void {
    if (depth == dims.size()) {
      Require(value.is_number() && !value.is_boolean(), name + ": expected number");
      Require(std::isfinite(value.get<double>()), name + ": non-finite value");
      return;
    }
    Require(value.is_array() && value.size() == dims[depth], name + ": invalid shape");
    for (const auto& child : value) self(self, child, depth + 1);
  };
  visit(visit, node, 0);
}

inline void NormalizeQuaternions(Json& node, size_t count) {
  Shape(node, {count, 4}, "quaternion");
  for (auto& q : node) {
    double squared = 0;
    for (const auto& v : q) squared += v.get<double>() * v.get<double>();
    Require(squared > 1e-12 && std::isfinite(squared), "invalid quaternion norm");
    for (auto& v : q) v = v.get<double>() / std::sqrt(squared);
  }
}

inline void ValidateJointReference(Json& data) {
  Shape(data.at("joint_pos"), {10, 29}, "joint_pos");
  Shape(data.at("joint_vel"), {10, 29}, "joint_vel");
  NormalizeQuaternions(data.at("root_quat"), 10);
}

struct Command {
  std::string kind, source, session;
  uint64_t seq;
  int64_t sent_at_ms;
  int ttl_ms;
  bool takeover;
  Json data;
};

inline Command ParseCommand(const Json& message) {
  Require(message.at("schema") == "sonic.h2.command.v1", "unsupported command schema");
  Command c;
  c.kind = message.at("kind").get<std::string>();
  c.source = message.at("source").get<std::string>();
  c.session = message.at("session").get<std::string>();
  Require(!c.source.empty() && c.source.size() <= 64 && !c.session.empty() &&
          c.session.size() <= 64, "invalid source/session");
  Require(message.at("seq").is_number_unsigned() ||
          (message.at("seq").is_number_integer() && message.at("seq").get<int64_t>() >= 0),
          "seq must be a nonnegative integer");
  c.seq = message.at("seq").get<uint64_t>();
  Require(message.at("sent_at_ms").is_number_integer(), "sent_at_ms must be an integer");
  Require(message.at("sent_at_ms") <= INT64_MAX, "timestamp out of range");
  c.sent_at_ms = message.at("sent_at_ms").get<int64_t>();
  Require(c.sent_at_ms >= 0, "negative timestamp");
  Require(message.at("ttl_ms").is_number_integer(), "ttl_ms must be an integer");
  Require(message.at("ttl_ms") >= 50 && message.at("ttl_ms") <= 1000, "invalid ttl_ms");
  c.ttl_ms = message.at("ttl_ms").get<int>();
  Require(c.ttl_ms >= 50 && c.ttl_ms <= 1000, "ttl_ms outside [50, 1000]");
  c.takeover = message.value("takeover", false);
  c.data = message.at("data");
  Require(c.data.is_object(), "data must be an object");
  if (c.kind == "g1" || c.kind == "h2") {
    ValidateJointReference(c.data);
  } else if (c.kind == "smpl") {
    Shape(c.data.at("joints"), {10, 24, 3}, "joints");
    Shape(c.data.at("wrist_joints"), {10, 6}, "wrist_joints");
    NormalizeQuaternions(c.data.at("root_quat"), 10);
  } else if (c.kind == "vr") {
    Shape(c.data.at("positions"), {3, 3}, "positions");
    NormalizeQuaternions(c.data.at("orientations"), 3);
    ValidateJointReference(c.data.at("lower_body"));
  } else if (c.kind == "planner") {
    Require(c.data.at("mode").is_number_integer(), "mode must be integer");
    Require(c.data.at("mode") >= 0 && c.data.at("mode") <= 26, "invalid locomotion mode");
    Shape(c.data.at("movement"), {3}, "movement");
    Shape(c.data.at("facing"), {3}, "facing");
    Shape(c.data.at("speed"), {}, "speed");
    Shape(c.data.at("height"), {}, "height");
  } else if (c.kind == "head") {
    Shape(c.data.at("pitch"), {}, "pitch");
    Shape(c.data.at("yaw"), {}, "yaw");
  } else if (c.kind == "hands") {
    Require(!c.data.empty(), "empty hands command");
    for (auto it = c.data.begin(); it != c.data.end(); ++it) {
      Require(it.key() == "left" || it.key() == "right", "unknown hand");
      Shape(it.value(), {7}, "hand");
    }
  } else if (c.kind == "token") {
    Shape(c.data.at("tokens"), {64}, "tokens");
  } else {
    Require((c.kind == "stand" || c.kind == "stop") && c.data.empty(), "unknown command kind");
  }
  Require(!message.contains("head") && !message.contains("hands"),
          "head and hands must use their independent channels");
  return c;
}

// A producer holds the input until expiry, release, or explicit takeover. A stopped
// or superseded producer cannot clear the active producer's command on shutdown.
class CommandInbox {
 public:
  bool Accept(Command command, Clock::time_point now, int64_t wall_ms) {
    const int64_t age = wall_ms - command.sent_at_ms;
    if (age < -250 || age >= command.ttl_ms) return Reject("expired command or clock skew");
    for (auto it = sequences_.begin(); it != sequences_.end();) {
      if (now - it->second.second > std::chrono::minutes(1)) it = sequences_.erase(it);
      else ++it;
    }
    const auto key = std::make_pair(command.source, command.session);
    auto previous = sequences_.find(key);
    if (previous != sequences_.end() && command.seq <= previous->second.first)
      return Reject("out-of-order command");
    if (Active(now) && command.kind != "stop" && !command.takeover &&
        (command.source != active_->source || command.session != active_->session))
      return Reject("another producer owns the input");
    if (previous == sequences_.end() && sequences_.size() >= 128)
      return Reject("too many producer sessions");
    sequences_[key] = {command.seq, now};
    if (command.kind == "stop") stopped_ = true;
    if (command.kind == "stand" || command.kind == "stop") {
      active_.reset();
    } else {
      expires_ = now + std::chrono::milliseconds(command.ttl_ms - std::max<int64_t>(0, age));
      received_ = now;
      active_ = std::move(command);
    }
    ++generation_;
    return true;
  }
  const Command* Active(Clock::time_point now) const {
    return active_ && now < expires_ && !stopped_ ? &*active_ : nullptr;
  }
  bool Stopped() const { return stopped_; }
  void DiscardActive() { active_.reset(); }
  uint64_t Generation() const { return generation_; }
  uint64_t Rejected() const { return rejected_; }
  const std::string& LastRejection() const { return rejection_; }
  Clock::time_point Received() const { return received_; }
 private:
  bool Reject(const std::string& reason) { ++rejected_; rejection_ = reason; return false; }
  std::optional<Command> active_;
  Clock::time_point received_{}, expires_{};
  std::map<std::pair<std::string, std::string>, std::pair<uint64_t, Clock::time_point>> sequences_;
  uint64_t generation_ = 0, rejected_ = 0;
  bool stopped_ = false;
  std::string rejection_;
};
}  // namespace sonic_runtime
