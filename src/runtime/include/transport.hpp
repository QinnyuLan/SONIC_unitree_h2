#pragma once
#include "command.hpp"
#include <zmq.hpp>
#include <set>

namespace sonic_runtime {
inline int64_t WallMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}

// Each socket is used only by the control thread. Bounded queues and polling
// prevent a stalled or misbehaving producer from blocking motor control.
class Receiver {
 public:
  Receiver(zmq::context_t& context, const std::string& endpoint,
           std::set<std::string> kinds) : socket_(context, zmq::socket_type::pull),
                                        kinds_(std::move(kinds)) {
    socket_.set(zmq::sockopt::linger, 0);
    socket_.set(zmq::sockopt::rcvhwm, 8);
    socket_.set(zmq::sockopt::maxmsgsize, int64_t{131072});
    socket_.bind(endpoint);
  }
  void Poll() {
    for (int i = 0; i < 16; ++i) {
      zmq::message_t message;
      if (!socket_.recv(message, zmq::recv_flags::dontwait)) break;
      try {
        auto command = ParseCommand(Json::parse(message.to_string()));
        Require(kinds_.count(command.kind) != 0, "command on wrong channel");
        inbox.Accept(std::move(command), Clock::now(), WallMilliseconds());
      } catch (const std::exception& e) { ++invalid; last_error = e.what(); }
    }
  }
  CommandInbox inbox;
  uint64_t invalid = 0;
  std::string last_error;
 private:
  zmq::socket_t socket_;
  std::set<std::string> kinds_;
};
}  // namespace sonic_runtime
