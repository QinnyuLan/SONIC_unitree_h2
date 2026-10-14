#pragma once
#include <cerrno>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <termios.h>
#include <unistd.h>

namespace sonic_runtime {
// Keep ISIG enabled so Ctrl+C reaches the runtime's shutdown handler.
class TerminalKeyboard {
 public:
  explicit TerminalKeyboard(int fd = STDIN_FILENO) : fd_(fd) {
    if (!::isatty(fd_)) return;  // A pipe can also drive offline validation.
    if (::tcgetattr(fd_, &saved_) != 0) throw std::runtime_error("cannot read terminal settings");
    auto mode = saved_;
    mode.c_lflag &= ~(ICANON | ECHO);
    mode.c_cc[VMIN] = 0;
    mode.c_cc[VTIME] = 0;
    if (::tcsetattr(fd_, TCSANOW, &mode) != 0) throw std::runtime_error("cannot configure runtime keyboard");
    restore_ = true;
    ::tcflush(fd_, TCIFLUSH);  // Do not arm from keys typed during model loading.
  }
  ~TerminalKeyboard() { if (restore_) ::tcsetattr(fd_, TCSANOW, &saved_); }
  TerminalKeyboard(const TerminalKeyboard&) = delete;
  TerminalKeyboard& operator=(const TerminalKeyboard&) = delete;

  std::string Poll() {
    pollfd descriptor{fd_, POLLIN, 0};
    const int result = ::poll(&descriptor, 1, 0);
    if (result < 0 && errno == EINTR) return {};
    if (result < 0 || (descriptor.revents & (POLLERR | POLLNVAL)))
      throw std::runtime_error("runtime keyboard disconnected");
    if (result == 0) return {};
    char bytes[64];
    const auto count = ::read(fd_, bytes, sizeof(bytes));
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) return {};
    if (count <= 0) throw std::runtime_error("runtime stdin closed; keep the control terminal open");
    std::string keys;
    for (ssize_t i = 0; i < count; ++i) {
      const unsigned char ch = bytes[i];
      if (ch == 27) { escape_ = 1; continue; }
      if (escape_ == 1) { escape_ = (ch == '[' || ch == 'O') ? 2 : 0; continue; }
      if (escape_ == 2) { if (ch >= 0x40 && ch <= 0x7e) escape_ = 0; continue; }
      keys += ch;
    }
    return keys;
  }
 private:
  int fd_, escape_ = 0;
  termios saved_{};
  bool restore_ = false;
};
}  // namespace sonic_runtime
