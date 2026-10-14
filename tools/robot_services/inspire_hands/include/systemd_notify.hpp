#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// systemd's AF_UNIX notification protocol; no libsystemd build dependency.
// The main loop owns this object. A separate heartbeat thread could conceal
// a stuck DDS loop or serial worker, so notifications stay on the main loop.
class SystemdNotify
{
public:
  using Clock = std::chrono::steady_clock;

  SystemdNotify()
  {
    const char* path = std::getenv("NOTIFY_SOCKET");
    if (!path || !*path) return;
    const auto size = std::strlen(path);
    if ((path[0] != '/' && path[0] != '@') || size >= sizeof(address_.sun_path))
      throw std::invalid_argument("invalid NOTIFY_SOCKET");
    address_.sun_family = AF_UNIX;
    std::memcpy(address_.sun_path, path, size + 1);
    address_size_ = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + size + 1);
    if (path[0] == '@')
    {
      address_.sun_path[0] = '\0';
      --address_size_;
    }
    if (const char* usec = std::getenv("WATCHDOG_USEC"))
    {
      const char* pid = std::getenv("WATCHDOG_PID");
      if (!pid || std::stol(pid) == ::getpid())
      {
        const auto timeout = std::stoll(usec);
        if (timeout <= 0) throw std::invalid_argument("invalid WATCHDOG_USEC");
        interval_ = std::chrono::microseconds(std::max<int64_t>(1, timeout / 3));
        watchdog_ = true;
      }
    }
    socket_ = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (socket_ < 0) throw std::runtime_error("cannot create systemd notification socket");
  }

  ~SystemdNotify() { if (socket_ >= 0) ::close(socket_); }
  SystemdNotify(const SystemdNotify&) = delete;
  SystemdNotify& operator=(const SystemdNotify&) = delete;

  void Ready() { Send("READY=1\nSTATUS=DDS ready; serial workers running"); }
  void Stopping() { Send("STOPPING=1"); }

  void Tick(bool workers_healthy)
  {
    if (!watchdog_) return;
    if (!workers_healthy)
    {
      if (!stalled_) Send("STATUS=Serial worker stalled; withholding watchdog heartbeat");
      stalled_ = true;
      return;
    }
    const auto now = Clock::now();
    if (now >= next_heartbeat_)
    {
      if (Send("WATCHDOG=1\nSTATUS=DDS and serial loops progressing"))
        next_heartbeat_ = now + interval_;
      stalled_ = false;
    }
  }

private:
  bool Send(const char* state)
  {
    if (socket_ < 0) return false;
    const auto size = std::strlen(state);
    return ::sendto(socket_, state, size, MSG_NOSIGNAL,
        reinterpret_cast<const sockaddr*>(&address_), address_size_) == static_cast<ssize_t>(size);
  }

  int socket_{-1};
  sockaddr_un address_{};
  socklen_t address_size_{};
  bool watchdog_{false};
  bool stalled_{false};
  std::chrono::microseconds interval_{1000000};
  Clock::time_point next_heartbeat_{};
};
