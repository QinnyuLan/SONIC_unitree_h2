#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/select.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

#include <string>

class SerialPort
{
public:
  using SharedPtr = std::shared_ptr<SerialPort>;

  SerialPort(const std::string& port, speed_t baudrate, int timeout_ms = 20)
      : port_(port)
  {
    set_timeout(timeout_ms);
    Init(port, baudrate);
  }

  ~SerialPort()
  {
    if (fd_ >= 0)
    {
      ::close(fd_);
    }
  }

  bool is_device_connected() const noexcept
  {
    if (fd_ < 0)
    {
      return false;
    }
    termios option{};
    while (tcgetattr(fd_, &option) != 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      return false;
    }

    struct stat opened{};
    struct stat current{};
    if (::fstat(fd_, &opened) != 0 || ::stat(port_.c_str(), &current) != 0)
    {
      return false;
    }
    if (opened.st_dev != current.st_dev || opened.st_ino != current.st_ino ||
        opened.st_rdev != current.st_rdev)
    {
      return false;
    }
    return true;
  }

  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;

  ssize_t send(const uint8_t* data, size_t len)
  {
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeout_ms_);
    size_t total = 0;
    while (total < len)
    {
      const ssize_t ret = ::write(fd_, data + total, len - total);
      if (ret > 0)
      {
        total += static_cast<size_t>(ret);
        continue;
      }
      if (ret < 0 && errno == EINTR)
      {
        continue;
      }
      if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
      {
        if (!wait_for_fd(false, deadline))
        {
          break;
        }
        continue;
      }
      return -1;
    }
    return static_cast<ssize_t>(total);
  }

  ssize_t recv(uint8_t* data, size_t len)
  {
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeout_ms_);
    size_t total = 0;
    while (total < len)
    {
      if (!wait_for_fd(true, deadline))
      {
        break;
      }
      const ssize_t ret = ::read(fd_, data + total, len - total);
      if (ret > 0)
      {
        total += static_cast<size_t>(ret);
        continue;
      }
      if (ret == 0 || (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)))
      {
        continue;
      }
      if (ret < 0 && errno == EINTR)
      {
        continue;
      }
      return -1;
    }
    return static_cast<ssize_t>(total);
  }

  void set_timeout(int timeout_ms)
  {
    if (timeout_ms <= 0)
    {
      throw std::invalid_argument("serial timeout must be positive");
    }
    timeout_ms_ = timeout_ms;
  }

  void discard_input()
  {
    if (tcflush(fd_, TCIFLUSH) != 0)
    {
      throw std::runtime_error("failed to flush serial input");
    }
  }

private:
  bool wait_for_fd(bool readable, std::chrono::steady_clock::time_point deadline)
  {
    while (true)
    {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline)
      {
        return false;
      }
      const auto remaining =
          std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
      timeval timeout{};
      timeout.tv_sec = static_cast<time_t>(remaining / 1000000);
      timeout.tv_usec = static_cast<suseconds_t>(remaining % 1000000);

      fd_set read_set;
      fd_set write_set;
      FD_ZERO(&read_set);
      FD_ZERO(&write_set);
      if (readable)
      {
        FD_SET(fd_, &read_set);
      }
      else
      {
        FD_SET(fd_, &write_set);
      }

      const int ret = select(
          fd_ + 1,
          readable ? &read_set : nullptr,
          readable ? nullptr : &write_set,
          nullptr,
          &timeout);
      if (ret > 0)
      {
        return true;
      }
      if (ret == 0)
      {
        return false;
      }
      if (errno != EINTR)
      {
        throw std::runtime_error("serial select failed: " + std::string(std::strerror(errno)));
      }
    }
  }

  void Init(const std::string& port, speed_t baudrate)
  {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
    {
      throw std::runtime_error(
          "open serial port " + port + " failed: " + std::string(std::strerror(errno)));
    }
    if (ioctl(fd_, TIOCEXCL) != 0 && errno != ENOTTY && errno != EINVAL)
    {
      const std::string message =
          "claim serial port " + port + " failed: " + std::string(std::strerror(errno));
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }

    termios option{};
    if (tcgetattr(fd_, &option) != 0)
    {
      const std::string message = "tcgetattr failed for " + port;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }
    cfmakeraw(&option);
    if (cfsetispeed(&option, baudrate) != 0 || cfsetospeed(&option, baudrate) != 0)
    {
      const std::string message = "failed to set baud rate for " + port;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }
    option.c_cflag |= CLOCAL | CREAD;
    option.c_cflag &= ~CSIZE;
    option.c_cflag |= CS8;
    option.c_cflag &= ~PARENB;
    option.c_cflag &= ~CSTOPB;
#ifdef CRTSCTS
    option.c_cflag &= ~CRTSCTS;
#endif
    option.c_cc[VTIME] = 0;
    option.c_cc[VMIN] = 0;
    if (tcsetattr(fd_, TCSANOW, &option) != 0 || tcflush(fd_, TCIOFLUSH) != 0)
    {
      const std::string message = "failed to configure serial port " + port;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }
  }

  int fd_{-1};
  int timeout_ms_{20};
  std::string port_;
};

#endif // SERIAL_PORT_H
