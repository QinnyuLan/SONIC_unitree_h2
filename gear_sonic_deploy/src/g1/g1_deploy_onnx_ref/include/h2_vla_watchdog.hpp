#ifndef H2_VLA_WATCHDOG_HPP
#define H2_VLA_WATCHDOG_HPP

#include <bit>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace h2_vla {
// The deploy target uses -ffast-math; std::isfinite can be optimized away.
inline bool Finite(double value) {
  constexpr std::uint64_t mask = UINT64_C(0x7ff0000000000000);
  return (std::bit_cast<std::uint64_t>(value) & mask) != mask;
}

inline bool Enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GEAR_SONIC_H2_VLA");
    return value && std::string(value) == "1";
  }();
  return enabled;
}

inline double MonotonicSeconds() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline bool ValidPacket(const std::vector<double>& token, double deadline, double now) {
  if (token.size() != 64 || !Finite(deadline) ||
      !Finite(now) || deadline <= now || deadline - now > 0.81) return false;
  for (double value : token) {
    if (!Finite(value) || std::abs(value) > 1.25) return false;
  }
  return true;
}

inline bool Expired(double now, double last_accepted, double deadline) {
  return !Finite(now) || !Finite(last_accepted) ||
         !Finite(deadline) || now < last_accepted ||
         now - last_accepted > 0.250 || now >= deadline;
}
}  // namespace h2_vla
#endif
