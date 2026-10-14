#ifndef HAND_BACKEND_HPP
#define HAND_BACKEND_HPP

#include <stdexcept>
#include <string>
#include <string_view>

enum class HandBackendKind {
  NONE,
  DEX3,
  INSPIRE,
};

inline HandBackendKind ParseHandBackend(std::string_view value) {
  if (value == "none") return HandBackendKind::NONE;
  if (value == "dex3") return HandBackendKind::DEX3;
  if (value == "inspire") return HandBackendKind::INSPIRE;
  throw std::invalid_argument(
      "invalid hand backend '" + std::string(value) +
      "' (expected dex3, inspire, or none)");
}

inline const char* HandBackendName(HandBackendKind backend) {
  switch (backend) {
    case HandBackendKind::NONE: return "none";
    case HandBackendKind::DEX3: return "dex3";
    case HandBackendKind::INSPIRE: return "inspire";
  }
  return "unknown";
}

#endif  // HAND_BACKEND_HPP
