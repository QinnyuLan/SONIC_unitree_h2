/**
 * @file error_monitor.hpp
 * @brief Monitors motor error states from LowState and prints warnings on
 *        state transitions, with periodic reminders for persistent faults.
 *
 * Unitree humanoids report per-motor fault codes via the `motorstate()` field
 * in each MotorState_ message, which is part of LowState_ on `rt/lowstate`.
 * A non-zero value indicates a motor fault.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "robot_parameters.hpp"

class ErrorMonitor {
 public:
  explicit ErrorMonitor(int reminder_interval = 500)
      : reminder_interval_(reminder_interval) {
    prev_motorstate_.fill(0);
  }

  template <std::size_t N>
  void update(const std::array<uint32_t, MAX_NUM_MOTOR>& motorstates,
              int motor_count,
              const std::array<const char*, N>& joint_names) {
    const int count = std::min({motor_count, static_cast<int>(N), MAX_NUM_MOTOR});
    for (int i = 0; i < count; ++i) {
      uint32_t state = motorstates[i];
      if (state != prev_motorstate_[i]) {
        if (state != 0 && prev_motorstate_[i] == 0) {
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i, joint_names)
                    << ") FAULT: code 0x" << std::hex << state << std::dec
                    << std::endl;
          error_count_++;
        } else if (state == 0 && prev_motorstate_[i] != 0) {
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i, joint_names)
                    << ") fault CLEARED" << std::endl;
          if (error_count_ > 0) error_count_--;
        } else {
          std::cout << "[ErrorMonitor] Motor " << i << " (" << jointName(i, joint_names)
                    << ") fault CHANGED: 0x" << std::hex << prev_motorstate_[i]
                    << " -> 0x" << state << std::dec << std::endl;
        }
        prev_motorstate_[i] = state;
        reminder_counter_ = 0;
      }
    }

    if (error_count_ > 0) {
      reminder_counter_++;
      if (reminder_counter_ >= reminder_interval_) {
        reminder_counter_ = 0;
        std::cout << "[ErrorMonitor] " << error_count_ << " motor(s) still faulted:";
        for (int i = 0; i < count; ++i) {
          if (prev_motorstate_[i] != 0) {
            std::cout << " " << jointName(i, joint_names) << "(0x" << std::hex
                      << prev_motorstate_[i] << std::dec << ")";
          }
        }
        std::cout << std::endl;
      }
    }
  }

  int getErrorCount() const { return error_count_; }
  bool hasErrors() const { return error_count_ > 0; }

 private:
  template <std::size_t N>
  static const char* jointName(int index, const std::array<const char*, N>& names) {
    if (index >= 0 && index < static_cast<int>(N)) return names[index];
    return "Unknown";
  }

  std::array<uint32_t, MAX_NUM_MOTOR> prev_motorstate_;
  int error_count_ = 0;
  int reminder_interval_ = 2500;
  int reminder_counter_ = 0;
};
