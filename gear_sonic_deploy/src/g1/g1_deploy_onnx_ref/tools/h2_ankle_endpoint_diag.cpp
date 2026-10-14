#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

constexpr int kModePR = 0;
constexpr int kH2MotorCount = 31;
constexpr int kH2LegWaistMotorCount = 15;
constexpr int kHgMotorCount = 35;
constexpr int kLeftAnkleRoll = 4;
constexpr int kLeftAnklePitch = 5;
constexpr int kRightAnkleRoll = 10;
constexpr int kRightAnklePitch = 11;
constexpr std::array<int, 4> kAnkleSlots{4, 5, 10, 11};
constexpr std::array<const char*, 4> kAnkleNames{
    "left_ankle_roll", "left_ankle_pitch", "right_ankle_roll",
    "right_ankle_pitch"};
constexpr const char* kLowCmdTopic = "rt/lowcmd";
constexpr const char* kLowStateTopic = "rt/lowstate";

// These limits deliberately cannot be relaxed from the command line.
constexpr double kAbsoluteMaxTorque = 8.0;
constexpr double kAbsoluteMaxRampRate = 0.5;
constexpr double kBackgroundDampingKd = 8.0;
constexpr double kSafeStopDampingKd = 8.0;
constexpr double kOtherAnkleVelocityLimit = 0.5;
constexpr double kOtherMotorVelocityLimit = 2.0;
constexpr double kUnexpectedTorqueRiseLimit = 10.0;
constexpr double kRampResumeVelocity = 0.08;
constexpr double kSoftTargetVelocity = 0.25;
constexpr double kTorqueBackoffRate = 4.0;

// Unitree ROS H2 ankle-pitch limits. The guard adds only a small diagnostic
// margin; reaching it is an abort, not an endpoint declaration.
constexpr double kExpectedPitchLower = -1.13446401358;
constexpr double kExpectedPitchUpper = 0.61086523808;

std::atomic<bool> g_stop_requested{false};

void SignalHandler(int) { g_stop_requested.store(true); }

uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t crc32 = 0xFFFFFFFF;
  constexpr uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1U << 31;
    const uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (crc32 & 0x80000000U) {
        crc32 = (crc32 << 1U) ^ polynomial;
      } else {
        crc32 <<= 1U;
      }
      if (data & xbit) crc32 ^= polynomial;
      xbit >>= 1U;
    }
  }
  return crc32;
}

std::string TimestampForPath() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&tt, &tm);
  std::ostringstream os;
  os << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return os.str();
}

struct Args {
  std::string network_interface;
  int domain_id = 0;
  int slot = -1;
  int direction = 0;
  bool suspended_ack = false;
  bool preflight_only = false;
  bool skip_release_mode = false;
  bool require_max_torque = false;
  double lowstate_timeout_s = 5.0;
  double rate_hz = 250.0;
  double max_torque = 2.0;
  double ramp_rate = 0.5;
  double ramp_down_rate = 4.0;
  double target_damping_kd = 1.0;
  double warmup_s = 1.0;
  double max_torque_hold_s = 0.6;
  double state_timeout_s = 0.05;
  double roll_excursion_limit = 0.10;
  double temperature_limit = 70.0;
  double temperature_rise_limit = 10.0;
  double pitch_guard_lower = -1.20;
  double pitch_guard_upper = 0.68;
  double minimum_travel = 0.04;
  double plateau_window_s = 0.35;
  double plateau_q_range = 0.003;
  double plateau_dq = 0.04;
  std::string log_path;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " <network_interface> --slot 5|11 --direction positive|negative "
         "--i-understand-suspended [options]\n\n"
      << "Moves exactly one H2 ankle-pitch joint in one direction using only "
         "damping plus feedforward torque.\n"
      << "It never writes calibration data. Run one direction at a time and "
         "inspect the result before continuing.\n\n"
      << "Required in command mode:\n"
      << "  --slot 5|11                  H2 mapped left/right ankle-pitch slot.\n"
      << "  --direction positive|negative\n"
      << "  --i-understand-suspended     Robot is securely suspended and clear.\n\n"
      << "Options:\n"
      << "  --preflight-only             Read state and exit; publish no LowCmd.\n"
      << "  --domain-id N                DDS domain, default 0.\n"
      << "  --max-torque NM              Default 2.0, hard maximum 8.0 N.m.\n"
      << "  --ramp-rate NM_S             Default and hard maximum 0.5 N.m/s.\n"
      << "  --damping-kd D               Target pitch damping, default 1.0, max 4.\n"
      << "  --roll-limit RAD             Paired ankle-roll excursion, default 0.10.\n"
      << "  --require-max-torque         Confirm a plateau only at max torque.\n"
      << "  --skip-release-mode          Skip MotionSwitcher check/release.\n"
      << "  --log PATH                   CSV path under /tmp by default.\n"
      << "  -h, --help                   Show this help.\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* option) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(option) + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--slot") {
      args.slot = std::stoi(value("--slot"));
    } else if (arg == "--direction") {
      const std::string direction = value("--direction");
      if (direction == "positive" || direction == "+") {
        args.direction = 1;
      } else if (direction == "negative" || direction == "-") {
        args.direction = -1;
      } else {
        throw std::runtime_error(
            "--direction must be positive or negative");
      }
    } else if (arg == "--i-understand-suspended") {
      args.suspended_ack = true;
    } else if (arg == "--preflight-only") {
      args.preflight_only = true;
    } else if (arg == "--require-max-torque") {
      args.require_max_torque = true;
    } else if (arg == "--skip-release-mode") {
      args.skip_release_mode = true;
    } else if (arg == "--domain-id") {
      args.domain_id = std::stoi(value("--domain-id"));
    } else if (arg == "--max-torque") {
      args.max_torque = std::stod(value("--max-torque"));
    } else if (arg == "--ramp-rate") {
      args.ramp_rate = std::stod(value("--ramp-rate"));
    } else if (arg == "--damping-kd") {
      args.target_damping_kd = std::stod(value("--damping-kd"));
    } else if (arg == "--roll-limit") {
      args.roll_excursion_limit = std::stod(value("--roll-limit"));
    } else if (arg == "--log") {
      args.log_path = value("--log");
    } else if (!arg.empty() && arg.front() == '-') {
      throw std::runtime_error("unknown option: " + arg);
    } else if (args.network_interface.empty()) {
      args.network_interface = arg;
    } else {
      throw std::runtime_error("unexpected positional argument: " + arg);
    }
  }

  if (args.network_interface.empty()) {
    throw std::runtime_error("network_interface is required");
  }
  if (!args.preflight_only && args.slot != kLeftAnklePitch &&
      args.slot != kRightAnklePitch) {
    throw std::runtime_error("--slot must be 5 or 11");
  }
  if (!args.preflight_only && args.direction == 0) {
    throw std::runtime_error("--direction is required");
  }
  if (!args.preflight_only && !args.suspended_ack) {
    throw std::runtime_error(
        "command mode requires --i-understand-suspended");
  }
  if (!(args.max_torque > 0.0) ||
      args.max_torque > kAbsoluteMaxTorque) {
    throw std::runtime_error("--max-torque must be in (0, 8.0]");
  }
  if (!(args.ramp_rate > 0.0) ||
      args.ramp_rate > kAbsoluteMaxRampRate) {
    throw std::runtime_error("--ramp-rate must be in (0, 0.5]");
  }
  if (args.target_damping_kd < 0.0 || args.target_damping_kd > 4.0) {
    throw std::runtime_error("--damping-kd must be in [0, 4]");
  }
  if (!(args.roll_excursion_limit > 0.0) ||
      args.roll_excursion_limit > 0.15) {
    throw std::runtime_error("--roll-limit must be in (0, 0.15]");
  }
  if (args.log_path.empty()) {
    args.log_path = "/tmp/h2_ankle_endpoint_" + TimestampForPath() + ".csv";
  }
  return args;
}

struct TimedState {
  LowState_ state;
  std::chrono::steady_clock::time_point received_at;
  uint64_t sequence = 0;
};

class LowStateBuffer {
 public:
  void Set(const LowState_& state) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sample_ = TimedState{state, std::chrono::steady_clock::now(), ++sequence_};
    }
    cv_.notify_all();
  }

  std::optional<TimedState> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sample_;
  }

  std::optional<TimedState> Wait(double timeout_s) const {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, std::chrono::duration<double>(timeout_s),
                 [&] { return sample_.has_value(); });
    return sample_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::optional<TimedState> sample_;
  uint64_t sequence_ = 0;
};

int PairedRollSlot(int pitch_slot) {
  return pitch_slot == kLeftAnklePitch ? kLeftAnkleRoll : kRightAnkleRoll;
}

const char* SlotName(int slot) {
  return slot == kLeftAnklePitch ? "left_ankle_pitch" : "right_ankle_pitch";
}

double MaxTemperature(const LowState_& state, int slot) {
  const auto& temperature = state.motor_state().at(slot).temperature();
  return std::max<double>(temperature[0], temperature[1]);
}

void PrintAnkleState(const LowState_& state) {
  std::cout << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < kAnkleSlots.size(); ++i) {
    const int slot = kAnkleSlots[i];
    const auto& motor = state.motor_state().at(slot);
    std::cout << "  slot " << std::setw(2) << slot << " "
              << std::setw(19) << std::left << kAnkleNames[i] << std::right
              << " q=" << std::showpos << motor.q() << " dq=" << motor.dq()
              << " tau=" << motor.tau_est() << std::noshowpos
              << " temp=" << motor.temperature()[0] << "/"
              << motor.temperature()[1] << " sensor=" << motor.sensor()[0]
              << "/" << motor.sensor()[1]
              << " motorstate=" << motor.motorstate() << "\n";
  }
}

LowCmd_ BuildDampingCommand(int mode_machine, double target_damping_kd,
                            int torque_slot, double torque) {
  LowCmd_ command;
  command.mode_pr() = static_cast<uint8_t>(kModePR);
  command.mode_machine() = static_cast<uint8_t>(mode_machine);
  for (int i = 0; i < kHgMotorCount; ++i) {
    auto& motor = command.motor_cmd().at(i);
    if (i < kH2LegWaistMotorCount) {
      motor.mode() = 1;
      motor.q() = 0.0F;
      motor.dq() = 0.0F;
      motor.tau() = i == torque_slot ? static_cast<float>(torque) : 0.0F;
      motor.kp() = 0.0F;
      motor.kd() = static_cast<float>(i == torque_slot
                                          ? target_damping_kd
                                          : kBackgroundDampingKd);
    } else {
      motor.mode() = 0;
      motor.q() = 0.0F;
      motor.dq() = 0.0F;
      motor.tau() = 0.0F;
      motor.kp() = 0.0F;
      motor.kd() = 0.0F;
    }
  }
  command.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&command),
                            (sizeof(command) >> 2U) - 1U);
  return command;
}

std::string CsvEscape(const std::string& text) {
  std::string escaped = "\"";
  for (const char c : text) {
    if (c == '"') escaped += '"';
    escaped += c;
  }
  escaped += '"';
  return escaped;
}

class CsvLogger {
 public:
  explicit CsvLogger(const std::string& path)
      : start_(std::chrono::steady_clock::now()) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    output_.open(path);
    if (!output_) throw std::runtime_error("failed to open CSV: " + path);
    output_ << "elapsed_s,phase,command_tau_nm,state_age_s,mode_pr,mode_machine";
    for (const char* name : kAnkleNames) {
      output_ << "," << name << "_mode"
              << "," << name << "_q"
              << "," << name << "_dq"
              << "," << name << "_tau_est"
              << "," << name << "_temp0"
              << "," << name << "_temp1"
              << "," << name << "_sensor0"
              << "," << name << "_sensor1"
              << "," << name << "_motorstate";
    }
    output_ << ",note\n" << std::setprecision(9);
  }

  void Write(const std::string& phase, double command_tau,
             const TimedState& sample, const std::string& note = {}) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - start_).count();
    const double state_age =
        std::chrono::duration<double>(now - sample.received_at).count();
    output_ << elapsed << "," << phase << "," << command_tau << ","
            << state_age << "," << int(sample.state.mode_pr()) << ","
            << int(sample.state.mode_machine());
    for (const int slot : kAnkleSlots) {
      const auto& motor = sample.state.motor_state().at(slot);
      output_ << "," << int(motor.mode()) << "," << motor.q() << ","
              << motor.dq() << "," << motor.tau_est() << ","
              << motor.temperature()[0] << "," << motor.temperature()[1]
              << "," << motor.sensor()[0] << "," << motor.sensor()[1]
              << "," << motor.motorstate();
    }
    output_ << "," << CsvEscape(note) << "\n";
    if (++rows_since_flush_ >= 50) {
      output_.flush();
      rows_since_flush_ = 0;
    }
  }

  void Flush() { output_.flush(); }

 private:
  std::ofstream output_;
  std::chrono::steady_clock::time_point start_;
  int rows_since_flush_ = 0;
};

bool CheckAndReleaseMotionMode(bool skip_release, std::string* error) {
  if (skip_release) {
    std::cout << "[WARN] MotionSwitcher check/release explicitly skipped.\n";
    return true;
  }

  auto client = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
  client->SetTimeout(5.0F);
  client->Init();
  for (int attempt = 0; attempt < 4; ++attempt) {
    std::string form;
    std::string name;
    const int32_t check_code = client->CheckMode(form, name);
    if (check_code != 0) {
      *error = "MotionSwitcher CheckMode failed with code " +
               std::to_string(check_code);
      return false;
    }
    if (name.empty()) {
      std::cout << "[INFO] Motion-control service is released.\n";
      return true;
    }
    std::cout << "[INFO] Releasing active service form='" << form
              << "' name='" << name << "'.\n";
    const int32_t release_code = client->ReleaseMode();
    if (release_code != 0) {
      *error = "MotionSwitcher ReleaseMode failed with code " +
               std::to_string(release_code);
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  *error = "MotionSwitcher still reports an active service after retries";
  return false;
}

std::optional<std::string> ValidateState(
    const TimedState& sample, const Args& args, int mode_machine,
    const std::array<double, kH2MotorCount>& initial_temperature,
    const std::array<double, kH2MotorCount>& initial_tau_est,
    const std::optional<double>& roll_baseline) {
  const auto now = std::chrono::steady_clock::now();
  const double age =
      std::chrono::duration<double>(now - sample.received_at).count();
  if (age > args.state_timeout_s) {
    return "lowstate watchdog expired: age=" + std::to_string(age) + " s";
  }
  if (sample.state.mode_pr() != kModePR) {
    return "mode_pr changed to " + std::to_string(sample.state.mode_pr());
  }
  if (sample.state.mode_machine() != mode_machine) {
    return "mode_machine changed from " + std::to_string(mode_machine) +
           " to " + std::to_string(sample.state.mode_machine());
  }
  for (int i = 0; i < kH2MotorCount; ++i) {
    const auto& motor = sample.state.motor_state().at(i);
    if (motor.motorstate() != 0) {
      return "motor fault at slot " + std::to_string(i) + ": " +
             std::to_string(motor.motorstate());
    }
    if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
        !std::isfinite(motor.tau_est())) {
      return "non-finite motor state at slot " + std::to_string(i);
    }
    const bool ankle_slot =
        std::find(kAnkleSlots.begin(), kAnkleSlots.end(), i) !=
        kAnkleSlots.end();
    const double velocity_limit =
        ankle_slot ? kOtherAnkleVelocityLimit : kOtherMotorVelocityLimit;
    if (i != args.slot && std::abs(motor.dq()) > velocity_limit) {
      return "unexpected velocity at non-target slot " + std::to_string(i) +
             ": dq=" + std::to_string(motor.dq());
    }
    if (i != args.slot &&
        std::abs(static_cast<double>(motor.tau_est()) - initial_tau_est[i]) >
            kUnexpectedTorqueRiseLimit) {
      return "unexpected torque rise at non-target slot " +
             std::to_string(i) + ": tau_est=" +
             std::to_string(motor.tau_est());
    }
    const double temperature = MaxTemperature(sample.state, i);
    if (temperature > args.temperature_limit) {
      return "temperature limit at slot " + std::to_string(i) + ": " +
             std::to_string(temperature) + " C";
    }
    if (temperature - initial_temperature[i] >
        args.temperature_rise_limit) {
      return "temperature rise limit at slot " + std::to_string(i);
    }
  }

  const auto& pitch = sample.state.motor_state().at(args.slot);
  if (!std::isfinite(pitch.q()) || !std::isfinite(pitch.dq())) {
    return "non-finite ankle-pitch state";
  }
  if (pitch.q() < args.pitch_guard_lower ||
      pitch.q() > args.pitch_guard_upper) {
    return "ankle-pitch q guard: q=" + std::to_string(pitch.q());
  }
  if (roll_baseline.has_value()) {
    const double roll_q =
        sample.state.motor_state().at(PairedRollSlot(args.slot)).q();
    if (std::abs(roll_q - *roll_baseline) > args.roll_excursion_limit) {
      return "paired ankle-roll excursion: delta=" +
             std::to_string(roll_q - *roll_baseline);
    }
  }
  return std::nullopt;
}

struct PlateauPoint {
  std::chrono::steady_clock::time_point time;
  double q = 0.0;
  double dq = 0.0;
};

bool IsPlateau(const std::deque<PlateauPoint>& history, const Args& args) {
  if (history.size() < 20) return false;
  const double duration =
      std::chrono::duration<double>(history.back().time - history.front().time)
          .count();
  if (duration < args.plateau_window_s * 0.90) return false;

  double q_min = std::numeric_limits<double>::infinity();
  double q_max = -std::numeric_limits<double>::infinity();
  size_t slow_count = 0;
  for (const auto& point : history) {
    q_min = std::min(q_min, point.q);
    q_max = std::max(q_max, point.q);
    if (std::abs(point.dq) <= args.plateau_dq) ++slow_count;
  }
  return q_max - q_min <= args.plateau_q_range &&
         static_cast<double>(slow_count) / history.size() >= 0.90;
}

void PublishSafeStop(ChannelPublisher<LowCmd_>& publisher,
                     const LowStateBuffer& state_buffer, CsvLogger& logger,
                     const Args& args, int mode_machine, double starting_tau,
                     bool emergency) {
  const double period_s = 1.0 / args.rate_hz;
  const auto period = std::chrono::duration<double>(period_s);
  const double down_rate = emergency ? 20.0 : args.ramp_down_rate;
  double tau = starting_tau;
  auto next = std::chrono::steady_clock::now();
  while (std::abs(tau) > 1e-6) {
    const double step = down_rate * period_s;
    if (std::abs(tau) <= step) {
      tau = 0.0;
    } else {
      tau -= std::copysign(step, tau);
    }
    publisher.Write(BuildDampingCommand(mode_machine, kSafeStopDampingKd,
                                        args.slot, tau));
    if (const auto sample = state_buffer.Get()) {
      logger.Write("ramp_down", tau, *sample,
                   emergency ? "emergency" : "normal");
    }
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        period);
    std::this_thread::sleep_until(next);
  }

  const auto zero_end = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(1.0);
  while (std::chrono::steady_clock::now() < zero_end) {
    publisher.Write(BuildDampingCommand(mode_machine, kSafeStopDampingKd,
                                        args.slot, 0.0));
    if (const auto sample = state_buffer.Get()) {
      logger.Write("zero_damping", 0.0, *sample);
    }
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        period);
    std::this_thread::sleep_until(next);
  }
  logger.Flush();
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = ParseArgs(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    PrintUsage(argv[0]);
    return 2;
  }

  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);

  std::cout << "\n=== H2 Ankle-Pitch Endpoint Diagnostic ===\n"
            << "DDS: domain=" << args.domain_id
            << " interface=" << args.network_interface << "\n"
            << "This tool does not write or modify calibration data.\n";
  if (!args.preflight_only) {
    std::cout << "Target: slot " << args.slot << " (" << SlotName(args.slot)
              << "), direction="
              << (args.direction > 0 ? "positive" : "negative") << "\n"
              << "Command: Kp=0, target Kd=" << args.target_damping_kd
              << ", background Kd=" << kBackgroundDampingKd
              << ", max |tau_ff|=" << args.max_torque
              << " N.m, ramp=" << args.ramp_rate << " N.m/s\n"
              << "Guards: paired roll delta<=" << args.roll_excursion_limit
              << " rad, pitch q in [" << args.pitch_guard_lower << ", "
              << args.pitch_guard_upper << "] rad\n"
              << "Governor: ramp only below " << kRampResumeVelocity
              << " rad/s; back off above " << kSoftTargetVelocity
              << " rad/s\n";
  }

  try {
    ChannelFactory::Instance()->Init(args.domain_id, args.network_interface);
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] DDS init failed: " << error.what() << "\n";
    return 3;
  }

  LowStateBuffer state_buffer;
  std::atomic<uint64_t> crc_errors{0};
  ChannelSubscriber<LowState_> subscriber(kLowStateTopic);
  subscriber.InitChannel(
      [&](const void* message) {
        LowState_ state = *static_cast<const LowState_*>(message);
        const uint32_t crc = Crc32Core(reinterpret_cast<uint32_t*>(&state),
                                       (sizeof(state) >> 2U) - 1U);
        if (state.crc() != crc) {
          ++crc_errors;
          return;
        }
        state_buffer.Set(state);
      },
      10);

  std::cout << "[INFO] Waiting for valid rt/lowstate...\n";
  const auto first = state_buffer.Wait(args.lowstate_timeout_s);
  if (!first) {
    std::cerr << "[ERROR] No valid lowstate within " << args.lowstate_timeout_s
              << " s (CRC errors=" << crc_errors.load() << ").\n";
    return 3;
  }
  std::cout << "[INFO] mode_pr=" << int(first->state.mode_pr())
            << " mode_machine=" << int(first->state.mode_machine())
            << "\n";
  PrintAnkleState(first->state);

  bool preflight_ok = first->state.mode_pr() == kModePR;
  for (int i = 0; i < kH2MotorCount; ++i) {
    if (first->state.motor_state().at(i).motorstate() != 0) {
      std::cerr << "[ERROR] Motor fault at slot " << i << ": "
                << first->state.motor_state().at(i).motorstate() << "\n";
      preflight_ok = false;
    }
  }
  if (!preflight_ok) {
    std::cerr << "[ERROR] Preflight failed; no command was sent.\n";
    return 4;
  }
  if (args.preflight_only) {
    std::cout << "[OK] Read-only preflight passed; no command was sent.\n";
    return 0;
  }

  std::cout << "\n[ARM] This will take over leg/waist slots 0..14 with Kp=0, "
            << "background Kd=" << kBackgroundDampingKd
            << ", target pitch Kd=" << args.target_damping_kd
            << ", apply torque only to slot " << args.slot
            << ", and leave upper-body slots disabled.\n"
            << "      Confirm the robot is suspended, nobody is in its motion "
               "envelope, and an operator can cut power.\n"
            << "      Type RUN to continue: " << std::flush;
  std::string confirmation;
  std::getline(std::cin, confirmation);
  if (confirmation != "RUN") {
    std::cout << "[INFO] Cancelled; no command was sent.\n";
    return 0;
  }

  std::string release_error;
  if (!CheckAndReleaseMotionMode(args.skip_release_mode, &release_error)) {
    std::cerr << "[ERROR] " << release_error << ". No LowCmd was sent.\n"
              << "        If the robot is deliberately in low-level debug "
                 "damping mode, re-run with --skip-release-mode.\n";
    return 4;
  }

  const auto before_control = state_buffer.Get();
  if (!before_control) {
    std::cerr << "[ERROR] Lowstate disappeared before control.\n";
    return 4;
  }
  const int mode_machine = before_control->state.mode_machine();
  std::array<double, kH2MotorCount> initial_temperature{};
  std::array<double, kH2MotorCount> initial_tau_est{};
  for (int i = 0; i < kH2MotorCount; ++i) {
    initial_temperature[i] = MaxTemperature(before_control->state, i);
    initial_tau_est[i] = before_control->state.motor_state().at(i).tau_est();
  }

  CsvLogger logger(args.log_path);
  ChannelPublisher<LowCmd_> publisher(kLowCmdTopic);
  publisher.InitChannel();
  std::cout << "[INFO] CSV: " << args.log_path << "\n"
            << "[INFO] Taking over with zero-torque damping for "
            << args.warmup_s << " s...\n";

  const auto period = std::chrono::duration<double>(1.0 / args.rate_hz);
  auto next = std::chrono::steady_clock::now();
  const auto warmup_end = next + std::chrono::duration<double>(args.warmup_s);
  std::string outcome;
  bool emergency = false;
  double command_tau = 0.0;
  while (std::chrono::steady_clock::now() < warmup_end) {
    publisher.Write(BuildDampingCommand(mode_machine, args.target_damping_kd,
                                        args.slot, 0.0));
    const auto sample = state_buffer.Get();
    if (!sample) {
      outcome = "lowstate unavailable during warmup";
      emergency = true;
      break;
    }
    logger.Write("warmup", 0.0, *sample);
    if (const auto error =
            ValidateState(*sample, args, mode_machine, initial_temperature,
                          initial_tau_est, std::nullopt)) {
      outcome = *error;
      emergency = true;
      break;
    }
    if (g_stop_requested.load()) {
      outcome = "operator interrupt during warmup";
      emergency = true;
      break;
    }
    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        period);
    std::this_thread::sleep_until(next);
  }

  std::optional<TimedState> baseline = state_buffer.Get();
  double start_q = 0.0;
  double roll_baseline = 0.0;
  if (baseline) {
    start_q = baseline->state.motor_state().at(args.slot).q();
    roll_baseline =
        baseline->state.motor_state().at(PairedRollSlot(args.slot)).q();
  } else if (outcome.empty()) {
    outcome = "lowstate unavailable after warmup";
    emergency = true;
  }

  double q_min = start_q;
  double q_max = start_q;
  double max_abs_dq = 0.0;
  std::deque<PlateauPoint> history;
  bool plateau_detected = false;

  if (outcome.empty()) {
    std::cout << "[INFO] Baseline q=" << std::showpos << std::fixed
              << std::setprecision(6) << start_q << std::noshowpos
              << " rad. Starting torque ramp now.\n";
    const auto ramp_start = std::chrono::steady_clock::now();
    auto control_previous = ramp_start;
    std::optional<std::chrono::steady_clock::time_point> max_torque_since;
    double magnitude = 0.0;
    next = ramp_start;
    while (true) {
      const auto now = std::chrono::steady_clock::now();
      const double control_dt = std::clamp(
          std::chrono::duration<double>(now - control_previous).count(), 0.0,
          0.02);
      control_previous = now;
      if (const auto control_state = state_buffer.Get()) {
        const double dq = control_state->state.motor_state().at(args.slot).dq();
        const double directed_velocity = args.direction * dq;
        if (directed_velocity > kSoftTargetVelocity) {
          magnitude =
              std::max(0.0, magnitude - kTorqueBackoffRate * control_dt);
        } else if (std::abs(dq) < kRampResumeVelocity) {
          magnitude = std::min(args.max_torque,
                               magnitude + args.ramp_rate * control_dt);
        }
      }
      command_tau = args.direction * magnitude;
      publisher.Write(BuildDampingCommand(mode_machine, args.target_damping_kd,
                                          args.slot, command_tau));

      const auto sample = state_buffer.Get();
      if (!sample) {
        outcome = "lowstate unavailable during torque ramp";
        emergency = true;
        break;
      }
      logger.Write("torque_ramp", command_tau, *sample);
      if (const auto error = ValidateState(
              *sample, args, mode_machine, initial_temperature, initial_tau_est,
              std::optional<double>(roll_baseline))) {
        outcome = *error;
        emergency = true;
        break;
      }
      if (g_stop_requested.load()) {
        outcome = "operator interrupt";
        emergency = true;
        break;
      }

      const auto& pitch = sample->state.motor_state().at(args.slot);
      const double q = pitch.q();
      q_min = std::min(q_min, q);
      q_max = std::max(q_max, q);
      max_abs_dq =
          std::max(max_abs_dq, std::abs(static_cast<double>(pitch.dq())));
      const double travel = q - start_q;
      if (magnitude >= 0.5 && args.direction * travel < -0.02) {
        outcome = "ankle moved opposite commanded direction: delta_q=" +
                  std::to_string(travel);
        emergency = true;
        break;
      }

      history.push_back({now, q, pitch.dq()});
      while (!history.empty() &&
             std::chrono::duration<double>(now - history.front().time).count() >
                 args.plateau_window_s) {
        history.pop_front();
      }

      const bool moved_enough = std::abs(travel) >= args.minimum_travel;
      const bool already_toward_endpoint = args.direction * q >= 0.25;
      const bool near_expected_endpoint =
          args.direction > 0
              ? q >= kExpectedPitchUpper - 0.12
              : q <= kExpectedPitchLower + 0.12;
      const bool at_max_torque =
          magnitude >= args.max_torque - 1e-6;
      const bool endpoint_context = args.require_max_torque
                                        ? at_max_torque &&
                                              (moved_enough ||
                                               already_toward_endpoint)
                                        : (moved_enough &&
                                           (near_expected_endpoint ||
                                            at_max_torque)) ||
                                              (already_toward_endpoint &&
                                               at_max_torque);
      if (endpoint_context && IsPlateau(history, args)) {
        plateau_detected = true;
        outcome = "stationary endpoint/obstruction plateau detected";
        break;
      }

      if (at_max_torque) {
        if (!max_torque_since) max_torque_since = now;
        if (std::chrono::duration<double>(now - *max_torque_since).count() >=
            args.max_torque_hold_s) {
          outcome = "maximum torque completed without a stable plateau";
          break;
        }
      } else {
        max_torque_since.reset();
      }

      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          period);
      std::this_thread::sleep_until(next);
    }
  }

  const auto endpoint_sample = state_buffer.Get();
  if (endpoint_sample) {
    logger.Write("outcome", command_tau, *endpoint_sample, outcome);
  }
  std::cout << "[INFO] Stopping torque: " << outcome << "\n";
  PublishSafeStop(publisher, state_buffer, logger, args, mode_machine,
                  command_tau, emergency);

  const auto final_sample = state_buffer.Get();
  if (final_sample) {
    const auto& pitch = final_sample->state.motor_state().at(args.slot);
    std::cout << "\n=== Result ===\n"
              << "  status: " << outcome << "\n"
              << "  slot: " << args.slot << " (" << SlotName(args.slot)
              << ")\n"
              << "  start q: " << std::showpos << std::fixed
              << std::setprecision(6) << start_q << " rad\n"
              << "  observed q range: [" << q_min << ", " << q_max
              << "] rad\n"
              << "  final q after zero damping: " << pitch.q() << " rad\n"
              << "  maximum |dq|: " << std::noshowpos << max_abs_dq
              << " rad/s\n"
              << "  endpoint sample sensor: " << pitch.sensor()[0] << "/"
              << pitch.sensor()[1] << "\n"
              << "  plateau detected: " << (plateau_detected ? "yes" : "no")
              << "\n"
              << "  CSV: " << args.log_path << "\n";
  }
  PrintAnkleState(final_sample ? final_sample->state : before_control->state);

  return emergency ? 5 : 0;
}
