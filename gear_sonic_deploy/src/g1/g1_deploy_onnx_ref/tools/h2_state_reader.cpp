#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/MainBoardState_.hpp>
#include <unitree/idl/hg/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using HgBmsState = unitree_hg::msg::dds_::BmsState_;
using HgImuState = unitree_hg::msg::dds_::IMUState_;
using HgLowState = unitree_hg::msg::dds_::LowState_;
using HgMainBoardState = unitree_hg::msg::dds_::MainBoardState_;
using HgSportModeState = unitree_hg::msg::dds_::SportModeState_;
using GoSportModeState = unitree_go::msg::dds_::SportModeState_;
using GoWirelessController = unitree_go::msg::dds_::WirelessController_;

constexpr int kH2MotorCount = 31;
constexpr int kHgMotorSlots = 35;

struct Args {
  std::string network_interface;
  int domain_id = 0;
  double wait_seconds = 5.0;
  int motor_count = kH2MotorCount;
  std::string ankle_capture_csv;
};

constexpr std::array<int, 4> kAnkleSlots{4, 5, 10, 11};
constexpr std::array<const char*, 4> kAnkleNames{
    "left_ankle_roll", "left_ankle_pitch", "right_ankle_roll",
    "right_ankle_pitch"};

struct AnkleMotorSample {
  uint8_t mode = 0;
  float q = 0.0F;
  float dq = 0.0F;
  float tau_est = 0.0F;
  std::array<int16_t, 2> temperature{};
  std::array<uint32_t, 2> sensor{};
  uint32_t motorstate = 0;
};

struct AnkleSample {
  double elapsed_seconds = 0.0;
  uint8_t mode_pr = 0;
  uint8_t mode_machine = 0;
  std::array<AnkleMotorSample, kAnkleSlots.size()> motors{};
};

class AnkleCapture {
 public:
  void Add(const HgLowState& state) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!start_time_) start_time_ = now;

    AnkleSample sample;
    sample.elapsed_seconds =
        std::chrono::duration<double>(now - *start_time_).count();
    sample.mode_pr = state.mode_pr();
    sample.mode_machine = state.mode_machine();
    for (size_t i = 0; i < kAnkleSlots.size(); ++i) {
      const auto& motor = state.motor_state().at(kAnkleSlots.at(i));
      auto& output = sample.motors.at(i);
      output.mode = motor.mode();
      output.q = motor.q();
      output.dq = motor.dq();
      output.tau_est = motor.tau_est();
      output.temperature = motor.temperature();
      output.sensor = motor.sensor();
      output.motorstate = motor.motorstate();
    }
    samples_.push_back(sample);
  }

  std::vector<AnkleSample> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<std::chrono::steady_clock::time_point> start_time_;
  std::vector<AnkleSample> samples_;
};

void WriteAnkleCapture(const std::string& path,
                       const std::vector<AnkleSample>& samples) {
  const std::filesystem::path output_path(path);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) throw std::runtime_error("failed to open capture CSV: " + path);

  output << "elapsed_s,mode_pr,mode_machine";
  for (const char* name : kAnkleNames) {
    output << "," << name << "_mode"
           << "," << name << "_q"
           << "," << name << "_dq"
           << "," << name << "_tau_est"
           << "," << name << "_temp0"
           << "," << name << "_temp1"
           << "," << name << "_sensor0"
           << "," << name << "_sensor1"
           << "," << name << "_motorstate";
  }
  output << "\n" << std::setprecision(9);

  for (const auto& sample : samples) {
    output << sample.elapsed_seconds << "," << int(sample.mode_pr) << ","
           << int(sample.mode_machine);
    for (const auto& motor : sample.motors) {
      output << "," << int(motor.mode) << "," << motor.q << "," << motor.dq
             << "," << motor.tau_est << "," << motor.temperature[0] << ","
             << motor.temperature[1] << "," << motor.sensor[0] << ","
             << motor.sensor[1] << "," << motor.motorstate;
    }
    output << "\n";
  }
}

void PrintAnkleExtrema(const std::vector<AnkleSample>& samples) {
  if (samples.empty()) return;
  std::cout << "\n=== read-only ankle extrema ===\n"
            << "  Raw extrema include motion samples; endpoint plateaus must be "
               "checked in the CSV.\n";
  for (size_t i = 0; i < kAnkleSlots.size(); ++i) {
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    double stationary_min = std::numeric_limits<double>::infinity();
    double stationary_max = -std::numeric_limits<double>::infinity();
    size_t stationary_count = 0;
    for (const auto& sample : samples) {
      const auto& motor = sample.motors.at(i);
      q_min = std::min(q_min, static_cast<double>(motor.q));
      q_max = std::max(q_max, static_cast<double>(motor.q));
      if (std::abs(motor.dq) <= 0.08F) {
        stationary_min =
            std::min(stationary_min, static_cast<double>(motor.q));
        stationary_max =
            std::max(stationary_max, static_cast<double>(motor.q));
        ++stationary_count;
      }
    }
    std::cout << "  slot " << kAnkleSlots.at(i) << " " << kAnkleNames.at(i)
              << ": raw=[" << std::showpos << std::fixed
              << std::setprecision(6) << q_min << ", " << q_max << "] rad";
    if (stationary_count != 0) {
      std::cout << ", |dq|<=0.08=[" << stationary_min << ", "
                << stationary_max << "] rad (n=" << std::noshowpos
                << stationary_count << ")";
    }
    std::cout << std::noshowpos << "\n";
  }
}

template <typename T>
class LatestSample {
 public:
  void Set(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = value;
    ++count_;
  }

  std::optional<T> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value_;
  }

  uint64_t Count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<T> value_;
  uint64_t count_ = 0;
};

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " <network_interface> [options]\n\n"
      << "Read-only H2 state snapshot. This program creates no command publisher.\n\n"
      << "Options:\n"
      << "  --domain-id N          DDS domain, default 0.\n"
      << "  --wait SEC             Collection window, default 5 seconds.\n"
      << "  --motor-count N        Motor slots to print, default 31, maximum 35.\n"
      << "  --ankle-capture-csv P  Continuously record ankle slots 4/5/10/11 to CSV.\n"
      << "  -h, --help             Show this help.\n";
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
    } else if (arg == "--domain-id") {
      args.domain_id = std::stoi(value("--domain-id"));
    } else if (arg == "--wait") {
      args.wait_seconds = std::stod(value("--wait"));
    } else if (arg == "--motor-count") {
      args.motor_count = std::stoi(value("--motor-count"));
    } else if (arg == "--ankle-capture-csv") {
      args.ankle_capture_csv = value("--ankle-capture-csv");
    } else if (arg.starts_with("-")) {
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
  if (args.wait_seconds <= 0.0 || args.wait_seconds > 60.0) {
    throw std::runtime_error("--wait must be in (0, 60]");
  }
  if (args.motor_count <= 0 || args.motor_count > kHgMotorSlots) {
    throw std::runtime_error("--motor-count must be in [1, 35]");
  }
  return args;
}

template <typename T>
void PrintScalar(T value) {
  if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
    std::cout << static_cast<int>(value);
  } else {
    std::cout << value;
  }
}

template <typename T, size_t N>
void PrintArray(const std::string& label, const std::array<T, N>& values) {
  std::cout << "  " << label << "=[";
  for (size_t i = 0; i < N; ++i) {
    if (i != 0) std::cout << ", ";
    PrintScalar(values[i]);
  }
  std::cout << "]\n";
}

void PrintImu(const std::string& label, const HgImuState& imu) {
  std::cout << label << ":\n";
  PrintArray("quaternion", imu.quaternion());
  PrintArray("gyroscope_rad_s", imu.gyroscope());
  PrintArray("accelerometer_m_s2", imu.accelerometer());
  PrintArray("rpy_rad", imu.rpy());
  std::cout << "  temperature=" << imu.temperature() << "\n";
}

const std::array<const char*, kH2MotorCount> kMotorNames{
    "left_hip_pitch",      "left_hip_roll",       "left_hip_yaw",
    "left_knee",           "left_ankle_roll",     "left_ankle_pitch",
    "right_hip_pitch",     "right_hip_roll",      "right_hip_yaw",
    "right_knee",          "right_ankle_roll",    "right_ankle_pitch",
    "waist_yaw",           "waist_roll_or_a",     "waist_pitch_or_b",
    "left_shoulder_pitch", "left_shoulder_roll",  "left_shoulder_yaw",
    "left_elbow",          "left_wrist_roll",     "left_wrist_pitch",
    "left_wrist_yaw",      "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw",  "right_elbow",         "right_wrist_roll",
    "right_wrist_pitch",   "right_wrist_yaw",     "head_pitch",
    "head_yaw"};

void PrintLowState(const std::string& topic, const HgLowState& state,
                   int motor_count, uint64_t samples) {
  std::cout << "\n=== " << topic << " (samples=" << samples << ") ===\n";
  PrintArray("version", state.version());
  std::cout << "  mode_pr=" << static_cast<int>(state.mode_pr())
            << " mode_machine=" << static_cast<int>(state.mode_machine())
            << " tick=" << state.tick() << " crc=0x" << std::hex
            << state.crc() << std::dec << "\n";
  PrintImu("pelvis_imu", state.imu_state());

  std::cout << "motors:\n"
            << "  idx name mode q dq ddq tau_est temp0 temp1 voltage "
               "sensor0 sensor1 motorstate reserve0 reserve1 reserve2 reserve3\n";
  for (int i = 0; i < motor_count; ++i) {
    const auto& motor = state.motor_state().at(i);
    const std::string name = i < kH2MotorCount ? kMotorNames.at(i) : "unused_slot";
    std::cout << "  " << std::setw(2) << i << " " << std::setw(21) << std::left
              << name << std::right << " " << std::setw(3)
              << static_cast<int>(motor.mode()) << " " << std::showpos
              << std::fixed << std::setprecision(6) << motor.q() << " "
              << motor.dq() << " " << motor.ddq() << " " << motor.tau_est()
              << std::noshowpos << " " << motor.temperature()[0] << " "
              << motor.temperature()[1] << " " << motor.vol() << " "
              << motor.sensor()[0] << " " << motor.sensor()[1] << " "
              << motor.motorstate();
    for (const uint32_t value : motor.reserve()) std::cout << " " << value;
    std::cout << "\n";
  }

  std::cout << "  wireless_remote_hex=";
  for (const uint8_t byte : state.wireless_remote()) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(byte);
  }
  std::cout << std::dec << std::setfill(' ') << "\n";
  PrintArray("reserve", state.reserve());
}

void PrintBms(const std::string& topic, const HgBmsState& state,
              uint64_t samples) {
  std::cout << "\n=== " << topic << " (samples=" << samples << ") ===\n"
            << "  version=" << static_cast<int>(state.version_high()) << "."
            << static_cast<int>(state.version_low())
            << " fn=" << static_cast<int>(state.fn())
            << " current=" << state.current()
            << " soc=" << static_cast<int>(state.soc())
            << " soh=" << static_cast<int>(state.soh())
            << " cycle=" << state.cycle()
            << " manufacturer_date=" << state.manufacturer_date() << "\n";
  PrintArray("cell_voltage", state.cell_vol());
  PrintArray("bms_voltage", state.bmsvoltage());
  PrintArray("temperature", state.temperature());
  PrintArray("bms_state", state.bmsstate());
  PrintArray("reserve", state.reserve());
}

void PrintMainBoard(const HgMainBoardState& state, uint64_t samples) {
  std::cout << "\n=== rt/lf/mainboardstate (samples=" << samples << ") ===\n";
  PrintArray("fan_state", state.fan_state());
  PrintArray("temperature", state.temperature());
  PrintArray("value", state.value());
  PrintArray("state", state.state());
}

void PrintHgSport(const HgSportModeState& state, uint64_t samples) {
  std::cout << "\n=== rt/sportmodestate (samples=" << samples << ") ===\n"
            << "  fsm_id=" << state.fsm_id()
            << " fsm_mode=" << state.fsm_mode()
            << " task_id=" << state.task_id()
            << " task_time=" << state.task_time() << "\n";
}

void PrintGoSport(const GoSportModeState& state, uint64_t samples) {
  std::cout << "\n=== rt/odommodestate (samples=" << samples << ") ===\n"
            << "  stamp=" << state.stamp().sec() << "." << std::setw(9)
            << std::setfill('0') << state.stamp().nanosec() << std::setfill(' ')
            << " error_code=" << state.error_code()
            << " mode=" << static_cast<int>(state.mode())
            << " progress=" << state.progress()
            << " gait_type=" << static_cast<int>(state.gait_type())
            << " foot_raise_height=" << state.foot_raise_height()
            << " body_height=" << state.body_height()
            << " yaw_speed=" << state.yaw_speed() << "\n";
  PrintArray("position", state.position());
  PrintArray("velocity", state.velocity());
  PrintArray("range_obstacle", state.range_obstacle());
  PrintArray("foot_force", state.foot_force());
  PrintArray("foot_position_body", state.foot_position_body());
  PrintArray("foot_speed_body", state.foot_speed_body());
  std::cout << "  path_points:\n";
  for (size_t i = 0; i < state.path_point().size(); ++i) {
    const auto& point = state.path_point()[i];
    std::cout << "    " << i << ": t=" << point.t_from_start()
              << " x=" << point.x() << " y=" << point.y()
              << " yaw=" << point.yaw() << " vx=" << point.vx()
              << " vy=" << point.vy() << " vyaw=" << point.vyaw() << "\n";
  }
}

void PrintWireless(const GoWirelessController& state, uint64_t samples) {
  std::cout << "\n=== rt/wirelesscontroller (samples=" << samples << ") ===\n"
            << "  lx=" << state.lx() << " ly=" << state.ly()
            << " rx=" << state.rx() << " ry=" << state.ry()
            << " keys=0x" << std::hex << state.keys() << std::dec << "\n";
}

template <typename T>
void InitSubscriber(ChannelSubscriber<T>& subscriber, LatestSample<T>& sample) {
  subscriber.InitChannel(
      [&sample](const void* message) {
        sample.Set(*static_cast<const T*>(message));
      },
      10);
}

template <typename T>
void ReportMissing(const std::string& topic, const LatestSample<T>& sample) {
  if (sample.Count() != 0) return;
  std::cout << "  MISSING  " << topic
            << " (not published or local IDL/QoS is incompatible)\n";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = ParseArgs(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n";
    PrintUsage(argv[0]);
    return 2;
  }

  std::cout << "H2 read-only state reader\n"
            << "  DDS domain=" << args.domain_id
            << " interface=" << args.network_interface
            << " wait=" << args.wait_seconds << "s\n"
            << "  Safety: subscribers only; no publisher, RPC client, or mode switch.\n";

  try {
    ChannelFactory::Instance()->Init(args.domain_id, args.network_interface);
  } catch (const std::exception& e) {
    std::cerr << "DDS initialization failed: " << e.what() << "\n";
    return 3;
  }

  LatestSample<HgLowState> lowstate;
  LatestSample<HgLowState> lowstate_raw;
  LatestSample<HgLowState> lowstate_lf;
  LatestSample<HgImuState> secondary_imu;
  LatestSample<HgBmsState> bms;
  LatestSample<HgBmsState> secondary_bms;
  LatestSample<HgMainBoardState> mainboard;
  LatestSample<HgSportModeState> hg_sport;
  LatestSample<GoSportModeState> odom;
  LatestSample<GoWirelessController> wireless;
  AnkleCapture ankle_capture;

  ChannelSubscriber<HgLowState> lowstate_sub("rt/lowstate");
  ChannelSubscriber<HgLowState> lowstate_raw_sub("rt/lowstate_raw");
  ChannelSubscriber<HgLowState> lowstate_lf_sub("rt/lf/lowstate");
  ChannelSubscriber<HgImuState> secondary_imu_sub("rt/secondary_imu");
  ChannelSubscriber<HgBmsState> bms_sub("rt/lf/bmsstate");
  ChannelSubscriber<HgBmsState> secondary_bms_sub("rt/lf/secondary_bmsstate");
  ChannelSubscriber<HgMainBoardState> mainboard_sub("rt/lf/mainboardstate");
  ChannelSubscriber<HgSportModeState> hg_sport_sub("rt/sportmodestate");
  ChannelSubscriber<GoSportModeState> odom_sub("rt/odommodestate");
  ChannelSubscriber<GoWirelessController> wireless_sub("rt/wirelesscontroller");

  lowstate_sub.InitChannel(
      [&lowstate, &ankle_capture, capture = !args.ankle_capture_csv.empty()](
          const void* message) {
        const auto& state = *static_cast<const HgLowState*>(message);
        lowstate.Set(state);
        if (capture) ankle_capture.Add(state);
      },
      10);
  InitSubscriber(lowstate_raw_sub, lowstate_raw);
  InitSubscriber(lowstate_lf_sub, lowstate_lf);
  InitSubscriber(secondary_imu_sub, secondary_imu);
  InitSubscriber(bms_sub, bms);
  InitSubscriber(secondary_bms_sub, secondary_bms);
  InitSubscriber(mainboard_sub, mainboard);
  InitSubscriber(hg_sport_sub, hg_sport);
  InitSubscriber(odom_sub, odom);
  InitSubscriber(wireless_sub, wireless);

  std::this_thread::sleep_for(std::chrono::duration<double>(args.wait_seconds));

  if (!args.ankle_capture_csv.empty()) {
    try {
      const auto samples = ankle_capture.Get();
      WriteAnkleCapture(args.ankle_capture_csv, samples);
      std::cout << "\n[INFO] Wrote " << samples.size()
                << " read-only ankle samples to " << args.ankle_capture_csv
                << "\n";
      PrintAnkleExtrema(samples);
    } catch (const std::exception& e) {
      std::cerr << "Failed to write ankle capture: " << e.what() << "\n";
      return 5;
    }
  }

  if (const auto state = lowstate.Get())
    PrintLowState("rt/lowstate", *state, args.motor_count, lowstate.Count());
  else if (const auto state = lowstate_lf.Get())
    PrintLowState("rt/lf/lowstate", *state, args.motor_count, lowstate_lf.Count());
  else if (const auto state = lowstate_raw.Get())
    PrintLowState("rt/lowstate_raw", *state, args.motor_count, lowstate_raw.Count());

  if (const auto state = secondary_imu.Get())
    PrintImu("\n=== rt/secondary_imu (samples=" +
                 std::to_string(secondary_imu.Count()) + ") ===",
             *state);
  if (const auto state = bms.Get()) PrintBms("rt/lf/bmsstate", *state, bms.Count());
  if (const auto state = secondary_bms.Get())
    PrintBms("rt/lf/secondary_bmsstate", *state, secondary_bms.Count());
  if (const auto state = mainboard.Get()) PrintMainBoard(*state, mainboard.Count());
  if (const auto state = hg_sport.Get()) PrintHgSport(*state, hg_sport.Count());
  if (const auto state = odom.Get()) PrintGoSport(*state, odom.Count());
  if (const auto state = wireless.Get()) PrintWireless(*state, wireless.Count());

  std::cout << "\n=== coverage ===\n";
  bool any = lowstate.Count() || lowstate_raw.Count() || lowstate_lf.Count() ||
             secondary_imu.Count() || bms.Count() || secondary_bms.Count() ||
             mainboard.Count() || hg_sport.Count() || odom.Count() || wireless.Count();
  if (!lowstate.Count() && !lowstate_raw.Count() && !lowstate_lf.Count()) {
    std::cout << "  MISSING  rt/lowstate (including aliases)"
              << " (not published or local IDL/QoS is incompatible)\n";
  }
  ReportMissing("rt/secondary_imu", secondary_imu);
  ReportMissing("rt/lf/bmsstate", bms);
  ReportMissing("rt/lf/secondary_bmsstate", secondary_bms);
  ReportMissing("rt/lf/mainboardstate", mainboard);
  ReportMissing("rt/sportmodestate", hg_sport);
  ReportMissing("rt/odommodestate", odom);
  ReportMissing("rt/wirelesscontroller", wireless);

  return any ? 0 : 4;
}
