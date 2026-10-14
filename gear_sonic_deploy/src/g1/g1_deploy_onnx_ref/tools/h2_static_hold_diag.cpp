#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "policy_parameters.hpp"
#include "robot_parameters.hpp"

#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using unitree_hg::msg::dds_::IMUState_;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

constexpr int kModePR = 0;
constexpr int kHgMotorCount = 35;
constexpr double kPi = 3.14159265358979323846;

struct Args {
  std::string network_interface;
  int domain_id = 0;
  std::string target_mode = "current";
  bool risk_ack = false;
  bool no_prompt = false;
  bool skip_release_mode = false;
  bool allow_large_move = false;
  int num_motors = H2_NUM_MOTOR;
  double lowstate_timeout = 5.0;
  double torso_imu_timeout = 1.0;
  double ramp_duration = 4.0;
  double hold_duration = 12.0;
  double rate_hz = 250.0;
  double kp_scale = 1.0;
  double kd_scale = 1.0;
  double max_target_diff = 1.0;
  double summary_tail_s = 2.0;
  std::vector<std::pair<int, double>> offsets;
  std::vector<int> watch_indices{0, 3, 4, 5, 6, 9, 10, 11, 13, 14};
  std::string log_path;
};

struct ImuSnapshot {
  bool valid = false;
  std::array<double, 4> quat{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  std::array<double, 3> rpy{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  std::array<double, 3> gyro{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  std::array<double, 3> accel{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
};

struct Sample {
  double t_wall = 0.0;
  double elapsed = 0.0;
  std::string phase;
  int mode_pr = 0;
  int mode_machine = 0;
  ImuSnapshot base_imu;
  ImuSnapshot torso_imu;
  std::array<double, H2_NUM_MOTOR> target{};
  std::array<double, H2_NUM_MOTOR> q{};
  std::array<double, H2_NUM_MOTOR> dq{};
  std::array<double, H2_NUM_MOTOR> tau{};
};

uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t crc32 = 0xFFFFFFFF;
  const uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (crc32 & 0x80000000) {
        crc32 <<= 1;
        crc32 ^= polynomial;
      } else {
        crc32 <<= 1;
      }
      if (data & xbit) {
        crc32 ^= polynomial;
      }
      xbit >>= 1;
    }
  }
  return crc32;
}

double WallTimeSeconds() {
  using clock = std::chrono::system_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
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

double RadToDeg(double rad) {
  return rad * 180.0 / kPi;
}

std::array<double, 3> QuaternionToRpy(std::array<double, 4> q) {
  const double norm =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < 1e-9 || !std::isfinite(norm)) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  for (double& v : q) {
    v /= norm;
  }
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
  const double pitch = std::asin(sinp);

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(siny_cosp, cosy_cosp);
  return {roll, pitch, yaw};
}

std::array<double, 4> ReadQuat(const IMUState_& imu) {
  std::array<double, 4> out{};
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<double>(imu.quaternion()[i]);
  }
  return out;
}

std::array<double, 3> ReadGyro(const IMUState_& imu) {
  std::array<double, 3> out{};
  for (int i = 0; i < 3; ++i) {
    out[i] = static_cast<double>(imu.gyroscope()[i]);
  }
  return out;
}

std::array<double, 3> ReadAccel(const IMUState_& imu) {
  std::array<double, 3> out{};
  for (int i = 0; i < 3; ++i) {
    out[i] = static_cast<double>(imu.accelerometer()[i]);
  }
  return out;
}

ImuSnapshot SnapshotImu(const IMUState_& imu) {
  ImuSnapshot out;
  out.valid = true;
  out.quat = ReadQuat(imu);
  out.rpy = QuaternionToRpy(out.quat);
  out.gyro = ReadGyro(imu);
  out.accel = ReadAccel(imu);
  return out;
}

std::string JointName(int dds_index) {
  if (dds_index >= 0 && dds_index < H2_NUM_MOTOR) {
    return h2_joint_names_dds.at(dds_index);
  }
  return "unknown";
}

std::vector<int> ParseIndices(const std::string& text) {
  std::vector<int> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stoi(item));
    }
  }
  if (out.empty()) {
    throw std::runtime_error("index list must contain at least one index");
  }
  return out;
}

std::pair<int, double> ParseOffset(const std::string& text) {
  const size_t pos = text.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("--offset expects INDEX:RAD, for example 14:-0.03");
  }
  const int index = std::stoi(text.substr(0, pos));
  const double value = std::stod(text.substr(pos + 1));
  return {index, value};
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " <network_interface> [options]\n\n"
      << "Static H2 hold diagnostic. Sends PR LowCmd packets and logs fixed-target\n"
      << "tracking without running the policy.\n\n"
      << "Options:\n"
      << "  --target current|default      Hold current measured q or H2 default, default current.\n"
      << "  --offset INDEX:RAD            Add target offset in DDS slot order. Repeatable.\n"
      << "  --watch A,B,C                 Console summary slots, default legs/waist subset.\n"
      << "  --domain-id N                 DDS domain id, default 0.\n"
      << "  --lowstate-timeout SEC        First LowState timeout, default 5.\n"
      << "  --torso-imu-timeout SEC       Secondary IMU wait timeout, default 1.\n"
      << "  --ramp-duration SEC           Ramp from current q to target, default 4.\n"
      << "  --hold-duration SEC           Fixed target hold duration, default 12.\n"
      << "  --rate-hz HZ                  LowCmd publish rate, default 250.\n"
      << "  --kp-scale S --kd-scale S     Scale H2 deploy gains, default 1/1.\n"
      << "  --max-target-diff RAD         Refuse big initial moves, default 1.0.\n"
      << "  --allow-large-move            Override max target diff refusal.\n"
      << "  --skip-release-mode           Do not call MotionSwitcher ReleaseMode.\n"
      << "  --no-prompt                   Do not pause before sending LowCmd.\n"
      << "  --i-understand-ground-test    Required before command mode.\n"
      << "  --log PATH                    CSV output path.\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto require_value = [&](const std::string& opt) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(opt + " requires a value");
      }
      return argv[++i];
    };

    if (a == "-h" || a == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (a == "--target") {
      args.target_mode = require_value(a);
    } else if (a == "--offset") {
      args.offsets.push_back(ParseOffset(require_value(a)));
    } else if (a == "--watch") {
      args.watch_indices = ParseIndices(require_value(a));
    } else if (a == "--domain-id") {
      args.domain_id = std::stoi(require_value(a));
    } else if (a == "--lowstate-timeout") {
      args.lowstate_timeout = std::stod(require_value(a));
    } else if (a == "--torso-imu-timeout") {
      args.torso_imu_timeout = std::stod(require_value(a));
    } else if (a == "--ramp-duration") {
      args.ramp_duration = std::stod(require_value(a));
    } else if (a == "--hold-duration") {
      args.hold_duration = std::stod(require_value(a));
    } else if (a == "--rate-hz") {
      args.rate_hz = std::stod(require_value(a));
    } else if (a == "--kp-scale") {
      args.kp_scale = std::stod(require_value(a));
    } else if (a == "--kd-scale") {
      args.kd_scale = std::stod(require_value(a));
    } else if (a == "--max-target-diff") {
      args.max_target_diff = std::stod(require_value(a));
    } else if (a == "--summary-tail-s") {
      args.summary_tail_s = std::stod(require_value(a));
    } else if (a == "--log") {
      args.log_path = require_value(a);
    } else if (a == "--skip-release-mode") {
      args.skip_release_mode = true;
    } else if (a == "--no-prompt") {
      args.no_prompt = true;
    } else if (a == "--allow-large-move") {
      args.allow_large_move = true;
    } else if (a == "--i-understand-ground-test") {
      args.risk_ack = true;
    } else if (!a.empty() && a[0] == '-') {
      throw std::runtime_error("unknown option: " + a);
    } else if (args.network_interface.empty()) {
      args.network_interface = a;
    } else {
      throw std::runtime_error("unexpected positional argument: " + a);
    }
  }
  return args;
}

template <typename T>
class LatestBuffer {
 public:
  void Set(const T& value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      value_ = value;
    }
    cv_.notify_all();
  }

  std::optional<T> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return value_;
  }

  std::optional<T> Wait(double timeout_s) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto timeout = std::chrono::duration<double>(timeout_s);
    cv_.wait_for(lock, timeout, [&] { return value_.has_value(); });
    return value_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::optional<T> value_;
};

std::array<double, H2_NUM_MOTOR> ReadMotorQ(const LowState_& state) {
  std::array<double, H2_NUM_MOTOR> q{};
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    q[i] = state.motor_state().at(i).q();
  }
  return q;
}

std::array<double, H2_NUM_MOTOR> MakeDefaultTarget() {
  std::array<double, H2_NUM_MOTOR> target{};
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    target[i] = H2DefaultAngleDds(i);
  }
  return target;
}

void ApplyOffsets(std::array<double, H2_NUM_MOTOR>* target,
                  const std::vector<std::pair<int, double>>& offsets) {
  for (const auto& [idx, value] : offsets) {
    if (idx < 0 || idx >= H2_NUM_MOTOR) {
      throw std::runtime_error("--offset index out of H2 motor range: " +
                               std::to_string(idx));
    }
    target->at(idx) += value;
  }
}

LowCmd_ BuildCmd(const std::array<double, H2_NUM_MOTOR>& target_q,
                 int mode_machine, const Args& args) {
  LowCmd_ cmd;
  cmd.mode_pr() = static_cast<uint8_t>(kModePR);
  cmd.mode_machine() = static_cast<uint8_t>(mode_machine);
  for (int i = 0; i < kHgMotorCount; ++i) {
    auto& motor = cmd.motor_cmd().at(i);
    if (i < args.num_motors) {
      motor.mode() = 1;
      motor.q() = static_cast<float>(target_q.at(i));
      motor.dq() = 0.0f;
      motor.tau() = 0.0f;
      motor.kp() = static_cast<float>(H2KpDds(i) * args.kp_scale);
      motor.kd() = static_cast<float>(H2KdDds(i) * args.kd_scale);
    } else {
      motor.mode() = 0;
      motor.q() = 0.0f;
      motor.dq() = 0.0f;
      motor.tau() = 0.0f;
      motor.kp() = 0.0f;
      motor.kd() = 0.0f;
    }
  }
  cmd.crc() = Crc32Core(reinterpret_cast<uint32_t*>(&cmd), (sizeof(cmd) >> 2) - 1);
  return cmd;
}

Sample MakeSample(const LowState_& lowstate, const std::optional<IMUState_>& torso_imu,
                  const std::array<double, H2_NUM_MOTOR>& target,
                  const std::string& phase, double elapsed) {
  Sample sample;
  sample.t_wall = WallTimeSeconds();
  sample.elapsed = elapsed;
  sample.phase = phase;
  sample.mode_pr = lowstate.mode_pr();
  sample.mode_machine = lowstate.mode_machine();
  sample.base_imu = SnapshotImu(lowstate.imu_state());
  if (torso_imu.has_value()) {
    sample.torso_imu = SnapshotImu(*torso_imu);
  }
  sample.target = target;
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    sample.q[i] = lowstate.motor_state().at(i).q();
    sample.dq[i] = lowstate.motor_state().at(i).dq();
    sample.tau[i] = lowstate.motor_state().at(i).tau_est();
  }
  return sample;
}

void PublishFor(ChannelPublisher<LowCmd_>& publisher,
                const LatestBuffer<LowState_>& lowstate_buffer,
                const LatestBuffer<IMUState_>& torso_imu_buffer,
                const Args& args, int mode_machine,
                const std::array<double, H2_NUM_MOTOR>& start_target,
                const std::array<double, H2_NUM_MOTOR>& final_target,
                double duration_s, const std::string& phase,
                std::vector<Sample>* samples, double* elapsed_accum) {
  const auto period = std::chrono::duration<double>(1.0 / args.rate_hz);
  auto next = std::chrono::steady_clock::now();
  const double start_elapsed = *elapsed_accum;
  while (*elapsed_accum - start_elapsed < duration_s) {
    const double local_elapsed = *elapsed_accum - start_elapsed;
    const double alpha =
        duration_s > 1e-9 ? std::clamp(local_elapsed / duration_s, 0.0, 1.0) : 1.0;
    std::array<double, H2_NUM_MOTOR> target{};
    if (phase == "ramp") {
      for (int i = 0; i < H2_NUM_MOTOR; ++i) {
        target[i] = start_target[i] + alpha * (final_target[i] - start_target[i]);
      }
    } else {
      target = final_target;
    }

    publisher.Write(BuildCmd(target, mode_machine, args));

    const auto lowstate = lowstate_buffer.Get();
    if (lowstate.has_value()) {
      samples->push_back(MakeSample(*lowstate, torso_imu_buffer.Get(), target,
                                    phase, *elapsed_accum));
    }

    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    std::this_thread::sleep_until(next);
    *elapsed_accum += 1.0 / args.rate_hz;
  }
}

void WriteLog(const std::string& path, const std::vector<Sample>& samples) {
  std::filesystem::path out_path(path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream f(path);
  f << "time_wall_s,elapsed_s,phase,mode_pr,mode_machine";
  const std::array<std::string, 3> axes{"roll", "pitch", "yaw"};
  f << ",base_qw,base_qx,base_qy,base_qz";
  for (const auto& axis : axes) {
    f << ",base_" << axis;
  }
  f << ",base_gx,base_gy,base_gz,base_ax,base_ay,base_az";
  f << ",torso_has,torso_qw,torso_qx,torso_qy,torso_qz";
  for (const auto& axis : axes) {
    f << ",torso_" << axis;
  }
  f << ",torso_gx,torso_gy,torso_gz,torso_ax,torso_ay,torso_az";
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    f << ",target_" << i << ",q_" << i << ",dq_" << i << ",tau_" << i
      << ",err_" << i;
  }
  f << "\n";

  f << std::fixed << std::setprecision(9);
  for (const auto& sample : samples) {
    f << sample.t_wall << "," << sample.elapsed << "," << sample.phase << ","
      << sample.mode_pr << "," << sample.mode_machine;
    for (double v : sample.base_imu.quat) {
      f << "," << v;
    }
    for (double v : sample.base_imu.rpy) {
      f << "," << v;
    }
    for (double v : sample.base_imu.gyro) {
      f << "," << v;
    }
    for (double v : sample.base_imu.accel) {
      f << "," << v;
    }
    f << "," << (sample.torso_imu.valid ? 1 : 0);
    for (double v : sample.torso_imu.quat) {
      f << "," << v;
    }
    for (double v : sample.torso_imu.rpy) {
      f << "," << v;
    }
    for (double v : sample.torso_imu.gyro) {
      f << "," << v;
    }
    for (double v : sample.torso_imu.accel) {
      f << "," << v;
    }
    for (int i = 0; i < H2_NUM_MOTOR; ++i) {
      f << "," << sample.target[i] << "," << sample.q[i] << "," << sample.dq[i]
        << "," << sample.tau[i] << "," << sample.target[i] - sample.q[i];
    }
    f << "\n";
  }
}

struct MeanStats {
  int count = 0;
  double base_pitch = 0.0;
  double torso_pitch = 0.0;
  bool has_torso = false;
  std::array<double, H2_NUM_MOTOR> target{};
  std::array<double, H2_NUM_MOTOR> q{};
  std::array<double, H2_NUM_MOTOR> tau{};
};

std::optional<MeanStats> SummarizeTail(const std::vector<Sample>& samples,
                                       double tail_s) {
  if (samples.empty()) {
    return std::nullopt;
  }
  const double last_elapsed = samples.back().elapsed;
  MeanStats stats;
  for (const auto& sample : samples) {
    if (sample.phase != "hold" || sample.elapsed < last_elapsed - tail_s) {
      continue;
    }
    ++stats.count;
    stats.base_pitch += sample.base_imu.rpy[1];
    if (sample.torso_imu.valid) {
      stats.has_torso = true;
      stats.torso_pitch += sample.torso_imu.rpy[1];
    }
    for (int i = 0; i < H2_NUM_MOTOR; ++i) {
      stats.target[i] += sample.target[i];
      stats.q[i] += sample.q[i];
      stats.tau[i] += sample.tau[i];
    }
  }
  if (stats.count == 0) {
    return std::nullopt;
  }
  const double inv = 1.0 / static_cast<double>(stats.count);
  stats.base_pitch *= inv;
  if (stats.has_torso) {
    stats.torso_pitch *= inv;
  }
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    stats.target[i] *= inv;
    stats.q[i] *= inv;
    stats.tau[i] *= inv;
  }
  return stats;
}

void MaybeReleaseMotionMode(bool skip_release) {
  if (skip_release) {
    std::cout << "[INFO] Skipping MotionSwitcher ReleaseMode.\n";
    return;
  }

  std::cout << "[INFO] Releasing motion-control service via MotionSwitcherClient...\n";
  auto client = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
  client->SetTimeout(5.0f);
  client->Init();
  for (int attempt = 0; attempt < 5; ++attempt) {
    std::string form;
    std::string name;
    const int32_t check_code = client->CheckMode(form, name);
    if (check_code != 0) {
      std::cout << "[WARN] MotionSwitcher CheckMode failed with code "
                << check_code << ".\n";
      return;
    }
    if (name.empty()) {
      std::cout << "[INFO] Motion-control service already released.\n";
      return;
    }
    std::cout << "[INFO] Active service form='" << form << "', name='" << name
              << "'; calling ReleaseMode().\n";
    const int32_t release_code = client->ReleaseMode();
    if (release_code != 0) {
      std::cout << "[WARN] ReleaseMode failed with code " << release_code << ".\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "[WARN] MotionSwitcher still reports active service after retries.\n";
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

  if (args.network_interface.empty()) {
    std::cerr << "Missing network_interface.\n\n";
    PrintUsage(argv[0]);
    return 2;
  }
  if (args.target_mode != "current" && args.target_mode != "default") {
    std::cerr << "--target must be 'current' or 'default'.\n";
    return 2;
  }
  if (!args.risk_ack) {
    std::cerr
        << "Refusing to send LowCmd without explicit acknowledgement.\n"
        << "Re-run with --i-understand-ground-test after the robot is supported\n"
        << "or you are ready to catch/disable it if the fixed pose is unstable.\n";
    return 2;
  }
  if (args.num_motors != H2_NUM_MOTOR) {
    std::cerr << "This diagnostic is H2-only and expects " << H2_NUM_MOTOR
              << " motors.\n";
    return 2;
  }
  for (int idx : args.watch_indices) {
    if (idx < 0 || idx >= H2_NUM_MOTOR) {
      std::cerr << "--watch index out of H2 motor range: " << idx << "\n";
      return 2;
    }
  }
  if (args.log_path.empty()) {
    args.log_path = "/tmp/h2_static_hold_diag/h2_static_hold_" +
                    TimestampForPath() + ".csv";
  }

  std::cout << "\n=== H2 Static Hold Diagnostic ===\n"
            << "DDS: domain=" << args.domain_id
            << ", interface=" << args.network_interface << "\n"
            << "Target mode: " << args.target_mode << "; ramp="
            << args.ramp_duration << "s; hold=" << args.hold_duration
            << "s; rate=" << args.rate_hz << "Hz\n"
            << "Gains: kp_scale=" << args.kp_scale
            << ", kd_scale=" << args.kd_scale << "\n";
  if (!args.offsets.empty()) {
    std::cout << "Target offsets:\n";
    for (const auto& [idx, value] : args.offsets) {
      std::cout << "  " << std::setw(2) << idx << " (" << JointName(idx)
                << "): " << std::showpos << std::fixed << std::setprecision(4)
                << value << std::noshowpos << " rad\n";
    }
  }
  std::cout << "Watch slots:\n";
  for (int idx : args.watch_indices) {
    std::cout << "  " << std::setw(2) << idx << ": " << JointName(idx) << "\n";
  }
  std::cout << "\n";

  try {
    ChannelFactory::Instance()->Init(args.domain_id, args.network_interface);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] DDS ChannelFactory init failed: " << e.what() << "\n";
    return 3;
  }

  LatestBuffer<LowState_> lowstate_buffer;
  LatestBuffer<IMUState_> torso_imu_buffer;
  bool crc_warned = false;

  ChannelSubscriber<LowState_> lowstate_subscriber(HG_STATE_TOPIC);
  lowstate_subscriber.InitChannel(
      [&](const void* message) {
        LowState_ state = *static_cast<const LowState_*>(message);
        const uint32_t crc =
            Crc32Core(reinterpret_cast<uint32_t*>(&state), (sizeof(state) >> 2) - 1);
        if (state.crc() != crc && !crc_warned) {
          crc_warned = true;
          std::cerr << "[WARN] LowState CRC mismatch once; keeping samples.\n";
        }
        lowstate_buffer.Set(state);
      },
      10);

  ChannelSubscriber<IMUState_> torso_imu_subscriber(HG_IMU_TORSO);
  torso_imu_subscriber.InitChannel(
      [&](const void* message) {
        torso_imu_buffer.Set(*static_cast<const IMUState_*>(message));
      },
      10);

  std::cout << "[INFO] Waiting for rt/lowstate...\n";
  const auto first_state = lowstate_buffer.Wait(args.lowstate_timeout);
  if (!first_state.has_value()) {
    std::cerr << "[ERROR] Timed out after " << args.lowstate_timeout
              << "s waiting for rt/lowstate.\n";
    return 3;
  }
  const auto first_torso = torso_imu_buffer.Wait(args.torso_imu_timeout);
  if (!first_torso.has_value()) {
    std::cout << "[WARN] No rt/secondary_imu within " << args.torso_imu_timeout
              << "s. Continuing with base IMU only.\n";
  }

  const int mode_machine = first_state->mode_machine();
  const auto start_q = ReadMotorQ(*first_state);
  std::array<double, H2_NUM_MOTOR> final_target =
      args.target_mode == "default" ? MakeDefaultTarget() : start_q;
  try {
    ApplyOffsets(&final_target, args.offsets);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    return 2;
  }

  double max_diff = 0.0;
  int max_diff_idx = 0;
  for (int i = 0; i < H2_NUM_MOTOR; ++i) {
    const double diff = std::abs(final_target[i] - start_q[i]);
    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }
  }
  std::cout << "[INFO] Received lowstate. mode_pr=" << int(first_state->mode_pr())
            << ", mode_machine=" << mode_machine << "\n"
            << "[INFO] Largest start->target difference: slot " << max_diff_idx
            << " (" << JointName(max_diff_idx) << ") = "
            << std::fixed << std::setprecision(4) << max_diff << " rad\n";

  if (max_diff > args.max_target_diff && !args.allow_large_move) {
    std::cerr << "[ERROR] Refusing because largest target jump exceeds "
              << args.max_target_diff << " rad. Move the robot closer to the pose,"
              << " increase --ramp-duration, or add --allow-large-move if intended.\n";
    return 2;
  }

  std::cout << "[INFO] Initial watched slots:\n";
  for (int idx : args.watch_indices) {
    std::cout << "  " << std::setw(2) << idx << " " << JointName(idx)
              << " q=" << std::showpos << std::fixed << std::setprecision(5)
              << start_q[idx] << " target=" << final_target[idx]
              << " target-q=" << final_target[idx] - start_q[idx]
              << std::noshowpos << "\n";
  }

  if (!args.no_prompt) {
    std::cout << "\nThis will send fixed-position LowCmd packets. Press Enter to start, "
                 "or Ctrl+C to abort...";
    std::cout.flush();
    std::string line;
    std::getline(std::cin, line);
  }

  MaybeReleaseMotionMode(args.skip_release_mode);

  ChannelPublisher<LowCmd_> publisher(HG_CMD_TOPIC);
  publisher.InitChannel();

  std::vector<Sample> samples;
  double elapsed = 0.0;
  std::cout << "[INFO] Ramping to target for " << args.ramp_duration << "s...\n";
  PublishFor(publisher, lowstate_buffer, torso_imu_buffer, args, mode_machine,
             start_q, final_target, args.ramp_duration, "ramp", &samples,
             &elapsed);

  std::cout << "[INFO] Holding fixed target for " << args.hold_duration << "s...\n";
  PublishFor(publisher, lowstate_buffer, torso_imu_buffer, args, mode_machine,
             final_target, final_target, args.hold_duration, "hold", &samples,
             &elapsed);

  // Keep the last target alive briefly while the final samples are flushed.
  PublishFor(publisher, lowstate_buffer, torso_imu_buffer, args, mode_machine,
             final_target, final_target, 0.25, "final_hold", &samples,
             &elapsed);

  WriteLog(args.log_path, samples);
  std::cout << "[INFO] Wrote CSV log: " << args.log_path << "\n";

  const auto summary = SummarizeTail(samples, args.summary_tail_s);
  if (summary.has_value()) {
    std::cout << "\n=== Tail Summary (last " << args.summary_tail_s
              << "s of hold) ===\n";
    std::cout << "samples=" << summary->count
              << ", base_pitch=" << std::showpos << std::fixed
              << std::setprecision(3) << RadToDeg(summary->base_pitch)
              << " deg";
    if (summary->has_torso) {
      std::cout << ", torso_pitch=" << RadToDeg(summary->torso_pitch) << " deg";
    }
    std::cout << std::noshowpos << "\n";
    std::cout << "slot,joint,target,q,target-q,tau_est\n";
    for (int idx : args.watch_indices) {
      std::cout << std::setw(2) << idx << "," << JointName(idx) << ","
                << std::showpos << std::fixed << std::setprecision(6)
                << summary->target[idx] << "," << summary->q[idx] << ","
                << summary->target[idx] - summary->q[idx] << ","
                << summary->tau[idx] << std::noshowpos << "\n";
    }
  } else {
    std::cout << "[WARN] Not enough hold samples for summary.\n";
  }

  return 0;
}
