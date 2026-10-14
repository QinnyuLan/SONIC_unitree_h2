#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
constexpr int kHgMotorCount = 35;
constexpr int kDefaultH2MotorCount = 31;
constexpr const char* kLowCmdTopic = "rt/lowcmd";
constexpr const char* kLowStateTopic = "rt/lowstate";

struct Args {
  std::string network_interface;
  int domain_id = 0;
  bool read_only = false;
  bool suspended_ack = false;
  bool include_waist = false;
  bool negative = false;
  bool skip_release_mode = false;
  bool no_prompt = false;
  int num_motors = kDefaultH2MotorCount;
  double delta = 0.10;
  double lowstate_timeout = 5.0;
  double warmup_duration = 1.0;
  double settle_duration = 0.7;
  double step_duration = 1.2;
  double rate_hz = 250.0;
  double kp = 40.0;
  double kd = 1.0;
  double hold_kp = 20.0;
  double hold_kd = 1.0;
  double tail_s = 0.35;
  std::vector<int> indices{4, 5, 10, 11};
  std::string log_path;
};

struct Sample {
  double t = 0.0;
  std::string phase;
  int commanded_index = -1;
  std::vector<double> target;
  std::vector<double> q;
  std::vector<double> dq;
  std::vector<double> tau;
  int mode_pr = 0;
  int mode_machine = 0;
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

std::string WatchName(int idx) {
  static const std::map<int, std::string> names{
      {4, "slot4 / code-assumed L ankle roll / Unitree-PR-example L ankle pitch"},
      {5, "slot5 / code-assumed L ankle pitch / Unitree-PR-example L ankle roll"},
      {10, "slot10 / code-assumed R ankle roll / Unitree-PR-example R ankle pitch"},
      {11, "slot11 / code-assumed R ankle pitch / Unitree-PR-example R ankle roll"},
      {12, "slot12 / waist yaw"},
      {13, "slot13 / waist roll or waist A"},
      {14, "slot14 / waist pitch or waist B"},
  };
  const auto it = names.find(idx);
  return it == names.end() ? "unnamed slot" : it->second;
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
    throw std::runtime_error("--indices must contain at least one index");
  }
  return out;
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " <network_interface> [options]\n\n"
      << "Options:\n"
      << "  --read-only                 Subscribe to rt/lowstate only; send no LowCmd.\n"
      << "  --i-understand-suspended    Required for command mode.\n"
      << "  --domain-id N               DDS domain id, default 0.\n"
      << "  --lowstate-timeout SEC      First LowState timeout, default 5.\n"
      << "  --indices A,B,C             Command slots, default 4,5,10,11.\n"
      << "  --include-waist             Also test 13 and 14.\n"
      << "  --num-motors N              Active H2 motor slots, default 31.\n"
      << "  --delta RAD                 Step size, default 0.10.\n"
      << "  --negative                  Also test negative step.\n"
      << "  --kp K --kd D               Active slot gains, default 40/1.\n"
      << "  --hold-kp K --hold-kd D     Hold gains, default 20/1.\n"
      << "  --rate-hz HZ                Publish rate, default 250.\n"
      << "  --skip-release-mode         Do not call MotionSwitcher ReleaseMode.\n"
      << "  --no-prompt                 Do not pause before each step.\n"
      << "  --log PATH                  CSV output path.\n";
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
    } else if (a == "--read-only") {
      args.read_only = true;
    } else if (a == "--i-understand-suspended") {
      args.suspended_ack = true;
    } else if (a == "--include-waist") {
      args.include_waist = true;
    } else if (a == "--negative") {
      args.negative = true;
    } else if (a == "--skip-release-mode") {
      args.skip_release_mode = true;
    } else if (a == "--no-prompt") {
      args.no_prompt = true;
    } else if (a == "--domain-id") {
      args.domain_id = std::stoi(require_value(a));
    } else if (a == "--lowstate-timeout") {
      args.lowstate_timeout = std::stod(require_value(a));
    } else if (a == "--indices") {
      args.indices = ParseIndices(require_value(a));
    } else if (a == "--num-motors") {
      args.num_motors = std::stoi(require_value(a));
    } else if (a == "--delta") {
      args.delta = std::stod(require_value(a));
    } else if (a == "--warmup-duration") {
      args.warmup_duration = std::stod(require_value(a));
    } else if (a == "--settle-duration") {
      args.settle_duration = std::stod(require_value(a));
    } else if (a == "--step-duration") {
      args.step_duration = std::stod(require_value(a));
    } else if (a == "--rate-hz") {
      args.rate_hz = std::stod(require_value(a));
    } else if (a == "--kp") {
      args.kp = std::stod(require_value(a));
    } else if (a == "--kd") {
      args.kd = std::stod(require_value(a));
    } else if (a == "--hold-kp") {
      args.hold_kp = std::stod(require_value(a));
    } else if (a == "--hold-kd") {
      args.hold_kd = std::stod(require_value(a));
    } else if (a == "--tail-s") {
      args.tail_s = std::stod(require_value(a));
    } else if (a == "--log") {
      args.log_path = require_value(a);
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

class LowStateBuffer {
 public:
  void Set(const LowState_& state) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      state_ = state;
    }
    cv_.notify_all();
  }

  std::optional<LowState_> Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::optional<LowState_> Wait(double timeout_s) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto timeout = std::chrono::duration<double>(timeout_s);
    cv_.wait_for(lock, timeout, [&] { return state_.has_value(); });
    return state_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::optional<LowState_> state_;
};

void ReadQDtau(const LowState_& state, int count, std::vector<double>* q,
              std::vector<double>* dq, std::vector<double>* tau) {
  q->assign(count, 0.0);
  dq->assign(count, 0.0);
  tau->assign(count, 0.0);
  for (int i = 0; i < count; ++i) {
    (*q)[i] = state.motor_state().at(i).q();
    (*dq)[i] = state.motor_state().at(i).dq();
    (*tau)[i] = state.motor_state().at(i).tau_est();
  }
}

LowCmd_ BuildCmd(const std::vector<double>& target_q, int mode_machine,
                 int num_motors, const std::set<int>& active_indices,
                 double kp, double kd, double hold_kp, double hold_kd) {
  LowCmd_ cmd;
  cmd.mode_pr() = static_cast<uint8_t>(kModePR);
  cmd.mode_machine() = static_cast<uint8_t>(mode_machine);
  for (int i = 0; i < kHgMotorCount; ++i) {
    auto& motor = cmd.motor_cmd().at(i);
    if (i < num_motors) {
      motor.mode() = 1;
      motor.q() = static_cast<float>(target_q.at(i));
      motor.dq() = 0.0f;
      motor.tau() = 0.0f;
      if (active_indices.count(i) != 0) {
        motor.kp() = static_cast<float>(kp);
        motor.kd() = static_cast<float>(kd);
      } else {
        motor.kp() = static_cast<float>(hold_kp);
        motor.kd() = static_cast<float>(hold_kd);
      }
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

void PublishPhase(ChannelPublisher<LowCmd_>& publisher,
                  const LowStateBuffer& state_buffer,
                  const std::vector<double>& target_q, int mode_machine,
                  int num_motors, const std::set<int>& active_indices,
                  double kp, double kd, double hold_kp, double hold_kd,
                  double duration_s, double rate_hz, const std::string& phase,
                  int commanded_index, std::vector<Sample>* samples) {
  const auto period = std::chrono::duration<double>(1.0 / rate_hz);
  auto next = std::chrono::steady_clock::now();
  const auto end_time = next + std::chrono::duration<double>(duration_s);
  while (std::chrono::steady_clock::now() < end_time) {
    LowCmd_ cmd = BuildCmd(target_q, mode_machine, num_motors, active_indices,
                           kp, kd, hold_kp, hold_kd);
    publisher.Write(cmd);

    const auto state = state_buffer.Get();
    if (state.has_value()) {
      Sample sample;
      sample.t = WallTimeSeconds();
      sample.phase = phase;
      sample.commanded_index = commanded_index;
      sample.target = target_q;
      ReadQDtau(*state, num_motors, &sample.q, &sample.dq, &sample.tau);
      sample.mode_pr = state->mode_pr();
      sample.mode_machine = state->mode_machine();
      samples->push_back(std::move(sample));
    }

    next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
    std::this_thread::sleep_until(next);
  }
}

std::optional<std::vector<double>> MeanTail(const std::vector<Sample>& samples,
                                           const std::string& phase, int index,
                                           double tail_s) {
  std::vector<const Sample*> phase_samples;
  for (const auto& sample : samples) {
    if (sample.phase == phase && sample.commanded_index == index) {
      phase_samples.push_back(&sample);
    }
  }
  if (phase_samples.empty()) {
    return std::nullopt;
  }
  const double last_t = phase_samples.back()->t;
  std::vector<const Sample*> tail;
  for (const Sample* sample : phase_samples) {
    if (sample->t >= last_t - tail_s) {
      tail.push_back(sample);
    }
  }
  if (tail.empty()) {
    tail = phase_samples;
  }
  std::vector<double> mean(tail.front()->q.size(), 0.0);
  for (const Sample* sample : tail) {
    for (size_t i = 0; i < mean.size(); ++i) {
      mean[i] += sample->q[i];
    }
  }
  for (double& v : mean) {
    v /= static_cast<double>(tail.size());
  }
  return mean;
}

void WriteLog(const std::string& path, const std::vector<Sample>& samples,
              const std::vector<int>& watch_indices) {
  std::filesystem::path out_path(path);
  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream f(path);
  f << "time_wall_s,phase,commanded_index,mode_pr,mode_machine";
  for (int idx : watch_indices) {
    f << ",target_" << idx << ",q_" << idx << ",dq_" << idx << ",tau_" << idx;
  }
  f << "\n";
  f << std::fixed << std::setprecision(9);
  for (const auto& sample : samples) {
    f << sample.t << "," << sample.phase << "," << sample.commanded_index
      << "," << sample.mode_pr << "," << sample.mode_machine;
    for (int idx : watch_indices) {
      f << "," << sample.target.at(idx) << "," << sample.q.at(idx) << ","
        << sample.dq.at(idx) << "," << sample.tau.at(idx);
    }
    f << "\n";
  }
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
  if (!args.read_only && !args.suspended_ack) {
    std::cerr << "Refusing to run command mode: robot must be suspended/supported.\n"
              << "Re-run with --i-understand-suspended after confirming it cannot fall or walk.\n";
    return 2;
  }
  if (args.num_motors <= 0 || args.num_motors > kHgMotorCount) {
    std::cerr << "--num-motors must be in [1, " << kHgMotorCount << "].\n";
    return 2;
  }
  if (args.include_waist) {
    args.indices.push_back(13);
    args.indices.push_back(14);
  }
  std::sort(args.indices.begin(), args.indices.end());
  args.indices.erase(std::unique(args.indices.begin(), args.indices.end()),
                     args.indices.end());
  for (const int idx : args.indices) {
    if (idx < 0 || idx >= args.num_motors) {
      std::cerr << "Test index " << idx << " is outside active motor range 0.."
                << args.num_motors - 1 << ".\n";
      return 2;
    }
  }
  std::set<int> watch_set{4, 5, 10, 11, 13, 14};
  watch_set.insert(args.indices.begin(), args.indices.end());
  std::vector<int> watch_indices;
  for (const int idx : watch_set) {
    if (idx >= 0 && idx < args.num_motors) {
      watch_indices.push_back(idx);
    }
  }
  if (args.log_path.empty()) {
    args.log_path = "/tmp/h2_pr_mapping_test/h2_pr_mapping_cpp_" +
                    TimestampForPath() + ".csv";
  }

  std::cout << "\n=== H2 PR Mapping Diagnostic (C++) ===\n";
  if (args.read_only) {
    std::cout << "Read-only mode: subscribing to rt/lowstate only. No LowCmd packets will be sent.\n";
  } else {
    std::cout << "Safety: robot must be suspended/supported. This tool sends real LowCmd packets.\n";
  }
  std::cout << "DDS: domain=" << args.domain_id
            << ", interface=" << args.network_interface << "\n";
  std::cout << "Mode: PR (" << kModePR << "); active motors: " << args.num_motors
            << "; step: " << std::showpos << std::fixed << std::setprecision(4)
            << args.delta << std::noshowpos << " rad\n";
  std::cout << "Watch slots:\n";
  for (int idx : watch_indices) {
    std::cout << "  " << std::setw(2) << idx << ": " << WatchName(idx) << "\n";
  }
  std::cout << "\n";

  try {
    ChannelFactory::Instance()->Init(args.domain_id, args.network_interface);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] DDS ChannelFactory init failed: " << e.what() << "\n";
    std::cerr << "        Use the interface name from ifconfig/ip, e.g. eno1, not the IP address.\n";
    return 3;
  }

  LowStateBuffer state_buffer;
  bool crc_warned = false;
  ChannelSubscriber<LowState_> lowstate_subscriber(kLowStateTopic);
  lowstate_subscriber.InitChannel(
      [&](const void* message) {
        LowState_ state = *static_cast<const LowState_*>(message);
        const uint32_t crc =
            Crc32Core(reinterpret_cast<uint32_t*>(&state), (sizeof(state) >> 2) - 1);
        if (state.crc() != crc && !crc_warned) {
          crc_warned = true;
          std::cerr << "[WARN] LowState CRC mismatch once; keeping samples for diagnostics.\n";
        }
        state_buffer.Set(state);
      },
      10);

  std::cout << "[INFO] Waiting for rt/lowstate...\n";
  const auto first_state = state_buffer.Wait(args.lowstate_timeout);
  if (!first_state.has_value()) {
    std::cerr << "[ERROR] Timed out after " << args.lowstate_timeout
              << "s waiting for rt/lowstate.\n";
    return 3;
  }

  const int mode_machine = first_state->mode_machine();
  std::vector<double> base_q;
  std::vector<double> base_dq;
  std::vector<double> base_tau;
  ReadQDtau(*first_state, args.num_motors, &base_q, &base_dq, &base_tau);

  std::cout << "[INFO] Received lowstate. mode_pr=" << int(first_state->mode_pr())
            << ", mode_machine=" << mode_machine << "\n";
  std::cout << "[INFO] Captured baseline q for tested/watch slots:\n";
  for (int idx : watch_indices) {
    std::cout << "  q[" << std::setw(2) << idx << "] = "
              << std::showpos << std::fixed << std::setprecision(6)
              << base_q.at(idx) << std::noshowpos << "  ("
              << WatchName(idx) << ")\n";
  }

  if (args.read_only) {
    std::cout << "\n[INFO] Streaming lowstate watch slots for 3 seconds:\n";
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < end) {
      const auto state = state_buffer.Get();
      if (state.has_value()) {
        std::vector<double> q;
        std::vector<double> dq;
        std::vector<double> tau;
        ReadQDtau(*state, args.num_motors, &q, &dq, &tau);
        std::cout << "  mode_pr=" << int(state->mode_pr())
                  << " mode_machine=" << int(state->mode_machine());
        for (int idx : watch_indices) {
          std::cout << " " << idx << ":" << std::showpos << std::fixed
                    << std::setprecision(4) << q.at(idx) << std::noshowpos;
        }
        std::cout << "\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
  }

  MaybeReleaseMotionMode(args.skip_release_mode);

  ChannelPublisher<LowCmd_> publisher(kLowCmdTopic);
  publisher.InitChannel();

  std::vector<Sample> samples;
  std::cout << "[INFO] Holding baseline for " << args.warmup_duration << "s...\n";
  PublishPhase(publisher, state_buffer, base_q, mode_machine, args.num_motors,
               std::set<int>(args.indices.begin(), args.indices.end()), args.kp,
               args.kd, args.hold_kp, args.hold_kd, args.warmup_duration,
               args.rate_hz, "baseline", -1, &samples);

  std::vector<double> step_values{args.delta};
  if (args.negative) {
    step_values.push_back(-args.delta);
  }

  struct Result {
    int idx = 0;
    double delta = 0.0;
    std::vector<double> response;
  };
  std::vector<Result> results;

  for (const int idx : args.indices) {
    for (const double delta : step_values) {
      std::cout << "\n[STEP] About to command slot " << idx << " by "
                << std::showpos << std::fixed << std::setprecision(4)
                << delta << std::noshowpos << " rad.\n"
                << "       " << WatchName(idx) << "\n"
                << "       Visually note whether motion is pitch (toe/front-back) or roll (side tilt).\n";
      if (!args.no_prompt) {
        std::cout << "       Press Enter to run this step, or Ctrl+C to stop...";
        std::cout.flush();
        std::string line;
        std::getline(std::cin, line);
      }

      const std::string delta_tag =
          (delta >= 0.0 ? "+" : "") + std::to_string(delta).substr(0, 6);
      const std::string pre_phase = "pre_" + std::to_string(idx) + "_" + delta_tag;
      PublishPhase(publisher, state_buffer, base_q, mode_machine, args.num_motors,
                   {idx}, args.kp, args.kd, args.hold_kp, args.hold_kd,
                   args.settle_duration, args.rate_hz, pre_phase, idx, &samples);
      const auto before = MeanTail(samples, pre_phase, idx, args.tail_s);

      std::vector<double> target = base_q;
      target.at(idx) += delta;
      const std::string step_phase = "step_" + std::to_string(idx) + "_" + delta_tag;
      PublishPhase(publisher, state_buffer, target, mode_machine, args.num_motors,
                   {idx}, args.kp, args.kd, args.hold_kp, args.hold_kd,
                   args.step_duration, args.rate_hz, step_phase, idx, &samples);
      const auto after = MeanTail(samples, step_phase, idx, args.tail_s);

      if (before.has_value() && after.has_value()) {
        std::vector<double> response(args.num_motors, 0.0);
        for (int j = 0; j < args.num_motors; ++j) {
          response[j] = after->at(j) - before->at(j);
        }
        results.push_back({idx, delta, response});
        std::vector<std::pair<double, int>> ranked;
        for (int j : watch_indices) {
          ranked.push_back({std::abs(response[j]), j});
        }
        std::sort(ranked.rbegin(), ranked.rend());
        std::cout << "[RESULT] LowState q response over tail window:\n";
        for (const auto& [_, j] : ranked) {
          std::cout << "  dq_meas[" << std::setw(2) << j << "] = "
                    << std::showpos << std::fixed << std::setprecision(6)
                    << response[j] << std::noshowpos << " rad  ("
                    << WatchName(j) << ")\n";
        }
      } else {
        std::cout << "[WARN] Not enough samples to summarize this step.\n";
      }

      std::cout << "[INFO] Returning slot " << idx << " to baseline for "
                << args.settle_duration << "s...\n";
      PublishPhase(publisher, state_buffer, base_q, mode_machine, args.num_motors,
                   {idx}, args.kp, args.kd, args.hold_kp, args.hold_kd,
                   args.settle_duration, args.rate_hz,
                   "post_" + std::to_string(idx) + "_" + delta_tag, idx,
                   &samples);
    }
  }

  std::cout << "\n[INFO] Holding captured baseline briefly before exit...\n";
  PublishPhase(publisher, state_buffer, base_q, mode_machine, args.num_motors,
               {}, args.kp, args.kd, args.hold_kp, args.hold_kd, 0.5,
               args.rate_hz, "final_hold", -1, &samples);

  WriteLog(args.log_path, samples, watch_indices);
  std::cout << "[INFO] Wrote CSV log: " << args.log_path << "\n";

  if (!results.empty()) {
    std::cout << "\n=== Summary ===\n";
    for (const auto& result : results) {
      std::vector<std::pair<double, int>> ranked;
      for (int j : watch_indices) {
        ranked.push_back({std::abs(result.response[j]), j});
      }
      std::sort(ranked.rbegin(), ranked.rend());
      const int best_idx = ranked.front().second;
      std::cout << "Command slot " << std::setw(2) << result.idx << " "
                << std::showpos << std::fixed << std::setprecision(4)
                << result.delta << std::noshowpos
                << " rad -> largest lowstate response slot " << std::setw(2)
                << best_idx << ": " << std::showpos << std::fixed
                << std::setprecision(6) << result.response[best_idx]
                << std::noshowpos << " rad\n";
    }
    std::cout << "\nInterpretation aid:\n"
              << "  If physical slot 4 is ankle pitch and slot 5 is ankle roll, swap H2 ankle DDS mapping in deploy.\n"
              << "  If physical slot 4 is ankle roll and slot 5 is ankle pitch, current H2 ankle order matches the code.\n"
              << "  Repeat the same logic for slots 10/11 on the right ankle.\n";
  }

  return 0;
}
