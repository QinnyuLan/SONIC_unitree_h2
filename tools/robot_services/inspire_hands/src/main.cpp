#include "inspire.h"
#include "systemd_notify.hpp"

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace po = boost::program_options;
using MotorCmds = unitree_go::msg::dds_::MotorCmds_;
using MotorStates = unitree_go::msg::dds_::MotorStates_;

namespace
{

constexpr size_t kHandDof = 6;
constexpr size_t kBothHandsDof = 12;
std::atomic<bool> stop_requested{false};

void HandleSignal(int)
{
  stop_requested.store(true);
}

struct Options
{
  std::string network{"eth0"};
  int domain{0};
  std::string topic_namespace{"inspire"};
  std::string right_port{"/dev/ttyUN0"};
  std::string left_port{"/dev/ttyUN1"};
  std::string hands{"both"};
  int right_id{1};
  int left_id{1};
  int serial_timeout_ms{20};
  int command_timeout_ms{500};
  int worker_period_ms{20};
  int offline_poll_ms{500};
  int usb_retry_ms{1000};
  int offline_failures{3};
  int state_period_ms{50};
  int state_stale_ms{250};
  int probe_attempts{3};
  bool probe_only{false};
  bool require_both{false};
};

Options ParseOptions(int argc, char** argv)
{
  Options options;
  po::options_description description("H2 native RS485 Inspire hand service");
  description.add_options()
      ("help,h", "show this help message")
      ("network", po::value<std::string>(&options.network)->default_value(options.network),
       "DDS network interface")
      ("domain", po::value<int>(&options.domain)->default_value(options.domain),
       "DDS domain ID (match the policy runtime)")
      ("namespace", po::value<std::string>(&options.topic_namespace)
          ->default_value(options.topic_namespace), "DDS topic namespace")
      ("right-port", po::value<std::string>(&options.right_port)
          ->default_value(options.right_port), "right-hand RS485 serial port")
      ("left-port", po::value<std::string>(&options.left_port)
          ->default_value(options.left_port), "left-hand RS485 serial port")
      ("hands", po::value<std::string>(&options.hands)->default_value(options.hands),
       "ports to probe/use: both, right, or left")
      ("right-id", po::value<int>(&options.right_id)->default_value(options.right_id),
       "right-hand device ID")
      ("left-id", po::value<int>(&options.left_id)->default_value(options.left_id),
       "left-hand device ID")
      ("serial-timeout-ms", po::value<int>(&options.serial_timeout_ms)
          ->default_value(options.serial_timeout_ms), "one serial response timeout")
      ("command-timeout-ms", po::value<int>(&options.command_timeout_ms)
          ->default_value(options.command_timeout_ms), "discard queued DDS commands older than this")
      ("worker-period-ms", po::value<int>(&options.worker_period_ms)
          ->default_value(options.worker_period_ms), "per-hand serial worker period")
      ("offline-poll-ms", po::value<int>(&options.offline_poll_ms)
          ->default_value(options.offline_poll_ms), "per-hand RS485 poll period after consecutive failures")
      ("usb-retry-ms", po::value<int>(&options.usb_retry_ms)
          ->default_value(options.usb_retry_ms), "per-hand serial reopen period while USB is absent")
      ("offline-failures", po::value<int>(&options.offline_failures)
          ->default_value(options.offline_failures), "consecutive read failures before offline polling")
      ("state-period-ms", po::value<int>(&options.state_period_ms)
          ->default_value(options.state_period_ms), "DDS state publication period")
      ("state-stale-ms", po::value<int>(&options.state_stale_ms)
          ->default_value(options.state_stale_ms), "mark DDS state lost after no successful read")
      ("probe-attempts", po::value<int>(&options.probe_attempts)
          ->default_value(options.probe_attempts), "read-only probe attempts per requested port")
      ("probe-only", po::bool_switch(&options.probe_only),
       "only query positions; do not initialize DDS or send position commands")
      ("require-both", po::bool_switch(&options.require_both),
       "in probe-only mode, require both requested hands to answer");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, description), variables);
  po::notify(variables);
  if (variables.count("help"))
  {
    std::cout << description << std::endl;
    std::exit(0);
  }

  if (options.hands != "both" && options.hands != "right" && options.hands != "left")
  {
    throw std::invalid_argument("--hands must be both, right, or left");
  }
  if (options.domain < 0 || options.domain > 232)
    throw std::invalid_argument("--domain must be in [0, 232]");
  if (options.topic_namespace.empty())
  {
    throw std::invalid_argument("--namespace must not be empty");
  }
  if (options.right_id < 1 || options.right_id > 254 ||
      options.left_id < 1 || options.left_id > 254)
  {
    throw std::invalid_argument("hand IDs must be in [1, 254]");
  }
  if (options.serial_timeout_ms <= 0 || options.command_timeout_ms <= 0 ||
      options.worker_period_ms <= 0 || options.offline_poll_ms <= 0 ||
      options.usb_retry_ms <= 0 || options.offline_failures <= 0 ||
      options.state_period_ms <= 0 || options.state_stale_ms <= 0 ||
      options.probe_attempts <= 0)
  {
    throw std::invalid_argument("timeout, period, and probe values must be positive");
  }
  if (options.hands == "both" && options.right_port == options.left_port)
  {
    throw std::invalid_argument("right and left hands must use different serial ports");
  }
  if (options.require_both && options.hands != "both")
  {
    throw std::invalid_argument("--require-both requires --hands both");
  }
  if (options.require_both && !options.probe_only)
  {
    throw std::invalid_argument("--require-both is only valid with --probe-only");
  }
  return options;
}

const char* SerialResultName(int16_t result)
{
  switch (result)
  {
    case 0: return "ok";
    case 1: return "write failed";
    case 2: return "response timeout or wrong length";
    case 3: return "invalid response header";
    case 4: return "response device ID mismatch";
    case 5: return "response checksum mismatch";
    default: return "serial exception";
  }
}

struct HandSnapshot
{
  std::array<float, kHandDof> q{};
  uint32_t lost{1};
  bool online{false};
  std::chrono::steady_clock::time_point last_success_at{};
};

class HandWorker
{
public:
  HandWorker(
      std::string side,
      std::string port,
      uint8_t device_id,
      int serial_timeout_ms,
      int command_timeout_ms,
      int worker_period_ms,
      int offline_poll_ms,
      int usb_retry_ms,
      int offline_failures)
      : side_(std::move(side)),
        port_(std::move(port)),
        device_id_(device_id),
        serial_timeout_ms_(serial_timeout_ms),
        command_timeout_(command_timeout_ms),
        worker_period_(worker_period_ms),
        offline_poll_period_(offline_poll_ms),
        usb_retry_period_(usb_retry_ms),
        offline_failure_threshold_(offline_failures)
  {
    state_.q.fill(std::numeric_limits<float>::quiet_NaN());
  }

  ~HandWorker()
  {
    Stop();
  }

  bool Probe(int attempts)
  {
    if (!TryOpenSerial(std::chrono::steady_clock::now()))
    {
      std::cerr << "[" << side_ << "] NO RESPONSE port=" << port_
                << " id=" << static_cast<int>(device_id_) << std::endl;
      return false;
    }

    for (int attempt = 1; attempt <= attempts; ++attempt)
    {
      Eigen::Matrix<double, kHandDof, 1> position;
      int16_t result = -1;
      try
      {
        result = hand_->GetPosition(position);
      }
      catch (const std::exception& error)
      {
        std::cerr << "[" << side_ << "] probe exception on " << port_
                  << ": " << error.what() << std::endl;
      }
      if (result == 0)
      {
        RecordReadSuccess(position);
        std::cout << "[" << side_ << "] ONLINE port=" << port_
                  << " id=" << static_cast<int>(device_id_)
                  << " q=" << position.transpose() << std::endl;
        return true;
      }
      RecordReadFailure(result);
      std::cerr << "[" << side_ << "] probe " << attempt << "/" << attempts
                << " failed on " << port_ << ": " << SerialResultName(result)
                << std::endl;
      if (!serial_->is_device_connected())
      {
        HandleUsbDisconnect("tty disappeared during probe");
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cerr << "[" << side_ << "] NO RESPONSE port=" << port_
              << " id=" << static_cast<int>(device_id_) << std::endl;
    return false;
  }

  void Start()
  {
    if (running_.exchange(true))
    {
      return;
    }
    last_progress_ms_.store(NowMilliseconds());
    thread_ = std::thread(&HandWorker::Loop, this);
  }

  void Stop()
  {
    running_.store(false);
    if (thread_.joinable())
    {
      thread_.join();
    }
  }

  void SubmitCommand(
      const std::array<double, kHandDof>& command,
      std::chrono::steady_clock::time_point received_at)
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (!accepting_commands_)
    {
      return;
    }
    pending_command_ = PendingCommand{command, received_at};
  }

  HandSnapshot Snapshot() const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
  }

  bool Progressing() const
  {
    // USB absence and protocol timeouts are expected. Only a worker which
    // stops advancing beyond its longest normal cycle suppresses the watchdog.
    const auto budget = std::max({worker_period_, offline_poll_period_, usb_retry_period_})
        + std::chrono::milliseconds(4LL * serial_timeout_ms_ + 2000);
    return running_.load() && NowMilliseconds() - last_progress_ms_.load() < budget.count();
  }

private:
  static int64_t NowMilliseconds()
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  struct PendingCommand
  {
    std::array<double, kHandDof> value{};
    std::chrono::steady_clock::time_point received_at{};
  };

  bool TryOpenSerial(std::chrono::steady_clock::time_point now)
  {
    try
    {
      auto serial = std::make_shared<SerialPort>(port_, B115200, serial_timeout_ms_);
      auto hand = std::make_unique<inspire::InspireHand>(serial, device_id_);
      serial_ = std::move(serial);
      hand_ = std::move(hand);
      usb_open_failures_ = 0;
      consecutive_read_failures_ = 0;
      offline_polling_ = false;
      DisableCommands();
      std::cout << "[" << side_ << "] USB serial opened: " << port_ << std::endl;
      return true;
    }
    catch (const std::exception& error)
    {
      hand_.reset();
      serial_.reset();
      ++usb_open_failures_;
      if (usb_open_failures_ == 1 || usb_open_failures_ % 60 == 0)
      {
        std::cerr << "[" << side_ << "] USB serial unavailable; retrying every "
                  << usb_retry_period_.count() << " ms: " << error.what() << std::endl;
      }
      next_usb_attempt_ = now + usb_retry_period_;
      MarkUnavailable();
      return false;
    }
  }

  void HandleUsbDisconnect(const char* reason)
  {
    if (!serial_)
    {
      return;
    }
    std::cerr << "[" << side_ << "] USB serial disconnected (" << reason
              << "); DDS remains active and this port will retry every "
              << usb_retry_period_.count() << " ms" << std::endl;
    DisableCommands();
    MarkUnavailable();
    hand_.reset();
    serial_.reset();
    consecutive_read_failures_ = 0;
    offline_polling_ = false;
    usb_open_failures_ = 0;
    next_usb_attempt_ = std::chrono::steady_clock::now() + usb_retry_period_;
  }

  void MarkUnavailable()
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (state_.lost == 0)
    {
      state_.lost = 1;
    }
    else if (state_.lost < std::numeric_limits<uint32_t>::max())
    {
      ++state_.lost;
    }
    state_.online = false;
  }

  void DisableCommands()
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    accepting_commands_ = false;
    pending_command_.reset();
  }

  void EnableFreshCommands()
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    pending_command_.reset();
    accepting_commands_ = true;
  }

  void RecordReadSuccess(const Eigen::Matrix<double, kHandDof, 1>& position)
  {
    bool recovered = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      recovered = !state_.online;
      for (size_t i = 0; i < kHandDof; ++i)
      {
        state_.q[i] = static_cast<float>(position(static_cast<int>(i)));
      }
      state_.lost = 0;
      state_.online = true;
      state_.last_success_at = std::chrono::steady_clock::now();
    }
    const bool leaving_backoff = offline_polling_;
    consecutive_read_failures_ = 0;
    offline_polling_ = false;
    if (recovered)
    {
      EnableFreshCommands();
      std::cout << "[" << side_ << "] ONLINE after RS485 response; accepting fresh commands"
                << (leaving_backoff ? " and resuming 50 Hz polling" : "")
                << std::endl;
    }
  }

  void RecordReadFailure(int16_t result)
  {
    ++consecutive_read_failures_;
    DisableCommands();
    uint32_t lost = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (state_.lost < std::numeric_limits<uint32_t>::max())
      {
        ++state_.lost;
      }
      state_.online = false;
      lost = state_.lost;
    }
    if (!offline_polling_ &&
        consecutive_read_failures_ >= static_cast<uint32_t>(offline_failure_threshold_))
    {
      offline_polling_ = true;
      std::cerr << "[" << side_ << "] RS485 offline after "
                << consecutive_read_failures_ << " consecutive read failures; polling at "
                << 1000.0 / static_cast<double>(offline_poll_period_.count()) << " Hz"
                << std::endl;
    }
    if (consecutive_read_failures_ == 1 || consecutive_read_failures_ % 120 == 0)
    {
      std::cerr << "[" << side_ << "] position read failed (lost=" << lost
                << "): " << SerialResultName(result) << std::endl;
    }
  }

  void ApplyLatestCommand()
  {
    std::optional<PendingCommand> command;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (!accepting_commands_ || !pending_command_)
      {
        return;
      }
      command = std::move(pending_command_);
      pending_command_.reset();
    }

    if (std::chrono::steady_clock::now() - command->received_at > command_timeout_)
    {
      std::cerr << "[" << side_ << "] discarded stale DDS command" << std::endl;
      return;
    }

    Eigen::Matrix<double, kHandDof, 1> target;
    for (size_t i = 0; i < kHandDof; ++i)
    {
      target(static_cast<int>(i)) = command->value[i];
    }
    int16_t result = -1;
    try
    {
      result = hand_->SetPosition(target);
    }
    catch (const std::exception& error)
    {
      std::cerr << "[" << side_ << "] position write exception: "
                << error.what() << std::endl;
      return;
    }
    if (result != 0)
    {
      std::cerr << "[" << side_ << "] position write failed: "
                << SerialResultName(result) << std::endl;
    }
  }

  void PollPosition()
  {
    Eigen::Matrix<double, kHandDof, 1> position;
    int16_t result = -1;
    try
    {
      result = hand_->GetPosition(position);
    }
    catch (const std::exception& error)
    {
      std::cerr << "[" << side_ << "] position read exception: "
                << error.what() << std::endl;
      RecordReadFailure(result);
      return;
    }
    if (result == 0)
    {
      RecordReadSuccess(position);
    }
    else
    {
      RecordReadFailure(result);
    }
  }

  void Loop()
  {
    while (running_.load())
    {
      last_progress_ms_.store(NowMilliseconds());
      const auto cycle_started = std::chrono::steady_clock::now();
      if (!serial_)
      {
        if (cycle_started < next_usb_attempt_)
        {
          std::this_thread::sleep_until(next_usb_attempt_);
          continue;
        }
        if (!TryOpenSerial(cycle_started))
        {
          std::this_thread::sleep_until(next_usb_attempt_);
          continue;
        }
      }

      if (!serial_->is_device_connected())
      {
        HandleUsbDisconnect("tty handle is gone or was re-enumerated");
        continue;
      }

      PollPosition();
      if (!serial_->is_device_connected())
      {
        HandleUsbDisconnect("tty disappeared while reading");
        continue;
      }
      ApplyLatestCommand();
      if (!serial_->is_device_connected())
      {
        HandleUsbDisconnect("tty disappeared while writing");
        continue;
      }
      const auto period = offline_polling_ ? offline_poll_period_ : worker_period_;
      std::this_thread::sleep_until(cycle_started + period);
    }
  }

  std::string side_;
  std::string port_;
  uint8_t device_id_;
  int serial_timeout_ms_;
  std::chrono::milliseconds command_timeout_;
  std::chrono::milliseconds worker_period_;
  std::chrono::milliseconds offline_poll_period_;
  std::chrono::milliseconds usb_retry_period_;
  int offline_failure_threshold_;
  SerialPort::SharedPtr serial_;
  std::unique_ptr<inspire::InspireHand> hand_;
  std::chrono::steady_clock::time_point next_usb_attempt_{};
  uint32_t consecutive_read_failures_{0};
  uint64_t usb_open_failures_{0};
  bool offline_polling_{false};

  std::atomic<bool> running_{false};
  std::atomic<int64_t> last_progress_ms_{0};
  std::thread thread_;

  mutable std::mutex state_mutex_;
  HandSnapshot state_;

  std::mutex command_mutex_;
  std::optional<PendingCommand> pending_command_;
  bool accepting_commands_{false};
};

struct DetectedHands
{
  std::unique_ptr<HandWorker> right;
  std::unique_ptr<HandWorker> left;
};

std::unique_ptr<HandWorker> OpenAndProbe(
    const std::string& side,
    const std::string& port,
    int device_id,
    const Options& options)
{
  try
  {
    auto worker = std::make_unique<HandWorker>(
        side,
        port,
        static_cast<uint8_t>(device_id),
        options.serial_timeout_ms,
        options.command_timeout_ms,
        options.worker_period_ms,
        options.offline_poll_ms,
        options.usb_retry_ms,
        options.offline_failures);
    worker->Probe(options.probe_attempts);
    return worker;
  }
  catch (const std::exception& error)
  {
    std::cerr << "[" << side << "] failed to open/probe " << port
              << ": " << error.what() << std::endl;
  }
  return nullptr;
}

DetectedHands DetectHands(const Options& options)
{
  DetectedHands detected;
  if (options.hands == "both" || options.hands == "right")
  {
    detected.right = OpenAndProbe(
        "right", options.right_port, options.right_id, options);
  }
  if (options.hands == "both" || options.hands == "left")
  {
    detected.left = OpenAndProbe(
        "left", options.left_port, options.left_id, options);
  }
  return detected;
}

class Native485Service
{
public:
  Native485Service(const Options& options, DetectedHands detected)
      : options_(options),
        right_(std::move(detected.right)),
        left_(std::move(detected.left))
  {
    const std::string base_topic = "rt/" + options_.topic_namespace;
    command_subscriber_ = std::make_shared<
        unitree::robot::ChannelSubscriber<MotorCmds>>(base_topic + "/cmd");
    state_publisher_ = std::make_shared<
        unitree::robot::ChannelPublisher<MotorStates>>(base_topic + "/state");
    state_publisher_->InitChannel();
    command_subscriber_->InitChannel(
        [this](const void* message) { OnCommand(message); }, 1);

    if (right_)
    {
      right_->Start();
    }
    if (left_)
    {
      left_->Start();
    }
    std::cout << "DDS ready: " << base_topic << "/cmd -> native RS485, "
              << base_topic << "/state <- native RS485" << std::endl;
  }

  ~Native485Service()
  {
    command_subscriber_->CloseChannel();
    if (right_)
    {
      right_->Stop();
    }
    if (left_)
    {
      left_->Stop();
    }
    state_publisher_->CloseChannel();
  }

  int Run()
  {
    notifier_.Ready();
    auto next_publish = std::chrono::steady_clock::now();
    while (!stop_requested.load())
    {
      PublishState();
      notifier_.Tick((!right_ || right_->Progressing()) && (!left_ || left_->Progressing()));
      next_publish += std::chrono::milliseconds(options_.state_period_ms);
      const auto after_publish = std::chrono::steady_clock::now();
      if (next_publish < after_publish)
      {
        next_publish = after_publish;
      }
      std::this_thread::sleep_until(next_publish);
    }
    notifier_.Stopping();
    return 0;
  }

private:
  static bool DecodeSide(
      const MotorCmds& message,
      size_t offset,
      std::array<double, kHandDof>& output)
  {
    for (size_t i = 0; i < kHandDof; ++i)
    {
      const double value = static_cast<double>(message.cmds()[offset + i].q());
      if (!std::isfinite(value) || value < 0.0 || value > 1.0)
      {
        return false;
      }
      output[i] = value;
    }
    return true;
  }

  void OnCommand(const void* raw_message)
  {
    const auto& message = *static_cast<const MotorCmds*>(raw_message);
    if (message.cmds().size() != kBothHandsDof)
    {
      std::cerr << "Rejected Inspire command with " << message.cmds().size()
                << " values; expected 12" << std::endl;
      return;
    }

    std::array<double, kHandDof> right_command{};
    std::array<double, kHandDof> left_command{};
    if (!DecodeSide(message, 0, right_command) ||
        !DecodeSide(message, kHandDof, left_command))
    {
      std::cerr << "Rejected Inspire command outside finite [0, 1] range" << std::endl;
      return;
    }

    const auto received_at = std::chrono::steady_clock::now();
    if (right_)
    {
      right_->SubmitCommand(right_command, received_at);
    }
    if (left_)
    {
      left_->SubmitCommand(left_command, received_at);
    }
  }

  static void FillState(
      MotorStates& message,
      size_t offset,
      const HandWorker* worker,
      std::chrono::milliseconds stale_after)
  {
    HandSnapshot snapshot;
    snapshot.q.fill(std::numeric_limits<float>::quiet_NaN());
    snapshot.lost = 1;
    if (worker != nullptr)
    {
      snapshot = worker->Snapshot();
      const auto now = std::chrono::steady_clock::now();
      if (snapshot.online &&
          (snapshot.last_success_at == std::chrono::steady_clock::time_point{} ||
           now - snapshot.last_success_at > stale_after))
      {
        snapshot.lost = std::max<uint32_t>(snapshot.lost, 1);
        snapshot.online = false;
      }
    }
    for (size_t i = 0; i < kHandDof; ++i)
    {
      message.states()[offset + i].q(snapshot.q[i]);
      message.states()[offset + i].lost(snapshot.lost);
    }
  }

  void PublishState()
  {
    MotorStates message;
    message.states().resize(kBothHandsDof);
    const auto stale_after = std::chrono::milliseconds(options_.state_stale_ms);
    FillState(message, 0, right_.get(), stale_after);
    FillState(message, kHandDof, left_.get(), stale_after);
    state_publisher_->Write(message);
  }

  Options options_;
  std::unique_ptr<HandWorker> right_;
  std::unique_ptr<HandWorker> left_;
  unitree::robot::ChannelSubscriberPtr<MotorCmds> command_subscriber_;
  unitree::robot::ChannelPublisherPtr<MotorStates> state_publisher_;
  SystemdNotify notifier_;
};

}  // namespace

int main(int argc, char** argv)
{
  try
  {
    const Options options = ParseOptions(argc, argv);
    std::cout << "H2 native RS485 mapping: right=" << options.right_port
              << " id=" << options.right_id << ", left=" << options.left_port
              << " id=" << options.left_id << std::endl;
    std::cout << "Read-only probing requested hand ports; no position command is sent"
              << std::endl;

    DetectedHands detected = DetectHands(options);
    const bool right_online = detected.right && detected.right->Snapshot().online;
    const bool left_online = detected.left && detected.left->Snapshot().online;
    const int online_count = static_cast<int>(right_online) + static_cast<int>(left_online);
    std::cout << "Probe summary: right=" << (right_online ? "ONLINE" : "OFFLINE")
              << ", left=" << (left_online ? "ONLINE" : "OFFLINE") << std::endl;

    if (options.probe_only)
    {
      if (online_count == 0)
      {
        std::cerr << "No requested Inspire hand answered" << std::endl;
        return 2;
      }
      if (options.require_both && (!right_online || !left_online))
      {
        std::cerr << "--require-both was set but both hands are not online" << std::endl;
        return 3;
      }
      return 0;
    }

    unitree::robot::ChannelFactory::Instance()->Init(options.domain, options.network);
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    Native485Service service(options, std::move(detected));
    const int run_result = service.Run();
    std::cout << "Service stopped; no shutdown hand pose was sent" << std::endl;
    return run_result;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Fatal: " << error.what() << std::endl;
    return 1;
  }
}
