/**
 * @file hand_client.cpp
 * @brief Read Inspire DDS state, or explicitly request a position test.
 */

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
#include <memory>
#include <mutex>
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
  double position{1.0};
  int duration_ms{2000};
  int publish_hz{10};
  bool cycle{false};
  bool monitor{true};
  double closed_position{0.7};
  int interval_ms{1500};
  int cycles{3};
};

Options ParseOptions(int argc, char** argv)
{
  Options options;
  po::options_description description("Inspire DDS client (read-only unless --position or --cycle is given)");
  description.add_options()
      ("help,h", "show this help message")
      ("network", po::value<std::string>(&options.network)->default_value(options.network),
       "DDS network interface")
      ("domain", po::value<int>(&options.domain)->default_value(options.domain), "DDS domain ID")
      ("namespace", po::value<std::string>(&options.topic_namespace)
          ->default_value(options.topic_namespace), "DDS topic namespace")
      ("position", po::value<double>(&options.position),
       "one-shot uniform target: 0=fully closed, 1=fully open")
      ("duration-ms", po::value<int>(&options.duration_ms)->default_value(options.duration_ms),
       "one-shot command publication duration")
      ("publish-hz", po::value<int>(&options.publish_hz)->default_value(options.publish_hz),
       "DDS command publication rate")
      ("cycle", po::bool_switch(&options.cycle),
       "explicitly enable alternating open/closed-position motion")
      ("closed-position", po::value<double>(&options.closed_position)
          ->default_value(options.closed_position), "cycle closed target; 0=fully closed")
      ("interval-ms", po::value<int>(&options.interval_ms)->default_value(options.interval_ms),
       "cycle hold time for each target")
      ("cycles", po::value<int>(&options.cycles)->default_value(options.cycles),
       "number of close/open cycles before exiting");

  po::variables_map variables;
  po::store(po::parse_command_line(argc, argv, description), variables);
  po::notify(variables);
  if (variables.count("help"))
  {
    std::cout << description << std::endl;
    std::exit(0);
  }

  if (options.topic_namespace.empty())
  {
    throw std::invalid_argument("--namespace must not be empty");
  }
  if (options.domain < 0 || options.domain > 232)
    throw std::invalid_argument("--domain must be in [0, 232]");
  options.monitor = !variables.count("position") && !options.cycle;
  if (!std::isfinite(options.position) || options.position < 0.0 || options.position > 1.0 ||
      !std::isfinite(options.closed_position) || options.closed_position < 0.0 ||
      options.closed_position > 1.0)
  {
    throw std::invalid_argument("--position and --closed-position must be in [0, 1]");
  }
  if (options.duration_ms <= 0 || options.publish_hz <= 0 ||
      options.interval_ms <= 0 || options.cycles <= 0)
  {
    throw std::invalid_argument("duration, rate, interval, and cycles must be positive");
  }
  return options;
}

class HandExample
{
public:
  explicit HandExample(const std::string& topic_namespace, bool monitor)
  {
    const std::string base_topic = "rt/" + topic_namespace;
    if (!monitor)
    {
      command_publisher_ = std::make_shared<
          unitree::robot::ChannelPublisher<MotorCmds>>(base_topic + "/cmd");
      command_publisher_->InitChannel();
    }
    state_subscriber_ = std::make_shared<
        unitree::robot::ChannelSubscriber<MotorStates>>(base_topic + "/state");
    state_subscriber_->InitChannel(
        [this](const void* message) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          state_ = *static_cast<const MotorStates*>(message);
          has_state_ = true;
        },
        1);
  }

  ~HandExample()
  {
    state_subscriber_->CloseChannel();
    if (command_publisher_) command_publisher_->CloseChannel();
  }

  void PublishUniform(double position)
  {
    MotorCmds command;
    command.cmds().resize(kBothHandsDof);
    for (auto& motor : command.cmds())
    {
      motor.q(static_cast<float>(position));
    }
    command_publisher_->Write(command);
  }

  void PrintState() const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!has_state_ || state_.states().size() != kBothHandsDof)
    {
      std::cout << "State: waiting for rt/inspire/state" << std::endl;
      return;
    }
    std::cout << "State right q=[";
    PrintSide(0);
    std::cout << "] lost=" << state_.states()[0].lost() << " left q=[";
    PrintSide(kHandDof);
    std::cout << "] lost=" << state_.states()[kHandDof].lost() << std::endl;
  }

private:
  void PrintSide(size_t offset) const
  {
    for (size_t i = 0; i < kHandDof; ++i)
    {
      if (i != 0)
      {
        std::cout << ", ";
      }
      std::cout << state_.states()[offset + i].q();
    }
  }

  unitree::robot::ChannelPublisherPtr<MotorCmds> command_publisher_;
  unitree::robot::ChannelSubscriberPtr<MotorStates> state_subscriber_;
  mutable std::mutex state_mutex_;
  MotorStates state_;
  bool has_state_{false};
};

void PublishFor(
    HandExample& example,
    double position,
    std::chrono::milliseconds duration,
    std::chrono::milliseconds publish_period)
{
  const auto deadline = std::chrono::steady_clock::now() + duration;
  int print_counter = 0;
  while (!stop_requested.load() && std::chrono::steady_clock::now() < deadline)
  {
    example.PublishUniform(position);
    if (print_counter++ % 5 == 0)
    {
      std::cout << "Command q=" << position << std::endl;
      example.PrintState();
    }
    std::this_thread::sleep_for(publish_period);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try
  {
    const Options options = ParseOptions(argc, argv);
    unitree::robot::ChannelFactory::Instance()->Init(options.domain, options.network);
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    HandExample example(options.topic_namespace, options.monitor);
    const auto publish_period = std::chrono::milliseconds(
        std::max(1, 1000 / options.publish_hz));
    std::cout << "DDS interface=" << options.network
              << " topic=rt/" << options.topic_namespace << "/cmd" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (options.monitor)
    {
      const auto deadline = std::chrono::steady_clock::now()
          + std::chrono::milliseconds(options.duration_ms);
      while (!stop_requested.load() && std::chrono::steady_clock::now() < deadline)
      {
        example.PrintState();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
      return 0;
    }

    if (!options.cycle)
    {
      std::cout << "One-shot position test. No automatic cycling is enabled." << std::endl;
      PublishFor(
          example,
          options.position,
          std::chrono::milliseconds(options.duration_ms),
          publish_period);
      return 0;
    }

    std::cout << "WARNING: explicit cycle mode enabled: open=1, closed-position="
              << options.closed_position << std::endl;
    for (int cycle = 0; cycle < options.cycles && !stop_requested.load(); ++cycle)
    {
      PublishFor(
          example,
          1.0,
          std::chrono::milliseconds(options.interval_ms),
          publish_period);
      PublishFor(
          example,
          options.closed_position,
          std::chrono::milliseconds(options.interval_ms),
          publish_period);
    }
    if (!stop_requested.load())
    {
      std::cout << "Cycle complete; commanding open before exit" << std::endl;
      PublishFor(example, 1.0, std::chrono::milliseconds(1000), publish_period);
    }
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Fatal: " << error.what() << std::endl;
    return 1;
  }
}
