// G1 locomotion generation is a separate process from the H2 policy runtime.
#include "transport.hpp"
#include "localmotion_kplanner_tensorrt.hpp"
#include <csignal>
#include <thread>
#include <unistd.h>
using namespace sonic_runtime;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Signal(int) { interrupted = 1; }
Json Sparse(const MotionSequence& motion, int cursor) {
  Json q = Json::array(), dq = Json::array(), root = Json::array();
  for (int f = 0; f < 10; ++f) {
    const int index = std::clamp(cursor + f * 5, 0, motion.timesteps - 1);
    q.push_back(std::vector<double>(motion.JointPositions(index), motion.JointPositions(index) + 29));
    dq.push_back(index == motion.timesteps - 1 ? std::vector<double>(29, 0) :
                 std::vector<double>(motion.JointVelocities(index), motion.JointVelocities(index) + 29));
    root.push_back(motion.BodyQuaternions(index)[0]);
  }
  Json data = {{"joint_pos", q}, {"joint_vel", dq}, {"root_quat", root}};
  ValidateG1(data);
  return data;
}
std::shared_ptr<MotionSequence> Merge(const MotionSequence& old, const MotionSequence& generated,
                                    int cursor, int generated_start) {
  auto result = std::make_shared<MotionSequence>();
  const int count = std::max(1, generated_start - cursor + generated.timesteps);
  Require(count <= 1500, "planner horizon exceeds capacity");
  result->ReserveCapacity(count, 29, 1, 1);
  result->timesteps = count;
  for (int f = 0; f < count; ++f) {
    const int previous = std::clamp(cursor + f, 0, old.timesteps - 1);
    const int next = std::clamp(cursor + f - generated_start, 0, generated.timesteps - 1);
    const double weight = std::clamp((f - std::max(0, generated_start - cursor)) / 8., 0., 1.);
    for (int j = 0; j < 29; ++j) {
      result->JointPositions(f)[j] = (1 - weight) * old.JointPositions(previous)[j] + weight * generated.JointPositions(next)[j];
      result->JointVelocities(f)[j] = (1 - weight) * old.JointVelocities(previous)[j] + weight * generated.JointVelocities(next)[j];
    }
    for (int j = 0; j < 3; ++j)
      result->BodyPositions(f)[0][j] = (1 - weight) * old.BodyPositions(previous)[0][j] + weight * generated.BodyPositions(next)[0][j];
    result->BodyQuaternions(f)[0] = quat_slerp_d(old.BodyQuaternions(previous)[0], generated.BodyQuaternions(next)[0], weight);
  }
  return result;
}
}
int main(int argc, char** argv) {
  std::signal(SIGINT, Signal); std::signal(SIGTERM, Signal);
  try {
    std::string model = "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx";
    std::string input = "tcp://127.0.0.1:5560", output = "tcp://127.0.0.1:5561";
    std::string runtime = "tcp://127.0.0.1:5558", feedback = "tcp://127.0.0.1:5559";
    bool publish_only = false;
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key == "--help") {
        std::cout << "G1 planner service: --model PATH --input-bind ENDPOINT --output-bind ENDPOINT\n"
                     "  --runtime ENDPOINT --state ENDPOINT --publish-only\n"
                     "Waits for planner commands on :5560; publishes sparse G1 on :5561.\n";
        return 0;
      }
      if (key == "--publish-only") { publish_only = true; continue; }
      Require(++i < argc, "missing value for " + key);
      if (key == "--model") model = argv[i];
      else if (key == "--input-bind") input = argv[i];
      else if (key == "--output-bind") output = argv[i];
      else if (key == "--runtime") runtime = argv[i];
      else if (key == "--state") feedback = argv[i];
      else throw std::invalid_argument("unknown option " + key);
    }
    PlannerConfig config; config.model_path = model; config.version = 2;
    LocalMotionPlannerTensorRT planner(false, 0, config);
    planner.planner_state_.enabled = true;
    zmq::context_t context(1);
    Receiver receiver(context, input, {"planner", "stand"});
    zmq::socket_t pub(context, zmq::socket_type::pub), push(context, zmq::socket_type::push), sub(context, zmq::socket_type::sub);
    for (auto* socket : {&pub, &push, &sub}) socket->set(zmq::sockopt::linger, 0);
    pub.set(zmq::sockopt::sndhwm, 2); pub.bind(output);
    push.set(zmq::sockopt::sndhwm, 2); push.set(zmq::sockopt::immediate, 1); push.connect(runtime);
    sub.set(zmq::sockopt::subscribe, ""); sub.set(zmq::sockopt::conflate, 1); sub.connect(feedback);
    Json state;
    std::shared_ptr<MotionSequence> motion;
    Clock::time_point timeline, last_plan;
    std::string active_session;
    const std::string session = std::to_string(WallMilliseconds()) + "-" + std::to_string(getpid());
    uint64_t seq = 0;
    auto send = [&](const std::string& kind, Json data) {
      Json message = {{"schema", "sonic.h2.command.v1"}, {"kind", kind}, {"data", data},
          {"source", "planner"}, {"session", session}, {"seq", seq++},
          {"sent_at_ms", WallMilliseconds()}, {"ttl_ms", 250}, {"takeover", false}};
      auto encoded = message.dump();
      pub.send(zmq::buffer(encoded), zmq::send_flags::dontwait);
      if (!publish_only) push.send(zmq::buffer(encoded), zmq::send_flags::dontwait);
    };
    std::cout << "Planner ready; waiting for runtime state and planner commands\n";
    while (!interrupted) {
      const auto tick = Clock::now();
      receiver.Poll();
      zmq::message_t message;
      if (sub.recv(message, zmq::recv_flags::dontwait)) {
        try {
          auto candidate = Json::parse(message.to_string());
          Require(candidate.at("schema") == "sonic.h2.state.v1", "invalid runtime state");
          Shape(candidate.at("joint_pos"), {31}, "state joints");
          Shape(candidate.at("root_quat"), {4}, "state quaternion");
          state = std::move(candidate);
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
      }
      const auto* command = receiver.inbox.Active(tick);
      const bool fresh = !state.empty() && WallMilliseconds() - state.at("sent_at_ms").get<int64_t>() < 500;
      if (!command || !fresh || state.value("phase", "") != "running") {
        if (motion) send("stand", Json::object());
        motion.reset(); active_session.clear();
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue;
      }
      if (!motion || active_session != command->session) {
        std::array<double, 29> physical{};
        for (int j = 0; j < 29; ++j) physical[j] = state["joint_pos"][h2lora_body_to_h2_dds[j]].get<double>();
        const auto joints = G1SemanticToMujocoOrder(H2PhysicalToG1Semantic(physical));
        Require(planner.Initialize(state["root_quat"].get<std::array<double, 4>>(), joints), "planner initialization failed");
        Require(planner.planner_motion_50hz_.timesteps > 0, "planner returned empty trajectory");
        motion = std::make_shared<MotionSequence>(planner.planner_motion_50hz_);
        timeline = Clock::now(); last_plan = timeline - std::chrono::seconds(1);
        active_session = command->session;
      }
      if (Clock::now() - last_plan >= std::chrono::milliseconds(200)) {
        const int cursor = std::max(0, static_cast<int>(std::chrono::duration<double>(Clock::now() - timeline).count() * 50));
        const auto data = command->data;
        Require(planner.UpdatePlanning(cursor, motion, data["mode"].get<int>(), data["speed"].get<float>(),
            data["height"].get<float>(), data["movement"].get<std::array<float, 3>>(),
            data["facing"].get<std::array<float, 3>>()), "planner inference failed");
        const int advanced = std::max(0, static_cast<int>(std::chrono::duration<double>(Clock::now() - timeline).count() * 50));
        motion = Merge(*motion, planner.planner_motion_50hz_, advanced, planner.gen_frame_);
        timeline += std::chrono::milliseconds(advanced * 20); last_plan = Clock::now();
      }
      // Do not renew a command that expired during a slow inference.
      if (receiver.inbox.Active(Clock::now())) {
        const int cursor = static_cast<int>(std::chrono::duration<double>(Clock::now() - timeline).count() * 50);
        send("g1", Sparse(*motion, cursor));
      }
      std::this_thread::sleep_until(tick + std::chrono::milliseconds(20));
    }
    send("stand", Json::object());
  } catch (const std::exception& e) { std::cerr << "Planner stopped: " << e.what() << '\n'; return 1; }
}
