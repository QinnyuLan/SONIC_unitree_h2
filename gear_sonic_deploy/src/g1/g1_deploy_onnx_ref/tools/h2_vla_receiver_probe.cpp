// Real ZMQManager/endpoint integration probe. NO DDS, policy, or motor writer.
#include "../include/input_interface/zmq_manager.hpp"
#include <fstream>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  setenv("GEAR_SONIC_H2_VLA", "1", 1);
  ZMQManager manager("127.0.0.1", std::stoi(argv[1]));
  MotionDataReader reader;
  auto reference = std::make_shared<MotionSequence>();
  reference->name = "h2_probe_standing_reference";
  reference->timesteps = 1;
  reference->ReserveCapacity(1);
  reference->SetEncodeMode(0);
  reader.motions.push_back(reference);
  std::shared_ptr<const MotionSequence> current = reference;
  int frame = 0;
  OperatorState operator_state;
  bool reinitialize = false, temperature = false;
  DataBuffer<HeadingState> heading;
  heading.SetData(HeadingState());
  DataBuffer<MovementState> movement;
  movement.SetData(MovementState());
  PlannerState planner;
  std::mutex mutex;
  std::ofstream status(argv[2]);
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::seconds(15)) {
    manager.update();
    manager.handle_input(reader, current, frame, operator_state, reinitialize,
                         heading, true, planner, movement, mutex, temperature);
    const auto [has_token, tokens] = manager.GetExternalTokenState();
    const auto [has_left, left] = manager.GetHandPose(true);
    status << "{\"has_token\":" << (has_token ? "true" : "false")
           << ",\"start\":" << (operator_state.start ? "true" : "false")
           << ",\"play\":" << (operator_state.play ? "true" : "false")
           << ",\"has_left\":" << (has_left ? "true" : "false")
           << ",\"left_1\":" << left[1]
           << ",\"reference\":\"" << current->name << "\"}"
           << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}
