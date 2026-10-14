/**
 * @file g1_deploy_onnx_ref.cpp
 * @brief Main application: deploys an RL locomotion policy on the Unitree G1 robot.
 *
 * This single-file application contains:
 *
 *   - **G1Deploy** class – the top-level controller that wires together all
 *     sub-systems (input, output, policy, encoder, planner, hands).
 *   - **main()** – CLI argument parser and run-loop.
 *
 * ## Threading Model
 *
 * Four real-time threads are created by the Unitree SDK's
 * `CreateRecurrentThreadEx`:
 *
 *   Thread           | Rate     | Method              | Responsibility
 *   -----------------|----------|---------------------|-------------------------------
 *   Input            | 100 Hz   | G1Deploy::Input     | Poll input interface, handle commands.
 *   Control          | 50 Hz    | G1Deploy::Control   | Gather obs, run policy, compute motor targets.
 *   Planner          | 10 Hz    | G1Deploy::Planner   | Re-plan locomotion trajectory.
 *   Command Writer   | 500 Hz   | G1Deploy::LowCommandWriter | Publish motor commands via DDS.
 *
 * ## Control-Loop State Machine (ProgramState)
 *
 *   INIT → WAIT_FOR_CONTROL → CONTROL
 *
 *   - **INIT**: Wait for a valid LowState message from the robot.
 *   - **WAIT_FOR_CONTROL**: Robot is ready; wait for operator "start" signal.
 *   - **CONTROL**: Active policy execution – gather observations, infer actions,
 *     write motor commands.  Exits on operator "stop" or error.
 *
 * ## CLI Arguments (selected)
 *
 *   Flag                  | Description
 *   ----------------------|------------
 *   --network             | DDS network interface (e.g. "eth0")
 *   --model               | Path to the policy ONNX model
 *   --motion-data         | Directory of pre-loaded reference motions
 *   --obs-config          | Observation config YAML
 *   --encoder-model       | Encoder ONNX model (for token_state)
 *   --planner-model       | Locomotion planner ONNX model
 *   --input-type          | keyboard / gamepad / zmq / ros2 / interface_manager / gamepad_manager / zmq_manager
 *   --output-type         | zmq / ros2 / all
 *   --disable-crc-check   | Skip CRC validation (for MuJoCo sim)
 *   --planner-fp16        | Use FP16 for planner TensorRT engine
 *   --policy-fp16         | Use FP16 for policy TensorRT engine
 */
#include <cmath>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <pthread.h>
#include <sched.h>
#include <array>
#include <vector>
#include <algorithm>
#include <chrono>
#include <utility>
#include <unistd.h>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <numeric>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <iterator>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

// TRTInference
#include <TRTInference/InferenceEngine.h>

// ONNX
#include <onnxruntime_cxx_api.h>

// Motion Data Reader
#include "../include/motion_data_reader.hpp"

// Math utilities
#include "../include/math_utils.hpp"

// Observation configuration
#include "../include/observation_config.hpp"
#include "../include/h2_lora_observation.hpp"
#include "../include/h2_native_contract.hpp"
#include "../include/h2_head_control.hpp"

// New Planner Classes
#include "../include/localmotion_kplanner.hpp"
#include "../include/localmotion_kplanner_tensorrt.hpp"

// Utility classes
#include "../include/utils.hpp"

// Robot parameters
#include "../include/robot_parameters.hpp"
#include "../include/policy_parameters.hpp"

// Input interface and input handlers
#include "../include/input_interface/keyboard_handler.hpp"
#include "../include/input_interface/gamepad.hpp"
#include "../include/input_interface/zmq_endpoint_interface.hpp"
#include "../include/input_interface/interface_manager.hpp"
#include "../include/input_interface/gamepad_manager.hpp"
#include "../include/input_interface/zmq_manager.hpp"

// Output interface and output handlers
#include "../include/output_interface/output_interface.hpp"

// Optional ROS2 input handler
#if HAS_ROS2
#include "../include/input_interface/ros2_input_handler.hpp"
#include "../include/output_interface/ros2_output_handler.hpp"
#endif

#include "../include/output_interface/zmq_output_handler.hpp"

#include <cuda_runtime.h>
#include "../include/state_logger.hpp"

// Encoder
#include "../include/encoder.hpp"

// Control policy
#include "../include/control_policy.hpp"

// Selectable DDS hand backends
#include "../include/dex3_hands.hpp"
#include "../include/hand_backend.hpp"
#include "../include/inspire_hands.hpp"

// Error monitor
#include "../include/error_monitor.hpp"

#include "audio_thread/audio_thread.hpp"

// DDS
using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;



/**
 * @class G1Deploy
 * @brief Top-level controller that orchestrates the entire G1 deployment pipeline.
 *
 * Owns and wires together:
 *  - InputInterface (keyboard / gamepad / ZMQ / ROS2 / managers)
 *  - PolicyEngine (TensorRT control policy)
 *  - EncoderEngine (optional TensorRT observation encoder)
 *  - LocalMotionPlannerBase (optional TensorRT locomotion planner)
 *  - Dex3 or Inspire DDS hand backend (optional)
 *  - StateLogger (ring buffer + CSV persistence)
 *  - OutputInterface(s) (ZMQ / ROS2 state publishers)
 *  - MotionDataReader (pre-loaded reference motions)
 *
 * Four real-time threads handle Input (100 Hz), Control (50 Hz),
 * Planner (10 Hz), and Command Writing (500 Hz).
 */
class G1Deploy {
  private:
    /// State machine for the control loop lifecycle.
    enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL };
    
    // =========================================================================
    // Core timing, mode, and counters
    // =========================================================================
    double time_;          ///< Elapsed time since start (used during INIT ramp-up).
    double publish_dt_;    ///< Command writer period (500 Hz = 0.002 s).
    double control_dt_;    ///< Control loop period  (50 Hz = 0.02 s).
    double planner_dt_;    ///< Planner loop period  (10 Hz = 0.1 s).
    double input_dt_;      ///< Input poll period    (100 Hz = 0.01 s).
    double duration_;      ///< Duration of the INIT ramp-up to default pose (3 s).
    int robot_future_step_ = 1; ///< Reference-frame stride for G1/teleop encoder inputs.
    int smpl_future_step_ = 1;  ///< Reference-frame stride for SMPL encoder inputs.
    double reference_future_step_fps_ = -1.0; ///< FPS used to configure future-frame strides.
    int counter_;          ///< General-purpose tick counter.
    Mode mode_pr_;         ///< Ankle control mode (series PR vs. parallel AB).
    uint8_t mode_machine_; ///< Robot variant code received from LowState.
    
    // =========================================================================
    // Input interface and buffered input data
    // =========================================================================
    std::unique_ptr<InputInterface> input_interface_;
    
    // Buffered input data (the real data to the policy engine)
    // has_*_data_ flags indicate whether the getter returned valid buffered data (true) or defaults (false)
    bool has_vr_3point_data_ = false;
    std::array<double, 9> vr_3point_position_buffer_;
    std::array<double, 12> vr_3point_orientation_buffer_;
    std::array<double, 3> vr_3point_compliance_buffer_;
    TimestampedData<HeadPoseCommand> head_pose_command_buffer_;
    bool has_vr_5point_data_ = false;
    std::array<double, 15> vr_5point_position_buffer_;
    std::array<double, 20> vr_5point_orientation_buffer_;
    bool has_left_hand_data_ = false;
    bool has_right_hand_data_ = false;
    std::array<double, 7> left_hand_joint_buffer_;
    std::array<double, 7> right_hand_joint_buffer_;
    bool has_upper_body_data_ = false;
    std::array<double, 17> upper_body_joint_positions_buffer_;
    std::array<double, 17> upper_body_joint_velocities_buffer_;
    std::vector<double> token_state_data_;  // Token buffer (size from config)
    
    // =========================================================================
    // Motion data, current motion, and recording
    // =========================================================================
    // Motion data reader and current motion
    MotionDataReader motion_reader_;
    
    // Current motion and frame (using shared_ptr for thread safety)
    std::shared_ptr<const MotionSequence> current_motion_ = nullptr;
    int current_frame_ = 0;
    int saved_frame_for_observation_window_ = 0; // for observation window
    std::mutex current_motion_mutex_; // for current motion and frame synchronization
    
    // =========================================================================
    // Local motion planner and movement state
    // =========================================================================
    // New unified planner interface
    std::string planner_path;
    std::unique_ptr<LocalMotionPlannerBase> planner_;
    std::shared_ptr<MotionSequence> planner_motion_;
    
    // Movement momentum system
    MovementState last_movement_state_ = MovementState(static_cast<int>(LocomotionMode::IDLE), {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, -1.0f, -1.0f);
    float replan_interval_running_ = 0.1;
    float replan_interval_crawling_ = 0.2;
    float replan_interval_boxing_ = 1.0;
    float replan_interval_ = 1.0;
    float replan_interval_counter_ = 0.0f;

    // Idle-mode error-based readaptation with double-threshold state machine.
    // States: IDLE (do nothing), ADAPTING (toward robot state), RECOVERING (toward planner target).
    // Each state has a trigger threshold (enter) and stop threshold (exit).
    enum class IdleReadaptState { IDLE, ADAPTING, RECOVERING };
    std::array<double, 29> idle_readapt_original_targets_{};
    bool idle_readapt_stored_ = false;
    IdleReadaptState idle_readapt_state_ = IdleReadaptState::IDLE;
    static constexpr double kAdaptTrigger  = 0.10;   // rad: start adapting
    static constexpr double kAdaptStop     = 0.05;   // rad: stop adapting → IDLE
    static constexpr double kRecoverTrigger = 0.045;  // rad: start recovering

    
    // =========================================================================
    // Low-level robot I/O buffers, channels, and threads
    // =========================================================================
    // Flag to disable CRC checking for MuJoCo simulation
    bool disable_crc_check_ = false;
    std::string robot_profile_ = "g1";
    H2GainScales h2_gain_scales_;
    bool use_h2_lora_ = false;
    bool use_h2_native_ = false;
    bool use_h2_robot_ = false;
    bool encoder_universal_full_ = false;
    int planner_encoder_mode_ = 0;
    bool project_h2_native_planner_reference_ = false;
    bool enable_h2_vr_head_control_ = false;
    HandBackendKind hand_backend_ = HandBackendKind::DEX3;
    bool project_h2_lora_planner_reference_ = true;
    int robot_motor_count_ = G1_NUM_MOTOR;
    double policy_action_scale_ = 1.0;
    double policy_action_scale_default_ = 1.0;
    static constexpr double kH2LoraMujocoSimPolicyActionScale = 1.0;
    bool h2_lora_mujoco_sim_ = false;
    bool auto_play_triggered_ = false;
    bool auto_next_triggered_ = false;
    std::size_t auto_next_trigger_count_ = 0;
    bool reinitialize_heading_ = true;
    bool report_temperature_ = false;
    std::string pending_tts_;  // One-shot TTS message, consumed by GatherInputInterfaceData()
    std::array<double, 2> h2_head_targets_ = {0.0, 0.0};
    bool h2_head_command_was_fresh_ = false;
    static constexpr std::chrono::milliseconds H2_HEAD_COMMAND_TIMEOUT{250};

    // Per-motor high temperature hysteresis (enter at >= 90, exit at < 85)
    std::array<bool, MAX_NUM_MOTOR> motor_high_temp_ = {};
    static constexpr int16_t HIGH_TEMP_ENTER = 90;
    static constexpr int16_t HIGH_TEMP_EXIT = 85;
    bool high_temp_warning_ = false;
    std::string high_temp_message_;

    std::array<double, 4> init_ref_data_root_rot_array_;

    DataBuffer<LowState_> low_state_buffer_;
    DataBuffer<MotorCommand> motor_command_buffer_;
    DataBuffer<IMUState_> imu_torso_buffer_;
    DataBuffer<HeadingState> heading_state_buffer_;
    DataBuffer<MovementState> movement_state_buffer_;
    
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
    ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;
    ThreadPtr input_thread_ptr_, command_writer_ptr_, control_thread_ptr_, planner_thread_ptr_;
    
    // =========================================================================
    // External clients and peripheral managers
    // =========================================================================
    std::unique_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;
    
    // Exactly one hand backend is active at a time.
    Dex3Hands dex3_hands_;
    InspireHands inspire_hands_;

    // Motor error monitor (tracks fault state transitions)
    ErrorMonitor error_monitor_;

    static constexpr std::chrono::milliseconds STREAMING_DATA_ABSENT_THRESHOLD{150};
    CounterDebouncer streaming_data_absent_debouncer_{100, 500, 50, 1};
    RollingStats<1000> streaming_data_delay_rolling_stats_;
    std::unique_ptr<AudioThread> audio_thread_;
    
    // =========================================================================
    // Program state and last commanded actions
    // =========================================================================
    static constexpr std::chrono::milliseconds LOW_STATE_LATE_THRESHOLD{50};
    static constexpr std::chrono::milliseconds LOW_STATE_ABSENT_THRESHOLD{500};
    static constexpr std::chrono::milliseconds COMMAND_LATENCY_LOG_INTERVAL{1000};
    ProgramState program_state_;
    std::chrono::steady_clock::time_point wall_start_time_;
    std::chrono::steady_clock::time_point last_command_latency_log_time_{};
    bool init_pose_snapshot_valid_ = false;
    std::array<double, MAX_NUM_MOTOR> init_pose_snapshot_ = {};
    std::array<double, POLICY_ACTION_DIM> last_action;
    std::array<double, 7> last_left_hand_action;
    std::array<double, 7> last_right_hand_action;
    
    // =========================================================================
    // Logging / recording streams
    // =========================================================================
    std::unique_ptr<std::ofstream> target_motion_file_;
    std::unique_ptr<std::ofstream> planner_motion_file_;
    std::unique_ptr<std::ofstream> policy_input_file_;
    std::unique_ptr<std::ofstream> encoder_input_file_;
    std::unique_ptr<std::ofstream> lowcmd_command_file_;
    std::unique_ptr<std::ofstream> record_input_file_;
    std::unique_ptr<std::ifstream> playback_input_file_;

    // Motion recorders for streamed and planner_motion
    MotionRecorder zmq_motion_recorder_;
    MotionRecorder planner_motion_recorder_;
    bool enable_motion_recording_ = false;
    
    // =========================================================================
    // Initial compliance values for VR 3-point control (set from command line)
    // These are keyboard-controlled and are the ONLY source of compliance values
    // =========================================================================
    std::array<double, 3> initial_vr_3point_compliance_ = {0.5, 0.5, 0.0};
    
    // Initial max close ratio for the selected hand backend (set from command line)
    // Default 1.0 allows full closure, use --max-close-ratio to limit
    // Keyboard controls (J/K) always available for runtime adjustment
    double initial_max_close_ratio_ = 1.0;
    
    // Track if vr_3point_compliance is observed by the policy
    // If false, adjusting compliance via keyboard has no effect on the policy
    bool has_vr_3point_compliance_obs_ = false;
    
    // Independent logging counter (not affected by planner buffer cleaning)
    int logging_counter_ = 0;
    TimestampedData<LowState_> used_low_state_data_;
    TimestampedData<IMUState_> used_imu_torso_data_;

    // State logger
    std::unique_ptr<StateLogger> state_logger_;
    
    // Output interfaces (supports multiple simultaneous outputs)
    std::vector<std::unique_ptr<OutputInterface>> output_interfaces_;

    // =========================================================================
    // Core ML components (encoder + policy) and interfaces
    // =========================================================================
    // Encoder engine
    std::string model_path;
    std::unique_ptr<EncoderEngine> encoder_engine_;
    EncoderConfig encoder_config_;  // Encoder configuration from observation config
    bool is_using_encoder_ = false;
    int initial_encoder_mode_ = -2;  // -2: no token state. -1: need token state but no encoder. 0,1,2,...: encoder mode.
    int last_logged_encoder_mode_ = -999;  // Track last logged mode to avoid spam
    
    // Token input safety tracking
    bool first_token_received_ = false;  // True once we've received at least one token
    std::optional<std::chrono::steady_clock::time_point> last_token_time_;  // Timestamp of last token received
    static constexpr std::chrono::milliseconds TOKEN_TIMEOUT_MS{200};  // Max time between tokens before warning (200ms = 5Hz min)
    
    // Control policy
    std::unique_ptr<PolicyEngine> policy_engine_;
    
    // =========================================================================
    // Observation system configuration and runtime state
    // =========================================================================
    // Function pointer type for observation functions (returns bool for success/failure)
    // Functions read from internal state (state_logger_, current_motion_, etc.)
    // and write to the provided target buffer at the given offset
    using ObservationFunction = std::function<bool(std::vector<double>&, size_t)>;
    
    // Observation registry - single place to define all observations
    struct ObservationRegistry {
      std::string name;
      size_t dimension;
      ObservationFunction function;
      
      ObservationRegistry(const std::string& n, size_t d, ObservationFunction f) 
        : name(n), dimension(d), function(f) {}
    };
 
    // Structure for active observation function with metadata
    struct ActiveObservation {
      std::string name;
      ObservationFunction function;
      size_t offset;
      size_t dimension;
      
      ActiveObservation(const std::string& n, ObservationFunction f, size_t off, size_t dim) 
        : name(n), function(f), offset(off), dimension(dim) {}
    };
    
    // Pre-allocated observation buffers to avoid allocation in control loop
    std::vector<double> obs_buffer_;          // Policy observation buffer
    std::vector<double> encoder_obs_buffer_;  // Encoder observation buffer
    
    // Observation configuration
    std::vector<ObservationConfig> obs_config_; 
    
    // Active observation functions (validated and ready for runtime)
    std::vector<ActiveObservation> active_obs_functions_;
    
    // Active encoder observation functions (for encoder input)
    std::vector<ActiveObservation> active_encoder_obs_functions_;
    
    // VR3Point index
    std::array<int, 3> actual_vr_3point_index = {-1, -1, -1};

    // VR5Point index
    std::array<int, 5> actual_vr_5point_index = {-1, -1, -1, -1, -1};

    // =========================================================================
    // Motion-based observation gatherers
    // 
    // These functions read from the currently-active MotionSequence to build
    // observation vectors for the policy neural network.  They support
    // multi-frame look-ahead (gathering future frames at a configurable
    // step interval) for temporal context.
    //
    // Each gatherer writes its output into `target_buffer` at `offset` and
    // updates `saved_frame_for_observation_window_` so that streamed motions
    // know how many look-ahead frames are needed before the end.
    // =========================================================================

    std::array<double, POLICY_ACTION_DIM> LowStateToG1PlannerSemantic(
        const LowState_& low_state) const {
      const auto& motor_state = low_state.motor_state();
      if (use_h2_robot_) {
        std::array<double, POLICY_ACTION_DIM> h2_physical_positions = {};
        for (int semantic_index = 0; semantic_index < POLICY_ACTION_DIM; ++semantic_index) {
          const int h2_dds_index = h2lora_body_to_h2_dds.at(semantic_index);
          h2_physical_positions.at(semantic_index) = motor_state.at(h2_dds_index).q();
        }
        return H2PhysicalToG1Semantic(h2_physical_positions);
      }

      std::array<double, POLICY_ACTION_DIM> mujoco_positions = {};
      for (int mujoco_index = 0; mujoco_index < POLICY_ACTION_DIM; ++mujoco_index) {
        mujoco_positions.at(mujoco_index) = motor_state.at(mujoco_index).q();
      }
      return G1MujocoToSemanticOrder(mujoco_positions);
    }

    std::array<double, POLICY_ACTION_DIM> PrepareReferenceJointsForPolicy(
        const double* semantic_values) const {
      std::array<double, POLICY_ACTION_DIM> values = {};
      std::copy_n(semantic_values, POLICY_ACTION_DIM, values.begin());
      if (use_h2_lora_ && project_h2_lora_planner_reference_ && current_motion_ &&
          current_motion_.get() == planner_motion_.get()) {
        return G1PlannerReferenceToH2LoraSemantic(values);
      }
      return values;
    }

    /// Gather root Z position from N future frames (1 value per frame).
    bool GatherMotionRootZPositionMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid body data
      if (current_motion_->GetNumBodies() == 0) {
        std::cerr << "✗ Error: Motion has no bodies - cannot gather root z position" << std::endl;
        return false;
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather root z position from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_root_z_pos = current_motion_->BodyPositions(target_frame)[0][2];
        target_buffer[offset + frame_idx * 1] = motion_root_z_pos;
      }
      return true;
    }

    /// Gather body positions from N future frames (3 values per body per frame).
    /// If body_part_indexes is empty, gathers all bodies; otherwise only the specified subset.
    bool GatherMotionBodyPositionsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> body_part_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid body data
      if (current_motion_->GetNumBodies() == 0) {
        std::cerr << "✗ Error: Motion has no bodies - cannot gather body positions" << std::endl;
        return false;
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather positions from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_body_pos = current_motion_->BodyPositions(target_frame);

        // If body part indexes are empty, gather all bodies (all x, y, z coordinates)
        if (body_part_indexes.empty()) {
          size_t num_bodies = current_motion_->GetNumBodies();
          size_t frame_offset = offset + frame_idx * num_bodies * 3;  // 3 coordinates per body
          
          std::vector<double> all_body_pos(num_bodies * 3);
          for (size_t i = 0; i < num_bodies; i++) {
            all_body_pos[i * 3 + 0] = motion_body_pos[i][0];  // x
            all_body_pos[i * 3 + 1] = motion_body_pos[i][1];  // y
            all_body_pos[i * 3 + 2] = motion_body_pos[i][2];  // z
          }
          std::copy(
            all_body_pos.begin(),
            all_body_pos.end(),
            target_buffer.begin() + frame_offset
          );
        // If body part indexes are not empty, gather only the needed bodies 
        } else {
          // Get motion's body part indexes to map desired body index to storage index
          const auto& motion_body_indexes = current_motion_->BodyPartIndexes();
          
          size_t frame_offset = offset + frame_idx * body_part_indexes.size() * 3;  // 3 coordinates per body
          std::vector<double> needed_body_pos(body_part_indexes.size() * 3);
          
          for (size_t i = 0; i < body_part_indexes.size(); i++) {
            int desired_body_index = body_part_indexes[i];
            
            // Find which storage index corresponds to this body index
            int storage_index = -1;
            for (size_t j = 0; j < motion_body_indexes.size(); j++) {
              if (motion_body_indexes[j] == desired_body_index) {
                storage_index = static_cast<int>(j);
                break;
              }
            }
            
            // Extract all x, y, z coordinates
            if (storage_index < 0) {
              std::cerr << "✗ Error: Body index " << desired_body_index 
                        << " not found in motion data. Available indices: [";
              for (size_t j = 0; j < motion_body_indexes.size(); j++) {
                std::cerr << motion_body_indexes[j];
                if (j + 1 < motion_body_indexes.size()) std::cerr << ", ";
              }
              std::cerr << "]" << std::endl;
              return false;
            }
            
            needed_body_pos[i * 3 + 0] = motion_body_pos[storage_index][0];  // x
            needed_body_pos[i * 3 + 1] = motion_body_pos[storage_index][1];  // y
            needed_body_pos[i * 3 + 2] = motion_body_pos[storage_index][2];  // z
          }
          std::copy(
            needed_body_pos.begin(),
            needed_body_pos.end(),
            target_buffer.begin() + frame_offset
          );
        }
      }
      return true;
    }

    /**
     * @brief Gather the heading-corrected anchor orientation from N future frames.
     *
     * Computes the relative rotation from the robot's current base orientation
     * to the heading-corrected reference orientation at each future frame.
     * Outputs the first two columns of the 3×3 rotation matrix (6 values per frame).
     *
     * Also handles heading reinitialisation when `reinitialize_heading_` is set.
     */
    /**
     * @brief Update heading state — must be called once per control tick BEFORE gathering observations.
     *
     * Handles heading reinitialisation (when operator triggers reset) and
     * ensures init_ref_data_root_rot_array_ is set when starting from frame 0.
     * This shared state is consumed by all orientation observation functions.
     */
    void UpdateHeadingState() {
      if (!current_motion_ || current_motion_->timesteps == 0) { return; }
      if (current_motion_->GetNumBodyQuaternions() == 0) { return; }
      if (!state_logger_) { return; }

      if (reinitialize_heading_) {
        double sample_dt = control_dt_;
        auto hist = state_logger_->GetLatest(1, sample_dt);
        const auto& base_quat = hist[0].base_quat;

        heading_state_buffer_.SetData(HeadingState(base_quat, 0.0));
        reinitialize_heading_ = false;
        const auto motion_body_quat_init = current_motion_->BodyQuaternions(current_frame_);
        init_ref_data_root_rot_array_ = motion_body_quat_init[0];
        std::cout << "Reset heading state to " << base_quat[0] << ", " << base_quat[1] << ", " << base_quat[2] << ", " << base_quat[3] << std::endl;
        std::cout << "Reset delta heading to 0" << std::endl;
        std::cout << "Reset init reference data root rotation to current frame: " << init_ref_data_root_rot_array_[0] << ", " << init_ref_data_root_rot_array_[1] << ", " << init_ref_data_root_rot_array_[2] << ", " << init_ref_data_root_rot_array_[3] << std::endl;
        std::cout << "**********************************************************" << std::endl;
        std::cout << "Reference motion name: " << current_motion_->name << std::endl;
        std::cout << "**********************************************************" << std::endl;
      }

      if (current_frame_ == 0) {
        const auto motion_body_quat_init = current_motion_->BodyQuaternions(0);
        init_ref_data_root_rot_array_ = motion_body_quat_init[0];
      }
    }

    /**
     * @brief Compute the apply_delta_heading quaternion from current heading state.
     *
     * This is a pure computation with no side effects — safe to call from any
     * observation function. Reads heading_state_buffer_ and init_ref_data_root_rot_array_.
     */
    std::array<double, 4> ComputeApplyDeltaHeading() {
      auto heading_state_data = heading_state_buffer_.GetDataWithTime().data;
      HeadingState heading_state = heading_state_data ? *heading_state_data : HeadingState();

      auto init_heading = calc_heading_quat_d(heading_state.init_base_quat);
      auto data_heading_inv = calc_heading_quat_inv_d(init_ref_data_root_rot_array_);
      auto apply_delta_heading = quat_mul_d(init_heading, data_heading_inv);

      if (heading_state.delta_heading != 0.0) {
        auto delta_quat = euler_z_to_quat_d(heading_state.delta_heading);
        apply_delta_heading = quat_mul_d(delta_quat, apply_delta_heading);
      }
      return apply_delta_heading;
    }

    int ComputeFutureReferenceFrame(int frame_idx, int step_size) const {
      if (!current_motion_ || current_motion_->timesteps <= 0) {
        return 0;
      }
      int target_frame = static_cast<int>(current_frame_);
      if (operator_state.play) {
        target_frame += frame_idx * step_size;
        if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
          target_frame = static_cast<int>(current_motion_->timesteps) - 1;
        }
      }
      return std::clamp(target_frame, 0, static_cast<int>(current_motion_->timesteps) - 1);
    }

    /**
     * @brief Gather anchor orientation from N future frames with configurable orientation mode.
     *
     * @param orientation_mode Controls which quaternion is used as the "left" side of the diff:
     *   - 0: full base quaternion (existing motion_anchor_ori_b)
     *   - 1: robot heading only (heading setting: motion_anchor_ori_heading)
     *   - 2: reference first-frame heading (refheading setting: motion_anchor_ori_refheading)
     */
    bool GatherMotionAnchorOrientationMutiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, int orientation_mode = 0) {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }

      if (current_motion_->GetNumBodyQuaternions() == 0) {
        std::cout << "✗ Error: No motion_body_quat data available. Stopping control system." << std::endl;
        return false;
      }
      if (!state_logger_) { return false; }

      double sample_dt = control_dt_;
      auto hist = state_logger_->GetLatest(1, sample_dt);
      const auto& base_quat = hist[0].base_quat;

      auto apply_delta_heading = ComputeApplyDeltaHeading();

      // For refheading mode (2): pre-compute the first future frame's ref heading
      // This is used as the "left" quaternion for ALL frames in the loop
      std::array<double, 4> ref_first_heading = {1.0, 0.0, 0.0, 0.0};
      if (orientation_mode == 2) {
        int first_target_frame = static_cast<int>(current_frame_);
        // first future frame is always frame_idx=0, so no step offset
        if (first_target_frame >= static_cast<int>(current_motion_->timesteps)) {
          first_target_frame = static_cast<int>(current_motion_->timesteps) - 1;
        }
        const auto first_body_quat = current_motion_->BodyQuaternions(first_target_frame);
        auto first_ref_rot = quat_mul_d(apply_delta_heading, first_body_quat[0]);
        ref_first_heading = calc_heading_quat_d(first_ref_rot);
      }

      // Gather orientations from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        // Get reference data root rotation at this target frame (first body part, index 0)
        const auto motion_body_quat = current_motion_->BodyQuaternions(target_frame);
        std::array<double, 4> ref_data_root_rot_array = motion_body_quat[0];

        // Calculate new reference root rotation with heading applied
        auto new_ref_root_rot = quat_mul_d(apply_delta_heading, ref_data_root_rot_array);

        // Calculate relative rotation based on orientation_mode:
        //   0: full base quaternion (motion_anchor_ori_b)
        //   1: robot heading only (motion_anchor_ori_heading)
        //   2: reference first-frame heading (motion_anchor_ori_refheading)
        std::array<double, 4> left_quat;
        if (orientation_mode == 1) {
          left_quat = h2_lora_observation::SonicHeadingQuaternion(base_quat);
        } else if (orientation_mode == 2) {
          left_quat = ref_first_heading;
        } else {
          left_quat = base_quat;
        }
        auto base_to_ref_quat = quat_mul_d(quat_conjugate_d(left_quat), new_ref_root_rot);
        auto rotation_matrix = quat_to_rotation_matrix_d(base_to_ref_quat);

        // Extract first 2 columns of rotation matrix and flatten ROW-WISE to 1D array (6 elements)
        // Python: .as_matrix()[..., :2].reshape(1, -1) flattens row by row
        std::array<double, 6> motion_anchor_ori_b = {
            rotation_matrix[0][0], rotation_matrix[0][1], // Row 0, cols 0-1
            rotation_matrix[1][0], rotation_matrix[1][1], // Row 1, cols 0-1
            rotation_matrix[2][0], rotation_matrix[2][1] // Row 2, cols 0-1
        };

        // Copy to observation buffer at the appropriate offset for this frame
        size_t frame_offset = offset + frame_idx * 6; // 6 values per frame
        std::copy(motion_anchor_ori_b.begin(), motion_anchor_ori_b.end(), target_buffer.begin() + frame_offset);
      }

      return true;
    }

    /**
     * @brief Gather heading_diff_robot_ref: yaw diff from robot to reference first frame.
     *
     * Used by g1_dyn decoder in the refheading setting.
     * Output: 6D rotation = quat_to_6d(quat_inv(robot_heading) * ref_first_heading)
     */
    bool GatherHeadingDiffRobotRef(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { return false; }
      if (current_motion_->GetNumBodyQuaternions() == 0) { return false; }
      if (!state_logger_) { return false; }

      double sample_dt = control_dt_;
      auto hist = state_logger_->GetLatest(1, sample_dt);
      const auto& base_quat = hist[0].base_quat;

      auto apply_delta_heading = ComputeApplyDeltaHeading();

      // Get reference root quat at current frame (first future frame = frame_0)
      int first_frame = static_cast<int>(current_frame_);
      if (first_frame >= static_cast<int>(current_motion_->timesteps)) {
        first_frame = static_cast<int>(current_motion_->timesteps) - 1;
      }
      const auto first_body_quat = current_motion_->BodyQuaternions(first_frame);
      auto ref_root_rot = quat_mul_d(apply_delta_heading, first_body_quat[0]);

      // Compute: quat_inv(robot_heading) * ref_first_heading
      auto robot_heading = calc_heading_quat_d(base_quat);
      auto ref_first_heading = calc_heading_quat_d(ref_root_rot);
      auto diff_quat = quat_mul_d(quat_conjugate_d(robot_heading), ref_first_heading);

      auto rotation_matrix = quat_to_rotation_matrix_d(diff_quat);
      std::array<double, 6> heading_diff = {
          rotation_matrix[0][0], rotation_matrix[0][1],
          rotation_matrix[1][0], rotation_matrix[1][1],
          rotation_matrix[2][0], rotation_matrix[2][1]
      };
      std::copy(heading_diff.begin(), heading_diff.end(), target_buffer.begin() + offset);
      return true;
    }

    /// Gather joint positions from N future frames.  If joint_indexes is empty, gathers all 29.
    /// When upper-body control is active (has_upper_body_data_), replaces upper-body joints with
    /// the externally-provided targets.
    bool GatherMotionJointPositionsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> joint_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid joint data
      const auto num_joints = current_motion_->GetNumJoints();
      if (num_joints == 0) {
        std::cerr << "✗ Error: Motion has no joints - cannot gather joint positions" << std::endl;
        return false;
      }
      // Validate requested joint indexes
      for (int idx : joint_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_joints)) {
          std::cerr << "✗ Error: joint index " << idx
                    << " is out of range [0, " << (num_joints - 1)
                    << "] in GatherMotionJointPositionsMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather positions from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }

        const auto motion_joint_pos = PrepareReferenceJointsForPolicy(
            current_motion_->JointPositions(target_frame));

        // If body part indexes are empty, gather all joints
        if (joint_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * 29;  // 29 joints per frame
          std::copy(
              motion_joint_pos.begin(),
              motion_joint_pos.end(),
              target_buffer.begin() + frame_offset
          );
          
          // If upper body control is enabled, use upper body joint positions in buffer to replace the motion_joint_pos
          if (has_upper_body_data_) {
            std::array<double, 29> current_motion_joint_pos = motion_joint_pos;
            for (size_t i = 0; i < 17; i++) {
              current_motion_joint_pos[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = upper_body_joint_positions_buffer_[i];
            }
            std::copy(
              current_motion_joint_pos.begin(),
              current_motion_joint_pos.end(),
              target_buffer.begin() + frame_offset
            );
          }

        // If body part indexes are not empty, gather only the needed joints
        } else {
          size_t frame_offset = offset + frame_idx * joint_indexes.size(); 
          std::vector<double> needed_joints_pos(joint_indexes.size());
          for (size_t i = 0; i < joint_indexes.size(); i++) {
            size_t body_part_index = joint_indexes[i];
            needed_joints_pos[i] = motion_joint_pos.at(body_part_index);
          }
          std::copy(
            needed_joints_pos.begin(),
            needed_joints_pos.end(),
            target_buffer.begin() + frame_offset
          );
        }
      }
      
      return true;
    }
    
    /// Gather joint velocities from N future frames.  Zeroes velocities when not playing.
    bool GatherMotionJointVelocitiesMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> joint_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid joint data
      const auto num_joints = current_motion_->GetNumJoints();
      if (num_joints == 0) {
        std::cerr << "✗ Error: Motion has no joints - cannot gather joint velocities" << std::endl;
        return false;
      }
      // Validate requested joint indexes
      for (int idx : joint_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_joints)) {
          std::cerr << "✗ Error: joint index " << idx
                    << " is out of range [0, " << (num_joints - 1)
                    << "] in GatherMotionJointVelocitiesMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      // Gather velocities from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate frame index (current frame + frame_idx * step_size for future frames)
        int target_frame = static_cast<int>(current_frame_) + frame_idx * step_size;
        
        // If beyond motion length, clamp to last frame (hold final pose)
        if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
          target_frame = static_cast<int>(current_motion_->timesteps) - 1;
        }
        
        const auto motion_joint_vel = PrepareReferenceJointsForPolicy(
            current_motion_->JointVelocities(target_frame));

        // If body part indexes are empty, gather all joints
        if (joint_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * 29;  // 29 joints per frame
          if (operator_state.play) {
            std::copy(
              motion_joint_vel.begin(),
              motion_joint_vel.end(),
              target_buffer.begin() + frame_offset
            );
            // If upper body control is enabled, use upper body joint velocities in buffer to replace the motion_joint_vel
            if (has_upper_body_data_) {
              std::array<double, 29> current_motion_joint_vel = motion_joint_vel;
              for (size_t i = 0; i < 17; i++) {
                current_motion_joint_vel[upper_body_joint_isaaclab_order_in_isaaclab_index[i]] = upper_body_joint_velocities_buffer_[i];
              }
              std::copy(
                current_motion_joint_vel.begin(),
                current_motion_joint_vel.end(),
                target_buffer.begin() + frame_offset
              );
            }
          } else {
            std::fill_n(
              target_buffer.begin() + frame_offset,
              num_joints,
              0.0
            );
          }
        // If body part indexes are not empty, gather only the needed joints
        } else {
          size_t frame_offset = offset + frame_idx * joint_indexes.size(); 
          if (operator_state.play) {
            std::vector<double> needed_joints_vel(joint_indexes.size());
            for (size_t i = 0; i < joint_indexes.size(); i++) {
              size_t body_part_index = joint_indexes[i];
              needed_joints_vel[i] = motion_joint_vel.at(body_part_index);
            }
            std::copy(
              needed_joints_vel.begin(),
              needed_joints_vel.end(),
              target_buffer.begin() + frame_offset
            );
          } else {
            std::fill_n(
              target_buffer.begin() + frame_offset,
              joint_indexes.size(),
              0.0
            );
          }
        }
      }
      
      return true;
    }

    bool GatherH2NativeReferenceMultiFrame(
        std::vector<double>& target_buffer,
        size_t offset,
        int num_frames = 10,
        int step_size = 5) {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }

      const int encode_mode = current_motion_->GetEncodeMode();
      const bool use_g1_reference = encode_mode == 3;
      const bool use_h2_reference = encode_mode == 0;
      const bool project_g1_planner = use_h2_reference &&
          project_h2_native_planner_reference_ &&
          current_motion_.get() == planner_motion_.get();
      if ((use_g1_reference || project_g1_planner) &&
          current_motion_->GetNumJoints() != POLICY_ACTION_DIM) {
        std::cerr << "✗ Error: G1 reference encoder requires 29D native G1 joint_pos/joint_vel"
                  << std::endl;
        return false;
      }
      if (use_h2_reference && !project_g1_planner &&
          (!current_motion_->HasH2JointPositions() ||
           !current_motion_->HasH2JointVelocities() ||
           current_motion_->GetNumH2Joints() != H2_NUM_MOTOR)) {
        std::cerr << "✗ Error: H2 reference encoder requires 31D h2_joint_pos.csv "
                     "and h2_joint_vel.csv sidecars"
                  << std::endl;
        return false;
      }
      if (!use_g1_reference && !use_h2_reference) {
        std::cerr << "✗ Error: Native reference gatherer does not support encoder mode "
                  << encode_mode << std::endl;
        return false;
      }

      saved_frame_for_observation_window_ = std::max(
          saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const int target_frame = ComputeFutureReferenceFrame(frame_idx, step_size);
        const size_t position_offset =
            offset + static_cast<size_t>(frame_idx) * POLICY_ACTION_DIM;
        const size_t velocity_offset =
            offset + static_cast<size_t>(num_frames) * POLICY_ACTION_DIM +
            static_cast<size_t>(frame_idx) * POLICY_ACTION_DIM;

        if (use_g1_reference || project_g1_planner) {
          std::array<double, POLICY_ACTION_DIM> g1_positions = {};
          std::array<double, POLICY_ACTION_DIM> g1_velocities = {};
          std::copy_n(
              current_motion_->JointPositions(target_frame),
              POLICY_ACTION_DIM,
              g1_positions.begin());
          std::copy_n(
              current_motion_->JointVelocities(target_frame),
              POLICY_ACTION_DIM,
              g1_velocities.begin());
          // Keep the planner buffer in G1 space. The opt-in selector-0 path
          // maps each sampled q/qdot once, without altering future replanning.
          if (project_g1_planner) {
            g1_positions = G1PlannerReferenceToH2NativePolicy(g1_positions);
            g1_velocities = G1PlannerReferenceToH2NativePolicy(g1_velocities);
          }
          std::copy(
              g1_positions.begin(),
              g1_positions.end(),
              target_buffer.begin() + position_offset);
          if (operator_state.play) {
            std::copy(
                g1_velocities.begin(),
                g1_velocities.end(),
                target_buffer.begin() + velocity_offset);
          } else {
            std::fill_n(
                target_buffer.begin() + velocity_offset,
                POLICY_ACTION_DIM,
                0.0);
          }
        } else {
          const double* positions = current_motion_->H2JointPositions(target_frame);
          const double* velocities = current_motion_->H2JointVelocities(target_frame);
          for (int policy_index = 0; policy_index < POLICY_ACTION_DIM; ++policy_index) {
            const int local_index = h2_native_policy_to_h2_mujoco[policy_index];
            target_buffer[position_offset + policy_index] = positions[local_index];
            target_buffer[velocity_offset + policy_index] =
                operator_state.play ? velocities[local_index] : 0.0;
          }
        }
      }
      return true;
    }
    
    /// Gather SMPL joint positions from N future frames (3 values × num_smpl_joints per frame).
    bool GatherMotionSmplJointsMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> smpl_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid SMPL joint data
      const auto num_smpl_joints = current_motion_->GetNumSmplJoints();
      if (num_smpl_joints == 0) {
        std::cerr << "✗ Error: Motion has no SMPL joints - cannot gather SMPL joint data" << std::endl;
        return false;
      }
      // Validate requested SMPL joint indexes
      for (int idx : smpl_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_smpl_joints)) {
          std::cerr << "✗ Error: SMPL joint index " << idx
                    << " is out of range [0, " << (num_smpl_joints - 1)
                    << "] in GatherMotionSmplJointsMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      
      // Gather SMPL joints from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }
        
        const auto motion_smpl_joints = current_motion_->SmplJoints(target_frame);

        // If indexes are empty, gather all SMPL joints (all x, y, z coordinates)
        if (smpl_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * num_smpl_joints * 3;  // 3 coordinates per joint
          
          for (size_t j = 0; j < num_smpl_joints; ++j) {
            target_buffer[frame_offset + j * 3 + 0] = motion_smpl_joints[j][0]; // x
            target_buffer[frame_offset + j * 3 + 1] = motion_smpl_joints[j][1]; // y
            target_buffer[frame_offset + j * 3 + 2] = motion_smpl_joints[j][2]; // z
          }
        // If indexes are not empty, gather only the needed SMPL joints
        } else {
          size_t frame_offset = offset + frame_idx * smpl_indexes.size() * 3; 
          for (size_t i = 0; i < smpl_indexes.size(); i++) {
            size_t smpl_idx = smpl_indexes[i];
            target_buffer[frame_offset + i * 3 + 0] = motion_smpl_joints[smpl_idx][0]; // x
            target_buffer[frame_offset + i * 3 + 1] = motion_smpl_joints[smpl_idx][1]; // y
            target_buffer[frame_offset + i * 3 + 2] = motion_smpl_joints[smpl_idx][2]; // z
          }
        }
      }
      
      return true;
    }

    bool GatherMotionSmplJointsLocalMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 10, int step_size = 1, std::vector<int> smpl_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }
      const auto num_smpl_joints = current_motion_->GetNumSmplJoints();
      if (num_smpl_joints == 0) {
        std::cerr << "✗ Error: Motion has no SMPL joints - cannot gather local SMPL joint data" << std::endl;
        return false;
      }
      if (current_motion_->GetNumSmplRootQuaternions() == 0) {
        std::cerr << "✗ Error: Motion has no smpl_root_quat.csv - cannot match training SMPL local observation" << std::endl;
        return false;
      }
      for (int idx : smpl_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_smpl_joints)) {
          std::cerr << "✗ Error: SMPL joint index " << idx
                    << " is out of range [0, " << (num_smpl_joints - 1)
                    << "] in GatherMotionSmplJointsLocalMultiFrame" << std::endl;
          return false;
        }
      }

      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      const size_t joints_per_frame = smpl_indexes.empty() ? static_cast<size_t>(num_smpl_joints) : smpl_indexes.size();

      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        const int target_frame = ComputeFutureReferenceFrame(frame_idx, step_size);
        const auto motion_smpl_joints = current_motion_->SmplJoints(target_frame);
        const auto smpl_root_quat = current_motion_->SmplRootQuaternions(target_frame)[0];
        const auto inv_smpl_root_quat = quat_conjugate_d(smpl_root_quat);
        const size_t frame_offset = offset + frame_idx * joints_per_frame * 3;

        if (smpl_indexes.empty()) {
          for (size_t j = 0; j < static_cast<size_t>(num_smpl_joints); ++j) {
            const std::array<double, 3> joint_vec = {
                motion_smpl_joints[j][0],
                motion_smpl_joints[j][1],
                motion_smpl_joints[j][2],
            };
            const auto local_joint = quat_rotate_d(inv_smpl_root_quat, joint_vec);
            target_buffer[frame_offset + j * 3 + 0] = local_joint[0];
            target_buffer[frame_offset + j * 3 + 1] = local_joint[1];
            target_buffer[frame_offset + j * 3 + 2] = local_joint[2];
          }
        } else {
          for (size_t i = 0; i < smpl_indexes.size(); i++) {
            const size_t smpl_idx = static_cast<size_t>(smpl_indexes[i]);
            const std::array<double, 3> joint_vec = {
                motion_smpl_joints[smpl_idx][0],
                motion_smpl_joints[smpl_idx][1],
                motion_smpl_joints[smpl_idx][2],
            };
            const auto local_joint = quat_rotate_d(inv_smpl_root_quat, joint_vec);
            target_buffer[frame_offset + i * 3 + 0] = local_joint[0];
            target_buffer[frame_offset + i * 3 + 1] = local_joint[1];
            target_buffer[frame_offset + i * 3 + 2] = local_joint[2];
          }
        }
      }

      return true;
    }

    bool GatherSmplRootOrientationBodyMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 10, int step_size = 1) {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }
      if (current_motion_->GetNumSmplRootQuaternions() == 0) {
        std::cerr << "✗ Error: Motion has no smpl_root_quat.csv - cannot gather SMPL root orientation" << std::endl;
        return false;
      }
      if (!state_logger_) { return false; }

      double sample_dt = control_dt_;
      auto hist = state_logger_->GetLatest(1, sample_dt);
      const auto& base_quat = hist[0].base_quat;
      const auto apply_delta_heading = ComputeApplyDeltaHeading();

      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        const int target_frame = ComputeFutureReferenceFrame(frame_idx, step_size);
        const auto smpl_root_quat = current_motion_->SmplRootQuaternions(target_frame)[0];
        const auto root_rot_dif =
            h2_lora_observation::HeadingAlignedSmplRootDifference(
                base_quat, apply_delta_heading, smpl_root_quat);
        const auto rotation_matrix = quat_to_rotation_matrix_d(root_rot_dif);
        const std::array<double, 6> rot6d = {
            rotation_matrix[0][0], rotation_matrix[0][1],
            rotation_matrix[1][0], rotation_matrix[1][1],
            rotation_matrix[2][0], rotation_matrix[2][1],
        };
        const size_t frame_offset = offset + frame_idx * 6;
        std::copy(rot6d.begin(), rot6d.end(), target_buffer.begin() + frame_offset);
      }
      return true;
    }

    bool GatherSmplRootOrientationHeadingMultiFrame(
        std::vector<double>& target_buffer,
        size_t offset,
        int num_frames = 10,
        int step_size = 1) {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }
      if (current_motion_->GetNumSmplRootQuaternions() == 0) {
        std::cerr << "✗ Error: Motion has no smpl_root_quat.csv - cannot gather SONIC v1.1 SMPL root orientation" << std::endl;
        return false;
      }
      if (!state_logger_) { return false; }

      const auto hist = state_logger_->GetLatest(1, control_dt_);
      const auto& base_quat = hist[0].base_quat;
      const auto apply_delta_heading = ComputeApplyDeltaHeading();

      saved_frame_for_observation_window_ = std::max(
          saved_frame_for_observation_window_,
          (num_frames - 1) * step_size + 1);
      for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const int target_frame = ComputeFutureReferenceFrame(frame_idx, step_size);
        const auto smpl_root_quat =
            current_motion_->SmplRootQuaternions(target_frame)[0];
        const auto root_rot_dif =
            h2_lora_observation::HeadingCanonicalizedSmplRootDifference(
                base_quat, apply_delta_heading, smpl_root_quat);
        const auto rotation_matrix = quat_to_rotation_matrix_d(root_rot_dif);
        const std::array<double, 6> rot6d = {
            rotation_matrix[0][0], rotation_matrix[0][1],
            rotation_matrix[1][0], rotation_matrix[1][1],
            rotation_matrix[2][0], rotation_matrix[2][1],
        };
        std::copy(
            rot6d.begin(), rot6d.end(),
            target_buffer.begin() + offset + frame_idx * 6);
      }
      return true;
    }
    
    /// Gather SMPL pose (axis-angle) data from N future frames (3 values × num_smpl_poses per frame).
    bool GatherMotionSmplPosesMultiFrame(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 5, std::vector<int> pose_indexes = {}) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Check if motion has valid SMPL pose data
      const auto num_smpl_poses = current_motion_->GetNumSmplPoses();
      if (num_smpl_poses == 0) {
        std::cerr << "✗ Error: Motion has no SMPL poses - cannot gather SMPL pose data" << std::endl;
        return false;
      }
      // Validate requested SMPL pose indexes
      for (int idx : pose_indexes) {
        if (idx < 0 || idx >= static_cast<int>(num_smpl_poses)) {
          std::cerr << "✗ Error: SMPL pose index " << idx
                    << " is out of range [0, " << (num_smpl_poses - 1)
                    << "] in GatherMotionSmplPosesMultiFrame" << std::endl;
          return false;
        }
      }
      saved_frame_for_observation_window_ = std::max(saved_frame_for_observation_window_, (num_frames - 1) * step_size + 1);
      
      // Gather SMPL poses from multiple future frames with step intervals
      for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Calculate target frame: if playing, advance through frames; if not playing, hold current frame
        int target_frame = static_cast<int>(current_frame_);
        if (operator_state.play) {
          target_frame += frame_idx * step_size;
          // If beyond motion length, clamp to last frame (hold final pose)
          if (target_frame >= static_cast<int>(current_motion_->timesteps)) {
            target_frame = static_cast<int>(current_motion_->timesteps) - 1;
          }
        }
        
        const auto motion_smpl_poses = current_motion_->SmplPoses(target_frame);

        // If indexes are empty, gather all SMPL poses (all x, y, z coordinates)
        if (pose_indexes.empty()) {
          size_t frame_offset = offset + frame_idx * num_smpl_poses * 3;  // 3 coordinates per pose
          
          for (size_t p = 0; p < num_smpl_poses; ++p) {
            target_buffer[frame_offset + p * 3 + 0] = motion_smpl_poses[p][0]; // x
            target_buffer[frame_offset + p * 3 + 1] = motion_smpl_poses[p][1]; // y
            target_buffer[frame_offset + p * 3 + 2] = motion_smpl_poses[p][2]; // z
          }
        // If indexes are not empty, gather only the needed SMPL poses
        } else {
          size_t frame_offset = offset + frame_idx * pose_indexes.size() * 3; 
          for (size_t i = 0; i < pose_indexes.size(); i++) {
            size_t pose_idx = pose_indexes[i];
            target_buffer[frame_offset + i * 3 + 0] = motion_smpl_poses[pose_idx][0]; // x
            target_buffer[frame_offset + i * 3 + 1] = motion_smpl_poses[pose_idx][1]; // y
            target_buffer[frame_offset + i * 3 + 2] = motion_smpl_poses[pose_idx][2]; // z
          }
        }
      }
      
      return true;
    }
    
    // =========================================================================
    // VR 3/5-point observation helpers and gatherers
    //
    // VR tracking data can come from two sources:
    //   1. External input interface (ZMQ / ROS2) → use buffered values directly.
    //   2. Computed from motion body positions with per-body offsets, then
    //      normalised to the root body frame.
    //
    // The UpdateVR*PointIndices() helpers map the canonical VR body indices
    // (e.g. left wrist = body 28) to storage indices in the current
    // motion's BodyPartIndexes array (which varies between motions).
    // =========================================================================

    /// Map canonical VR 3-point body indices to current motion's storage indices.
    /// @return True if all 3 required indices were found.
    bool UpdateVR3PointIndices() {
      if (!current_motion_ || current_motion_->timesteps == 0) {
        return false;
      }
      
      auto body_part_indexes = current_motion_->BodyPartIndexes();
      
      // Reset indices
      actual_vr_3point_index = {-1, -1, -1};
      
      // Find positions of required indices
      for (size_t i = 0; i < body_part_indexes.size(); i++) {
        for (size_t j = 0; j < vr_3point_index.size(); j++) {
          if (body_part_indexes[i] == vr_3point_index[j] && actual_vr_3point_index[j] == -1) {
            actual_vr_3point_index[j] = static_cast<int>(i);
            break; // Take first occurrence, move to next body part
          }
        }
      }
      
      // Validate that all required indices were found
      for (size_t j = 0; j < actual_vr_3point_index.size(); j++) {
        if (actual_vr_3point_index[j] == -1) {
          // VR3Point index not found - will fall back to interface values
          return false;
        }
      }
      
      return true;
    }

    // Helper function to update VR3Point indices for current motion
    // Returns true if all indices found, false otherwise
    bool UpdateVR5PointIndices() {
      if (!current_motion_ || current_motion_->timesteps == 0) { return false; }

      auto body_part_indexes = current_motion_->BodyPartIndexes();

      // Reset indices
      actual_vr_5point_index = {-1, -1, -1, -1, -1};

      // Find positions of required indices
      for (size_t i = 0; i < body_part_indexes.size(); i++) {
        for (size_t j = 0; j < vr_5point_index.size(); j++) {
          if (body_part_indexes[i] == vr_5point_index[j] && actual_vr_5point_index[j] == -1) {
            actual_vr_5point_index[j] = static_cast<int>(i);
            break; // Take first occurrence, move to next body part
          }
        }
      }

      // Validate that all required indices were found
      for (size_t j = 0; j < actual_vr_5point_index.size(); j++) {
        if (actual_vr_5point_index[j] == -1) {
          // VR5Point index not found - will fall back to interface values
          return false;
        }
      }

      return true;
    }

    bool GatherVR3PointPosition(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }


      // 1. Input interface providing VR 3-point (ZMQ, ROS2) -> use buffer
      // 2. Otherwise -> compute from motion body positions with offsets
      bool use_buffered_vr3 = has_vr_3point_data_;
      
      if (use_buffered_vr3) {
        // Use VR 3-point data from input interface directly
        // The sender (ZMQ/ROS2/planner) has already applied offsets and root normalization
        std::copy(vr_3point_position_buffer_.begin(), vr_3point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // No external VR 3-point data - compute from motion's body positions
      // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
      // If indices not found, fall back to buffered interface values (fallback to default values)
      if (!UpdateVR3PointIndices()) {
        std::copy(vr_3point_position_buffer_.begin(), vr_3point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // read from current motion
      int frame = current_frame_ % current_motion_->timesteps;
      const auto& motion_body_pos = current_motion_->BodyPositions(frame);
      const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

      // VR 3-point offsets in body's local frame (matching ros2_input_handler.hpp and visualize_motion.py)
      // Order: left_wrist (index 0), right_wrist (index 1), head/torso (index 2)
      // NOTE: Y-axis is mirrored for right wrist (symmetric robot)
      constexpr std::array<std::array<double, 3>, 3> VR_3POINT_OFFSETS = {{
        {0.18, -0.025, 0.0},   // left wrist offset
        {0.18, +0.025, 0.0},   // right wrist offset (Y mirrored for symmetry)
        {0.0, 0.0, 0.35}       // head offset (applied to torso body)
      }};

      // Apply offsets to body positions to get VR 3-point positions
      std::array<double, 9> vr_3point_position;
      for (int i = 0; i < 3; ++i) {
        // Get body position and orientation
        const auto& body_pos = motion_body_pos[actual_vr_3point_index[i]];
        const auto& body_quat = motion_body_quat[actual_vr_3point_index[i]];
        
        // Rotate offset by body orientation to get offset in world frame
        auto offset_world = quat_rotate_d(body_quat, VR_3POINT_OFFSETS[i]);
        
        // Add rotated offset to body position to get VR point position
        vr_3point_position[i * 3 + 0] = body_pos[0] + offset_world[0];
        vr_3point_position[i * 3 + 1] = body_pos[1] + offset_world[1];
        vr_3point_position[i * 3 + 2] = body_pos[2] + offset_world[2];
      }

      // normalize positions to the root by - root_pos rotate and normalize the root_quat
      auto root_pos = current_motion_->BodyPositions(frame)[0];
      auto root_quat = current_motion_->BodyQuaternions(frame)[0];
      
      // Normalize each VR 3-point position relative to root
      // Apply inverse transformation: subtract root position, then rotate by inverse root quaternion
      auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
      
      for (int i = 0; i < 3; ++i) {
        // Extract position for this VR point
        std::array<double, 3> point_pos = {
          vr_3point_position[i * 3 + 0],
          vr_3point_position[i * 3 + 1], 
          vr_3point_position[i * 3 + 2]
        };
        
        // Make position relative to root (subtract root position)
        std::array<double, 3> relative_pos = {
          point_pos[0] - root_pos[0],
          point_pos[1] - root_pos[1],
          point_pos[2] - root_pos[2]
        };
        
        // Rotate relative position by inverse root quaternion
        auto normalized_pos = quat_rotate_d(root_quat_inv, relative_pos);
        
        // Store normalized position back to the array
        vr_3point_position[i * 3 + 0] = normalized_pos[0];
        vr_3point_position[i * 3 + 1] = normalized_pos[1];
        vr_3point_position[i * 3 + 2] = normalized_pos[2];
      }
      
      std::copy(
        vr_3point_position.begin(),
        vr_3point_position.end(),
        target_buffer.begin() + offset
      );
      
      // Update buffer with the computed value from dataset 
      vr_3point_position_buffer_ = vr_3point_position;
      
      return true;
    }

    bool GatherVR3PointOrientation(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      
      bool use_buffered_vr3 = has_vr_3point_data_;

      if (use_buffered_vr3) {
        std::copy(vr_3point_orientation_buffer_.begin(), vr_3point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }
      
      // No external VR 3-point data - compute from motion's body positions
      // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
      // If indices not found, fall back to buffered interface values (fallback to default values)
      if (!UpdateVR3PointIndices()) {
        std::copy(vr_3point_orientation_buffer_.begin(), vr_3point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      }

      // read from current motion
      int frame = current_frame_ % current_motion_->timesteps;
      const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

      // Copy quaternions to observation buffer (flatten 3x4D quaternions to 12 values)
      std::array<double, 12> vr_3point_orientation = {
        motion_body_quat[actual_vr_3point_index[0]][0], motion_body_quat[actual_vr_3point_index[0]][1], motion_body_quat[actual_vr_3point_index[0]][2], motion_body_quat[actual_vr_3point_index[0]][3], // Quat 0: w,x,y,z
        motion_body_quat[actual_vr_3point_index[1]][0], motion_body_quat[actual_vr_3point_index[1]][1], motion_body_quat[actual_vr_3point_index[1]][2], motion_body_quat[actual_vr_3point_index[1]][3], // Quat 1: w,x,y,z
        motion_body_quat[actual_vr_3point_index[2]][0], motion_body_quat[actual_vr_3point_index[2]][1], motion_body_quat[actual_vr_3point_index[2]][2], motion_body_quat[actual_vr_3point_index[2]][3]  // Quat 2: w,x,y,z
      };

      // Normalize quaternions relative to root quaternion: quat_mul(quat_inv(root_quat), vr_quat)
      auto root_quat = current_motion_->BodyQuaternions(frame)[0];
      auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
      
      for (int i = 0; i < 3; ++i) {
        // Extract quaternion for this VR point
        std::array<double, 4> point_quat = {
          vr_3point_orientation[i * 4 + 0], // w
          vr_3point_orientation[i * 4 + 1], // x
          vr_3point_orientation[i * 4 + 2], // y
          vr_3point_orientation[i * 4 + 3]  // z
        };
        
        // Apply quaternion normalization: quat_mul(quat_inv(root_quat), vr_quat)
        auto normalized_quat = quat_mul_d(root_quat_inv, point_quat);
        
        // Store normalized quaternion back to the array
        vr_3point_orientation[i * 4 + 0] = normalized_quat[0]; // w
        vr_3point_orientation[i * 4 + 1] = normalized_quat[1]; // x
        vr_3point_orientation[i * 4 + 2] = normalized_quat[2]; // y
        vr_3point_orientation[i * 4 + 3] = normalized_quat[3]; // z
      }
      
      std::copy(
        vr_3point_orientation.begin(),
        vr_3point_orientation.end(),
        target_buffer.begin() + offset
      );
      
      // Update buffer with the computed value from dataset
      vr_3point_orientation_buffer_ = vr_3point_orientation;
      
      return true;
    }


    bool GatherVR5PointPosition(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }

      if (current_motion_ && current_motion_ == planner_motion_){
        // if planner motion, read from buffered interface data
        std::copy(vr_5point_position_buffer_.begin(), vr_5point_position_buffer_.end(), target_buffer.begin() + offset);
        return true;
      } else {
        // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
        // If indices not found, fall back to buffered interface values
        if (!UpdateVR5PointIndices()) {
          std::copy(vr_5point_position_buffer_.begin(), vr_5point_position_buffer_.end(), target_buffer.begin() + offset);
          return true;
        }
        
        // read from current motion
        int frame = current_frame_ % current_motion_->timesteps;
        const auto& motion_body_pos = current_motion_->BodyPositions(frame);
        const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

        // VR 3-point offsets in body's local frame (matching ros2_input_handler.hpp and visualize_motion.py)
        // Order: left_wrist (index 0), right_wrist (index 1), head/torso (index 2)
        // NOTE: Y-axis is mirrored for right wrist (symmetric robot)
        constexpr std::array<std::array<double, 3>, 5> VR_5POINT_OFFSETS = {
          std::array<double, 3>{0.0, 0.0, 0.0},   // left wrist offset
          std::array<double, 3>{0.0, 0.0, 0.0},   // right wrist offset (Y mirrored for symmetry)
          std::array<double, 3>{0.0, 0.0, 0.0},   // head offset (applied to torso body)
          std::array<double, 3>{0.0, 0.0, 0.0},   // left ankle offset
          std::array<double, 3>{0.0, 0.0, 0.0}    // right ankle offset
        };

        // Apply offsets to body positions to get VR 3-point positions
        std::array<double, 15> vr_5point_position;
        for (int i = 0; i < 5; ++i) {
          // Get body position and orientation
          const auto& body_pos = motion_body_pos[actual_vr_5point_index[i]];
          const auto& body_quat = motion_body_quat[actual_vr_5point_index[i]];
          
          // Rotate offset by body orientation to get offset in world frame
          auto offset_world = quat_rotate_d(body_quat, VR_5POINT_OFFSETS[i]);
          
          // Add rotated offset to body position to get VR point position
          vr_5point_position[i * 3 + 0] = body_pos[0] + offset_world[0];
          vr_5point_position[i * 3 + 1] = body_pos[1] + offset_world[1];
          vr_5point_position[i * 3 + 2] = body_pos[2] + offset_world[2];
        }

        // normalize positions to the root by - root_pos rotate and normalize the root_quat
        auto root_pos = current_motion_->BodyPositions(frame)[0];
        auto root_quat = current_motion_->BodyQuaternions(frame)[0];
        
        // Normalize each VR 3-point position relative to root
        // Apply inverse transformation: subtract root position, then rotate by inverse root quaternion
        auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
        
        for (int i = 0; i < 5; ++i) {
          // Extract position for this VR point
          std::array<double, 3> point_pos = {
            vr_5point_position[i * 3 + 0],
            vr_5point_position[i * 3 + 1], 
            vr_5point_position[i * 3 + 2]
          };
          
          // Make position relative to root (subtract root position)
          std::array<double, 3> relative_pos = {
            point_pos[0] - root_pos[0],
            point_pos[1] - root_pos[1],
            point_pos[2] - root_pos[2]
          };
          
          // Rotate relative position by inverse root quaternion
          auto normalized_pos = quat_rotate_d(root_quat_inv, relative_pos);
          
          // Store normalized position back to the array
          vr_5point_position[i * 3 + 0] = normalized_pos[0];
          vr_5point_position[i * 3 + 1] = normalized_pos[1];
          vr_5point_position[i * 3 + 2] = normalized_pos[2];
        }
        
        std::copy(
          vr_5point_position.begin(),
          vr_5point_position.end(),
          target_buffer.begin() + offset
        );
        
        // Update buffer with the computed value from dataset 
        vr_5point_position_buffer_ = vr_5point_position;
        
        return true;
      }
      return false;
    }

    bool GatherVR5PointOrientation(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      
      if (current_motion_ && current_motion_ == planner_motion_){
        std::copy(vr_5point_orientation_buffer_.begin(), vr_5point_orientation_buffer_.end(), target_buffer.begin() + offset);
        return true;
      } else {
        // Update VR3Point indices for current motion (different motions have different BodyPartIndexes)
        // If indices not found, fall back to buffered interface values
        if (!UpdateVR5PointIndices()) {
          std::copy(vr_5point_orientation_buffer_.begin(), vr_5point_orientation_buffer_.end(), target_buffer.begin() + offset);
          return true;
        }
        
        // read from current motion
        int frame = current_frame_ % current_motion_->timesteps;
        const auto& motion_body_quat = current_motion_->BodyQuaternions(frame);

        // Copy quaternions to observation buffer (flatten 3x4D quaternions to 12 values)
        std::array<double, 20> vr_5point_orientation = {
          motion_body_quat[actual_vr_5point_index[0]][0], motion_body_quat[actual_vr_5point_index[0]][1], motion_body_quat[actual_vr_5point_index[0]][2], motion_body_quat[actual_vr_5point_index[0]][3], // Quat 0: w,x,y,z
          motion_body_quat[actual_vr_5point_index[1]][0], motion_body_quat[actual_vr_5point_index[1]][1], motion_body_quat[actual_vr_5point_index[1]][2], motion_body_quat[actual_vr_5point_index[1]][3], // Quat 1: w,x,y,z
          motion_body_quat[actual_vr_5point_index[2]][0], motion_body_quat[actual_vr_5point_index[2]][1], motion_body_quat[actual_vr_5point_index[2]][2], motion_body_quat[actual_vr_5point_index[2]][3], // Quat 2: w,x,y,z
          motion_body_quat[actual_vr_5point_index[3]][0], motion_body_quat[actual_vr_5point_index[3]][1], motion_body_quat[actual_vr_5point_index[3]][2], motion_body_quat[actual_vr_5point_index[3]][3], // Quat 3: w,x,y,z
          motion_body_quat[actual_vr_5point_index[4]][0], motion_body_quat[actual_vr_5point_index[4]][1], motion_body_quat[actual_vr_5point_index[4]][2], motion_body_quat[actual_vr_5point_index[4]][3]  // Quat 4: w,x,y,z
        };

        // Normalize quaternions relative to root quaternion: quat_mul(quat_inv(root_quat), vr_quat)
        auto root_quat = current_motion_->BodyQuaternions(frame)[0];
        auto root_quat_inv = quat_conjugate_d(root_quat); // Get inverse quaternion
        
        for (int i = 0; i < 5; ++i) {
          // Extract quaternion for this VR point
          std::array<double, 4> point_quat = {
            vr_5point_orientation[i * 4 + 0], // w
            vr_5point_orientation[i * 4 + 1], // x
            vr_5point_orientation[i * 4 + 2], // y
            vr_5point_orientation[i * 4 + 3]  // z
          };
          
          // Apply quaternion normalization: quat_mul(quat_inv(root_quat), vr_quat)
          auto normalized_quat = quat_mul_d(root_quat_inv, point_quat);
          
          // Store normalized quaternion back to the array
          vr_5point_orientation[i * 4 + 0] = normalized_quat[0]; // w
          vr_5point_orientation[i * 4 + 1] = normalized_quat[1]; // x
          vr_5point_orientation[i * 4 + 2] = normalized_quat[2]; // y
          vr_5point_orientation[i * 4 + 3] = normalized_quat[3]; // z
        }
        
        std::copy(
          vr_5point_orientation.begin(),
          vr_5point_orientation.end(),
          target_buffer.begin() + offset
        );
        
        // Update buffer with the computed value from dataset
        vr_5point_orientation_buffer_ = vr_5point_orientation;
        
        return true;
      }
      return false;
    }

    bool GatherVR3PointCompliance(std::vector<double>& target_buffer, size_t offset) {
      if (!current_motion_ || current_motion_->timesteps == 0) { 
        return false; 
      }
      // Compliance always comes from interface (not from motion dataset)
      std::copy(vr_3point_compliance_buffer_.begin(), vr_3point_compliance_buffer_.end(), target_buffer.begin() + offset);
      return true;
    }

    // =========================================================================
    // History-based observation gatherers (from StateLogger)
    //
    // These functions read past robot state from the StateLogger ring buffer
    // to provide temporal context to the policy.  They support multi-frame
    // history with configurable step size (e.g., 4 frames at 1× dt = last
    // 4 ticks; 10 frames at 2× dt = last 20 ticks sampled every other one).
    //
    // Each returns false if the required historical data is unavailable.
    // =========================================================================

    /// Gather historical body joint positions (num_frames × joints_per_frame).
    bool GatherHisBodyJointPositions(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      // step_size here means: sample every step_size * dt (e.g., step_size=2 means 0, 2*dt, 4*dt, ...)
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].body_q.empty()) {
        std::cerr << "✗ Error: No historical joint position data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].body_q.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisBodyJointPositions" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          // All joints
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.body_q.empty()) {
            std::copy(entry.body_q.begin(), entry.body_q.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          // Specific joints
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.body_q.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.body_q.size()) ? entry.body_q[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisBodyJointVelocities(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].body_dq.empty()) {
        std::cerr << "✗ Error: No historical joint velocity data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].body_dq.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisBodyJointVelocities" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.body_dq.empty()) {
            std::copy(entry.body_dq.begin(), entry.body_dq.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.body_dq.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.body_dq.size()) ? entry.body_dq[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisLastActions(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, std::vector<int> body_part_indexes = {}, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty() || hist[0].last_action.empty()) {
        std::cerr << "✗ Error: No historical last action data available" << std::endl;
        return false;
      }
      const auto joints = static_cast<int>(hist[0].last_action.size());
      // Validate requested joint indexes
      for (int idx : body_part_indexes) {
        if (idx < 0 || idx >= joints) {
          std::cerr << "✗ Error: body joint index " << idx
                    << " is out of range [0, " << (joints - 1)
                    << "] in GatherHisLastActions" << std::endl;
          return false;
        }
      }

      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        if (body_part_indexes.empty()) {
          size_t frame_offset = offset + static_cast<size_t>(f) * joints;
          if (!entry.last_action.empty()) {
            std::copy(entry.last_action.begin(), entry.last_action.begin() + joints, target_buffer.begin() + frame_offset);
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, joints, 0.0);
          }
        } else {
          size_t frame_offset = offset + static_cast<size_t>(f) * body_part_indexes.size();
          if (!entry.last_action.empty()) {
            for (size_t i = 0; i < body_part_indexes.size(); ++i) {
              int idx = body_part_indexes[i];
              target_buffer[frame_offset + i] = (idx >= 0 && static_cast<size_t>(idx) < entry.last_action.size()) ? entry.last_action[static_cast<size_t>(idx)] : 0.0;
            }
          } else {
            std::fill_n(target_buffer.begin() + frame_offset, body_part_indexes.size(), 0.0);
          }
        }
      }
      return true;
    }

    bool GatherHisBaseAngularVelocity(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty()) {
        std::cerr << "✗ Error: No historical base angular velocity data available" << std::endl;
        return false;
      }
      
      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        size_t frame_offset = offset + static_cast<size_t>(f) * 3;
        target_buffer[frame_offset + 0] = entry.base_ang_vel[0];
        target_buffer[frame_offset + 1] = entry.base_ang_vel[1];
        target_buffer[frame_offset + 2] = entry.base_ang_vel[2];
      }
      return true;
    }

    bool GatherHisGravityDir(std::vector<double>& target_buffer, size_t offset, int num_frames = 5, int step_size = 1, bool newest_first = false) {
      if (!state_logger_) { return false; }
      double sample_dt = control_dt_ * step_size;
      auto hist = state_logger_->GetLatest(static_cast<size_t>(num_frames), sample_dt, newest_first);
      
      // Check if we have valid historical data
      if (hist.empty()) {
        std::cerr << "✗ Error: No historical gravity direction data available" << std::endl;
        return false;
      }
      
      for (int f = 0; f < num_frames; ++f) {
        const auto &entry = hist[static_cast<size_t>(f)];
        auto gravity_dir_inv = quat_conjugate_d(entry.base_quat);
        std::array<double, 3> gravity_dir = quat_rotate_d(gravity_dir_inv, {0.0, 0.0, -1.0});
        size_t frame_offset = offset + static_cast<size_t>(f) * 3;
        target_buffer[frame_offset + 0] = gravity_dir[0];
        target_buffer[frame_offset + 1] = gravity_dir[1];
        target_buffer[frame_offset + 2] = gravity_dir[2];
      }
      return true;
    }

    // =========================================================================
    // Token / encoder observation gatherers
    //
    // The token state can be produced in two ways:
    //   1. Local encoder: GatherEncoderObservations() fills the encoder's
    //      input buffer, Encode() runs TensorRT inference, and the output
    //      tokens are copied into token_state_data_.
    //   2. External source: token_state_data_ was already populated in
    //      GatherInputInterfaceData() from ZMQ / ROS2.
    // =========================================================================

    /// Populate the token_state observation (either from local encoder or external data).
    bool GatherTokenState(std::vector<double>& target_buffer, size_t offset) {
      if (!is_using_encoder_) {
        // No encoder configured; use token_state_data_ (can be set externally via ROS2/ZMQ)
        std::copy(token_state_data_.begin(), token_state_data_.end(), target_buffer.begin() + offset);
        return true;
      }

      if (!encoder_engine_) {
        std::cerr << "✗ Error: Encoder engine not initialized!" << std::endl;
        return false;
      }

      if (!GatherEncoderObservations()) {
        std::cerr << "✗ Error: Failed to gather encoder observations!" << std::endl;
        return false;
      }

      // Run encoder inference (handles CPU→GPU transfer, inference, GPU→CPU transfer)
      if (!encoder_engine_->Encode()) {
        std::cerr << "✗ Error: Encoder inference failed" << std::endl;
        return false;
      }

      // Access encoded tokens from encoder's internal buffer (already populated by Encode)
      auto& token_buffer = encoder_engine_->GetTokenBuffer();
      const size_t token_dim = encoder_engine_->GetTokenDimension();
      
      // Validate token dimension matches expected size
      if (token_dim != token_state_data_.size()) {
        std::cerr << "✗ Error: Encoder output dimension (" << token_dim 
                  << ") doesn't match token state buffer size (" << token_state_data_.size() << ")!" << std::endl;
        return false;
      }
      
      for (size_t i = 0; i < token_dim; ++i) {
        const double v = static_cast<double>(token_buffer[i]);
        token_state_data_[i] = v;
        target_buffer[offset + i] = v;
      }
      return true;
    }

    bool GatherEncoderMode(std::vector<double>& target_buffer, size_t offset, int fill_zeros_num = 0) {
      target_buffer[offset] = static_cast<float>(current_motion_->GetEncodeMode());
      for (int i = 1; i <= fill_zeros_num; ++i) {
        target_buffer[offset + i] = 0;
      }
      return true;
    }

    bool GatherEncoderIndex(std::vector<double>& target_buffer, size_t offset) {
      int mode = current_motion_->GetEncodeMode();
      if (mode < 0) {
        mode = 0;
      }
      target_buffer[offset] = static_cast<double>(mode);
      for (int i = 0; i < 3; ++i) {
        target_buffer[offset + 1 + i] = (i == mode) ? 1.0 : 0.0;
      }
      return true;
    }

    int ComputeReferenceFrameStep(double dt_future_ref_frames, double reference_fps) const {
      if (dt_future_ref_frames <= 0.0 || reference_fps <= 0.0) {
        return 1;
      }
      const double exact_steps = dt_future_ref_frames * reference_fps;
      const int frame_step = std::max(1, static_cast<int>(std::floor(exact_steps + 1e-6)));
      if (std::abs(static_cast<double>(frame_step) - exact_steps) > 1e-4) {
        std::cout << "⚠ Warning: future ref dt " << dt_future_ref_frames
                  << "s at " << reference_fps << " FPS maps to non-integer frame step "
                  << exact_steps << "; using floor step " << frame_step << std::endl;
      }
      return frame_step;
    }

    void ConfigureReferenceFutureSteps(const MotionSequence& motion) {
      const double reference_fps = motion.GetFPS();
      if (std::abs(reference_future_step_fps_ - reference_fps) < 1e-6) {
        return;
      }
      if (use_h2_robot_) {
        // Match sonic_h2lora_release/config.yaml:
        //   dt_future_ref_frames=0.1 for G1/teleop robot references
        //   smpl_dt_future_ref_frames=0.02 for SMPL references
        robot_future_step_ = ComputeReferenceFrameStep(0.1, reference_fps);
        smpl_future_step_ = ComputeReferenceFrameStep(0.02, reference_fps);
      } else {
        // Preserve the existing G1 deploy default unless an observation explicitly
        // requests a fixed *_step5 variant.
        robot_future_step_ = 1;
        smpl_future_step_ = 1;
      }
      std::cout << "[Reference Timing] motion FPS=" << reference_fps
                << ", robot_future_step=" << robot_future_step_
                << ", smpl_future_step=" << smpl_future_step_ << std::endl;
      reference_future_step_fps_ = reference_fps;
    }

    // =========================================================================
    // Observation registry
    //
    // Single source of truth mapping observation names (as they appear in
    // observation_config.yaml) to their dimension and gatherer lambda.
    // Both the policy and encoder observation systems use this registry.
    // =========================================================================

    /// Build the complete observation registry (name, dimension, gatherer function).
    std::vector<ObservationRegistry> GetObservationRegistry() {
      // Use token dimension from encoder config (defaults to 64 if not configured)
      size_t token_dim = (encoder_config_.dimension > 0) ? encoder_config_.dimension : 64;
      return {{"token_state", token_dim, [this](std::vector<double>& buf, size_t offset) { return GatherTokenState(buf, offset); }},
              {"encoder_mode", 3, [this](std::vector<double>& buf, size_t offset) { return GatherEncoderMode(buf, offset, 2); }},
              {"encoder_mode_4", 4, [this](std::vector<double>& buf, size_t offset) { return GatherEncoderMode(buf, offset, 3); }},
              {"encoder_index", 4, [this](std::vector<double>& buf, size_t offset) { return GatherEncoderIndex(buf, offset); }},
              {"motion_joint_positions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 1, 1); }},
              {"motion_joint_velocities", 29, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 1, 1); }},
              {"motion_anchor_orientation", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1); }},
              {"motion_root_z_position", 1, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 1, 1); }},
              {"motion_root_z_position_10frame_step5", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 5); }},
              {"motion_root_z_position_10frame_step1", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 1); }},
              {"motion_root_z_position_3frame_step1", 3, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 3, 1); }},
              {"motion_anchor_orientation_10frame_step5", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 5); }},
              {"motion_anchor_orientation_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1); }},
              {"command_multi_future_nonflat", 580, [this](std::vector<double>& buf, size_t offset) {
                return GatherMotionJointPositionsMultiFrame(buf, offset, 10, robot_future_step_)
                    && GatherMotionJointVelocitiesMultiFrame(buf, offset + 290, 10, robot_future_step_);
              }},
              {"h2_native_reference", 580, [this](std::vector<double>& buf, size_t offset) {
                return GatherH2NativeReferenceMultiFrame(buf, offset, 10, robot_future_step_);
              }},
              {"motion_anchor_ori_b_mf_nonflat", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, robot_future_step_); }},
              {"motion_anchor_ori_heading_mf_nonflat", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, robot_future_step_, 1); }},
              {"command_multi_future_lower_body", 240, [this](std::vector<double>& buf, size_t offset) {
                return GatherMotionJointPositionsMultiFrame(buf, offset, 10, robot_future_step_, lower_body_joint_mujoco_order_in_isaaclab_index)
                    && GatherMotionJointVelocitiesMultiFrame(buf, offset + 120, 10, robot_future_step_, lower_body_joint_mujoco_order_in_isaaclab_index);
              }},
              {"motion_anchor_ori_b", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1); }},
              {"motion_anchor_ori_heading", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1, 1); }},
              // Heading-only variants (mode=1): use robot yaw-only quaternion
              // Matches Python heading setting: quat_mul(quat_inv(get_heading_q(robot_pelvis_quat)), ref_quat)
              {"motion_anchor_orientation_heading", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1, 1); }},
              {"motion_anchor_orientation_heading_10frame_step5", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 5, 1); }},
              {"motion_anchor_orientation_heading_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1, 1); }},
              // Refheading variants (mode=2): use reference first-frame yaw-only quaternion (no robot state)
              // Matches Python refheading setting: quat_mul(quat_inv(get_heading_q(ref_first_frame)), ref_quat)
              {"motion_anchor_orientation_refheading", 6, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 1, 1, 2); }},
              {"motion_anchor_orientation_refheading_10frame_step5", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 5, 2); }},
              {"motion_anchor_orientation_refheading_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1, 2); }},
              {"motion_joint_positions_10frame_step5", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 5); }},
              {"motion_joint_velocities_10frame_step5", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 5); }},
              {"motion_joint_positions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1); }},
              {"motion_joint_velocities_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1); }},
              {"motion_joint_positions_3frame_step1", 87, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 3, 1); }},
              {"motion_joint_velocities_3frame_step1", 87, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 3, 1); }},
              {"motion_joint_positions_lowerbody_10frame_step5", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 5, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_velocities_lowerbody_10frame_step5", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 5, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_positions_lowerbody_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_velocities_lowerbody_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1, lower_body_joint_mujoco_order_in_isaaclab_index); }},
              {"motion_joint_positions_wrists_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"joint_pos_multi_future_wrist_for_smpl", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 10, smpl_future_step_, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_positions_wrists_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 2, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_velocities_wrists_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 10, 1, wrist_joint_isaaclab_order_in_isaaclab_index); }},
              {"motion_joint_positions_5frame_step5", 145, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointPositionsMultiFrame(buf, offset, 5, 5); }},
              {"motion_joint_velocities_5frame_step5", 145, [this](std::vector<double>& buf, size_t offset) { return GatherMotionJointVelocitiesMultiFrame(buf, offset, 5, 5); }},
              // SMPL data gathering (dimensions assume 24 joints, 21 poses - adjust based on actual data)
              {"smpl_joints", 72, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 1, 1); }},  // 24*3
              {"smpl_joints_5frame_step5", 360, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 5, 5); }},  // 24*3*5
              {"smpl_joints_10frame_step5", 720, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 10, 5); }},  // 24*3*10
              {"smpl_joints_10frame_step1", 720, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsLocalMultiFrame(buf, offset, 10, 1); }},  // 24*3*10
              {"smpl_joints_multi_future_local_nonflat", 720, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsLocalMultiFrame(buf, offset, 10, smpl_future_step_); }},
              {"smpl_joints_lower_10frame_step1", 270, [this](std::vector<double>& buf, size_t offset) {return GatherMotionSmplJointsMultiFrame(buf, offset, 10, 1, {0,1,2,4,5,7,8,10,11}); }},  // 9*3*10 lower body joints
              {"smpl_joints_2frame_step1", 144, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplJointsMultiFrame(buf, offset, 2, 1); }},  // 24*3*10
              {"smpl_pose", 63, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 1, 1); }},  // 21*3
              {"smpl_pose_5frame_step5", 315, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 5, 5); }},  // 21*3*5
              {"smpl_pose_10frame_step5", 630, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 5); }},  // 21*3*10
              {"smpl_pose_10frame_step1", 630, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 1); }},  // 21*3*10
              {"smpl_elbow_wrist_poses_10frame_step1", 120, [this](std::vector<double>& buf, size_t offset) { return GatherMotionSmplPosesMultiFrame(buf, offset, 10, 1, {17, 18, 19, 20}); }},  // 4*3*10
              {"smpl_root_z_10frame_step1", 10, [this](std::vector<double>& buf, size_t offset) { return GatherMotionRootZPositionMultiFrame(buf, offset, 10, 1); }},
              {"smpl_anchor_orientation_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1); }},
              {"smpl_root_ori_b_multi_future", 60, [this](std::vector<double>& buf, size_t offset) { return GatherSmplRootOrientationBodyMultiFrame(buf, offset, 10, smpl_future_step_); }},
              {"smpl_root_ori_heading_multi_future", 60, [this](std::vector<double>& buf, size_t offset) { return GatherSmplRootOrientationHeadingMultiFrame(buf, offset, 10, smpl_future_step_); }},
              {"smpl_anchor_orientation_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 2, 1); }},
              // SMPL heading-only variants (mode=1, matches Python smpl_root_ori_heading_multi_future)
              {"smpl_anchor_orientation_heading_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1, 1); }},
              {"smpl_anchor_orientation_heading_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 2, 1, 1); }},
              // SMPL refheading variants (mode=2, matches Python smpl_root_ori_refheading_multi_future)
              {"smpl_anchor_orientation_refheading_10frame_step1", 60, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 10, 1, 2); }},
              {"smpl_anchor_orientation_refheading_2frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherMotionAnchorOrientationMutiFrame(buf, offset, 2, 1, 2); }},
              // Heading diff: yaw from robot to reference first frame (refheading decoder input)
              {"heading_diff_robot_ref", 6, [this](std::vector<double>& buf, size_t offset) { return GatherHeadingDiffRobotRef(buf, offset); }},
              // VR 3-point data gathering
              {"vr_3point_local_target", 9, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointPosition(buf, offset); }},
              // NOTE: In the TELEOP time, the vr_3point_local_target == vr_3point_local_target_compliant
              {"vr_3point_local_target_compliant", 9, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointPosition(buf, offset); }},
              {"vr_3point_local_orn_target", 12, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointOrientation(buf, offset); }},
              {"vr_3point_compliance", 3, [this](std::vector<double>& buf, size_t offset) { return GatherVR3PointCompliance(buf, offset); }},
              // VR 5-point data gathering
              {"vr_5point_local_target", 15, [this](std::vector<double>& buf, size_t offset) { return GatherVR5PointPosition(buf, offset); }},
              {"vr_5point_local_orn_target", 20, [this](std::vector<double>& buf, size_t offset) { return GatherVR5PointOrientation(buf, offset); }},
              // History robot state gathering
              {"base_angular_velocity", 3, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 1, 1); }},
              {"body_joint_positions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 1, 1); }},
              {"body_joint_velocities", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 1, 1); }},
              {"last_actions", 29, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 1, 1); }},
              {"gravity_dir", 3, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 1, 1); }},
              {"his_body_joint_positions_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 4, 1); }},
              {"his_body_joint_velocities_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 4, 1); }},
              {"his_last_actions_4frame_step1", 116, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 4, 1); }},
              {"his_base_angular_velocity_4frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 4, 1); }},
              {"his_gravity_dir_4frame_step1", 12, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 4, 1); }},
              {"his_body_joint_positions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointPositions(buf, offset, 10, 1); }},
              {"his_body_joint_velocities_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisBodyJointVelocities(buf, offset, 10, 1); }},
              {"his_last_actions_10frame_step1", 290, [this](std::vector<double>& buf, size_t offset) { return GatherHisLastActions(buf, offset, 10, 1); }},
              {"his_base_angular_velocity_10frame_step1", 30, [this](std::vector<double>& buf, size_t offset) { return GatherHisBaseAngularVelocity(buf, offset, 10, 1); }},
              {"his_gravity_dir_10frame_step1", 30, [this](std::vector<double>& buf, size_t offset) { return GatherHisGravityDir(buf, offset, 10, 1); }}};
    }
    
    // Initialize observation functions
    void InitializeObservationFunctions() {
      // Get all available observations from registry (single source of truth!)
      auto registry = GetObservationRegistry();
      
      // =========================================================================
      // Initialize policy observations
      // =========================================================================
      active_obs_functions_.clear();
      size_t current_offset = 0;
      
      for (const auto& config : obs_config_) {
        if (!config.enabled) continue;
        
        // Find observation in registry
        auto registry_it = std::find_if(registry.begin(), registry.end(),
          [&config](const ObservationRegistry& obs) {
            return obs.name == config.name;
          });
        
        if (registry_it == registry.end()) {
          std::cerr << "✗ Error: Unknown observation function '" << config.name << "' in configuration!" << std::endl;
          throw std::runtime_error("Invalid observation configuration: unknown function '" + config.name + "'");
        }
        
        // Get dimension directly from registry entry
        size_t dimension = registry_it->dimension;
        if (dimension == 0) {
          std::cerr << "✗ Error: Observation function '" << config.name << "' has invalid dimension!" << std::endl;
          throw std::runtime_error("Invalid observation configuration: zero dimension for '" + config.name + "'");
        }
        
        // Add to active functions (get function directly from registry)
        active_obs_functions_.emplace_back(config.name, registry_it->function, current_offset, dimension);
        current_offset += dimension;
      }
      
      // Validate total dimension matches model input
      size_t model_input_dim = policy_engine_->GetInputDimension();
      if (current_offset != model_input_dim) {
        std::cerr << "✗ Error: Total observation dimension (" << current_offset 
                  << ") doesn't match model input dimension (" << model_input_dim << ")!" << std::endl;
        throw std::runtime_error("Observation dimension mismatch");
      }

      ValidateRobotSpecificObservationLayout();
      
      // =========================================================================
      // Initialize encoder observations (only if encoder is actually loaded)
      // =========================================================================
      active_encoder_obs_functions_.clear();
      if (encoder_engine_ && encoder_engine_->IsInitialized() && encoder_config_.dimension > 0 && !encoder_config_.encoder_observations.empty()) {
        std::cout << "Initializing encoder observations..." << std::endl;
        size_t encoder_offset = 0;
        size_t encoder_input_size = encoder_engine_->GetInputDimension();
        
        for (const auto& config : encoder_config_.encoder_observations) {
          if (!config.enabled) continue;
          
          // Find observation in registry
          auto registry_it = std::find_if(registry.begin(), registry.end(),
            [&config](const ObservationRegistry& obs) {
              return obs.name == config.name;
            });
          
          if (registry_it == registry.end()) {
            std::cerr << "✗ Error: Unknown encoder observation function '" << config.name << "'!" << std::endl;
            throw std::runtime_error("Invalid encoder observation: unknown function '" + config.name + "'");
          }
          
          size_t dimension = registry_it->dimension;
          if (dimension == 0) {
            std::cerr << "✗ Error: Encoder observation '" << config.name << "' has invalid dimension!" << std::endl;
            throw std::runtime_error("Invalid encoder observation: zero dimension for '" + config.name + "'");
          }
          
          // Add to active encoder functions
          active_encoder_obs_functions_.emplace_back(config.name, registry_it->function, encoder_offset, dimension);
          encoder_offset += dimension;
        }
        
        // Validate encoder input dimension matches expected
        if (encoder_offset != encoder_input_size) {
          std::ostringstream oss;
          oss << "Encoder observation dimension mismatch: got " << encoder_offset
              << ", expected " << encoder_input_size;
          throw std::runtime_error(oss.str());
        }
        ValidateRobotSpecificEncoderObservationLayout(encoder_offset);
        
        std::cout << "✓ Initialized " << active_encoder_obs_functions_.size() 
                  << " encoder observations (total dim: " << encoder_offset << ")" << std::endl;
      }
      
      // =========================================================================
      // Check if vr_3point_compliance observation is enabled (in main or encoder obs)
      // This affects whether keyboard compliance controls (g/h/b/v) have any effect
      // =========================================================================
      has_vr_3point_compliance_obs_ = false;
      
      // Check main observations
      for (const auto& obs : active_obs_functions_) {
        if (obs.name == "vr_3point_compliance") {
          has_vr_3point_compliance_obs_ = true;
          break;
        }
      }
      
      // Check encoder observations if not found in main
      if (!has_vr_3point_compliance_obs_) {
        for (const auto& obs : active_encoder_obs_functions_) {
          if (obs.name == "vr_3point_compliance") {
            has_vr_3point_compliance_obs_ = true;
            break;
          }
        }
      }
    }

    void ValidateRobotSpecificObservationLayout() const {
      if (!use_h2_robot_) {
        return;
      }

      const std::string profile_label = use_h2_native_ ? "H2-native" : "H2LoRA";

      const std::vector<std::pair<std::string, size_t>> expected = {
          {"token_state", 64},
          {"his_base_angular_velocity_10frame_step1", 30},
          {"his_body_joint_positions_10frame_step1", 290},
          {"his_body_joint_velocities_10frame_step1", 290},
          {"his_last_actions_10frame_step1", 290},
          {"his_gravity_dir_10frame_step1", 30},
      };

      if (active_obs_functions_.size() != expected.size()) {
        std::ostringstream oss;
        oss << profile_label << " policy observation layout mismatch: got "
            << active_obs_functions_.size() << " terms, expected "
            << expected.size()
            << ". The decoder ONNX expects token_state followed by the 930D "
               "actor_obs slice in training order.";
        throw std::runtime_error(oss.str());
      }

      for (size_t i = 0; i < expected.size(); ++i) {
        const auto& obs = active_obs_functions_[i];
        const auto& [expected_name, expected_dim] = expected[i];
        if (obs.name != expected_name || obs.dimension != expected_dim) {
          std::ostringstream oss;
          oss << profile_label << " policy observation layout mismatch at term " << i
              << ": got '" << obs.name << "' (" << obs.dimension << "D), expected '"
              << expected_name << "' (" << expected_dim
              << "D). Keep deploy obs order identical to "
                 "the exported ONNX actor_obs order: base_ang_vel, joint_pos, "
                 "joint_vel, actions, gravity_dir.";
          throw std::runtime_error(oss.str());
        }
      }

      if (policy_engine_->GetInputDimension() != 994) {
        std::ostringstream oss;
        oss << profile_label << " decoder input dimension mismatch: got "
            << policy_engine_->GetInputDimension()
            << ", expected 994 (64 token + 930 actor_obs).";
        throw std::runtime_error(oss.str());
      }
    }

    void ValidateRobotSpecificEncoderObservationLayout(size_t encoder_input_size) const {
      if (!use_h2_robot_) {
        return;
      }

      const bool heading_contract = IsH2NativeV11Profile(robot_profile_);
      if (encoder_input_size == 640) {
        if (!use_h2_native_ || heading_contract) {
          throw std::runtime_error(
              "The 640D H2-only encoder is supported only by the non-v1.1 "
              "h2_native profile.");
        }
        const std::vector<std::pair<std::string, size_t>> h2_only_expected = {
            {"h2_native_reference", 580},
            {"motion_anchor_ori_b_mf_nonflat", 60},
        };
        if (active_encoder_obs_functions_.size() != h2_only_expected.size()) {
          throw std::runtime_error(
              "H2-only encoder observation layout must contain exactly "
              "h2_native_reference and motion_anchor_ori_b_mf_nonflat.");
        }
        for (size_t i = 0; i < h2_only_expected.size(); ++i) {
          const auto& obs = active_encoder_obs_functions_[i];
          const auto& [expected_name, expected_dim] = h2_only_expected[i];
          if (obs.name != expected_name || obs.dimension != expected_dim) {
            std::ostringstream oss;
            oss << "H2-only encoder observation layout mismatch at term " << i
                << ": got '" << obs.name << "' (" << obs.dimension
                << "D), expected '" << expected_name << "' ("
                << expected_dim << "D).";
            throw std::runtime_error(oss.str());
          }
        }
        return;
      }

      const std::vector<std::pair<std::string, size_t>> expected = {
          {"encoder_index", 4},
          {use_h2_native_ ? "h2_native_reference" : "command_multi_future_nonflat", 580},
          {heading_contract ? "motion_anchor_ori_heading" : "motion_anchor_ori_b", 6},
          {heading_contract ? "motion_anchor_ori_heading_mf_nonflat" : "motion_anchor_ori_b_mf_nonflat", 60},
          {"command_multi_future_lower_body", 240},
          {"vr_3point_local_target", 9},
          {"vr_3point_local_orn_target", 12},
          {"smpl_joints_multi_future_local_nonflat", 720},
          {heading_contract ? "smpl_root_ori_heading_multi_future" : "smpl_root_ori_b_multi_future", 60},
          {"joint_pos_multi_future_wrist_for_smpl", 60},
      };

      if (active_encoder_obs_functions_.size() != expected.size()) {
        std::ostringstream oss;
        oss << (use_h2_native_ ? "H2-native" : "H2LoRA")
            << " encoder observation layout mismatch: got "
            << active_encoder_obs_functions_.size() << " terms, expected "
            << expected.size()
            << ". The encoder ONNX input must follow the universal SONIC selector order.";
        throw std::runtime_error(oss.str());
      }

      for (size_t i = 0; i < expected.size(); ++i) {
        const auto& obs = active_encoder_obs_functions_[i];
        const auto& [expected_name, expected_dim] = expected[i];
        if (obs.name != expected_name || obs.dimension != expected_dim) {
          std::ostringstream oss;
          oss << (use_h2_native_ ? "H2-native" : "H2LoRA")
              << " encoder observation layout mismatch at term " << i
              << ": got '" << obs.name << "' (" << obs.dimension << "D), expected '"
              << expected_name << "' (" << expected_dim
              << "D). Keep deploy encoder obs order identical to the exported ONNX contract.";
          throw std::runtime_error(oss.str());
        }
      }

      if (encoder_input_size != 1751) {
        std::ostringstream oss;
        oss << (use_h2_native_ ? "H2-native" : "H2LoRA")
            << " encoder input dimension mismatch: got " << encoder_input_size
            << ", expected 1751 (1 encoder selector + 1750 tokenizer obs).";
        throw std::runtime_error(oss.str());
      }
    }

    // Log observation configuration
    void LogObservationConfiguration() {
      std::cout << "\n=== Observation Configuration ===" << std::endl;
      std::cout << "Active policy observations (runtime order):" << std::endl;
      
      size_t total_enabled_dim = 0;
      
      // Show only active observations (clean and focused)
      for (size_t i = 0; i < active_obs_functions_.size(); i++) {
        const auto& active_obs = active_obs_functions_[i];
        
        std::cout << "  " << (i + 1) << ". " 
                  << active_obs.name << " (" << active_obs.dimension << "D)"
                  << " [offset: " << active_obs.offset << "]" << std::endl;
        
        total_enabled_dim += active_obs.dimension;
      }
      
      std::cout << "\nPolicy Summary:" << std::endl;
      std::cout << "  Total dimension: " << total_enabled_dim << std::endl;
      size_t model_input_dim = policy_engine_->GetInputDimension();
      std::cout << "  Model dimension: " << model_input_dim << std::endl;
      
      if (total_enabled_dim == model_input_dim) {
        std::cout << "  ✓ Dimension match: Configuration is valid!" << std::endl;
      } else {
        std::ostringstream oss;
        oss << "Policy observation dimension mismatch: total_enabled_dim=" << total_enabled_dim
            << ", model_input_dim=" << model_input_dim
            << ", difference=" << (int64_t)model_input_dim - (int64_t)total_enabled_dim;
        throw std::runtime_error(oss.str());
      }
      
      // Show encoder observations if enabled
      if (!active_encoder_obs_functions_.empty()) {
        std::cout << "\n--- Encoder Observations ---" << std::endl;
        size_t total_encoder_dim = 0;
        
        for (size_t i = 0; i < active_encoder_obs_functions_.size(); i++) {
          const auto& active_obs = active_encoder_obs_functions_[i];
          
          std::cout << "  " << (i + 1) << ". " 
                    << active_obs.name << " (" << active_obs.dimension << "D)"
                    << " [offset: " << active_obs.offset << "]" << std::endl;
          
          total_encoder_dim += active_obs.dimension;
        }
        
        std::cout << "\nEncoder Summary:" << std::endl;
        std::cout << "  Total dimension: " << total_encoder_dim << std::endl;
        if (encoder_engine_ && encoder_engine_->IsInitialized()) {
          size_t encoder_input_size = encoder_engine_->GetInputDimension();
          std::cout << "  Model dimension: " << encoder_input_size << std::endl;
          
          if (total_encoder_dim == encoder_input_size) {
            std::cout << "  ✓ Dimension match: Encoder configuration is valid!" << std::endl;
          } else {
            std::ostringstream oss;
            oss << "Encoder configuration dimension mismatch: total_encoder_dim=" << total_encoder_dim
                << ", encoder_model_input_dim=" << encoder_input_size;
            throw std::runtime_error(oss.str());
          }
        }
      }
      
      std::cout << "================================\n" << std::endl;
    }
    
    // Gather observations (simplified - functions read from internal state)
    bool GatherObservations() {
      // Clear observation buffer
      std::fill(obs_buffer_.begin(), obs_buffer_.end(), 0.0);
      
      // Process observations using pre-built active functions (no map lookup needed!)
      for (const auto& active_obs : active_obs_functions_) {
        bool success = active_obs.function(obs_buffer_, active_obs.offset);
        
        if (!success) {
          std::cout << "Error: Failed to gather observation: " << active_obs.name << std::endl;
          return false;
        }
      }

      RefreshH2LoraLatestActionHistory();
      
      return true;
    }

    void RefreshH2LoraLatestActionHistory() {
      if (!use_h2_lora_) {
        return;
      }

      const auto action_history = std::find_if(
          active_obs_functions_.begin(),
          active_obs_functions_.end(),
          [](const ActiveObservation& observation) {
            return observation.name == "his_last_actions_10frame_step1";
          });
      if (action_history == active_obs_functions_.end()) {
        throw std::runtime_error(
            "H2LoRA action history observation is not active");
      }

      // Match the corrected training wrapper: replace only the newest frame
      // inside the action history and preserve the following gravity history.
      h2_lora_observation::ReplaceNewestHistoryFrame(
          std::span<double>(obs_buffer_),
          action_history->offset,
          action_history->dimension,
          std::span<const double>(last_action));
    }

    // Gather encoder observations (populate encoder input buffer)
    // Observation functions read from internal state (state_logger_, current_motion_, etc.)
    // Mode-specific optimization: Only compute observations required for current encoder mode
    bool GatherEncoderObservations() {
      if (!is_using_encoder_ || active_encoder_obs_functions_.empty()) {
        return true;  // Nothing to do
      }

      // Get reference to encoder's internal input buffer
      auto& encoder_input_buffer = encoder_engine_->GetInputBuffer();
      
      // Clear encoder observation buffer (safe default for unused observations)
      std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0.0);
      
      // Store the intended encoder mode for warning purposes
      int intended_encoder_mode = current_motion_->GetEncodeMode();
      
      // Build list of modes to try (start with current mode, then all others as fallback)
      std::vector<int> modes_to_try;
      
      if (!encoder_config_.encoder_modes.empty()) {
        // Start with intended mode if valid
        if (current_motion_->GetEncodeMode() >= 0) {
          modes_to_try.push_back(current_motion_->GetEncodeMode());
        }
        
        // Add all other modes as fallbacks
        for (const auto& mode_config : encoder_config_.encoder_modes) {
          if (mode_config.mode_id != current_motion_->GetEncodeMode()) {
            modes_to_try.push_back(mode_config.mode_id);
          }
        }
      }
      
      // If no modes configured, try without mode filter
      if (modes_to_try.empty()) {
        modes_to_try.push_back(-1);  // -1 means no mode filter (all observations required)
      }
      
      // Try each mode until one succeeds
      for (size_t attempt = 0; attempt < modes_to_try.size(); ++attempt) {
        int mode_to_try = modes_to_try[attempt];
        current_motion_->SetEncodeMode(mode_to_try);
        
        // Determine which observations to gather based on encoder mode
        std::vector<std::string> required_observations;
        bool use_mode_filter = false;
        std::string mode_name;
        
        // Check if mode-specific filtering is configured
        if (!encoder_config_.encoder_modes.empty() && current_motion_->GetEncodeMode() >= 0) {
          // Find the current mode configuration
          for (const auto& mode_config : encoder_config_.encoder_modes) {
            if (mode_config.mode_id == current_motion_->GetEncodeMode()) {
              required_observations = mode_config.required_observations;
              use_mode_filter = true;
              mode_name = mode_config.name;
              
              // Only log when mode changes to avoid spam
              if (current_motion_->GetEncodeMode() != last_logged_encoder_mode_) {
                std::cout << "[Mode Filter] Switched to mode '" << mode_config.name << "' (ID=" << current_motion_->GetEncodeMode() 
                          << ") with " << required_observations.size() << " required observations" << std::endl;
                last_logged_encoder_mode_ = current_motion_->GetEncodeMode();
              }
              break;
            }
          }
        }
        
        // Process encoder observations using pre-built active functions
        bool all_success = true;
        for (const auto& active_obs : active_encoder_obs_functions_) {
          const bool required =
              !use_mode_filter ||
              std::find(required_observations.begin(), required_observations.end(),
                        active_obs.name) != required_observations.end();
          if (!required) {
            // The exported universal-token ONNX input is a full tokenizer
            // vector.  The selected encoder branch is controlled by
            // encoder_index, so inactive branch slices should remain at the
            // zero-filled default instead of forcing branch-specific reference
            // data (e.g. SMPL joints for a G1 motion).
            continue;
          }

          bool success = active_obs.function(encoder_obs_buffer_, active_obs.offset);
          
          if (!success) {
            std::cerr << "✗ Error: Failed to gather encoder observation: "
                      << active_obs.name
                      << " for mode " << current_motion_->GetEncodeMode()
                      << (!mode_name.empty() ? " ('" + mode_name + "')" : "") << std::endl;
            all_success = false;
            break;
          }
        }
        
        // Check if this mode succeeded
        if (all_success) {
          // Success! Copy encoder observations to encoder input buffer (double -> float)
          for (size_t i = 0; i < encoder_input_buffer.size() && i < encoder_obs_buffer_.size(); ++i) {
            encoder_input_buffer[i] = static_cast<float>(encoder_obs_buffer_[i]);
          }

          if (encoder_input_file_) {
            for (const double d : encoder_obs_buffer_) {
              (*encoder_input_file_) << d << ",";
            }
            (*encoder_input_file_) << std::endl;
          }
          
          // Warn if we had to switch from intended mode
          if (current_motion_->GetEncodeMode() != intended_encoder_mode && attempt > 0) {
            std::cerr << "⚠ Warning: Intended encoder mode " << intended_encoder_mode 
                      << " failed, successfully switched to fallback mode " << current_motion_->GetEncodeMode() 
                      << (!mode_name.empty() ? " ('" + mode_name + "')" : "") << std::endl;
          }
          
          return true;
        } else {
          // This mode failed - warn if it was the intended mode
          if (current_motion_->GetEncodeMode() == intended_encoder_mode && attempt == 0) {
            std::cerr << "⚠ Warning: Intended encoder mode " << intended_encoder_mode 
                      << (!mode_name.empty() ? " ('" + mode_name + "')" : "")
                      << " failed, trying fallback modes..." << std::endl;
          }
          
          // Clear buffer and try next mode
          std::fill(encoder_obs_buffer_.begin(), encoder_obs_buffer_.end(), 0.0);
        }
      }
      
      // All modes failed
      std::cerr << "✗ Error: All available encoder modes failed to gather observations" << std::endl;
      return false;
    }



  public:
    OperatorState operator_state;

    G1Deploy(
      std::string networkInterface,
      std::string model_file_path,
      std::string motion_data_path,
      bool disable_crc_check = false,
      std::string obs_config_path = "",
      std::string encoder_file_path = "",
      std::string planner_file_path = "",
      std::string target_motion_file_path = "",
      std::string planner_motion_file_path = "",
      std::string policy_input_file_path = "",
      std::string input_type = "keyboard",
      std::string output_type = "zmq",
      std::string record_input_file_path = "",
      std::string playback_input_file_path = "",
      bool planner_fp16 = false,
      bool policy_fp16 = false,
      std::string logs_dir = "",
      bool enable_csv_logs = false,
      std::string zmq_host = "localhost",
      int zmq_port = 5556,
      std::string zmq_topic = "pose",
      bool zmq_conflate = false,
      bool zmq_verbose = false,
      int zmq_out_port = 5557,
      std::string zmq_out_topic = "g1_debug",
      bool enable_motion_recording = false,
      std::array<double, 3> initial_compliance = {0.05, 0.05, 0.0},
      double initial_max_close_ratio = 1.0,
      std::string robot_profile = "g1",
      bool enable_h2_vr_head_control = false,
      HandBackendKind hand_backend = HandBackendKind::DEX3,
      bool project_h2_native_planner_reference = false,
      H2GainScales h2_gain_scales = {})
      : time_(0.0),
        publish_dt_(0.002),
        control_dt_(0.02),
        planner_dt_(0.1),
        input_dt_(0.01),
        duration_(3.0),
        counter_(0),
        mode_pr_(Mode::PR),
        mode_machine_(0),
        disable_crc_check_(disable_crc_check),
        robot_profile_(std::move(robot_profile)),
        h2_gain_scales_(h2_gain_scales),
        use_h2_lora_(IsH2LoraProfile(robot_profile_)),
        use_h2_native_(IsH2NativeProfile(robot_profile_)),
        use_h2_robot_(IsH2Profile(robot_profile_)),
        project_h2_native_planner_reference_(project_h2_native_planner_reference),
        enable_h2_vr_head_control_(enable_h2_vr_head_control),
        hand_backend_(hand_backend),
        robot_motor_count_(ProfileMotorCount(robot_profile_)),
        program_state_(ProgramState::INIT),
        wall_start_time_(std::chrono::steady_clock::now()),
        last_action {0.0},
        last_left_hand_action {0.0},
        last_right_hand_action {0.0},
        enable_motion_recording_(enable_motion_recording),
        initial_vr_3point_compliance_(initial_compliance),
        initial_max_close_ratio_(initial_max_close_ratio),
        //env(ORT_LOGGING_LEVEL_WARNING, "G1Deploy"),
        model_path(model_file_path),
        planner_path(planner_file_path) {

      h2_gain_scales_.Validate();
      if (!h2_gain_scales_.IsDefault() && robot_profile_ != "h2_native") {
        throw std::invalid_argument("H2 gain scaling requires --robot-profile h2_native");
      }
      if (project_h2_native_planner_reference_ &&
          (!use_h2_native_ || IsH2NativeV11Profile(robot_profile_) ||
           planner_path.empty() || encoder_file_path.empty())) {
        throw std::runtime_error(
            "--planner-g1-to-h2 requires the h2_native release profile, "
            "a planner file and a full encoder file");
      }

	      std::cout << "[Robot Profile] " << robot_profile_
	                << " (policy_actions=" << POLICY_ACTION_DIM
	                << ", command_motors=" << robot_motor_count_ << ")" << std::endl;

      h2_lora_mujoco_sim_ =
          use_h2_lora_ && disable_crc_check_ && (output_type == "zmq" || output_type == "all");
      policy_action_scale_default_ =
          h2_lora_mujoco_sim_ ? kH2LoraMujocoSimPolicyActionScale : 1.0;
      policy_action_scale_ = policy_action_scale_default_;
      if (const char* env = std::getenv("GEAR_SONIC_POLICY_ACTION_SCALE");
          env != nullptr && std::string(env).empty() == false) {
        try {
          policy_action_scale_ = std::stod(std::string(env));
        } catch (...) {
          std::cerr << "⚠ Warning: invalid GEAR_SONIC_POLICY_ACTION_SCALE='"
                    << env << "', using " << policy_action_scale_default_ << std::endl;
          policy_action_scale_ = policy_action_scale_default_;
        }
      }
      if (use_h2_lora_) {
        if (const char* env = std::getenv("GEAR_SONIC_H2LORA_PROJECT_PLANNER_REFERENCE");
            env != nullptr && std::string(env).empty() == false) {
          std::string value(env);
          std::transform(value.begin(), value.end(), value.begin(), ::tolower);
          project_h2_lora_planner_reference_ =
              !(value == "0" || value == "false" || value == "off" || value == "no");
        }
        std::cout << "[H2LoRA] raw policy action scale: " << policy_action_scale_
                  << " (default=" << policy_action_scale_default_
                  << ", override with GEAR_SONIC_POLICY_ACTION_SCALE)" << std::endl;
        std::cout << "[H2LoRA] MuJoCo sim mode: "
                  << (h2_lora_mujoco_sim_ ? "true" : "false") << std::endl;
        std::cout << "[H2LoRA Planner] Phi(Phi+) reference projection: "
                  << (project_h2_lora_planner_reference_ ? "enabled" : "disabled")
                  << " (override with GEAR_SONIC_H2LORA_PROJECT_PLANNER_REFERENCE)"
                  << std::endl;
        std::cout << "[H2 Head] VR pitch/yaw control: "
                  << (enable_h2_vr_head_control_ ? "enabled" : "disabled")
                  << std::endl;
      } else if (use_h2_native_) {
        std::cout << "[H2-native] actuator profile: "
                  << H2_NATIVE_ACTUATOR_PROFILE << " (Unitree commit "
                  << H2_NATIVE_ACTUATOR_REFERENCE_COMMIT << ")" << std::endl;
        std::cout << "[H2-native] direct H2 reference/proprioception/action path enabled"
                  << std::endl;
        std::cout << "[H2 Kp] scales: waist_pitch=" << h2_gain_scales_.waist_pitch_kp
                  << ", waist_roll=" << h2_gain_scales_.waist_roll_kp
                  << ", ankle_pitch=" << h2_gain_scales_.ankle_pitch_kp
                  << ", ankle_roll=" << h2_gain_scales_.ankle_roll_kp << std::endl;
        std::cout << "[H2 Kd] ankle_pitch scale=" << h2_gain_scales_.ankle_pitch_kd
                  << " (other joints and damping-only commands unchanged)" << std::endl;
        for (const int dds_index : {14, 13, 5, 11, 3, 9, 4, 10}) {
          std::cout << "[H2 Gains] " << h2_joint_names_dds[dds_index] << ": Kp "
                    << H2NativeKpDds(dds_index) << " -> "
                    << h2_gain_scales_.ApplyKp(dds_index, H2NativeKpDds(dds_index))
                    << ", Kd " << H2NativeKdDds(dds_index) << " -> "
                    << h2_gain_scales_.ApplyKd(dds_index, H2NativeKdDds(dds_index))
                    << std::endl;
        }
        std::cout << "[H2-native] reference orientation contract: "
                  << (IsH2NativeV11Profile(robot_profile_)
                          ? "sonic_robot_heading_v1"
                          : "sonic_full_root_relative_v0")
                  << std::endl;
        std::cout << "[H2-native Planner] G1->H2 reference projection: "
                  << (project_h2_native_planner_reference_ ? "enabled (comparison mode)"
                                                          : "disabled")
                  << "; branch is selected after encoder shape validation" << std::endl;
        std::cout << "[H2 Head] reference control: enabled; VR override: "
                  << (enable_h2_vr_head_control_ ? "enabled" : "disabled")
                  << std::endl;
      } else if (enable_h2_vr_head_control_) {
        throw std::runtime_error(
            "--h2-vr-head-control requires an H2 robot profile");
      }
      int domain_id = 0;
      if (const char* domain_env = std::getenv("GEAR_SONIC_DOMAIN_ID");
          domain_env != nullptr && std::string(domain_env).empty() == false) {
        try {
          domain_id = std::stoi(std::string(domain_env));
        } catch (...) {
          throw std::runtime_error(
              "Invalid GEAR_SONIC_DOMAIN_ID='" + std::string(domain_env) + "'");
        }
      }
      std::cout << "[DDS] ChannelFactory domain_id=" << domain_id
                << ", interface=" << networkInterface << std::endl;
      ChannelFactory::Instance()->Init(domain_id, networkInterface);

      if (hand_backend_ == HandBackendKind::DEX3) {
        dex3_hands_.initialize("");
        std::cout << "[Hand Backend] Dex3 DDS enabled" << std::endl;
      } else if (hand_backend_ == HandBackendKind::INSPIRE) {
        inspire_hands_.initialize("");
        std::cout << "[Hand Backend] Inspire DDS enabled (rt/inspire)" << std::endl;
        std::cout << "[Hand Backend] Release both Triggers once to arm hand commands"
                  << std::endl;
      } else {
        std::cout << "[Hand Backend] Hand control disabled" << std::endl;
      }

      audio_thread_ = std::make_unique<AudioThread>();

      if(!target_motion_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(target_motion_file_path);
        f.close();

        // open file in append mode:
        target_motion_file_ = std::make_unique<std::ofstream>(target_motion_file_path, std::ios::app);
      }

      if(!planner_motion_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(planner_motion_file_path);
        f.close();

        // open file in append mode:
        planner_motion_file_ = std::make_unique<std::ofstream>(planner_motion_file_path, std::ios::app);
      }

      if(!policy_input_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(policy_input_file_path);
        f.close();

        // open file in append mode:
        policy_input_file_ = std::make_unique<std::ofstream>(policy_input_file_path, std::ios::app);
      }

      if(!record_input_file_path.empty())
      {
        // clear existing file:
        std::ofstream f(record_input_file_path);
        f.close();

        // open file in append mode:
        record_input_file_ = std::make_unique<std::ofstream>(record_input_file_path, std::ios::app);
        std::cout << "Initialized record input file: " << record_input_file_path << std::endl;
      }

      if(!playback_input_file_path.empty())
      {
        // open file in append mode:
        playback_input_file_ = std::make_unique<std::ifstream>(playback_input_file_path);
        std::cout << "Initialized playback input file: " << playback_input_file_path << std::endl;
      }

      movement_state_buffer_.SetData(MovementState());

      // Initialize planner motion state as shared_ptr
      planner_motion_ = std::make_shared<MotionSequence>();
      planner_motion_->ReserveCapacity(1500, 29, 1, 1, 0, 0);
      planner_motion_->timesteps = 0;
      planner_motion_->name = "planner_motion";
      const bool skip_motion_switcher =
          disable_crc_check_ ||
          (std::getenv("GEAR_SONIC_SKIP_MOTION_SWITCHER") != nullptr &&
           std::string(std::getenv("GEAR_SONIC_SKIP_MOTION_SWITCHER")) != "0" &&
           std::string(std::getenv("GEAR_SONIC_SKIP_MOTION_SWITCHER")) != "false");
      if (skip_motion_switcher) {
        std::cout << "[INFO] Skipping MotionSwitcherClient initialization for simulation/diagnostics" << std::endl;
      } else {
        // Try to shutdown motion control-related service on real hardware.
        msc_ = std::make_unique<unitree::robot::b2::MotionSwitcherClient>();
        msc_->SetTimeout(5.0f);
        msc_->Init();
        std::string form, name;
        while (msc_->CheckMode(form, name), !name.empty()) {
          if (msc_->ReleaseMode()) std::cout << "Failed to switch to Release Mode\n";
          sleep(5);
        }
      }

      // create publisher
      lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
      lowcmd_publisher_->InitChannel();
      // create subscriber
      lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
      lowstate_subscriber_->InitChannel(std::bind(&G1Deploy::LowStateHandler, this, std::placeholders::_1), 1);
      imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
      imutorso_subscriber_->InitChannel(std::bind(&G1Deploy::imuTorsoHandler, this, std::placeholders::_1), 1);
      // Load motion data
      if (use_h2_robot_) {
        const std::filesystem::path h2_manifest =
            std::filesystem::path(motion_data_path) / "h2_lora_reference_manifest.json";
        if (!std::filesystem::exists(h2_manifest)) {
          std::cout << "✗ Error: H2 deploy requires H2 MotionLib-exported reference data." << std::endl;
          std::cout << "Expected manifest: " << h2_manifest << std::endl;
          std::cout << "Do not use the G1 reference/example directory with an H2 profile." << std::endl;
          throw std::runtime_error("Missing H2 reference manifest");
        }
      }
      if (motion_reader_.ReadFromCSV(motion_data_path)) {
        if (!motion_reader_.motions.empty()) {
          std::cout << "✓ Motion data loaded successfully!" << std::endl;
          // motion_reader_.PrintSummary();
          motion_reader_.current_motion_index_ = 0;
          if (const char* initial_motion_name = std::getenv("GEAR_SONIC_INITIAL_MOTION_NAME");
              initial_motion_name != nullptr && initial_motion_name[0] != '\0') {
            const int requested_index = motion_reader_.FindMotionIndexByName(initial_motion_name);
            if (requested_index < 0) {
              std::cout << "✗ Error: Requested initial motion not found: " << initial_motion_name << std::endl;
              std::cout << "Available motions:" << std::endl;
              for (size_t i = 0; i < motion_reader_.motions.size(); ++i) {
                std::cout << "  [" << i << "] " << motion_reader_.motions[i]->name << std::endl;
              }
              throw std::runtime_error("Requested initial motion not found");
            }
            motion_reader_.current_motion_index_ = requested_index;
            std::cout << "Using requested initial motion: " << initial_motion_name
                      << " (index " << requested_index << ")" << std::endl;
          }
          std::string motion_name;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
            current_frame_ = 0;
            motion_name = current_motion_->name;
            ConfigureReferenceFutureSteps(*current_motion_);
          }
          operator_state.play = false;
          std::cout << "Started with motion: " << motion_name << " (paused at frame 0)" << std::endl;
        } else {
          std::cout << "✗ Error: Motion directory found but no valid motions loaded from " << motion_data_path
                    << std::endl;
          std::cout << "Check that motion folders contain valid CSV files." << std::endl;
          std::cout << "Control system cannot function without motion data!" << std::endl;
          std::cout << "Exiting..." << std::endl;
          throw std::runtime_error("No valid motions found in motion data directory");
        }
      } else {
        std::cout << "✗ Error: Could not access motion data from " << motion_data_path << std::endl;
        std::cout << "Make sure the path exists and contains motion folders." << std::endl;
        std::cout << "Control system cannot function without motion data!" << std::endl;
        std::cout << "Exiting..." << std::endl;
        throw std::runtime_error("Failed to load motion data");
      }
      
      if (use_h2_native_) {
        h2_native_contract::ValidateDeploymentModels(
            encoder_file_path,
            model_path,
            IsH2NativeV11Profile(robot_profile_)
                ? "sonic_robot_heading_v1"
                : "sonic_full_root_relative_v0");
        std::cout << "✓ H2-native ONNX shapes and metadata validated" << std::endl;
      }

      // Initialize control policy
      policy_engine_ = std::make_unique<PolicyEngine>();
      
      if (!policy_engine_->Initialize(model_path, policy_fp16)) {
        throw std::runtime_error("Failed to initialize control policy from: " + model_path);
      }
      
      // Initialize observation buffer with correct size (zero-initialized)
      size_t obs_dim = policy_engine_->GetInputDimension();
      obs_buffer_.resize(obs_dim, 0.0);
      
      // Capture CUDA graph for optimized execution
      if (!policy_engine_->CaptureGraph()) {
        throw std::runtime_error("Failed to capture control policy CUDA graph");
      }
      
      std::cout << "✓ Policy model loaded successfully!" << std::endl;

      // Load observation configuration FIRST (before encoder/planner initialization)
      std::cout << "Loading observation configuration..." << std::endl;
      FullObservationConfig full_obs_config;
      if (!obs_config_path.empty()) {
        std::cout << "Using observation config file: " << obs_config_path << std::endl;
        full_obs_config = ObservationConfigParser::ParseFullConfig(obs_config_path);
      } else {
        std::cout << "Using default observation configuration" << std::endl;
        full_obs_config.observations = ObservationConfigParser::ParseConfig();
      }
      
      // Check if config parsing failed (empty observations returned)
      if (full_obs_config.observations.empty()) {
        throw std::runtime_error("Failed to parse observation configuration - check config file for errors");
      }
      
      obs_config_ = full_obs_config.observations;
      encoder_config_ = full_obs_config.encoder;
      
      // Initialize token buffer size from encoder config
      if (encoder_config_.dimension > 0) {
        token_state_data_.resize(encoder_config_.dimension, 0.0);
        std::cout << "✓ Token state buffer initialized (size: " << encoder_config_.dimension << ")" << std::endl;
      }
      
      // =========================================================================
      // Initialize encoder engine
      // =========================================================================
      // Note: Parser already validates that if token_state is enabled, encoder_config has dimension > 0
      // So if dimension <= 0 here, it means token_state is disabled - just ignore encoder_file_path
      if (!encoder_file_path.empty() && encoder_config_.dimension > 0) {
        std::cout << "Initializing encoder..." << std::endl;
        encoder_engine_ = std::make_unique<EncoderEngine>();
        
        if (!encoder_engine_->Initialize(encoder_file_path, encoder_config_.use_fp16)) {
          throw std::runtime_error("Failed to initialize encoder engine from: " + encoder_file_path);
        }
        
        // Validate encoder output dimension matches config
        if (encoder_engine_->GetTokenDimension() != static_cast<size_t>(encoder_config_.dimension)) {
          std::cerr << "⚠ Warning: Encoder model output dimension (" << encoder_engine_->GetTokenDimension() 
                    << ") doesn't match config dimension (" << encoder_config_.dimension << ")" << std::endl;
        }
        
        // Initialize encoder observation buffer with correct size
        size_t encoder_input_size = encoder_engine_->GetInputDimension();
        encoder_obs_buffer_.resize(encoder_input_size, 0.0);

        encoder_universal_full_ = encoder_input_size == 1751;
        if (project_h2_native_planner_reference_ && !encoder_universal_full_) {
          throw std::runtime_error(
              "--planner-g1-to-h2 requires the 1751D model_encoder_full.onnx");
        }
        planner_encoder_mode_ = 0;
        if (encoder_universal_full_) {
          const bool has_native_g1_mode = std::any_of(
              encoder_config_.encoder_modes.begin(), encoder_config_.encoder_modes.end(),
              [](const EncoderModeConfig& mode) { return mode.mode_id == 3; });
          if (!has_native_g1_mode) {
            throw std::runtime_error(
                "Full universal encoder requires encoder mode 3 for native G1 planner references");
          }
          planner_encoder_mode_ = project_h2_native_planner_reference_ ? 0 : 3;
          std::cout << "[Encoder] Full universal selector detected; planner references use "
                    << (project_h2_native_planner_reference_
                            ? "G1->H2 projection and H2 mode 0"
                            : "native G1 mode 3") << std::endl;
        }

        // Capture CUDA graph for optimized execution
        if (!encoder_engine_->CaptureGraph()) {
          throw std::runtime_error("Failed to capture encoder CUDA graph");
        }
        
        std::cout << "✓ Encoder model loaded successfully!" << std::endl;
        is_using_encoder_ = true;
        initial_encoder_mode_ = 0;  // Encoder available, default to mode 0.
      } else {
        if (encoder_config_.dimension > 0) {
          std::cout << "Encoder config found but no encoder file provided - tokens can be set externally" << std::endl;
          initial_encoder_mode_ = -1;
        } else {
          std::cout << "No encoder configured" << std::endl;
          initial_encoder_mode_ = -2;
        }
        is_using_encoder_ = false;
      }

      // =========================================================================
      // Initialize encode_mode for all loaded reference motions and planner motion
      // =========================================================================
      // Set the encode_mode for all loaded reference motions based on encoder availability
      std::cout << "Setting encode_mode for " << motion_reader_.motions.size() << " loaded reference motions..." << std::endl;
      const auto reference_type_name = [](int mode) {
        switch (mode) {
          case 0: return "h2";
          case 1: return "teleop";
          case 2: return "smpl";
          case 3: return "g1";
          default: return "unknown";
        }
      };
      for (auto& motion : motion_reader_.motions) {
        int motion_encode_mode = initial_encoder_mode_;
        if (initial_encoder_mode_ >= 0 && motion->HasPreferredEncodeMode()) {
          motion_encode_mode = motion->GetPreferredEncodeMode();
        }
        if (use_h2_native_ && !encoder_universal_full_ && motion_encode_mode != 0) {
          throw std::runtime_error(
              "The H2-only encoder supports only reference_type h2; use the full "
              "model_encoder_full.onnx for g1/smpl/teleop reference types");
        }
        if (use_h2_native_ && motion_encode_mode == 0 &&
            (!motion->HasH2JointPositions() || !motion->HasH2JointVelocities() ||
             motion->GetNumH2Joints() != H2_NUM_MOTOR)) {
          throw std::runtime_error(
              "H2 reference '" + motion->name +
              "' requires complete 31D h2_joint_pos.csv and h2_joint_vel.csv sidecars");
        }
        if (use_h2_native_ && motion_encode_mode == 3 &&
            motion->GetNumJoints() != POLICY_ACTION_DIM) {
          throw std::runtime_error(
              "G1 reference '" + motion->name +
              "' requires native 29D joint_pos.csv and joint_vel.csv");
        }
        if (use_h2_native_ && motion_encode_mode == 2 &&
            (motion->GetNumSmplJoints() != 24 || motion->GetNumSmplRootQuaternions() != 1)) {
          throw std::runtime_error(
              "SMPL reference '" + motion->name +
              "' requires smpl_joint.csv (24 joints) and smpl_root_quat.csv");
        }
        if (use_h2_lora_ && motion_encode_mode == 2 &&
            (motion->GetNumSmplJoints() != 24 || motion->GetNumSmplRootQuaternions() != 1)) {
          std::ostringstream oss;
          oss << "H2LoRA motion '" << motion->name
              << "' requests SMPL encoder mode 2 but does not provide smpl_joint.csv "
              << "with 24 joints and smpl_root_quat.csv with one wxyz root quaternion. "
              << "Re-export the reference with matching SMPL data; "
              << "do not silently fall back to the G1 encoder for SMPL golden rollouts.";
          throw std::runtime_error(oss.str());
        }
        motion->SetEncodeMode(motion_encode_mode);
        std::cout << "  Motion '" << motion->name << "' encode_mode set to: "
                  << motion_encode_mode;
        std::cout << " (reference_type=" << reference_type_name(motion_encode_mode) << ")";
        if (motion->HasPreferredEncodeMode()) {
          std::cout << " (preferred)";
        }
        std::cout << std::endl;
      }
      
      // Set encode_mode for planner motion
      planner_motion_->SetEncodeMode(planner_encoder_mode_);
      std::cout << "Planner motion encode_mode set to: " << planner_encoder_mode_
                << (project_h2_native_planner_reference_
                        ? " (reference_type=h2; projected G1 planner q/qdot)"
                        : " (reference_type=g1; native G1 planner branch)")
                << std::endl;

      // Initialize planner
      if (!planner_path.empty()) {
        PlannerConfig planner_config;
        planner_config.model_path = planner_path;
        if (planner_path.find("V0") != std::string::npos)
        {
          planner_config.version = 0;
        }
        else if (planner_path.find("V1") != std::string::npos)
        {
          planner_config.version = 1;
        }
        else if (planner_path.find("V2") != std::string::npos)
        {
          planner_config.version = 2;
        }
        else
        {
          std::cout << "Unsupported planner version: " << planner_path << std::endl;
          throw std::runtime_error("Unsupported planner version: " + planner_path);
        }
        planner_ = std::make_unique<LocalMotionPlannerTensorRT>(planner_fp16, 0, planner_config);
      }
      
      // Initialize observation function map
      InitializeObservationFunctions();
      
      // Log observation configuration details
      LogObservationConfiguration();

      // Prepare robot configuration for data collection (after all initialization is complete)
      std::map<std::string, std::variant<std::string, int, double, bool>> robot_config;
      robot_config["model_path"] = model_path;
      robot_config["reference_motion_path"] = motion_data_path;
      robot_config["planner_path"] = planner_path.empty() ? "none" : planner_path;
      robot_config["obs_config_path"] = obs_config_path.empty() ? "none" : obs_config_path;
      robot_config["encoder_file"] = encoder_file_path.empty() ? "none" : encoder_file_path;
      robot_config["robot_profile"] = robot_profile_;
      robot_config["h2_vla_protocol"] = 1;
      robot_config["h2_vla_watchdog"] = h2_vla::Enabled();
      if (use_h2_native_) {
        robot_config["h2_waist_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.waist_pitch_kp);
        robot_config["h2_waist_roll_kp_scale"] = static_cast<double>(h2_gain_scales_.waist_roll_kp);
        robot_config["h2_ankle_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.ankle_pitch_kp);
        robot_config["h2_ankle_roll_kp_scale"] = static_cast<double>(h2_gain_scales_.ankle_roll_kp);
        robot_config["h2_ankle_pitch_kd_scale"] = static_cast<double>(h2_gain_scales_.ankle_pitch_kd);
        robot_config["h2_left_ankle_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.KpScaleForDdsIndex(5));
        robot_config["h2_left_ankle_roll_kp_scale"] = static_cast<double>(h2_gain_scales_.KpScaleForDdsIndex(4));
        robot_config["h2_right_ankle_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.KpScaleForDdsIndex(11));
        robot_config["h2_right_ankle_roll_kp_scale"] = static_cast<double>(h2_gain_scales_.KpScaleForDdsIndex(10));
        robot_config["h2_left_knee_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.left_knee_pitch_kp);
        robot_config["h2_right_knee_pitch_kp_scale"] = static_cast<double>(h2_gain_scales_.right_knee_pitch_kp);
      }
      robot_config["command_motor_count"] = robot_motor_count_;
      robot_config["policy_action_dim"] = POLICY_ACTION_DIM;
      robot_config["h2_vr_head_control"] = enable_h2_vr_head_control_;
      robot_config["project_h2_lora_planner_reference"] =
          project_h2_lora_planner_reference_;
      robot_config["project_h2_native_planner_reference"] =
          project_h2_native_planner_reference_;
      robot_config["planner_encoder_mode"] = planner_encoder_mode_;
      robot_config["control_frequency"] = 1.0 / control_dt_;
      robot_config["planner_frequency"] = 1.0 / planner_dt_;
      robot_config["is_using_encoder"] = is_using_encoder_;
      robot_config["policy_fp16"] = policy_fp16;
      robot_config["planner_fp16"] = planner_fp16;

      // Initialize state logger with complete robot configuration
      try {
        std::string resolved_logs_dir = logs_dir;
        // dt is control_dt_; pass complete robot_config to constructor
        state_logger_ = std::make_unique<StateLogger>(resolved_logs_dir, 10000, POLICY_ACTION_DIM, POLICY_ACTION_DIM,
                                                       control_dt_, enable_csv_logs, robot_config);
        if (enable_csv_logs) {
          if (!policy_input_file_) {
            const std::filesystem::path policy_input_path =
                std::filesystem::path(state_logger_->csvPath()) / "policy_input.csv";
            policy_input_file_ = std::make_unique<std::ofstream>(policy_input_path, std::ios::out);
            if (!policy_input_file_ || !policy_input_file_->good()) {
              throw std::runtime_error("Failed to open policy input CSV log");
            }
            std::cout << "✓ Policy input CSV logging enabled: " << policy_input_path << std::endl;
          }

          if (encoder_engine_ && encoder_engine_->IsInitialized()) {
            const std::filesystem::path encoder_input_path =
                std::filesystem::path(state_logger_->csvPath()) / "encoder_input.csv";
            encoder_input_file_ = std::make_unique<std::ofstream>(encoder_input_path, std::ios::out);
            if (!encoder_input_file_ || !encoder_input_file_->good()) {
              throw std::runtime_error("Failed to open encoder input CSV log");
            }
            std::cout << "✓ Encoder input CSV logging enabled: " << encoder_input_path << std::endl;
          }

          const std::filesystem::path lowcmd_path =
              std::filesystem::path(state_logger_->csvPath()) / "lowcmd_command.csv";
          lowcmd_command_file_ = std::make_unique<std::ofstream>(lowcmd_path, std::ios::out);
          if (lowcmd_command_file_ && lowcmd_command_file_->good()) {
            (*lowcmd_command_file_) << "time_monotonic_ms,program_state,start,play";
            for (int i = 0; i < robot_motor_count_; ++i) {
              (*lowcmd_command_file_) << ",q_target_" << i;
            }
            for (int i = 0; i < robot_motor_count_; ++i) {
              (*lowcmd_command_file_) << ",dq_target_" << i;
            }
            for (int i = 0; i < robot_motor_count_; ++i) {
              (*lowcmd_command_file_) << ",kp_" << i;
            }
            for (int i = 0; i < robot_motor_count_; ++i) {
              (*lowcmd_command_file_) << ",kd_" << i;
            }
            for (int i = 0; i < robot_motor_count_; ++i) {
              (*lowcmd_command_file_) << ",tau_ff_" << i;
            }
            (*lowcmd_command_file_) << std::endl;
            std::cout << "✓ LowCmd command CSV logging enabled: " << lowcmd_path << std::endl;
          } else {
            throw std::runtime_error("Failed to open lowcmd command CSV log");
          }
        }
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to initialize state logger: " << e.what() << std::endl;
        std::cerr << "State logger is required for operation. Exiting..." << std::endl;
        throw;  // Re-throw to stop program initialization
      }

      // Initialize input interface based on type
      if (input_type == "gamepad") {
        input_interface_ = std::make_unique<unitree::common::Gamepad>();
        std::cout << "Initialized gamepad input interface" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      } 
      else if (input_type == "gamepad_manager") {
        input_interface_ = std::make_unique<GamepadManager>(zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose);
        std::cout << "Initialized demo gamepad manager input interface" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Verbose: " << (zmq_verbose ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "manager") {
        input_interface_ = std::make_unique<InterfaceManager>(
          zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose
        );
        std::cout << "Initialized interface manager (Shift+1/2/3/4 [! @ # $] to switch: keyboard, gamepad, zmq"
#if HAS_ROS2
                  " , ros2"
#endif
                  ")" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "zmq") {
        input_interface_ = std::make_unique<ZMQEndpointInterface>(
          zmq_host, zmq_port, zmq_topic, zmq_conflate, zmq_verbose
        );
        std::cout << "Initialized ZMQ endpoint interface" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Topic: " << zmq_topic << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
      else if (input_type == "zmq_manager") {
        input_interface_ = std::make_unique<ZMQManager>(
          zmq_host,
          zmq_port,
          zmq_topic,
          "command",
          "planner",
          zmq_conflate,
          zmq_verbose
        );
        std::cout << "Initialized ZMQ manager" << std::endl;
        std::cout << "  Host: " << zmq_host << ":" << zmq_port << std::endl;
        std::cout << "  Pose topic: " << zmq_topic << std::endl;
        std::cout << "  Command topic: command" << std::endl;
        std::cout << "  Planner topic: planner" << std::endl;
        std::cout << "  Conflate: " << (zmq_conflate ? "enabled" : "disabled") << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
#if HAS_ROS2
      else if (input_type == "ros2") {
        // ROS2 input interface will be initialized without encoder mode parameter
        input_interface_ = std::make_unique<ROS2InputHandler>(true, "g1_deploy_ros2_handler");
        std::cout << "Initialized ROS2 input interface" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }
#endif
      else {
        input_interface_ = std::make_unique<SimpleKeyboard>();
        std::cout << "Initialized keyboard input interface (default)" << std::endl;
        std::cout << "  Initial encoder mode: " << initial_encoder_mode_ << std::endl;
      }

      if (input_interface_) {
        input_interface_->SetPlannerEncoderMode(planner_encoder_mode_);
      }

      // Set initial VR 3-point compliance values for all input interfaces
      // These are keyboard-controlled: g/h for left hand, b/v for right hand
      if (input_interface_) {
        input_interface_->SetVR3PointCompliance(initial_vr_3point_compliance_);
        // Set initial max close ratio for hands (keyboard-controlled: X/C keys)
        input_interface_->SetMaxCloseRatio(initial_max_close_ratio_);
        if (hand_backend_ == HandBackendKind::DEX3) {
          dex3_hands_.SetMaxCloseRatio(initial_max_close_ratio_);
        } else if (hand_backend_ == HandBackendKind::INSPIRE) {
          inspire_hands_.SetMaxCloseRatio(initial_max_close_ratio_);
        }
        std::cout << "[INFO] Initial VR 3-point compliance: ["
                  << initial_vr_3point_compliance_[0] << ", "
                  << initial_vr_3point_compliance_[1] << ", "
                  << initial_vr_3point_compliance_[2] << "]" << std::endl;
        std::cout << "[INFO] Initial hand max close ratio: " << initial_max_close_ratio_ 
                  << " (1.0 = full closure allowed, 0.2 = limited)" << std::endl;
        std::cout << "[INFO] Keyboard controls: g/h = left hand +/- 0.1, b/v = right hand +/- 0.1 (range: 0.01-0.5)" << std::endl;
        std::cout << "[INFO] Keyboard controls: x/c = hand max close ratio +/- 0.1 (range: 0.2-1.0)" << std::endl;
        
        // Info message about compliance observation status
        if (!has_vr_3point_compliance_obs_) {
          std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
          std::cout << "│  Compliance control: DISABLED                                               │" << std::endl;
          std::cout << "│  Policy does not observe 'vr_3point_compliance'.                            │" << std::endl;
          std::cout << "│  Keyboard controls (g/h/b/v) will be ignored for this policy.               │" << std::endl;
          std::cout << "│  To enable, use a compliance-aware policy that observes it natively.        │" << std::endl;
          std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n" << std::endl;
        } else {
          std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐" << std::endl;
          std::cout << "│  Compliance control: ENABLED                                                │" << std::endl;
          std::cout << "│  Policy observes 'vr_3point_compliance'. Keyboard controls (g/h/b/v) work.  │" << std::endl;
          std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n" << std::endl;
        }
      }

      // Initialize output interfaces based on type (now that state_logger_ is ready)
      // Supports multiple simultaneous outputs with --output-type all
      bool create_zmq = (output_type == "zmq" || output_type == "all");
      bool create_ros2 = false;
#if HAS_ROS2
      create_ros2 = (output_type == "ros2" || output_type == "all");
#endif

      if (create_zmq) {
        auto zmq_handler = std::make_unique<ZMQOutputHandler>(*state_logger_, zmq_out_port, zmq_out_topic);
        std::cout << "Initialized ZMQ output interface" << std::endl;
        // Publish robot config so subscribers can receive it before control loop starts
        zmq_handler->publish_config();
        output_interfaces_.push_back(std::move(zmq_handler));
      }

#if HAS_ROS2
      if (create_ros2) {
        auto ros2_handler = std::make_unique<ROS2OutputHandler>(*state_logger_, "g1_output_handler");
        std::cout << "Initialized ROS2 output interface" << std::endl;
        // Publish robot config so subscribers can receive it before control loop starts
        ros2_handler->publish_config();
        output_interfaces_.push_back(std::move(ros2_handler));
      }
#endif

      if (output_interfaces_.empty() && output_type != "all") {
        std::cout << "Unknown output type '" << output_type << "' - no output will be published" << std::endl;
      } else {
        std::cout << "Total output interfaces initialized: " << output_interfaces_.size() << std::endl;
      }

      // Start diagnostic auto-control timers after all heavyweight model
      // loading has completed, so short automated tests are measured from the
      // actual control-thread lifetime rather than from constructor entry.
      wall_start_time_ = std::chrono::steady_clock::now();

      // Seed the command buffer before the 500 Hz writer thread starts.
      MotorCommand motor_command_tmp;
      for (int i = 0; i < robot_motor_count_; ++i) {
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.q_target.at(i) =
            static_cast<float>(ProfileDefaultAngle(robot_profile_, i));
        motor_command_tmp.dq_target.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = ProfileInitKp(robot_profile_, i);
        motor_command_tmp.kd.at(i) = ProfileInitKd(robot_profile_, i);
      }
      motor_command_buffer_.SetData(motor_command_tmp);

      // create threads
      input_thread_ptr_ = CreateRecurrentThreadEx("Input", UT_CPU_ID_NONE, input_dt_ * 1e6, &G1Deploy::Input, this);
      command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, publish_dt_ * 1e6,
                                                    &G1Deploy::LowCommandWriter, this);
      control_thread_ptr_ =
          CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, control_dt_ * 1e6, &G1Deploy::Control, this);
      
      if (planner_) {
        planner_thread_ptr_ =
          CreateRecurrentThreadEx("planner", UT_CPU_ID_NONE, planner_dt_ * 1e6, &G1Deploy::Planner, this);
      }
          
      SetThreadPriority();
    }

    ~G1Deploy()
    {
      // CUDA resources are now cleaned up by the PolicyEngine and planner classes automatically
    }

    void SetThreadPriority() {
      struct sched_param param;
      param.sched_priority = sched_get_priority_max(SCHED_FIFO);
      pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }

    /// DDS callback: receives a 500 Hz LowState message from the robot SDK.
    /// Validates CRC (unless disabled for MuJoCo sim) and stores the result
    /// in the thread-safe low_state_buffer_.
    const char* ProfileJointNameDds(int motor_index) const {
      if (use_h2_robot_) {
        if (motor_index >= 0 && motor_index < H2_NUM_MOTOR) {
          return h2_joint_names_dds[motor_index];
        }
      } else if (motor_index >= 0 && motor_index < G1_NUM_MOTOR) {
        return G1_JOINT_NAMES_DDS[motor_index];
      }
      return "Unknown";
    }

    void LowStateHandler(const void* message) {
      LowState_ low_state = *(const LowState_*)message;

      // Only perform CRC check if not disabled (for MuJoCo simulation compatibility)
      if (!disable_crc_check_) {
        uint32_t received_crc = low_state.crc();
        uint32_t calculated_crc = Crc32Core((uint32_t*)&low_state, (sizeof(LowState_) >> 2) - 1);

        if (received_crc != calculated_crc) {
          static int crc_error_count = 0;
          crc_error_count++;
          std::cout << "[ERROR] CRC Error #" << crc_error_count << " - Received: 0x" << std::hex << received_crc
                    << ", Calculated: 0x" << calculated_crc << ", Size: " << sizeof(LowState_) << " bytes" << std::dec
                    << std::endl;
          return;
        }
      }

      // Check motor error states (prints only on transitions)
      std::array<uint32_t, MAX_NUM_MOTOR> motorstates = {};
      for (int i = 0; i < robot_motor_count_; ++i) {
        motorstates[i] = low_state.motor_state()[i].motorstate();
      }
      if (use_h2_robot_) {
        error_monitor_.update(motorstates, robot_motor_count_, h2_joint_names_dds);
      } else {
        error_monitor_.update(motorstates, robot_motor_count_, G1_JOINT_NAMES_DDS);
      }

      low_state_buffer_.SetData(low_state);

      // update mode machine
      if (mode_machine_ != low_state.mode_machine()) {
        if (mode_machine_ == 0) std::cout << "Robot mode machine: " << unsigned(low_state.mode_machine()) << std::endl;
        mode_machine_ = low_state.mode_machine();
      }
    }

    /// DDS callback: receives secondary (torso) IMU data.
    void imuTorsoHandler(const void* message) {
      IMUState_ imu_torso = *(const IMUState_*)message;
      imu_torso_buffer_.SetData(imu_torso);
    }

    /**
     * @brief Command-writer thread body (500 Hz).
     *
     * Reads the latest MotorCommand from motor_command_buffer_, packs it
     * into a LowCmd_ DDS message with CRC, and publishes via DDS.
     * Also services the selected DDS hand backend.
     */
    void LowCommandWriter() {
      LowCmd_ dds_low_command;
      dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
      dds_low_command.mode_machine() = mode_machine_;

      const auto motor_command_data = motor_command_buffer_.GetDataWithTime();
      const std::shared_ptr<const MotorCommand> mc = motor_command_data.data;
      if (mc) {
        const auto now = std::chrono::steady_clock::now();
        const double t_ms = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
        for (int i = 0; i < robot_motor_count_; i++) {
          dds_low_command.motor_cmd().at(i).mode() = 1; // 1:Enable, 0:Disable
          dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
          dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
          dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
          // Scale once at publication, consistently across INIT and CONTROL.
          dds_low_command.motor_cmd().at(i).kp() =
              use_h2_native_ ? h2_gain_scales_.ApplyKp(i, mc->kp.at(i)) : mc->kp.at(i);
          dds_low_command.motor_cmd().at(i).kd() =
              use_h2_native_
                  ? h2_gain_scales_.ApplyKd(i, mc->kd.at(i), mc->kp.at(i) == 0.0F)
                  : mc->kd.at(i);
        }

        if (lowcmd_command_file_ && lowcmd_command_file_->good()) {
          (*lowcmd_command_file_) << std::fixed << std::setprecision(6) << t_ms << ","
                                  << static_cast<int>(program_state_) << ","
                                  << static_cast<int>(operator_state.start) << ","
                                  << static_cast<int>(operator_state.play);
          for (int i = 0; i < robot_motor_count_; ++i) {
            (*lowcmd_command_file_) << "," << mc->q_target.at(i);
          }
          for (int i = 0; i < robot_motor_count_; ++i) {
            (*lowcmd_command_file_) << "," << mc->dq_target.at(i);
          }
          for (int i = 0; i < robot_motor_count_; ++i) {
            (*lowcmd_command_file_) << "," << dds_low_command.motor_cmd().at(i).kp();
          }
          for (int i = 0; i < robot_motor_count_; ++i) {
            (*lowcmd_command_file_) << "," << dds_low_command.motor_cmd().at(i).kd();
          }
          for (int i = 0; i < robot_motor_count_; ++i) {
            (*lowcmd_command_file_) << "," << mc->tau_ff.at(i);
          }
          (*lowcmd_command_file_) << std::endl;
        }

        dds_low_command.crc() = Crc32Core((uint32_t*)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
        // Temporarily disable periodic command-latency diagnostics in the operator console.
#if 0
        const auto low_state_data = low_state_buffer_.GetDataWithTime();
        if (low_state_data.data &&
            now - last_command_latency_log_time_ >= COMMAND_LATENCY_LOG_INTERVAL) {
          const double low_state_to_lowcmd_ms =
              std::chrono::duration<double, std::milli>(now - low_state_data.timestamp).count();
          const double motor_command_age_ms =
              std::chrono::duration<double, std::milli>(now - motor_command_data.timestamp).count();
          const double low_state_to_motor_command_ms =
              std::chrono::duration<double, std::milli>(motor_command_data.timestamp - low_state_data.timestamp).count();
          std::cout << "Command latency - LowState to LowCmd publish: "
                    << low_state_to_lowcmd_ms << "ms"
                    << ", LowState to MotorCommand: " << low_state_to_motor_command_ms << "ms"
                    << ", MotorCommand age: " << motor_command_age_ms << "ms"
                    << std::endl;
          last_command_latency_log_time_ = now;
        }
#endif
        lowcmd_publisher_->Write(dds_low_command);
      }

      if (hand_backend_ == HandBackendKind::DEX3) {
        dex3_hands_.writeOnce();
      } else if (hand_backend_ == HandBackendKind::INSPIRE) {
        // InspireHands internally throttles this 500 Hz writer to 20 Hz.
        inspire_hands_.writeOnce();
      }
    }

    /// Gracefully stop all threads and send a damping-only command.
    void Stop() {
      operator_state.stop = true;
      if (hand_backend_ == HandBackendKind::INSPIRE) {
        // Stop publishing; never synthesize an open/close command on shutdown.
        inspire_hands_.ResetInput();
      }

      if (control_thread_ptr_) {
        input_thread_ptr_->Wait();
        input_thread_ptr_.reset();
        control_thread_ptr_->Wait();
        control_thread_ptr_.reset();
        command_writer_ptr_->Wait();
        command_writer_ptr_.reset();
        if (planner_thread_ptr_) {
          planner_thread_ptr_->Wait();
          planner_thread_ptr_.reset();
        }
      }
      CreateDampingCommand();
      LowCommandWriter();
      std::cout << "Stop" << std::endl;
    }

    /// Write a zero-torque, damping-only motor command (safe shutdown pose).
    void CreateDampingCommand() {
      MotorCommand motor_command_tmp;
      const std::shared_ptr<const LowState_> ls = low_state_buffer_.GetDataWithTime().data;

      for (int i = 0; i < robot_motor_count_; ++i) {
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.q_target.at(i) = 0.0;
        motor_command_tmp.dq_target.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = 0;
        motor_command_tmp.kd.at(i) = 8;
      }
      motor_command_buffer_.SetData(motor_command_tmp);
    }

    /**
     * @brief INIT state handler: ramp the robot from its initial pose to the
     *        default standing angles over `duration_` seconds (linear interpolation).
     *
     * Called at 50 Hz until the ramp completes, at which point the state machine
     * transitions to WAIT_FOR_CONTROL. Legacy DEX3 initialization retains its
     * close/open sequence; Inspire never moves during robot initialization.
     * @return True once LowState data is available; false if not yet ready.
     */
    bool InitControl() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      if (!ls) {
        return false;
      }
      std::array<double, MAX_NUM_MOTOR> init_target = {};
      for (int i = 0; i < robot_motor_count_; ++i) {
        init_target.at(i) = ProfileDefaultAngle(robot_profile_, i);
      }
      if (!init_pose_snapshot_valid_) {
        for (int i = 0; i < robot_motor_count_; ++i) {
          init_pose_snapshot_.at(i) = ls->motor_state()[i].q();
        }
        init_pose_snapshot_valid_ = true;
      }
      MotorCommand motor_command_tmp;
      for (int i = 0; i < robot_motor_count_; ++i) {
        motor_command_tmp.tau_ff.at(i) = 0.0;
        motor_command_tmp.q_target.at(i) = static_cast<float>(init_target.at(i));
        motor_command_tmp.dq_target.at(i) = 0.0;
        motor_command_tmp.kp.at(i) = ProfileInitKp(robot_profile_, i);
        motor_command_tmp.kd.at(i) = ProfileInitKd(robot_profile_, i);
      }
      time_ += control_dt_;
      if (time_ < duration_) {
        double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
        for (int i = 0; i < robot_motor_count_; i++) {
          motor_command_tmp.q_target.at(i) =
              static_cast<float>(
                  init_pose_snapshot_.at(i) * (1.0 - ratio)
                  + init_target.at(i) * ratio);
        }
        if (hand_backend_ == HandBackendKind::DEX3) {
          dex3_hands_.close(true);
          dex3_hands_.close(false);
        }
      } else {
        program_state_ = ProgramState::WAIT_FOR_CONTROL;
        if (hand_backend_ == HandBackendKind::DEX3) {
          dex3_hands_.open(true);
          dex3_hands_.open(false);
        }
        std::cout << "Init Done" << std::endl;
      }
      motor_command_buffer_.SetData(motor_command_tmp);
      return true;
    }

    /// Check for valid LowState data and recent updates; if invalid, transition to ERROR state.
    bool CheckSafety() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      if (!ls) {
        std::cout << "[ERROR] LowState data is not available in the middle of the control loop!" << std::endl;
        return false;
      }

      auto now = std::chrono::steady_clock::now();
      if (now - low_state_data.timestamp > LOW_STATE_ABSENT_THRESHOLD) {
        std::cout << "[ERROR] Lost LowState data connection from robot!" << std::endl;
        return false;
      }

      return true;
    }

    /// Get current ROS 2 timestamp (seconds) from the first ROS2 output interface, or 0.0 if none.
    double GetRosTimestamp() {
      // Get ROS timestamp from the first ROS2 output interface (0.0 if none available)
      double ros_timestamp = 0.0;
#if HAS_ROS2
      for (auto& output_interface : output_interfaces_) {
        if (output_interface && output_interface->GetType() == OutputInterface::OutputType::ROS2) {
          auto* ros2_handler = dynamic_cast<ROS2OutputHandler*>(output_interface.get());
          if (ros2_handler) {
            ros_timestamp = ros2_handler->GetROSTimestamp();
            break;  // Use timestamp from first ROS2 interface found
          }
        }
      }
#endif
      return ros_timestamp;
    }

    /**
     * @brief Read the latest robot state (IMU + joints + hands) and log it.
     *
     * Reads LowState and torso IMU from their DataBuffers, remaps joint positions
     * from hardware order to IsaacLab order (subtracting default_angles), and
     * calls StateLogger::LogFullState() with the complete snapshot.
     *
     * @return False if LowState or IMU data is missing or a joint velocity
     *         exceeds the safety threshold (35 rad/s).
     */
    bool GatherRobotStateToLogger() {
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      auto imu_data = imu_torso_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      const std::shared_ptr<const IMUState_> imu_torso = imu_data.data;
      if (!ls || !imu_torso) {
        std::cout << "✗ Error: LowState or IMUState is not available in the middle of the control loop!" << std::endl;
        return false;
      }
      used_low_state_data_ = (low_state_data);
      used_imu_torso_data_ = (imu_data);
      // robot state data
      std::array<double, POLICY_ACTION_DIM> body_q = {0.0};
      std::array<double, POLICY_ACTION_DIM> body_dq = {0.0};

      auto unitree_joint_state = ls->motor_state();
      std::array<double, MAX_NUM_MOTOR * 2> motor_temperature = {0.0};
      std::array<double, MAX_NUM_MOTOR> motor_error = {0.0};
      std::array<double, MAX_NUM_MOTOR> motor_torque = {0.0};

      if (use_h2_native_) {
        for (int policy_index = 0; policy_index < POLICY_ACTION_DIM; ++policy_index) {
          const int h2_dds_index = h2_native_policy_to_h2_dds[policy_index];
          body_q[policy_index] =
              unitree_joint_state[h2_dds_index].q() -
              h2_native_default_angles_policy[policy_index];
          body_dq[policy_index] = unitree_joint_state[h2_dds_index].dq();
        }
      } else if (use_h2_lora_) {
        std::array<double, POLICY_ACTION_DIM> h2_physical_q = {0.0};
        std::array<double, POLICY_ACTION_DIM> h2_physical_dq = {0.0};
        for (int i = 0; i < POLICY_ACTION_DIM; i++) {
          const int h2_local_index = h2lora_body_to_h2_mujoco[i];
          const int h2_dds_index = h2lora_body_to_h2_dds[i];
          h2_physical_q[i] = unitree_joint_state[h2_dds_index].q() - h2_default_angles[h2_local_index];
          h2_physical_dq[i] = unitree_joint_state[h2_dds_index].dq();
        }
        body_q = H2PhysicalToG1Semantic(h2_physical_q);
        body_dq = H2PhysicalToG1Semantic(h2_physical_dq);
      } else {
        for (int i = 0; i < POLICY_ACTION_DIM; i++) {
          body_q[i] =
              unitree_joint_state[mujoco_to_isaaclab[i]].q() - default_angles[mujoco_to_isaaclab[i]]; // URDF order
          body_dq[i] = unitree_joint_state[mujoco_to_isaaclab[i]].dq(); // URDF order
        }
      }

      for (int i = 0; i < POLICY_ACTION_DIM; i++) {
        if (body_dq[i] > 35 && !disable_crc_check_) {
          std::cout << "✗ Error: body_dq[" << i << "] = " << body_dq[i] << " > 35."
                    << std::endl;
          return false;
        }
      }

      for (int i = 0; i < robot_motor_count_; ++i) {
        motor_temperature[i * 2] = static_cast<double>(unitree_joint_state[i].temperature()[0]);
        motor_temperature[i * 2 + 1] = static_cast<double>(unitree_joint_state[i].temperature()[1]);
        motor_error[i] = static_cast<double>(unitree_joint_state[i].motorstate());
        motor_torque[i] = static_cast<double>(unitree_joint_state[i].tau_est());

        const int16_t max_temp = std::max(unitree_joint_state[i].temperature()[0],
                                          unitree_joint_state[i].temperature()[1]);
        if (motor_high_temp_[i]) {
          if (max_temp < HIGH_TEMP_EXIT) { motor_high_temp_[i] = false; }
        } else {
          if (max_temp >= HIGH_TEMP_ENTER) { motor_high_temp_[i] = true; }
        }
      }

      // Build high temperature warning message for audio
      {
        bool any_high = false;
        std::string high_temp_msg = "Warning! High temperature at ";
        std::string high_temp_print_msg = "Warning! High temperature at ";
        bool first = true;
        for (int i = 0; i < robot_motor_count_; ++i) {
          if (motor_high_temp_[i]) {
            any_high = true;
            const int16_t t = std::max(unitree_joint_state[i].temperature()[0],
                                       unitree_joint_state[i].temperature()[1]);
            if (!first) { high_temp_msg += ", "; high_temp_print_msg += ", "; }
            const std::string joint_name = ProfileJointNameDds(i);
            high_temp_msg += joint_name;
            high_temp_print_msg += joint_name + "(" + std::to_string(t) + ")";
            first = false;
          }
        }
        high_temp_warning_ = any_high;
        high_temp_message_ = any_high ? high_temp_msg : "";

        // Print to console at ~1s intervals while overheating (50 Hz control loop)
        if (any_high) {
          static int high_temp_print_counter = 0;
          if (high_temp_print_counter % 50 == 0) {
            std::cout << "[HighTemp] " << high_temp_print_msg << std::endl;
          }
          high_temp_print_counter++;
        }
      }

      std::array<double, 4> base_quat = float_to_double<4>(ls->imu_state().quaternion()); // qw, qx, qy, qz
      std::array<double, 3> base_ang_vel = float_to_double<3>(ls->imu_state().gyroscope());
      std::array<double, 3> base_accel = float_to_double<3>(ls->imu_state().accelerometer());

      std::array<double, 4> body_torso_quat = float_to_double<4>(imu_torso->quaternion()); //qw, qx, qy, qz 
      std::array<double, 3> body_torso_ang_vel = float_to_double<3>(imu_torso->gyroscope());
      std::array<double, 3> body_torso_accel = float_to_double<3>(imu_torso->accelerometer());

      // Collect the selected backend's state in the legacy 7-D logging slots.
      std::array<double, 7> left_hand_q = {0.0};
      std::array<double, 7> left_hand_dq = {0.0};
      std::array<double, 7> right_hand_q = {0.0};
      std::array<double, 7> right_hand_dq = {0.0};
      
      if (hand_backend_ == HandBackendKind::DEX3) {
        auto left_hand_state_ptr = dex3_hands_.getState(true);
        if (left_hand_state_ptr && left_hand_state_ptr->motor_state().size() >= 7) {
          for (int i = 0; i < 7; ++i) {
            left_hand_q[i] = left_hand_state_ptr->motor_state()[i].q();
            left_hand_dq[i] = left_hand_state_ptr->motor_state()[i].dq();
          }
        }

        auto right_hand_state_ptr = dex3_hands_.getState(false);
        if (right_hand_state_ptr && right_hand_state_ptr->motor_state().size() >= 7) {
          for (int i = 0; i < 7; ++i) {
            right_hand_q[i] = right_hand_state_ptr->motor_state()[i].q();
            right_hand_dq[i] = right_hand_state_ptr->motor_state()[i].dq();
          }
        }
      } else if (hand_backend_ == HandBackendKind::INSPIRE) {
        const auto state = inspire_hands_.getStateWithTime();
        inspire_hand_control::CombinedTarget positions{};
        if (state.data &&
            inspire_hand_control::DecodeUsableState(*state.data, positions)) {
          for (std::size_t i = 0; i < inspire_hand_control::kHandDof; ++i) {
            right_hand_q[i] = positions[inspire_hand_control::kRightOffset + i];
            left_hand_q[i] = positions[inspire_hand_control::kLeftOffset + i];
          }
        }
      }

      // Log robot state for analysis and debugging
      if (state_logger_) {
        // Get ROS timestamp if using ROS2 output interface (0.0 for non-ROS2 interfaces)
        double ros_timestamp = GetRosTimestamp();
        state_logger_->LogFullState(base_quat, base_ang_vel, base_accel, body_torso_quat, body_torso_ang_vel, body_torso_accel,
                                    std::span(body_q),
                                    std::span(body_dq),
                                    std::span(last_action),
                                    std::span(motor_temperature).first(static_cast<size_t>(robot_motor_count_) * 2),
                                    std::span(motor_error).first(static_cast<size_t>(robot_motor_count_)),
                                    std::span(motor_torque).first(static_cast<size_t>(robot_motor_count_)),
                                    std::span(left_hand_q),
                                    std::span(left_hand_dq),
                                    std::span(right_hand_q),
                                    std::span(right_hand_dq),
                                    std::span(last_left_hand_action),
                                    std::span(last_right_hand_action),
                                    ros_timestamp);
      }
      return true;
    }

    /**
     * @brief Snapshot all input-interface data into local buffers.
     *
     * Called once per control tick **before** GatherObservations().  Copies VR
     * positions / orientations / compliance, hand joints, upper-body targets,
     * and external token state from the active InputInterface into member
     * buffers so that observations, logging, and output all use a consistent
     * snapshot.
     *
     * Also handles the external-token-state fallback logic:
     *  - If encoder is loaded → token_state_data_ will be populated later
     *    by GatherTokenState() during observation gathering.
     *  - If no encoder and external token data arrives (ZMQ/ROS2) → copies it.
     *  - If no encoder and no external data → error recovery depending on
     *    initial_encoder_mode_ (-1 = stop, 0+ = fall back to local encoder).
     */
    bool GatherInputInterfaceData() {
      // Snapshot input interface data into buffers before gathering observations
      // This ensures observations, logging, and output all use the same data
      // some of the data will be overwritten by the motion dataset if we are not using the buffered interface data
      std::tie(has_vr_3point_data_, vr_3point_position_buffer_) = input_interface_->GetVR3PointPosition();
      std::tie(std::ignore, vr_3point_orientation_buffer_) = input_interface_->GetVR3PointOrientation();
      vr_3point_compliance_buffer_ = input_interface_->GetVR3PointCompliance();
      head_pose_command_buffer_ = input_interface_->GetHeadPoseCommand();
      std::tie(has_vr_5point_data_, vr_5point_position_buffer_) = input_interface_->GetVR5PointPosition();
      std::tie(std::ignore, vr_5point_orientation_buffer_) = input_interface_->GetVR5PointOrientation();
      std::tie(has_left_hand_data_, left_hand_joint_buffer_) = input_interface_->GetHandPose(true);
      std::tie(has_right_hand_data_, right_hand_joint_buffer_) = input_interface_->GetHandPose(false);
      std::tie(has_upper_body_data_, upper_body_joint_positions_buffer_) = input_interface_->GetUpperBodyJointPositions();
      std::tie(std::ignore, upper_body_joint_velocities_buffer_) = input_interface_->GetUpperBodyJointVelocities();

      auto last_update_time = input_interface_->GetLastUpdateTime();
      if (last_update_time.has_value()) {
        auto streaming_data_delay = std::chrono::steady_clock::now() - last_update_time.value();
        streaming_data_delay_rolling_stats_.push(static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(streaming_data_delay).count()));

        bool streaming_data_absent = streaming_data_delay > STREAMING_DATA_ABSENT_THRESHOLD;
        streaming_data_absent_debouncer_.update(streaming_data_absent);
      }

      auto low_state_data = low_state_buffer_.GetDataWithTime();
      bool low_state_late = (std::chrono::steady_clock::now() - low_state_data.timestamp) > LOW_STATE_LATE_THRESHOLD;

      audio_thread_->SetCommand(
        AudioCommand{
          .streaming_data_absent = streaming_data_absent_debouncer_.state(),
          .motor_error = error_monitor_.hasErrors(),
          .low_state_late = low_state_late,
          .tts_message = std::move(pending_tts_),
          .high_temperature = high_temp_warning_,
          .high_temperature_message = high_temp_message_,
        });
      pending_tts_.clear();

      // if encoder mode is not -2, we need to get the token state data either from the encoder or the external token state
      if (initial_encoder_mode_ != -2) {
        // get the external token state from the input interface
        bool has_ext_token_data;
        std::vector<double> temp_token_state_data;
        {
            std::lock_guard<std::mutex> motion_lock(current_motion_mutex_);
            std::tie(has_ext_token_data, temp_token_state_data) = input_interface_->GetExternalTokenState();
        }

        // if the external token state is not empty, we need to copy the data to the token_state_data_
        // and set the is_using_encoder_ to false
        if (has_ext_token_data) {
          if (temp_token_state_data.empty()) {
            std::cout << "⚠ Warning: The external token state is empty" << std::endl;
            std::cout << "but interface says it has external token state" << std::endl;
            std::cout << "No copying happened." << std::endl;
            return true;
          }

          // check if the size of the temp_token_state_data is the same as the token_state_data_
          if (temp_token_state_data.size() != token_state_data_.size()) {
            std::cout << "⚠ Warning: The size of the external token state is not the same as what defined in the config file" << std::endl;
            std::cout << "temp_token_state_data.size(): " << temp_token_state_data.size() << std::endl;
            std::cout << "token_state_data_.size(): " << token_state_data_.size() << std::endl;
            std::cout << "No copying happened." << std::endl;
          } else {
            std::copy(temp_token_state_data.begin(), temp_token_state_data.end(), token_state_data_.begin());
            is_using_encoder_ = false;
            // Safety: Mark that we've received at least one token
            if (!first_token_received_) {
              first_token_received_ = true;
              std::cout << "[Token Safety] ✓ First token received! Control can now be started." << std::endl;
            }
            last_token_time_ = std::chrono::steady_clock::now();
            
            static int token_copy_count = 0;
            if (token_copy_count % 50 == 0) {  // Warn every ~1 second
              std::cout << "[Token Flow] Copied external tokens, tokens[0]=" << token_state_data_[0] << std::endl;

            }
            token_copy_count++;
          }
        } 
        else if (!has_ext_token_data && !is_using_encoder_) { // if encoder is not used and the external token state is empty
          if (initial_encoder_mode_ == -1) { // no encoder is loaded, we need to wait for the external tokens
            // Token input mode: waiting for external tokens
            auto now = std::chrono::steady_clock::now();
            
            // Check if we've received first token
            if (!first_token_received_) {
              // Warn but don't block - allow control to start
              static auto startup_time = std::chrono::steady_clock::now();
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
              
              // Reset the current token_state_data_ to zeros
              token_state_data_.assign(token_state_data_.size(), 0.0);
              
              static int wait_count = 0;
              if (wait_count % 250 == 0) {  // Print every ~5 seconds (50Hz * 250 = 5s)
                std::cout << "⚠ [Token Safety] WARNING: No tokens received yet (" 
                          << elapsed.count() << "s elapsed). "
                          << "Robot will use zero tokens until tokens start streaming." << std::endl;
              }
              wait_count++;
            } else {
              // We've received tokens before, check freshness
              if (last_token_time_.has_value()) {
                auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_token_time_.value());
                
                if (time_since_last > TOKEN_TIMEOUT_MS) {
                  static int stale_count = 0;
                  if (stale_count % 50 == 0) {  // Warn every ~1 second
                    std::cout << "⚠ [Token Safety] WARNING: No tokens received for " 
                              << time_since_last.count() << "ms (timeout: " 
                              << TOKEN_TIMEOUT_MS.count() << "ms). "
                              << "Robot will use last received tokens." << std::endl;
                  }
                  stale_count++;
                  // Don't pause - just warn
                }
              }
            }
            
          } else { 
            // if encoder mode is not -1, we need to turn on the encoder
            std::cout << "No encoder is used and not using external tokens" << std::endl;
            std::cout << "Turning on the encoder now" << std::endl;
            is_using_encoder_ = true;
          }
        }
      }
      return true;
    }

    void ApplyH2HeadCommand(MotorCommand& motor_command) {
      if (!use_h2_native_ && !enable_h2_vr_head_control_) {
        return;
      }

      const auto now = std::chrono::steady_clock::now();
      const bool command_fresh =
          head_pose_command_buffer_.data &&
          now - head_pose_command_buffer_.timestamp <= H2_HEAD_COMMAND_TIMEOUT;
      std::array<double, 2> desired = {0.0, 0.0};
      if (use_h2_native_ && current_motion_ && current_motion_->HasH2JointPositions()) {
        const int frame = std::clamp(
            current_frame_, 0, static_cast<int>(current_motion_->timesteps) - 1);
        const double* reference = current_motion_->H2JointPositions(frame);
        desired = h2_head_control::Clamp(std::array<double, 2>{
            reference[h2_head_mujoco_indices[0]],
            reference[h2_head_mujoco_indices[1]]});
      }
      if (enable_h2_vr_head_control_ && command_fresh) {
        desired = h2_head_control::Clamp(*head_pose_command_buffer_.data);
      }
      // The VLA was trained with a fixed camera pitch. Keep this pose also
      // after the Python client exits and its head heartbeat expires.
      if (use_h2_native_ && h2_vla::Enabled()) {
        desired = {40.0 * 3.14159265358979323846 / 180.0, 0.0};
      }
      h2_head_targets_ = h2_head_control::Slew(
          h2_head_targets_, desired, control_dt_);

      if (enable_h2_vr_head_control_ && command_fresh != h2_head_command_was_fresh_) {
        std::cout << "[H2 Head] VR command "
                  << (command_fresh
                          ? "active"
                          : (use_h2_native_ ? "timed out; returning to reference"
                                            : "timed out; returning to zero"))
                  << std::endl;
        h2_head_command_was_fresh_ = command_fresh;
      }

      h2_head_control::MergeTargets(
          motor_command, h2_head_targets_, h2_head_dds_indices);
    }

    /**
     * @brief Run the policy neural network and produce a motor command.
     *
     * Converts the double-precision observation buffer to float, copies it
     * into the PolicyEngine's pinned input buffer, runs TensorRT inference,
     * then maps the action output (IsaacLab order) to a MotorCommand
     * (hardware order) using `g1_action_scale` and `default_angles`.
     */
    bool CreatePolicyCommand() {
      // Convert double observation to float and populate policy's internal input buffer
      auto& obs_buffer_float = policy_engine_->GetInputBuffer();
      for (size_t i = 0; i < obs_buffer_.size(); i++) { 
        obs_buffer_float[i] = static_cast<float>(obs_buffer_[i]); 
      }

      // Run policy inference (handles CPU→GPU transfer, inference, GPU→CPU transfer)
      if (!policy_engine_->Infer()) {
        std::cerr << "✗ Error: Control policy inference failed" << std::endl;
        return false;
      }
      
      // Access actions from control policy's internal buffer (already populated by Infer).
      // GEAR_SONIC_POLICY_ACTION_SCALE lets sim2sim explicitly tune the raw
      // residual gain before the same train/deploy action adapter consumes it.
      auto& action_buffer = policy_engine_->GetActionBuffer();
      float* floatarr = action_buffer.data();
      static const double policy_action_clip =
          [this]() {
            const char* env = std::getenv("GEAR_SONIC_POLICY_ACTION_CLIP");
            if (env == nullptr || std::string(env).empty()) {
              // Decoder-only ONNX bypasses Actor.forward(), so H2LoRA deploy
              // must mirror sonic_h2lora_release/config.yaml action_mean_clip.
              return use_h2_robot_ ? 6.0 : 0.0;
            }
            try {
              return std::stod(std::string(env));
            } catch (...) {
              std::cerr << "⚠ Warning: invalid GEAR_SONIC_POLICY_ACTION_CLIP='"
                        << env << "', using "
                        << (use_h2_robot_ ? "6.0" : "disabled") << std::endl;
              return use_h2_robot_ ? 6.0 : 0.0;
            }
          }();
      std::array<float, POLICY_ACTION_DIM> command_action = {};
      for (int i = 0; i < POLICY_ACTION_DIM; i++) {
        double raw_policy_action = static_cast<double>(floatarr[i]);
        if (!std::isfinite(raw_policy_action)) {
          raw_policy_action = 0.0;
        }
        if (policy_action_clip > 0.0) {
          raw_policy_action =
              std::clamp(raw_policy_action, -policy_action_clip, policy_action_clip);
        }
        command_action[i] =
            static_cast<float>(raw_policy_action * policy_action_scale_);
      }

      MotorCommand motor_command_tmp;
      for (int i = 0; i < POLICY_ACTION_DIM; i++) {
        last_action[i] = static_cast<double>(command_action[i]);
      }

      if (use_h2_native_) {
        for (int policy_index = 0; policy_index < POLICY_ACTION_DIM; ++policy_index) {
          const int h2_dds_index = h2_native_policy_to_h2_dds[policy_index];
          motor_command_tmp.q_target.at(h2_dds_index) = static_cast<float>(
              h2_native_default_angles_policy[policy_index] +
              h2_native_action_scale_policy[policy_index] * command_action[policy_index]);
          motor_command_tmp.tau_ff.at(h2_dds_index) = 0.0;
          motor_command_tmp.kp.at(h2_dds_index) = H2NativeKpDds(h2_dds_index);
          motor_command_tmp.kd.at(h2_dds_index) = H2NativeKdDds(h2_dds_index);
          motor_command_tmp.dq_target.at(h2_dds_index) = 0.0;
        }
        for (size_t i = 0; i < h2_head_mujoco_indices.size(); ++i) {
          const int h2_head_local_index = h2_head_mujoco_indices[i];
          const int h2_head_dds_index = h2_head_dds_indices[i];
          motor_command_tmp.q_target.at(h2_head_dds_index) =
              static_cast<float>(h2_default_angles[h2_head_local_index]);
          motor_command_tmp.tau_ff.at(h2_head_dds_index) = 0.0;
          motor_command_tmp.kp.at(h2_head_dds_index) = H2NativeKpDds(h2_head_dds_index);
          motor_command_tmp.kd.at(h2_head_dds_index) = H2NativeKdDds(h2_head_dds_index);
          motor_command_tmp.dq_target.at(h2_head_dds_index) = 0.0;
        }
        ApplyH2HeadCommand(motor_command_tmp);
      } else if (use_h2_lora_) {
        std::array<double, POLICY_ACTION_DIM> g1_delta_semantic = {0.0};
        for (int i = 0; i < POLICY_ACTION_DIM; i++) {
          g1_delta_semantic[i] = static_cast<double>(command_action[i]) * G1ActionScaleSemantic(i);
        }
        const auto h2_delta_physical = G1SemanticToH2Physical(g1_delta_semantic);

        for (int i = 0; i < POLICY_ACTION_DIM; i++) {
          const int h2_local_index = h2lora_body_to_h2_mujoco[i];
          const int h2_dds_index = h2lora_body_to_h2_dds[i];
          motor_command_tmp.q_target.at(h2_dds_index) =
              static_cast<float>(h2_default_angles[h2_local_index] + h2_delta_physical[i]);
          motor_command_tmp.tau_ff.at(h2_dds_index) = 0.0;
          motor_command_tmp.kp.at(h2_dds_index) = H2KpDds(h2_dds_index);
          motor_command_tmp.kd.at(h2_dds_index) = H2KdDds(h2_dds_index);
          motor_command_tmp.dq_target.at(h2_dds_index) = 0.0;
        }
        for (size_t i = 0; i < h2_head_mujoco_indices.size(); ++i) {
          const int h2_head_local_index = h2_head_mujoco_indices[i];
          const int h2_head_dds_index = h2_head_dds_indices[i];
          motor_command_tmp.q_target.at(h2_head_dds_index) =
              static_cast<float>(h2_default_angles[h2_head_local_index]);
          motor_command_tmp.tau_ff.at(h2_head_dds_index) = 0.0;
          motor_command_tmp.kp.at(h2_head_dds_index) = H2KpDds(h2_head_dds_index);
          motor_command_tmp.kd.at(h2_head_dds_index) = H2KdDds(h2_head_dds_index);
          motor_command_tmp.dq_target.at(h2_head_dds_index) = 0.0;
        }
        ApplyH2HeadCommand(motor_command_tmp);
      } else {
        for (int i = 0; i < G1_NUM_MOTOR; i++) {
          const double action_value = static_cast<double>(command_action[isaaclab_to_mujoco[i]]) * g1_action_scale[i];
          motor_command_tmp.q_target.at(i) = static_cast<float>(default_angles[i] + action_value);
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = kps[i];
          motor_command_tmp.kd.at(i) = kds[i];
          motor_command_tmp.dq_target.at(i) = 0.0;
        }
      }
      motor_command_buffer_.SetData(motor_command_tmp);
      return true;
    }

    /**
     * @brief Advance the playback cursor and blend planner output.
     *
     * Called at the end of each control tick.  Two cases:
     *
     * 1. **Planner active**: Checks if a new planner animation is available.
     *    If so, blends it into the existing planner_motion_ using a linear
     *    cross-fade over 8 frames.  Then advances current_frame_ by 1.
     *
     * 2. **Reference-motion mode**: Advances current_frame_ by 1.  Clamps or
     *    resets to 0 when the motion ends (except for "streamed" motions,
     *    which hold the last valid frame while waiting for new data).
     *
     * @return True on success.
     */
    bool CurrentFrameAdvancement() {
      // get current motion and frame from planner when planner is enabled and initialized
      std::lock_guard<std::mutex> motion_lock(current_motion_mutex_);
      if (planner_ && planner_->planner_state_.enabled && planner_->planner_state_.initialized) {
        std::lock_guard<std::mutex> planner_lock(planner_->planner_motion_mutex_);
        
        // Check if the planner thread has finished generating a valid animation:
        if (planner_->motion_available_ && planner_->planner_motion_50hz_.timesteps > 0) {
          
          const auto &planner_motion_gen = planner_->planner_motion_50hz_;
          planner_->motion_available_ = false;

          // special case if this is the first time we're receiving a planner motion:
          bool is_the_first_time = false;
          bool success = false;
          if(planner_motion_->timesteps == 0)
          {
            // step through the frames and copy the animation over:
            for(int f = 0; f < planner_motion_gen.timesteps; ++f)
            {
              std::copy(
                planner_motion_gen.JointPositions(f),
                planner_motion_gen.JointPositions(f) + planner_motion_gen.GetNumJoints(),
                planner_motion_->JointPositions(f)
              );

              std::copy(
                planner_motion_gen.JointVelocities(f),
                planner_motion_gen.JointVelocities(f) + planner_motion_gen.GetNumJoints(),
                planner_motion_->JointVelocities(f)
              );

              planner_motion_->BodyPositions(f)[0] = planner_motion_gen.BodyPositions(f)[0];
              planner_motion_->BodyQuaternions(f)[0] = planner_motion_gen.BodyQuaternions(f)[0];
            }
            planner_motion_->timesteps = planner_motion_gen.timesteps;
            success = true;
            is_the_first_time = true;
          }
          else
          {

            auto fgen = planner_->gen_frame_;

            // Ok, a new planner animation is available. We want to do these things:
            //  * rebase the old planner_motion_ so current_frame_ is at frame 0
            //  * fade in the new planner animation, starting at frame fgen - current_frame_
            //    (or frame 0 if fgen < current_frame_)

            // The newly generated animation starts at frame fgen - current_frame_ on this
            // new timeline, so the total length of the animation will be this:
            auto new_anim_length = fgen - current_frame_ + planner_motion_gen.timesteps;

            // guess we've got to sit this out if things arrived so late that we don't have any frames
            // we can copy over...
            if(new_anim_length > 0)
            {
              // We want to insert a region where the old animation is blended with the new one
              // using a linearly decaying weight. If current_frame_ > fgen, this needs to start
              // immediately, otherwise it should start at at frame fgen - current_frame_:
              auto blend_start_frame = std::max(0, fgen - current_frame_);

              // choosing 8 for the width of the blend region because this is about the size of the
              // context we use for conditioning the planner model
              const int blend_num_frames = 8;

              // step through the frames and blend the animations:
              for(int f = 0; f < new_anim_length; ++f)
              {
                // corresponding frames in the old animation and the new one from the
                // planner that we're blending in:
                int f_old = f + current_frame_;
                int f_new = f_old - fgen;

                // clamp to valid ranges:
                f_old = std::clamp(f_old, 0, planner_motion_->timesteps - 1);
                f_new = std::clamp(f_new, 0, planner_motion_gen.timesteps - 1);
                
                // calculate linearly decaying blend weight:
                double w_new = double(f - blend_start_frame) / blend_num_frames;
                w_new = std::clamp(w_new, 0.0, 1.0);
                double w_old = 1.0 - w_new;

                // blend joint_positions + joint_velocities:
                for(size_t j=0; j < planner_motion_->GetNumJoints(); ++j)
                {
                  planner_motion_->JointPositions(f)[j] = 
                    w_old * planner_motion_->JointPositions(f_old)[j] +
                    w_new * planner_motion_gen.JointPositions(f_new)[j];
                  planner_motion_->JointVelocities(f)[j] = 
                      w_old * planner_motion_->JointVelocities(f_old)[j] +
                      w_new * planner_motion_gen.JointVelocities(f_new)[j];
                }

                // blend the global xyz position:
                for(int j=0; j < 3; ++j)
                {
                  planner_motion_->BodyPositions(f)[0][j] = 
                      w_old * planner_motion_->BodyPositions(f_old)[0][j] +
                      w_new * planner_motion_gen.BodyPositions(f_new)[0][j];
                }
                
                // blend the global rotation quaternion using a slerp:
                std::array<double, 4> q0 = planner_motion_->BodyQuaternions(f_old)[0];
                std::array<double, 4> q1 = planner_motion_gen.BodyQuaternions(f_new)[0];
                planner_motion_->BodyQuaternions(f)[0] = quat_slerp_d(q0, q1, w_new);
              }

              // set the new timestep count and reset the current frame to 0:
              planner_motion_->timesteps = new_anim_length;
              success = true;
            }
          }

          if(success)
          {
            if(planner_motion_file_)
            {
              // log the motion that we just copied/blended over:
              for(int frame = 0; frame < planner_motion_->timesteps; frame++) {
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][0] << ",";
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][1] << ",";
                (*planner_motion_file_) << planner_motion_->BodyPositions(frame)[0][2] << ",";

                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][0] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][1] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][2] << ",";
                (*planner_motion_file_) << planner_motion_->BodyQuaternions(frame)[0][3] << ",";

                for(int i = 0; i < 29; i++) {
                  (*planner_motion_file_) << planner_motion_->JointPositions(frame)[isaaclab_to_mujoco[i]] << ",";
                }

                (*planner_motion_file_) << std::endl;
              }

              // insert a blank line to separate motions:
              (*planner_motion_file_) << std::endl;
            }

            // switch the current motion to planner_motion:
            current_frame_ = 0;
            // Assign shared_ptr directly - planner_motion_ is already a shared_ptr
            current_motion_ = planner_motion_;
            idle_readapt_stored_ = false;
            if(is_the_first_time) {
              reinitialize_heading_ = true;
            }
          }
        }
          
        // Frame advancement at control frequency (50Hz) for smooth playback
        if (current_motion_->timesteps > 0 && operator_state.play) {
          int new_frame = current_frame_ + 1;
            if (new_frame >= current_motion_->timesteps) {
              new_frame = current_motion_->timesteps - 1; // Clamp to last frame
              // Error-based readaptation in idle mode with double-threshold state machine.
              // IDLE: do nothing. ADAPTING: blend toward robot state. RECOVERING: blend toward planner target.
              auto movement_state_data = movement_state_buffer_.GetDataWithTime().data;
              if(movement_state_data && movement_state_data->locomotion_mode == static_cast<int>(LocomotionMode::IDLE)) {
                auto low_state = low_state_buffer_.GetDataWithTime().data;
                if (low_state) {
                  const auto actual_joint_positions =
                      LowStateToG1PlannerSemantic(*low_state);
                  const bool project_reference =
                      use_h2_lora_ && project_h2_lora_planner_reference_;
                  std::array<double, POLICY_ACTION_DIM> raw_planner_target = {};
                  std::copy_n(
                      planner_motion_->JointPositions(new_frame),
                      POLICY_ACTION_DIM,
                      raw_planner_target.begin());
                  const auto policy_planner_target = project_reference
                      ? G1PlannerReferenceToH2LoraSemantic(raw_planner_target)
                      : raw_planner_target;
                  const auto raw_adaptation_target = project_reference
                      ? H2LoraSemanticToG1PlannerReference(actual_joint_positions)
                      : actual_joint_positions;

                  // Store original planner targets on first entry
                  if (!idle_readapt_stored_) {
                    for (int idx : lower_body_joint_isaaclab_order_in_isaaclab_index) {
                      idle_readapt_original_targets_[idx] = planner_motion_->JointPositions(new_frame)[idx];
                    }
                    idle_readapt_stored_ = true;
                    idle_readapt_state_ = IdleReadaptState::IDLE;
                  }

                  // Compute average error across all lower-body joints
                  double total_error = 0.0;
                  for (int idx : lower_body_joint_isaaclab_order_in_isaaclab_index) {
                    total_error += std::abs(policy_planner_target.at(idx)
                                            - actual_joint_positions.at(idx));
                  }
                  double avg_error = total_error / static_cast<double>(lower_body_joint_isaaclab_order_in_isaaclab_index.size());

                  // State transitions
                  switch (idle_readapt_state_) {
                    case IdleReadaptState::IDLE:
                      if (avg_error > kAdaptTrigger) {
                        idle_readapt_state_ = IdleReadaptState::ADAPTING;
                      } else if (avg_error < kRecoverTrigger) {
                        idle_readapt_state_ = IdleReadaptState::RECOVERING;
                      }
                      break;
                    case IdleReadaptState::ADAPTING:
                      if (avg_error < kAdaptStop) {
                        idle_readapt_state_ = IdleReadaptState::IDLE;
                      }
                      break;
                    case IdleReadaptState::RECOVERING:
                      if (avg_error > kAdaptTrigger) {
                        idle_readapt_state_ = IdleReadaptState::ADAPTING;
                      }
                      break;
                  }

                  // Apply blend based on current state
                  if (idle_readapt_state_ == IdleReadaptState::ADAPTING) {
                    for (int idx : lower_body_joint_isaaclab_order_in_isaaclab_index) {
                      planner_motion_->JointPositions(new_frame)[idx] =
                          0.98 * planner_motion_->JointPositions(new_frame)[idx]
                          + 0.02 * raw_adaptation_target.at(idx);
                    }
                  } else if (idle_readapt_state_ == IdleReadaptState::RECOVERING) {
                    for (int idx : lower_body_joint_isaaclab_order_in_isaaclab_index) {
                      planner_motion_->JointPositions(new_frame)[idx] =
                          0.98 * planner_motion_->JointPositions(new_frame)[idx] + 0.02 * idle_readapt_original_targets_[idx];
                    }
                  }
                  // IDLE state: do nothing
                }
              }
            }
          current_frame_ = new_frame;
        }
      }

      // Update motion frame playback for non-planner motion
      bool use_planner_motion = (current_motion_ && current_motion_ == planner_motion_);
      if (!use_planner_motion && current_motion_ && current_motion_->timesteps > 0 && operator_state.play) {
        // Update display motion current frame
        current_frame_++;
        // Check if motion completed (reached the end)
        if (current_motion_->name != "streamed") {
          if (current_frame_ >= current_motion_->timesteps) {
            operator_state.play = false;
            if (current_motion_ ->name != "temporary_motion") {
              std::cout << "______________________________________________________" << std::endl;
              std::cout << "Motion index: " << motion_reader_.current_motion_index_ << " : " << current_motion_->name << " completed." << std::endl;
              std::cout << "______________________________________________________" << std::endl;
            } else {
              std::cout << "Temporary motion completed." << std::endl;
            }
            current_frame_ = 0; // Reset to beginning
            // Total reset: both base quaternion and delta heading
            reinitialize_heading_ = true;
            std::cout << "Reset to frame 0." << std::endl;
          }
        } else {
          if (current_frame_ >= current_motion_->timesteps - saved_frame_for_observation_window_) {
            current_frame_ = current_frame_ - 1;
            std::cout << "Motion " << current_motion_->name << " completed and waiting following motion" << std::endl;                    
          }
        }
      }
      return true;
    }

    /**
     * @brief Input thread body (100 Hz).
     *
     * 1. Calls input_interface_->update() to poll keyboard / gamepad / network.
     * 2. Forwards raw gamepad data from LowState to the active interface.
     * 3. Calls input_interface_->handle_input() to translate flags into
     *    system-state changes (motion switching, planner commands, etc.).
     * 4. Optionally records / plays back input state for offline replay.
     */
    void Input() {
      if (operator_state.stop) { return; }
      
      // Update input interface (poll for new data)
      input_interface_->update();

      // Handle input using the interface - input handler updates everything directly
      const std::shared_ptr<const LowState_> low_state_data = low_state_buffer_.GetDataWithTime().data;
      std::array<double, 4> current_quat = {0.0, 0.0, 0.0, 1.0};
      if(low_state_data) {
        current_quat = float_to_double<4>(low_state_data->imu_state().quaternion());
      }
      
      // Update gamepad data if using gamepad interface (or manager)
      if (auto gamepad = dynamic_cast<unitree::common::Gamepad*>(input_interface_.get())) {
        if(low_state_data) {
          memcpy(gamepad->gamepad_data.buff, &low_state_data->wireless_remote()[0], 40);
        }else{
          for(int i = 0; i < 40; i++) {
            gamepad->gamepad_data.buff[i] = 0;
          }
        }
      }
      else if (auto manager = dynamic_cast<InterfaceManager*>(input_interface_.get())) {
        if (low_state_data) {
          manager->UpdateGamepadRemoteData(&low_state_data->wireless_remote()[0], 40);
        } else {
          uint8_t zeros[40] = {0};
          manager->UpdateGamepadRemoteData(zeros, 40);
        }
      }
      else if (auto gamepad_mgr = dynamic_cast<GamepadManager*>(input_interface_.get())) {
        if (low_state_data) {
          gamepad_mgr->UpdateGamepadRemoteData(&low_state_data->wireless_remote()[0], 40);
        } else {
          uint8_t zeros[40] = {0};
          gamepad_mgr->UpdateGamepadRemoteData(zeros, 40);
        }
      }
    
     
      bool has_planner = static_cast<bool>(planner_);
      
      if (has_planner) {
        input_interface_->handle_input(motion_reader_, current_motion_, current_frame_, operator_state,
                                      reinitialize_heading_, heading_state_buffer_, has_planner, planner_->planner_state_, movement_state_buffer_, current_motion_mutex_, report_temperature_);
      } else {
        PlannerState planner_state{false, false};
        input_interface_->handle_input(motion_reader_, current_motion_, current_frame_, operator_state,
                                      reinitialize_heading_, heading_state_buffer_, has_planner, planner_state, movement_state_buffer_, current_motion_mutex_, report_temperature_);
      }

      if (playback_input_file_ && program_state_ == ProgramState::CONTROL) {
        std::string line;
        PlannerState ps{false, false};
        auto &planner_state = has_planner ? planner_->planner_state_ : ps;
        if (std::getline(*playback_input_file_, line)) {

          MovementState movement_state;
          std::istringstream iss(line);
          std::string token;
          std::getline(iss, token, ',');
          motion_reader_.current_motion_index_ = std::stoi(token);
          std::getline(iss, token, ',');
          //current_frame_ = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.play = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.start = std::stoi(token);
          std::getline(iss, token, ',');
          operator_state.stop = std::stoi(token);
          std::getline(iss, token, ',');
          planner_state.enabled = std::stoi(token);
          std::getline(iss, token, ',');
          planner_state.initialized = std::stoi(token);
          std::getline(iss, token, ',');
          movement_state.locomotion_mode = std::stoi(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[0] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[1] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_direction[2] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[0] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[1] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.facing_direction[2] = std::stof(token);
          std::getline(iss, token, ',');
          movement_state.movement_speed = std::stof(token);
          movement_state_buffer_.SetData(movement_state);

          if(!(has_planner && planner_state.enabled && planner_state.initialized)) {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
          }
        }
      }

      if (record_input_file_ && program_state_ == ProgramState::CONTROL) {
        auto movement_state_data = movement_state_buffer_.GetDataWithTime();
        PlannerState ps{false, false};
        const auto &planner_state = has_planner ? planner_->planner_state_ : ps;
        (*record_input_file_) << motion_reader_.current_motion_index_ << ",";
        (*record_input_file_) << current_frame_ << ",";
        (*record_input_file_) << operator_state.play << ",";
        (*record_input_file_) << operator_state.start << ",";
        (*record_input_file_) << operator_state.stop << ",";

        (*record_input_file_) << planner_state.enabled.load() << ",";
        (*record_input_file_) << planner_state.initialized.load() << ",";
        
        (*record_input_file_) << movement_state_data.data->locomotion_mode << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[0] << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[1] << ",";
        (*record_input_file_) << movement_state_data.data->movement_direction[2] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[0] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[1] << ",";
        (*record_input_file_) << movement_state_data.data->facing_direction[2] << ",";
        (*record_input_file_) << movement_state_data.data->movement_speed << ",";
        (*record_input_file_) << movement_state_data.data->height << ",";
        (*record_input_file_) << std::endl;
      }

    }

    /**
     * @brief Planner thread body (10 Hz).
     *
     * When the planner is enabled:
     *  - First call: initialises the planner with the robot's current pose
     *    (base quaternion + joint positions) and runs an initial inference.
     *  - Subsequent calls: reads the latest MovementState from the
     *    movement_state_buffer_, decides whether a replan is needed (mode
     *    change, direction change, or periodic timer), and if so calls
     *    UpdatePlanning() which runs TensorRT inference and produces a new
     *    30 Hz trajectory (resampled to 50 Hz by the base class).
     *
     * Replan intervals vary by locomotion type:
     *  - Running: every 0.1 s
     *  - Crawling: every 0.2 s
     *  - Boxing: every 1.0 s
     *  - Default: every 1.0 s
     *
     * The generated animation is stored in planner_->planner_motion_50hz_
     * and picked up by CurrentFrameAdvancement() in the control thread.
     */
    void Planner() {
      if (operator_state.stop) { return; }
      auto low_state_data = low_state_buffer_.GetDataWithTime();
      const std::shared_ptr<const LowState_> ls = low_state_data.data;
      
      if (planner_ && planner_->planner_state_.enabled) {
        if (ls) {
          if (!planner_->planner_state_.initialized) {
            try {
              std::cout << "Initializing planner..." << std::endl;
              planner_motion_->timesteps = 0;
              std::cout << "Setting planner_motion encode mode to " << planner_encoder_mode_ << std::endl;
              planner_motion_->SetEncodeMode(planner_encoder_mode_);
              // Get base quaternion and joint positions from robot state
              std::array<double, 4> base_quat = float_to_double<4>(ls->imu_state().quaternion());
              const auto planner_semantic_positions =
                  LowStateToG1PlannerSemantic(*ls);
              const auto joint_positions =
                  G1SemanticToMujocoOrder(planner_semantic_positions);
              if (use_h2_robot_) {
                std::cout << "[H2 Planner] Initialized G1 context from H2 physical state via Phi"
                          << std::endl;
              }
              // Initialize planner with robot state
              if(!planner_->Initialize(base_quat, joint_positions)) {
                throw std::runtime_error("Error when initializing planner");
              }

              std::cout << "Planner initialized successfully!" << std::endl;
              
              // Start recording session for planner motion (if enabled)
              if (enable_motion_recording_) {
                if (planner_motion_recorder_.IsActive()) {
                  planner_motion_recorder_.Finalize();
                }
                planner_motion_recorder_.StartSession("planner_motion");
              }
              
            } catch (const std::exception& e) {
              std::cout << "✗ Error initializing planner: " << e.what() << std::endl;
              std::cout << "Disabling planner to prevent further errors..." << std::endl;
              planner_->planner_state_.enabled = false;
              planner_->planner_state_.initialized = false;
              planner_motion_->timesteps = 0;
              // reset operator state and current motion and frame
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex_);
                operator_state.play = false;
                current_frame_ = 0;
                current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
              }
              // Finalize recording if initialization failed
              if (planner_motion_recorder_.IsActive()) {
                planner_motion_recorder_.Finalize();
              }
              
              return;
            }
          } 
          else {
            if (planner_motion_->timesteps == 0) {
              std::cout << "Planner() no data in planner_motion_, returning. If control loop is started, this should only happen once when planner is initialized. If control loop is not started, this should keep returning until control loop is started." << std::endl;
              return;
            }
            bool need_replan = false;

            // bool of different reason to replan
            bool time_to_replan = false;
            bool facing_direction_changed = false;
            bool height_changed = false;
            bool movement_mode_changed = false;
            bool movement_speed_changed = false;
            bool movement_direction_changed = false;
            bool under_static_motion_mode = is_static_motion_mode(static_cast<LocomotionMode>(last_movement_state_.locomotion_mode));

            // Read current movement mode from thread-safe buffer
            auto movement_state_data = movement_state_buffer_.GetDataWithTime();

            if (movement_state_data.data) {
              // bool of checking if facing direction is changed
              facing_direction_changed = movement_state_data.data->facing_direction[0] != last_movement_state_.facing_direction[0] || 
                                          movement_state_data.data->facing_direction[1] != last_movement_state_.facing_direction[1] || 
                                          movement_state_data.data->facing_direction[2] != last_movement_state_.facing_direction[2];
              // bool of checking if height is changed
              height_changed = movement_state_data.data->height != last_movement_state_.height;
              // bool of checking if movement mode is changed
              movement_mode_changed = movement_state_data.data->locomotion_mode != last_movement_state_.locomotion_mode;
              // bool of checking if movement speed is changed
              movement_speed_changed = movement_state_data.data->movement_speed != last_movement_state_.movement_speed;
              // bool of checking if movement direction is changed
              movement_direction_changed = movement_state_data.data->movement_direction[0] != last_movement_state_.movement_direction[0] || 
                                          movement_state_data.data->movement_direction[1] != last_movement_state_.movement_direction[1] || 
                                          movement_state_data.data->movement_direction[2] != last_movement_state_.movement_direction[2];
              // bool of checking if the last movement value is under the static motion mode
              under_static_motion_mode = is_static_motion_mode(static_cast<LocomotionMode>(movement_state_data.data->locomotion_mode));
            }
            bool is_running = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RUN);
            bool is_boxing = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::LEFT_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RIGHT_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RANDOM_PUNCH) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::LEFT_HOOK) ||
                             movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::RIGHT_HOOK);
            bool is_crawling = movement_state_data.data->locomotion_mode == static_cast<int>(LocomotionMode::CRAWLING);
            // increment replan interval counter
            replan_interval_counter_ += planner_dt_;
            // check if need to replan
            if(is_running) {
              if(replan_interval_counter_ >= replan_interval_running_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else if(is_crawling) {
              if(replan_interval_counter_ >= replan_interval_crawling_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else if(is_boxing) {
              if(replan_interval_counter_ >= replan_interval_boxing_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            } else {
              if(replan_interval_counter_ >= replan_interval_) {
                // reset replan interval counter
                replan_interval_counter_ = 0.0f;
                // set need to replan to true
                time_to_replan = true;
              }
            }
            
            if (movement_mode_changed || facing_direction_changed || height_changed) {
              need_replan = true;
            } else if (!under_static_motion_mode && (movement_speed_changed || movement_direction_changed || (time_to_replan && movement_state_data.data->movement_speed != 0))) {
              need_replan = true;
            }

            if (!need_replan) {
              // no need to replan, return
              return;
            }

            auto gather_input_start_time = std::chrono::steady_clock::now(); 
            
            int current_mode = static_cast<int>(LocomotionMode::IDLE);
            std::array<float, 3> movement_direction = {0.0f, 0.0f, 0.0f};
            std::array<float, 3> facing_direction = {1.0f, 0.0f, 0.0f};  // Default forward
            float movement_speed = -1.0f;
            float target_height = -1.0f;
            
            if (movement_state_data.data) {
              current_mode = movement_state_data.data->locomotion_mode;
              movement_direction[0] = movement_state_data.data->movement_direction[0];
              movement_direction[1] = movement_state_data.data->movement_direction[1]; 
              movement_direction[2] = movement_state_data.data->movement_direction[2];
              facing_direction[0] = movement_state_data.data->facing_direction[0];
              facing_direction[1] = movement_state_data.data->facing_direction[1];
              facing_direction[2] = movement_state_data.data->facing_direction[2];
              movement_speed = movement_state_data.data->movement_speed;
              target_height = movement_state_data.data->height;

              // Update last movement state for next iteration comparison (update here to avoid missing the update)
              last_movement_state_.locomotion_mode = movement_state_data.data->locomotion_mode;
              last_movement_state_.movement_direction = movement_state_data.data->movement_direction;
              last_movement_state_.facing_direction = movement_state_data.data->facing_direction;
              last_movement_state_.movement_speed = movement_state_data.data->movement_speed;
              last_movement_state_.height = movement_state_data.data->height;
            }
            
            try {
              // Update planning with current movement parameters

              // gotta make sure planner_motion actually has data...
              if(!planner_->UpdatePlanning(
                current_frame_,
                planner_motion_,
                current_mode,
                movement_speed,
                target_height,
                movement_direction,
                facing_direction
              )) {
                throw std::runtime_error("Error when updating planner");
              }
              
            } catch (const std::exception& e) {
              std::cout << "✗ Error during planning update: " << e.what() << std::endl;
              std::cout << "Disabling planner to prevent further errors..." << std::endl;
              planner_->planner_state_.enabled = false;
              planner_->planner_state_.initialized = false;
              planner_motion_->timesteps = 0;
              // reset operator state and current motion and frame
              {
                std::lock_guard<std::mutex> lock(current_motion_mutex_);
                operator_state.play = false;
                current_frame_ = 0;
                current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
              }
              // Finalize recording if planning update failed
              if (planner_motion_recorder_.IsActive()) {
                planner_motion_recorder_.Finalize();
              }
              
              return;
            }
          }
        }
      } else if (planner_ && !planner_->planner_state_.enabled) {
        // since the planner is disabled, we make sure to reset initialized and timesteps
        planner_->planner_state_.initialized = false;
        
        // Finalize planner motion recording when planner is disabled
        if (planner_motion_recorder_.IsActive()) {
          planner_motion_recorder_.Finalize();
        }
        
        return;
      }
    }

    /**
     * @brief Control thread body (50 Hz) — the main real-time control loop.
     *
     * State machine:
     *  - INIT: ramps the robot to the default standing pose (InitControl).
     *  - WAIT_FOR_CONTROL: idle, waiting for operator "start" signal.
     *  - CONTROL: each tick performs the full pipeline:
     *    1. GatherRobotStateToLogger — read IMU + joints, log to StateLogger.
     *    2. GatherInputInterfaceData — snapshot VR / hand / token data.
     *    3. GatherObservations — fill the policy observation vector.
     *    4. LogPostState — append encoder token to the latest log entry.
     *    5. CreatePolicyCommand — run TensorRT policy, produce MotorCommand.
     *    6. Update the selected hand backend (max-close ratio + targets).
     *    7. Publish state to all output interfaces (ZMQ / ROS2).
     *    8. Handle motion recording (streamed + planner).
     *    9. CurrentFrameAdvancement — advance playback cursor, blend planner.
     *    10. Periodic timing log every 50 ticks (~1 s).
     */
    void Control() {
      if (operator_state.stop) { return; }

      static const auto env_seconds_or_disabled = [](const char* name) {
        const char* env = std::getenv(name);
        if (env == nullptr || std::string(env).empty()) {
          return -1.0;
        }
        try {
          return std::stod(std::string(env));
        } catch (...) {
          std::cerr << "⚠ Warning: invalid " << name << "='" << env
                    << "', disabling this auto-control hook" << std::endl;
          return -1.0;
        }
      };
      static const auto env_seconds_list = []() {
        std::vector<double> values;
        const char* env = std::getenv("GEAR_SONIC_AUTO_NEXT_AFTER_SEC");
        if (env == nullptr || std::string(env).empty()) {
          return values;
        }
        std::stringstream ss{std::string(env)};
        std::string token;
        while (std::getline(ss, token, ',')) {
          token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
          if (token.empty()) {
            continue;
          }
          try {
            values.push_back(std::stod(token));
          } catch (...) {
            std::cerr << "⚠ Warning: invalid GEAR_SONIC_AUTO_NEXT_AFTER_SEC entry='"
                      << token << "', ignoring" << std::endl;
          }
        }
        return values;
      };
      static const double auto_start_after_sec =
          env_seconds_or_disabled("GEAR_SONIC_AUTO_START_AFTER_SEC");
      static const double auto_play_after_sec =
          env_seconds_or_disabled("GEAR_SONIC_AUTO_PLAY_AFTER_SEC");
      static const std::vector<double> auto_next_after_sec_list = env_seconds_list();
      static const double auto_stop_after_sec =
          env_seconds_or_disabled("GEAR_SONIC_AUTO_STOP_AFTER_SEC");
      const double wall_elapsed_sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start_time_).count();
      if (auto_stop_after_sec >= 0.0 && wall_elapsed_sec >= auto_stop_after_sec) {
        operator_state.stop = true;
        return;
      }
      if (auto_start_after_sec >= 0.0 && wall_elapsed_sec >= auto_start_after_sec &&
          program_state_ == ProgramState::WAIT_FOR_CONTROL) {
        operator_state.start = true;
      }
      if (auto_play_after_sec >= 0.0 && !auto_play_triggered_ &&
          wall_elapsed_sec >= auto_play_after_sec &&
          program_state_ == ProgramState::CONTROL && !operator_state.play) {
        auto_play_triggered_ = true;
        operator_state.play = true;
      }
      if (!auto_next_triggered_ &&
          auto_next_trigger_count_ < auto_next_after_sec_list.size() &&
          wall_elapsed_sec >= auto_next_after_sec_list[auto_next_trigger_count_] &&
          program_state_ == ProgramState::CONTROL && current_motion_ &&
          motion_reader_.motions.size() > 1) {
        auto_next_trigger_count_ += 1;
        auto_next_triggered_ =
            auto_next_trigger_count_ >= auto_next_after_sec_list.size();
        {
          std::lock_guard<std::mutex> lock(current_motion_mutex_);
          motion_reader_.current_motion_index_ =
              (motion_reader_.current_motion_index_ + 1) % motion_reader_.motions.size();
          current_motion_ = motion_reader_.GetMotionShared(motion_reader_.current_motion_index_);
          current_frame_ = 0;
          operator_state.play = false;
          reinitialize_heading_ = true;
          std::fill(last_action.begin(), last_action.end(), 0.0);
          std::cout << "[Auto Control] Switched to next motion "
                    << motion_reader_.current_motion_index_ << ": "
                    << current_motion_->name << std::endl;
        }
        operator_state.play = true;
      }

      switch (program_state_) {
        case ProgramState::INIT:
          if (!InitControl()) {
            std::cout << "LowState is not available, waiting for robot to be ready" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          // Re-publish robot_config so late-joining subscribers can receive it
          // before the policy is activated (ZMQ PUB has no persistence).
          for (auto& oi : output_interfaces_) { if (oi) oi->publish_config(); }
          break;

        case ProgramState::WAIT_FOR_CONTROL:
          if (!CheckSafety()) {
            std::cout << "[ERROR] Safety check failed, cannot start control." << std::endl;
            operator_state.stop = true;
            break;
          }

          // Re-publish robot_config so late-joining subscribers can receive it
          // before the policy is activated (ZMQ PUB has no persistence).
          for (auto& oi : output_interfaces_) { if (oi) oi->publish_config(); }
          if (operator_state.start) {
            // Warn if starting control in token mode without tokens, but allow it
            if (initial_encoder_mode_ == -1 && !first_token_received_) {
              static int warn_count = 0;
              if (warn_count % 50 == 0) {  // Warn every ~1 second
                std::cout << "⚠⚠⚠ [Token Safety] WARNING: Starting control before tokens arrive! "
                          << "Robot will use zero tokens until tokens start streaming." << std::endl;
              }
              warn_count++;
            }
            std::cout << "[Control] Starting policy control" << std::endl;
            program_state_ = ProgramState::CONTROL;
          }
          break;

        case ProgramState::CONTROL: {
          if (!CheckSafety()) {
            std::cout << "[ERROR] Safety check failed, stopping control." << std::endl;
            operator_state.stop = true;
            break;
          }

          // NEW: Get data with timestamps for loop timing analysis
          auto obs_start_time = std::chrono::steady_clock::now();
          // Increment independent logging counter every control iteration
          logging_counter_++;

          if (!GatherRobotStateToLogger()) {
            std::cout << "✗ Error: Failed to gather robot state to logger in the middle of the control loop!" << std::endl;
            operator_state.stop = true;
            std::cout << "Stopping control system." << std::endl;
            return;
          }

          // Handle temperature report request (F key)
          if (report_temperature_) {
            report_temperature_ = false;
            const auto ls = used_low_state_data_.data;
            if (ls) {
              std::cout << "\n[Temperature Report] Motor temperatures (winding / driver):" << std::endl;
              std::vector<std::pair<int16_t, int>> max_temps;
              max_temps.reserve(static_cast<size_t>(robot_motor_count_));
              for (int i = 0; i < robot_motor_count_; ++i) {
                auto temp = ls->motor_state()[i].temperature();
                std::cout << "  " << std::setw(2) << i << " " << std::setw(22) << std::left
                          << ProfileJointNameDds(i) << std::right << ": "
                          << std::setw(4) << temp[0] << " / " << std::setw(4) << temp[1] << std::endl;
                max_temps.push_back({std::max(temp[0], temp[1]), i});
              }
              std::cout << std::endl;

              std::sort(max_temps.begin(), max_temps.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
              std::string tts = "Hottest motors: ";
              for (int k = 0; k < 3 && k < static_cast<int>(max_temps.size()); ++k) {
                auto [temp_val, idx] = max_temps[k];
                if (k > 0) tts += ", ";
                tts += std::string(ProfileJointNameDds(idx)) + " " + std::to_string(temp_val);
              }
              pending_tts_ = tts;
            }
          }

          if (!GatherInputInterfaceData()) {
            return;
          }

          // Lock mutex for observation gathering and output sending to ensure consistency
          // This ensures all observations use the same snapshot of current_motion_ and current_frame_
          int current_frame_copy;
          int current_encoder_mode_copy;
          bool current_play_copy;
          std::shared_ptr<const MotionSequence> current_motion_copy = nullptr;
          {
            std::lock_guard<std::mutex> lock(current_motion_mutex_);
            current_frame_copy = current_frame_;
            current_motion_copy = current_motion_;
            if (current_motion_copy) {
              ConfigureReferenceFutureSteps(*current_motion_copy);
            }
            current_encoder_mode_copy = current_motion_copy->GetEncodeMode();
            current_play_copy = operator_state.play;

            // reset saved frame for observation window before gathering observations so that we can get the latest value
            saved_frame_for_observation_window_ = 0;

            // Update heading state (handles reinitialize_heading_ and frame-0 init)
            // Must run before any orientation observation functions
            UpdateHeadingState();

            // Gather all observations using modular functions
            // This may update token_state_data_ with encoder output (if encoder is used)
            if (!GatherObservations()) {
              std::cout << "✗ Error: Failed to gather observations in the middle of the control loop!" << std::endl;
              std::cout << "Stopping control system." << std::endl;
              operator_state.stop = true;
              return;
            }
          } // Release lock after all observation-dependent operations

          // Log post-state data (token state) to the most recent state logger entry
          // This must be called after GatherObservations() which populates token_state_data_
          if (state_logger_) {
            std::string motion_name = current_motion_copy ? current_motion_copy->name : "";
            if (!state_logger_->LogPostState(std::span(token_state_data_), current_encoder_mode_copy, motion_name, current_play_copy)) {
              std::cerr << "[WARNING] Failed to log token state to state logger" << std::endl;
            }
          }

          auto obs_end_time = std::chrono::steady_clock::now();

          if (!CreatePolicyCommand()) {
            std::cout << "✗ Error: Failed to create policy command in the middle of the control loop!" << std::endl;
            std::cout << "Stopping control system." << std::endl;
            operator_state.stop = true;
            return;
          }
          auto motor_command_end_time = std::chrono::steady_clock::now();

          const double hand_close_ratio = input_interface_->GetMaxCloseRatio();
          if (hand_backend_ == HandBackendKind::DEX3) {
            dex3_hands_.SetMaxCloseRatio(hand_close_ratio);
            if (has_left_hand_data_) {
              dex3_hands_.setAllJointsCommand(true, left_hand_joint_buffer_);
              last_left_hand_action = left_hand_joint_buffer_;
            }
            if (has_right_hand_data_) {
              dex3_hands_.setAllJointsCommand(false, right_hand_joint_buffer_);
              last_right_hand_action = right_hand_joint_buffer_;
            }
          } else if (hand_backend_ == HandBackendKind::INSPIRE) {
            constexpr auto kHandInputTimeout = std::chrono::milliseconds(500);
            bool input_fresh = has_left_hand_data_ || has_right_hand_data_;
            if (const auto last_update = input_interface_->GetLastUpdateTime();
                last_update.has_value()) {
              input_fresh = input_fresh &&
                  std::chrono::steady_clock::now() - *last_update <=
                      kHandInputTimeout;
            }
            inspire_hands_.SetMaxCloseRatio(hand_close_ratio);
            inspire_hands_.UpdateTargets(
                has_left_hand_data_, left_hand_joint_buffer_,
                has_right_hand_data_, right_hand_joint_buffer_, input_fresh);
            if (input_fresh && has_left_hand_data_) {
              last_left_hand_action = left_hand_joint_buffer_;
            }
            if (input_fresh && has_right_hand_data_) {
              last_right_hand_action = right_hand_joint_buffer_;
            }
          }
          
          auto hand_joint_end_time = std::chrono::steady_clock::now();

          // Publish output data (state logger data, robot config, command/motion data) to all output interfaces
          for (auto& output_interface : output_interfaces_) {
            if (output_interface) {
              output_interface->publish(
                vr_3point_position_buffer_, vr_3point_orientation_buffer_, vr_3point_compliance_buffer_,
                left_hand_joint_buffer_, right_hand_joint_buffer_, init_ref_data_root_rot_array_,
                heading_state_buffer_, current_motion_copy, current_frame_copy
              );
            }
          }

          // Handle recording of streamed motion (if enabled)
          if (enable_motion_recording_) {
            if (current_motion_copy && current_motion_copy->name == "streamed") {
              // Start recording session when streamed starts at frame 0
              if (current_frame_copy == 0 && !zmq_motion_recorder_.IsActive()) {
                zmq_motion_recorder_.StartSession("streamed");
              }
              
              // Write current frame if recording is active
              if (zmq_motion_recorder_.IsActive() && operator_state.play) {
                zmq_motion_recorder_.WriteFrame(current_motion_copy, current_frame_copy);
              }
            } else {
              // Finalize recording when switching away from streamed
              if (zmq_motion_recorder_.IsActive()) {
                zmq_motion_recorder_.Finalize();
              }
            }
            
            // Handle recording of planner_motion (closed-loop result)
            if (current_motion_copy && current_motion_copy == planner_motion_) {
              // Write current frame if recording is active
              if (planner_motion_recorder_.IsActive() && operator_state.play) {
                planner_motion_recorder_.WriteFrame(current_motion_copy, current_frame_copy);
              }
            }
          }

          // Write target motion to file
          if (target_motion_file_) {
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][0] << ",";
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][1] << ",";
            (*target_motion_file_) << current_motion_copy->BodyPositions(current_frame_copy)[0][2] << ",";

            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][0] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][1] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][2] << ",";
            (*target_motion_file_) << current_motion_copy->BodyQuaternions(current_frame_copy)[0][3] << ",";

            for(int i = 0; i < 29; i++) {
              (*target_motion_file_) << current_motion_copy->JointPositions(current_frame_copy)[isaaclab_to_mujoco[i]] << ",";
            }

            (*target_motion_file_) << std::endl;
          }

          if(policy_input_file_)
          {
            // dump neural network inputs to a log file:
            for(auto d : obs_buffer_)
            {
              (*policy_input_file_) << d << ",";
            }
            (*policy_input_file_) << std::endl;
          }

          if (!CurrentFrameAdvancement()) {
            std::cout << "✗ Error: Failed to advance current frame in the middle of the control loop!" << std::endl;
            std::cout << "Stopping control system." << std::endl;
            operator_state.stop = true;
            return;
          }

          // Temporarily disable the periodic timing line; it floods the operator console.
#if 0
          if (logging_counter_ % 50 == 0) {
            auto control_loop_end_time = std::chrono::steady_clock::now();
            auto obs_duration = std::chrono::duration_cast<std::chrono::microseconds>(obs_end_time - obs_start_time);
            auto policy_duration = std::chrono::duration_cast<std::chrono::microseconds>(motor_command_end_time - obs_end_time);
            auto motor_command_duration = std::chrono::duration_cast<std::chrono::microseconds>(motor_command_end_time - obs_start_time);
            auto post_hand_joint_duration = std::chrono::duration_cast<std::chrono::microseconds>(control_loop_end_time - hand_joint_end_time);
            
            std::cout << "Loop timing - LowState age: " << used_low_state_data_.GetAgeMs() << "ms"
                      << ", Streaming data mean delay: " << streaming_data_delay_rolling_stats_.mean() << "ms"
                      << ", Streaming data std delay: " << streaming_data_delay_rolling_stats_.stddev() << "ms"
                      << ", IMU age: " << used_imu_torso_data_.GetAgeMs() << "ms"
                      << ", Obs: " << obs_duration.count() << "us"
                      << ", Policy: " << policy_duration.count() << "us"
                      << ", Obs 2 Motor Command: " << motor_command_duration.count() << "us"
                      << ", Post processing: " << post_hand_joint_duration.count() << "us";
            
            // Add planner timing if planner is enabled and initialized
            if (planner_ && planner_->planner_state_.enabled && planner_->planner_state_.initialized) {
              std::cout << ", Planner - Gather Input: " << planner_->last_timing_.gather_input_duration.count() << "us"
                        << ", Model: " << planner_->last_timing_.model_duration.count() << "us"
                        << ", Convert50Hz: " << planner_->last_timing_.extract_duration.count() << "us"
                        << ", Total: " << planner_->last_timing_.total_duration.count() << "us";
            }
            
            // Only print compliance levels if the policy observes them
            if (has_vr_3point_compliance_obs_) {
              std::cout << " | Compliance [L,R,H]: [" 
                        << vr_3point_compliance_buffer_[0] << ", "
                        << vr_3point_compliance_buffer_[1] << ", "
                        << vr_3point_compliance_buffer_[2] << "]";
            }
            
            if (hand_backend_ == HandBackendKind::DEX3) {
              std::cout << " | HandCloseRatio: " << dex3_hands_.GetMaxCloseRatio();
            } else if (hand_backend_ == HandBackendKind::INSPIRE) {
              std::cout << " | HandCloseRatio: " << inspire_hands_.GetMaxCloseRatio();
            }
            
            std::cout << std::endl;
          }
#endif
          break;
        }
      }
    }
};

/**
 * @brief Entry point: parse CLI arguments and run the G1 deployment application.
 *
 * Required positional arguments:
 *   1. network_interface – DDS network interface (e.g. "eth0")
 *   2. policy_file       – Path to the ONNX policy model
 *   3. motion_data_path  – Directory of pre-loaded reference motions
 *
 * All other arguments are optional flags (see --help for full list).
 * The main loop sleeps until the operator issues a stop signal or ROS2 shuts down.
 */
int main(int argc, char const* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <network_interface> <policy_file> <motion_data_path> [OPTIONS]"
              << std::endl;
    std::cout << "  network_interface: network interface for DDS communication" << std::endl;
    std::cout << "  policy_file: path to ONNX policy file" << std::endl;
    std::cout << "  motion_data_path: path to motion data directory (e.g., reference/bones_072925_test/)" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --planner-file <path>: specify planner file (optional)" << std::endl;
    std::cout << "  --input-type <keyboard|gamepad|gamepad_manager|manager|zmq|zmq_manager";
#if HAS_ROS2
    std::cout << "|ros2";
#endif
    std::cout << ">: input interface type (default: keyboard)" << std::endl;
    std::cout << "  --output-type <zmq|all";
#if HAS_ROS2
    std::cout << "|ros2";
#endif
    std::cout << ">: output interface type (default: zmq, 'all' creates all available)" << std::endl;
    std::cout << "  --target-motion-logfile <path>: write target motion to a csv file if provided" << std::endl;
    std::cout << "  --planner-motion-logfile <path>: write planner motion to a csv file if provided" << std::endl;
    std::cout << "  --policy-input-logfile <path>: write policy input tensors to a csv file if provided" << std::endl;
    std::cout << "  --disable-crc-check: disable CRC validation for MuJoCo simulation" << std::endl;
    std::cout << "  --obs-config <path>: specify observation configuration YAML file" << std::endl;
    std::cout << "  --encoder-file <path>: specify encoder ONNX file (optional)" << std::endl;
    std::cout << "  --robot-profile <g1|h2_lora|h2_native|h2_native_v1_1>: select deploy policy interface (default: g1)" << std::endl;
    std::cout << "  --h2-vr-head-control: replace H2's zero head targets with fresh VR head_pose commands" << std::endl;
    std::cout << "  --planner-g1-to-h2: opt-in H2-native release planner projection to selector 0 (default: native G1 selector 3)" << std::endl;
    std::cout << "  --h2-waist-pitch-kp-scale <S>: waist pitch Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-waist-roll-kp-scale <S>: waist roll Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-ankle-pitch-kp-scale <S>: both ankle pitch Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-left-ankle-pitch-kp-scale <S>: left ankle pitch Kp; overrides paired scale" << std::endl;
    std::cout << "  --h2-left-ankle-roll-kp-scale <S>: left ankle roll Kp; overrides paired scale" << std::endl;
    std::cout << "  --h2-right-ankle-pitch-kp-scale <S>: right ankle pitch Kp; overrides paired scale" << std::endl;
    std::cout << "  --h2-right-ankle-roll-kp-scale <S>: right ankle roll Kp; overrides paired scale" << std::endl;
    std::cout << "  --h2-left-knee-pitch-kp-scale <S>: left knee Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-right-knee-pitch-kp-scale <S>: right knee Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-ankle-roll-kp-scale <S>: both ankle roll Kp multiplier (default 1)" << std::endl;
    std::cout << "  --h2-ankle-pitch-kd-scale <S>: both ankle pitch Kd multiplier (default 1)" << std::endl;
    std::cout << "    Gain scales require h2_native, apply during INIT and CONTROL; action scale and shutdown damping unchanged." << std::endl;
    std::cout << "  --hand-backend <dex3|inspire|none>: select DDS hand backend (default: dex3)" << std::endl;
    std::cout << "  --disable-dex3: deprecated alias for --hand-backend none" << std::endl;
    std::cout << "  --planner-precision <16|32>: specify precision to run the planner model at (default: 16)" << std::endl;
    std::cout << "  --policy-precision <16|32>: specify precision to run the policy model at (default: 32)" << std::endl;
    std::cout << "  --zmq-host <host>: ZMQ server host (default: localhost)" << std::endl;
    std::cout << "  --zmq-port <port>: ZMQ server port (default: 5556)" << std::endl;
    std::cout << "  --zmq-topic <topic>: ZMQ topic/prefix (default: pose)" << std::endl;
    std::cout << "  --zmq-conflate: enable ZMQ CONFLATE (default: disabled)" << std::endl;
    std::cout << "  --zmq-verbose: enable ZMQ subscriber verbose logs" << std::endl;
    std::cout << "  --zmq-out-port <port>: ZMQ port for output (default: 5557)" << std::endl;
    std::cout << "  --zmq-out-topic <topic>: ZMQ topic/prefix for output (default: g1_debug)" << std::endl;
    std::cout << "  --logs-dir <path>: optional logs output base directory (default: logs/<timestamp>/)" << std::endl;
    std::cout << "  --enable-csv-logs: enable writing CSV logs (default: OFF)" << std::endl;
    std::cout << "  --enable-motion-recording: enable motion recording for ZMQ/planner (default: OFF)" << std::endl;
    std::cout << "  --set-compliance <value>: set initial VR 3-point compliance (0.01=rigid, 0.5=compliant; default: [0.5, 0.5, 0.0])" << std::endl;
    std::cout << "                                 Can specify 1 value (both hands) or 3 values (left_wrist,right_wrist,head)" << std::endl;
    std::cout << "                                 Keyboard controls: g/h = left hand +/- 0.1, b/v = right hand +/- 0.1" << std::endl;
    std::cout << "  --max-close-ratio <value>: set initial hand max close ratio (0.2-1.0; default: 1.0 = full closure)" << std::endl;
    std::cout << "                             0.2 = limited (80% open), 1.0 = full closure allowed" << std::endl;
    std::cout << "                             Keyboard controls: x/c = +/- 0.1 (always available)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --planner-file policy/planner.onnx --obs-config policy/single_frame/observation_config.yaml --disable-crc-check" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/token/model.onnx reference/bones_072925_test/ --obs-config policy/token/observation_config.yaml --encoder-file policy/token/encoder.onnx" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type gamepad --planner-file policy/planner.onnx" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type gamepad_manager --planner-file policy/planner.onnx --zmq-host localhost --zmq-port 5556" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type zmq --zmq-host 192.168.1.2 --zmq-port 5556 --zmq-topic pose --zmq-conflate" << std::endl;
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type zmq_manager --planner-file policy/planner.onnx --zmq-host localhost --zmq-port 5556" << std::endl;
#if HAS_ROS2
    std::cout << "  " << argv[0] << " enp5s0 policy/single_frame/model.onnx reference/bones_072925_test/ --input-type ros2 --planner-file policy/planner.onnx" << std::endl;
#endif
    exit(0);
  }
  std::string networkInterface = argv[1];
  std::string modelFile = argv[2];
  std::string motionDataPath = argv[3];
  std::string plannerFile = "";

  // Parse optional arguments
  bool disableCrcCheck = false;
  std::string obsConfigPath = "";
  std::string encoderFile = "";
  std::string targetMotionLogfile = "";
  std::string plannerMotionLogfile = "";
  std::string policyInputLogfile = "";
  std::string recordInputFile = "";
  std::string playbackInputFile = "";
  std::string inputType = "keyboard"; // Default to keyboard
  std::string outputType = "zmq"; // Default to zmq
  std::string robotProfile = "g1";
  bool enableH2VrHeadControl = false;
  bool projectH2NativePlannerReference = false;
  H2GainScales h2GainScales;
  bool h2GainScalesExplicit = false;
  std::string handBackendValue = "dex3";
  bool handBackendExplicit = false;
  bool disableDex3Requested = false;
  bool plannerFp16 = false;
  bool policyFp16 = false;
  std::string logsDir = "";
  bool enableCsvLogs = false;
  std::string zmq_host = "localhost";
  int zmq_port = 5556;
  std::string zmq_topic = "pose";
  bool zmq_conflate = false;  // default off; enable with --zmq-conflate
  bool zmq_verbose = false;
  bool enableMotionRecording = false;  // default off; enable with --enable-motion-recording
  int zmq_out_port = 5557;
  std::string zmq_out_topic = "g1_debug";
  std::array<double, 3> initial_compliance = {0.5, 0.5, 0.0}; // initial compliance is 0.5 for both hands (keyboard controllable)
  double initial_max_close_ratio = 1.0; // default allows full closure, use --max-close-ratio to limit
  for (int i = 4; i < argc; i++) {
    if (std::string(argv[i]) == "--disable-crc-check") {
      disableCrcCheck = true;
      std::cout << "[INFO] CRC checking disabled for MuJoCo simulation" << std::endl;
    } else if (std::string(argv[i]) == "--obs-config") {
      if (i + 1 < argc) {
        obsConfigPath = argv[i + 1];
        std::cout << "[INFO] Using observation config: " << obsConfigPath << std::endl;
        i++; // Skip the next argument since it's the config path
      } else {
        std::cerr << "Error: --obs-config requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--encoder-file") {
      if (i + 1 < argc) {
        encoderFile = argv[i + 1];
        std::cout << "[INFO] Using encoder file: " << encoderFile << std::endl;
        i++; // Skip the next argument since it's the encoder file path
      } else {
        std::cerr << "Error: --encoder-file requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--robot-profile") {
      if (i + 1 < argc) {
        robotProfile = argv[i + 1];
        if (!(robotProfile == "g1" || IsH2Profile(robotProfile))) {
          std::cerr << "Error: --robot-profile must be 'g1', 'h2_lora', 'h2_native', or 'h2_native_v1_1'" << std::endl;
          exit(1);
        }
        std::cout << "[INFO] Using robot profile: " << robotProfile << std::endl;
        i++;
      } else {
        std::cerr << "Error: --robot-profile requires a profile argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--h2-vr-head-control") {
      enableH2VrHeadControl = true;
      std::cout << "[INFO] H2 VR head pitch/yaw control enabled" << std::endl;
    } else if (std::string(argv[i]) == "--h2-waist-pitch-kp-scale" ||
               std::string(argv[i]) == "--h2-waist-roll-kp-scale" ||
               std::string(argv[i]) == "--h2-ankle-pitch-kp-scale" ||
               std::string(argv[i]) == "--h2-ankle-roll-kp-scale" ||
               std::string(argv[i]) == "--h2-ankle-pitch-kd-scale" ||
               std::string(argv[i]) == "--h2-left-ankle-pitch-kp-scale" ||
               std::string(argv[i]) == "--h2-left-ankle-roll-kp-scale" ||
               std::string(argv[i]) == "--h2-right-ankle-pitch-kp-scale" ||
               std::string(argv[i]) == "--h2-right-ankle-roll-kp-scale" ||
               std::string(argv[i]) == "--h2-left-knee-pitch-kp-scale" ||
               std::string(argv[i]) == "--h2-right-knee-pitch-kp-scale") {
      const std::string option = argv[i];
      if (i + 1 >= argc) {
        std::cerr << "Error: " << option << " requires a positive numeric value" << std::endl;
        return 1;
      }
      try {
        const float value = H2GainScales::ParseValue(argv[++i]);
        if (option == "--h2-waist-pitch-kp-scale") h2GainScales.waist_pitch_kp = value;
        else if (option == "--h2-waist-roll-kp-scale") h2GainScales.waist_roll_kp = value;
        else if (option == "--h2-ankle-pitch-kp-scale") h2GainScales.ankle_pitch_kp = value;
        else if (option == "--h2-ankle-roll-kp-scale") h2GainScales.ankle_roll_kp = value;
        else if (option == "--h2-ankle-pitch-kd-scale") h2GainScales.ankle_pitch_kd = value;
        else if (option == "--h2-left-ankle-pitch-kp-scale") h2GainScales.left_ankle_pitch_kp = value;
        else if (option == "--h2-left-ankle-roll-kp-scale") h2GainScales.left_ankle_roll_kp = value;
        else if (option == "--h2-right-ankle-pitch-kp-scale") h2GainScales.right_ankle_pitch_kp = value;
        else if (option == "--h2-right-ankle-roll-kp-scale") h2GainScales.right_ankle_roll_kp = value;
        else if (option == "--h2-left-knee-pitch-kp-scale") h2GainScales.left_knee_pitch_kp = value;
        else if (option == "--h2-right-knee-pitch-kp-scale") h2GainScales.right_knee_pitch_kp = value;
      } catch (const std::exception& error) {
        std::cerr << "Error: " << option << ": " << error.what() << std::endl;
        return 1;
      }
      h2GainScalesExplicit = true;
    } else if (std::string(argv[i]) == "--planner-g1-to-h2") {
      projectH2NativePlannerReference = true;
      std::cout << "[INFO] G1->H2 planner reference comparison enabled (selector 0)" << std::endl;
    } else if (std::string(argv[i]) == "--hand-backend") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --hand-backend requires dex3, inspire, or none"
                  << std::endl;
        exit(1);
      }
      handBackendValue = argv[++i];
      try {
        (void)ParseHandBackend(handBackendValue);
      } catch (const std::invalid_argument& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        exit(1);
      }
      handBackendExplicit = true;
      std::cout << "[INFO] Hand backend: " << handBackendValue << std::endl;
    } else if (std::string(argv[i]) == "--disable-dex3") {
      disableDex3Requested = true;
      std::cout << "[INFO] --disable-dex3 maps to --hand-backend none"
                << std::endl;
    } else if (std::string(argv[i]) == "--planner-file") {
      if (i + 1 < argc) {
        plannerFile = argv[i + 1];
        std::cout << "[INFO] Using planner file: " << plannerFile << std::endl;
        i++; // Skip the next argument since it's the planner path
      } else {
        std::cerr << "Error: --planner-file requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--target-motion-logfile") {
      if (i + 1 < argc) {
        targetMotionLogfile = argv[i + 1];
        std::cout << "[INFO] Using target motion logfile: " << targetMotionLogfile << std::endl;
        i++; // Skip the next argument since it's the target motion logfile
      }
    } else if (std::string(argv[i]) == "--planner-motion-logfile") {
      if (i + 1 < argc) {
        plannerMotionLogfile = argv[i + 1];
        std::cout << "[INFO] Using planner motion logfile: " << plannerMotionLogfile << std::endl;
        i++; // Skip the next argument since it's the planner motion logfile
      }
    } else if (std::string(argv[i]) == "--policy-input-logfile") {
      if (i + 1 < argc) {
        policyInputLogfile = argv[i + 1];
        std::cout << "[INFO] Using policy input logfile: " << policyInputLogfile << std::endl;
        i++; // Skip the next argument since it's the policy input logfile
      }
    } else if (std::string(argv[i]) == "--input-type") {
      if (i + 1 < argc) {
        inputType = argv[i + 1];
        // Validate input type based on what's available
        bool valid_input = (inputType == "keyboard" || inputType == "gamepad" || inputType == "gamepad_manager" || inputType == "zmq" || inputType == "zmq_manager" || inputType == "manager");
#if HAS_ROS2
        valid_input = valid_input || (inputType == "ros2");
#endif
        if (!valid_input) {
          std::cerr << "Error: --input-type must be 'keyboard', 'gamepad', 'gamepad_manager', 'manager', 'zmq', or 'zmq_manager'";
#if HAS_ROS2
          std::cerr << ", or 'ros2'";
#endif
          std::cerr << std::endl;
          exit(1);
        }
        std::cout << "[INFO] Using input type: " << inputType << std::endl;
        i++; // Skip the next argument since it's the input type
      } else {
        std::cerr << "Error: --input-type requires a type argument (keyboard, gamepad, gamepad_manager, manager, zmq, zmq_manager";
#if HAS_ROS2
        std::cerr << ", or ros2";
#endif
        std::cerr << ")" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--output-type") {
      if (i + 1 < argc) {
        outputType = argv[i + 1];
        bool valid_output = (outputType == "zmq" || outputType == "all");
#if HAS_ROS2
        valid_output = valid_output || (outputType == "ros2");
#endif
        if (!valid_output) {
          std::cerr << "Error: --output-type must be 'zmq', 'all'";
#if HAS_ROS2
          std::cerr << ", or 'ros2'";
#endif
          std::cerr << std::endl;
          exit(1);
        }
        std::cout << "[INFO] Using output type: " << outputType << std::endl;
        i++; // Skip the next argument since it's the output type
      } else {
        std::cerr << "Error: --output-type requires a type argument (zmq, all";
#if HAS_ROS2
        std::cerr << ", or ros2";
#endif
        std::cerr << ")" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--zmq-out-port") {
      if (i + 1 < argc) { zmq_out_port = std::stoi(argv[i + 1]); i++; }
    } else if (std::string(argv[i]) == "--zmq-out-topic") {
      if (i + 1 < argc) { zmq_out_topic = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--record-input-file") {
      if (i + 1 < argc) {
        recordInputFile = argv[i + 1];
        std::cout << "[INFO] Recording input to logfile: " << recordInputFile << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--playback-input-file") {
      if (i + 1 < argc) {
        playbackInputFile = argv[i + 1];
        std::cout << "[INFO] Playing input from logfile: " << playbackInputFile << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--planner-precision") {
      if (i + 1 < argc) {
        if(argv[i+1] == std::string("16"))
        {
          plannerFp16 = true;
        }
        else if(argv[i+1] == std::string("32"))
        {
          plannerFp16 = false;
        }
        else{
          throw std::runtime_error("invalid --planner-precision (should be 16 or 32): " + std::string(argv[i+1]));
        }
        std::cout << "[INFO] Planner precision: " << argv[i+1] << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
    } else if (std::string(argv[i]) == "--policy-precision") {
      if (i + 1 < argc) {
        if(argv[i+1] == std::string("16"))
        {
          policyFp16 = true;
        }
        else if(argv[i+1] == std::string("32"))
        {
          policyFp16 = false;
        }
        else{
          throw std::runtime_error("invalid --policy-precision (should be 16 or 32): " + std::string(argv[i+1]));
        }
        std::cout << "[INFO] Policy precision: " << argv[i+1] << std::endl;
        i++; // Skip the next argument since it's the input logfile
      }
      else{
        std::cerr << "Error: --policy-precision requires 16 or 32" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--enable-csv-logs") {
      enableCsvLogs = true;
      std::cout << "[INFO] CSV logging enabled" << std::endl;
    } else if (std::string(argv[i]) == "--logs-dir") {
      if (i + 1 < argc) {
        logsDir = argv[i + 1];
        std::cout << "[INFO] Using logs directory: " << logsDir << std::endl;
        i++; // Skip next arg (the path)
      } else {
        std::cerr << "Error: --logs-dir requires a path argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--zmq-host") {
      if (i + 1 < argc) { zmq_host = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--zmq-port") {
      if (i + 1 < argc) { zmq_port = std::stoi(argv[i + 1]); i++; }
    } else if (std::string(argv[i]) == "--zmq-topic") {
      if (i + 1 < argc) { zmq_topic = argv[i + 1]; i++; }
    } else if (std::string(argv[i]) == "--zmq-conflate") {
      zmq_conflate = true;
    } else if (std::string(argv[i]) == "--zmq-verbose") {
      zmq_verbose = true;
    } else if (std::string(argv[i]) == "--enable-motion-recording") {
      enableMotionRecording = true;
      std::cout << "[INFO] Motion recording enabled" << std::endl;
    } else if (std::string(argv[i]) == "--set-compliance") {
      if (i + 1 < argc) {
        // Parse compliance values (can be 1 or 3 values)
        std::string compliance_str = argv[i + 1];
        std::vector<double> values;
        std::stringstream ss(compliance_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
          try {
            values.push_back(std::stod(token));
          } catch (...) {
            std::cerr << "Error: Invalid compliance value: " << token << std::endl;
            exit(1);
          }
        }
        if (values.size() == 1) {
          initial_compliance = {values[0], values[0], 0.0};
          std::cout << "[INFO] Initial VR 3-point compliance set to: [" << values[0] 
                    << ", " << values[0] << ", 0.0]" << std::endl;
        } else if (values.size() == 3) {
          initial_compliance = {values[0], values[1], values[2]};
          std::cout << "[INFO] Initial VR 3-point compliance set to: ["
                    << values[0] << ", " << values[1] << ", " << values[2] << "]" << std::endl;
        } else {
          std::cerr << "Error: --set-compliance requires 1 or 3 comma-separated values" << std::endl;
          exit(1);
        }
        i++; // Skip the next argument since it's the compliance value
      } else {
        std::cerr << "Error: --set-compliance requires a value argument" << std::endl;
        exit(1);
      }
    } else if (std::string(argv[i]) == "--max-close-ratio") {
      if (i + 1 < argc) {
        try {
          initial_max_close_ratio = std::stod(argv[i + 1]);
          // Validate range [0.2, 1.0]
          if (!inspire_hand_control::IsFinite(initial_max_close_ratio) ||
              initial_max_close_ratio < 0.2 || initial_max_close_ratio > 1.0) {
            std::cerr << "Error: --max-close-ratio must be between 0.2 and 1.0" << std::endl;
            exit(1);
          }
          std::cout << "[INFO] Initial hand max close ratio set to: " << initial_max_close_ratio 
                    << " (0.2 = limited, 1.0 = full closure)" << std::endl;
        } catch (...) {
          std::cerr << "Error: Invalid max close ratio value: " << argv[i + 1] << std::endl;
          exit(1);
        }
        i++; // Skip the next argument since it's the ratio value
      } else {
        std::cerr << "Error: --max-close-ratio requires a value argument" << std::endl;
        exit(1);
      }
    }
  }

  if (disableDex3Requested && handBackendExplicit) {
    std::cerr << "Error: --disable-dex3 cannot be combined with --hand-backend"
              << std::endl;
    exit(1);
  }
  if (h2GainScalesExplicit && robotProfile != "h2_native") {
    std::cerr << "Error: H2 gain scaling requires --robot-profile h2_native" << std::endl;
    return 1;
  }
  if (disableDex3Requested) handBackendValue = "none";
  const HandBackendKind handBackend = ParseHandBackend(handBackendValue);
  if (projectH2NativePlannerReference &&
      (!IsH2NativeProfile(robotProfile) || IsH2NativeV11Profile(robotProfile) ||
       plannerFile.empty() || encoderFile.empty())) {
    std::cerr << "Error: --planner-g1-to-h2 requires --robot-profile h2_native, "
                 "--planner-file and --encoder-file" << std::endl;
    return 1;
  }

  std::cout << "[INFO] Creating deploy controller..." << std::endl;
  G1Deploy custom(
    networkInterface,
    modelFile,
    motionDataPath,
    disableCrcCheck,
    obsConfigPath,
    encoderFile,
    plannerFile,
    targetMotionLogfile,
    plannerMotionLogfile,
    policyInputLogfile,
    inputType,
    outputType,
    recordInputFile,
    playbackInputFile,
    plannerFp16,
    policyFp16,
    logsDir,
    enableCsvLogs,
    zmq_host,
    zmq_port,
    zmq_topic,
    zmq_conflate,
    zmq_verbose,
    zmq_out_port,
    zmq_out_topic,
    enableMotionRecording,
    initial_compliance,
    initial_max_close_ratio,
    robotProfile,
    enableH2VrHeadControl,
    handBackend,
    projectH2NativePlannerReference,
    h2GainScales
  );
  std::cout << "[INFO] Deploy controller ready" << std::endl;
  const bool trace_stop_state =
      []() {
        const char* env = std::getenv("GEAR_SONIC_TRACE_STOP_STATE");
        return env != nullptr && std::string(env) != "0" && std::string(env) != "false";
      }();
  if (trace_stop_state) {
    std::cout << "[TRACE] operator_state.stop after constructor: "
              << static_cast<int>(custom.operator_state.stop) << std::endl;
  }
  
  // Main application loop - check both operator_state.stop and ROS2 status if using ROS2
#if HAS_ROS2
  if (inputType == "ros2") {
    while (!custom.operator_state.stop && rclcpp::ok()) { 
      sleep(0.02); 
    }
    if (!rclcpp::ok()) {
      std::cout << "[INFO] ROS2 shutdown detected (Ctrl+C)" << std::endl;
    }
  } else {
    while (!custom.operator_state.stop) { sleep(0.02); }
  }
#else
  while (!custom.operator_state.stop) { sleep(0.02); }
#endif
  if (trace_stop_state) {
    std::cout << "[TRACE] main loop exited with operator_state.stop="
              << static_cast<int>(custom.operator_state.stop) << std::endl;
  }
  
  custom.Stop();
  sleep(0.5);
  std::cout << "[INFO] Program exiting normally" << std::endl;
  return 0;
}
