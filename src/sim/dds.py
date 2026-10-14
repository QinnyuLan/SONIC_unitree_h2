"""Bridge between Unitree SDK2 DDS topics and the MuJoCo simulation.

Subscribes to low-level motor commands (body + hands) and publishes
simulated sensor state (joint pos/vel, IMU, odometry) back over DDS,
so the WBC policy sees the sim as a real robot.
"""

import os
import sys
import threading
import time
from typing import Dict, Tuple

import numpy as np
import scipy.spatial.transform
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__WirelessController_,
    unitree_hg_msg_dds__HandCmd_ as HandCmd_default,
    unitree_hg_msg_dds__HandState_ as HandState_default,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from sim.odometry import OdoState_, default_odometry

H2_JOINT_NAMES_LOCAL_MUJOCO = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

H2_DDS_TO_LOCAL_MUJOCO = np.asarray(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        15,
        16,
    ],
    dtype=np.int64,
)
H2_LOCAL_MUJOCO_TO_DDS = np.empty_like(H2_DDS_TO_LOCAL_MUJOCO)
H2_LOCAL_MUJOCO_TO_DDS[H2_DDS_TO_LOCAL_MUJOCO] = np.arange(H2_DDS_TO_LOCAL_MUJOCO.size)

H2_JOINT_NAMES_DDS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "head_pitch_joint",
    "head_yaw_joint",
]


def apply_local_pitch_bias_wxyz(quaternion, pitch_bias_rad: float) -> np.ndarray:
    """Right-compose a local-frame pitch bias onto a wxyz quaternion."""
    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must contain 4 values, got shape {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"quaternion must have a finite non-zero norm, got {quat}")

    w, x, y, z = quat / norm
    half_angle = 0.5 * float(pitch_bias_rad)
    c = np.cos(half_angle)
    s = np.sin(half_angle)
    biased = np.asarray(
        [
            w * c - y * s,
            x * c - z * s,
            w * s + y * c,
            x * s + z * c,
        ],
        dtype=float,
    )
    return biased / np.linalg.norm(biased)


def build_h2_dds_to_sim_joint_mapping(sim_joint_names) -> Tuple[np.ndarray, np.ndarray]:
    """Build official H2 DDS motor order <-> actual MuJoCo body joint order maps."""
    if len(sim_joint_names) != len(H2_JOINT_NAMES_DDS):
        raise ValueError(
            "H2 DDS/sim motor mapping expects "
            f"{len(H2_JOINT_NAMES_DDS)} joints, got {len(sim_joint_names)}."
        )

    name_to_sim: dict[str, int] = {}
    duplicates = []
    for sim_i, name in enumerate(sim_joint_names):
        if name in name_to_sim:
            duplicates.append(name)
        name_to_sim[name] = sim_i
    if duplicates:
        raise ValueError(f"Duplicate H2 MuJoCo joint names: {duplicates}")

    missing = [name for name in H2_JOINT_NAMES_DDS if name not in name_to_sim]
    if missing:
        raise ValueError(
            "Cannot map H2 DDS motors to MuJoCo joints; missing joints: "
            f"{missing}. sim_joint_names={list(sim_joint_names)}"
        )

    motor_to_sim = np.asarray(
        [name_to_sim[name] for name in H2_JOINT_NAMES_DDS], dtype=np.int64
    )
    sim_to_motor = np.empty_like(motor_to_sim)
    sim_to_motor[motor_to_sim] = np.arange(motor_to_sim.size, dtype=np.int64)
    return motor_to_sim, sim_to_motor


class UnitreeSdk2Bridge:
    """
    This class is responsible for bridging the Unitree SDK2 with the Groot environment.
    It is responsible for sending and receiving messages to and from the Unitree SDK2.
    Both the body and hand are supported.
    """

    def __init__(self, config):
        # Note that we do not give the mjdata and mjmodel to the UnitreeSdk2Bridge.
        # It is unsafe and would be unflexible if we use a hand-plugged robot model

        robot_type = config["ROBOT_TYPE"]
        self._low_cmd_factory = None
        if "g1" in robot_type or "h1-2" in robot_type or "h2" in robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
                unitree_hg_msg_dds__LowCmd_,
                unitree_hg_msg_dds__LowState_ as LowState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

            self._low_cmd_factory = unitree_hg_msg_dds__LowCmd_
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowCmd_,
                unitree_go_msg_dds__LowState_ as LowState_default,
                unitree_hg_msg_dds__IMUState_ as IMUState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import IMUState_, LowCmd_, LowState_

            self._low_cmd_factory = unitree_go_msg_dds__LowCmd_
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', 'h2', or 'go2'.")

        self.num_body_motor = config["NUM_MOTORS"]
        self.num_hand_motor = config.get("NUM_HAND_MOTORS", 0)
        self.use_sensor = config["USE_SENSOR"]
        self.use_local_base_gyro = bool(config.get("USE_LOCAL_BASE_GYRO", False))
        self.pelvis_imu_pitch_bias_deg = float(
            config.get("PELVIS_IMU_PITCH_BIAS_DEG", 0.0)
        )
        self.pelvis_imu_pitch_bias_rad = np.deg2rad(self.pelvis_imu_pitch_bias_deg)
        if self.pelvis_imu_pitch_bias_deg != 0.0:
            print(
                "[MuJoCo] Pelvis IMU pitch bias enabled: "
                f"{self.pelvis_imu_pitch_bias_deg:+.6f} deg "
                "(reported = true * local RotY(bias)); odometry remains unbiased."
            )
        self.low_cmd_timeout_sec = float(
            os.environ.get(
                "GEAR_SONIC_MUJOCO_LOW_CMD_TIMEOUT_SEC",
                config.get("LOW_CMD_TIMEOUT_SEC", 0.25),
            )
        )
        self.motor_to_sim_joint_index = np.arange(self.num_body_motor, dtype=np.int64)
        self.sim_joint_to_motor_index = np.arange(self.num_body_motor, dtype=np.int64)
        if "h2" in robot_type:
            if self.num_body_motor != H2_DDS_TO_LOCAL_MUJOCO.size:
                raise ValueError(
                    f"H2 DDS/local motor mapping expects 31 motors, got {self.num_body_motor}."
                )
            sim_joint_names = config.get(
                "MUJOCO_BODY_JOINT_LOGICAL_NAMES",
                config.get("MUJOCO_BODY_JOINT_NAMES"),
            )
            if sim_joint_names:
                (
                    self.motor_to_sim_joint_index,
                    self.sim_joint_to_motor_index,
                ) = build_h2_dds_to_sim_joint_mapping(sim_joint_names)
            else:
                # Fallback for older callers that did not expose actual MuJoCo
                # joint names.  The local training/MJCF asset keeps head before
                # arms, while Unitree H2 LowCmd/LowState use official DDS order
                # (legs + waist + arms + head).
                self.motor_to_sim_joint_index = H2_DDS_TO_LOCAL_MUJOCO.copy()
                self.sim_joint_to_motor_index = H2_LOCAL_MUJOCO_TO_DDS.copy()

        self.have_imu_ = False
        self.have_frame_sensor_ = False

        self.low_cmd_lock = threading.Lock()
        self.left_hand_cmd_lock = threading.Lock()
        self.right_hand_cmd_lock = threading.Lock()
        self.low_cmd_received = False
        self.new_low_cmd = False
        self.low_cmd_count = 0
        self.low_cmd_last_time = None
        self.left_hand_cmd_received = False
        self.new_left_hand_cmd = False
        self.right_hand_cmd_received = False
        self.new_right_hand_cmd = False

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        # Only create odo_state for supported robot types
        if "g1" in robot_type or "h1-2" in robot_type or "h2" in robot_type:
            self.odo_state = default_odometry()
            self.odo_state_puber = ChannelPublisher("rt/odostate", OdoState_)
            self.odo_state_puber.Init()
        else:
            self.odo_state = None
            self.odo_state_puber = None
        self.torso_imu_state = IMUState_default()
        self.torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self.torso_imu_puber.Init()

        self.left_hand_state = HandState_default()
        self.left_hand_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.left_hand_state_puber.Init()
        self.right_hand_state = HandState_default()
        self.right_hand_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.right_hand_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.left_hand_cmd = HandCmd_default()
        self.left_hand_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.left_hand_cmd_suber.Init(self.LeftHandCmdHandler, 1)
        self.right_hand_cmd = HandCmd_default()
        self.right_hand_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.right_hand_cmd_suber.Init(self.RightHandCmdHandler, 1)

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        self.reset()

    def reset(self):
        with self.low_cmd_lock:
            if self._low_cmd_factory is not None:
                self.low_cmd = self._low_cmd_factory()
            self.low_cmd_received = False
            self.new_low_cmd = False
            self.low_cmd_count = 0
            self.low_cmd_last_time = None
        with self.left_hand_cmd_lock:
            self.left_hand_cmd_received = False
            self.new_left_hand_cmd = False
        with self.right_hand_cmd_lock:
            self.right_hand_cmd_received = False
            self.new_right_hand_cmd = False

    def LowCmdHandler(self, msg):
        with self.low_cmd_lock:
            self.low_cmd = msg
            self.low_cmd_received = True
            self.new_low_cmd = True
            self.low_cmd_count += 1
            self.low_cmd_last_time = time.monotonic()

    def LeftHandCmdHandler(self, msg):
        with self.left_hand_cmd_lock:
            self.left_hand_cmd = msg
            self.left_hand_cmd_received = True
            self.new_left_hand_cmd = True

    def RightHandCmdHandler(self, msg):
        with self.right_hand_cmd_lock:
            self.right_hand_cmd = msg
            self.right_hand_cmd_received = True
            self.new_right_hand_cmd = True

    def cmd_received(self):
        with self.low_cmd_lock:
            low_cmd_received = self.low_cmd_received
        with self.left_hand_cmd_lock:
            left_hand_cmd_received = self.left_hand_cmd_received
        with self.right_hand_cmd_lock:
            right_hand_cmd_received = self.right_hand_cmd_received
        return low_cmd_received or left_hand_cmd_received or right_hand_cmd_received

    def body_low_cmd_is_fresh(self) -> bool:
        with self.low_cmd_lock:
            if not self.low_cmd_received or self.low_cmd_last_time is None:
                return False
            if self.low_cmd_timeout_sec <= 0:
                return True
            return (time.monotonic() - self.low_cmd_last_time) <= self.low_cmd_timeout_sec

    def body_low_cmd_available(self) -> bool:
        """Return whether a body LowCmd has ever been received.

        The official unitree_mujoco bridge updates MuJoCo ctrl inside the
        LowCmd callback and leaves that ctrl active until another command
        arrives.  For sim2sim we keep the same hold-last-command behavior for
        the low-level PD torque calculation, while body_low_cmd_is_fresh()
        remains available for logging and optional wait/freeze diagnostics.
        """
        with self.low_cmd_lock:
            return bool(self.low_cmd_received and self.low_cmd is not None)

    def WarmupSerialization(self):
        """Build lazy DDS codecs locally before the realtime physics epoch."""
        messages = [self.low_state, self.low_cmd, self.odo_state, self.torso_imu_state]
        if self.num_hand_motor:
            messages += [self.left_hand_state, self.right_hand_state]
        for message in messages:
            type(message).deserialize(message.serialize())

    def PublishLowState(self, obs: Dict[str, any]):
        # publish body state
        if self.use_sensor:
            raise NotImplementedError("Sensor data is not implemented yet.")
        else:
            for motor_i in range(self.num_body_motor):
                sim_i = int(self.motor_to_sim_joint_index[motor_i])
                self.low_state.motor_state[motor_i].q = obs["body_q"][sim_i]
                self.low_state.motor_state[motor_i].dq = obs["body_dq"][sim_i]
                self.low_state.motor_state[motor_i].ddq = obs["body_ddq"][sim_i]
                self.low_state.motor_state[motor_i].tau_est = obs["body_tau_est"][sim_i]

        if self.use_sensor and self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        else:
            # Get data from ground truth
            self.odo_state.position[:] = obs["floating_base_pose"][:3]
            self.odo_state.linear_velocity[:] = obs["floating_base_vel"][:3]
            self.odo_state.orientation[:] = obs["floating_base_pose"][3:7]
            self.odo_state.angular_velocity[:] = obs["floating_base_vel"][3:6]
            # Only the LowState pelvis IMU receives the optional mounting/calibration
            # bias. Odometry stays as MuJoCo ground truth for an observable A/B test.
            self.low_state.imu_state.quaternion[:] = apply_local_pitch_bias_wxyz(
                obs["floating_base_pose"][3:7], self.pelvis_imu_pitch_bias_rad
            )
            if self.use_local_base_gyro:
                # Unitree IMU gyro and IsaacLab base_ang_vel are expressed in the root/body frame.
                self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel_local"][3:6]
            else:
                self.low_state.imu_state.gyroscope[:] = obs["floating_base_vel"][3:6]
            # linear acceleration
            self.low_state.imu_state.accelerometer[:] = obs["floating_base_acc"][:3]

            self.torso_imu_state.quaternion[:] = obs["secondary_imu_quat"]
            self.torso_imu_state.gyroscope[:] = obs["secondary_imu_vel"][3:6]

        # acceleration: x, y, z (only available when frame sensor is enabled)
        if self.have_frame_sensor_:
            raise NotImplementedError("Frame sensor data is not implemented yet.")
        self.low_state.tick = int(obs["time"] * 1e3)
        self.low_state_puber.Write(self.low_state)

        self.odo_state.tick = int(obs["time"] * 1e3)
        self.odo_state_puber.Write(self.odo_state)

        self.torso_imu_puber.Write(self.torso_imu_state)

        # publish hand state
        if self.num_hand_motor == 0:
            return
        for i in range(self.num_hand_motor):
            self.left_hand_state.motor_state[i].q = obs["left_hand_q"][i]
            self.left_hand_state.motor_state[i].dq = obs["left_hand_dq"][i]
        self.left_hand_state_puber.Write(self.left_hand_state)

        for i in range(self.num_hand_motor):
            self.right_hand_state.motor_state[i].q = obs["right_hand_q"][i]
            self.right_hand_state.motor_state[i].dq = obs["right_hand_dq"][i]
        self.right_hand_state_puber.Write(self.right_hand_state)

    def GetAction(self) -> Tuple[np.ndarray, bool, bool]:
        with self.low_cmd_lock:
            body_q = [self.low_cmd.motor_cmd[i].q for i in range(self.num_body_motor)]
        with self.left_hand_cmd_lock:
            left_hand_q = [self.left_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.right_hand_cmd_lock:
            right_hand_q = [self.right_hand_cmd.motor_cmd[i].q for i in range(self.num_hand_motor)]
        with self.low_cmd_lock and self.left_hand_cmd_lock and self.right_hand_cmd_lock:
            if self.num_hand_motor == 0:
                is_new_action = self.new_low_cmd
            else:
                is_new_action = (
                    self.new_low_cmd and self.new_left_hand_cmd and self.new_right_hand_cmd
                )
            if is_new_action:
                self.new_low_cmd = False
                self.new_left_hand_cmd = False
                self.new_right_hand_cmd = False

        if self.num_hand_motor == 0:
            action = np.asarray(body_q, dtype=float)
        else:
            action = np.concatenate([body_q[:-7], left_hand_q, body_q[-7:], right_hand_q])

        return (
            action,
            self.cmd_received(),
            is_new_action,
        )

    def PublishWirelessController(self):
        import pygame

        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            # Pack 16 button states into a single integer via bit-shifting
            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 3, "RY": 4,
                    "LT": 2, "RT": 5, "DX": 6, "DY": 7,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 4, "RB": 5, "SELECT": 6, "START": 7,
                    "XBOX": 8, "LSB": 9, "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                    "LT": 4, "RT": 5,
                }
                self.button_id = {
                    "X": 2, "Y": 3, "B": 1, "A": 0,
                    "LB": 9, "RB": 10, "SELECT": 4, "START": 6,
                    "XBOX": 5, "LSB": 7, "RSB": 8,
                    "DYU": 11, "DYD": 12, "DXL": 13, "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3,
                "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0,
                "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        import mujoco
        from loguru import logger
        from termcolor import colored

        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                logger.info(
                    f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}"
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(
        self,
        point=None,
        length=0.0,
        kp_pos=10000.0,
        kd_pos=1000.0,
        kp_ang=1000.0,
        kd_ang=10.0,
    ):
        self.kp_pos = float(kp_pos)
        self.kd_pos = float(kd_pos)
        self.kp_ang = float(kp_ang)
        self.kd_ang = float(kd_ang)
        self.point = np.array([0, 0, 1] if point is None else point, dtype=float)
        self.length = float(length)
        self.enable = True

    def Advance(self, pose):
        if not self.enable:
            return np.zeros(6)

        pos = pose[0:3]
        quat = pose[3:7]
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        delta_x = self.point - pos
        f = self.kp_pos * (delta_x + np.array([0.0, 0.0, self.length])) - self.kd_pos * lin_vel

        torque = np.zeros(3, dtype=float)
        if self.kp_ang != 0.0 or self.kd_ang != 0.0:
            # Convert quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w].
            # H2LoRA disables this term to match unitree_mujoco's translational
            # sling; keeping it configurable preserves existing G1 debug behavior.
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
            rot = scipy.spatial.transform.Rotation.from_quat(quat)
            rotvec = rot.as_rotvec()
            torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        import glfw

        if key == glfw.KEY_7:
            self.length -= 0.1
            print(f"ElasticBand length: {self.length:.3f}")
        if key == glfw.KEY_8:
            self.length += 0.1
            print(f"ElasticBand length: {self.length:.3f}")
        if key == glfw.KEY_9:
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")

    def handle_keyboard_button(self, key):
        if key == "9":
            self.enable = not self.enable
            print(f"ElasticBand enable: {self.enable}")
