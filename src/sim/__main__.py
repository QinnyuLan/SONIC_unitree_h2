"""Entry point for running a MuJoCo simulation loop.

Parses a YAML-based WBC config via tyro CLI and launches the simulator
(optionally with offscreen image publishing).
"""

import os

import tyro

from sim.factory import SimulatorFactory
from sim.config import SimLoopConfig

ArgsConfig = SimLoopConfig


def _optional_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.lower() not in {"0", "false", "no", "off"}


def _optional_env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _optional_env_float(name: str) -> float | None:
    value = _optional_env_value(name)
    if value is None:
        return None
    return float(value)


class SimWrapper:
    def __init__(self, env_name: str, config: dict[str, object], **kwargs):
        self.config = config

        # Create simulator using factory
        self.sim = SimulatorFactory.create_simulator(
            config=self.config,
            env_name=env_name,
            **kwargs,
        )


def main(config: ArgsConfig):
    wbc_config = config.load_wbc_yaml()
    # NOTE: we will override the interface to local if it is not specified
    wbc_config["ENV_NAME"] = config.env_name
    diagnostic_dir = _optional_env_value("GEAR_SONIC_SIM_DIAGNOSTICS_DIR")
    if diagnostic_dir is not None:
        wbc_config["DIAGNOSTICS_DIR"] = diagnostic_dir
        wbc_config["DIAGNOSTICS_HZ"] = float(os.environ.get("GEAR_SONIC_SIM_DIAGNOSTICS_HZ", "200"))
    domain_override = _optional_env_value("GEAR_SONIC_DOMAIN_ID")
    if domain_override is not None:
        wbc_config["DOMAIN_ID"] = int(domain_override)
    elastic_override = _optional_env_bool("GEAR_SONIC_MUJOCO_ENABLE_ELASTIC")
    if elastic_override is not None:
        wbc_config["ENABLE_ELASTIC_BAND"] = elastic_override
    foot_contact_override = _optional_env_bool(
        "GEAR_SONIC_MUJOCO_H2_OFFICIAL_FOOT_CONTACTS"
    )
    if foot_contact_override is not None:
        wbc_config["ADD_H2_OFFICIAL_FOOT_CONTACT_GEOMS"] = foot_contact_override
    pelvis_imu_pitch_bias_override = _optional_env_float(
        "GEAR_SONIC_MUJOCO_PELVIS_IMU_PITCH_BIAS_DEG"
    )
    if pelvis_imu_pitch_bias_override is not None:
        wbc_config["PELVIS_IMU_PITCH_BIAS_DEG"] = pelvis_imu_pitch_bias_override

    if config.enable_image_publish:
        assert (
            config.enable_offscreen
        ), "enable_offscreen must be True when enable_image_publish is True"

    sim_wrapper = SimWrapper(
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )
    # Start simulator as independent process
    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
