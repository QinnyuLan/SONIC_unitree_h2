"""Configuration for the released H2 MuJoCo simulator."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from gear_sonic.utils.network.network_utils import resolve_interface


@dataclass
class SimLoopConfig:
    interface: str = "sim"
    wbc_version: Literal["h2_native_unitree_ros_loop"] = "h2_native_unitree_ros_loop"
    sim_frequency: int = 1000
    realtime_max_lag_ms: float = 20.0
    env_name: str = "default"
    enable_onscreen: bool = True
    enable_offscreen: bool = False
    enable_image_publish: bool = False
    camera_port: int = 5555
    mp_start_method: str = "spawn"
    verbose: bool = False
    visual_style: Literal["default", "paper"] = "default"
    presentation_camera: Literal["fixed", "follow"] = "fixed"
    presentation_distance: float = 4.5
    presentation_azimuth: float = 135.0
    presentation_elevation: float = -15.0
    presentation_lookat: tuple[float, float, float] = (0.0, 0.0, 0.85)

    def load_wbc_yaml(self) -> dict:
        if self.sim_frequency <= 0:
            raise ValueError("sim_frequency must be positive")
        interface, env_type = resolve_interface(self.interface)
        path = Path(__file__).parent / "wbc_configs/h2_native_sonic_unitree_ros_loop.yaml"
        config = yaml.safe_load(path.read_text())
        config.update({
            "INTERFACE": interface,
            "ENV_TYPE": env_type,
            "VERSION": self.wbc_version,
            "SIMULATE_DT": 1.0 / self.sim_frequency,
            "REALTIME_MAX_LAG_MS": self.realtime_max_lag_ms,
            "ENABLE_ONSCREEN": self.enable_onscreen,
            "ENABLE_OFFSCREEN": self.enable_offscreen,
            "VISUAL_STYLE": self.visual_style,
            "PRESENTATION_CAMERA": self.presentation_camera,
            "PRESENTATION_DISTANCE": self.presentation_distance,
            "PRESENTATION_AZIMUTH": self.presentation_azimuth,
            "PRESENTATION_ELEVATION": self.presentation_elevation,
            "PRESENTATION_LOOKAT": self.presentation_lookat,
            "verbose": self.verbose,
        })
        return config
