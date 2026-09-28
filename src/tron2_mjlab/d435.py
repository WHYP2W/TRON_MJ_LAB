"""Intel RealSense D435 depth views given to the policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.utils.noise.noise_cfg import NoiseCfg
from mujoco._functions import mju_mat2Quat
from mujoco._specs import MjSpec

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

MIN_RANGE = 0.2
MAX_RANGE = 3.0
# 24 x 14 rays keep the 87 x 58 degree depth field of view.
PATTERN = PinholeCameraPatternCfg(width=24, height=14, fovy=58.0)


@dataclass(frozen=True)
class Mount:
    body: str
    """Body whose mesh contains the housing, so raycasts skip it."""
    pos: tuple[float, float, float]
    """Depth origin (left IR imager) in that body's frame, in metres."""
    pitch: float
    """Lens tilt below horizontal, in degrees; both lenses face forward."""


# Measured from the two housings in the official meshes. The model frames
# only the chest camera, and its optical frame points backward.
CAMERAS = {
    "d435_chest": Mount("base_Link", (0.0968, 0.0176, -0.0041), 66.0),
    "d435_mast": Mount("camera_mount_Link", (0.0407, 0.0175, -0.0529), 40.0),
}


def add_sites(spec: MjSpec) -> None:
    """One site per camera: x image right, y image up, lens along -z."""
    for name, mount in CAMERAS.items():
        pitch = math.radians(mount.pitch)
        axes = np.array(
            [
                [0.0, math.sin(pitch), -math.cos(pitch)],
                [-1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), math.sin(pitch)],
            ]
        )
        quat = np.zeros(4)
        mju_mat2Quat(quat, axes.flatten())
        spec.body(mount.body).add_site(
            name=name, pos=mount.pos, quat=quat, size=(0.01, 0.0, 0.0)
        )


def sensor_cfgs() -> tuple[RayCastSensorCfg, ...]:
    return tuple(
        RayCastSensorCfg(
            name=name,
            frame=ObjRef(type="site", name=name, entity="robot"),
            pattern=PATTERN,
            max_distance=MAX_RANGE,
            # Robot visual meshes included so the legs occlude the view.
            include_geom_groups=(0, 1),
        )
        for name in CAMERAS
    )


def depth(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Range along each ray in metres; invalid pixels read MAX_RANGE."""
    distances = env.scene[sensor_name].data.distances
    return torch.where(distances < MIN_RANGE, MAX_RANGE, distances)


@dataclass(kw_only=True)
class DepthNoiseCfg(NoiseCfg):
    """Stereo depth error growing with range squared, plus missing pixels."""

    error_per_metre: float = 0.01
    hole_probability: float = 0.02

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        error = self.error_per_metre * data.square()
        noisy = data + error * (2.0 * torch.rand_like(data) - 1.0)
        holes = torch.rand_like(data) < self.hole_probability
        return torch.where(holes, MAX_RANGE, noisy).clamp(MIN_RANGE, MAX_RANGE)
