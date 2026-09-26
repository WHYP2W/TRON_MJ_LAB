"""SFYG locomotion environments with independent upper-body control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from tron2_mjlab.control import UpperBodyActionCfg, upper_body_targets
from tron2_mjlab.robot import (
    LEG_JOINTS,
    UPPER_JOINTS,
    robot_cfg,
)

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Tron2VelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> Tron2VelocityCommand:
        return Tron2VelocityCommand(self, env)


class Tron2VelocityCommand(UniformVelocityCommand):
    """Keep fixed velocity axes disabled in the playback joystick."""

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        from viser import Icon

        sliders: list[viser.GuiSliderHandle] = []
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                minimum, maximum = getattr(self.cfg.ranges, axis)
                fixed = minimum == maximum
                slider = server.gui.add_slider(
                    axis,
                    min=minimum - 0.05 if fixed else minimum,
                    max=maximum + 0.05 if fixed else maximum,
                    step=0.05,
                    initial_value=min(maximum, max(minimum, 0.0)),
                    disabled=fixed,
                )
                sliders.append(slider)

            zero_button = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_button.on_click
            def zero_command(_event) -> None:
                for slider in sliders:
                    if not slider.disabled:
                        slider.value = min(slider.max, max(slider.min, 0.0))

        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx


@dataclass(kw_only=True)
class StaticObstacleTerrainCfg(SubTerrainCfg):
    """A straight obstacle lane with a flat approach and landing area."""

    kind: Literal["platform", "stairs", "gap"]
    height_range: tuple[float, float] = (0.05, 0.50)
    gap_width_range: tuple[float, float] = (0.10, 0.60)
    approach_length: float = 3.0
    platform_length: float = 1.5
    step_width: float = 0.30
    num_steps: int = 4
    pit_depth: float = 1.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        if self.kind not in ("platform", "stairs", "gap"):
            raise ValueError(f"Unknown obstacle kind: {self.kind}")
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("Obstacle difficulty must be between 0 and 1")
        dimensions = (
            *self.size,
            self.approach_length,
            self.platform_length,
            self.step_width,
            self.pit_depth,
        )
        if any(not np.isfinite(value) or value <= 0 for value in dimensions):
            raise ValueError("Obstacle dimensions must be finite and positive")
        for limits in (self.height_range, self.gap_width_range):
            if not all(np.isfinite(value) for value in limits):
                raise ValueError("Obstacle ranges must be finite")
            if not 0.0 < limits[0] <= limits[1]:
                raise ValueError("Obstacle ranges must be positive and ordered")
        if self.num_steps < 1:
            raise ValueError("A staircase must have at least one step")
        if self.approach_length < 2.0 or self.size[1] < 1.5:
            raise ValueError("Obstacle lane must leave room for a flat spawn area")

        height = self.height_range[0] + difficulty * (
            self.height_range[1] - self.height_range[0]
        )
        segments = [(0.0, self.approach_length, 0.0)]
        cursor = self.approach_length
        if self.kind == "platform":
            segments.append((cursor, cursor + self.platform_length, height))
            cursor += self.platform_length
        elif self.kind == "stairs":
            for step in range(1, self.num_steps + 1):
                segments.append((cursor, cursor + self.step_width, step * height))
                cursor += self.step_width
            segments.append(
                (cursor, cursor + self.platform_length, self.num_steps * height)
            )
            cursor += self.platform_length
            for step in range(self.num_steps - 1, 0, -1):
                segments.append((cursor, cursor + self.step_width, step * height))
                cursor += self.step_width
        else:
            gap_width = self.gap_width_range[0] + difficulty * (
                self.gap_width_range[1] - self.gap_width_range[0]
            )
            segments.append((cursor, cursor + gap_width, -self.pit_depth))
            cursor += gap_width
        if cursor > self.size[0] - 2.0:
            raise ValueError("Obstacle lane must leave at least 2 m for landing")
        segments.append((cursor, self.size[0], 0.0))

        body = spec.body("terrain")
        bottom = -self.pit_depth - 0.1
        obstacle_color = {
            "platform": (0.15, 0.60, 0.45, 1.0),
            "stairs": (0.20, 0.45, 0.80, 1.0),
            "gap": (0.25, 0.25, 0.25, 1.0),
        }[self.kind]
        geometries = []
        for start, end, top in segments:
            color = obstacle_color if top != 0.0 else (0.55, 0.58, 0.60, 1.0)
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=((start + end) / 2, self.size[1] / 2, (bottom + top) / 2),
                size=((end - start) / 2, self.size[1] / 2, (top - bottom) / 2),
                friction=(0.8, 0.005, 0.0001),
                rgba=color,
            )
            geometries.append(TerrainGeometry(geom=geom, color=color))
        return TerrainOutput(
            origin=np.array((1.0, self.size[1] / 2, 0.0)),
            geometries=geometries,
        )


def make_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.num_envs = 1 if play else 64
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern="^ankle_pitch_[LR]_Link$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
    )
    cfg.actions = {
        "legs": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=LEG_JOINTS,
            scale=0.3,
            use_default_offset=True,
        ),
    }
    cfg.actions["upper_body"] = UpperBodyActionCfg(entity_name="robot")

    for group in cfg.observations.values():
        group.terms.pop("height_scan", None)
        group.terms.pop("foot_height", None)
        group.terms["base_lin_vel"] = ObservationTermCfg(
            func=env_mdp.base_lin_vel
        )
        group.terms["base_ang_vel"] = ObservationTermCfg(
            func=env_mdp.base_ang_vel
        )
        group.terms["joint_pos"] = ObservationTermCfg(
            func=env_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINTS + UPPER_JOINTS,
                    preserve_order=True,
                )
            },
        )
        group.terms["upper_targets"] = ObservationTermCfg(
            func=upper_body_targets
        )
    cfg.observations["actor"].enable_corruption = not play

    twist = cfg.commands["twist"]
    if not isinstance(twist, UniformVelocityCommandCfg):
        raise TypeError("The twist command must be a velocity command")
    twist.heading_command = False
    twist.rel_heading_envs = 0.0
    twist.ranges.heading = None
    twist.ranges.lin_vel_x = (-0.5, 1.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (-0.6, 0.6)
    twist.debug_vis = False
    if play:
        cfg.commands["twist"] = Tron2VelocityCommandCfg(
            **{
                parameter.name: getattr(twist, parameter.name)
                for parameter in fields(twist)
                if parameter.init
            }
        )

    cfg.events = {
        name: cfg.events[name] for name in ("reset_base", "reset_robot_joints")
    }
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.0, 0.02),
        "yaw": (-0.2, 0.2),
    }
    reward_names = [
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "dof_pos_limits",
        "action_rate_l2",
        "soft_landing",
        "air_time",
        "foot_slip",
    ]
    cfg.rewards = {name: cfg.rewards[name] for name in reward_names}
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_Link",)
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["pose"].weight = 0.2
    cfg.rewards["pose"].params.update(
        asset_cfg=SceneEntityCfg("robot", joint_names=LEG_JOINTS),
        std_standing={".*": 0.3},
        std_walking={".*": 0.6},
        std_running={".*": 0.9},
    )
    cfg.rewards["air_time"].weight = 0.5
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = (
        "foot_L",
        "foot_R",
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["root_too_low"] = TerminationTermCfg(
        func=env_mdp.root_height_below_minimum, params={"minimum_height": 0.4}
    )
    cfg.curriculum = {}
    cfg.decimation = 4
    cfg.episode_length_s = 20.0
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.viewer.body_name = "base_Link"
    return cfg


def out_of_obstacle_lane(
    env: ManagerBasedRlEnv, margin: float = 0.4
) -> torch.Tensor:
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    length, width = terrain.cfg.terrain_generator.size
    position = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    longitudinal = position[:, 0] + 1.0
    return (
        (longitudinal < margin)
        | (longitudinal > length - margin)
        | (position[:, 1].abs() > width / 2 - margin)
    )


def make_obstacle_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Static obstacle infrastructure; not a trained perceptive parkour policy."""
    cfg = make_env_cfg(play=play)
    cfg.scene.num_envs = 3 if play else 64
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(12.0, 4.0),
            num_rows=6,
            curriculum=True,
            sub_terrains={
                "platform": StaticObstacleTerrainCfg(kind="platform"),
                "stairs": StaticObstacleTerrainCfg(
                    kind="stairs", height_range=(0.05, 0.15)
                ),
                "gap": StaticObstacleTerrainCfg(kind="gap"),
            },
        ),
    )
    cfg.actions["upper_body"] = UpperBodyActionCfg(
        entity_name="robot", automatic_motion=False
    )
    cfg.commands["twist"].ranges.lin_vel_x = (0.0, 1.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (0.0, 0.02),
        "yaw": (-0.05, 0.05),
    }
    cfg.terminations["out_of_lane"] = TerminationTermCfg(
        func=out_of_obstacle_lane, time_out=True
    )
    return cfg
