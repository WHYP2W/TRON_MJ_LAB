"""SFYG locomotion environments with independent upper-body control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from tron2_mjlab.control import UpperBodyActionCfg, upper_body_targets
from tron2_mjlab.robot import (
    LEG_JOINTS,
    PROJECT_ROOT,
    UPPER_JOINTS,
    get_perceptive_spec,
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


@dataclass(kw_only=True)
class ReferenceTerrainCfg(SubTerrainCfg):
    motion_file: str

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        with np.load(self.motion_file, allow_pickle=False) as motion:
            if str(motion["terrain"]) != "paired_boxes":
                raise ValueError("Expert motion must include its paired terrain")
            positions = motion["terrain_positions"]
            sizes = motion["terrain_half_sizes"]
            quaternions = motion["terrain_quaternions"]
        if positions.shape != sizes.shape or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Invalid paired terrain box layout")
        if quaternions.shape != (len(positions), 4) or np.any(sizes <= 0):
            raise ValueError("Invalid paired terrain sizes or orientations")
        if not all(np.isfinite(value).all() for value in (positions, sizes, quaternions)):
            raise ValueError("Paired terrain must be finite")
        origin = np.array((self.size[0] / 2, self.size[1] / 2, 0.0))
        body = spec.body("terrain")
        floor = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.size[0] / 2, self.size[1] / 2, 0.05),
            pos=origin + (0.0, 0.0, -0.05), friction=(0.8, 0.005, 0.0001),
        )
        geometries = [TerrainGeometry(geom=floor)]
        for position, size, quaternion in zip(positions, sizes, quaternions, strict=True):
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=position + origin,
                quat=quaternion, friction=(0.8, 0.005, 0.0001),
                rgba=(0.20, 0.55, 0.65, 1.0),
            )
            geometries.append(TerrainGeometry(geom=geom))
        return TerrainOutput(origin=origin, geometries=geometries)


def motion_finished(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = env.command_manager.get_term("motion")
    return command.time_steps >= command.motion.time_step_total - 1


@dataclass(kw_only=True)
class PhpMotionCommandCfg(tracking_mdp.MotionCommandCfg):
    start_sampling_probability: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> PhpMotionCommand:
        if not 0.0 <= self.start_sampling_probability <= 1.0:
            raise ValueError("Start sampling probability must lie in [0, 1]")
        return PhpMotionCommand(self, env)


class PhpMotionCommand(tracking_mdp.MotionCommand):
    def __init__(self, cfg: PhpMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.metrics["start_sample"] = torch.zeros(self.num_envs, device=self.device)

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        super()._adaptive_sampling(env_ids)
        start = torch.rand(len(env_ids), device=self.device) < self.cfg.start_sampling_probability
        self.time_steps[env_ids[start]] = 0
        self.metrics["start_sample"][env_ids] = start.float()


def make_expert_env_cfg(*, play: bool = False, motion_file: str | None = None) -> ManagerBasedRlEnvCfg:
    """PHP-style privileged motion-tracking expert with independent upper body."""
    cfg = make_tracking_env_cfg()
    flat = make_env_cfg(play=play)
    cfg.scene.entities = {"robot": robot_cfg()}
    cfg.scene.entities["robot"].spec_fn = get_perceptive_spec
    cfg.scene.num_envs = 1 if play else 64
    motion_file = motion_file or str(PROJECT_ROOT / "downloads/retargeted/tron2_climb16_validated_fps50.npz")
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator", terrain_generator=TerrainGeneratorCfg(
            seed=0, size=(12.0, 12.0), sub_terrains={
                "reference": ReferenceTerrainCfg(motion_file=motion_file),
            },
        ),
    )
    cfg.scene.sensors = (
        *flat.scene.sensors,
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_Link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_Link", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        RayCastSensorCfg(
            name="terrain_scan", frame=ObjRef(type="body", name="base_Link", entity="robot"),
            ray_alignment="yaw", pattern=GridPatternCfg(size=(0.7, 0.7), resolution=0.1),
            max_distance=3.0, exclude_parent_body=True, include_geom_groups=(0,),
        ),
    )
    cfg.actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=LEG_JOINTS, scale=1.0, use_default_offset=True,
        ),
        "upper_body": UpperBodyActionCfg(entity_name="robot", automatic_motion=False),
    }
    motion = cfg.commands["motion"]
    cfg.commands["motion"] = motion = PhpMotionCommandCfg(**{
        parameter.name: getattr(motion, parameter.name)
        for parameter in fields(motion) if parameter.init
    })
    motion.motion_file = motion_file
    motion.anchor_body_name = "base_Link"
    motion.body_names = ("base_Link",) + tuple(name.replace("_Joint", "_Link") for name in LEG_JOINTS)
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start" if play else "adaptive"
    motion.debug_vis = play
    for group in cfg.observations.values():
        group.terms["base_lin_vel"] = ObservationTermCfg(func=env_mdp.base_lin_vel)
        group.terms["base_ang_vel"] = ObservationTermCfg(func=env_mdp.base_ang_vel)
        group.terms["height_scan"] = ObservationTermCfg(
            func=env_mdp.height_scan, params={"sensor_name": "terrain_scan"},
        )
        group.terms["upper_targets"] = ObservationTermCfg(func=upper_body_targets)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_Link",)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = "collision_.*"
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.1)
    cfg.events["push_robot"].params["velocity_range"] = {
        axis: (-0.1, 0.1) for axis in ("x", "y", "roll", "pitch", "yaw")
    } | {"z": (-0.05, 0.05)}
    cfg.rewards["motion_global_root_pos"].weight = 1.0
    cfg.rewards["motion_global_root_ori"].weight = 1.0
    cfg.rewards["self_collisions"].weight = -0.5
    cfg.rewards["self_collisions"].params["force_threshold"] = 1.0
    cfg.terminations["anchor_pos"] = TerminationTermCfg(
        func=tracking_mdp.bad_anchor_pos, params={"command_name": "motion", "threshold": 0.5},
    )
    cfg.terminations["ee_body_pos"].params["body_names"] = ("ankle_pitch_L_Link", "ankle_pitch_R_Link")
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
    cfg.terminations["reference_finished"] = TerminationTermCfg(func=motion_finished, time_out=True)
    cfg.sim = flat.sim
    cfg.viewer = flat.viewer
    cfg.episode_length_s = 20.0
    if play:
        cfg.events = {}
        cfg.observations["actor"].enable_corruption = False
        motion.pose_range = {}
        motion.velocity_range = {}
    return cfg


class DepthObservation:
    """Hold camera samples at 30 Hz; observation-manager delay is applied later."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
        self.sensor = env.scene[cfg.params["sensor_name"]]
        self.cached = torch.full(
            (env.num_envs, 1, self.sensor.cfg.height, self.sensor.cfg.width),
            3.0, device=env.device,
        )
        self.last_frame = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.offset = torch.zeros((env.num_envs, 1, 1, 1), device=env.device)
        self.noisy = cfg.params.get("noise_std", 0.0) > 0
        self.reset()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        self.last_frame[selection] = -1
        self.cached[selection] = 3.0
        self.offset[selection] = (torch.rand_like(self.offset[selection]) - 0.5) * (0.06 if self.noisy else 0.0)

    def __call__(
        self, env: ManagerBasedRlEnv, sensor_name: str, noise_std: float = 0.0
    ) -> torch.Tensor:
        del sensor_name
        camera_frame = int(env.common_step_counter * env.step_dt * 30.0 + 1e-8)
        update = self.last_frame != camera_frame
        if torch.any(update):
            depth = self.sensor.data.depth
            if depth is None:
                raise RuntimeError("The student requires a depth-rendering camera")
            depth = depth.permute(0, 3, 1, 2)
            depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 3.0)
            if noise_std > 0:
                depth = depth + self.offset + torch.randn_like(depth) * noise_std
            self.cached[update] = depth[update].clamp(0.1, 3.0)
            self.last_frame[update] = camera_frame
        return self.cached / 3.0


def teacher_validity(env: ManagerBasedRlEnv) -> torch.Tensor:
    failed = tracking_mdp.bad_anchor_pos(env, "motion", 0.5)
    failed |= tracking_mdp.bad_anchor_ori(env, SceneEntityCfg("robot"), "motion", 0.8)
    failed |= tracking_mdp.bad_motion_body_pos_z_only(
        env, "motion", 0.5, ("ankle_pitch_L_Link", "ankle_pitch_R_Link"),
    )
    return (~failed).unsqueeze(-1)


def student_velocity_command(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.command_manager.get_command("twist")[:, :2]


def make_student_env_cfg(*, play: bool = False, motion_file: str | None = None) -> ManagerBasedRlEnvCfg:
    """Depth-conditioned student environment; reference state is training-only."""
    cfg = make_expert_env_cfg(play=play, motion_file=motion_file)
    flat = make_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 16
    angle = np.deg2rad(15.0)
    camera_rotation = Rotation.from_matrix(np.array([
        [0.0, np.sin(angle), -np.cos(angle)],
        [-1.0, 0.0, 0.0],
        [0.0, np.cos(angle), np.sin(angle)],
    ])).as_quat()[[3, 0, 1, 2]]
    cfg.scene.sensors = (*cfg.scene.sensors, CameraSensorCfg(
        name="depth_camera", parent_body="robot/base_Link",
        pos=(0.25, 0.0, 0.25), quat=tuple(camera_rotation.tolist()),
        width=87, height=58, fovy=58.0, data_types=("depth",),
        use_textures=False, enabled_geom_groups=(0, 1),
    ))
    cfg.commands["twist"] = flat.commands["twist"]
    cfg.commands["twist"].ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.commands["motion"].sampling_mode = "start" if play else "uniform"
    teacher = cfg.observations["actor"]
    actor = flat.observations["actor"]
    actor.terms.pop("base_lin_vel")
    actor.terms["command"] = ObservationTermCfg(func=student_velocity_command)
    cfg.observations = {
        "actor": actor,
        "depth": ObservationGroupCfg(terms={
            "image": ObservationTermCfg(
                func=DepthObservation,
                params={"sensor_name": "depth_camera", "noise_std": 0.0 if play else 0.03},
                delay_min_lag=3, delay_max_lag=4,
            ),
        }),
        "critic": cfg.observations["critic"],
        "teacher": teacher,
        "teacher_valid": ObservationGroupCfg(terms={
            "valid": ObservationTermCfg(func=teacher_validity),
        }),
    }
    return cfg
