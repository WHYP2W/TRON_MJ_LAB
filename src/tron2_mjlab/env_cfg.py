"""Flat-ground SFYG locomotion with independent upper-body control."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg

from tron2_mjlab.control import UpperBodyActionCfg, upper_body_targets
from tron2_mjlab.robot import (
    LEG_JOINTS,
    UPPER_JOINTS,
    robot_cfg,
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
