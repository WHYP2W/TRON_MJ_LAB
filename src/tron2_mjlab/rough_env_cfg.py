"""SFYG locomotion on a terrain curriculum with sim-to-real randomization."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mujoco import MjSpec

from tron2_mjlab import d435
from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.robot import LEG_JOINTS, get_spec
from tron2_mjlab.terrains import terrain_generator_cfg

SOLE_SITES = ("sole_L", "sole_R")
TORSO_BODIES = (
    "base_Link",
    "transition_Link",
    "arm_base_Link",
    "antenna_[LR]_Link",
    "camera_mount_Link",
    "radar_Link",
    "proximal_(pitch|roll|yaw)_[LR]_Link",
)
SCAN_RANGE = 5.0
# Pit floors lie 2 m below grade; walkable surfaces stay above -1 m.
PIT_HEIGHT = -0.8


def terrain_spec() -> MjSpec:
    """SFYG spec with raycast-safe collision groups, soles and two D435s."""
    spec = get_spec()
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.group = 3
    for side, y in (("L", 0.0358488), ("R", -0.0358488)):
        foot = spec.body(f"ankle_pitch_{side}_Link")
        collision = [g for g in foot.geoms if g.contype or g.conaffinity]
        for index, geom in enumerate(collision):
            geom.name = f"foot_{side}_collision_{index}"
            # Priority lets the randomized foot friction govern contacts.
            geom.priority = 1
        foot.add_site(
            name=f"sole_{side}",
            pos=(-0.0144255, y, -0.0737),
            size=(0.01, 0.0, 0.0),
        )
    d435.add_sites(spec)
    return spec


def make_rough_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_env_cfg(play=play)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = terrain_spec
    assert robot.articulation is not None
    # 0-10 ms bus latency between the policy and the motor drivers.
    robot.articulation.actuators = tuple(
        replace(actuator, delay_min_lag=0, delay_max_lag=2)
        for actuator in robot.articulation.actuators
    )

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg(),
        max_init_terrain_level=None if play else 3,
    )
    cfg.scene.sensors = (
        *(cfg.scene.sensors or ()),
        *d435.sensor_cfgs(),
        RayCastSensorCfg(
            name="terrain_scan",
            frame=ObjRef(type="body", name="base_Link", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
            max_distance=SCAN_RANGE,
            include_geom_groups=(0,),
        ),
        TerrainHeightSensorCfg(
            name="foot_height_scan",
            frame=tuple(
                ObjRef(type="site", name=site, entity="robot")
                for site in SOLE_SITES
            ),
            ray_alignment="yaw",
            pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=6),
            max_distance=1.0,
            include_geom_groups=(0,),
        ),
        ContactSensorCfg(
            name="torso_terrain_contact",
            primary=ContactMatch(
                mode="body", pattern=TORSO_BODIES, entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
        ),
    )

    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    actor["base_lin_vel"].noise = Unoise(n_min=-0.1, n_max=0.1)
    actor["base_ang_vel"].noise = Unoise(n_min=-0.2, n_max=0.2)
    actor["joint_pos"].params["biased"] = True
    actor["joint_pos"].noise = Unoise(n_min=-0.01, n_max=0.01)
    # 20-60 ms from exposure to policy input, frames arriving at 30 Hz.
    for name in d435.CAMERAS:
        actor[name] = ObservationTermCfg(
            func=d435.depth,
            params={"sensor_name": name},
            noise=d435.DepthNoiseCfg(),
            scale=1 / d435.MAX_RANGE,
            delay_min_lag=1,
            delay_max_lag=3,
            delay_hold_prob=0.3,
        )
    critic["height_scan"] = ObservationTermCfg(
        func=mdp.height_scan,
        params={"sensor_name": "terrain_scan"},
        scale=1 / SCAN_RANGE,
    )
    critic["foot_height"] = ObservationTermCfg(
        func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}
    )

    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.25, 0.25),
        "y": (-0.25, 0.25),
        "z": (0.01, 0.05),
        "yaw": (-math.pi, math.pi),
    }
    cfg.events.update(
        foot_friction=EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=(r"foot_[LR]_collision_\d+",)
                ),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        encoder_bias=EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        # About 2 degrees of camera mounting and calibration error.
        d435_mount=EventTermCfg(
            mode="startup",
            func=dr.site_quat,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", site_names=tuple(d435.CAMERAS)
                ),
                "roll_range": (-0.035, 0.035),
                "pitch_range": (-0.035, 0.035),
                "yaw_range": (-0.035, 0.035),
            },
        ),
        base_inertia=EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("base_Link",)
                ),
                "alpha_range": (-0.05, 0.1),
                "t_range": (-0.03, 0.03),
            },
        ),
        leg_inertia=EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=(
                        r"proximal_(pitch|roll|yaw)_[LR]_Link",
                        r"(knee|ankle_pitch)_[LR]_Link",
                    ),
                ),
                "alpha_range": (-0.05, 0.05),
            },
        ),
        # Up to 0.5 kg carried by the gripper.
        gripper_payload=EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("gripper_base_Link",)
                ),
                "alpha_range": (0.0, 0.37),
            },
        ),
        motor_gains=EventTermCfg(
            mode="startup",
            func=dr.pd_gains,
            params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)},
        ),
        motor_strength=EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={"effort_limit_range": (0.85, 1.0)},
        ),
        joint_friction=EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
                "ranges": (0.0, 0.3),
            },
        ),
        joint_armature=EventTermCfg(
            mode="startup",
            func=dr.joint_armature,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
                "ranges": (0.9, 1.1),
                "operation": "scale",
            },
        ),
    )

    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=SOLE_SITES, preserve_order=True
            ),
        },
    )
    cfg.rewards["foot_swing_height"] = RewardTermCfg(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )

    cfg.terminations.pop("root_too_low", None)
    cfg.terminations["torso_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "torso_terrain_contact"},
    )
    cfg.terminations["fell_into_pit"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": PIT_HEIGHT},
    )
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=mdp.out_of_terrain_bounds, time_out=True
    )

    if play:
        cfg.events = {
            "randomize_terrain": EventTermCfg(
                func=mdp.randomize_terrain, mode="reset"
            ),
            **cfg.events,
        }
    else:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.3, 0.3),
                    "pitch": (-0.3, 0.3),
                    "yaw": (-0.5, 0.5),
                },
            },
        )
        cfg.events["gripper_bump"] = EventTermCfg(
            func=mdp.apply_body_impulse,
            mode="step",
            params={
                "force_range": (-20.0, 20.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.4),
                "cooldown_s": (3.0, 8.0),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=("gripper_base_Link",)
                ),
            },
        )
        cfg.curriculum = {
            "terrain_levels": CurriculumTermCfg(
                func=mdp.terrain_levels_vel, params={"command_name": "twist"}
            ),
        }

    # Contact buffers dominate GPU memory; measured demand peaks near 12.
    cfg.sim.nconmax = 64
    return cfg
