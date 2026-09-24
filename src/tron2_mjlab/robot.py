"""Official SFYG_TRON2A model and simulation actuator configuration."""

import os
from pathlib import Path

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mujoco._enums import mjtJoint
from mujoco._specs import MjSpec

LEG_JOINTS = tuple(
    f"{joint}_{side}_Joint"
    for side in ("L", "R")
    for joint in (
        "proximal_pitch",
        "proximal_roll",
        "proximal_yaw",
        "knee",
        "ankle_pitch",
    )
)
ARM_JOINTS = tuple(f"arm{index}_Joint" for index in range(1, 7))
GRIPPER_JOINTS = ("gripper1_Joint", "gripper2_Joint")
UPPER_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
ARM_HOME = (0.0, 0.35, -0.7, 0.0, 0.0, 0.0)
UPPER_HOME = ARM_HOME + (0.025, -0.025)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def model_path() -> Path:
    asset_root = Path(
        os.environ.get(
            "TRON2_ASSET_ROOT", PROJECT_ROOT / "assets" / "robot-description"
        )
    )
    path = asset_root / "tron2a" / "SFYG_TRON2A" / "xml" / "robot.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Official YG model not found: {path}. "
            "Run scripts/setup_assets.ps1 first, "
            "or set TRON2_ASSET_ROOT to the robot-description checkout."
        )
    return path


def get_spec() -> MjSpec:
    """Load SFYG with scene physics and actuators configured by mjlab."""
    spec = MjSpec.from_file(str(model_path()))
    spec.option.timestep = MjSpec().option.timestep
    spec.njmax = -1
    spec.nconmax = -1
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    for geom in list(spec.geoms):
        if geom.name == "ground":
            spec.delete(geom)
    for light in list(spec.lights):
        spec.delete(light)
    for camera in list(spec.cameras):
        spec.delete(camera)
    spec.body("base_Link").pos = (0.0, 0.0, 0.0)
    for joint in spec.joints:
        if joint.type == mjtJoint.mjJNT_FREE:
            joint.name = "floating_base"
            joint.stiffness[:] = 0.0
            joint.damping[:] = 0.0
    for index, geom in enumerate(spec.geoms):
        collision = geom.contype != 0 or geom.conaffinity != 0
        geom.name = f"{'collision' if collision else 'visual'}_{index}"
        if collision:
            geom.friction = (0.8, 0.005, 0.0001)
            geom.margin = 0.0
    for side in ("L", "R"):
        foot_body = f"ankle_pitch_{side}_Link"
        spec.body(foot_body).add_site(
            name=f"foot_{side}", size=(0.01, 0.0, 0.0)
        )
    spec.body("gripper_pick").add_site(name="tool_tip", size=(0.01, 0.0, 0.0))
    return spec


def robot_cfg() -> EntityCfg:
    initial_positions = dict.fromkeys(LEG_JOINTS, 0.0)
    initial_positions.update(dict(zip(UPPER_JOINTS, UPPER_HOME, strict=True)))
    initial_positions.update(
        proximal_yaw_L_Joint=-3.14159,
        proximal_yaw_R_Joint=3.14159,
    )
    actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "proximal_pitch_[LR]_Joint",
                "proximal_roll_[LR]_Joint",
                "knee_[LR]_Joint",
            ),
            stiffness=159.67,
            damping=10.16,
            effort_limit=140.0,
            armature=0.161777558,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                "proximal_yaw_[LR]_Joint",
                "ankle_pitch_[LR]_Joint",
            ),
            stiffness=53.22,
            damping=3.39,
            effort_limit=40.0,
            armature=0.053923687,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("arm[1-6]_Joint",),
            stiffness=40.0,
            damping=4.0,
            effort_limit=30.0,
            armature=0.0110718,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("gripper[12]_Joint",),
            stiffness=100.0,
            damping=5.0,
            effort_limit=10.0,
            armature=0.0110718,
        ),
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.85),
            joint_pos=initial_positions,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=0.95,
        ),
    )
