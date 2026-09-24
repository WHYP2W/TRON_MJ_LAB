import mujoco
from mjlab.entity import Entity

from tron2_mjlab.robot import (
    LEG_JOINTS,
    UPPER_HOME,
    UPPER_JOINTS,
    get_spec,
    robot_cfg,
)


def test_official_sfyg_model_has_movable_upper_body():
    spec = get_spec()
    assert spec.option.timestep == mujoco.MjSpec().option.timestep
    assert spec.njmax == -1
    assert spec.nconmax == -1
    model = spec.compile()
    assert model.nq == 25
    assert model.nv == 24
    assert model.neq == 0
    assert model.nu == 0
    for joint_name in UPPER_JOINTS:
        assert model.joint(joint_name).type[0] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        )
    for joint_name, value in zip(UPPER_JOINTS, UPPER_HOME, strict=True):
        lower, upper = model.joint(joint_name).range
        assert lower < value < upper


def test_entity_adds_actuators_for_every_joint():
    robot = Entity(robot_cfg())
    model = robot.spec.compile()
    assert model.nu == 18
    assert set(LEG_JOINTS + UPPER_JOINTS) == {
        model.joint(joint_index).name for joint_index in range(1, model.njnt)
    }
    for joint_name in UPPER_JOINTS:
        joint_id = model.joint(joint_name).id
        assert joint_id in model.actuator_trnid[:, 0]
