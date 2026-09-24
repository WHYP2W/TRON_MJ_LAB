from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tron2_mjlab.control import UpperBodyAction, UpperBodyActionCfg
from tron2_mjlab.robot import UPPER_HOME, UPPER_JOINTS


def make_action(mode, **overrides):
    home = torch.tensor(UPPER_HOME).repeat(3, 1)
    limits = torch.tensor(
        [
            [-2.6, 2.6],
            [0.08, 3.06],
            [-2.89, -0.07],
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-2.0, 2.0],
            [0.001, 0.049],
            [-0.049, -0.001],
        ]
    ).repeat(3, 1, 1)
    entity = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=home, soft_joint_pos_limits=limits
        ),
        find_joints=Mock(return_value=(list(range(8)), list(UPPER_JOINTS))),
        set_joint_position_target=Mock(),
    )
    env = SimpleNamespace(
        scene={"robot": entity},
        num_envs=3,
        device="cpu",
        physics_dt=0.005,
        step_dt=0.02,
    )
    return UpperBodyAction(
        UpperBodyActionCfg(entity_name="robot", mode=mode, **overrides), env
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "arm_scale",
        "gripper_scale",
        "arm_speed",
        "gripper_speed",
        "motion_frequency",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_control_configuration_requires_finite_positive_values(
    parameter, value
):
    with pytest.raises(ValueError, match=parameter):
        make_action("external", **{parameter: value})


@pytest.mark.parametrize("mode, dimension", [("policy", 8), ("external", 0)])
def test_modes_have_distinct_policy_dimensions(mode, dimension):
    term = make_action(mode)
    assert term.action_dim == dimension
    term.process_actions(torch.ones(3, dimension))
    term.apply_actions()
    assert torch.any(term.applied_targets != term.home)
    assert torch.all(
        (term.applied_targets - term.home).abs() <= term.max_delta + 1e-6
    )
    assert torch.all(term.applied_targets >= term.limits[..., 0])
    assert torch.all(term.applied_targets <= term.limits[..., 1])


def test_external_targets_are_batched_clamped_and_preserved():
    term = make_action("external")
    term.set_targets(torch.full((8,), 100.0), torch.tensor([1]))
    target = term.desired_targets[1].clone()
    for _ in range(3):
        term.process_actions(torch.zeros(3, 0))
        term.apply_actions()
    torch.testing.assert_close(term.desired_targets[1], target)
    torch.testing.assert_close(target, term.limits[1, :, 1])
    assert not torch.equal(term.desired_targets[0], target)
    term.release_targets(torch.tensor([1]))
    term.process_actions(torch.zeros(3, 0))
    assert not torch.equal(term.desired_targets[1], target)


@pytest.mark.parametrize(
    "bad_target",
    [torch.zeros(6), torch.zeros(2, 8), torch.full((8,), float("nan"))],
)
def test_invalid_external_targets_do_not_change_control_state(bad_target):
    term = make_action("external")
    before = term.desired_targets.clone()
    with pytest.raises(ValueError):
        term.set_targets(bad_target)
    torch.testing.assert_close(term.desired_targets, before)


def test_policy_mode_cannot_be_overridden_by_external_controller():
    with pytest.raises(RuntimeError):
        make_action("policy").set_targets(torch.zeros(8))


def test_reset_only_clears_selected_environments():
    term = make_action("external")
    term.set_targets(term.home + 0.01)
    before = term.desired_targets.clone()
    term.reset(torch.tensor([1]))
    torch.testing.assert_close(term.desired_targets[1], term.home[1])
    torch.testing.assert_close(term.desired_targets[[0, 2]], before[[0, 2]])
