import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from tron2_mjlab.control import UpperBodyAction
from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.robot import UPPER_JOINTS
from tron2_mjlab.tasks import register_tasks, task_id


@pytest.mark.parametrize("mode", ["policy", "external"])
def test_registered_configs_are_independent_and_local(mode):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    register_tasks()
    name = task_id(mode)
    cfg = load_env_cfg(name)
    cfg.scene.num_envs = 123
    assert load_env_cfg(name).scene.num_envs == 64
    assert load_env_cfg(name, play=True).scene.num_envs == 1
    assert load_rl_cfg(name).logger == "tensorboard"
    assert not load_rl_cfg(name).upload_model
    assert cfg.actions["upper_body"].mode == mode


def test_only_sfyg_tasks_are_registered():
    from mjlab.tasks.registry import list_tasks

    assert {
        name
        for name in list_tasks()
        if name.startswith("Mjlab-Velocity-Flat-TRON2-")
    } == {task_id("policy"), task_id("external")}


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["policy", "external"])
def test_native_cuda_environment(mode):
    assert torch.cuda.is_available(), (
        "This integration test requires native CUDA PyTorch"
    )
    cfg = make_env_cfg(mode)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        observations, _ = env.reset()
        dimension = 18 if mode == "policy" else 10
        assert env.action_manager.total_action_dim == dimension
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(observations["critic"]).all()
        term = env.action_manager.get_term("upper_body")
        assert isinstance(term, UpperBodyAction)
        actions = torch.zeros(2, dimension, device="cuda:0")
        if mode == "policy":
            actions[:, -8:] = 0.4
        else:
            term.set_targets(term.home + 0.01)
        robot = env.scene["robot"]
        upper_ids, _ = robot.find_joints(UPPER_JOINTS, preserve_order=True)
        initial_upper_pos = robot.data.joint_pos[:, upper_ids].clone()
        for _ in range(8):
            observations, rewards, _, _, _ = env.step(actions)
            assert torch.isfinite(observations["actor"]).all()
            assert torch.isfinite(observations["critic"]).all()
            assert torch.isfinite(rewards).all()
        assert torch.any(term.applied_targets != term.home)
        assert torch.any(
            (robot.data.joint_pos[:, upper_ids] - initial_upper_pos).abs()
            > 1e-5
        )
        assert torch.all(term.applied_targets >= term.limits[..., 0])
        assert torch.all(term.applied_targets <= term.limits[..., 1])
    finally:
        env.close()
