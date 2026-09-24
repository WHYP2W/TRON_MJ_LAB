import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from tron2_mjlab.control import UpperBodyAction
from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.robot import UPPER_JOINTS
from tron2_mjlab.tasks import TASK_ID, register_tasks


def test_registered_configs_are_independent_and_local():
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    register_tasks()
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 123
    assert load_env_cfg(TASK_ID).scene.num_envs == 64
    assert load_env_cfg(TASK_ID, play=True).scene.num_envs == 1
    assert load_rl_cfg(TASK_ID).logger == "tensorboard"
    assert not load_rl_cfg(TASK_ID).upload_model


def test_only_external_sfyg_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    assert {
        name
        for name in list_tasks()
        if name.startswith("Mjlab-Velocity-Flat-TRON2-")
    } == {TASK_ID}


@pytest.mark.gpu
def test_native_cuda_environment():
    assert torch.cuda.is_available(), (
        "This integration test requires native CUDA PyTorch"
    )
    cfg = make_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        observations, _ = env.reset()
        dimension = 10
        assert env.action_manager.total_action_dim == dimension
        assert observations["actor"].shape == (2, 74)
        assert observations["critic"].shape == (2, 84)
        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(observations["critic"]).all()
        term = env.action_manager.get_term("upper_body")
        assert isinstance(term, UpperBodyAction)
        actions = torch.zeros(2, dimension, device="cuda:0")
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
