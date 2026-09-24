"""Register SFYG tasks with policy and independent upper-body control."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.robot import ArmMode


def task_id(arm_mode: ArmMode) -> str:
    if arm_mode not in ("policy", "external"):
        raise ValueError(f"Unsupported arm mode: {arm_mode}")
    return f"Mjlab-Velocity-Flat-TRON2-SFYG-{arm_mode.title()}"


def runner_cfg(arm_mode: ArmMode) -> RslRlOnPolicyRunnerCfg:
    task_id(arm_mode)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
        clip_actions=1.0,
        logger="tensorboard",
        upload_model=False,
        experiment_name=f"tron2_sfyg_{arm_mode}",
        max_iterations=1500,
    )


def register_tasks() -> None:
    registered = set(list_tasks())
    for arm_mode in ("policy", "external"):
        name = task_id(arm_mode)
        if name not in registered:
            register_mjlab_task(
                task_id=name,
                env_cfg=make_env_cfg(arm_mode),
                play_env_cfg=make_env_cfg(arm_mode, play=True),
                rl_cfg=runner_cfg(arm_mode),
            )
