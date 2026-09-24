"""Register SFYG locomotion with independent upper-body control."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tron2_mjlab.env_cfg import make_env_cfg

TASK_ID = "Mjlab-Velocity-Flat-TRON2-SFYG-External"


def runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="tron2_sfyg_external",
        max_iterations=1500,
    )


def register_tasks() -> None:
    if TASK_ID not in list_tasks():
        register_mjlab_task(
            task_id=TASK_ID,
            env_cfg=make_env_cfg(),
            play_env_cfg=make_env_cfg(play=True),
            rl_cfg=runner_cfg(),
        )
