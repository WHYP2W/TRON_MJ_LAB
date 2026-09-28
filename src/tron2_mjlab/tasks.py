"""Register SFYG locomotion with independent upper-body control."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.rough_env_cfg import make_rough_env_cfg

TASK_ID = "Mjlab-Velocity-Flat-TRON2-SFYG-External"
ROUGH_TASK_ID = "Mjlab-Velocity-Rough-TRON2-SFYG-External"


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


def rough_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = runner_cfg()
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    cfg.experiment_name = "tron2_sfyg_rough"
    cfg.max_iterations = 10_000
    return cfg


def register_tasks() -> None:
    registered = list_tasks()
    for task_id, make_cfg, make_rl_cfg in (
        (TASK_ID, make_env_cfg, runner_cfg),
        (ROUGH_TASK_ID, make_rough_env_cfg, rough_runner_cfg),
    ):
        if task_id not in registered:
            register_mjlab_task(
                task_id=task_id,
                env_cfg=make_cfg(),
                play_env_cfg=make_cfg(play=True),
                rl_cfg=make_rl_cfg(),
            )
