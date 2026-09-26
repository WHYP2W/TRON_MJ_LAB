"""Register SFYG locomotion with independent upper-body control."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from tron2_mjlab.distillation import PhpPpoCfg, PhpStudentRunner
from tron2_mjlab.env_cfg import make_env_cfg, make_expert_env_cfg, make_obstacle_env_cfg, make_student_env_cfg

TASK_ID = "Mjlab-Velocity-Flat-TRON2-SFYG-External"
OBSTACLE_TASK_ID = "Mjlab-Velocity-Obstacles-TRON2-SFYG-External"
EXPERT_TASK_ID = "Mjlab-Tracking-TRON2-PHP-Expert"
STUDENT_TASK_ID = "Mjlab-Tracking-TRON2-PHP-Student"


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
    if OBSTACLE_TASK_ID not in list_tasks():
        obstacle_runner_cfg = runner_cfg()
        obstacle_runner_cfg.experiment_name = "tron2_sfyg_obstacles"
        register_mjlab_task(
            task_id=OBSTACLE_TASK_ID,
            env_cfg=make_obstacle_env_cfg(),
            play_env_cfg=make_obstacle_env_cfg(play=True),
            rl_cfg=obstacle_runner_cfg,
        )
    if EXPERT_TASK_ID not in list_tasks():
        expert_runner_cfg = runner_cfg()
        expert_runner_cfg.actor.hidden_dims = (512, 256, 128)
        expert_runner_cfg.critic.hidden_dims = (512, 256, 128)
        expert_runner_cfg.actor.distribution_cfg["init_std"] = 1.0
        expert_runner_cfg.algorithm.entropy_coef = 0.005
        expert_runner_cfg.clip_actions = None
        expert_runner_cfg.max_iterations = 20000
        expert_runner_cfg.experiment_name = "tron2_php_expert"
        register_mjlab_task(
            task_id=EXPERT_TASK_ID,
            env_cfg=make_expert_env_cfg(),
            play_env_cfg=make_expert_env_cfg(play=True),
            rl_cfg=expert_runner_cfg,
            runner_cls=MotionTrackingOnPolicyRunner,
        )
    if STUDENT_TASK_ID not in list_tasks():
        student_runner_cfg = runner_cfg()
        student_runner_cfg.actor = RslRlModelCfg(
            class_name="CNNModel", hidden_dims=(2048, 1024, 512, 256, 128),
            activation="elu", obs_normalization=True,
            cnn_cfg={
                "output_channels": (16, 32, 32), "kernel_size": (5, 3, 3),
                "stride": (2, 2, 2), "global_pool": "avg",
            },
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.01, "std_type": "scalar"},
        )
        student_runner_cfg.critic.hidden_dims = (512, 256, 128)
        student_runner_cfg.obs_groups = {"actor": ("actor", "depth"), "critic": ("critic",)}
        student_runner_cfg.algorithm = PhpPpoCfg(
            num_learning_epochs=2, num_mini_batches=96,
            learning_rate=3e-4, entropy_coef=0.001,
        )
        student_runner_cfg.clip_actions = None
        student_runner_cfg.max_iterations = 20000
        student_runner_cfg.experiment_name = "tron2_php_student"
        register_mjlab_task(
            task_id=STUDENT_TASK_ID,
            env_cfg=make_student_env_cfg(),
            play_env_cfg=make_student_env_cfg(play=True),
            rl_cfg=student_runner_cfg,
            runner_cls=PhpStudentRunner,
        )
