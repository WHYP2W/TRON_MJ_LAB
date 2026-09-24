"""Finite native-Windows CUDA rollout, checkpoint, and rendering checks."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from PIL import Image

from tron2_mjlab.control import UpperBodyAction
from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.robot import PROJECT_ROOT
from tron2_mjlab.tasks import runner_cfg


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-mode", choices=("policy", "external"), default="policy"
    )
    parser.add_argument("--num-envs", type=positive_int, default=2)
    parser.add_argument("--steps", type=positive_int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm-targets", type=float, nargs=8)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.arm_targets is not None and args.arm_mode != "external":
        parser.error("--arm-targets requires --arm-mode external")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is unavailable. Run uv sync --locked."
        )
    configure_torch_backends()
    wp.init()
    cfg = make_env_cfg(args.arm_mode, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.viewer.width = 960
    cfg.viewer.height = 720
    env = ManagerBasedRlEnv(
        cfg, device="cuda:0", render_mode="rgb_array" if args.image else None
    )
    try:
        agent_cfg = runner_cfg(args.arm_mode)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        observations = wrapper.get_observations()
        policy = None
        if args.checkpoint:
            runner = MjlabOnPolicyRunner(
                wrapper, asdict(agent_cfg), device="cuda:0"
            )
            runner.load(
                str(args.checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location="cuda:0",
            )
            policy = runner.get_inference_policy(device="cuda:0")
            observations, _ = wrapper.reset()
        term = env.action_manager.get_term("upper_body")
        assert isinstance(term, UpperBodyAction)
        robot = env.scene["robot"]
        initial_upper = robot.data.joint_pos[:, term.joint_ids].clone()
        max_upper_movement = 0.0
        reward_sum = 0.0
        resets = 0
        with torch.inference_mode():
            for step_index in range(args.steps):
                if args.arm_targets is not None:
                    term.set_targets(
                        torch.tensor(args.arm_targets, device=env.device)
                    )
                if policy is not None:
                    actions = policy(observations)
                else:
                    actions = torch.zeros(
                        args.num_envs,
                        env.action_manager.total_action_dim,
                        device=env.device,
                    )
                    if args.arm_mode == "policy":
                        actions[:, -8:] = 0.4 * np.sin(
                            step_index * env.step_dt + 0.5
                        )
                observations, rewards, dones, _ = wrapper.step(actions)
                for name, values in observations.items():
                    if not isinstance(values, torch.Tensor):
                        raise TypeError(
                            f"Expected tensor observations: {name}"
                        )
                    if not torch.isfinite(values).all():
                        raise RuntimeError(
                            f"Non-finite {name} observation "
                            f"at step {step_index}"
                        )
                if not torch.isfinite(rewards).all():
                    raise RuntimeError(
                        f"Non-finite reward at step {step_index}"
                    )
                max_upper_movement = max(
                    max_upper_movement,
                    (robot.data.joint_pos[:, term.joint_ids] - initial_upper)
                    .abs()
                    .max()
                    .item(),
                )
                reward_sum += rewards.mean().item()
                resets += dones.sum().item()
                if step_index == 0 and args.image:
                    pixels = env.render()
                    if pixels is None or np.ptp(pixels.astype(float)) < 20:
                        raise RuntimeError("Rendered frame is blank")
                    args.image.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pixels).save(args.image)
        if max_upper_movement < 1e-5:
            raise RuntimeError("Upper-body joints did not move")
        actor_observations = observations["actor"]
        if not isinstance(actor_observations, torch.Tensor):
            raise TypeError("Actor observations must be a tensor")
        report = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mjlab": version("mjlab"),
            "mujoco": version("mujoco"),
            "warp": version("warp-lang"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "robot": "SFYG_TRON2A",
            "arm_mode": args.arm_mode,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "action_dim": env.action_manager.total_action_dim,
            "actor_observation_dim": actor_observations.shape[-1],
            "upper_body_max_movement": max_upper_movement,
            "mean_reward": reward_sum / args.steps,
            "episode_resets": resets,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "image": str(args.image) if args.image else None,
        }
        output = (
            args.report
            or PROJECT_ROOT / "artifacts" / f"SFYG_{args.arm_mode}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        print(f"Report: {output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
