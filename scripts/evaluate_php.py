"""Evaluate the first complete reference attempt without counting reset retries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from tron2_mjlab.tasks import EXPERT_TASK_ID, STUDENT_TASK_ID


def evaluate(args: argparse.Namespace) -> dict:
    if args.num_envs < 1 or args.perturbation < 0:
        raise ValueError("Use a positive environment count and nonnegative perturbation")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    task_id = EXPERT_TASK_ID if args.policy == "expert" else STUDENT_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.commands["motion"].pose_range = {
        axis: (-args.perturbation, args.perturbation) for axis in ("x", "y", "yaw")
    }
    settings = load_rl_cfg(task_id)
    settings.algorithm.num_mini_batches = min(settings.algorithm.num_mini_batches, args.num_envs)
    environment = ManagerBasedRlEnv(cfg, device=args.device)
    env = RslRlVecEnvWrapper(environment, clip_actions=settings.clip_actions)
    try:
        runner = load_runner_cls(task_id)(env, asdict(settings), device=args.device, registry_name=None)
        runner.load(str(args.checkpoint), load_cfg={"actor": True, "iteration": True}, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        observations, _ = env.reset()
        command = environment.command_manager.get_term("motion")
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        completed = torch.zeros_like(active)
        steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        frame_progress = torch.zeros_like(steps)
        failure_terms = {
            name: torch.zeros_like(active)
            for name in ("anchor_pos", "anchor_ori", "ee_body_pos", "time_out")
        }
        maximum_steps = command.motion.time_step_total + 5
        with torch.inference_mode():
            for _ in range(maximum_steps):
                frame_progress[active] = torch.maximum(frame_progress[active], command.time_steps[active])
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError("Policy produced non-finite actions")
                observations, rewards, done, _ = env.step(actions)
                if not torch.isfinite(rewards).all():
                    raise FloatingPointError("Evaluation reward became non-finite")
                steps[active] += 1
                finished = active & done.bool()
                manager = environment.termination_manager
                completed[finished] = manager.get_term("reference_finished")[finished] & ~manager.terminated[finished]
                for name, mask in failure_terms.items():
                    mask[finished] = manager.get_term(name)[finished]
                active[finished] = False
                if not torch.any(active):
                    break
        with args.checkpoint.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        result = {
            "task": task_id,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checksum,
            "seed": args.seed,
            "initial_xy_yaw_perturbation": args.perturbation,
            "reference_frames": command.motion.time_step_total,
            "trials": args.num_envs,
            "completed": int(completed.sum().item()),
            "completion_rate": completed.float().mean().item(),
            "first_attempt_steps": steps.cpu().tolist(),
            "maximum_reference_frames": frame_progress.cpu().tolist(),
            "failure_counts": {name: int(mask.sum().item()) for name, mask in failure_terms.items()},
            "unfinished": int(active.sum().item()),
            "scope": "single_reference_tracking_not_multiskill_parkour",
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("expert", "student"), default="expert")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    print(json.dumps(evaluate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
