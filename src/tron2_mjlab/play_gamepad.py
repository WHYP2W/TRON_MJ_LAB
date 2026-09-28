"""Native policy replay with Xbox locomotion and independent arm control."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import PolicyProtocol

from tron2_mjlab.control import UpperBodyAction
from tron2_mjlab.env_cfg import make_env_cfg
from tron2_mjlab.gamepad import GamepadControl, PygameGamepad
from tron2_mjlab.tasks import runner_cfg


class GamepadVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.vel_command_b[env_ids] = 0.0

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Keep external commands intact between input updates and resets."""
        del dt, env_ids
        self._update_metrics()


@dataclass(kw_only=True)
class GamepadVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> GamepadVelocityCommand:
        return GamepadVelocityCommand(self, env)


def make_gamepad_env_cfg() -> ManagerBasedRlEnvCfg:
    cfg = make_env_cfg(play=True)
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    cfg.commands["twist"] = GamepadVelocityCommandCfg(
        entity_name=twist.entity_name,
        ranges=twist.ranges,
        resampling_time_range=twist.resampling_time_range,
        rel_heading_envs=0.0,
    )
    cfg.episode_length_s = 3600.0
    return cfg


class GamepadViewer(NativeMujocoViewer):
    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        policy: PolicyProtocol,
        gamepad: PygameGamepad,
        *,
        arm_speed: float = 0.6,
        gripper_speed: float = 0.03,
    ) -> None:
        super().__init__(env, policy)
        upper = env.unwrapped.action_manager.get_term("upper_body")
        twist = env.unwrapped.command_manager.get_term("twist")
        if not isinstance(upper, UpperBodyAction):
            raise TypeError("Gamepad replay requires independent arm control")
        if not isinstance(twist, GamepadVelocityCommand):
            raise TypeError(
                "Gamepad replay requires a manual velocity command"
            )
        if env.num_envs != 1:
            raise ValueError("Gamepad replay supports exactly one environment")
        for name, speed, maximum in (
            ("arm_speed", arm_speed, upper.cfg.arm_speed),
            ("gripper_speed", gripper_speed, upper.cfg.gripper_speed),
        ):
            if not math.isfinite(speed) or not 0.0 < speed <= maximum:
                raise ValueError(
                    f"{name} must be finite and in (0, {maximum}]"
                )
        self.gamepad = gamepad
        self.control = GamepadControl()
        self.upper = upper
        self.twist = twist
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.targets = upper.applied_targets.clone()
        self._home_requested = False
        self._forward = 0.0
        self._turn = 0.0
        self._hold_targets()

    def _hold_targets(self) -> None:
        self._home_requested = False
        self._forward = 0.0
        self._turn = 0.0
        self.twist.command.zero_()
        self.targets.copy_(self.upper.applied_targets)
        self.upper.set_targets(self.targets)

    def _disarm(self) -> None:
        self.control.disarm()
        self._hold_targets()

    def pause(self) -> None:
        self._disarm()
        super().pause()

    def resume(self) -> None:
        self._disarm()
        super().resume()

    def reset_environment(self) -> None:
        super().reset_environment()
        self._disarm()

    def tick(self) -> bool:
        was_enabled = self.control.command.enabled
        command = self.control.update(self.gamepad.poll())
        if command.reset:
            self.request_reset()
        if command.pause:
            self.request_toggle_pause()
        if self._is_paused:
            self.control.disarm()
        elif command.home:
            self._home_requested = True
        if was_enabled and not self.control.command.enabled:
            self._hold_targets()
        return super().tick()

    def _execute_step(self) -> bool:
        command = self.control.command
        if not command.enabled:
            self._hold_targets()
        else:
            ranges = self.twist.cfg.ranges
            speed = (
                ranges.lin_vel_x[1]
                if command.forward >= 0.0
                else -ranges.lin_vel_x[0]
            )
            self._forward = command.forward * speed
            self._turn = command.turn * ranges.ang_vel_z[1]
            self.twist.command[:, 0] = self._forward
            self.twist.command[:, 1] = 0.0
            self.twist.command[:, 2] = self._turn
            if self._home_requested:
                self.targets.copy_(self.upper.home)
                self._home_requested = False
            else:
                step_dt = self.env.unwrapped.step_dt
                joint_index = command.arm_pair * 2
                self.targets[:, joint_index] += (
                    command.arm_horizontal * self.arm_speed * step_dt
                )
                self.targets[:, joint_index + 1] += (
                    command.arm_vertical * self.arm_speed * step_dt
                )
                gripper_delta = command.gripper * self.gripper_speed * step_dt
                self.targets[:, 6] += gripper_delta
                self.targets[:, 7] -= gripper_delta
            self.targets.clamp_(
                self.upper.limits[..., 0], self.upper.limits[..., 1]
            )
            self.upper.set_targets(self.targets)
        success = super()._execute_step()
        if success and self.env.unwrapped.episode_length_buf[0].item() == 0:
            self._disarm()
        return success

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        status = self.get_status()
        first_joint = self.control.arm_pair * 2 + 1
        labels = (
            "Controller\nInput\nArm joints\nvx [m/s]\nwz [rad/s]\n"
            "Simulation\nStep"
        )
        values = (
            f"{self.gamepad.name or 'Disconnected'}\n"
            f"{'ENABLED' if self.control.command.enabled else 'DISARMED'}\n"
            f"arm{first_joint} / arm{first_joint + 1}\n"
            f"{self._forward:+.2f}\n{self._turn:+.2f}\n"
            f"{'PAUSED' if status.paused else 'RUNNING'} "
            f"({status.speed_label})\n{status.step_count}"
        )
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                labels,
                values,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument(
        "--agent", choices=("trained", "zero"), default="trained"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--controller-index", type=int, default=None)
    parser.add_argument("--deadzone", type=float, default=0.15)
    parser.add_argument("--arm-speed", type=float, default=0.6)
    parser.add_argument("--gripper-speed", type=float, default=0.03)
    parser.add_argument("--list-controllers", action="store_true")
    args = parser.parse_args()
    if not args.list_controllers:
        if args.agent == "trained" and args.checkpoint_file is None:
            parser.error("--checkpoint-file is required for a trained policy")
        if (
            args.checkpoint_file is not None
            and not args.checkpoint_file.is_file()
        ):
            parser.error(f"Checkpoint not found: {args.checkpoint_file}")
        if args.agent == "zero" and args.checkpoint_file is not None:
            parser.error("--agent zero does not load a checkpoint")

    gamepad = PygameGamepad(args.controller_index, args.deadzone)
    try:
        if args.list_controllers:
            devices = gamepad.devices()
            for index, name in devices:
                print(f"{index}: {name}")
            if not devices:
                print(
                    "No SDL game controllers found. "
                    "Connect your Xbox controller."
                )
            return
        configure_torch_backends()
        device = args.device or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        cfg = make_gamepad_env_cfg()
        agent_cfg = runner_cfg()
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        try:
            wrapped = RslRlVecEnvWrapper(
                env, clip_actions=agent_cfg.clip_actions
            )
            if args.agent == "zero":

                def policy(observations) -> torch.Tensor:
                    del observations
                    return torch.zeros((1, wrapped.num_actions), device=device)

            else:
                runner = MjlabOnPolicyRunner(
                    wrapped, asdict(agent_cfg), device=device
                )
                runner.load(
                    str(args.checkpoint_file),
                    load_cfg={"actor": True},
                    strict=True,
                    map_location=device,
                )
                policy = runner.get_inference_policy(device=device)
            viewer = GamepadViewer(
                wrapped,
                policy,
                gamepad,
                arm_speed=args.arm_speed,
                gripper_speed=args.gripper_speed,
            )
            print(
                "[GAMEPAD] Release LB and center sticks/triggers, "
                "then hold LB to enable."
            )
            if args.agent == "zero":
                print("[WARN] Zero actions cannot balance or walk the robot.")
            viewer.run()
        finally:
            env.close()
    finally:
        gamepad.close()


if __name__ == "__main__":
    main()
