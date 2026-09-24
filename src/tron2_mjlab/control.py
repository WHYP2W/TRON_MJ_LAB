"""Policy and independent upper-body control through the same actuators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from tron2_mjlab.robot import UPPER_JOINTS, ArmMode

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class UpperBodyActionCfg(ActionTermCfg):
    mode: ArmMode = "policy"
    arm_scale: float = 0.5
    gripper_scale: float = 0.025
    arm_speed: float = 1.5
    gripper_speed: float = 0.04
    motion_frequency: float = 0.15

    def build(self, env: ManagerBasedRlEnv) -> UpperBodyAction:
        return UpperBodyAction(self, env)


class UpperBodyAction(ActionTerm):
    cfg: UpperBodyActionCfg

    def __init__(self, cfg: UpperBodyActionCfg, env: ManagerBasedRlEnv):
        if cfg.mode not in ("policy", "external"):
            raise ValueError(f"Unknown arm control mode: {cfg.mode}")
        for parameter in (
            "arm_scale",
            "gripper_scale",
            "arm_speed",
            "gripper_speed",
            "motion_frequency",
        ):
            value = getattr(cfg, parameter)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{parameter} must be finite and positive")
        super().__init__(cfg, env)
        joint_ids, joint_names = self._entity.find_joints(
            UPPER_JOINTS, preserve_order=True
        )
        if tuple(joint_names) != UPPER_JOINTS:
            raise ValueError(
                f"Unexpected upper-body joint order: {joint_names}"
            )
        self.joint_ids = torch.tensor(
            joint_ids, device=self.device, dtype=torch.long
        )
        self.home = self._entity.data.default_joint_pos[
            :, self.joint_ids
        ].clone()
        self.limits = self._entity.data.soft_joint_pos_limits[
            :, self.joint_ids
        ].clone()
        self.scale = torch.tensor(
            [cfg.arm_scale] * 6 + [cfg.gripper_scale] * 2, device=self.device
        )
        self.max_delta = (
            torch.tensor(
                [cfg.arm_speed] * 6 + [cfg.gripper_speed] * 2,
                device=self.device,
            )
            * env.physics_dt
        )
        self.desired_targets = self.home.clone()
        self.applied_targets = self.home.clone()
        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._manual = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._phase = torch.zeros(self.num_envs, 8, device=self.device)
        self.reset()

    @property
    def action_dim(self) -> int:
        return 8 if self.cfg.mode == "policy" else 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self.cfg.mode == "policy":
            target = self.home + actions.clamp(-1.0, 1.0) * self.scale
        else:
            self._phase.add_(
                2.0 * torch.pi * self.cfg.motion_frequency * self._env.step_dt
            )
            automatic = self.home + 0.5 * self.scale * torch.sin(self._phase)
            target = torch.where(
                self._manual[:, None], self.desired_targets, automatic
            )
        self.desired_targets.copy_(
            target.clamp(self.limits[..., 0], self.limits[..., 1])
        )

    def apply_actions(self) -> None:
        delta = (self.desired_targets - self.applied_targets).clamp(
            -self.max_delta, self.max_delta
        )
        self.applied_targets.add_(delta)
        self._entity.set_joint_position_target(
            self.applied_targets, joint_ids=self.joint_ids
        )

    def set_targets(
        self,
        targets: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set absolute arm radians and finger metres within soft limits.

        Accepts shape (8,) or (selected_envs, 8), ordered as UPPER_JOINTS.
        Manual targets persist until reset or release_targets().
        """
        if self.cfg.mode != "external":
            raise RuntimeError(
                "Independent targets require arm mode 'external'"
            )
        selection = slice(None) if env_ids is None else env_ids
        target = torch.as_tensor(
            targets, dtype=self.home.dtype, device=self.device
        )
        expected = self.desired_targets[selection].shape
        if target.shape not in ((8,), expected):
            raise ValueError(
                f"Expected shape (8,) or {tuple(expected)}, got {target.shape}"
            )
        if not torch.isfinite(target).all():
            raise ValueError(
                "Upper-body targets must contain only finite values"
            )
        limits = self.limits[selection]
        self.desired_targets[selection] = target.clamp(
            limits[..., 0], limits[..., 1]
        )
        self._manual[selection] = True

    def release_targets(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        """Resume automatic motion for the selected environments."""
        selection = slice(None) if env_ids is None else env_ids
        self._manual[selection] = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        selection = slice(None) if env_ids is None else env_ids
        self._raw_actions[selection] = 0.0
        self._manual[selection] = False
        self.desired_targets[selection] = self.home[selection]
        self.applied_targets[selection] = self.home[selection]
        self._phase[selection] = torch.rand_like(self._phase[selection]) * (
            2.0 * torch.pi
        )


def upper_body_targets(env: ManagerBasedRlEnv) -> torch.Tensor:
    term = env.action_manager.get_term("upper_body")
    assert isinstance(term, UpperBodyAction)
    return torch.cat((term.desired_targets, term.applied_targets), dim=-1)
