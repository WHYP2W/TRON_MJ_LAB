"""Offline motion-matching primitives for PHP-style skill composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, yaw_quat

FUTURE_TIMES = (0.33, 0.67, 1.0)


@dataclass(frozen=True)
class MotionClip:
    """Retargeted, z-up robot motion; quaternions use wxyz ordering."""

    name: str
    fps: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    foot_pos: np.ndarray
    skill_start: int | None = None
    skill_end: int | None = None
    entry_frames: int = 0

    def __post_init__(self) -> None:
        if not self.name or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Motion name and a finite positive frame rate are required")
        for name in ("root_pos", "root_quat", "joint_pos", "foot_pos"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, array)
        if self.root_pos.ndim != 2:
            raise ValueError("root_pos must have shape (frames, 3)")
        frames = self.root_pos.shape[0]
        expected = {
            "root_pos": (frames, 3),
            "root_quat": (frames, 4),
            "foot_pos": (frames, 2, 3),
        }
        if frames < 2:
            raise ValueError("A motion clip needs at least two frames")
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.joint_pos.ndim != 2 or self.joint_pos.shape[0] != frames:
            raise ValueError("joint_pos must have shape (frames, joints)")
        if self.joint_pos.shape[1] == 0:
            raise ValueError("A motion clip needs at least one joint")
        norm = np.linalg.norm(self.root_quat, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Root quaternions must be nonzero")
        object.__setattr__(self, "root_quat", self.root_quat / norm)
        if (self.skill_start is None) != (self.skill_end is None):
            raise ValueError("Skill start and end must be specified together")
        if self.entry_frames < 0:
            raise ValueError("Entry window length cannot be negative")
        if self.skill_start is not None:
            if not 0 <= self.entry_frames <= self.skill_start <= self.skill_end < frames:
                raise ValueError("Invalid skill interval or pre-skill entry window")
        for name in ("root_pos", "root_quat", "joint_pos", "foot_pos"):
            getattr(self, name).setflags(write=False)

    @property
    def frames(self) -> int:
        return self.root_pos.shape[0]


def _heading_local(vectors: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    quaternions = quat_inv(yaw_quat(torch.tensor(root_quat, dtype=torch.float64)))
    values = torch.tensor(vectors, dtype=torch.float64)
    while quaternions.ndim < values.ndim:
        quaternions = quaternions.unsqueeze(-2)
    quaternions = quaternions.expand(*values.shape[:-1], 4)
    return quat_apply(quaternions, values).numpy()


def matching_features(clip: MotionClip) -> np.ndarray:
    """Build 27-D features in the root's yaw-local character frame."""
    sample_times = np.arange(clip.frames, dtype=np.float64) / clip.fps
    future_times = sample_times[:, None] + np.array(FUTURE_TIMES)
    future_positions = np.stack([
        np.interp(future_times, sample_times, clip.root_pos[:, axis])
        for axis in range(3)
    ], axis=-1)
    future_positions = _heading_local(
        future_positions - clip.root_pos[:, None, :], clip.root_quat
    )[..., :2]
    headings = yaw_quat(torch.tensor(clip.root_quat, dtype=torch.float64)).numpy()
    yaw = np.unwrap(2 * np.arctan2(headings[:, 3], headings[:, 0]))
    future_yaw = np.interp(future_times, sample_times, yaw)
    future_directions = np.stack(
        (np.cos(future_yaw), np.sin(future_yaw), np.zeros_like(future_yaw)), axis=-1
    )
    future_directions = _heading_local(future_directions, clip.root_quat)[..., :2]
    root_velocity = np.gradient(clip.root_pos, 1 / clip.fps, axis=0)
    foot_velocity = np.gradient(clip.foot_pos, 1 / clip.fps, axis=0)
    return np.concatenate((
        future_positions.reshape(clip.frames, 6),
        future_directions.reshape(clip.frames, 6),
        _heading_local(clip.foot_pos - clip.root_pos[:, None, :], clip.root_quat).reshape(clip.frames, 6),
        _heading_local(foot_velocity, clip.root_quat).reshape(clip.frames, 6),
        _heading_local(root_velocity, clip.root_quat),
    ), axis=-1)


def critical_spring(
    value: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    damping: float,
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the critically damped spring in PHP appendix equation 4."""
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("Spring damping must be finite and positive")
    if not np.isfinite(time) or time < 0:
        raise ValueError("Spring time must be finite and nonnegative")
    value, velocity, goal = np.broadcast_arrays(value, velocity, goal)
    if not all(np.isfinite(array).all() for array in (value, velocity, goal)):
        raise ValueError("Spring inputs must be finite")
    offset = value - goal
    coefficient = velocity + damping * offset
    decay = np.exp(-damping * time)
    return (
        decay * (offset + time * coefficient) + goal,
        decay * (velocity - damping * time * coefficient),
    )


def integrated_spring(
    value: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    damping: float,
    time: float,
) -> np.ndarray:
    """Integrate the spring analytically, as in PHP appendix equation 5."""
    critical_spring(value, velocity, goal, damping, time)
    value, velocity, goal = np.broadcast_arrays(value, velocity, goal)
    offset = value - goal
    coefficient = velocity + damping * offset
    decay = np.exp(-damping * time)
    return (
        goal * time
        + offset * (1 - decay) / damping
        + coefficient * (1 - (1 + damping * time) * decay) / damping**2
    )


def query_features(
    root_quat: np.ndarray,
    root_velocity: np.ndarray,
    foot_positions: np.ndarray,
    foot_velocities: np.ndarray,
    command_velocity: np.ndarray,
    *,
    acceleration: np.ndarray | None = None,
    heading_velocity: float = 0.0,
    damping: float = 4.0,
) -> np.ndarray:
    """Build a query from world-frame velocities and root-relative foot positions."""
    root_quat = np.asarray(root_quat, dtype=np.float64)
    root_velocity = np.asarray(root_velocity, dtype=np.float64)
    foot_positions = np.asarray(foot_positions, dtype=np.float64)
    foot_velocities = np.asarray(foot_velocities, dtype=np.float64)
    command_velocity = np.asarray(command_velocity, dtype=np.float64)
    acceleration = np.zeros(2) if acceleration is None else np.asarray(acceleration)
    for value, shape in (
        (root_quat, (4,)), (root_velocity, (3,)),
        (foot_positions, (2, 3)), (foot_velocities, (2, 3)),
        (command_velocity, (2,)), (acceleration, (2,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Expected a finite array of shape {shape}")
    norm = np.linalg.norm(root_quat)
    if norm < 1e-8:
        raise ValueError("Root quaternion must be nonzero")
    root_quat = root_quat / norm
    heading_quat = yaw_quat(torch.tensor(root_quat, dtype=torch.float64)).numpy()
    heading = 2 * np.arctan2(heading_quat[3], heading_quat[0])
    target_heading = heading
    if np.linalg.norm(command_velocity) > 1e-8:
        difference = np.arctan2(command_velocity[1], command_velocity[0]) - heading
        target_heading += np.arctan2(np.sin(difference), np.cos(difference))
    future_positions = np.zeros((3, 3))
    future_directions = np.zeros((3, 3))
    for index, time in enumerate(FUTURE_TIMES):
        future_positions[index, :2] = integrated_spring(
            root_velocity[:2], acceleration, command_velocity, damping, time
        )
        angle, _ = critical_spring(heading, heading_velocity, target_heading, damping, time)
        future_directions[index, :2] = (np.cos(angle), np.sin(angle))
    return np.concatenate((
        _heading_local(future_positions, root_quat)[..., :2].reshape(6),
        _heading_local(future_directions, root_quat)[..., :2].reshape(6),
        _heading_local(foot_positions, root_quat).reshape(6),
        _heading_local(foot_velocities, root_quat).reshape(6),
        _heading_local(root_velocity, root_quat),
    ))


@dataclass(frozen=True)
class MotionMatch:
    clip: MotionClip
    frame: int
    distance: float


class MotionDatabase:
    """Exact nearest-neighbor search with explicit locomotion and skill windows."""

    def __init__(self, clips: list[MotionClip]) -> None:
        if not clips:
            raise ValueError("Motion database cannot be empty")
        if len({clip.name for clip in clips}) != len(clips):
            raise ValueError("Motion clip names must be unique")
        if len({clip.fps for clip in clips}) != 1:
            raise ValueError("Resample all clips to the same frame rate first")
        if len({clip.joint_pos.shape[1] for clip in clips}) != 1:
            raise ValueError("Motion clips must share a joint layout")
        self.clips = {clip.name: clip for clip in clips}
        self.features = {clip.name: matching_features(clip) for clip in clips}

    def search(self, query: np.ndarray, *, skill: str | None = None) -> MotionMatch:
        query = np.asarray(query, dtype=np.float64)
        if query.shape != (27,) or not np.isfinite(query).all():
            raise ValueError("Motion query must be a finite 27-D vector")
        if skill is not None and skill not in self.clips:
            raise ValueError(f"Unknown skill: {skill}")
        best = None
        for clip in self.clips.values():
            if skill is None:
                if clip.skill_start is not None:
                    continue
                last = clip.frames - int(np.ceil(clip.fps))
                candidates = np.arange(max(0, last))
            else:
                if clip.name != skill or clip.skill_start is None:
                    continue
                candidates = np.arange(clip.skill_start - clip.entry_frames, clip.skill_start + 1)
            if candidates.size == 0:
                continue
            difference = self.features[clip.name][candidates] - query
            distances = np.einsum("ij,ij->i", difference, difference)
            index = int(np.argmin(distances))
            distance = float(np.sqrt(distances[index]))
            if best is None or distance < best.distance:
                best = MotionMatch(clip, int(candidates[index]), distance)
        if best is None:
            raise ValueError("No eligible motion frames in the requested search window")
        return best
