"""Validated motion conversion between retargeting, matching, and mjlab tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tron2_mjlab.motion_matching import MotionClip, MotionDatabase, compose_motion
from tron2_mjlab.robot import LEG_JOINTS, UPPER_JOINTS, get_spec


def export_tracking_motion(source: Path, output: Path, fps: float = 50.0) -> dict:
    if output.suffix != ".npz":
        raise ValueError("Tracking output must have an .npz suffix")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Output frame rate must be finite and positive")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if str(arrays.get("qpos_order", "")) != "xyz_wxyz_joints":
        raise ValueError("Input must explicitly declare MuJoCo qpos ordering")
    if tuple(arrays["joint_names"].tolist()) != LEG_JOINTS + UPPER_JOINTS:
        raise ValueError("Input joint ordering does not match TRON2")
    source_fps = float(arrays["fps"])
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 25 or qpos.shape[0] < 2:
        raise ValueError("Expected at least two frames with 25 TRON2 qpos values")
    if not np.isfinite(qpos).all() or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source motion must be finite and have a positive frame rate")
    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=-1)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError("Source has an invalid root quaternion")
    source_times = np.arange(len(qpos)) / source_fps
    target_times = np.arange(0.0, source_times[-1] + 1e-10, 1 / fps)
    if len(target_times) < 2:
        raise ValueError("Output motion must have at least two frames")
    target_qpos = np.stack([
        np.interp(target_times, source_times, qpos[:, index]) for index in range(25)
    ], axis=-1)
    rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    target_qpos[:, 3:7] = Slerp(source_times, rotations)(target_times).as_quat()[:, [3, 0, 1, 2]]

    model = get_spec().compile()
    data = mujoco.MjData(model)
    joint_names = tuple(model.joint(index).name for index in range(1, model.njnt))
    if joint_names != LEG_JOINTS + UPPER_JOINTS:
        raise ValueError("Compiled model joint ordering changed")
    limits = model.jnt_range[1:]
    if np.any(target_qpos[:, 7:] < limits[:, 0] - 1e-5) or np.any(target_qpos[:, 7:] > limits[:, 1] + 1e-5):
        raise ValueError("Reference motion exceeds the robot joint limits")
    body_names = tuple(model.body(index).name for index in range(1, model.nbody))
    frames = len(target_times)
    qvel = np.zeros((frames, model.nv))
    for frame in range(frames):
        first, last = max(frame - 1, 0), min(frame + 1, frames - 1)
        mujoco.mj_differentiatePos(
            model, qvel[frame], target_times[last] - target_times[first],
            target_qpos[first], target_qpos[last],
        )
    body_pos = np.empty((frames, model.nbody - 1, 3))
    body_quat = np.empty((frames, model.nbody - 1, 4))
    body_lin_vel = np.empty_like(body_pos)
    body_ang_vel = np.empty_like(body_pos)
    foot_pos = np.empty((frames, 2, 3))
    velocity = np.empty(6)
    for frame in range(frames):
        data.qpos[:] = target_qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[1:]
        body_quat[frame] = data.xquat[1:]
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            body_ang_vel[frame, body - 1] = velocity[:3]
            body_lin_vel[frame, body - 1] = velocity[3:]
        foot_pos[frame] = data.xpos[[model.body("ankle_pitch_L_Link").id, model.body("ankle_pitch_R_Link").id]]
    for name in ("costs", "human_joints"):
        arrays.pop(name, None)
    indices = np.clip(np.searchsorted(source_times, target_times + 1e-10, side="right") - 1, 0, len(qpos) - 1)
    for name in ("velocity_commands", "source_clip_names", "source_clip_frames"):
        if name in arrays:
            if len(arrays[name]) != len(qpos):
                raise ValueError(f"{name} must have one entry per source frame")
            arrays[name] = arrays[name][indices]
    arrays.update(
        fps=np.asarray(fps), qpos=target_qpos, qvel=qvel,
        joint_pos=target_qpos[:, 7:], joint_vel=qvel[:, 6:],
        body_pos_w=body_pos, body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel,
        body_names=np.asarray(body_names), foot_pos_w=foot_pos,
        root_pos=target_qpos[:, :3], root_quat=target_qpos[:, 3:7],
    )
    if not all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise ValueError("Converted motion contains non-finite data")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {"frames": frames, "fps": fps, "joints": len(joint_names), "bodies": len(body_names), "output": str(output)}


def load_matching_clip(
    path: Path,
    *,
    name: str | None = None,
    skill_start: int | None = None,
    skill_end: int | None = None,
    entry_frames: int = 0,
    start_frame: int = 0,
    stop_frame: int | None = None,
) -> MotionClip:
    """Load a clip; skill annotations are relative to the selected frame slice."""
    with np.load(path, allow_pickle=False) as motion:
        if tuple(motion["joint_names"].tolist()) != LEG_JOINTS + UPPER_JOINTS:
            raise ValueError("Motion matching requires the canonical TRON2 joint order")
        total = len(motion["root_pos"])
        stop = total if stop_frame is None else stop_frame
        if not isinstance(start_frame, int) or not isinstance(stop, int) or not 0 <= start_frame < stop <= total:
            raise ValueError("Invalid source motion frame slice")
        selection = slice(start_frame, stop)
        return MotionClip(
            name or path.stem, float(motion["fps"]), motion["root_pos"][selection],
            motion["root_quat"][selection], motion["joint_pos"][selection], motion["foot_pos_w"][selection],
            skill_start=skill_start, skill_end=skill_end, entry_frames=entry_frames,
        )


def compose_tracking_motion(manifest: Path, output: Path) -> dict:
    """Compose an annotated clip library and carry paired terrain into mjlab."""
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    specification = json.loads(manifest.read_text())
    clips = []
    paths = {}
    for entry in specification["clips"]:
        name = entry["name"]
        if name in paths:
            raise ValueError(f"Duplicate clip name: {name}")
        path = (manifest.parent / entry["path"]).resolve()
        paths[name] = path
        clips.append(load_matching_clip(
            path, name=name, start_frame=entry.get("start_frame", 0),
            stop_frame=entry.get("stop_frame"), skill_start=entry.get("skill_start"),
            skill_end=entry.get("skill_end"), entry_frames=entry.get("entry_frames", 0),
        ))
    cues = {}
    for cue in specification.get("skills", []):
        if cue["frame"] in cues:
            raise ValueError("Only one skill cue is allowed per output frame")
        cues[cue["frame"]] = cue["name"]
    model = get_spec().compile()
    data = mujoco.MjData(model)
    foot_ids = [model.body(name).id for name in ("ankle_pitch_L_Link", "ankle_pitch_R_Link")]

    def foot_kinematics(qpos: np.ndarray) -> np.ndarray:
        if qpos.shape != (model.nq,):
            raise ValueError("Composed qpos does not match the TRON2 model")
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return data.xpos[foot_ids].copy()

    result = compose_motion(
        MotionDatabase(clips), np.asarray(specification["commands"]), cues,
        foot_kinematics, initial_clip=specification["initial_clip"],
        search_interval=specification.get("search_interval", 10),
        damping=specification.get("damping", 12.0),
    )
    terrain_positions, terrain_sizes, terrain_quaternions = [], [], []
    for placement in result.placements:
        with np.load(paths[placement.name], allow_pickle=False) as source:
            if str(source["terrain"]) != "paired_boxes":
                raise ValueError("Each skill must include its reference terrain")
            rotation = Rotation.from_quat(placement.rotation[[1, 2, 3, 0]])
            terrain_positions.extend(rotation.apply(source["terrain_positions"]) + placement.translation)
            terrain_sizes.extend(source["terrain_half_sizes"])
            orientations = rotation * Rotation.from_quat(source["terrain_quaternions"][:, [1, 2, 3, 0]])
            terrain_quaternions.extend(orientations.as_quat()[:, [3, 0, 1, 2]])
    source_hashes = []
    for path in paths.values():
        with path.open("rb") as stream:
            source_hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    with tempfile.TemporaryDirectory(prefix="tron2-compose-") as directory:
        raw = Path(directory) / "composed.npz"
        motion = result.motion
        np.savez_compressed(
            raw, qpos=np.concatenate((motion.root_pos, motion.root_quat, motion.joint_pos), axis=-1),
            qpos_order="xyz_wxyz_joints", fps=motion.fps,
            joint_names=np.asarray(LEG_JOINTS + UPPER_JOINTS),
            terrain="paired_boxes", terrain_positions=np.asarray(terrain_positions).reshape(-1, 3),
            terrain_half_sizes=np.asarray(terrain_sizes).reshape(-1, 3),
            terrain_quaternions=np.asarray(terrain_quaternions).reshape(-1, 4),
            velocity_commands=result.commands, source_clip_names=np.asarray(result.source_names),
            source_clip_frames=result.source_frames, source_files=np.asarray([str(path) for path in paths.values()]),
            source_sha256=np.asarray(source_hashes), composition_manifest=json.dumps(specification),
            validation_scope="kinematic_composition_not_policy_quality",
        )
        report = export_tracking_motion(raw, output, fps=motion.fps)
    report["skill_instances"] = len(result.placements)
    report["terrain_boxes"] = len(terrain_positions)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--fps", type=float, default=50.0)
    compose = commands.add_parser("compose")
    compose.add_argument("--manifest", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "convert":
        result = export_tracking_motion(args.source, args.output, args.fps)
    else:
        result = compose_tracking_motion(args.manifest, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()