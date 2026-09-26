"""Retarget verified OmniRetarget human poses to TRON2 with the official solver."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import site
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "bccd4d7451640a2800ddc77e469d911a84f91994"
DATA_REVISION = "135259130158f1cb1799e4ab9b6566c938285fc9"
ARCHIVE_SHA256 = "f30bdc915287547dbb2225bdb4efe323dbfcc2237e2a4e82c7482dd05402aea6"


def configure_imports() -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site.addsitedir(str(ROOT / ".venv" / "lib" / version / "site-packages"))
    source = ROOT / "downloads/holosoma"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Run scripts/setup_retargeting.sh to obtain the pinned solver")
    sys.path.insert(0, str(source / "src/holosoma_retargeting"))


def retarget(args: argparse.Namespace) -> dict:
    configure_imports()
    import mujoco
    import numpy as np
    import yourdfpy
    from scipy.spatial.transform import Rotation

    from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import (
        calculate_laplacian_coordinates,
        create_interaction_mesh,
        extract_foot_sticking_sequence_velocity,
        get_adjacency_list,
    )
    from tron2_mjlab.robot import (
        LEG_JOINTS, UPPER_HOME, UPPER_JOINTS, get_retarget_spec, robot_cfg,
    )

    if args.output.exists():
        raise FileExistsError(f"Output already exists; it was not changed: {args.output}")
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Unexpected source archive checksum")
    with zipfile.ZipFile(args.archive) as archive:
        payload = archive.read(args.member)
    with np.load(io.BytesIO(payload), allow_pickle=False) as motion:
        if "human_joints" not in motion:
            raise ValueError("This member has no human poses; select an augmented trajectory")
        human = np.array(motion["human_joints"], dtype=np.float64)
        source_qpos = np.array(motion["qpos"], dtype=np.float64)
        fps = float(motion["fps"])
    if human.ndim != 3 or human.shape[1:] != (len(MOCAP_DEMO_JOINTS), 3):
        raise ValueError("Unexpected human joint layout")
    if source_qpos.shape != (len(human), 36):
        raise ValueError("Expected G1 poses in [wxyz, xyz, 29 joints] layout")
    if not np.isfinite(human).all() or not np.isfinite(source_qpos).all():
        raise ValueError("Source motion contains non-finite values")
    if fps != 30.0:
        raise ValueError("This adapter currently expects the dataset's 30 Hz human poses")
    if args.max_frames is not None:
        human = human[:args.max_frames]
        source_qpos = source_qpos[:args.max_frames]
    if len(human) < 2:
        raise ValueError("Select at least two frames")

    source_rotation = Rotation.from_quat(source_qpos[0, [1, 2, 3, 0]])
    forward = source_rotation.apply([1.0, 0.0, 0.0])
    yaw = np.arctan2(forward[1], forward[0])
    alignment = Rotation.from_rotvec([0.0, 0.0, -yaw])
    translation = np.array([source_qpos[0, 4], source_qpos[0, 5], 0.0])
    human = (human - translation) @ alignment.as_matrix().T * args.scale
    root_position = alignment.apply(source_qpos[0, 4:7] - translation) * args.scale
    root_quaternion = (alignment * source_rotation).as_quat()[[3, 0, 1, 2]]
    mapping = {
        "Spine1": "base_Link",
        "LeftUpLeg": "proximal_pitch_L_Link",
        "LeftLeg": "knee_L_Link",
        "LeftFoot": "ankle_pitch_L_Link",
        "LeftToeBase": "left_toe",
        "RightUpLeg": "proximal_pitch_R_Link",
        "RightLeg": "knee_R_Link",
        "RightFoot": "ankle_pitch_R_Link",
        "RightToeBase": "right_toe",
    }
    lower = {str(index): -1.0 for index in range(3, 7)}
    upper = {str(index): 1.0 for index in range(3, 7)}
    for index, value in enumerate(UPPER_HOME, start=7 + len(LEG_JOINTS)):
        lower[str(index)] = value
        upper[str(index)] = value

    floor_x = np.linspace(human[..., 0].min() - 0.5, human[..., 0].max() + 0.5, 6)
    floor_y = np.linspace(human[..., 1].min() - 0.5, human[..., 1].max() + 0.5, 6)
    grid_x, grid_y = np.meshgrid(floor_x, floor_y)
    ground = np.stack((grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)), axis=-1)
    scene_name = Path(args.member).stem.split("_z_scale_")[0]
    terrain_path = args.archive.parent / "models/terrain" / scene_name / "multi_boxes_z_scale_1.0.urdf"
    terrain = yourdfpy.URDF.load(
        str(terrain_path.resolve()), mesh_dir=str(terrain_path.parent.resolve()),
        load_meshes=True, build_scene_graph=True,
    )
    boxes = []
    terrain_points = [ground]
    for mesh in terrain.scene.dump():
        bounds = mesh.bounding_box_oriented
        if not np.isclose(abs(mesh.volume), bounds.volume, rtol=1e-5):
            raise ValueError("Published terrain mesh is not a box")
        transform = bounds.primitive.transform
        center = alignment.apply(transform[:3, 3] - translation) * args.scale
        orientation = alignment * Rotation.from_matrix(transform[:3, :3])
        boxes.append((center, bounds.primitive.extents * args.scale / 2, orientation.as_quat()[[3, 0, 1, 2]]))
        terrain_points.append((mesh.vertices - translation) @ alignment.as_matrix().T * args.scale)
    if len(boxes) != 2:
        raise ValueError("Expected two published terrain boxes")
    ground = np.concatenate(terrain_points)
    sticking = extract_foot_sticking_sequence_velocity(human, MOCAP_DEMO_JOINTS, ["LeftToeBase", "RightToeBase"])
    sticking[0] = {"LeftToeBase": False, "RightToeBase": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tron2-retarget-") as directory:
        model_path = Path(directory) / "tron2.xml"
        spec = get_retarget_spec()
        for index, (center, half_size, quaternion) in enumerate(boxes):
            spec.worldbody.add_geom(
                name=f"ground_obstacle_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=center, size=half_size, quat=quaternion,
                friction=(0.8, 0.005, 0.0001),
            )
        model_path.write_text(spec.to_xml())
        constants = SimpleNamespace(
            ROBOT_URDF_FILE=str(model_path), ROBOT_DOF=18, OBJECT_NAME="ground",
            FOOT_STICKING_LINKS=["left_toe", "left_heel", "right_toe", "right_heel"],
            DEMO_JOINTS=MOCAP_DEMO_JOINTS, JOINTS_MAPPING=mapping,
            MANUAL_LB=lower, MANUAL_UB=upper, MANUAL_COST={},
            NOMINAL_TRACKING_INDICES=np.arange(3, 7),
        )
        solver = InteractionMeshRetargeter(
            task_constants=constants, object_urdf_path=None,
            foot_sticking_tolerance=0.01, visualize=False, debug=False,
        )
        current = solver.robot_model.qpos0.copy()
        current[:3] = root_position
        current[3:7] = root_quaternion
        for name, value in robot_cfg().init_state.joint_pos.items():
            current[solver.robot_model.joint(name).qposadr[0]] = value
        motions = []
        costs = []
        for frame, joints in enumerate(human):
            mapped = joints[solver.smplh_mapped_joint_indices]
            vertices, tetrahedra = create_interaction_mesh(np.vstack((mapped, ground)))
            adjacency = get_adjacency_list(tetrahedra, len(vertices))
            laplacian = calculate_laplacian_coordinates(vertices, adjacency)
            previous = current.copy()
            current, cost = solver.iterate(
                q_locked=previous, q_n=previous.copy(), q_t_last=previous,
                target_laplacian=laplacian, adj_list=adjacency, obj_pts_local=ground,
                foot_sticking=sticking[frame], w_nominal_tracking=0.5,
                q_a_nominal=previous, init_t=frame == 0,
                n_iter=30 if frame == 0 else 5, frame_idx=frame,
            )
            if not np.isfinite(current).all():
                raise ValueError(f"Non-finite retargeted pose at frame {frame}")
            np.testing.assert_allclose(current[17:25], UPPER_HOME, atol=1e-5)
            limits = solver.robot_model.jnt_range[1:]
            if np.any(current[7:] < limits[:, 0] - 1e-5) or np.any(current[7:] > limits[:, 1] + 1e-5):
                raise ValueError(f"Joint limit violation at frame {frame}")
            motions.append(current.copy())
            costs.append(float(cost))
            if frame % 10 == 0 or frame == len(human) - 1:
                print(f"Retarget frame {frame + 1}/{len(human)}: cost={cost:.5f}", flush=True)

    result = np.stack(motions)
    np.savez_compressed(
        args.output, qpos=result, fps=fps, joint_names=np.array(LEG_JOINTS + UPPER_JOINTS),
        qpos_order="xyz_wxyz_joints", human_joints=human, costs=np.array(costs),
        source_member=args.member, source_archive_sha256=ARCHIVE_SHA256,
        source_member_sha256=hashlib.sha256(payload).hexdigest(),
        retargeter_revision=SOURCE_REVISION, scale=args.scale,
        terrain="paired_boxes", terrain_positions=np.array([box[0] for box in boxes]),
        terrain_half_sizes=np.array([box[1] for box in boxes]),
        terrain_quaternions=np.array([box[2] for box in boxes]),
        terrain_urdf_sha256=hashlib.sha256(terrain_path.read_bytes()).hexdigest(),
        validation_scope="kinematic_retargeting_not_policy_quality",
    )
    return {"output": str(args.output), "frames": len(result), "fps": fps, "max_cost": max(costs), "upper_body_held": True, "terrain": "paired_boxes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "downloads/omniretarget" / DATA_REVISION / "robot-terrain.zip")
    parser.add_argument("--member", default="robot-terrain/climb_16_z_scale_0.8.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")
    if not 0.1 <= args.scale <= 2.0:
        parser.error("--scale must be between 0.1 and 2")
    print(json.dumps(retarget(args), indent=2))


if __name__ == "__main__":
    main()
