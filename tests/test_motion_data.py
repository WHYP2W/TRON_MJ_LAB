"""Motion export tests using synthetic poses on the official TRON2 model."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tron2_mjlab.motion_data import compose_tracking_motion, export_tracking_motion
from tron2_mjlab.robot import LEG_JOINTS, UPPER_JOINTS, robot_cfg


def write_fixture(path: Path, translation: float = 0.0) -> None:
    frames = 151
    names = LEG_JOINTS + UPPER_JOINTS
    initial = robot_cfg().init_state.joint_pos
    qpos = np.zeros((frames, 25))
    qpos[:, 0] = translation + np.arange(frames) / 50
    qpos[:, 2] = 0.85
    qpos[:, 3] = 1.0
    qpos[:, 7:] = [initial[name] for name in names]
    np.savez_compressed(
        path, qpos=qpos, qpos_order="xyz_wxyz_joints", fps=50.0,
        joint_names=np.asarray(names), terrain="paired_boxes",
        terrain_positions=np.array([[translation + 2.0, 0.0, 0.1]]),
        terrain_half_sizes=np.array([[0.5, 0.5, 0.1]]),
        terrain_quaternions=np.array([[1.0, 0.0, 0.0, 0.0]]),
        velocity_commands=np.column_stack((np.arange(frames), np.zeros(frames))),
    )


class MotionDataTests(unittest.TestCase):
    def test_time_scaling_preserves_poses_and_scales_velocities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.npz", root / "half_speed.npz"
            write_fixture(source)
            report = export_tracking_motion(source, output, playback_speed=0.5)
            self.assertEqual(report["frames"], 301)
            with np.load(output, allow_pickle=False) as motion, np.load(source, allow_pickle=False) as original:
                np.testing.assert_allclose(motion["qpos"][::2], original["qpos"], atol=1e-10)
                np.testing.assert_allclose(motion["qvel"][:, 0], 0.5, atol=1e-10)
                np.testing.assert_allclose(motion["terrain_positions"], original["terrain_positions"])
                np.testing.assert_allclose(motion["velocity_commands"][::2], original["velocity_commands"] * 0.5)
                self.assertEqual(float(motion["playback_speed"]), 0.5)

    def test_commands_follow_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.npz", root / "output.npz"
            write_fixture(source)
            export_tracking_motion(source, output, fps=25.0)
            with np.load(output, allow_pickle=False) as motion:
                np.testing.assert_array_equal(motion["velocity_commands"][:, 0], np.arange(0, 151, 2))
                self.assertEqual(motion["joint_pos"].shape, (76, 18))
            with self.assertRaises(FileExistsError):
                export_tracking_motion(source, output)

    def test_composition_transforms_paired_terrain_and_records_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, translation in (("walk", 0.0), ("skill", 10.0)):
                raw, converted = root / f"{name}_raw.npz", root / f"{name}.npz"
                write_fixture(raw, translation)
                export_tracking_motion(raw, converted)
            manifest = root / "composition.json"
            manifest.write_text(json.dumps({
                "initial_clip": "walk",
                "clips": [
                    {"name": "walk", "path": "walk.npz"},
                    {"name": "skill", "path": "skill.npz", "skill_start": 40, "skill_end": 90, "entry_frames": 20},
                ],
                "commands": [[1.0, 0.0]] * 180,
                "skills": [{"frame": 20, "name": "skill"}],
            }))
            output = root / "composed.npz"
            report = compose_tracking_motion(manifest, output)
            self.assertEqual(report["skill_instances"], 1)
            self.assertEqual(report["terrain_boxes"], 1)
            with np.load(output, allow_pickle=False) as motion:
                frame = int(motion["source_clip_frames"][20])
                expected = 12.0 + motion["root_pos"][20, 0] - (10.0 + frame / 50)
                self.assertAlmostEqual(motion["terrain_positions"][0, 0], expected)
                self.assertEqual(motion["source_sha256"].shape, (2,))
                self.assertEqual(motion["velocity_commands"].shape, (180, 2))
                self.assertEqual(motion["source_clip_names"][20], "skill")
                np.testing.assert_allclose(motion["terrain_half_sizes"], [[0.5, 0.5, 0.1]])


if __name__ == "__main__":
    unittest.main()
