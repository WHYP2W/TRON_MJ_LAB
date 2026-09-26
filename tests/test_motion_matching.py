"""Algorithmic tests using synthetic fixtures, not training demonstrations."""

import unittest

import numpy as np

from tron2_mjlab.motion_matching import (
    MotionClip,
    MotionDatabase,
    critical_spring,
    integrated_spring,
    matching_features,
    query_features,
)


def make_clip(name="locomotion", **annotations):
    frames = 121
    positions = np.zeros((frames, 3))
    positions[:, 0] = np.arange(frames) / 60
    positions[:, 2] = 0.85
    feet = positions[:, None, :] + np.array([[0.0, 0.15, -0.85], [0.0, -0.15, -0.85]])
    return MotionClip(
        name, 60.0, positions,
        np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        np.zeros((frames, 10)), feet, **annotations,
    )


class MotionMatchingTests(unittest.TestCase):
    def test_features_and_future_horizons(self):
        clip = make_clip()
        features = matching_features(clip)
        self.assertEqual(features.shape, (121, 27))
        np.testing.assert_allclose(features[0, :6], [0.33, 0.0, 0.67, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(features[:, -3:], np.tile([1.0, 0.0, 0.0], (121, 1)), atol=1e-12)

    def test_global_yaw_and_translation_invariance(self):
        clip = make_clip()
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([3.0, -2.0, 0.0])
        rotated = MotionClip(
            "rotated", clip.fps, clip.root_pos @ rotation.T + translation,
            np.tile([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], (clip.frames, 1)),
            clip.joint_pos, clip.foot_pos @ rotation.T + translation,
        )
        np.testing.assert_allclose(matching_features(rotated), matching_features(clip), atol=1e-12)

    def test_constant_velocity_query_matches_database(self):
        clip = make_clip()
        query = query_features(
            clip.root_quat[0], np.array([1.0, 0.0, 0.0]),
            clip.foot_pos[0] - clip.root_pos[0],
            np.tile([1.0, 0.0, 0.0], (2, 1)), np.array([1.0, 0.0]),
        )
        np.testing.assert_allclose(query, matching_features(clip)[0], atol=1e-12)
        result = MotionDatabase([clip]).search(query)
        self.assertEqual(result.clip.name, "locomotion")
        self.assertLess(result.distance, 1e-10)
        self.assertLessEqual(result.frame, 60)

    def test_skill_search_is_confined_to_entry_window(self):
        skill = make_clip("jump", skill_start=40, skill_end=90, entry_frames=20)
        database = MotionDatabase([make_clip(), skill])
        result = database.search(database.features["jump"][80], skill="jump")
        self.assertGreaterEqual(result.frame, 20)
        self.assertLessEqual(result.frame, 40)
        self.assertEqual(database.search(database.features["jump"][80]).clip.name, "locomotion")
        with self.assertRaises(ValueError):
            database.search(np.zeros(27), skill="missing")

    def test_spring_initial_state_and_composition(self):
        value = np.array([1.0, -2.0])
        velocity = np.array([0.3, -0.2])
        goal = np.zeros(2)
        np.testing.assert_allclose(critical_spring(value, velocity, goal, 4.0, 0.0), (value, velocity))
        halfway = critical_spring(value, velocity, goal, 4.0, 0.1)
        np.testing.assert_allclose(critical_spring(*halfway, goal, 4.0, 0.1), critical_spring(value, velocity, goal, 4.0, 0.2), atol=1e-12)

    def test_integrated_spring_derivative(self):
        value, velocity, goal = np.array([0.1]), np.array([0.2]), np.array([2.0])
        epsilon = 1e-5
        derivative = (integrated_spring(value, velocity, goal, 4.0, 0.3 + epsilon) - integrated_spring(value, velocity, goal, 4.0, 0.3 - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(derivative, critical_spring(value, velocity, goal, 4.0, 0.3)[0], atol=1e-8)

    def test_invalid_motion_and_query_rejected(self):
        with self.assertRaises(ValueError):
            make_clip("bad", skill_start=5, skill_end=20, entry_frames=10)
        with self.assertRaises(ValueError):
            MotionDatabase([])
        with self.assertRaises(ValueError):
            MotionDatabase([make_clip()]).search(np.zeros(26))
        with self.assertRaises(ValueError):
            critical_spring(np.zeros(1), np.zeros(1), np.zeros(1), 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
