"""Numerical checks of the PHP distillation objective, without policy training."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.models import CNNModel, MLPModel
from rsl_rl.storage import RolloutStorage

from tron2_mjlab.distillation import PhpPPO, distillation_schedule, hybrid_loss, masked_dagger_loss
from tron2_mjlab.env_cfg import DepthObservation, make_student_env_cfg
from tron2_mjlab.tasks import STUDENT_TASK_ID


class DistillationTests(unittest.TestCase):
    def test_curriculum_keeps_dagger_and_relaxes_termination(self):
        first = distillation_schedule(0, 20000)
        middle = distillation_schedule(10000, 20000)
        last = distillation_schedule(20000, 20000)
        self.assertEqual(first.dagger_weight, 1.0)
        self.assertEqual(first.ppo_weight, 0.0)
        self.assertEqual(first.termination_threshold, 0.5)
        self.assertEqual(middle.dagger_weight, 0.1)
        self.assertEqual(last, middle)
        self.assertEqual(last.termination_threshold, 1.0)
        self.assertFalse(distillation_schedule(1000, 20000).adaptive_learning_rate)
        self.assertTrue(distillation_schedule(1001, 20000).adaptive_learning_rate)

    def test_invalid_teacher_states_do_not_contribute(self):
        student = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[2.0, 2.0], [float("nan"), float("nan")]], requires_grad=True)
        loss = masked_dagger_loss(student, teacher, torch.tensor([True, False]))
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad[1], torch.zeros(2))
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_empty_mask_is_differentiable_zero(self):
        student = torch.ones(2, 10, requires_grad=True)
        loss = masked_dagger_loss(student, torch.ones_like(student), torch.zeros(2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_joint_objective_sends_only_student_gradients(self):
        student = torch.zeros(1, 10, requires_grad=True)
        teacher = torch.ones(1, 10, requires_grad=True)
        ppo = torch.tensor(2.0, requires_grad=True)
        schedule = distillation_schedule(5000, 20000)
        loss, dagger = hybrid_loss(ppo, student, teacher, torch.tensor([True]), schedule)
        self.assertEqual(dagger.item(), 1.0)
        self.assertEqual(loss.item(), 6.0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(ppo.grad.item(), 0.5)
        self.assertTrue(torch.all(student.grad < 0))

    def test_depth_sample_hold_and_reset(self):
        sensor = SimpleNamespace(
            cfg=SimpleNamespace(height=58, width=87),
            data=SimpleNamespace(depth=torch.ones(2, 58, 87, 1)),
        )
        env = SimpleNamespace(scene={"depth_camera": sensor}, num_envs=2, device="cpu", common_step_counter=0, step_dt=0.02)
        term = DepthObservation(ObservationTermCfg(func=DepthObservation, params={"sensor_name": "depth_camera"}), env)
        first = term(env, "depth_camera").clone()
        sensor.data.depth.fill_(2.0)
        env.common_step_counter = 1
        torch.testing.assert_close(term(env, "depth_camera"), first)
        env.common_step_counter = 2
        torch.testing.assert_close(term(env, "depth_camera"), torch.full_like(first, 2 / 3))
        term.reset(torch.tensor([0]))
        sensor.data.depth[0].fill_(float("nan"))
        self.assertTrue(torch.isfinite(term(env, "depth_camera")).all())

    def test_cnn_cannot_read_teacher_observations(self):
        cfg = make_student_env_cfg()
        self.assertNotIn("base_lin_vel", cfg.observations["actor"].terms)
        self.assertNotIn("height_scan", cfg.observations["actor"].terms)
        self.assertEqual(cfg.commands["motion"].sampling_mode, "uniform")
        settings = load_rl_cfg(STUDENT_TASK_ID)
        self.assertEqual(settings.actor.hidden_dims, (2048, 1024, 512, 256, 128))
        observations = TensorDict({
            "actor": torch.randn(2, 70), "depth": torch.rand(2, 1, 58, 87),
            "teacher": torch.randn(2, 177), "critic": torch.randn(2, 276),
        }, batch_size=[2])
        actor = CNNModel(
            observations, settings.obs_groups, "actor", 10,
            hidden_dims=(32, 16), cnn_cfg=settings.actor.cnn_cfg,
        ).eval()
        with torch.no_grad():
            expected = actor(observations)
            observations["teacher"].fill_(1000.0)
            observations["critic"].fill_(-1000.0)
            torch.testing.assert_close(actor(observations), expected)

    def test_synthetic_update_and_curriculum_resume(self):
        observations = TensorDict({
            "actor": torch.randn(4, 6), "critic": torch.randn(4, 8),
            "teacher": torch.randn(4, 7), "teacher_valid": torch.ones(4, 1, dtype=torch.bool),
        }, batch_size=[4])
        actor = MLPModel(
            observations, {"actor": ["actor"]}, "actor", 10, hidden_dims=(16,),
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        )
        critic = MLPModel(observations, {"critic": ["critic"]}, "critic", 1, hidden_dims=(16,))
        teacher = MLPModel(
            observations, {"teacher": ["teacher"]}, "teacher", 10,
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "synthetic_fixture.pt"
            torch.save({"actor_state_dict": teacher.state_dict()}, checkpoint)
            storage = RolloutStorage("rl", 4, 4, observations, [10], "cpu")
            algorithm = PhpPPO(
                actor, critic, storage, teacher_checkpoints=(str(checkpoint),),
                num_learning_epochs=1, num_mini_batches=2,
            )
            before = [parameter.detach().clone() for parameter in actor.parameters()]
            with torch.no_grad():
                for _ in range(4):
                    algorithm.act(observations)
                    algorithm.process_env_step(observations, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
                algorithm.compute_returns(observations)
            metrics = algorithm.update()
            self.assertEqual(algorithm.php_iteration, 1)
            self.assertEqual(metrics["dagger_weight"], 1.0)
            self.assertTrue(any(not torch.equal(first, second) for first, second in zip(before, actor.parameters())))
            self.assertTrue(all(parameter.grad is None for parameter in algorithm.teachers[0].parameters()))
            saved = algorithm.save()
            algorithm.php_iteration = 0
            algorithm.load(saved, None, True)
            self.assertEqual(algorithm.php_iteration, 1)
            saved["php_teacher_sha256"] = ["wrong"]
            with self.assertRaises(ValueError):
                algorithm.load(saved, None, True)


if __name__ == "__main__":
    unittest.main()
