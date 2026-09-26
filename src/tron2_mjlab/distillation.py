"""PHP's joint DAgger/PPO objective and warmup curriculum."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel


@dataclass(frozen=True)
class DistillationSchedule:
    dagger_weight: float
    ppo_weight: float
    termination_threshold: float
    adaptive_learning_rate: bool


def distillation_schedule(iteration: int, total_iterations: int) -> DistillationSchedule:
    if iteration < 0 or total_iterations < 2:
        raise ValueError("Iteration must be nonnegative and total iterations at least two")
    progress = min(1.0, 2.0 * iteration / total_iterations)
    dagger_weight = max(0.1, 1.0 - progress)
    ppo_weight = 1.0 - dagger_weight
    return DistillationSchedule(
        dagger_weight=dagger_weight,
        ppo_weight=ppo_weight,
        termination_threshold=0.5 + 0.5 * progress,
        adaptive_learning_rate=ppo_weight > 0.1 + 1e-12,
    )


def masked_dagger_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> torch.Tensor:
    """Exclude states outside the teacher's original termination boundary."""
    if student_actions.ndim != 2 or student_actions.shape != teacher_actions.shape:
        raise ValueError("Student and teacher actions must have equal (batch, actions) shape")
    if teacher_valid.shape != student_actions.shape[:1] or teacher_valid.dtype != torch.bool:
        raise ValueError("Teacher validity must be a boolean vector with one entry per sample")
    if not torch.any(teacher_valid):
        return student_actions.sum() * 0.0
    student = student_actions[teacher_valid]
    teacher = teacher_actions.detach()[teacher_valid]
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Valid teacher/student actions must be finite")
    return (student - teacher).square().mean()


def hybrid_loss(
    ppo_loss: torch.Tensor,
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_valid: torch.Tensor,
    schedule: DistillationSchedule,
    dagger_coefficient: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dagger_coefficient <= 0:
        raise ValueError("DAgger coefficient must be positive")
    dagger = masked_dagger_loss(student_actions, teacher_actions, teacher_valid)
    return (
        schedule.ppo_weight * ppo_loss + schedule.dagger_weight * dagger_coefficient * dagger,
        dagger,
    )


@dataclass
class PhpPpoCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "tron2_mjlab.distillation:PhpPPO"
    teacher_checkpoints: tuple[str, ...] = ()
    curriculum_iterations: int = 20000
    dagger_coefficient: float = 10.0


class PhpPPO(PPO):
    """Single-GPU feed-forward PPO with jointly optimized teacher imitation."""

    def __init__(
        self, actor, critic, storage, *, teacher_checkpoints=(),
        curriculum_iterations=20000, dagger_coefficient=10.0, **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        if self.is_multi_gpu or self.rnd or self.symmetry or actor.is_recurrent or critic.is_recurrent:
            raise ValueError("PHP distillation currently supports feed-forward, single-GPU training only")
        if self.num_mini_batches > storage.num_envs * storage.num_transitions_per_env:
            raise ValueError("Number of mini-batches exceeds rollout samples")
        distillation_schedule(0, curriculum_iterations)
        if dagger_coefficient <= 0:
            raise ValueError("DAgger coefficient must be positive")
        self.curriculum_iterations = curriculum_iterations
        self.dagger_coefficient = dagger_coefficient
        self.php_iteration = 0
        self._environment = None
        self.teachers = []
        self.teacher_hashes = []
        prototype = storage.observations[0]
        for checkpoint in teacher_checkpoints:
            path = Path(checkpoint)
            with path.open("rb") as stream:
                self.teacher_hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
            teacher = MLPModel(
                prototype, {"teacher": ["teacher"]}, "teacher", storage.actions.shape[-1],
                hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
            )
            saved = torch.load(path, map_location="cpu", weights_only=True)
            teacher.load_state_dict(saved["actor_state_dict"], strict=True)
            self.teachers.append(teacher.to(self.device).eval().requires_grad_(False))

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        algorithm._environment = env.unwrapped
        algorithm._apply_curriculum()
        return algorithm

    def _apply_curriculum(self) -> None:
        if self._environment is None:
            return
        schedule = distillation_schedule(self.php_iteration, self.curriculum_iterations)
        for name in ("anchor_pos", "ee_body_pos"):
            self._environment.termination_manager.get_term_cfg(name).params["threshold"] = schedule.termination_threshold

    def _teacher_actions(self, observations) -> torch.Tensor:
        if not self.teachers:
            raise RuntimeError("Training requires --agent.algorithm.teacher-checkpoints with a validated expert checkpoint")
        with torch.no_grad():
            if len(self.teachers) == 1:
                return self.teachers[0](observations)
            if "teacher_id" not in observations.keys():
                raise ValueError("Multi-expert distillation requires a teacher_id observation")
            identifiers = observations["teacher_id"].reshape(-1).long()
            if torch.any((identifiers < 0) | (identifiers >= len(self.teachers))):
                raise ValueError("A rollout contains an invalid teacher identifier")
            result = torch.zeros((len(identifiers), self.storage.actions.shape[-1]), device=self.device)
            for index, teacher in enumerate(self.teachers):
                selected = identifiers == index
                if torch.any(selected):
                    result[selected] = teacher(observations[selected])
            return result

    def update(self) -> dict[str, float]:
        if not self.teachers:
            raise RuntimeError("No teacher checkpoints were configured for distillation")
        schedule = distillation_schedule(self.php_iteration, self.curriculum_iterations)
        totals = {name: 0.0 for name in ("value", "surrogate", "entropy", "dagger", "teacher_valid_fraction")}
        updates = 0
        batches = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in batches:
            self.actor(batch.observations, stochastic_output=True)
            values = self.critic(batch.observations)
            log_probabilities = self.actor.get_output_log_prob(batch.actions)
            if schedule.adaptive_learning_rate and self.schedule == "adaptive" and self.desired_kl is not None:
                with torch.no_grad():
                    divergence = self.actor.get_kl_divergence(batch.old_distribution_params, self.actor.output_distribution_params).mean().item()
                    if divergence > 2 * self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < divergence < self.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            advantages = batch.advantages.reshape(-1)
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            ratios = torch.exp(log_probabilities - batch.old_actions_log_prob.reshape(-1))
            surrogate = -torch.minimum(
                advantages * ratios,
                advantages * ratios.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            value_error = (values - batch.returns).square()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_error = torch.maximum(value_error, (clipped - batch.returns).square())
            value_loss = value_error.mean()
            entropy = self.actor.output_entropy.mean()
            valid = batch.observations["teacher_valid"].reshape(-1).bool()
            teacher_actions = self._teacher_actions(batch.observations)
            loss, dagger = hybrid_loss(
                surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy,
                self.actor.output_mean, teacher_actions, valid, schedule, self.dagger_coefficient,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PHP distillation loss")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for name, value in (
                ("value", value_loss), ("surrogate", surrogate), ("entropy", entropy),
                ("dagger", dagger), ("teacher_valid_fraction", valid.float().mean()),
            ):
                totals[name] += value.item()
            updates += 1
        if updates == 0:
            raise RuntimeError("No rollout batches were available for distillation")
        self.storage.clear()
        self.php_iteration += 1
        self._apply_curriculum()
        return {name: value / updates for name, value in totals.items()} | {
            "dagger_weight": schedule.dagger_weight, "ppo_weight": schedule.ppo_weight,
        }

    def save(self) -> dict:
        return super().save() | {
            "php_iteration": self.php_iteration,
            "php_teacher_sha256": self.teacher_hashes,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = load_cfg is None or load_cfg.get("iteration", False)
        if restore_iteration and self.teacher_hashes and loaded_dict.get("php_teacher_sha256") != self.teacher_hashes:
            raise ValueError("Teacher checkpoints differ from the saved distillation run")
        result = super().load(loaded_dict, load_cfg, strict)
        if restore_iteration:
            self.php_iteration = int(loaded_dict.get("php_iteration", 0))
            self._apply_curriculum()
        return result


class PhpStudentRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
        del registry_name
        super().__init__(env, train_cfg, log_dir, device)