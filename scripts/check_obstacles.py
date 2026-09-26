"""Check static obstacle geometry and optionally run a short simulation."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains import TerrainGenerator

from tron2_mjlab.control import UpperBodyAction, UpperBodyActionCfg
from tron2_mjlab.env_cfg import (
    StaticObstacleTerrainCfg,
    make_obstacle_env_cfg,
    out_of_obstacle_lane,
)
from tron2_mjlab.tasks import OBSTACLE_TASK_ID, TASK_ID, register_tasks


def check_geometry() -> dict:
    cfg = make_obstacle_env_cfg().scene.terrain.terrain_generator
    spec = mujoco.MjSpec()
    generator = TerrainGenerator(cfg)
    generator.compile(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def height_at(origin: np.ndarray, offset: float) -> float:
        point = origin + np.array((offset, 0.0, 2.0))
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, point, np.array((0.0, 0.0, -1.0)),
            None, 1, -1, geom_id,
        )
        assert distance >= 0.0, (origin, offset)
        return float(point[2] - distance)

    for column, terrain in enumerate(cfg.sub_terrains.values()):
        for row in range(cfg.num_rows):
            difficulty = row / (cfg.num_rows - 1)
            origin = generator.terrain_origins[row, column]
            assert abs(height_at(origin, 0.0)) < 1e-6
            assert abs(height_at(origin, cfg.size[0] - 2.0)) < 1e-6
            approach = terrain.approach_length - 1.0
            assert abs(height_at(origin, approach - 0.01)) < 1e-6
            if terrain.kind == "gap":
                width = terrain.gap_width_range[0] + difficulty * (
                    terrain.gap_width_range[1] - terrain.gap_width_range[0]
                )
                assert abs(height_at(origin, approach + width / 2) + terrain.pit_depth) < 1e-6
                assert abs(height_at(origin, approach + width + 0.01)) < 1e-6
            else:
                height = terrain.height_range[0] + difficulty * (
                    terrain.height_range[1] - terrain.height_range[0]
                )
                assert abs(height_at(origin, approach + 0.05) - height) < 1e-6
                if terrain.kind == "stairs":
                    plateau = approach + terrain.num_steps * terrain.step_width
                    assert abs(height_at(origin, plateau + 0.1) - terrain.num_steps * height) < 1e-6
                    descent = plateau + terrain.platform_length
                    for step in range(terrain.num_steps - 1):
                        offset = descent + (step + 0.5) * terrain.step_width
                        expected = (terrain.num_steps - 1 - step) * height
                        assert abs(height_at(origin, offset) - expected) < 1e-6
    assert np.all(model.geom_size > 0)
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)

    invalid_settings = (
        {"height_range": (0.5, 0.1)},
        {"gap_width_range": (0.0, 0.5)},
        {"pit_depth": float("nan")},
        {"size": (4.0, 4.0)},
        {"num_steps": 0},
    )
    for settings in invalid_settings:
        terrain = StaticObstacleTerrainCfg(kind="platform", **settings)
        try:
            terrain.function(0.0, mujoco.MjSpec(), np.random.default_rng(0))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid terrain configuration accepted: {settings}")
    return {"terrain_cells": cfg.num_rows * len(cfg.sub_terrains), "geometries": model.ngeom}


def check_configuration() -> None:
    register_tasks()
    register_tasks()
    flat = load_env_cfg(TASK_ID)
    obstacles = load_env_cfg(OBSTACLE_TASK_ID)
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.actions["upper_body"].automatic_motion
    assert not obstacles.actions["upper_body"].automatic_motion
    assert flat.observations == obstacles.observations
    assert flat.actions["legs"] == obstacles.actions["legs"]
    assert load_rl_cfg(TASK_ID).experiment_name == "tron2_sfyg_external"
    assert load_rl_cfg(OBSTACLE_TASK_ID).experiment_name == "tron2_sfyg_obstacles"

    term = object.__new__(UpperBodyAction)
    term.cfg = UpperBodyActionCfg(entity_name="robot", automatic_motion=False)
    term._env = SimpleNamespace(step_dt=0.02, device="cpu")
    term._raw_actions = torch.zeros(2, 0)
    term._phase = torch.ones(2, 8)
    term.home = torch.zeros(2, 8)
    term.scale = torch.ones(8)
    term.limits = torch.stack((-torch.ones(2, 8), torch.ones(2, 8)), dim=-1)
    term.desired_targets = term.home.clone()
    term._manual = torch.zeros(2, dtype=torch.bool)
    term.set_targets(torch.full((8,), 0.25), env_ids=slice(0, 1))
    for _ in range(10):
        term.process_actions(torch.empty(2, 0))
    assert torch.all(term.desired_targets[0] == 0.25)
    assert torch.all(term.desired_targets[1] == 0.0)
    assert term.action_dim == 0
    term.release_targets()
    term.process_actions(torch.empty(2, 0))
    assert torch.equal(term.desired_targets, term.home)
    term.cfg.automatic_motion = True
    term.process_actions(torch.empty(2, 0))
    assert torch.any(term.desired_targets != term.home)

    class Scene(dict):
        pass

    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=torch.tensor([
            [0.0, 0.0, 0.85], [-1.0, 0.0, 0.85],
            [0.0, 1.8, 0.85], [11.0, 0.0, 0.85],
        ])
    )))
    scene.env_origins = torch.zeros(4, 3)
    scene.terrain = SimpleNamespace(cfg=obstacles.scene.terrain)
    assert out_of_obstacle_lane(SimpleNamespace(scene=scene)).tolist() == [
        False, True, True, True,
    ]


def check_simulation(device: str, num_envs: int, steps: int) -> dict:
    from mjlab.envs import ManagerBasedRlEnv

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA is required for this device"
        assert torch.ones(4, device=device).sum().item() == 4.0
    cfg = make_obstacle_env_cfg(play=True)
    cfg.seed = 0
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        observations, _ = env.reset()
        assert observations["actor"].shape == (num_envs, 74)
        assert env.action_manager.total_action_dim == 10
        assert set(env.scene.terrain.terrain_types.tolist()) == {0, 1, 2}
        upper = env.action_manager.get_term("upper_body")
        assert upper.action_dim == 0
        target = upper.home.clone()
        target[:, 0] += 0.1
        upper.set_targets(target)
        expected = upper.desired_targets.clone()
        actions = torch.zeros((num_envs, 10), device=device)
        contact_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reset_count = 0
        hold_steps = min(20, steps // 2)
        with torch.inference_mode():
            for step in range(steps):
                previous = upper.applied_targets.clone()
                observations, rewards, terminated, truncated, _ = env.step(actions)
                assert all(torch.isfinite(value).all() for value in observations.values())
                assert torch.isfinite(rewards).all()
                assert torch.isfinite(env.sim.data.qpos).all()
                done = terminated | truncated
                reset_count += int(done.sum().item())
                contact = env.scene["feet_ground_contact"].data
                assert torch.isfinite(contact.force).all()
                contact_seen |= contact.found.reshape(num_envs, -1).any(dim=-1)
                active = upper._manual
                assert torch.allclose(upper.desired_targets[active], expected[active])
                change = (upper.applied_targets - previous).abs()
                assert torch.all(change[~done] <= upper.max_delta * cfg.decimation + 1e-6)
                if step + 1 == hold_steps:
                    upper.release_targets()
                elif step >= hold_steps:
                    assert torch.allclose(upper.desired_targets, upper.home)
        assert contact_seen.all(), contact_seen
        upper.set_targets(expected)
        env.reset()
        assert not upper._manual.any()
        assert torch.allclose(upper.desired_targets, upper.home)
        return {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "actor_obs_dim": 74,
            "action_dim": 10,
            "contact_seen": contact_seen.tolist(),
            "automatic_resets": reset_count,
            "upper_targets_and_rate_limits": "PASS",
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.num_envs < 3 or args.steps < 25:
        parser.error("Use at least 3 environments and 25 simulation steps")
    result = {"geometry": check_geometry()}
    check_configuration()
    result["configuration_and_control"] = "PASS"
    if args.simulate:
        result["simulation"] = check_simulation(args.device, args.num_envs, args.steps)
    result["result"] = "PASS"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
