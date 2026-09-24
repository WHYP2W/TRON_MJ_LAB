# TRON2 SFYG on Native Windows

An independent mjlab project for **SFYG_TRON2A**, with policy-controlled or
independently controlled upper-body joints. No WSL or reference training
repository is required. Project commands run from this repository root.

This is a runnable flat-ground locomotion training baseline, not a trained
walking/manipulation policy or a hardware deployment package.

## Verified Environment

- Windows native, NVIDIA RTX 4060 Laptop GPU, 8 GB, driver 591.59.
- Python 3.11.15, mjlab 1.6.0, MuJoCo/MuJoCo Warp 3.11.0.
- PyTorch 2.10.0+cu128 and Warp 1.17.0; exact dependencies in `uv.lock`.
- Default training batch: 64 environments, 200 Hz physics, 50 Hz control.
- Native Windows is locally verified; this is not a claim of upstream support
  for every Windows configuration. Single-GPU execution only has been tested.

## Setup

Prerequisites: NVIDIA driver, Git, and [uv](https://docs.astral.sh/uv/).
Run in PowerShell from the repository root:

```powershell
uv sync --locked
.\scripts\setup_assets.ps1
uv run tron2-check --arm-mode policy
```

`uv` creates `.venv` inside this project. Select
`.venv\Scripts\python.exe` as the interpreter when opening this project in
VS Code. CUDA PyTorch downloads approximately 2.7 GiB. The first simulation
compiles Warp kernels and can take several minutes; later runs use its cache.
Do not set `MUJOCO_GL=egl` on Windows; the native GLFW backend is used.

The asset script downloads only the SFYG XML/mesh directories
from [LimX's official model repository](https://github.com/limxdynamics/tron2-robot-description),
pinned to `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`. Its license and notices
remain in the asset checkout. Existing modified checkouts are never overwritten.
Set `TRON2_ASSET_ROOT` to use another compatible checkout with `tron2a/` inside.

## Tasks

| Task | Policy actions | Actor observations |
| --- | ---: | ---: |
| `Mjlab-Velocity-Flat-TRON2-SFYG-Policy` | 18 | 82 |
| `Mjlab-Velocity-Flat-TRON2-SFYG-External` | 10 | 74 |

Both tasks are discovered by mjlab through package entry points. This package
registers only SFYG tasks; upstream mjlab tasks are not modified.

- **Policy:** RL controls locomotion, six arm joints, and two gripper joints.
  Arm actions are bounded position offsets, not a fixed-pose lock. The current
  objective is locomotion; there is no grasp or end-effector tracking reward.
- **External:** RL controls only locomotion. A separate controller writes the
  arm/gripper targets. By default, smoothly changing sinusoidal targets expose
  the locomotion policy to a moving upper-body load during training and play.
- Both modes observe upper-body positions, velocities, desired targets, and
  applied rate-limited targets. All ten leg joints use position control.
- Do not transfer checkpoints between control modes; their observation/action
  contracts differ. Independent mode does not give the policy arm authority.

There is no fixed-pose arm-lock controller. Arm position actuators use
40/4 gains with a 30 Nm simulation torque cap. Grippers use 100/5 with a 10 N
cap. Targets respect soft limits and slew limits (1.5 rad/s arm, 0.04 m/s
fingers). These are simulation starting points, not calibrated hardware limits.
SFYG keeps the required hip-yaw offsets of -pi/+pi. No joint is welded.

## Train

```powershell
uv run train Mjlab-Velocity-Flat-TRON2-SFYG-Policy --env.scene.num-envs 64 --agent.max-iterations 1500
uv run train Mjlab-Velocity-Flat-TRON2-SFYG-External --env.scene.num-envs 64 --agent.max-iterations 1500
```

Run one training command at a time. Start with 64 environments on this 8 GB
GPU; reduce to 16 or 32 if other applications occupy VRAM. Logs/checkpoints
are under `logs/rsl_rl/tron2_sfyg_<mode>/<timestamp>/` when running from
this directory. TensorBoard is the default; no W&B login or upload is required.

```powershell
uv run tensorboard --logdir logs/rsl_rl
uv run train Mjlab-Velocity-Flat-TRON2-SFYG-Policy --agent.resume True --agent.load-run "<run-directory-name>" --agent.load-checkpoint "model_100.pt"
```

Use a checkpoint from the same task and unchanged observation/action layout.
Isaac Lab reference checkpoints are not compatible with these tasks.

## Play and Check

Choose a local checkpoint from your training run:

```powershell
$Checkpoint = '.\logs\rsl_rl\tron2_sfyg_policy\<run>\model_100.pt'
uv run play Mjlab-Velocity-Flat-TRON2-SFYG-Policy --checkpoint-file $Checkpoint --viewer native --num-envs 1
uv run tron2-check --arm-mode policy --checkpoint $Checkpoint --steps 128 --image artifacts\sfyg.png
```

Inspect an untrained model or the independent moving arm controller:

```powershell
uv run play Mjlab-Velocity-Flat-TRON2-SFYG-External --agent zero --viewer native --num-envs 1
uv run tron2-check --arm-mode external --steps 128 --image artifacts\sfyg_external.png
uv run tron2-check --arm-mode external --arm-targets 0.3 0.5 -0.9 0.1 0.2 0.0 0.03 -0.03
```

`tron2-check` runs a finite CUDA rollout, validates observations and rewards,
checks that the upper body actually moves, and writes a JSON report to
`artifacts/`. `--image` saves the first stepped frame, not a demonstration of a
converged gait. Untrained policies/zero agents can fall and automatically reset.

## Independent Controller API

In an existing `ManagerBasedRlEnv` created with `arm_mode="external"`:

```python
import torch

upper_body = env.action_manager.get_term("upper_body")
upper_body.set_targets(
    torch.tensor(
        [0.3, 0.5, -0.9, 0.1, 0.2, 0.0, 0.03, -0.03],
        device=env.device,
    )
)
observations, rewards, terminated, truncated, info = env.step(leg_actions)
upper_body.release_targets()
```

Order: `arm1_Joint` through `arm6_Joint` (radians), then `gripper1_Joint` and
`gripper2_Joint` (metres, opposite signed ranges). Accepts `(8,)` broadcast
targets or `(selected_envs, 8)` targets with an optional `env_ids` tensor.
Targets persist until replaced, explicitly released, or that environment
resets. An external live controller should refresh its command before each
step, including after resets. Non-finite or incorrectly shaped targets are
rejected. The policy mode refuses external overrides.

## Quality Gates

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pyright
uv run pytest -q -m "not gpu"
uv run pytest -q -m gpu
uv build --no-sources
```

The Windows CI workflow runs formatting, lint, standard Pyright checks, non-GPU
tests and package builds on Python 3.11/3.12. Actions and uv are pinned; runtime
and development dependencies are locked. CI uses read-only permissions and
does not run untrusted code on a private GPU runner. GPU tests run locally on
a native CUDA host, not on the GPU-less hosted CI workers.

Tests cover model compilation, movable arm/gripper joints, actuator coverage,
finite control parameters, input validation, control-mode separation, target
rate limits, partial resets, task config isolation and both GPU control modes.
Locally verified: SFYG PPO updates/checkpoint saves, checkpoint replay and
native image rendering. Temporary smoke checkpoints and reference code are
not shipped. Short validation runs are **not trained walking policies**.
Reward tuning, long-run convergence, rough terrain, manipulation tasks and
sim-to-real validation are outside this baseline.

The model adapter changes only the in-memory specification: it removes the
model's standalone ground/motors, adds mjlab actuators/sites, and sets contact
margin to zero as required by MuJoCo Warp MULTICCD. Scene time step and contact
capacity are owned by mjlab, not by the standalone XML options. Official XML
and mesh files remain unchanged. Assets, environments, caches, logs and generated
artifacts are local-only and excluded from Git and the Python package.
