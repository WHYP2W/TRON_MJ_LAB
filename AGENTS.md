# Agent Guidelines

## Environment and Commands

- Follow [README.md](README.md) for installation, asset setup, training, playback, and controller examples. Its commands assume Bash on Ubuntu 22.04 at the repository root.
- Preserve the Linux baseline in [pyproject.toml](pyproject.toml): Python `>=3.11,<3.13`, pinned mjlab/PyTorch dependencies, and the CUDA package index. Keep [uv.lock](uv.lock) consistent with intentional dependency changes and validate it with `uv lock --check`.
- Use `uv sync --locked` and the project interpreter `.venv/bin/python`; Ubuntu 22.04's system Python 3.10 does not satisfy the project requirements. Do not modify an unrelated Conda environment or replace the system Python.
- Check GPU driver compatibility with the locked CUDA runtime before simulation. On servers without a display, use the documented headless or browser-viewer path rather than the native GLFW viewer.

## Behavioral Boundaries

- Keep task discovery consistent across the `mjlab.tasks` entry point in [pyproject.toml](pyproject.toml), package import in [src/tron2_mjlab/__init__.py](src/tron2_mjlab/__init__.py), and idempotent registration in [src/tron2_mjlab/tasks.py](src/tron2_mjlab/tasks.py). Preserve the sole task ID `Mjlab-Velocity-Flat-TRON2-SFYG-External` unless the requested change requires otherwise; do not modify upstream mjlab tasks.
- Make environment, observation, and reward changes in [src/tron2_mjlab/env_cfg.py](src/tron2_mjlab/env_cfg.py). The policy controls only the 10 leg joints; the `upper_body` action term contributes zero policy action dimensions. Flag observation/action layout changes as checkpoint compatibility changes.
- Keep upper-body target processing in [src/tron2_mjlab/control.py](src/tron2_mjlab/control.py). Preserve joint ordering, soft-limit and rate-limit enforcement, and manual target persistence until replacement, release, or environment reset. Leg policy steps must not overwrite external upper-body targets.
- Keep joint definitions, actuator configuration, and in-memory model adaptation in [src/tron2_mjlab/robot.py](src/tron2_mjlab/robot.py). Adapt the loaded model there instead of editing the downloaded official XML or meshes.

## Assets and Generated Data

- Use [scripts/setup_assets.sh](scripts/setup_assets.sh) for the pinned model checkout. Preserve its revision and dirty-checkout protections; never overwrite locally modified assets to fix setup.
- `TRON2_ASSET_ROOT` must point to the model checkout containing `tron2a/`, not directly to the XML directory. Missing assets are a setup prerequisite, not a reason to replace the model with a stub.
- Keep downloaded assets, virtual environments, caches, logs, and checkpoints out of source control and distribution artifacts, as described in [README.md](README.md) and [.gitignore](.gitignore).

## Validation

- No test suite, lint configuration, or CI workflow is currently provided. Do not assume pytest or Ruff is a configured project check.
- For Python edits, a dependency-free syntax check is `python -m compileall -q src/tron2_mjlab` using an available interpreter in the supported range. This does not validate imports, task registration, or simulation behavior.
- For runtime checks, use the zero-action playback command in [README.md](README.md) after confirming the supported environment, installed dependencies, model assets, and GPU/viewer availability. Package import registers tasks; a successful import is not proof that model loading or simulation works.
- Do not launch dependency/asset downloads or long training runs as routine validation. When training is requested, follow the documented environment-count guidance and run only one training process at a time.
- Report exactly which checks ran and distinguish syntax-only validation, runtime smoke checks, and training results; do not claim simulation or policy quality from static checks.
