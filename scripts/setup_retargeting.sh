#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="$project_root/downloads/holosoma"
environment="$project_root/downloads/retarget-env"
revision='bccd4d7451640a2800ddc77e469d911a84f91994'

if [[ -e "$source_root" ]]; then
    if [[ ! -d "$source_root/.git" ]]; then
        printf 'Existing retargeting source is not a Git checkout.\n' >&2
        exit 1
    fi
    if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
        printf 'Existing retargeting source is not at the pinned revision.\n' >&2
        exit 1
    fi
    if [[ -n "$(git -C "$source_root" status --porcelain)" ]]; then
        printf 'Retargeting source has local changes; leaving it unchanged.\n' >&2
        exit 1
    fi
else
    GIT_TERMINAL_PROMPT=0 git clone --filter=blob:none --no-checkout --depth 1 \
        https://github.com/amazon-far/holosoma.git "$source_root"
    git -C "$source_root" sparse-checkout set \
        src/holosoma_retargeting/holosoma_retargeting/src \
        src/holosoma_retargeting/holosoma_retargeting/config_types
    git -C "$source_root" fetch --depth 1 origin "$revision"
    git -C "$source_root" checkout --detach "$revision"
fi

if [[ ! -x "$environment/bin/python" ]]; then
    uv venv --python "$project_root/.venv/bin/python" "$environment"
fi
uv pip install --python "$environment/bin/python" \
    'numpy==2.3.5' 'cvxpy==1.7.4' 'libigl==2.6.1' 'yourdfpy==0.0.58'
uv pip install --python "$environment/bin/python" --no-deps 'smplx==0.1.28'
"$environment/bin/python" - "$project_root" <<'PY'
import site
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = f"python{sys.version_info.major}.{sys.version_info.minor}"
site.addsitedir(str(root / ".venv" / "lib" / version / "site-packages"))
sys.path.insert(0, str(root / "downloads/holosoma/src/holosoma_retargeting"))
import cvxpy
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

assert "CLARABEL" in cvxpy.installed_solvers()
assert InteractionMeshRetargeter is not None
print("OmniRetarget imports and Clarabel solver: PASS")
PY
printf 'Pinned OmniRetarget source: %s\nIsolated tools: %s\n' "$source_root" "$environment"
