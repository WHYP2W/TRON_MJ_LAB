#!/usr/bin/env bash
set -euo pipefail

dataset='omniretarget/OmniRetarget_Dataset'
revision='135259130158f1cb1799e4ab9b6566c938285fc9'
archive='robot-terrain.zip'
checksum='f30bdc915287547dbb2225bdb4efe323dbfcc2237e2a4e82c7482dd05402aea6'
endpoint="${HF_ENDPOINT:-https://huggingface.co}"
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$project_root/downloads/omniretarget/$revision"
target="$destination/$archive"
scene="${1:-climb_16}"

if [[ "$endpoint" != https://* ]]; then
    printf 'HF_ENDPOINT must use HTTPS.\n' >&2
    exit 1
fi
if [[ ! "$scene" =~ ^climb_[0-9][0-9]$ ]]; then
    printf 'Expected a scene identifier such as climb_16.\n' >&2
    exit 1
fi
mkdir -p -- "$destination"
if [[ -e "$target" ]]; then
    if ! printf '%s  %s\n' "$checksum" "$target" | sha256sum --check --status; then
        printf 'Existing motion archive has an unexpected checksum; leaving it unchanged.\n' >&2
        exit 1
    fi
else
    temporary="$(mktemp "$destination/.motion-download.XXXXXX")"
    trap 'rm -f -- "$temporary"' EXIT
    curl -4 --fail --location --retry 2 --connect-timeout 10 --max-time 300 \
        --output "$temporary" \
        "${endpoint%/}/datasets/$dataset/resolve/$revision/$archive?download=true"
    printf '%s  %s\n' "$checksum" "$temporary" | sha256sum --check --status
    mv -- "$temporary" "$target"
    trap - EXIT
fi
printf 'Verified OmniRetarget terrain motions: %s\n' "$target"
printf 'Dataset: https://huggingface.co/datasets/%s/tree/%s\n' "$dataset" "$revision"
"$project_root/.venv/bin/python" - "$endpoint" "$dataset" "$revision" "$destination" "$scene" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

endpoint, dataset, revision, directory, scene = sys.argv[1:]

def fetch(url):
    return subprocess.check_output([
        "curl", "-4", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "10", "--max-time", "60", url,
    ])

metadata = json.loads(fetch(f"{endpoint}/api/datasets/{dataset}/revision/{revision}?blobs=true"))
if metadata["sha"] != revision or metadata["id"] != dataset:
    raise ValueError("Dataset identity or revision mismatch")
prefix = f"models/terrain/{scene}/"
entries = [entry for entry in metadata["siblings"] if
    entry["rfilename"] == "README.md" or
    (entry["rfilename"].startswith(prefix) and
     (entry["rfilename"].endswith(".obj") or entry["rfilename"].endswith("multi_boxes_z_scale_1.0.urdf")))]
if not any(entry["rfilename"].endswith(".urdf") for entry in entries):
    raise ValueError(f"No published terrain assets for {scene}")
root = Path(directory).resolve()
for entry in entries:
    name = entry["rfilename"]
    target = (root / name).resolve()
    if not target.is_relative_to(root) or entry["size"] > 2_000_000:
        raise ValueError(f"Unexpected terrain asset: {name}")
    payload = target.read_bytes() if target.exists() else fetch(
        f"{endpoint}/datasets/{dataset}/resolve/{revision}/{quote(name)}"
    )
    digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    if digest != entry["blobId"]:
        raise ValueError(f"Asset checksum mismatch; not overwriting {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(f"Verified terrain asset: {target}")
PY
