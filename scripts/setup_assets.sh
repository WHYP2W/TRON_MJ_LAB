#!/usr/bin/env bash
set -euo pipefail

repository='https://github.com/limxdynamics/tron2-robot-description.git'
revision='f547f5bc949f2a4c98e076e61cf6d3ca73d179a0'
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$project_root/assets/robot-description"

if [[ -e "$destination" || -L "$destination" ]]; then
    if [[ ! -e "$destination/.git" ]]; then
        printf 'Existing destination is not a Git checkout: %s\n' "$destination" >&2
        exit 1
    fi
    current_revision="$(git --no-optional-locks -C "$destination" rev-parse HEAD)"
    if [[ "$current_revision" != "$revision" ]]; then
        printf 'Expected asset revision %s. Existing files were not changed.\n' "$revision" >&2
        exit 1
    fi
    changes="$(git --no-optional-locks -C "$destination" status --porcelain --untracked-files=all)"
    if [[ -n "$changes" ]]; then
        printf 'Asset checkout has local changes; refusing to overwrite them.\n' >&2
        exit 1
    fi
else
    mkdir -p -- "$(dirname -- "$destination")"
    git -c core.autocrlf=false clone --depth 1 --filter=blob:none --sparse "$repository" "$destination"
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach "$revision"
fi

git -C "$destination" sparse-checkout set tron2a/SFYG_TRON2A/xml tron2a/SFYG_TRON2A/meshes

model="$destination/tron2a/SFYG_TRON2A/xml/robot.xml"
mesh="$destination/tron2a/SFYG_TRON2A/meshes/base_Link.STL"
if [[ ! -f "$model" || ! -f "$mesh" ]]; then
    printf 'Missing SFYG XML or meshes in %s\n' "$destination" >&2
    exit 1
fi

printf 'SFYG assets verified at %s\n%s\n' "$revision" "$destination"
