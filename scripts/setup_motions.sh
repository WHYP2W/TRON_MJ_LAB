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

if [[ "$endpoint" != https://* ]]; then
    printf 'HF_ENDPOINT must use HTTPS.\n' >&2
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
