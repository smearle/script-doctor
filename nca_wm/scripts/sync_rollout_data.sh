#!/usr/bin/env bash
# Bidirectional rsync of rollout_data/ between yalda (this box) and 210
# (or any other host). Newer-mtime-wins per file. Skips the merged
# dataset cache (rollout_data/_merged/, large npz blobs that each box
# regenerates on demand).
#
# By default syncs ONLY the per-game caches each side is missing;
# pass `--all` to mirror the whole tree.
#
# Usage:
#   nca_wm/scripts/sync_rollout_data.sh                  # bidirectional, missing-only
#   nca_wm/scripts/sync_rollout_data.sh push             # local -> remote only
#   nca_wm/scripts/sync_rollout_data.sh pull             # remote -> local only
#   REMOTE=210 nca_wm/scripts/sync_rollout_data.sh
#   nca_wm/scripts/sync_rollout_data.sh --all push       # full mirror push
set -uo pipefail

cd "$(dirname "$0")/../.."

REMOTE=${REMOTE:-210}
REMOTE_PATH=${REMOTE_PATH:-script-doctor/rollout_data}
LOCAL_PATH=rollout_data

DIRECTION="bidir"
MODE="missing"
for arg in "$@"; do
    case "$arg" in
        push) DIRECTION="push" ;;
        pull) DIRECTION="pull" ;;
        bidir) DIRECTION="bidir" ;;
        --all) MODE="all" ;;
        *) echo "unknown arg: $arg"; echo "usage: $0 [push|pull|bidir] [--all]"; exit 1 ;;
    esac
done

# rsync flags:
#   -a     archive mode (perms, times, recursive, ...)
#   -u     skip files that are newer on the receiver (newer-mtime-wins)
#   --info=stats2  one-line summary
# Always exclude _merged (per-machine, regenerated on demand).
RSYNC_FLAGS=(-a -u --info=stats2 --exclude='_merged' --exclude='_test*')

if [ "$MODE" = "missing" ]; then
    # --ignore-existing: send only files not present on receiver
    RSYNC_FLAGS+=(--ignore-existing)
fi

push() {
    echo "=== [$(date '+%F %T')] push: $LOCAL_PATH/ -> ${REMOTE}:${REMOTE_PATH}/ ($MODE) ==="
    rsync "${RSYNC_FLAGS[@]}" "$LOCAL_PATH/" "${REMOTE}:${REMOTE_PATH}/" 2>&1 | tail -5
}
pull() {
    echo "=== [$(date '+%F %T')] pull: ${REMOTE}:${REMOTE_PATH}/ -> $LOCAL_PATH/ ($MODE) ==="
    rsync "${RSYNC_FLAGS[@]}" "${REMOTE}:${REMOTE_PATH}/" "$LOCAL_PATH/" 2>&1 | tail -5
}

case "$DIRECTION" in
    push) push ;;
    pull) pull ;;
    bidir)
        # Push first, then pull. With --ignore-existing the receiver only
        # accepts files it doesn't already have, so this is union-like.
        push
        pull
        ;;
esac

echo "=== [$(date '+%F %T')] done ==="
