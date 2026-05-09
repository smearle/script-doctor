#!/usr/bin/env bash
# Reclaim disk by deleting stale merged-dataset caches at
# rollout_data/_merged/dataset_*.npz. These are per-machine and
# regenerated on demand by train.py, so deleting them only costs
# one-time recompute on the next training run that uses each cache key.
#
# Safe-by-default: only deletes files older than today AND larger than
# 100 MB (so today's in-flight cache files and tiny precollect outputs
# stay untouched). Any in-flight training run already has its merged
# cache loaded into RAM at startup, so deleting the on-disk file does
# not disrupt the run.
#
# Usage:
#   nca_wm/scripts/cleanup_merged_cache.sh                    # dry-run
#   nca_wm/scripts/cleanup_merged_cache.sh --apply            # actually delete
#   AGE_DAYS=3 SIZE_MIN=500M nca_wm/scripts/cleanup_merged_cache.sh --apply
set -uo pipefail

cd "$(dirname "$0")/../.."

ROOT=${ROOT:-rollout_data/_merged}
SIZE_MIN=${SIZE_MIN:-100M}
AGE_DAYS=${AGE_DAYS:-1}   # delete files older than this many days
APPLY=0
for arg in "$@"; do
    case "$arg" in
        --apply) APPLY=1 ;;
        --help|-h)
            echo "usage: $0 [--apply]"
            echo "env vars: ROOT (default $ROOT), SIZE_MIN (default $SIZE_MIN),"
            echo "          AGE_DAYS (default $AGE_DAYS, files older than this go)"
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

if [ ! -d "$ROOT" ]; then
    echo "no $ROOT directory; nothing to do"
    exit 0
fi

echo "scanning $ROOT for files older than $AGE_DAYS day(s) AND >= $SIZE_MIN ..."

# Build the candidate list once
mapfile -t TARGETS < <(find "$ROOT" -name "dataset_*.npz" \
    -mtime "+$AGE_DAYS" -size "+$SIZE_MIN" -print 2>/dev/null)

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "nothing matches; disk usage:"
    du -sh "$ROOT" 2>/dev/null
    exit 0
fi

# Total size to be freed
total_bytes=$(printf '%s\n' "${TARGETS[@]}" | xargs -I{} stat -c '%s' {} 2>/dev/null | awk '{s+=$1} END {print s}')
echo "candidates: ${#TARGETS[@]} files, $((total_bytes / 1024 / 1024 / 1024)) GB total"

# Show oldest + newest sample so we can sanity-check
printf '%s\n' "${TARGETS[@]}" | xargs -I{} stat -c '%y {}' {} 2>/dev/null | sort | head -3
echo "..."
printf '%s\n' "${TARGETS[@]}" | xargs -I{} stat -c '%y {}' {} 2>/dev/null | sort | tail -3

if [ "$APPLY" = 0 ]; then
    echo
    echo "(dry-run; pass --apply to actually delete)"
    exit 0
fi

# Sanity: refuse to delete a file that's currently held open by any
# python process (defensive — train.py loads into RAM at startup so
# this should never trip in practice, but cheap to check).
echo
echo "checking for open file descriptors..."
held=0
for pid in $(pgrep -f "nca_wm" || true); do
    open_files=$(ls -l /proc/$pid/fd 2>/dev/null | grep -F "$ROOT" | awk '{print $NF}' || true)
    if [ -n "$open_files" ]; then
        echo "  PID $pid holds:"
        echo "$open_files" | sed 's/^/    /'
        held=$((held + 1))
    fi
done
if [ "$held" -gt 0 ]; then
    echo "ABORT: $held process(es) hold $ROOT files open"
    exit 1
fi

echo "deleting ${#TARGETS[@]} files..."
printf '%s\n' "${TARGETS[@]}" | xargs -r rm -f
echo "done. new disk state:"
df -h . | head -2
du -sh "$ROOT"
