#!/usr/bin/env bash
# Dispatch a training command to a GPU slot across Yalda24 + ssh 210.
#
# Usage:
#   launch_slot.sh <slot> -- <command and args...>
#   slot: 0 | 1 | 210
#     0, 1: local Yalda24 GPU index
#     210:  hemeraTwo's only GPU (CUDA_VISIBLE_DEVICES=0 there)
#
# Commit-first: refuses if either box's working tree is dirty, so both
# boxes run the same git sha and results are comparable. The remote
# checkout is sha-pinned (detached HEAD) for the same reason.
#
# This script execs the command in the foreground. To run several slots
# in parallel, background each launcher invocation:
#   ./launch_slot.sh 0   -- python -m nca_wm.train ... &
#   ./launch_slot.sh 1   -- python -m nca_wm.train ... &
#   ./launch_slot.sh 210 -- python -m nca_wm.train ... &
#   wait
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  echo "Usage: $0 <slot> -- <command and args...>" >&2
  echo "  slot: 0 | 1 | 210" >&2
  exit 2
fi

slot="$1"
shift 2

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [ -n "$(git status --porcelain)" ]; then
  echo "launch_slot: local working tree is dirty. Commit/stash first." >&2
  git status -s >&2
  exit 3
fi
branch="$(git rev-parse --abbrev-ref HEAD)"
sha="$(git rev-parse HEAD)"

case "$slot" in
  0|1)
    export CUDA_VISIBLE_DEVICES="$slot"
    export PYTHONUNBUFFERED=1
    echo "launch_slot: local GPU $slot, sha=$sha" >&2
    exec "$@"
    ;;
  210)
    remote_dirty="$(ssh 210 'cd ~/script-doctor && git status --porcelain')"
    if [ -n "$remote_dirty" ]; then
      echo "launch_slot: 210 working tree is dirty:" >&2
      echo "$remote_dirty" >&2
      exit 4
    fi
    git push --quiet origin "$branch"

    quoted=""
    for a in "$@"; do
      quoted+=" $(printf %q "$a")"
    done

    echo "launch_slot: ssh 210 GPU 0, sha=$sha (branch=$branch)" >&2
    ssh 210 "cd ~/script-doctor && \
      git fetch --quiet origin && \
      git checkout --quiet '$sha' && \
      CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1$quoted"
    ;;
  *)
    echo "launch_slot: unknown slot '$slot' (use 0 | 1 | 210)" >&2
    exit 2
    ;;
esac
