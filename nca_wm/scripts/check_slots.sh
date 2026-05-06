#!/usr/bin/env bash
# Quick view of GPU availability across Yalda24 (local) and ssh 210.
# Use before dispatching with launch_slot.sh.
set -uo pipefail

print_section() {
  printf '\n=== %s ===\n' "$1"
}

print_section "Yalda24 (local, 2x RTX 4090)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null
echo "--- compute apps ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
  --format=csv,noheader 2>/dev/null
echo "--- gdrtodd processes (avoid GPU 0 if present) ---"
ps -eo user,pid,cmd 2>/dev/null | awk '$1=="gdrtodd"' | head -5

print_section "ssh 210 (hemeraTwo, 1x RTX 4090)"
ssh -o ConnectTimeout=5 210 '
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null
  echo "--- compute apps ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null
  echo "--- gdrtodd processes ---"
  ps -eo user,pid,cmd 2>/dev/null | awk "\$1==\"gdrtodd\"" | head -5
' 2>/dev/null || echo "ssh 210 unreachable"
