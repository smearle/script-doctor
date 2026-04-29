#!/usr/bin/env bash
# When overnight phase 1b (constellationz on GPU 0) exits, relaunch the
# overnight_growth.sh launcher. This causes the new launcher to re-read
# the script file and pick up edits made mid-overnight (max_transitions
# cap, phase 1c singletons, scaling_6 @ hid=256).
#
# IMPORTANT: uses pgrep -f with a POSIX extended regex (needs -E or we
# match literally "*").

set -u

REPO=/home/jupyter-smearle/script-doctor

# Wait for the specific constellationz training process to exit. Use the
# full cmdline substring to identify it unambiguously.
while pgrep -f 'nca_wm/train.py --games global_constellationz --conditional --balanced_sampling --axis_pool --axis_cummax --global_pool' >/dev/null; do
    sleep 60
done

echo "[restart] phase 1b constellationz finished, relaunching overnight_growth.sh"
# Kill any still-running old launcher (should be none, since launcher waits
# for its child; but be defensive).
pkill -f 'nca_wm/scripts/run_overnight_growth.sh' 2>/dev/null
sleep 2

cd "$REPO"
nohup bash nca_wm/scripts/run_overnight_growth.sh > "$REPO/nca_wm/sweep_launch_logs/overnight_growth_master.log" 2>&1 &
echo "[restart] new launcher pid=$!"
