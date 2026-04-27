#!/usr/bin/env python3
"""Overnight dashboard — prints a one-shot summary of every active NCA WM run.

Scans all logs under nca_wm/logs/, reports the latest step, loss, and
change_err for each, plus whether the process is still training or done.
Group by sweep_name for readability.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent

LOGS = _REPO / "nca_wm" / "logs"


def last_step_from_curves(d: Path):
    files = sorted(glob.glob(str(d / "curves_step*.npz")))
    if not files:
        return None
    import numpy as np
    def step(p):
        m = re.search(r"curves_step(\d+)", p)
        return int(m.group(1)) if m else 0
    latest = max(files, key=step)
    z = np.load(latest)
    total_steps = step(latest)
    if "losses" in z and len(z["losses"]):
        tail = min(200, len(z["losses"]))
        loss = float(z["losses"][-tail:].mean())
        cerr = float(1.0 - z["change_accs"][-tail:].mean())
        return total_steps, loss, cerr
    return total_steps, None, None


def is_process_alive(run_dir: Path) -> bool:
    # Grep ps for a running python that has this save_dir in its cmdline
    out = subprocess.run(
        ["ps", "-ef"], capture_output=True, text=True
    ).stdout
    want = os.path.basename(run_dir)
    for line in out.splitlines():
        if "python" in line and want in line:
            return True
    return False


def main():
    runs = []
    for cfg_path in sorted(LOGS.glob("*/config.json")):
        d = cfg_path.parent
        with open(cfg_path) as f:
            cfg = json.load(f)
        runs.append((d, cfg))

    # Group by sweep_name
    by_sweep: dict[str, list] = {}
    for d, cfg in runs:
        by_sweep.setdefault(cfg.get("sweep_name") or "(none)", []).append((d, cfg))

    for sweep in sorted(by_sweep):
        items = by_sweep[sweep]
        print(f"\n=== sweep={sweep!r}  ({len(items)} runs) ===")
        rows = []
        for d, cfg in items:
            steps_info = last_step_from_curves(d)
            alive = is_process_alive(d)
            pools = "".join([
                "a" if cfg.get("axis_pool") else "-",
                "c" if cfg.get("axis_cummax") else "-",
                "g" if cfg.get("global_pool") else "-",
            ])
            bal = "b" if cfg.get("balanced_sampling") else "-"
            spr = "s" if cfg.get("sprite_loss_weight", 0) > 0 else "-"
            gc = cfg.get("grad_clip", 0.0) or 0.0
            games = cfg.get("games") or cfg.get("game") or "?"
            hid = cfg.get("n_hid", "?")
            nca = cfg.get("n_nca_steps", "?")
            tag = f"{games:<22} hid={hid:<4} nca={nca:<3} [{pools}{bal}{spr}] gc={gc:<3}"
            if steps_info is None:
                rows.append((tag, "(no curves)", alive))
            else:
                st, loss, cerr = steps_info
                mark = "LIVE" if alive else "done"
                rows.append((tag,
                             f"step={st:>7,}  loss={loss:.2e}  cerr={cerr:.2e}  {mark}",
                             alive))
        # Sort: live first, then by worst cerr
        rows.sort(key=lambda r: (-int(r[2]), r[1]))
        for tag, summary, _ in rows:
            print(f"  {tag}  {summary}")


if __name__ == "__main__":
    main()
