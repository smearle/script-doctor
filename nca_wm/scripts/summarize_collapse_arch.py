"""Summarize the Collapse architecture-ablation runs into one table.

Reads each `nca_wm/logs_collapse_arch/collapse_L0_*` save dir and reports:
- best loss (from train_meta.json or last step in log)
- end-of-train change_err
- whether training appeared stable or diverged

Used to populate the depth/skip-connection table in
nca_wm/ARCHITECTURE_REPORT.md.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

LOGDIR = Path(__file__).resolve().parents[1] / "logs_collapse_arch"
STEP_RE = re.compile(
    r"step\s+([\d,]+)/[\d,]+.*?loss=([\d.eE+-]+).*?"
    r"err=([\d.eE+-]+).*?change_err=([\d.eE+-]+)"
)


def parse_log(log_path: Path) -> dict:
    """Pull last reported step's metrics + max+min observed loss."""
    info: dict = {"path": log_path.name}
    if not log_path.is_file():
        info["status"] = "MISSING"
        return info
    last_match = None
    losses: list[float] = []
    diverged = False
    text = log_path.read_text(errors="replace")
    if "RESOURCE_EXHAUSTED" in text or "OutOfMemoryError" in text:
        info["status"] = "OOM"
        return info
    for line in text.splitlines():
        if "Traceback" in line or "RuntimeError" in line:
            info["status"] = info.get("status", "ERROR")
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1).replace(",", ""))
        loss = float(m.group(2))
        err = float(m.group(3))
        cerr = float(m.group(4))
        losses.append(loss)
        last_match = (step, loss, err, cerr)
        if loss > 1e3:
            diverged = True
    if last_match is None:
        info["status"] = info.get("status", "NO_STEPS")
        return info
    step, loss, err, cerr = last_match
    info["last_step"] = step
    info["last_loss"] = loss
    info["last_err"] = err
    info["last_change_err"] = cerr
    info["min_loss"] = min(losses) if losses else None
    info["max_loss"] = max(losses) if losses else None
    info["status"] = "DIVERGED" if diverged else (info.get("status") or "OK")
    return info


def main() -> None:
    rows = []
    for log in sorted(LOGDIR.glob("collapse_L0_*.out")):
        rows.append(parse_log(log))
    if not rows:
        print(f"no logs in {LOGDIR}", file=sys.stderr)
        sys.exit(1)
    fmt = "{:<48}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("run", "status", "last_step", "min_loss", "last_loss",
                     "last_err", "last_chg"))
    for r in rows:
        if "last_step" not in r:
            print(fmt.format(r["path"], r.get("status", "?"), "-", "-", "-", "-", "-"))
            continue
        print(fmt.format(
            r["path"],
            r["status"],
            f"{r['last_step']:,}",
            f"{r['min_loss']:.3e}",
            f"{r['last_loss']:.3e}",
            f"{r['last_err']:.3e}",
            f"{r['last_change_err']:.3f}",
        ))


if __name__ == "__main__":
    main()
