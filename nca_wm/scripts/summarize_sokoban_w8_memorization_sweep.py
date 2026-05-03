"""Summarize the 8x8 synthetic sokoban memorization sweep.

Expected run dirs are named:

    sokoban_w8_n{synthetic_levels}_nca{n_nca_steps}_seed0

Each should contain ``heldout_eval_dual_control/results.json`` from
``run_sokoban_w8_memorization_sweep.sh``.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


RUN_RE = re.compile(r"sokoban_w8_n(?P<n_levels>\d+)_nca(?P<nca>\d+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class Row:
    n_levels: int
    nca_steps: int
    seed: int
    bucket: str
    game: str
    model_step1: float
    model_rollout: float
    identity_step1: float
    perfect: int
    total: int

    @property
    def beats_identity(self) -> bool:
        return self.model_step1 < self.identity_step1


def _avg(xs: list[float]) -> float:
    ys = [x for x in xs if x == x]
    return mean(ys) if ys else float("nan")


def _pct(x: float) -> str:
    return "nan" if x != x else f"{100.0 * x:.3f}%"


def _load_rows(run_dir: Path) -> list[Row]:
    m = RUN_RE.match(run_dir.name)
    if not m:
        return []
    path = run_dir / "heldout_eval_dual_control" / "results.json"
    if not path.is_file():
        return []
    with path.open() as f:
        results = json.load(f)

    rows: list[Row] = []
    for bucket in ("heldout", "train_sample"):
        for game, per_level in results.get(bucket, {}).items():
            step1 = []
            rollout = []
            ident = []
            perfect = 0
            for level in per_level.values():
                ar = level["random"]
                s1 = float(ar["model_cell_err_step1"])
                step1.append(s1)
                rollout.append(float(ar["model_cell_err_mean"]))
                ident.append(float(ar["identity_cell_err_step1"]))
                perfect += int(s1 < 1e-9)
            rows.append(Row(
                n_levels=int(m.group("n_levels")),
                nca_steps=int(m.group("nca")),
                seed=int(m.group("seed")),
                bucket=bucket,
                game=game,
                model_step1=_avg(step1),
                model_rollout=_avg(rollout),
                identity_step1=_avg(ident),
                perfect=perfect,
                total=len(step1),
            ))
    return rows


def _write_markdown(rows: list[Row], path: Path) -> None:
    lines = [
        "# Sokoban 8x8 Synthetic Memorization Sweep",
        "",
        "Question: does the 8x8 synthetic sokoban failure go away with more",
        "synthetic levels or with a shallower NCA?",
        "",
        "Verdict rule of thumb: the authored `sokoban_basic` control should beat",
        "the identity baseline and ideally approach the matched-size 7x7 result",
        "(0% step-1). Microban/Microban_I should also improve, but the 6x7",
        "authored control is the sharper memorization test.",
        "",
    ]

    for bucket, title in (
        ("heldout", "Native Authored Controls And Human Heldouts"),
        ("train_sample", "Padded Training-Frame Controls"),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| synth levels | nca steps | game | step-1 | rollout | identity step-1 | perfect | verdict |",
            "|---:|---:|---|---:|---:|---:|---:|---|",
        ])
        for r in sorted(
            [x for x in rows if x.bucket == bucket],
            key=lambda x: (x.game, x.n_levels, x.nca_steps, x.seed),
        ):
            verdict = "beats identity" if r.beats_identity else "fails identity"
            lines.append(
                f"| {r.n_levels} | {r.nca_steps} | {r.game} | "
                f"{_pct(r.model_step1)} | {_pct(r.model_rollout)} | "
                f"{_pct(r.identity_step1)} | {r.perfect}/{r.total} | {verdict} |"
            )
        lines.append("")

    if not rows:
        lines.append("No completed heldout evals found yet.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="nca_wm/logs_sokoban_w8_mem_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[Row] = []
    for run_dir in sorted(root.glob("sokoban_w8_n*_nca*_seed*")):
        if run_dir.is_dir():
            rows.extend(_load_rows(run_dir))

    out = Path(args.out) if args.out else root / "summary.md"
    _write_markdown(rows, out)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
