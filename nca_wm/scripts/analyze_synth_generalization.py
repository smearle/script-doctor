"""Summarize synthetic-level world-model generalization experiments.

Reads ``heldout_eval*/results.json`` files and writes compact CSV/Markdown
tables for the synthetic-vs-authored question:

  - Does synthetic sokoban training transfer to human Microban levels?
  - Does the pattern repeat in multi-game training?
  - Which games/settings fail, and do they beat the identity baseline?

The script is intentionally post-hoc: it does not train or re-evaluate models.
It lets us turn the existing expensive runs in ``nca_wm/logs`` into a stable
report that can be regenerated after new heldout evals finish.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_RUNS = {
    "single_synth_sokoban_w7": "nca_wm/logs/synth_sokoban_n256_w7h7_seed0",
    "single_synth_sokoban_w8": "nca_wm/logs/synth_sokoban_n256_w8h8_v7_seed0",
    "synth_4games_per_game_size": "nca_wm/logs/synth_4games_pgsize_v7_seed0",
    "synth_scaling6_v7_fallback": "nca_wm/logs/synth_scaling6_pgsize_fallback_v7_seed0",
    "synth_scaling6_k5": "nca_wm/logs/synth_scaling6_K5_tight_v8_seed0",
    "synth_scaling6_multigrid_k5": "nca_wm/logs/synth_scaling6_multigrid_K5_seed0",
    "authored_scaling6": "nca_wm/logs/scaling_6_joint_v1",
}


@dataclass(frozen=True)
class Row:
    run: str
    bucket: str
    game: str
    n_levels: int
    model_step1: float
    model_rollout: float
    identity_step1: float
    identity_rollout: float
    perfect_step1: int
    beats_identity_step1: bool
    eval_dir: str


def _find_results(run_dir: Path, preferred_subdir: str | None) -> Path | None:
    if preferred_subdir:
        path = run_dir / preferred_subdir / "results.json"
        if path.is_file():
            return path
    candidates = sorted(run_dir.glob("heldout_eval*/results.json"))
    if not candidates:
        return None
    full = [p for p in candidates if p.parent.name.endswith("_full")]
    return full[-1] if full else candidates[-1]


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return mean(finite) if finite else float("nan")


def _rows_from_results(label: str, run_dir: Path, path: Path) -> list[Row]:
    with path.open() as f:
        results = json.load(f)
    rows: list[Row] = []
    for bucket in ("heldout", "train_sample"):
        for game, per_level in results.get(bucket, {}).items():
            model_step1 = []
            model_rollout = []
            identity_step1 = []
            identity_rollout = []
            perfect = 0
            for level in per_level.values():
                ar = level["random"]
                s1 = float(ar["model_cell_err_step1"])
                model_step1.append(s1)
                model_rollout.append(float(ar["model_cell_err_mean"]))
                identity_step1.append(float(ar["identity_cell_err_step1"]))
                identity_rollout.append(float(ar["identity_cell_err_mean"]))
                perfect += int(s1 < 1e-9)
            m_s1 = _safe_mean(model_step1)
            i_s1 = _safe_mean(identity_step1)
            rows.append(Row(
                run=label,
                bucket=bucket,
                game=game,
                n_levels=len(model_step1),
                model_step1=m_s1,
                model_rollout=_safe_mean(model_rollout),
                identity_step1=i_s1,
                identity_rollout=_safe_mean(identity_rollout),
                perfect_step1=perfect,
                beats_identity_step1=(m_s1 < i_s1),
                eval_dir=str(path.parent.relative_to(run_dir)),
            ))
    return rows


def _pct(v: float) -> str:
    return "nan" if v != v else f"{100.0 * v:.3f}%"


def _write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run", "bucket", "game", "n_levels",
            "model_step1", "model_rollout",
            "identity_step1", "identity_rollout",
            "perfect_step1", "beats_identity_step1", "eval_dir",
        ])
        for r in rows:
            writer.writerow([
                r.run, r.bucket, r.game, r.n_levels,
                r.model_step1, r.model_rollout,
                r.identity_step1, r.identity_rollout,
                r.perfect_step1, int(r.beats_identity_step1), r.eval_dir,
            ])


def _write_markdown(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synthetic Level Generalization Summary",
        "",
        "Cell error rates are means across evaluated levels. Step-1 uses the",
        "autoregressive evaluator's first predicted transition; identity is the",
        "baseline that predicts the next state equals the current state.",
        "",
        "## Held-Out Human Levels",
        "",
        "| run | game | levels | step-1 | rollout | identity step-1 | perfect | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.bucket != "heldout":
            continue
        verdict = "beats identity" if r.beats_identity_step1 else "fails identity"
        lines.append(
            f"| {r.run} | {r.game} | {r.n_levels} | {_pct(r.model_step1)} | "
            f"{_pct(r.model_rollout)} | {_pct(r.identity_step1)} | "
            f"{r.perfect_step1}/{r.n_levels} | {verdict} |"
        )

    lines.extend([
        "",
        "## Authored Training-Game Controls",
        "",
        "| run | game | levels | step-1 | rollout | identity step-1 | perfect | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in rows:
        if r.bucket != "train_sample":
            continue
        verdict = "beats identity" if r.beats_identity_step1 else "fails identity"
        lines.append(
            f"| {r.run} | {r.game} | {r.n_levels} | {_pct(r.model_step1)} | "
            f"{_pct(r.model_rollout)} | {_pct(r.identity_step1)} | "
            f"{r.perfect_step1}/{r.n_levels} | {verdict} |"
        )

    lines.extend([
        "",
        "## Readout",
        "",
        "- Microban transfer is real: synthetic sokoban at the matched authored grid",
        "  size is perfect on the cached Microban/Microban_I evals.",
        "- The main repeatable failure is grid-size mismatch: the same synthetic",
        "  sokoban recipe at 8x8 stops beating identity on the 6x7 authored",
        "  sokoban control and becomes much worse on Microban.",
        "- Per-game-size and multi-grid synthetic training recover most of the",
        "  multi-game Microban transfer, roughly matching authored scaling_6.",
        "- Residual failures concentrate in games where synthetic data is sparse,",
        "  mixed-size, or dynamics-only: especially nekopuzzle, Zen, and kettle.",
        "",
    ])
    path.write_text("\n".join(lines))


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).name
    return label, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="*",
        default=[f"{k}={v}" for k, v in DEFAULT_RUNS.items()],
        help="Run specs as label=path. Defaults to known local synth/authored runs.",
    )
    ap.add_argument("--preferred_subdir", default=None)
    ap.add_argument("--out_dir", default="nca_wm/figures/synth_generalization")
    args = ap.parse_args()

    rows: list[Row] = []
    missing: list[str] = []
    for spec in args.runs:
        label, run_dir = parse_run_spec(spec)
        result_path = _find_results(run_dir, args.preferred_subdir)
        if result_path is None:
            missing.append(f"{label} ({run_dir})")
            continue
        rows.extend(_rows_from_results(label, run_dir, result_path))

    rows.sort(key=lambda r: (r.bucket != "heldout", r.game, r.run))
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if missing:
        print("missing heldout results:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
