"""Run inverse-fit few-shot diagnostics across games and transition budgets.

This is a lightweight experiment harness around `nca_wm.inverse_fit_slot`.
It freezes a trained rule-attention NCA checkpoint, optimizes only the game
slot matrix for each target game, and summarizes whether a small number of
observed transitions is enough to improve the frozen model's dynamics.

Example:
    .venv/bin/python -m nca_wm.scripts.run_inverse_fit_grid \
        --load nca_wm/logs/20260430-001831_vqdec_20g_rule_attn_allpool_200k_vq1024 \
        --games Microban,Bouncers \
        --shot_counts 10,50 \
        --n_steps 200 --skip_done
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _load_games(args) -> list[str]:
    names: list[str] = []
    if args.heldout_file:
        with open(args.heldout_file) as f:
            spec = json.load(f)
        if isinstance(spec, dict) and "heldout" in spec:
            names.extend([str(x["name"]) for x in spec["heldout"]])
        elif isinstance(spec, list):
            names.extend([str(x["name"] if isinstance(x, dict) else x)
                          for x in spec])
        else:
            raise ValueError(
                "heldout_file must be a list or a dict with a 'heldout' list"
            )
    if args.games:
        # PuzzleScript names can contain commas. For explicit lists, accept
        # semicolon as the unambiguous separator and comma for simple names.
        sep = ";" if ";" in args.games else ","
        names.extend([x.strip() for x in args.games.split(sep) if x.strip()])
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            out.append(name)
            seen.add(name)
    if args.max_games is not None:
        out = out[:args.max_games]
    if not out:
        raise ValueError("provide --games or --heldout_file")
    return out


def _load_summary(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _write_summary(root: Path, rows: list[dict], failures: list[dict]):
    root.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["game"], r["shots"]))
    with (root / "summary.json").open("w") as f:
        json.dump({"rows": rows, "failures": failures}, f, indent=2)

    with (root / "summary.md").open("w") as f:
        f.write("# Inverse-Fit Few-Shot Summary\n\n")
        f.write("| game | shots | init | pre loss | post loss | "
                "pre cell err | post cell err | identity cell err | "
                "1-NN train | status |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            f.write(
                f"| `{r['game']}` | {r['shots']} | `{r['init']}` | "
                f"{r['pre_loss']:.4g} | {r['post_loss']:.4g} | "
                f"{r['pre_cell_err']:.4f} | {r['post_cell_err']:.4f} | "
                f"{r['identity_cell_err']:.4f} | "
                f"`{r.get('nn_train_name', '')}` | {r['status']} |\n"
            )
        if failures:
            f.write("\n## Failures\n\n")
            f.write("| game | shots | reason |\n")
            f.write("|---|---:|---|\n")
            for fail in failures:
                reason = str(fail.get("reason", "")).replace("\n", " ")
                f.write(f"| `{fail['game']}` | {fail['shots']} | "
                        f"{reason[:300]} |\n")


def _discover_existing_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("shots_*_*/*/summary.json")):
        parent = summary_path.parent
        shot_dir = parent.parent.name
        parts = shot_dir.split("_")
        if len(parts) < 3 or parts[0] != "shots":
            continue
        try:
            shots = int(parts[1])
        except ValueError:
            continue
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        rows.append(_row_from_summary(parent.name, shots, summary))
    return rows


def _merge_rows(rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (row["game"], int(row["shots"]), str(row.get("init", "")))
        merged[key] = row
    return list(merged.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True,
                    help="Checkpoint run directory.")
    ap.add_argument("--games", default=None,
                    help="Comma-separated or semicolon-separated target games.")
    ap.add_argument("--heldout_file", default=None,
                    help="Optional heldout manifest JSON.")
    ap.add_argument("--max_games", type=int, default=None)
    ap.add_argument("--shot_counts", default="10,50,100,500")
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--init", default="knn",
                    choices=["knn", "mean", "random"])
    ap.add_argument("--out_subdir", default="inverse_fit_grid")
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_dir = Path(args.load)
    out_root = load_dir / args.out_subdir
    games = _load_games(args)
    shot_counts = [int(x) for x in args.shot_counts.split(",") if x.strip()]

    print(f"Checkpoint: {load_dir}")
    print(f"Targets: {len(games)} games")
    print(f"Shot counts: {shot_counts}")
    print(f"Output: {out_root}")

    rows: list[dict] = _discover_existing_rows(out_root)
    failures: list[dict] = []
    if rows:
        print(f"Discovered {len(rows)} existing completed cells")
    t_start = time.time()

    for game_i, game in enumerate(games, 1):
        for shot_i, shots in enumerate(shot_counts, 1):
            subdir = f"{args.out_subdir}/shots_{shots}_{args.init}"
            summary_path = (
                load_dir / subdir / game / "summary.json"
            )
            if args.skip_done and summary_path.exists():
                summary = _load_summary(summary_path)
                if summary is not None:
                    print(f"[{game_i}/{len(games)}:{shot_i}/{len(shot_counts)}] "
                          f"{game} shots={shots}: cached")
                    rows.append(_row_from_summary(game, shots, summary))
                    rows = _merge_rows(rows)
                    _write_summary(out_root, rows, failures)
                    continue

            cmd = [
                sys.executable, "-m", "nca_wm.inverse_fit_slot",
                "--load", str(load_dir),
                "--target_game", game,
                "--n_max_transitions", str(shots),
                "--n_steps", str(args.n_steps),
                "--lr", str(args.lr),
                "--init", args.init,
                "--out_subdir", subdir,
                "--seed", str(args.seed),
            ]
            print(f"[{game_i}/{len(games)}:{shot_i}/{len(shot_counts)}] "
                  f"{game} shots={shots}: running", flush=True)
            t0 = time.time()
            res = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
            )
            dt = time.time() - t0
            if res.returncode != 0:
                stderr_tail = "\n".join(res.stderr.splitlines()[-12:])
                print(f"  FAIL rc={res.returncode} ({dt:.1f}s)\n"
                      f"{stderr_tail}")
                failures.append({
                    "game": game,
                    "shots": shots,
                    "returncode": res.returncode,
                    "reason": stderr_tail,
                })
                rows = _merge_rows(rows)
                _write_summary(out_root, rows, failures)
                continue

            summary = _load_summary(summary_path)
            if summary is None:
                failures.append({
                    "game": game,
                    "shots": shots,
                    "returncode": 0,
                    "reason": f"missing summary at {summary_path}",
                })
                rows = _merge_rows(rows)
                _write_summary(out_root, rows, failures)
                continue
            row = _row_from_summary(game, shots, summary)
            rows.append(row)
            rows = _merge_rows(rows)
            print(f"  OK ({dt:.1f}s): loss {row['pre_loss']:.4g} -> "
                  f"{row['post_loss']:.4g}, cell_err "
                  f"{row['pre_cell_err']:.4f} -> {row['post_cell_err']:.4f}")
            _write_summary(out_root, rows, failures)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f}m")
    print(f"Summary: {out_root / 'summary.md'}")
    if failures:
        print(f"Failures: {len(failures)}")


def _row_from_summary(game: str, shots: int, summary: dict) -> dict:
    pre = summary.get("pre_fit_metrics", {})
    post = summary.get("post_fit_metrics", {})
    pre_loss = float(pre.get("loss", float("nan")))
    post_loss = float(post.get("loss", summary.get("final_loss", float("nan"))))
    pre_cell = float(pre.get("cell_err", float("nan")))
    post_cell = float(post.get("cell_err", float("nan")))
    identity_cell = float(post.get(
        "identity_cell_err",
        pre.get("identity_cell_err", float("nan")),
    ))
    status = "improved" if post_loss < pre_loss else "no_improve"
    return {
        "game": game,
        "shots": shots,
        "init": summary.get("init", ""),
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "pre_cell_err": pre_cell,
        "post_cell_err": post_cell,
        "identity_cell_err": identity_cell,
        "nn_train_name": summary.get("nn_train_name", ""),
        "nn_cos_dist": summary.get("nn_cos_dist"),
        "status": status,
        "summary_file": summary.get("summary_file", ""),
    }


if __name__ == "__main__":
    main()
