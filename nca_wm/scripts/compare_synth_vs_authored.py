"""Compare synthetic-trained vs authored-trained NCA world models on Microban.

Reads the per-game heldout-eval results.json from two runs and prints a
side-by-side table.

Usage:
    python nca_wm/scripts/compare_synth_vs_authored.py \\
        --synth nca_wm/logs/synth_sokoban_n256_w7h7_seed0 \\
        --authored nca_wm/logs/authored_sokoban_basic_seed0
"""
import argparse
import json
import os
from pathlib import Path


def _load_results(save_dir: str) -> dict:
    path = Path(save_dir) / "heldout_eval" / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"No heldout results at {path} -- run heldout_eval first")
    with open(path) as f:
        return json.load(f)


def _gather(results: dict, bucket: str) -> dict:
    """Returns {game: [(level_i, model_step1, model_mean, identity_step1, identity_mean), ...]}."""
    out = {}
    for game, per_level in results.get(bucket, {}).items():
        rows = []
        for li_str, mode in per_level.items():
            li = int(li_str)
            rng = mode.get("random", {})
            rows.append((
                li,
                rng.get("model_cell_err_step1", float("nan")),
                rng.get("model_cell_err_mean", float("nan")),
                rng.get("identity_cell_err_step1", float("nan")),
                rng.get("identity_cell_err_mean", float("nan")),
            ))
        out[game] = sorted(rows)
    return out


def _fmt(v: float) -> str:
    if v != v:  # nan
        return "  nan"
    return f"{100*v:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="save_dir of synthetic-trained run")
    ap.add_argument("--authored", required=True, help="save_dir of authored-trained run")
    args = ap.parse_args()

    synth = _load_results(args.synth)
    auth = _load_results(args.authored)

    print(f"\nSynthetic: {args.synth}")
    print(f"Authored:  {args.authored}")
    print()

    for bucket in ("heldout", "train_sample"):
        s = _gather(synth, bucket)
        a = _gather(auth, bucket)
        all_games = sorted(set(s.keys()) | set(a.keys()))
        if not all_games:
            continue
        print(f"=== {bucket} ===")
        # Print per-level table
        print(f"{'game':24s} {'L':>3s}  "
              f"{'synth_step1':>12s} {'auth_step1':>11s} "
              f"{'synth_mean':>11s} {'auth_mean':>10s}  "
              f"{'identity_step1':>14s}")
        for game in all_games:
            srows = {li: row for row in s.get(game, []) for li in [row[0]]}
            arows = {li: row for row in a.get(game, []) for li in [row[0]]}
            levels = sorted(set(srows.keys()) | set(arows.keys()))
            for li in levels:
                sr = srows.get(li)
                ar = arows.get(li)
                synth_s1 = _fmt(sr[1]) if sr else "  ---"
                auth_s1 = _fmt(ar[1]) if ar else "  ---"
                synth_m = _fmt(sr[2]) if sr else "  ---"
                auth_m = _fmt(ar[2]) if ar else "  ---"
                # Identity should be the same regardless of which model
                ident_s1 = _fmt(sr[3]) if sr else (_fmt(ar[3]) if ar else "  ---")
                print(f"{game:24s} {li:>3d}  "
                      f"{synth_s1:>12s} {auth_s1:>11s} "
                      f"{synth_m:>11s} {auth_m:>10s}  "
                      f"{ident_s1:>14s}")
        # Game-level mean
        print()
        print(f"{'game':24s} {'':3s}  "
              f"{'synth_mean(s1)':>14s} {'auth_mean(s1)':>13s}  "
              f"{'synth_mean':>11s} {'auth_mean':>10s}")
        for game in all_games:
            srs = s.get(game, [])
            ars = a.get(game, [])
            if not srs and not ars:
                continue
            def _mean(rows, idx):
                vals = [r[idx] for r in rows if r[idx] == r[idx]]
                return sum(vals) / len(vals) if vals else float("nan")
            ss1 = _mean(srs, 1); as1 = _mean(ars, 1)
            sm = _mean(srs, 2); am = _mean(ars, 2)
            print(f"{game:24s} {'avg':>3s}  "
                  f"{_fmt(ss1):>14s} {_fmt(as1):>13s}  "
                  f"{_fmt(sm):>11s} {_fmt(am):>10s}")
        print()


if __name__ == "__main__":
    main()
