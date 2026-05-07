#!/usr/bin/env python3
r"""Collate matched cond-vs-uncond runs into paper-ready CSV + LaTeX.

Reads each listed run dir's config.json + eval_multigame.npz +
heldout_v4_n30/results.json, computes the headline metrics, and writes
both a CSV (for inspection) and a LaTeX fragment (for \input into the
paper). Re-run after a new heldout finishes — paper auto-picks-up the
\input.

Output: nca_wm/paper/figures/cond_vs_uncond_match/{summary.csv,
match_table.tex, per_game_step1.csv}

Usage:
    python nca_wm/scripts/collate_match_table.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "cond_vs_uncond_match"

# Order matters: rows in the table follow this order. The collator
# silently skips any run dir whose results.json or eval_multigame.npz
# is missing, so adding a row before it lands is fine — re-run after
# the experiment finishes and the row populates.
RUNS = [
    {
        "run_dir": "multi_scaling_14_uncond_match_s0",
        "label": "Unconditional, Train-14",
        "kind": "uncond",
    },
    {
        "run_dir": "multi_scaling_14_cond_match_s0",
        "label": "Rule-conditional, Train-14",
        "kind": "cond",
    },
    {
        "run_dir": "multi_scaling_gallery_v2_uncond_match_s0",
        "label": "Unconditional, Train-59",
        "kind": "uncond",
    },
    {
        "run_dir": "multi_scaling_gallery_v2_cond_match_s0",
        "label": "Rule-conditional, Train-59",
        "kind": "cond",
    },
    {
        "run_dir": "multi_scaling_gallery_v4_cond_match_s0",
        "label": "Rule-conditional, Train-199",
        "kind": "cond",
    },
    {
        "run_dir": "multi_scaling_gallery_v4_uncond_match_s0",
        "label": "Unconditional, Train-199",
        "kind": "uncond",
    },
]


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"missing config.json: {p}")
    return json.loads(p.read_text())


def _count_params(cfg: dict) -> int | None:
    """Best-effort param count: prefer explicit field, else infer from
    pickle on disk. Returns None if neither works."""
    if "n_params" in cfg:
        return int(cfg["n_params"])
    return None


def _aggregate_indist(eval_npz_path: Path) -> dict:
    """Mean per-game cell-error over random + bfs rollouts, in-distribution."""
    if not eval_npz_path.exists():
        return {}
    d = np.load(eval_npz_path, allow_pickle=True)
    # Keys look like "<game>_L<i>_<algo>_cell_error_rate" with per-step arrays.
    per_game_random: dict[str, list[float]] = {}
    per_game_bfs: dict[str, list[float]] = {}
    for k in d.files:
        if not k.endswith("_cell_error_rate"):
            continue
        # game_<idx>_<algo>_cell_error_rate ; tf variants end with _tf
        # Skip teacher-forced for the AR rollout summary.
        if k.endswith("_tf_cell_error_rate"):
            continue
        # split by "_L<digit>_"
        parts = k.split("_")
        # find the index of the level token (matches L\d+)
        level_idx = None
        for i, tok in enumerate(parts):
            if tok.startswith("L") and tok[1:].isdigit():
                level_idx = i
                break
        if level_idx is None:
            continue
        game = "_".join(parts[:level_idx])
        algo = parts[level_idx + 1]  # random / bfs / astar
        arr = np.asarray(d[k], dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        mean_err = float(arr.mean())
        if algo == "random":
            per_game_random.setdefault(game, []).append(mean_err)
        elif algo == "bfs":
            per_game_bfs.setdefault(game, []).append(mean_err)
    def _summary(per_game: dict[str, list[float]]) -> dict:
        if not per_game:
            return {"n_games": 0, "mean": float("nan"), "median": float("nan")}
        per_game_means = [float(np.mean(v)) for v in per_game.values()]
        return {
            "n_games": len(per_game),
            "mean": float(np.mean(per_game_means)),
            "median": float(np.median(per_game_means)),
        }
    return {
        "random": _summary(per_game_random),
        "bfs": _summary(per_game_bfs),
    }


def _aggregate_heldout(results_json_path: Path) -> dict:
    """Mean step-1 + AR-30 cell-error vs identity baseline, plus per-game wins.

    Computes per-game first (averaging across that game's levels and
    episodes), then averages across games. Symmetric across runs because
    the same heldout list is used.
    """
    if not results_json_path.exists():
        return {}
    d = json.loads(results_json_path.read_text())
    ho = d.get("heldout", {})
    per_game_step1_m: dict[str, list[float]] = {}
    per_game_step1_i: dict[str, list[float]] = {}
    per_game_ar30_m: dict[str, list[float]] = {}
    per_game_ar30_i: dict[str, list[float]] = {}
    for game, levels in ho.items():
        for li, kinds in levels.items():
            rec = kinds.get("random") or {}
            mp = rec.get("model_cell_err_per_step") or []
            ip = rec.get("identity_cell_err_per_step") or []
            if not mp or not ip:
                continue
            per_game_step1_m.setdefault(game, []).append(mp[0])
            per_game_step1_i.setdefault(game, []).append(ip[0])
            per_game_ar30_m.setdefault(game, []).append(float(np.mean(mp)))
            per_game_ar30_i.setdefault(game, []).append(float(np.mean(ip)))
    if not per_game_step1_m:
        return {}
    games = sorted(per_game_step1_m.keys())
    s1m = [float(np.mean(per_game_step1_m[g])) for g in games]
    s1i = [float(np.mean(per_game_step1_i[g])) for g in games]
    arm = [float(np.mean(per_game_ar30_m[g])) for g in games]
    ari = [float(np.mean(per_game_ar30_i[g])) for g in games]
    wins_step1 = sum(1 for a, b in zip(s1m, s1i) if a < b)
    wins_ar30 = sum(1 for a, b in zip(arm, ari) if a < b)
    return {
        "step1_model_mean": float(np.mean(s1m)),
        "step1_model_median": float(np.median(s1m)),
        "step1_identity_mean": float(np.mean(s1i)),
        "step1_identity_median": float(np.median(s1i)),
        "ar30_model_mean": float(np.mean(arm)),
        "ar30_model_median": float(np.median(arm)),
        "ar30_identity_mean": float(np.mean(ari)),
        "ar30_identity_median": float(np.median(ari)),
        "wins_step1": wins_step1,
        "wins_ar30": wins_ar30,
        "n_games": len(games),
        "per_game": {
            g: {
                "step1_model": s1m[i],
                "step1_identity": s1i[i],
                "ar30_model": arm[i],
                "ar30_identity": ari[i],
            }
            for i, g in enumerate(games)
        },
    }


def collate(runs: list[dict]) -> tuple[list[dict], dict, dict]:
    """Returns (rows, per_game_step1_table, identity_per_game)."""
    rows: list[dict] = []
    # Per-game step-1: rows are games, columns are run labels + identity.
    per_game_step1: dict[str, dict[str, float]] = {}
    identity_per_game: dict[str, float] = {}
    for spec in runs:
        run_dir = LOGS_ROOT / spec["run_dir"]
        if not (run_dir / "config.json").exists():
            print(f"[collate] skip {spec['label']}: run dir not present yet "
                  f"({run_dir})")
            continue
        cfg = _load_config(run_dir)
        indist = _aggregate_indist(run_dir / "eval_multigame.npz")
        ho = _aggregate_heldout(run_dir / "heldout_v4_n30" / "results.json")
        row = {
            "run_dir": spec["run_dir"],
            "label": spec["label"],
            "kind": spec["kind"],
            "n_train_games": cfg.get("games", "?"),
            "n_hid": cfg.get("n_hid"),
            "n_nca_steps": cfg.get("n_nca_steps"),
            "input_skip": cfg.get("input_skip"),
            "encode_sprites": cfg.get("encode_sprites"),
            "token_decoder_loss_weight": cfg.get("token_decoder_loss_weight"),
            "indist_random_n_games": indist.get("random", {}).get("n_games"),
            "indist_random_mean": indist.get("random", {}).get("mean"),
            "indist_random_median": indist.get("random", {}).get("median"),
            "indist_bfs_mean": indist.get("bfs", {}).get("mean"),
            "ood_step1_model_mean": ho.get("step1_model_mean"),
            "ood_step1_model_median": ho.get("step1_model_median"),
            "ood_step1_identity_mean": ho.get("step1_identity_mean"),
            "ood_step1_identity_median": ho.get("step1_identity_median"),
            "ood_ar30_model_mean": ho.get("ar30_model_mean"),
            "ood_ar30_model_median": ho.get("ar30_model_median"),
            "ood_ar30_identity_mean": ho.get("ar30_identity_mean"),
            "ood_ar30_identity_median": ho.get("ar30_identity_median"),
            "ood_wins_step1": ho.get("wins_step1"),
            "ood_wins_ar30": ho.get("wins_ar30"),
            "ood_n_games": ho.get("n_games"),
        }
        rows.append(row)
        for g, m in (ho.get("per_game") or {}).items():
            per_game_step1.setdefault(g, {})[spec["label"]] = m["step1_model"]
            identity_per_game[g] = m["step1_identity"]
    return rows, per_game_step1, identity_per_game


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_per_game_csv(
    per_game_step1: dict[str, dict[str, float]],
    identity_per_game: dict[str, float],
    out_path: Path,
) -> None:
    if not per_game_step1:
        return
    labels = sorted({k for v in per_game_step1.values() for k in v})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game"] + labels + ["identity"])
        # Sort games by best run's gap vs identity (cond if present, else first)
        sort_label = "Rule-conditional" if "Rule-conditional" in labels else labels[0]
        sorted_games = sorted(
            per_game_step1.keys(),
            key=lambda g: (per_game_step1[g].get(sort_label, float("inf"))
                           - identity_per_game.get(g, 0.0)),
        )
        for g in sorted_games:
            w.writerow(
                [g]
                + [f"{per_game_step1[g].get(l, float('nan')):.4f}" for l in labels]
                + [f"{identity_per_game.get(g, float('nan')):.4f}"]
            )


def write_latex(rows: list[dict], out_path: Path) -> None:
    """OOD cell-error table mirroring Table 5's regime columns, plus a
    final column counting per-game wins vs the identity baseline under
    the AR-30 random-action rollout.

    Heldout-30 eval currently runs random actions only, so the BFS / A*
    columns render as `--` until the oracle-action heldout eval is
    added. The ID rollout column is intentionally omitted: in-distribution
    fidelity lives in Table 5.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _cell(mean_v, median_v):
        if mean_v is None or (isinstance(mean_v, float) and np.isnan(mean_v)):
            return "--"
        if median_v is None or (isinstance(median_v, float) and np.isnan(median_v)):
            return f"{100*mean_v:.2f}"
        return f"{100*mean_v:.2f} / {100*median_v:.2f}"

    def _wins(w, n):
        return "--" if w is None or n is None else f"{w} / {n}"

    body = []
    for r in rows:
        body.append(
            "  "
            + " & ".join([
                r["label"],
                _cell(r.get("ood_ar30_model_mean"), r.get("ood_ar30_model_median")),
                _cell(r.get("ood_step1_model_mean"), r.get("ood_step1_model_median")),
                "--",  # BFS oracle on heldout: not yet evaluated
                "--",  # A* oracle on heldout: not yet evaluated
                _wins(r.get("ood_wins_ar30"), r.get("ood_n_games")),
            ])
            + r" \\"
        )
    if rows:
        # Identity baseline holds for any random-action regime; per-cell
        # state-change rates would differ for BFS / A* trajectories, so we
        # leave those `--` until those regimes are evaluated on heldout.
        ident_ar = next((
            (r.get("ood_ar30_identity_mean"), r.get("ood_ar30_identity_median"))
            for r in rows
            if r.get("ood_ar30_identity_mean") is not None
        ), (None, None))
        ident_s1 = next((
            (r.get("ood_step1_identity_mean"), r.get("ood_step1_identity_median"))
            for r in rows
            if r.get("ood_step1_identity_mean") is not None
        ), (None, None))
        body.append(
            "  "
            + " & ".join([
                "Identity baseline",
                _cell(*ident_ar),
                _cell(*ident_s1),
                "--",
                "--",
                "--",
            ])
            + r" \\"
        )
    col_spec = "l c c c c c"
    table = (
        "% AUTOGENERATED by nca_wm/scripts/collate_match_table.py — do not edit.\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "  \\toprule\n"
        "  Model & random (AR, 30-step) & random (TF, 1-step)"
        " & BFS (oracle) & A* (oracle) & wins vs.\\ identity \\\\\n"
        "  & \\small mean / median (\\%) & \\small mean / median (\\%)"
        " & \\small mean / median (\\%) & \\small mean / median (\\%)"
        " & \\small AR-30, games \\\\\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    out_path.write_text(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    rows, per_game, identity = collate(RUNS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT_DIR / "summary.csv")
    write_per_game_csv(per_game, identity, OUT_DIR / "per_game_step1.csv")
    write_latex(rows, OUT_DIR / "match_table.tex")
    # Stdout summary so we can inspect at a glance.
    fmt = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "  --  "
    for r in rows:
        print(
            f"{r['label']:<32s}  "
            f"ID random mean={fmt(r['indist_random_mean'])}  "
            f"OOD step-1={fmt(r['ood_step1_model_mean'])}  "
            f"OOD AR-30={fmt(r['ood_ar30_model_mean'])}  "
            f"wins(s1)={r['ood_wins_step1']}/{r['ood_n_games']}  "
            f"wins(AR-30)={r['ood_wins_ar30']}/{r['ood_n_games']}"
        )
    print(f"\nWrote: {OUT_DIR}/summary.csv")
    print(f"       {OUT_DIR}/per_game_step1.csv")
    print(f"       {OUT_DIR}/match_table.tex")


if __name__ == "__main__":
    main()
