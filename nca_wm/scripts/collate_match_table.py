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
        "preset": "Train-14",
        "n_games": 14,
        "kind": "uncond",
        "label": "Unconditional, Train-14",  # used for per-game CSV column header
    },
    {
        "run_dir": "multi_scaling_14_cond_match_s0",
        "preset": "Train-14",
        "n_games": 14,
        "kind": "cond",
        "label": "Rule-conditional, Train-14",
    },
    {
        "run_dir": "multi_scaling_gallery_v2_uncond_match_s0",
        "preset": "Train-59",
        "n_games": 59,
        "kind": "uncond",
        "label": "Unconditional, Train-59",
    },
    {
        "run_dir": "multi_scaling_gallery_v2_cond_match_s0",
        "preset": "Train-59",
        "n_games": 59,
        "kind": "cond",
        "label": "Rule-conditional, Train-59",
    },
    {
        "run_dir": "multi_scaling_gallery_v4_uncond_match_s0",
        "preset": "Train-199",
        "n_games": 199,
        "kind": "uncond",
        "label": "Unconditional, Train-199",
    },
    {
        "run_dir": "multi_scaling_gallery_v4_cond_match_s0",
        "preset": "Train-199",
        "n_games": 199,
        "kind": "cond",
        "label": "Rule-conditional, Train-199",
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
    """Heldout metrics: 1-step (teacher-forced) and AR-30 (random) cell-error
    averaged across rollout steps and games, plus per-game wins-vs-identity
    counts.

    Both metrics follow the definitions in Section 4.4: teacher-forced
    cell-error feeds real previous states at every step (so each step is
    an independent 1-step prediction from a different real state) while
    AR-30 feeds the model's own prediction back. We aggregate per-game
    (averaging across that game's levels, episodes, and rollout steps),
    then across games. Symmetric across runs because the same heldout
    list is used.
    """
    if not results_json_path.exists():
        return {}
    d = json.loads(results_json_path.read_text())
    ho = d.get("heldout", {})
    per_game_tf_m: dict[str, list[float]] = {}
    per_game_tf_i: dict[str, list[float]] = {}
    per_game_ar_m: dict[str, list[float]] = {}
    per_game_ar_i: dict[str, list[float]] = {}
    for game, levels in ho.items():
        for li, kinds in levels.items():
            tf_rec = kinds.get("random_tf") or {}
            tf_mp = tf_rec.get("model_cell_err_per_step") or []
            tf_ip = tf_rec.get("identity_cell_err_per_step") or []
            if tf_mp and tf_ip:
                per_game_tf_m.setdefault(game, []).append(float(np.mean(tf_mp)))
                per_game_tf_i.setdefault(game, []).append(float(np.mean(tf_ip)))
            ar_rec = kinds.get("random") or {}
            ar_mp = ar_rec.get("model_cell_err_per_step") or []
            ar_ip = ar_rec.get("identity_cell_err_per_step") or []
            if ar_mp and ar_ip:
                per_game_ar_m.setdefault(game, []).append(float(np.mean(ar_mp)))
                per_game_ar_i.setdefault(game, []).append(float(np.mean(ar_ip)))
    if not per_game_tf_m and not per_game_ar_m:
        return {}
    def _summarize(per_m, per_i):
        if not per_m:
            return None
        games = sorted(per_m.keys())
        m = [float(np.mean(per_m[g])) for g in games]
        i_ = [float(np.mean(per_i[g])) for g in games]
        wins = sum(1 for a, b in zip(m, i_) if a < b)
        return {
            "model_mean": float(np.mean(m)),
            "model_median": float(np.median(m)),
            "identity_mean": float(np.mean(i_)),
            "identity_median": float(np.median(i_)),
            "wins": wins,
            "n_games": len(games),
            "per_game_model": dict(zip(games, m)),
            "per_game_identity": dict(zip(games, i_)),
        }
    tf = _summarize(per_game_tf_m, per_game_tf_i)
    ar = _summarize(per_game_ar_m, per_game_ar_i)
    games = sorted(set((tf or {}).get("per_game_model", {})) |
                   set((ar or {}).get("per_game_model", {})))
    return {
        "tf_model_mean": (tf or {}).get("model_mean"),
        "tf_model_median": (tf or {}).get("model_median"),
        "tf_identity_mean": (tf or {}).get("identity_mean"),
        "tf_identity_median": (tf or {}).get("identity_median"),
        "tf_wins": (tf or {}).get("wins"),
        "ar30_model_mean": (ar or {}).get("model_mean"),
        "ar30_model_median": (ar or {}).get("model_median"),
        "ar30_identity_mean": (ar or {}).get("identity_mean"),
        "ar30_identity_median": (ar or {}).get("identity_median"),
        "ar30_wins": (ar or {}).get("wins"),
        "n_games": (tf or ar or {}).get("n_games"),
        "per_game": {
            g: {
                "tf_model": (tf or {}).get("per_game_model", {}).get(g),
                "tf_identity": (tf or {}).get("per_game_identity", {}).get(g),
                "ar30_model": (ar or {}).get("per_game_model", {}).get(g),
                "ar30_identity": (ar or {}).get("per_game_identity", {}).get(g),
            }
            for g in games
        },
    }


def collate(runs: list[dict]) -> tuple[list[dict], dict, dict]:
    """Returns (rows, per_game_step1_table, identity_per_game).

    Rows are emitted for every spec in `runs`, even if the run dir
    isn't present yet (so the table keeps placeholder '--' cells in
    the right slot order).
    """
    rows: list[dict] = []
    per_game_step1: dict[str, dict[str, float]] = {}
    identity_per_game: dict[str, float] = {}
    for spec in runs:
        run_dir = LOGS_ROOT / spec["run_dir"]
        present = (run_dir / "config.json").exists()
        if present:
            cfg = _load_config(run_dir)
            indist = _aggregate_indist(run_dir / "eval_multigame.npz")
            ho = _aggregate_heldout(run_dir / "heldout_v4_n30" / "results.json")
        else:
            print(f"[collate] {spec['label']}: run dir not present, "
                  f"emitting placeholder row")
            cfg, indist, ho = {}, {}, {}
        row = {
            "run_dir": spec["run_dir"],
            "label": spec["label"],
            "preset": spec["preset"],
            "n_games": spec["n_games"],
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
            "ood_tf_model_mean": ho.get("tf_model_mean"),
            "ood_tf_model_median": ho.get("tf_model_median"),
            "ood_tf_identity_mean": ho.get("tf_identity_mean"),
            "ood_tf_identity_median": ho.get("tf_identity_median"),
            "ood_tf_wins": ho.get("tf_wins"),
            "ood_ar30_model_mean": ho.get("ar30_model_mean"),
            "ood_ar30_model_median": ho.get("ar30_model_median"),
            "ood_ar30_identity_mean": ho.get("ar30_identity_mean"),
            "ood_ar30_identity_median": ho.get("ar30_identity_median"),
            "ood_ar30_wins": ho.get("ar30_wins"),
            "ood_n_games": ho.get("n_games"),
        }
        rows.append(row)
        for g, m in (ho.get("per_game") or {}).items():
            if m.get("tf_model") is None:
                continue
            per_game_step1.setdefault(g, {})[spec["label"]] = m["tf_model"]
            if m.get("tf_identity") is not None:
                identity_per_game[g] = m["tf_identity"]
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
    """OOD cell-error table with the same Preset / Model layout as
    Table 5, plus a wins-vs-identity column.

    Heldout-30 eval currently runs random actions only, so BFS (AR) /
    A* (AR) render as `--` until those rollouts are added.
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

    # Group rows by preset so we can emit \multirow + a \midrule between
    # presets, matching the in-distribution scaling table.
    groups: list[list[dict]] = []
    for r in rows:
        if groups and groups[-1][0]["preset"] == r["preset"]:
            groups[-1].append(r)
        else:
            groups.append([r])

    body = []
    for gi, group in enumerate(groups):
        if gi > 0:
            body.append("  \\midrule")
        size = len(group)
        preset = group[0]["preset"]
        preset_cell = (
            f"\\multirow{{{size}}}{{*}}{{\\textsc{{{preset}}}}}"
            if size > 1
            else f"\\textsc{{{preset}}}"
        )
        for ri, r in enumerate(group):
            first = preset_cell if ri == 0 else ""
            body.append(
                "  "
                + " & ".join([
                    first,
                    r["kind"],
                    _cell(r.get("ood_ar30_model_mean"),
                          r.get("ood_ar30_model_median")),
                    _cell(r.get("ood_tf_model_mean"),
                          r.get("ood_tf_model_median")),
                    "--",  # BFS (AR) on heldout: not yet evaluated
                    "--",  # A* (AR) on heldout: not yet evaluated
                    _wins(r.get("ood_tf_wins"), r.get("ood_n_games")),
                ])
                + r" \\"
            )

    if rows:
        ident_ar = next((
            (r.get("ood_ar30_identity_mean"), r.get("ood_ar30_identity_median"))
            for r in rows
            if r.get("ood_ar30_identity_mean") is not None
        ), (None, None))
        ident_tf = next((
            (r.get("ood_tf_identity_mean"), r.get("ood_tf_identity_median"))
            for r in rows
            if r.get("ood_tf_identity_mean") is not None
        ), (None, None))
        body.append("  \\midrule")
        body.append(
            "  "
            + " & ".join([
                "\\multicolumn{2}{l}{Identity baseline}",
                _cell(*ident_ar),
                _cell(*ident_tf),
                "--",
                "--",
                "--",
            ])
            + r" \\"
        )

    col_spec = "l l c c c c c"
    table = (
        "% AUTOGENERATED by nca_wm/scripts/collate_match_table.py — do not edit.\n"
        "\\begin{adjustbox}{max width=\\linewidth}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "  \\toprule\n"
        "  Preset & Model & random (AR) & 1-step (TF)"
        " & BFS (AR) & A* (AR) & wins vs.\\ identity \\\\\n"
        "  & & \\small mean / median (\\%) & \\small mean / median (\\%)"
        " & \\small mean / median (\\%) & \\small mean / median (\\%)"
        " & \\small TF, games \\\\\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
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
            f"OOD TF mean={fmt(r['ood_tf_model_mean'])}  "
            f"OOD AR-30={fmt(r['ood_ar30_model_mean'])}  "
            f"wins(TF)={r['ood_tf_wins']}/{r['ood_n_games']}  "
            f"wins(AR-30)={r['ood_ar30_wins']}/{r['ood_n_games']}"
        )
    print(f"\nWrote: {OUT_DIR}/summary.csv")
    print(f"       {OUT_DIR}/per_game_step1.csv")
    print(f"       {OUT_DIR}/match_table.tex")


if __name__ == "__main__":
    main()
