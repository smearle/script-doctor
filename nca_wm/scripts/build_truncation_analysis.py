#!/usr/bin/env python3
"""Build the heldout-truncation impact summary used in the paper.

For each matched-recipe checkpoint with a refreshed `heldout_v4_n30/results.json`,
recomputes the OOD aggregates dropping the 4 games whose tokenized rule sets
exceed `max_seq_len=657` (so the conditional encoder is silently truncated):
`angize_by_ali_nikkhah`, `headless_people_problems_by_monakrom`,
`break_out_of_the_mine_by_jja_i.e._juan,_jose_&_andre`,
`Heroes_of_Sokoban_-_Ancient_Japan`.

Outputs:
  nca_wm/paper/figures/heldout_truncation/truncation_summary.csv
  nca_wm/paper/figures/heldout_truncation/truncation_table.tex

The LaTeX fragment is sized for an appendix subsection ("Truncation
sensitivity of Heldout-30"). Keep the main-body Table 6 unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "heldout_truncation"

TRUNCATED = {
    "angize_by_ali_nikkhah",
    "break_out_of_the_mine_by_jja_i.e._juan,_jose_&_andre",
    "headless_people_problems_by_monakrom",
    "Heroes_of_Sokoban_-_Ancient_Japan",
}

RUNS = [
    ("Train-14",  "uncond", "multi_scaling_14_uncond_match_s0"),
    ("Train-14",  "cond",   "multi_scaling_14_cond_match_s0"),
    ("Train-59",  "uncond", "multi_scaling_gallery_v2_uncond_match_s0"),
    ("Train-59",  "cond",   "multi_scaling_gallery_v2_cond_match_s0"),
    ("Train-199", "uncond", "multi_scaling_gallery_v4_uncond_match_s0"),
    ("Train-199", "cond",   "multi_scaling_gallery_v4_cond_match_s0"),
]


def per_game_metric(path: Path, kind: str):
    if not path.exists():
        return {}, {}
    r = json.loads(path.read_text())
    out_m, out_i = {}, {}
    for g, levels in r.get("heldout", {}).items():
        ms, ids = [], []
        for li, kinds in levels.items():
            rec = kinds.get(kind) or {}
            mp = rec.get("model_cell_err_per_step") or []
            ip = rec.get("identity_cell_err_per_step") or []
            if mp and ip:
                ms.append(float(np.mean(mp)))
                ids.append(float(np.mean(ip)))
        if ms:
            out_m[g] = float(np.mean(ms))
            out_i[g] = float(np.mean(ids))
    return out_m, out_i


def summarize(model_pg: dict, ident_pg: dict, drop=frozenset()):
    keep = [g for g in sorted(model_pg) if g not in drop]
    if not keep:
        return None
    m = np.array([model_pg[g] for g in keep])
    i = np.array([ident_pg[g] for g in keep])
    return {
        "n": len(keep),
        "mean": float(m.mean()),
        "median": float(np.median(m)),
        "wins": int(np.sum(m < i)),
        "ident_mean": float(i.mean()),
        "ident_median": float(np.median(i)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for preset, kind, run_dir in RUNS:
        path = LOGS_ROOT / run_dir / "heldout_v4_n30" / "results.json"
        m, i = per_game_metric(path, "random_tf")
        all_s = summarize(m, i)
        keep_s = summarize(m, i, drop=TRUNCATED)
        rows.append({
            "preset": preset,
            "kind": kind,
            "run_dir": run_dir,
            "all": all_s,
            "keep": keep_s,
        })

    # CSV — one row per (preset, kind), columns = all + keep summaries
    csv_path = OUT_DIR / "truncation_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "preset", "kind", "run_dir",
            "all_n", "all_mean_pct", "all_median_pct", "all_wins",
            "keep_n", "keep_mean_pct", "keep_median_pct", "keep_wins",
            "ident_mean_pct_all", "ident_median_pct_all",
            "ident_mean_pct_keep", "ident_median_pct_keep",
        ])
        for r in rows:
            a = r["all"] or {}
            k = r["keep"] or {}
            w.writerow([
                r["preset"], r["kind"], r["run_dir"],
                a.get("n"),
                f"{100*a['mean']:.2f}" if a else "",
                f"{100*a['median']:.2f}" if a else "",
                a.get("wins"),
                k.get("n"),
                f"{100*k['mean']:.2f}" if k else "",
                f"{100*k['median']:.2f}" if k else "",
                k.get("wins"),
                f"{100*a['ident_mean']:.2f}" if a else "",
                f"{100*a['ident_median']:.2f}" if a else "",
                f"{100*k['ident_mean']:.2f}" if k else "",
                f"{100*k['ident_median']:.2f}" if k else "",
            ])

    # LaTeX — appendix-sized table.
    def cell(s):
        if s is None:
            return "--"
        return f"{100*s['mean']:.2f} / {100*s['median']:.2f}"

    def wcell(s):
        if s is None:
            return "--"
        return f"{s['wins']} / {s['n']}"

    body = []
    groups = {}
    for r in rows:
        groups.setdefault(r["preset"], []).append(r)
    presets = ["Train-14", "Train-59", "Train-199"]
    for gi, preset in enumerate(presets):
        if gi > 0:
            body.append("  \\midrule")
        group = groups.get(preset, [])
        size = len(group)
        for ri, r in enumerate(group):
            first = (
                f"\\multirow{{{size}}}{{*}}{{\\textsc{{{preset}}}}}"
                if ri == 0 and size > 1
                else (f"\\textsc{{{preset}}}" if ri == 0 else "")
            )
            body.append(
                "  "
                + " & ".join([
                    first,
                    r["kind"],
                    cell(r["all"]),
                    wcell(r["all"]),
                    cell(r["keep"]),
                    wcell(r["keep"]),
                ])
                + r" \\"
            )

    # Identity baseline: match the convention used by collate_match_table.py
    # (first run with non-None identity). Different rows have slightly
    # different identity values when a game is missing from one run's
    # JSON (n=29 vs n=30); using the first-row identity keeps this table
    # consistent with main-body Table 6's identity row.
    ident_all = next((r["all"] for r in rows if r["all"]), None)
    ident_keep = next((r["keep"] for r in rows if r["keep"]), None)
    body.append("  \\midrule")
    body.append(
        "  "
        + " & ".join([
            "\\multicolumn{2}{l}{Identity baseline}",
            (f"{100*ident_all['ident_mean']:.2f} / {100*ident_all['ident_median']:.2f}"
             if ident_all else "--"),
            "--",
            (f"{100*ident_keep['ident_mean']:.2f} / {100*ident_keep['ident_median']:.2f}"
             if ident_keep else "--"),
            "--",
        ])
        + r" \\"
    )

    table = (
        "% AUTOGENERATED by nca_wm/scripts/build_truncation_analysis.py.\n"
        "% Heldout-26 = Heldout-30 minus 4 games whose tokenized rule sets\n"
        "% exceed max_seq_len=657: angize_by_ali_nikkhah,\n"
        "% break_out_of_the_mine_by_..., headless_people_problems_by_monakrom,\n"
        "% Heroes_of_Sokoban_-_Ancient_Japan.\n"
        "\\begin{adjustbox}{max width=\\linewidth}\n"
        "\\begin{tabular}{l l c c c c}\n"
        "  \\toprule\n"
        "  & & \\multicolumn{2}{c}{\\textsc{Heldout-30}}"
        " & \\multicolumn{2}{c}{\\textsc{Heldout-26} (no trunc.)} \\\\\n"
        "  \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
        "  Preset & Model & 1-step (TF) & wins vs.\\ id."
        " & 1-step (TF) & wins vs.\\ id. \\\\\n"
        "  & & \\small mean / median (\\%) & \\small games"
        " & \\small mean / median (\\%) & \\small games \\\\\n"
        "  \\midrule\n"
        + "\n".join(body)
        + "\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
    )
    (OUT_DIR / "truncation_table.tex").write_text(table)

    print("Wrote:")
    print(f"  {csv_path}")
    print(f"  {OUT_DIR / 'truncation_table.tex'}")
    for r in rows:
        a = r["all"]; k = r["keep"]
        if a is None and k is None:
            print(f"  {r['preset']:<10s} {r['kind']:<7s}: (missing)")
            continue
        print(f"  {r['preset']:<10s} {r['kind']:<7s}: "
              f"all n={a['n']:2d} {100*a['mean']:5.2f}/{100*a['median']:5.2f}  wins {a['wins']:2d} "
              f"|  keep n={k['n']:2d} {100*k['mean']:5.2f}/{100*k['median']:5.2f}  wins {k['wins']:2d}")


if __name__ == "__main__":
    main()
