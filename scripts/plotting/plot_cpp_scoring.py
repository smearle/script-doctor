"""Plot the controlled C++ scoring comparison, separate from archived curves."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = [json.loads(p.read_text()) for p in args.results]
    for field in ("host", "library_hashes", "source_hashes", "trials", "seeds", "steps",
                  "thread_environment", "cpu_affinity", "idle_workers"):
        if len({json.dumps(d[field], sort_keys=True) for d in data}) != 1:
            raise ValueError(f"Mixed configuration: {field}")
    rows = [r for d in data for r in d["results"]]
    games = list(dict.fromkeys(r["game"] for r in rows))
    configs = [(1, 1), (32, 1), (256, 8)]
    seeds = data[0]["seeds"]
    expected = {(g, b, t, s) for g in games for b, t in configs for s in seeds}
    actual = [(r["game"], r["batch"], r["threads"], r["seed"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected or len(games) != 8:
        raise ValueError("Incomplete or duplicated eight-game comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7, 4.8))
    table = []
    for i, (batch, threads) in enumerate(configs):
        means, lows, highs = [], [], []
        for game in games:
            selected = [r for r in rows if (r["game"], r["batch"], r["threads"]) == (game, batch, threads)]
            ratios = np.array([r["speedup"] for r in selected])
            mean = float(np.exp(np.mean(np.log(ratios))))
            means.append(mean)
            lows.append(mean - min(ratios))
            highs.append(max(ratios) - mean)
            table.append({"game": game, "batch": batch, "threads": threads, "speedup": mean,
                          "min_seed_ratio": min(ratios), "max_seed_ratio": max(ratios),
                          "baseline_fps": float(np.exp(np.mean(np.log([r["baseline_fps"] for r in selected])))),
                          "candidate_fps": float(np.exp(np.mean(np.log([r["candidate_fps"] for r in selected]))))})
        y = np.arange(len(games)) + (i - 1) * 0.24
        ax.barh(y, means, height=0.22, xerr=[lows, highs], capsize=2,
                color=["#9ecae1", "#4292c6", "#08519c"][i],
                label=f"Batch {batch}, {threads} thread{'s' if threads != 1 else ''}",
                error_kw={"elinewidth": 0.7})
    ax.axvline(1, color="0.3", linewidth=0.8, linestyle="--")
    titles = {"sokoban_basic": "Sokoban Basic", "blocks": "Blocks", "kettle": "Kettle",
              "notsnake": "Notsnake", "limerick": "Lime Rick"}
    ax.set_yticks(np.arange(len(games)), [titles.get(g, g.replace("_", " ")) for g in games], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Throughput / corrected C++ baseline", fontsize=9)
    ax.set_title("C++ scoring optimization · complete RL outputs", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=7, frameon=False)
    fig.text(0.5, 0.012, "Geometric mean of two seed ratios; whiskers span seeds. Both builds include the parallel flag fix.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"cpp_scoring_speedup.{suffix}", dpi=220)
    plt.close(fig)
    with (args.output_dir / "cpp_scoring_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    manifest = {"summary": table, "configuration": {k: v for k, v in data[0].items() if k != "results"},
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.results}}
    (args.output_dir / "cpp_scoring_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
