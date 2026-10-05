"""Plot all games and every measured batch, with explicit stopping reasons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

from scripts.benchmarks.benchmark_paper_throughput import BENCHMARK_GAMES

TITLES = {"sokoban_basic": "Sokoban Basic", "notsnake": "Notsnake", "limerick": "Lime Rick",
          "kettle": "Kettle", "atlas shrank": "Atlas Shrank", "blocks": "Blocks",
          "nekopuzzle": "Nekopuzzle", "sokoban_match3": "Sokoban Match 3",
          "Travelling_salesman": "Travelling Salesman"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads((args.results / f"{game}.json").read_text()) for game in BENCHMARK_GAMES]
    for game, row in zip(BENCHMARK_GAMES, rows):
        if row["game"] != game or row.get("stop_reason") not in ("plateau", "regression", "max_batch"):
            raise ValueError(f"Unfinished/mislabeled sweep: {game}")
        if sorted(row["requested_batches"]) != sorted(r["batch"] for r in row["results"]):
            raise ValueError(f"Missing requested measurements: {game}")
    for field in ("engine_sha256", "jax_version", "devices", "trials", "seed", "base_steps", "min_steps", "level",
                  "output_mode", "max_episode_steps", "action_generation", "adaptive"):
        if len({json.dumps(row[field], sort_keys=True) for row in rows}) != 1:
            raise ValueError(f"Mixed configuration: {field}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 9))
    table, peaks = [], []
    for ax, row in zip(axes.flat, rows):
        points = sorted(row["results"], key=lambda r: r["batch"])
        peak = max(points, key=lambda r: r["median_fps"])
        x = [p["batch"] for p in points]
        ax.plot(x, [p["median_fps"] for p in points], color="#d62728", marker="x", markersize=3)
        ax.fill_between(x, [p["q25_fps"] for p in points], [p["q75_fps"] for p in points],
                        color="#d62728", alpha=0.18, linewidth=0)
        ax.scatter([peak["batch"]], [peak["median_fps"]], facecolors="none", edgecolors="black", s=45, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7)
        ax.set_title(TITLES.get(row["game"], row["game"].replace("_", " ")), fontsize=9)
        rate = (f"{peak['median_fps']/1e6:.2f}M" if peak['median_fps'] >= 1e6
                else f"{peak['median_fps']/1e3:.1f}K")
        ax.text(0.04, 0.97, f"Best: {rate}/s @ {peak['batch']:,}\nStop: {row['stop_reason']}",
                transform=ax.transAxes, va="top", fontsize=6.5)
        peaks.append({"game": row["game"], "best_batch": peak["batch"], "median_fps": peak["median_fps"],
                      "last_batch": x[-1], "stop_reason": row["stop_reason"]})
        for point in points:
            table.append({"game": row["game"], **{k: point.get(k) for k in
                          ("batch", "steps", "median_fps", "q25_fps", "q75_fps", "compile_s", "temporary_bytes")},
                          "stop_reason": row["stop_reason"]})
    fig.supxlabel("Batch size", fontsize=10, y=0.045)
    fig.supylabel("Environment steps/s", fontsize=10, x=0.008)
    fig.text(0.5, 0.013, "H200 · final-carry rollouts · median/IQR of five warmed calls · circles mark best measured batches",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.06, 1, 1), h_pad=2, w_pad=1.4)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"adaptive_throughput_h200.{suffix}", dpi=220)
    plt.close(fig)
    with (args.output_dir / "adaptive_throughput_h200.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    manifest = {"peaks": peaks, "engine_sha256": rows[0]["engine_sha256"], "adaptive": rows[0]["adaptive"],
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs": {str(args.results / f"{game}.json"): hashlib.sha256((args.results / f"{game}.json").read_bytes()).hexdigest()
                           for game in BENCHMARK_GAMES}}
    (args.output_dir / "adaptive_throughput_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
