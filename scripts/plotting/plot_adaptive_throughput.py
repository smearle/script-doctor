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
from scripts.plotting.plot_engine_throughput import load_cpp_results

TITLES = {"sokoban_basic": "Sokoban Basic", "notsnake": "Notsnake", "limerick": "Lime Rick",
          "kettle": "Kettle", "atlas shrank": "Atlas Shrank", "blocks": "Blocks",
          "nekopuzzle": "Nekopuzzle", "sokoban_match3": "Sokoban Match 3",
          "Travelling_salesman": "Travelling Salesman"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpp-results", type=Path)
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
    cpp = load_cpp_results(args.cpp_results, BENCHMARK_GAMES) if args.cpp_results else None
    if cpp:
        cpu_label = cpp[BENCHMARK_GAMES[0]]["cpu_model"].replace("Intel(R) Core(TM)", "Core").replace(" CPU @ 3.00GHz", "")
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 9))
    table, peaks = [], []
    for ax, row in zip(axes.flat, rows):
        points = sorted(row["results"], key=lambda r: r["batch"])
        peak = max(points, key=lambda r: r["median_fps"])
        x = [p["batch"] for p in points]
        ax.plot(x, [p["median_fps"] for p in points], color="#d62728", marker="x", markersize=3, label="JAX (H200)")
        ax.fill_between(x, [p["q25_fps"] for p in points], [p["q75_fps"] for p in points],
                        color="#d62728", alpha=0.18, linewidth=0)
        ax.scatter([peak["batch"]], [peak["median_fps"]], facecolors="none", edgecolors="black", s=45, zorder=4)
        if cpp:
            cpu_points = cpp[row["game"]]["results"]
            cx = [p["batch"] for p in cpu_points]
            ax.plot(cx, [p["median_fps"] for p in cpu_points], color="#1f77b4", marker="*", markersize=3,
                    label=f"C++ optimized ({cpu_label}, ≤{cpp[row['game']]['max_threads']} threads)")
            ax.fill_between(cx, [p["q25_fps"] for p in cpu_points], [p["q75_fps"] for p in cpu_points],
                            color="#1f77b4", alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if cpp:
            low, high = ax.get_ylim()
            ax.set_ylim(low, high * (high / low) ** 0.22)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7)
        ax.set_title(TITLES.get(row["game"], row["game"].replace("_", " ")), fontsize=9)
        rate = (f"{peak['median_fps']/1e6:.2f}M" if peak['median_fps'] >= 1e6
                else f"{peak['median_fps']/1e3:.1f}K")
        ax.text(0.04, 0.97, f"{'JAX best' if cpp else 'Best'}: {rate}/s @ {peak['batch']:,}\nStop: {row['stop_reason']}",
                transform=ax.transAxes, va="top", fontsize=6.5)
        peaks.append({"game": row["game"], "best_batch": peak["batch"], "median_fps": peak["median_fps"],
                      "last_batch": x[-1], "stop_reason": row["stop_reason"]})
        for point in points:
            table.append({"game": row["game"], "engine": "jax_h200", **{k: point.get(k) for k in
                          ("batch", "steps", "median_fps", "q25_fps", "q75_fps", "compile_s", "temporary_bytes")},
                          "stop_reason": row["stop_reason"]})
        if cpp:
            for point in cpu_points:
                table.append({"game": row["game"], "engine": "cpp_optimized_cpu", **{k: point.get(k) for k in
                              ("batch", "steps", "median_fps", "q25_fps", "q75_fps", "compile_s", "temporary_bytes")},
                              "stop_reason": cpp[row["game"]]["stop_reason"]})
    fig.supxlabel("Batch size", fontsize=10, y=0.07 if cpp else 0.045)
    fig.supylabel("Environment steps/s", fontsize=10, x=0.008)
    if cpp:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.033), ncol=2, frameon=False, fontsize=8)
        fig.text(0.5, 0.018, "JAX: final carry, continuing rollouts. C++: full RL outputs, reset between trials.",
                 ha="center", fontsize=7.5)
        fig.text(0.5, 0.003, "Median / IQR of five warmed calls; circles mark best JAX batches.",
                 ha="center", fontsize=7.5)
    else:
        fig.text(0.5, 0.013, "H200 · final-carry rollouts · median/IQR of five warmed calls · circles mark best measured batches",
                 ha="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0.10 if cpp else 0.06, 1, 1), h_pad=2, w_pad=1.4)
    stem = "adaptive_throughput_h200_cpp" if cpp else "adaptive_throughput_h200"
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)
    with (args.output_dir / f"{stem}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    manifest = {"peaks": peaks, "engine_sha256": rows[0]["engine_sha256"], "adaptive": rows[0]["adaptive"],
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs": {str(args.results / f"{game}.json"): hashlib.sha256((args.results / f"{game}.json").read_bytes()).hexdigest()
                           for game in BENCHMARK_GAMES}}
    if cpp:
        manifest["cpp_configuration"] = {k: v for k, v in cpp[BENCHMARK_GAMES[0]].items()
                                         if k not in ("game", "compiled_sha256", "results", "stop_reason", "extension_history")}
        manifest["cpp_peaks"] = []
        for game in BENCHMARK_GAMES:
            best = max(cpp[game]["results"], key=lambda p: p["median_fps"])
            manifest["cpp_peaks"].append({"game": game, "best_batch": best["batch"],
                                           "median_fps": best["median_fps"],
                                           "last_batch": cpp[game]["results"][-1]["batch"],
                                           "stop_reason": cpp[game]["stop_reason"]})
        manifest["inputs"].update({str(args.cpp_results / f"{g}.json"): hashlib.sha256((args.cpp_results / f"{g}.json").read_bytes()).hexdigest()
                                   for g in BENCHMARK_GAMES})
    (args.output_dir / "adaptive_throughput_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
