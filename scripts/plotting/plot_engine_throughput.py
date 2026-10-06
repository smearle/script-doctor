"""Publication figures for updated H200 throughput and paired engine changes.

Historical CPU curves retain the selection/statistics of plot_rand_profile.py.
They are explicitly labeled archival references, not newly measured CPU runs.
New JAX curves use every measured batch, with median and interquartile bands.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np


GAMES = ["sokoban_basic", "notsnake", "Zen_Puzzle_Garden", "Slidings",
         "limerick", "kettle", "Take_Heart_Lass", "atlas shrank"]
TITLES = ["Sokoban Basic", "Notsnake", "Zen Puzzle Garden", "Slidings",
          "Lime Rick", "Kettle", "Take Heart Lass", "Atlas Shrank"]
CPU = "Intel(R)_Core(TM)_i9-9980XE_CPU_@_3.00GHz"
KEY = re.compile(r"(\d+)-([a-z_]+)(?:-threads-(\d+))?$")
STYLES = {
    "cpp_batched": ("C++ batched (archived CPU)", "#1f77b4", "*", "-"),
    "nodejs_batched": ("NodeJS batched (archived CPU)", "#2ca02c", "D", "--"),
    "single_process": ("NodeJS (archived CPU)", "#F0C078", None, (0, (5, 3))),
    "nodejs_native": ("NodeJS engine-only (archived CPU)", "#E8943A", None, (0, (5, 3))),
}


def historical_series(path, mode):
    data = json.loads(path.read_text())
    candidates = {}
    for key, stats in data.items():
        match = KEY.fullmatch(key)
        if match is None or match[2] != mode or not stats.get("fps"):
            continue
        batch = int(match[1])
        best = max(stats["fps"])
        value = best if mode == "cpp_batched" else stats["fps"][-1]
        point = (batch, value, best)
        if batch not in candidates or best > candidates[batch][2]:
            candidates[batch] = point
    # Reproduce the archived figure's stopping rule for its archived curves.
    result = []
    for point in sorted(candidates.values()):
        result.append(point)
        if len(result) > 1 and point[2] < result[-2][2]:
            break
    return result


def save(fig, path):
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_cpp_results(results_dir, games):
    rows = {}
    for game in games:
        data = json.loads((results_dir / f"{game}.json").read_text())
        if data["game"] != game or data.get("level") != 0:
            raise ValueError(f"Mislabeled C++ result: {game}")
        if data.get("stop_reason") not in ("plateau", "regression", "max_batch") or "pending_batch" in data:
            raise ValueError(f"Unfinished C++ sweep: {game}")
        points = data["results"]
        if not points or [p["batch"] for p in points] != [2**i for i in range(len(points))]:
            raise ValueError(f"Missing C++ batch measurements: {game}")
        for p in points:
            if p["threads"] != min(p["batch"], data["max_threads"]) or len(p["samples_s"]) != data["trials"]:
                raise ValueError(f"Incomplete/inconsistent C++ timing: {game}/{p['batch']}")
        rows[game] = data
    for field in ("library_sha256", "source_hashes", "cpu_model", "cpu_affinity", "host",
                  "max_threads", "trials", "seed", "base_steps", "min_steps", "timing", "thread_environment", "adaptive"):
        if len({json.dumps(r[field], sort_keys=True) for r in rows.values()}) != 1:
            raise ValueError(f"Mixed C++ configuration: {field}")
    return rows


def plot_paper(results_dir, data_dir, output_dir, manifest, cpp_results=None):
    rows = [json.loads((results_dir / f"{game}.json").read_text()) for game in GAMES]
    for game, row in zip(GAMES, rows):
        if row["game"] != game:
            raise ValueError(f"Mislabeled game result: expected {game}, got {row['game']}")
        if not row["results"]:
            raise ValueError(f"No completed measurements for {row['game']}")
        if row["devices"] != ["NVIDIA H200"]:
            raise ValueError(f"Expected H200 results, got {row['devices']}")
        expected_batches = row.get("requested_batches", [1, 16, 256, 1024, 4096, 16384])
        if sorted(r["batch"] for r in row["results"]) != sorted(expected_batches):
            raise ValueError(f"Incomplete batch sweep for {row['game']}")
        if "adaptive" in row and row.get("stop_reason") not in ("plateau", "regression", "max_batch"):
            raise ValueError(f"Unfinished adaptive sweep for {row['game']}")
    if not all("adaptive" in row for row in rows) and len({tuple(r["batch"] for r in row["results"]) for row in rows}) != 1:
        raise ValueError("Paper sweep is incomplete: games have different batch sets")
    for field in ("engine_sha256", "jax_version", "level", "trials", "seed",
                  "base_steps", "min_steps", "max_episode_steps", "output_mode",
                  "action_generation"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"Inconsistent paper benchmark configuration: {field}")
    manifest["paper_configuration"] = {
        field: rows[0][field] for field in (
            "engine_sha256", "jax_version", "devices", "level", "trials", "seed",
            "base_steps", "min_steps", "max_episode_steps", "timing",
            "output_mode", "action_generation")}
    metadata = json.loads((data_dir / "games_to_n_rules.json").read_text())
    manifest["inputs"].append(str(data_dir / "games_to_n_rules.json"))
    cpp = load_cpp_results(cpp_results, GAMES) if cpp_results else None
    if cpp:
        manifest["cpp_configuration"] = {k: v for k, v in cpp[GAMES[0]].items()
                                         if k not in ("game", "compiled_sha256", "results", "stop_reason", "extension_history")}
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.8))
    handles = {}
    for ax, game, title, row in zip(axes.flat, GAMES, TITLES, rows):
        points = sorted(row["results"], key=lambda point: point["batch"])
        x = [p["batch"] for p in points]
        line, = ax.plot(x, [p["median_fps"] for p in points], color="#d62728",
                        marker="x", markersize=3, linewidth=1.2, label="JAX (H200, updated)")
        handles[line.get_label()] = line
        ax.fill_between(x, [p["q25_fps"] for p in points], [p["q75_fps"] for p in points],
                        color="#d62728", alpha=0.18, linewidth=0)
        manifest["inputs"].append(str(results_dir / f"{game}.json"))
        if cpp:
            points = cpp[game]["results"]
            cx = [p["batch"] for p in points]
            line, = ax.plot(cx, [p["median_fps"] for p in points], color="#1f77b4", marker="*",
                            markersize=3, linewidth=1.2, label="C++ optimized (CPU)")
            handles[line.get_label()] = line
            ax.fill_between(cx, [p["q25_fps"] for p in points], [p["q75_fps"] for p in points],
                            color="#1f77b4", alpha=0.18, linewidth=0)
            manifest["inputs"].append(str(cpp_results / f"{game}.json"))
        for mode, (label, color, marker, style) in STYLES.items():
            if cpp and mode == "cpp_batched":
                continue
            folder = "cpp_profiling_results" if mode == "cpp_batched" else "nodejs_profiling_results"
            path = data_dir / folder / CPU / "5000-step_rollout" / game / "level-0.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing historical reference: {path}")
            historic = historical_series(path, mode)
            if not historic:
                raise ValueError(f"Missing historical mode {mode} in {path}")
            manifest["inputs"].append(str(path))
            if marker is None:
                line = ax.axhline(historic[0][1], color=color, linestyle=style, linewidth=0.9, label=label)
            else:
                line, = ax.plot([p[0] for p in historic], [p[1] for p in historic], color=color,
                                marker=marker, markersize=2.5, linestyle=style, linewidth=0.9, label=label)
            handles[label] = line
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        n_rules, random = metadata[f"{game}.txt"]
        details = f"{n_rules} rule{'s' if n_rules != 1 else ''}" + (", stochastic" if random else "")
        ax.set_title(f"{title}\n({details})", fontsize=7)
        ax.tick_params(labelsize=6, width=0.5, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.supxlabel("Batch size", fontsize=8, y=0.18)
    fig.supylabel("Environment steps/s", fontsize=8, x=0.005, y=0.61)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.52, 0.05))
    if cpp:
        fig.text(0.52, 0.029, "JAX: final carry, continuing rollouts. C++: full RL outputs, reset between trials. Both: median / IQR.",
                 ha="center", fontsize=5.6)
        cpu_label = cpp[GAMES[0]]["cpu_model"].replace("Intel(R) Core(TM)", "Core").replace(" CPU @ 3.00GHz", "")
        fig.text(0.52, 0.006, f"C++: {cpu_label}, up to {cpp[GAMES[0]]['max_threads']} threads. NodeJS: archived CPU references.",
                 ha="center", fontsize=5.6)
    else:
        fig.text(0.52, 0.015,
                 f"JAX: median / IQR of {rows[0]['trials']} warmed, continuing rollouts. Archived CPU: Core i9-9980XE; original statistics.",
                 ha="center", fontsize=5.8)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.92, bottom=0.30, wspace=0.46, hspace=0.70)
    save(fig, output_dir / "random_rollout_profile_h200_updated")


def plot_movement(path, output_dir, manifest):
    data = json.loads(path.read_text())
    rows = [r for r in data["results"] if r["kind"] == "rollout"]
    games = ["sokoban_basic", "blocks", "Zen_Puzzle_Garden"]
    batches = sorted({r["batch"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    width = 0.23
    for i, batch in enumerate(batches):
        selected = [next(r for r in rows if r["game"] == game and r["batch"] == batch) for game in games]
        x = np.arange(len(games)) + (i - (len(batches) - 1) / 2) * width
        values = [100 * (r["speedup"] - 1) for r in selected]
        bars = ax.bar(x, values, width=width, label=f"Batch {batch:,}", color=["#9ecae1", "#4292c6", "#08519c"][i])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)
    ax.axhline(0, color="0.35", linewidth=0.6)
    ax.set_xticks(np.arange(3), ["Sokoban Basic", "Blocks", "Zen Puzzle Garden"])
    ax.set_ylabel("Throughput improvement (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_ylim(top=max(100 * (r["speedup"] - 1) for r in rows) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_title(f"Movement-coordinate compaction · H200 · {data['trials']} paired trials", fontsize=9)
    fig.tight_layout()
    save(fig, output_dir / "movement_coordinate_speedup_h200")
    manifest["inputs"].append(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path)
    parser.add_argument("--movement-results", type=Path)
    parser.add_argument("--cpp-results", type=Path, help="Replace archived C++ with fresh optimized CPU curves.")
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/benchmarks/results/historical-cpu"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.paper_results is None and args.movement_results is None:
        parser.error("provide --paper-results and/or --movement-results")
    if args.cpp_results and not args.paper_results:
        parser.error("--cpp-results requires --paper-results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    manifest = {"inputs": [], "matplotlib_version": matplotlib.__version__,
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "historical_cpu_note": "NodeJS curves are archived; C++ is newly measured when cpp_configuration is present.",
                "updated_jax_note": "Median/IQR of warmed final-carry rollouts, random actions generated inside the scan.",
                "movement_note": "Separate paired full-output rollout workload; only movement coordinate collection changes."}
    if args.paper_results:
        plot_paper(args.paper_results, args.data_dir, args.output_dir, manifest, args.cpp_results)
    if args.movement_results:
        plot_movement(args.movement_results, args.output_dir, manifest)
    manifest["inputs"] = [{"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                          for p in sorted(set(manifest["inputs"]))]
    (args.output_dir / "throughput_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
