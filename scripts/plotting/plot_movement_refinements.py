"""Plot paired movement refinements and controlled rollout chunking."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

TITLES = {"sokoban_basic": "Sokoban", "blocks": "Blocks", "Zen_Puzzle_Garden": "Zen",
          "atlas shrank": "Atlas Shrank"}


def load(paths, manifest):
    records = []
    configurations = set()
    for path in paths:
        data = json.loads(path.read_text())
        if len(data["devices"]) != 1 or not data["results"]:
            raise ValueError(f"Expected completed single-device measurements in {path}")
        configurations.add((data["devices"][0], data["jax_version"], data["engine_sha256"], data["trials"]))
        manifest["inputs"][str(path)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "engine_sha256": data["engine_sha256"], "jax_version": data["jax_version"],
            "trials": data["trials"], "devices": data["devices"],
        }
        records.extend(data["results"])
    if len(configurations) != 1:
        raise ValueError("Each plotted comparison must have a consistent device, JAX version, engine and trial count")
    device, _, _, trials = configurations.pop()
    return records, device.removeprefix("NVIDIA ").removeprefix("GeForce "), trials


def save(fig, path):
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_groups(paths, manifest):
    groups = {}
    for path in paths:
        device = tuple(json.loads(path.read_text())["devices"])
        groups.setdefault(device, []).append(path)
    return [load(group, manifest) for group in groups.values()]


def plot_combined(rows, output, accepted, device, trials):
    if {r["steps"] for r in rows} != {100}:
        raise ValueError("Combined figure expects 100-step rollouts")
    games = list(dict.fromkeys(row["game"] for row in rows))
    fig, axes = plt.subplots(1, len(games), figsize=(7.1, 2.65), squeeze=False, sharey=True)
    labels = {"all": "Cleanup + slice + barriers", "cleanup_barriers": "Cleanup + barriers"}
    colors = {"all": "#176d9c", "cleanup_barriers": "#cd8432"}
    for ax, game in zip(axes.flat, games):
        for candidate, label in labels.items():
            points = sorted((r for r in rows if r["game"] == game and r["candidate"] == candidate),
                            key=lambda r: r["batch"])
            if not points:
                continue
            if len({r["batch"] for r in points}) != len(points):
                raise ValueError(f"Duplicate {game}/{candidate} measurements")
            ax.plot([r["batch"] for r in points], [r["speedup"] for r in points],
                    marker="o", markersize=3, linewidth=1.4, color=colors[candidate],
                    linestyle="-" if candidate == accepted else "--", label=label)
        ax.axhline(1, color="0.5", linewidth=0.8)
        ax.set_xscale("log", base=2)
        batches = sorted({r["batch"] for r in rows if r["game"] == game})
        ax.set_xticks(batches, [f"{b:,}" for b in batches], rotation=25)
        ax.set_title(TITLES.get(game, game))
        ax.set_xlabel("Environments per batch")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Speedup vs. preceding engine (×)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    fig.suptitle(f"{device} · 100-step full rollouts · {trials} alternating trials")
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    save(fig, output / f"movement_refinements_{device.lower().replace(' ', '')}")


def plot_chunked(rows, output, device):
    if any(r["batch"] != 16384 or r["chunk_size"] != 4096 or r["mode"] != "full" for r in rows):
        raise ValueError("Chunk figure expects full outputs at batch 16384, chunk size 4096")
    if len({r["game"] for r in rows}) != len(rows):
        raise ValueError("Duplicate chunk measurements")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(len(rows))
    values = [r["speedup"] for r in rows]
    ax.bar(x, values, width=0.55, color=["#176d9c" if v >= 1 else "#cd8432" for v in values])
    ax.axhline(1, color="0.3", linestyle="--", linewidth=0.8)
    for i, value in enumerate(values):
        ax.text(i, value + 0.025, f"{value:.2f}×", ha="center", va="bottom")
    ax.set_xticks(x, [TITLES.get(r["game"], r["game"]) for r in rows])
    ax.set_ylim(0, max(1, max(values)) * 1.18)
    ax.set_ylabel("Chunked / monolithic throughput")
    ax.set_title(f"{device} · 16,384 environments → four chunks of 4,096")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01, "Identical actions and PRNG streams; complete output trajectories checked", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, output / f"rollout_chunking_{device.lower().replace(' ', '')}")


def plot_ablations(rows, output, device):
    games = ["sokoban_basic", "blocks", "Zen_Puzzle_Garden"]
    candidates = [("parallel", "Parallel cleanup"), ("slice", "Force-array slice"),
                  ("prefix_barrier", "Prefix barrier only"),
                  ("mask_prefix_barrier", "Mask + prefix barriers")]
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    x = np.arange(len(games))
    for i, (candidate, label) in enumerate(candidates):
        values = []
        for game in games:
            points = [r for r in rows if r["candidate"] == candidate and r["game"] == game and r["batch"] == 4096]
            if len(points) != 1:
                raise ValueError(f"Missing or duplicate ablation: {candidate}/{game}")
            values.append(points[0]["speedup"])
        ax.bar(x + (i - 1.5) * 0.19, values, width=0.18, label=label)
    ax.axhline(1, color="0.3", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [TITLES[g] for g in games])
    ax.set_ylabel("A/B throughput ratio (×)")
    ax.set_title(f"{device} · separate movement ablations · batch 4,096")
    ax.set_ylim(0, 1.5)
    ax.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.01, "Layout and barrier comparisons include parallel cleanup in both variants", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, output / f"movement_ablations_{device.lower().replace(' ', '')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-results", nargs="+", type=Path)
    parser.add_argument("--combined-results", nargs="+", type=Path)
    parser.add_argument("--chunk-results", nargs="+", type=Path)
    parser.add_argument("--accepted", choices=("all", "cleanup_barriers"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not any((args.ablation_results, args.combined_results, args.chunk_results)):
        parser.error("At least one result group is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    manifest = {"inputs": {}, "accepted_candidate": args.accepted,
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
                "note": "Full-output paired experiments; separate from final-carry paper throughput curves."}
    if args.ablation_results:
        for rows, device, trials in load_groups(args.ablation_results, manifest):
            plot_ablations(rows, args.output_dir, device)
    if args.combined_results:
        for rows, device, trials in load_groups(args.combined_results, manifest):
            plot_combined(rows, args.output_dir, args.accepted, device, trials)
    if args.chunk_results:
        for rows, device, trials in load_groups(args.chunk_results, manifest):
            plot_chunked(rows, args.output_dir, device)
    (args.output_dir / "refinement_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
