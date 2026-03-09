import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from puzzlejax.globals import (
    CPP_SOLS_DIR,
    GAMES_TO_N_RULES_PATH,
    JS_SOLS_DIR,
    PLOTS_DIR,
)


RESULT_FILENAME_RE = re.compile(r"([a-z_]+)_(\d+)-steps_level-(\d+)\.json$")
BACKEND_STYLES = {
    "NodeJS": {"color": "C2", "marker": "o"},
    "C++": {"color": "C0", "marker": "s"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot search FPS versus number of rules for NodeJS and C++ backends."
    )
    parser.add_argument("--algo", default=None, help="Optional algorithm filter, e.g. bfs or astar.")
    parser.add_argument("--steps", type=int, default=None, help="Optional n_steps filter.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to plots/search_fps_vs_rules.png.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis for FPS.",
    )
    return parser.parse_args()


def load_games_to_n_rules():
    with open(GAMES_TO_N_RULES_PATH, "r") as f:
        raw = json.load(f)
    return {game_name[:-4]: metadata[0] for game_name, metadata in raw.items()}


def parse_result_filename(filename: str):
    match = RESULT_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse result filename: {filename}")
    algo, n_steps, level_i = match.groups()
    return algo, int(n_steps), int(level_i)


def collect_records(results_dir: str, backend_name: str):
    root = Path(results_dir)
    if not root.is_dir():
        return []

    per_run_game_records = defaultdict(lambda: defaultdict(list))
    records = []
    for game_dir in sorted(root.iterdir()):
        if not game_dir.is_dir():
            continue

        for result_path in sorted(game_dir.glob("*.json")):
            try:
                algo, n_steps, _level_i = parse_result_filename(result_path.name)
            except ValueError:
                continue

            with open(result_path, "r") as f:
                level_result = json.load(f)

            fps = level_result.get("FPS")
            if fps is None:
                iterations = level_result.get("iterations")
                elapsed = level_result.get("time")
                if iterations is None or elapsed in (None, 0):
                    continue
                fps = iterations / max(elapsed, 1.0e-4)
            if fps <= 0:
                continue

            per_run_game_records[(algo, n_steps)][game_dir.name].append(float(fps))

    for (algo, n_steps), game_records in per_run_game_records.items():
        for game, fps_values in game_records.items():
            records.append(
                {
                    "backend": backend_name,
                    "algo": algo,
                    "n_steps": n_steps,
                    "game": game,
                    "fps": sum(fps_values) / len(fps_values),
                    "n_levels": len(fps_values),
                }
            )

    return records


def filter_records(records, algo=None, steps=None):
    filtered = []
    for record in records:
        if algo is not None and record["algo"] != algo:
            continue
        if steps is not None and record["n_steps"] != steps:
            continue
        filtered.append(record)
    return filtered


def group_records(records, games_to_n_rules):
    grouped = defaultdict(list)
    for record in records:
        n_rules = games_to_n_rules.get(record["game"])
        if n_rules is None:
            continue
        key = (record["algo"], record["n_steps"])
        grouped[key].append({**record, "n_rules": n_rules})
    return dict(grouped)


def make_figure(grouped_records, *, log_y: bool):
    panel_keys = sorted(grouped_records.keys(), key=lambda item: (item[0], item[1]))
    n_panels = len(panel_keys)
    n_cols = min(3, max(1, math.ceil(math.sqrt(n_panels))))
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5), squeeze=False)

    for ax, (algo, n_steps) in zip(axes.flat, panel_keys):
        panel_records = grouped_records[(algo, n_steps)]
        for backend_name, style in BACKEND_STYLES.items():
            backend_records = [record for record in panel_records if record["backend"] == backend_name]
            if not backend_records:
                continue
            ax.scatter(
                [record["n_rules"] for record in backend_records],
                [record["fps"] for record in backend_records],
                label=backend_name,
                alpha=0.75,
                s=32,
                color=style["color"],
                marker=style["marker"],
            )

        ax.set_title(f"{algo.upper()} | {n_steps:,} steps")
        ax.set_xlabel("number of rules")
        ax.set_ylabel("FPS")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Search FPS vs. Game Rule Count", fontsize=16)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    games_to_n_rules = load_games_to_n_rules()
    records = []
    records.extend(collect_records(JS_SOLS_DIR, "NodeJS"))
    records.extend(collect_records(CPP_SOLS_DIR, "C++"))
    records = filter_records(records, algo=args.algo, steps=args.steps)
    grouped_records = group_records(records, games_to_n_rules)

    if not grouped_records:
        raise SystemExit("No matching search results found to plot.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = args.output or os.path.join(PLOTS_DIR, "search_fps_vs_rules.png")
    fig = make_figure(grouped_records, log_y=args.log_y)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
