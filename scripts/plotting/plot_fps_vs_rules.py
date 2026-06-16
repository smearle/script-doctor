import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from puzzlescript_jax.globals import (
    CPP_PROFILING_RESULTS_DIR,
    CPP_SOLS_DIR,
    GAMES_TO_N_RULES_PATH,
    JAX_PROFILING_RESULTS_DIR,
    JS_SOLS_DIR,
    NODEJS_PROFILING_RESULTS_DIR,
    PLOTS_DIR,
)


SEARCH_RESULT_FILENAME_RE = re.compile(r"([a-z_]+)_(\d+)-steps_level-(\d+)\.json$")
PROFILE_LEVEL_RE = re.compile(r"level-(\d+)(?:-vmap-(True|False))?\.json$")
PROFILE_STATS_KEY_RE = re.compile(r"(?P<n_envs>\d+)-(?P<execution_mode>[a-z_]+)(?:-threads-(?P<num_threads>\d+))?$")
SEARCH_RUN_STYLES = {
    "NodeJS": {"label": "NodeJS", "color": "C2", "marker": "o"},
    "C++": {"label": "C++", "color": "C0", "marker": "s"},
}
JAX_RUN_STYLES = {
    (True, False): {"label": "PuzzleJAX", "color": "C1", "marker": "^"},
    (False, False): {"label": "PuzzleJAX (for loop)", "color": "C4", "marker": "x"},
    (True, True): {"label": "PuzzleJAX (switch)", "color": "C6", "marker": "P"},
    (False, True): {"label": "PuzzleJAX (switch for loop)", "color": "C8", "marker": "X"},
}
NODEJS_RUN_STYLES = {
    "single_process": {"label": "NodeJS", "color": "C3", "marker": "o"},
    "nodejs_native": {"label": "NodeJS (native)", "color": "C4", "marker": "^"},
    "nodejs_batched": {"label": "NodeJS (batched)", "color": "C5", "marker": "D"},
    "multiprocess": {"label": "NodeJS (multiprocess)", "color": "C2", "marker": "s"},
    "nodejs_native_multiprocess": {"label": "NodeJS (native multiprocess)", "color": "C9", "marker": "v"},
}
CPP_RUN_STYLES = {
    "cpp_batched": {"label": "C++ (batched)", "color": "C8", "marker": "*"},
    "cpp_native": {"label": "C++ (native)", "color": "C6", "marker": "P"},
    "cpp_native_multiprocess": {"label": "C++ (native multiprocess)", "color": "C7", "marker": "v"},
}
INCLUDED_JAX_RUN_TYPES = [
    (True, False),
]
INCLUDED_NODEJS_RUN_TYPES = [
    "nodejs_native_multiprocess",
    # "nodejs_batched",
]
INCLUDED_CPP_RUN_TYPES = [
    "cpp_native_multiprocess",
    # "cpp_batched",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot FPS versus number of rules from random-profile or search results."
    )
    parser.add_argument(
        "--source",
        choices=("rand", "search"),
        default="rand",
        help="Which result set to plot. Defaults to random-profile results.",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help="Optional algorithm filter for search results, e.g. bfs or astar.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional n_steps filter.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to plots/fps_vs_rules.png or plots/search_fps_vs_rules.png.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=True,
        help="Use a logarithmic y-axis for FPS.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_false",
        dest="log_y",
        help="Use a linear y-axis for FPS.",
    )
    return parser.parse_args()


def normalize_game_name(game_name: str) -> str:
    return re.sub(r"[\s_]+", "", game_name).lower()


def load_games_to_n_rules():
    with open(GAMES_TO_N_RULES_PATH, "r") as f:
        raw = json.load(f)
    return {
        normalize_game_name(game_name[:-4]): {"game": game_name[:-4], "n_rules": metadata[0]}
        for game_name, metadata in raw.items()
    }


def parse_search_result_filename(filename: str):
    match = SEARCH_RESULT_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse result filename: {filename}")
    algo, n_steps, level_i = match.groups()
    return algo, int(n_steps), int(level_i)


def parse_profile_level_filename(filename: str):
    match = PROFILE_LEVEL_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Could not parse profile result filename: {filename}")
    level_i, vmap_flag = match.groups()
    return int(level_i), None if vmap_flag is None else (vmap_flag == "True")


def parse_profile_stats_key(stats_key: str):
    match = PROFILE_STATS_KEY_RE.fullmatch(stats_key)
    if match is None:
        return None
    return int(match.group("n_envs")), match.group("execution_mode"), match.group("num_threads")


def collect_search_records(results_dir: str, backend_name: str):
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
                algo, n_steps, _level_i = parse_search_result_filename(result_path.name)
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
                    "series_key": backend_name,
                    "series_label": SEARCH_RUN_STYLES[backend_name]["label"],
                    "color": SEARCH_RUN_STYLES[backend_name]["color"],
                    "marker": SEARCH_RUN_STYLES[backend_name]["marker"],
                }
            )

    return records


def collect_jax_profile_records(results_dir: str):
    root = Path(results_dir)
    if not root.is_dir():
        return []

    per_game_records = defaultdict(list)
    for result_path in root.rglob("level-*.json"):
        try:
            _level_i, vmap_flag = parse_profile_level_filename(result_path.name)
        except ValueError:
            continue

        if len(result_path.parts) < 4:
            continue
        game = result_path.parent.name
        steps_dir = result_path.parents[2].name
        try:
            n_steps = int(steps_dir.split("-", 1)[0])
        except ValueError:
            continue

        with open(result_path, "r") as f:
            level_result = json.load(f)

        sample_stats = next(iter(level_result.values()), None)
        use_switch_env = bool(sample_stats.get("use_switch_env", False)) if isinstance(sample_stats, dict) else False
        run_type = (True if vmap_flag is None else vmap_flag, use_switch_env)
        if run_type not in INCLUDED_JAX_RUN_TYPES:
            continue

        for n_envs, stats in level_result.items():
            fps_values = stats.get("fps", [])
            if not fps_values:
                continue
            per_game_records[(n_steps, game, run_type)].append(max(float(fps) for fps in fps_values))

    records = []
    for (n_steps, game, run_type), fps_values in per_game_records.items():
        style = JAX_RUN_STYLES[run_type]
        records.append(
            {
                "backend": "JAX",
                "algo": "rand",
                "n_steps": n_steps,
                "game": game,
                "fps": max(fps_values),
                "series_key": f"jax:{run_type}",
                "series_label": style["label"],
                "color": style["color"],
                "marker": style["marker"],
            }
        )
    return records


def collect_profile_records(results_dir: str, backend_name: str):
    root = Path(results_dir)
    if not root.is_dir():
        return []

    per_game_records = defaultdict(list)
    for result_path in root.rglob("level-*.json"):
        try:
            _level_i, _vmap_flag = parse_profile_level_filename(result_path.name)
        except ValueError:
            continue

        if len(result_path.parts) < 3:
            continue
        game = result_path.parent.name
        steps_dir = result_path.parents[1].name
        try:
            n_steps = int(steps_dir.split("-", 1)[0])
        except ValueError:
            continue

        with open(result_path, "r") as f:
            level_result = json.load(f)

        for stats_key, stats in level_result.items():
            parsed_key = parse_profile_stats_key(stats_key)
            if parsed_key is None:
                if backend_name == "NodeJS" and stats_key.isdigit() and "multiprocess" in INCLUDED_NODEJS_RUN_TYPES:
                    execution_mode = "multiprocess"
                else:
                    continue
            else:
                _n_envs, execution_mode, num_threads = parsed_key
                if backend_name == "C++" and execution_mode == "cpp_batched" and num_threads is None:
                    raise ValueError(
                        f"C++ batched profiling result is missing thread count in key: {stats_key}"
                    )
            if backend_name == "NodeJS":
                if execution_mode not in INCLUDED_NODEJS_RUN_TYPES:
                    continue
                style = NODEJS_RUN_STYLES[execution_mode]
            else:
                if execution_mode not in INCLUDED_CPP_RUN_TYPES:
                    continue
                style = CPP_RUN_STYLES[execution_mode]
            fps_values = stats.get("fps", [])
            if not fps_values:
                continue
            per_game_records[(n_steps, game, execution_mode)].append(max(float(fps) for fps in fps_values))

    records = []
    for (n_steps, game, execution_mode), fps_values in per_game_records.items():
        style = NODEJS_RUN_STYLES[execution_mode] if backend_name == "NodeJS" else CPP_RUN_STYLES[execution_mode]
        records.append(
            {
                "backend": backend_name,
                "algo": "rand",
                "n_steps": n_steps,
                "game": game,
                "fps": max(fps_values),
                "series_key": f"{backend_name}:{execution_mode}",
                "series_label": style["label"],
                "color": style["color"],
                "marker": style["marker"],
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
        game_entry = games_to_n_rules.get(normalize_game_name(record["game"]))
        if game_entry is None:
            continue
        key = (record["algo"], record["n_steps"])
        grouped[key].append(
            {
                **record,
                "game": game_entry["game"],
                "n_rules": game_entry["n_rules"],
            }
        )
    return dict(grouped)


def make_figure(grouped_records, *, log_y: bool, source: str):
    panel_keys = sorted(grouped_records.keys(), key=lambda item: (item[0], item[1]))
    n_panels = len(panel_keys)
    n_cols = min(3, max(1, math.ceil(math.sqrt(n_panels))))
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 4.5), squeeze=False)

    for ax, (algo, n_steps) in zip(axes.flat, panel_keys):
        panel_records = grouped_records[(algo, n_steps)]
        for series_key in dict.fromkeys(record["series_key"] for record in panel_records):
            series_records = [record for record in panel_records if record["series_key"] == series_key]
            if not series_records:
                continue
            style = series_records[0]
            ax.scatter(
                [record["n_rules"] for record in series_records],
                [record["fps"] for record in series_records],
                label=style["series_label"],
                alpha=0.75,
                s=32,
                color=style["color"],
                marker=style["marker"],
            )

        title_prefix = algo.upper() if source == "search" else "RAND"
        ax.set_title(f"{title_prefix} | {n_steps:,} steps")
        ax.set_xlabel("number of rules")
        ax.set_ylabel("FPS")
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Search FPS vs. Game Rule Count" if source == "search" else "Random Rollout FPS vs. Game Rule Count",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


def collect_records(source: str):
    if source == "search":
        records = []
        records.extend(collect_search_records(JS_SOLS_DIR, "NodeJS"))
        records.extend(collect_search_records(CPP_SOLS_DIR, "C++"))
        return records

    records = []
    records.extend(collect_jax_profile_records(JAX_PROFILING_RESULTS_DIR))
    records.extend(collect_profile_records(NODEJS_PROFILING_RESULTS_DIR, "NodeJS"))
    records.extend(collect_profile_records(CPP_PROFILING_RESULTS_DIR, "C++"))
    return records


def default_output_path(source: str):
    filename = "search_fps_vs_rules.png" if source == "search" else "fps_vs_rules.png"
    return os.path.join(PLOTS_DIR, filename)


def main():
    args = parse_args()
    games_to_n_rules = load_games_to_n_rules()
    records = collect_records(args.source)
    records = filter_records(records, algo=args.algo, steps=args.steps)
    grouped_records = group_records(records, games_to_n_rules)

    if not grouped_records:
        raise SystemExit(f"No matching {args.source} results found to plot.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = args.output or default_output_path(args.source)
    fig = make_figure(grouped_records, log_y=args.log_y, source=args.source)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
