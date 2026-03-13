import json
import os
import re

import hydra
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from conf.config import PlotRandProfileConfig
from profile_rand_jax import get_level_int, get_step_int, get_vmap
from puzzlescript_jax.globals import (
    CPP_PROFILING_RESULTS_DIR,
    GAMES_TO_N_RULES_PATH,
    JAX_PROFILING_RESULTS_DIR,
    NODEJS_PROFILING_RESULTS_DIR,
    PLOTS_DIR,
    PRIORITY_GAMES,
)
from puzzlescript_jax.preprocessing import count_rules
from puzzlescript_jax.utils import init_ps_env


GAMES_TO_PLOT = PRIORITY_GAMES
JAX_RUN_STYLES = {
    True: {"label": "PuzzleJAX", "color": "C0", "marker": "x", "linestyle": "-"},
    False: {"label": "PuzzleJAX (for loop)", "color": "C1", "marker": "x", "linestyle": "-"},
}
NODEJS_RUN_STYLES = {
    "single_process": {"label": "NodeJS", "color": "C3", "marker": "o", "linestyle": "--"},
    "nodejs_native": {"label": "NodeJS (native)", "color": "C4", "marker": "^", "linestyle": "--"},
    "nodejs_batched": {"label": "NodeJS (batched)", "color": "C5", "marker": "D", "linestyle": "--"},
    "multiprocess": {"label": "NodeJS (multiprocess)", "color": "C2", "marker": "s", "linestyle": "--"},
    "nodejs_native_multiprocess": {
        "label": "NodeJS (native multiprocess)",
        "color": "C9",
        "marker": "D",
        "linestyle": "--",
    },
}
CPP_RUN_STYLES = {
    "cpp_batched": {"label": "C++ (batched)", "color": "C8", "marker": "*", "linestyle": "-"},
    "cpp_native": {"label": "C++ (native)", "color": "C6", "marker": "P", "linestyle": "-."},
    "cpp_native_multiprocess": {
        "label": "C++ (native multiprocess)",
        "color": "C7",
        "marker": "v",
        "linestyle": "-.",
    },
}
INCLUDED_JAX_RUN_TYPES = [
    True,
    # False,
]
INCLUDED_NODEJS_RUN_TYPES = [
    "nodejs_batched",
    # "single_process",
    # "nodejs_native",
    # "multiprocess",
    # "nodejs_native_multiprocess",
]
INCLUDED_CPP_RUN_TYPES = [
    "cpp_batched",
    # "cpp_native",
    # "cpp_native_multiprocess",
]
LEGEND_LABEL_ORDER = [
    "PuzzleJAX",
    "PuzzleJAX (for loop)",
    "NodeJS",
    "NodeJS (native)",
    "NodeJS (batched)",
    "NodeJS (multiprocess)",
    "NodeJS (native multiprocess)",
    "C++ (batched)",
    "C++ (native)",
    "C++ (native multiprocess)",
]
PROFILE_STATS_KEY_RE = re.compile(r"(?P<n_envs>\d+)-(?P<execution_mode>[a-z_]+)(?:-threads-(?P<num_threads>\d+))?$")
CPP_BATCHED_THREAD_MARKERS = {
    1: "o",
    2: "s",
    4: "^",
    8: "v",
    16: "D",
    24: "P",
    32: "X",
}


def _get_best_fps(stats: dict) -> float:
    fpss = stats.get("fps", ())
    if not fpss:
        return 0.0
    return float(max(fpss))


def _has_valid_fps(stats: dict) -> bool:
    fpss = stats.get("fps")
    return isinstance(fpss, list) and len(fpss) > 0


def _truncate_series_on_first_best_fps_drop(points: list[dict]) -> list[dict]:
    truncated_points = []
    prev_best_fps = None

    for point in points:
        current_best_fps = point["best_fps"]
        truncated_points.append(point)
        if prev_best_fps is not None and current_best_fps < prev_best_fps:
            break
        prev_best_fps = current_best_fps

    return truncated_points


def _load_games_to_n_rules() -> dict:
    with open(GAMES_TO_N_RULES_PATH, "r") as f:
        return json.load(f)


def _get_game_metadata(game: str, games_to_n_rules: dict) -> tuple[int, bool]:
    key = f"{game}.txt"
    if key not in games_to_n_rules:
        print(f"Game {game} not found in games_to_n_rules. Computing metadata on demand.")
        env = init_ps_env(game=game, level_i=0, max_episode_steps=1000, vmap=True)
        games_to_n_rules[key] = (count_rules(env.tree), env.has_randomness())
    return tuple(games_to_n_rules[key])


def _normalize_device_label(device: str) -> str:
    return device.replace("_", " ")


def _parse_profile_stats_key(stats_key: str) -> tuple[int, str, int | None] | None:
    match = PROFILE_STATS_KEY_RE.fullmatch(stats_key)
    if match is None:
        return None
    n_envs = int(match.group("n_envs"))
    execution_mode = match.group("execution_mode")
    num_threads = match.group("num_threads")
    return n_envs, execution_mode, (None if num_threads is None else int(num_threads))


def _get_cpp_batched_thread_marker(num_threads: int) -> str:
    return CPP_BATCHED_THREAD_MARKERS.get(num_threads, "H")


def _format_cpp_batched_thread_label(base_label: str, num_threads: int) -> str:
    suffix = f", {num_threads} thread{'s' if num_threads != 1 else ''}"
    if base_label.endswith(")"):
        return f"{base_label[:-1]}{suffix})"
    return f"{base_label}{suffix}"


def _discover_rollout_lengths() -> list[str]:
    rollout_lengths = set()
    for root_dir in (
        JAX_PROFILING_RESULTS_DIR,
        NODEJS_PROFILING_RESULTS_DIR,
        CPP_PROFILING_RESULTS_DIR,
    ):
        if not os.path.isdir(root_dir):
            continue
        for device in os.listdir(root_dir):
            device_dir = os.path.join(root_dir, device)
            if not os.path.isdir(device_dir):
                continue
            for rollout_len_str in os.listdir(device_dir):
                rollout_dir = os.path.join(device_dir, rollout_len_str)
                if os.path.isdir(rollout_dir):
                    rollout_lengths.add(rollout_len_str)
    return sorted(rollout_lengths, key=get_step_int)


def _collect_jax_series(rollout_len_str: str) -> dict[str, list[dict]]:
    series_by_game = {}
    if not os.path.isdir(JAX_PROFILING_RESULTS_DIR):
        return series_by_game

    for device in os.listdir(JAX_PROFILING_RESULTS_DIR):
        rollout_dir = os.path.join(JAX_PROFILING_RESULTS_DIR, device, rollout_len_str)
        if not os.path.isdir(rollout_dir):
            continue

        for profile_variant in os.listdir(rollout_dir):
            variant_dir = os.path.join(rollout_dir, profile_variant)
            if not os.path.isdir(variant_dir):
                continue

            for game in os.listdir(variant_dir):
                game_dir = os.path.join(variant_dir, game)
                if not os.path.isdir(game_dir):
                    continue

                for level_path in os.listdir(game_dir):
                    if not level_path.endswith(".json"):
                        continue
                    level_str = level_path[:-5]
                    if get_level_int(level_str) != 0:
                        continue

                    level_results_path = os.path.join(game_dir, level_path)
                    with open(level_results_path, "r") as f:
                        n_envs_to_stats = json.load(f)

                    n_envs = sorted(
                        int(n_env)
                        for n_env, stats in n_envs_to_stats.items()
                        if _has_valid_fps(stats)
                    )
                    points = [
                        {
                            "x": n_env,
                            "y": n_envs_to_stats[str(n_env)]["fps"][-1],
                            "best_fps": _get_best_fps(n_envs_to_stats[str(n_env)]),
                        }
                        for n_env in n_envs
                    ]
                    points = _truncate_series_on_first_best_fps_drop(points)
                    vmap = get_vmap(level_str)
                    if vmap not in INCLUDED_JAX_RUN_TYPES:
                        continue
                    style = JAX_RUN_STYLES[vmap]
                    label = style["label"]
                    if len(os.listdir(JAX_PROFILING_RESULTS_DIR)) > 1:
                        label = f"{label} [{_normalize_device_label(device)}]"

                    series_by_game.setdefault(game, []).append({
                        "label": label,
                        "x": [point["x"] for point in points],
                        "y": [point["y"] for point in points],
                        "color": style["color"],
                        "linestyle": style["linestyle"],
                        "marker": style["marker"],
                    })
    return series_by_game


def _collect_nodejs_series(rollout_len_str: str) -> dict[str, list[dict]]:
    series_by_game = {}
    if not os.path.isdir(NODEJS_PROFILING_RESULTS_DIR):
        return series_by_game

    node_devices = [d for d in os.listdir(NODEJS_PROFILING_RESULTS_DIR) if os.path.isdir(os.path.join(NODEJS_PROFILING_RESULTS_DIR, d))]
    include_device_in_label = len(node_devices) > 1

    for device in node_devices:
        rollout_dir = os.path.join(NODEJS_PROFILING_RESULTS_DIR, device, rollout_len_str)
        if not os.path.isdir(rollout_dir):
            continue

        for game in os.listdir(rollout_dir):
            game_dir = os.path.join(rollout_dir, game)
            if not os.path.isdir(game_dir):
                continue

            level_results_path = os.path.join(game_dir, "level-0.json")
            if not os.path.isfile(level_results_path):
                continue

            with open(level_results_path, "r") as f:
                stats_by_key = json.load(f)

            mode_to_points = {run_type: [] for run_type in INCLUDED_NODEJS_RUN_TYPES}
            for stats_key, stats in stats_by_key.items():
                if not _has_valid_fps(stats):
                    continue
                parsed_key = _parse_profile_stats_key(stats_key)
                if parsed_key is None:
                    continue
                n_envs, execution_mode, _num_threads = parsed_key
                if execution_mode not in mode_to_points:
                    continue
                mode_to_points[execution_mode].append({
                    "x": n_envs,
                    "y": stats["fps"][-1],
                    "best_fps": _get_best_fps(stats),
                })

            for execution_mode, points in mode_to_points.items():
                if not points:
                    continue
                points.sort(key=lambda point: point["x"])
                points = _truncate_series_on_first_best_fps_drop(points)
                style = NODEJS_RUN_STYLES[execution_mode]
                base_label = style["label"]
                if include_device_in_label:
                    base_label = f"{base_label} [{_normalize_device_label(device)}]"
                series_by_game.setdefault(game, []).append({
                    "label": base_label,
                    "x": [point["x"] for point in points],
                    "y": [point["y"] for point in points],
                    "color": style["color"],
                    "linestyle": style["linestyle"],
                    "marker": style["marker"],
                })

    return series_by_game


def _collect_cpp_series(rollout_len_str: str) -> dict[str, list[dict]]:
    series_by_game = {}
    if not os.path.isdir(CPP_PROFILING_RESULTS_DIR):
        return series_by_game

    cpp_devices = [d for d in os.listdir(CPP_PROFILING_RESULTS_DIR) if os.path.isdir(os.path.join(CPP_PROFILING_RESULTS_DIR, d))]
    include_device_in_label = len(cpp_devices) > 1

    for device in cpp_devices:
        rollout_dir = os.path.join(CPP_PROFILING_RESULTS_DIR, device, rollout_len_str)
        if not os.path.isdir(rollout_dir):
            continue

        for game in os.listdir(rollout_dir):
            game_dir = os.path.join(rollout_dir, game)
            if not os.path.isdir(game_dir):
                continue

            level_results_path = os.path.join(game_dir, "level-0.json")
            if not os.path.isfile(level_results_path):
                continue

            with open(level_results_path, "r") as f:
                stats_by_key = json.load(f)

            mode_to_points = {
                run_type: [] for run_type in INCLUDED_CPP_RUN_TYPES if run_type != "cpp_batched"
            }
            cpp_batched_thread_points: dict[int, list[dict]] = {}
            cpp_batched_envelope_candidates: list[dict] = []
            for stats_key, stats in stats_by_key.items():
                if not _has_valid_fps(stats):
                    continue
                parsed_key = _parse_profile_stats_key(stats_key)
                if parsed_key is None:
                    continue
                n_envs, execution_mode, num_threads = parsed_key
                if execution_mode not in INCLUDED_CPP_RUN_TYPES:
                    continue
                best_fps = _get_best_fps(stats)
                if execution_mode == "cpp_batched":
                    if num_threads is None:
                        raise ValueError(
                            f"C++ batched profiling result is missing thread count in key: {stats_key}"
                        )
                    point = {
                        "x": n_envs,
                        "y": best_fps,
                        "best_fps": best_fps,
                        "num_threads": num_threads,
                    }
                    cpp_batched_thread_points.setdefault(num_threads, []).append(point)
                    cpp_batched_envelope_candidates.append(point)
                else:
                    mode_to_points[execution_mode].append({
                        "x": n_envs,
                        "y": best_fps,
                        "best_fps": best_fps,
                    })

            for execution_mode, points in mode_to_points.items():
                if not points:
                    continue
                points.sort(key=lambda point: point["x"])
                points = _truncate_series_on_first_best_fps_drop(points)
                style = CPP_RUN_STYLES[execution_mode]
                base_label = style["label"]
                if include_device_in_label:
                    base_label = f"{base_label} [{_normalize_device_label(device)}]"
                series_by_game.setdefault(game, []).append({
                    "label": base_label,
                    "x": [point["x"] for point in points],
                    "y": [point["y"] for point in points],
                    "color": style["color"],
                    "linestyle": style["linestyle"],
                    "marker": style["marker"],
                })

            if "cpp_batched" in INCLUDED_CPP_RUN_TYPES and cpp_batched_envelope_candidates:
                envelope_by_n_env = {}
                for point in cpp_batched_envelope_candidates:
                    prev = envelope_by_n_env.get(point["x"])
                    if prev is None or point["y"] > prev["y"]:
                        envelope_by_n_env[point["x"]] = point

                envelope_points = sorted(envelope_by_n_env.values(), key=lambda point: point["x"])
                envelope_points = _truncate_series_on_first_best_fps_drop(envelope_points)
                style = CPP_RUN_STYLES["cpp_batched"]
                base_label = style["label"]
                if include_device_in_label:
                    base_label = f"{base_label} [{_normalize_device_label(device)}]"

                for num_threads, points in sorted(
                    cpp_batched_thread_points.items(),
                    key=lambda item: item[0],
                ):
                    points = sorted(points, key=lambda point: point["x"])
                    thread_label = _format_cpp_batched_thread_label(style["label"], num_threads)
                    if include_device_in_label:
                        thread_label = f"{thread_label} [{_normalize_device_label(device)}]"
                    series_by_game.setdefault(game, []).append({
                        "label": thread_label,
                        "x": [point["x"] for point in points],
                        "y": [point["y"] for point in points],
                        "color": style["color"],
                        "linestyle": "None",
                        "marker": _get_cpp_batched_thread_marker(num_threads),
                        "alpha": 0.6,
                    })

                series_by_game.setdefault(game, []).append({
                    "label": base_label,
                    "x": [point["x"] for point in envelope_points],
                    "y": [point["y"] for point in envelope_points],
                    "color": style["color"],
                    "linestyle": style["linestyle"],
                    "marker": style["marker"],
                    "linewidth": 1.5,
                    "zorder": 3,
                })

    return series_by_game


def _get_games_to_plot(
    cfg: PlotRandProfileConfig,
    jax_series: dict,
    nodejs_series: dict,
    cpp_series: dict,
) -> list[str]:
    available_games = sorted(set(jax_series) | set(nodejs_series) | set(cpp_series))
    if cfg.all_games:
        return available_games
    return [game for game in GAMES_TO_PLOT if game in available_games]


def _plot_peak_reference_line(ax, series: dict) -> None:
    if not series["y"]:
        return
    ax.axhline(
        y=max(series["y"]),
        color=series["color"],
        linestyle=series["linestyle"],
        linewidth=1,
        alpha=0.8,
        zorder=0,
        label="_nolegend_",
    )


def _set_log_axes(ax, all_series: list[dict]) -> None:
    positive_x = [x for series in all_series for x in series["x"] if x > 0]
    positive_y = [y for series in all_series for y in series["y"] if y > 0]
    if not positive_x or not positive_y:
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=min(positive_x))
    ax.set_ylim(bottom=min(positive_y))


@hydra.main(version_base="1.3", config_path="conf", config_name="plot_rand_profile_config")
def main(cfg: PlotRandProfileConfig):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    games_to_n_rules = _load_games_to_n_rules()
    rollout_len_strs = _discover_rollout_lengths()

    if not rollout_len_strs:
        print("No profiling results found.")
        return

    for rollout_len_str in rollout_len_strs:
        print(f"Rollout len: {rollout_len_str}")
        jax_series = _collect_jax_series(rollout_len_str)
        nodejs_series = _collect_nodejs_series(rollout_len_str)
        cpp_series = _collect_cpp_series(rollout_len_str)
        games = _get_games_to_plot(cfg, jax_series, nodejs_series, cpp_series)

        if not games:
            print(f"No games with profiling results for {rollout_len_str}.")
            continue

        games_n_rules = []
        for game in games:
            games_n_rules.append((game, _get_game_metadata(game, games_to_n_rules)))
        games_n_rules.sort(key=lambda x: x[1][0])

        n_games = len(games_n_rules)
        n_rows = int(n_games ** 0.5)
        n_cols = int(n_games / n_rows) + (n_games % n_rows > 0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

        if n_games == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)

        for game_i, (game, (n_rules, has_randomness)) in enumerate(games_n_rules):
            ax_x, ax_y = (game_i // n_cols, game_i % n_cols)
            if n_games == 1 or n_rows == 1:
                ax = axes[game_i]
            else:
                ax = axes[ax_x, ax_y]

            all_series = jax_series.get(game, []) + nodejs_series.get(game, []) + cpp_series.get(game, [])
            for series in all_series:
                ax.plot(
                    series["x"],
                    series["y"],
                    label=series["label"],
                    marker=series["marker"],
                    markersize=5,
                    linestyle=series["linestyle"],
                    color=series["color"],
                    alpha=series.get("alpha", 1.0),
                    linewidth=series.get("linewidth", 1.0),
                    zorder=series.get("zorder", 2),
                )
                # _plot_peak_reference_line(ax, series)

            _set_log_axes(ax, all_series)
            ax.set_xlabel("batch size")
            ax.set_ylabel("FPS")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.grid(True)
            ax.set_title(
                f"{game}\n({n_rules} rule{'s' if n_rules != 1 else ''}"
                f"{', stochastic' if has_randomness else ''})"
            )

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                label_to_handle = dict(zip(labels, handles))
                ordered_labels = [label for label in LEGEND_LABEL_ORDER if label in label_to_handle]
                remaining_labels = [label for label in labels if label not in ordered_labels]
                final_labels = ordered_labels + remaining_labels
                final_handles = [label_to_handle[label] for label in final_labels]
                ax.legend(handles=final_handles, labels=final_labels)

        total_axes = n_rows * n_cols
        for empty_i in range(n_games, total_axes):
            ax_x, ax_y = (empty_i // n_cols, empty_i % n_cols)
            if n_games == 1 or n_rows == 1:
                axes[empty_i].axis("off")
            else:
                axes[ax_x, ax_y].axis("off")

        rollout_len = get_step_int(rollout_len_str)
        fig.suptitle(f"{rollout_len}-step random rollout", fontsize=16)
        fig.tight_layout()
        plot_path = os.path.join(
            PLOTS_DIR,
            f"random_rollout_profile_{rollout_len_str}{('_select' if not cfg.all_games else '')}.png",
        )
        plot_path = plot_path.replace(" ", "_")
        print(f"Saving plot to {plot_path}")
        fig.savefig(plot_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
