import json
import os

import hydra
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from conf.config import PlotRandProfileConfig
from profile_rand_jax import get_level_int, get_step_int, get_vmap
from puzzlejax.globals import (
    GAMES_TO_N_RULES_PATH,
    JAX_PROFILING_RESULTS_DIR,
    NODEJS_PROFILING_RESULTS_DIR,
    PLOTS_DIR,
    PRIORITY_GAMES,
)
from puzzlejax.preprocessing import count_rules
from puzzlejax.utils import init_ps_env


GAMES_TO_PLOT = PRIORITY_GAMES


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


def _discover_rollout_lengths() -> list[str]:
    rollout_lengths = set()
    for root_dir in (JAX_PROFILING_RESULTS_DIR, NODEJS_PROFILING_RESULTS_DIR):
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

                    n_envs = sorted(int(n_env) for n_env in n_envs_to_stats)
                    fps = [n_envs_to_stats[str(n_env)]["fps"][-1] for n_env in n_envs]
                    vmap = get_vmap(level_str)
                    label = "PuzzleJAX" if vmap else "PuzzleJAX (for loop)"
                    if len(os.listdir(JAX_PROFILING_RESULTS_DIR)) > 1:
                        label = f"{label} [{_normalize_device_label(device)}]"
                    color = "C0" if vmap else "C1"

                    series_by_game.setdefault(game, []).append({
                        "label": label,
                        "x": n_envs,
                        "y": fps,
                        "color": color,
                        "linestyle": "-",
                        "marker": "x",
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

            mode_to_points = {
                "single_process": [],
                "nodejs_native": [],
                "multiprocess": [],
                "nodejs_native_multiprocess": [],
            }
            for stats_key, stats in stats_by_key.items():
                if "error_type" in stats:
                    continue
                if "-" not in stats_key:
                    continue
                n_envs_str, execution_mode = stats_key.split("-", 1)
                if execution_mode not in mode_to_points:
                    continue
                mode_to_points[execution_mode].append((int(n_envs_str), stats["fps"][-1]))

            for execution_mode, points in mode_to_points.items():
                if not points:
                    continue
                points.sort()
                if execution_mode == "single_process":
                    base_label = "NodeJS"
                    color = "C3"
                    marker = "o"
                elif execution_mode == "nodejs_native":
                    base_label = "NodeJS (native)"
                    color = "C4"
                    marker = "^"
                elif execution_mode == "nodejs_native_multiprocess":
                    base_label = "NodeJS (native multiprocess)"
                    color = "C5"
                    marker = "D"
                else:
                    base_label = "NodeJS (multiprocess)"
                    color = "C2"
                    marker = "s"
                if include_device_in_label:
                    base_label = f"{base_label} [{_normalize_device_label(device)}]"
                linestyle = "--"
                series_by_game.setdefault(game, []).append({
                    "label": base_label,
                    "x": [x for x, _ in points],
                    "y": [y for _, y in points],
                    "color": color,
                    "linestyle": linestyle,
                    "marker": marker,
                })

    return series_by_game


def _get_games_to_plot(cfg: PlotRandProfileConfig, jax_series: dict, nodejs_series: dict) -> list[str]:
    available_games = sorted(set(jax_series) | set(nodejs_series))
    if cfg.all_games:
        return available_games
    return [game for game in GAMES_TO_PLOT if game in available_games]


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
        games = _get_games_to_plot(cfg, jax_series, nodejs_series)

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

            all_series = jax_series.get(game, []) + nodejs_series.get(game, [])
            for series in all_series:
                ax.plot(
                    series["x"],
                    series["y"],
                    label=series["label"],
                    marker=series["marker"],
                    markersize=5,
                    linestyle=series["linestyle"],
                    color=series["color"],
                )

            ax.set_yscale("linear")
            ax.set_xscale("linear")
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
                labels_order = [
                    "PuzzleJAX",
                    "PuzzleJAX (for loop)",
                    "NodeJS",
                    "NodeJS (native)",
                    "NodeJS (multiprocess)",
                    "NodeJS (native multiprocess)",
                ]
                ordered_labels = [label for label in labels_order if label in label_to_handle]
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
