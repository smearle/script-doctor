#!/usr/bin/env python3
"""Sweep launcher and cross-evaluation plotter for NCA world models.

Named sweep presets live in ``_NAMED_SWEEPS`` and define the sweep axes plus
base training arguments. A single ``sweep_name`` drives both training and
plotting: training writes ``sweep_name`` into each run's ``config.json``, and
plotting discovers runs by matching on it.

Usage (from repo root):
    python nca_wm/sweep.py cond_vs_uncond --mode train
    python nca_wm/sweep.py cond_vs_uncond --mode train --slurm
    python nca_wm/sweep.py cond_vs_uncond          # plot
    python nca_wm/sweep.py cond_vs_uncond --per_level
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOGS_DIR = "nca_wm/logs"
PLOTS_DIR = "nca_wm/plots"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_SCRIPT = str(_REPO_ROOT / "nca_wm" / "train.py")

# Config fields that are sweep metadata / not meaningful to compare
META_FIELDS = {
    "save_dir", "load", "render_gif", "play", "serve", "port",
    "wandb", "wandb_project", "wandb_name", "log_interval", "sweep_name",
}


# ===================================================================
# Named sweep presets
# ===================================================================

@dataclass
class NCAWMSweepConfig:
    """A named sweep preset.

    ``sweep_axes`` maps train_nca_world_model.py arg names to lists of values.
    ``base_args`` are CLI args shared across all runs in the sweep.
    """
    sweep_axes: dict[str, list] = field(default_factory=dict)
    base_args: list[str] = field(default_factory=list)


@dataclass
class CondVsUncond(NCAWMSweepConfig):
    """Compare conditional vs unconditional on the small game set."""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "conditional": [True, False],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "small",
        "--n_hid", "128",
        "--n_updates", "200_000",
        "--wandb",
    ])


@dataclass
class HiddenSize(NCAWMSweepConfig):
    """Sweep hidden size for conditional model."""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "n_hid": [8, 32, 128],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "small",
        "--conditional",
        "--n_updates", "1_000_000",
        "--wandb",
    ])


@dataclass
class NCASteps(NCAWMSweepConfig):
    """Sweep NCA update steps per forward pass."""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "n_nca_steps": [2, 4, 8],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "small",
        "--conditional",
        "--n_hid", "32",
        "--n_updates", "200000",
        "--wandb",
    ])


@dataclass
class SmokeTest(NCAWMSweepConfig):
    """Quick smoke test."""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "conditional": [True, False],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "small",
        "--n_hid", "8",
        "--n_updates", "1000",
    ])


@dataclass
class NCAStepsNeko(NCAWMSweepConfig):
    """Sweep n_nca_steps on nekopuzzle alone — does extra propagation fix `...`?"""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "n_nca_steps": [4, 8, 16, 32],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "scaling_1_neko",
        "--conditional",
        "--n_hid", "128",
        "--n_updates", "200000",
        "--wandb",
    ])


@dataclass
class NGamesScaling(NCAWMSweepConfig):
    """Scale conditional training from 1 → 9 games (nested subsets of `small`)."""
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        "games": ["scaling_1", "scaling_2", "scaling_4", "scaling_6", "small"],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--conditional",
        "--n_hid", "128",
        "--n_updates", "200000",
        "--wandb",
    ])


# ---------------------------------------------------------------------------
# Architecture sweep: test pool flags on games that use specific rule types.
# 4 game presets × 4 architecture variants. Each game is a singleton preset
# (defined in train_nca_world_model.MULTI_GAME_PRESETS) so dataset balance is
# moot. Use `--mode train` to launch.
# ---------------------------------------------------------------------------

@dataclass
class GlobalArchSingleGame(NCAWMSweepConfig):
    """Sweep architecture variants on one game at a time (set via base_args).

    Variants:
      - baseline   (no global pooling)
      - axis_pool  (row + col max pool — for `...` rules, axis-aligned)
      - global_pool (full-grid max pool — for `[X] [Y]` rules)
      - axis_cummax + global_pool (most expressive, handles both)

    The game is fixed via base_args; instantiate one preset per game we
    want to test.
    """
    sweep_axes: dict[str, list] = field(default_factory=lambda: {
        # encode the variant as a single string -> we expand into flags later
        "_arch_variant": ["baseline", "axis_pool", "global_pool", "cummax_global"],
    })
    base_args: list[str] = field(default_factory=lambda: [
        "--conditional",
        "--n_hid", "128",
        "--n_nca_steps", "4",  # keep small to expose architecture's value
        "--n_updates", "200000",
        "--wandb",
    ])


@dataclass
class GlobalArchNeko(GlobalArchSingleGame):
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "global_neko", "--conditional", "--n_hid", "128",
        "--n_nca_steps", "4", "--n_updates", "200000", "--wandb",
    ])


@dataclass
class GlobalArchConstellationZ(GlobalArchSingleGame):
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "global_constellationz", "--conditional", "--n_hid", "128",
        "--n_nca_steps", "4", "--n_updates", "200000", "--wandb",
    ])


@dataclass
class GlobalArchClearing(GlobalArchSingleGame):
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "global_clearing", "--conditional", "--n_hid", "128",
        "--n_nca_steps", "4", "--n_updates", "200000", "--wandb",
    ])


@dataclass
class GlobalArchNirvana(GlobalArchSingleGame):
    base_args: list[str] = field(default_factory=lambda: [
        "--games", "global_nirvana", "--conditional", "--n_hid", "128",
        "--n_nca_steps", "4", "--n_updates", "200000", "--wandb",
    ])


_NAMED_SWEEPS: dict[str, type[NCAWMSweepConfig]] = {
    "cond_vs_uncond": CondVsUncond,
    "hidden_size": HiddenSize,
    "nca_steps": NCASteps,
    "smoke_test": SmokeTest,
    "n_games_scaling": NGamesScaling,
    "nca_steps_neko": NCAStepsNeko,
    "global_arch_neko": GlobalArchNeko,
    "global_arch_constellationz": GlobalArchConstellationZ,
    "global_arch_clearing": GlobalArchClearing,
    "global_arch_nirvana": GlobalArchNirvana,
}


# ===================================================================
# Discovery
# ===================================================================

def discover_runs(sweep_name: str) -> list[dict]:
    """Find all runs whose config.json has the given sweep_name."""
    runs = []
    for d in sorted(glob.glob(os.path.join(LOGS_DIR, "*"))):
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("sweep_name") != sweep_name:
            continue
        cfg["_dir"] = d
        runs.append(cfg)
    return runs


# ===================================================================
# Varying-field detection
# ===================================================================

def _canonical(v):
    if isinstance(v, list):
        return tuple(v)
    # Treat missing fields (None) the same as their boolean default (False)
    # so newly-added flags don't spuriously appear "varying" against pre-flag runs.
    if v is None:
        return False
    return v


def _format_value(v):
    if isinstance(v, float):
        return f"{v:.2e}" if (abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e3)) else f"{v:g}"
    if isinstance(v, bool):
        return str(v)
    return str(v)


def infer_varying_fields(runs: list[dict]) -> list[str]:
    """Return field names whose values differ across runs."""
    if not runs:
        return []
    exclude = META_FIELDS | {"_dir"}
    all_keys = sorted(set().union(*(r.keys() for r in runs)) - exclude)
    varying = []
    for k in all_keys:
        vals = {_canonical(r.get(k)) for r in runs}
        if len(vals) > 1:
            varying.append(k)
    return varying


def curve_key(run: dict, fields: list[str]) -> tuple:
    return tuple((f, _canonical(run.get(f))) for f in fields)


def curve_label(key: tuple) -> str:
    if not key:
        return "default"
    return ", ".join(f"{k}={_format_value(v)}" for k, v in key)


# ===================================================================
# Loading metrics
# ===================================================================

def load_training_curves(run: dict) -> dict | None:
    """Load the latest curves_step*.npz for a run."""
    d = run["_dir"]
    curve_files = sorted(glob.glob(os.path.join(d, "curves_step*.npz")))
    if not curve_files:
        return None
    def _step(path):
        m = re.search(r"curves_step(\d+)", path)
        return int(m.group(1)) if m else 0
    return dict(np.load(max(curve_files, key=_step)))


def load_eval_metrics(run: dict) -> dict | None:
    """Load eval_multigame.npz for a run."""
    path = os.path.join(run["_dir"], "eval_multigame.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


_EVAL_METRICS = (
    "error_rate", "wrong_tiles",
    "cell_error_rate", "wrong_cells",
    "mean_wrong_cells", "mean_first_div", "first_div_step",
)
_EVAL_ROLLOUT_TYPES = ("random_tf", "random", "bfs", "astar")
_EVAL_KEY_RE = re.compile(
    r"^(?P<game>.+)_L(?P<level>\d+)_"
    r"(?P<rtype>" + "|".join(_EVAL_ROLLOUT_TYPES) + r")_"
    r"(?P<metric>" + "|".join(_EVAL_METRICS) + r")$"
)


def parse_eval_key(key: str):
    """Parse '{game}_L{level}_{rollout_type}_{metric}'."""
    m = _EVAL_KEY_RE.match(key)
    if not m:
        return None
    return m.group("game"), int(m.group("level")), m.group("rtype"), m.group("metric")


# ===================================================================
# Aggregation helpers
# ===================================================================

def _aggregate_eval_per_game(eval_data: dict, rollout_type: str = "random",
                              metric: str = "error_rate") -> dict[str, float]:
    """Mean metric per game (across levels)."""
    game_vals: dict[str, list[float]] = {}
    for key, arr in eval_data.items():
        parsed = parse_eval_key(key)
        if parsed is None:
            continue
        game, level, rtype, mtype = parsed
        if rtype != rollout_type or mtype != metric:
            continue
        game_vals.setdefault(game, []).append(float(arr.mean()))
    return {g: np.mean(vals) for g, vals in game_vals.items()}


def _aggregate_eval_per_game_level(eval_data: dict, rollout_type: str = "random",
                                     metric: str = "error_rate") -> dict[tuple[str, int], float]:
    results = {}
    for key, arr in eval_data.items():
        parsed = parse_eval_key(key)
        if parsed is None:
            continue
        game, level, rtype, mtype = parsed
        if rtype != rollout_type or mtype != metric:
            continue
        results[(game, level)] = float(arr.mean())
    return results


def _group_runs(runs: list[dict], varying: list[str]) -> dict[tuple, list[dict]]:
    """Group runs by non-seed varying fields."""
    curve_fields = [f for f in varying if f != "seed"]
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        k = curve_key(run, curve_fields)
        groups.setdefault(k, []).append(run)
    return groups


# ===================================================================
# Plotting: training curves
# ===================================================================

def _ema(arr: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Exponential moving average. alpha is the weight of the new sample."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def plot_training_curves(runs: list[dict], varying: list[str], save_dir: str,
                         ema_alpha: float = 0.01):
    groups = _group_runs(runs, varying)

    # (npz_key, display_label, use_log_scale, invert_to_error)
    metrics_to_plot = [
        ("losses", "Training Loss", True, False),
        ("accs", "Error Rate", True, True),
        ("change_accs", "Changed-Tile Error Rate", True, True),
    ]

    for metric_key, ylabel, use_log, invert in metrics_to_plot:
        # Two-panel figure: full range (left), zoomed to last 20% (right)
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 5),
                                                gridspec_kw={"width_ratios": [3, 2]})
        for key in sorted(groups, key=str):
            all_curves = []
            log_interval = None
            for run in groups[key]:
                data = load_training_curves(run)
                if data is None or metric_key not in data:
                    continue
                vals = data[metric_key]
                if invert:
                    vals = 1.0 - vals
                all_curves.append(vals)
                log_interval = log_interval or run.get("log_interval", 100)

            if not all_curves:
                continue

            min_len = min(len(c) for c in all_curves)
            stacked = np.array([c[:min_len] for c in all_curves])
            mean = stacked.mean(axis=0)
            smoothed = _ema(mean, alpha=ema_alpha)
            steps = np.arange(len(mean)) * (log_interval or 100)

            label = curve_label(key)
            color = ax_full._get_lines.get_next_color()

            # Full range: faint raw + bold smoothed
            ax_full.plot(steps, mean, color=color, alpha=0.15, linewidth=0.5)
            ax_full.plot(steps, smoothed, color=color, label=label, linewidth=1.5)
            if stacked.shape[0] > 1:
                std = stacked.std(axis=0)
                smoothed_std = _ema(std, alpha=ema_alpha)
                ax_full.fill_between(steps, smoothed - smoothed_std,
                                     smoothed + smoothed_std, color=color, alpha=0.15)

            # Zoomed: last 20%
            n_zoom = max(1, len(mean) // 5)
            ax_zoom.plot(steps[-n_zoom:], mean[-n_zoom:],
                         color=color, alpha=0.15, linewidth=0.5)
            ax_zoom.plot(steps[-n_zoom:], smoothed[-n_zoom:],
                         color=color, label=label, linewidth=1.5)
            if stacked.shape[0] > 1:
                ax_zoom.fill_between(
                    steps[-n_zoom:],
                    smoothed[-n_zoom:] - smoothed_std[-n_zoom:],
                    smoothed[-n_zoom:] + smoothed_std[-n_zoom:],
                    color=color, alpha=0.15)

        for ax in (ax_full, ax_zoom):
            ax.set_xlabel("Training Step")
            ax.set_ylabel(ylabel)
            if use_log:
                ax.set_yscale("log")
            ax.legend(fontsize=8)

        ax_full.set_title(f"{ylabel}")
        ax_zoom.set_title(f"{ylabel} (last 20%)")
        fig.suptitle("NCA World Model", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(save_dir, f"nca_wm_{metric_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ===================================================================
# Plotting: summary bars
# ===================================================================

def plot_summary_bar(runs: list[dict], varying: list[str], save_dir: str):
    groups = _group_runs(runs, varying)

    labels, final_loss, final_err, final_change_err = [], [], [], []
    for key in sorted(groups, key=str):
        losses_all, errs_all, cerrs_all = [], [], []
        for run in groups[key]:
            data = load_training_curves(run)
            if data is None:
                continue
            losses_all.append(data["losses"][-100:].mean())
            errs_all.append(1.0 - data["accs"][-100:].mean())
            cerrs_all.append(1.0 - data["change_accs"][-100:].mean())
        if not losses_all:
            continue
        labels.append(curve_label(key))
        final_loss.append(np.mean(losses_all))
        final_err.append(np.mean(errs_all))
        final_change_err.append(np.mean(cerrs_all))

    if not labels:
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(x, final_loss, color="steelblue")
    axes[0].set_ylabel("Final Loss (log)")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss")

    axes[1].bar(x, final_err, color="seagreen")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_yscale("log")
    axes[1].set_title("Prediction Error")

    axes[2].bar(x, final_change_err, color="coral")
    axes[2].set_ylabel("Change Error Rate")
    axes[2].set_yscale("log")
    axes[2].set_title("Changed-Tile Error")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.suptitle("NCA World Model — Final Training Metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "nca_wm_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ===================================================================
# Plotting: eval heatmaps
# ===================================================================

def plot_eval_heatmap(runs: list[dict], varying: list[str], save_dir: str,
                       metric: str = "error_rate"):
    groups = _group_runs(runs, varying)
    metric_label = {
        "error_rate": "Per-Bit Error Rate",
        "cell_error_rate": "Per-Cell Error Rate",
        "mean_first_div": "Mean First-Divergence Step",
    }.get(metric, metric)
    higher_is_better = metric == "mean_first_div"
    cmap = "RdYlGn" if higher_is_better else "RdYlGn_r"
    # For each eligible rollout type (skip if no data)
    for rollout_type in ["random", "random_tf", "bfs", "astar"]:
        group_labels = []
        group_game_errors: list[dict[str, float]] = []
        for key in sorted(groups, key=str):
            seed_errors = []
            for run in groups[key]:
                ev = load_eval_metrics(run)
                if ev is None:
                    continue
                seed_errors.append(_aggregate_eval_per_game(
                    ev, rollout_type, metric=metric))
            if not seed_errors or all(not s for s in seed_errors):
                continue
            all_games = sorted(set().union(*seed_errors))
            avg = {g: float(np.nanmean([se.get(g, float("nan")) for se in seed_errors]))
                   for g in all_games}
            group_labels.append(curve_label(key))
            group_game_errors.append(avg)

        if not group_labels:
            continue

        all_games = sorted(set().union(*group_game_errors))
        matrix = np.full((len(all_games), len(group_labels)), np.nan)
        for j, errs in enumerate(group_game_errors):
            for i, g in enumerate(all_games):
                matrix[i, j] = errs.get(g, np.nan)

        fig, ax = plt.subplots(
            figsize=(max(6, 2 * len(group_labels)), max(4, 0.5 * len(all_games))))
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_games)))
        ax.set_yticklabels(all_games, fontsize=8)

        thresh = 0.1 if not higher_is_better else 10.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isnan(v):
                    fmt = f"{v:.3f}" if not higher_is_better else f"{v:.1f}"
                    light = v > thresh if not higher_is_better else v < thresh
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                            color="white" if light else "black")

        ax.set_title(f"{metric_label} — {rollout_type} rollouts")
        fig.colorbar(im, ax=ax, label=metric_label)
        fig.tight_layout()
        path = os.path.join(save_dir, f"nca_wm_eval_{rollout_type}_{metric}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_per_game_training_curves(runs: list[dict], varying: list[str], save_dir: str):
    """Plot per-game change-error trajectories from per_game_* arrays in curves.

    Produces one subplot per run group showing change_err vs step for each
    game the run was trained on. Reveals which games under/over-fit.
    """
    groups = _group_runs(runs, varying)
    # Filter to groups that actually have per-game data in their latest curves
    valid = []
    for key in sorted(groups, key=str):
        for run in groups[key]:
            data = load_training_curves(run)
            if data is None:
                continue
            if any(k.startswith("per_game_") and k.endswith("_change_acc") for k in data):
                valid.append((key, groups[key]))
                break
    if not valid:
        return

    n = len(valid)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)
    for idx, (key, runs_in_group) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        # Use first run in group
        run = runs_in_group[0]
        data = load_training_curves(run)
        game_names = set()
        for k in data:
            if k.startswith("per_game_") and k.endswith("_change_acc"):
                game_names.add(k[len("per_game_"):-len("_change_acc")])
        for g in sorted(game_names):
            steps = data.get(f"per_game_{g}_step")
            cacc = data.get(f"per_game_{g}_change_acc")
            if steps is None or cacc is None or len(steps) == 0:
                continue
            cerr = 1.0 - cacc
            ax.plot(steps, cerr, label=g, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Change Error Rate")
        ax.set_title(curve_label(key))
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    # Hide unused subplots
    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Per-Game Training Change-Error", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "nca_wm_per_game_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_game_training_curves_by_game(runs: list[dict], varying: list[str],
                                           save_dir: str):
    """Transposed view of plot_per_game_training_curves.

    One subplot per game, one curve per run-group showing that game's
    training change-error trajectory for that config. Reveals whether a
    given game converges faster/slower as you add other games to the set.
    """
    groups = _group_runs(runs, varying)
    # game_name -> list[(group_key, steps, change_err)]
    by_game: dict[str, list[tuple]] = {}
    for key in sorted(groups, key=str):
        run = groups[key][0]
        data = load_training_curves(run)
        if data is None:
            continue
        for k in data:
            if not (k.startswith("per_game_") and k.endswith("_change_acc")):
                continue
            g = k[len("per_game_"):-len("_change_acc")]
            steps = data.get(f"per_game_{g}_step")
            cacc = data.get(f"per_game_{g}_change_acc")
            if steps is None or cacc is None or len(steps) == 0:
                continue
            by_game.setdefault(g, []).append((key, np.asarray(steps),
                                               1.0 - np.asarray(cacc)))
    if not by_game:
        return

    games = sorted(by_game)
    n = len(games)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)
    # Stable color per group across subplots
    all_keys = sorted({k for entries in by_game.values() for (k, _, _) in entries},
                      key=str)
    color_map = {k: plt.cm.tab10.colors[i % 10] for i, k in enumerate(all_keys)}

    for idx, g in enumerate(games):
        ax = axes[idx // ncols][idx % ncols]
        for key, steps, cerr in by_game[g]:
            ax.plot(steps, cerr, label=curve_label(key),
                    color=color_map[key], linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Change Error Rate")
        ax.set_title(g)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Per-Game Training Change-Error (curves by config)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "nca_wm_per_game_training_by_game.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_teacher_vs_autoregressive(runs: list[dict], varying: list[str], save_dir: str):
    """Scatter teacher-forced vs autoregressive error per (run-group, game).

    Points on y=x mean compounding doesn't matter (single-step is the
    bottleneck); points above y=x indicate autoregressive drift dominates.
    """
    groups = _group_runs(runs, varying)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10.colors
    had_any = False
    for ci, key in enumerate(sorted(groups, key=str)):
        tf_vals, ar_vals, games = [], [], []
        for run in groups[key]:
            ev = load_eval_metrics(run)
            if ev is None:
                continue
            tf = _aggregate_eval_per_game(ev, "random_tf", "error_rate")
            ar = _aggregate_eval_per_game(ev, "random", "error_rate")
            common = sorted(set(tf) & set(ar))
            for g in common:
                tf_vals.append(tf[g])
                ar_vals.append(ar[g])
                games.append(g)
        if not tf_vals:
            continue
        had_any = True
        ax.scatter(tf_vals, ar_vals, color=colors[ci % len(colors)],
                   label=curve_label(key), s=40, alpha=0.8)
        # Label the game with highest autoregressive error for this group
        if games:
            worst = int(np.argmax(ar_vals))
            ax.annotate(games[worst], (tf_vals[worst], ar_vals[worst]),
                        fontsize=7, alpha=0.7)
    if not had_any:
        plt.close(fig)
        return
    lims = [max(1e-5, min(ax.get_xlim()[0], ax.get_ylim()[0])),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Teacher-Forced Error Rate (single-step)")
    ax.set_ylabel("Autoregressive Error Rate (open-loop rollout)")
    ax.set_title("Single-step vs Compounding Error (random rollouts, per game)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(save_dir, "nca_wm_tf_vs_ar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_eval_per_level(runs: list[dict], varying: list[str], save_dir: str):
    groups = _group_runs(runs, varying)

    for rollout_type in ["bfs", "astar"]:
        group_labels = []
        group_level_errors: list[dict[tuple[str, int], float]] = []
        for key in sorted(groups, key=str):
            seed_errors = []
            for run in groups[key]:
                ev = load_eval_metrics(run)
                if ev is None:
                    continue
                seed_errors.append(_aggregate_eval_per_game_level(ev, rollout_type))
            if not seed_errors:
                continue
            all_keys = sorted(set().union(*seed_errors))
            avg = {k: float(np.nanmean([se.get(k, float("nan")) for se in seed_errors]))
                   for k in all_keys}
            group_labels.append(curve_label(key))
            group_level_errors.append(avg)

        if not group_labels:
            continue

        all_levels = sorted(set().union(*group_level_errors))
        level_labels = [f"{g} L{l}" for g, l in all_levels]
        matrix = np.full((len(all_levels), len(group_labels)), np.nan)
        for j, errs in enumerate(group_level_errors):
            for i, lk in enumerate(all_levels):
                matrix[i, j] = errs.get(lk, np.nan)

        fig, ax = plt.subplots(
            figsize=(max(6, 2 * len(group_labels)), max(6, 0.3 * len(all_levels))))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(group_labels)))
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_levels)))
        ax.set_yticklabels(level_labels, fontsize=6)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5,
                            color="white" if v > 0.1 else "black")

        ax.set_title(f"Per-Level Error Rate — {rollout_type} rollouts")
        fig.colorbar(im, ax=ax, label="Error Rate")
        fig.tight_layout()
        path = os.path.join(save_dir, f"nca_wm_eval_{rollout_type}_per_level.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ===================================================================
# Sweep command generation
# ===================================================================

# Boolean/store_true flags in train_nca_world_model.py
_STORE_TRUE_FLAGS = {"conditional", "render_gif", "wandb", "balanced_sampling",
                     "axis_pool", "axis_cummax", "global_pool"}

# Convenience axis: a single string maps to a set of architecture pool flags.
# Used by GlobalArch* sweep presets via the special axis name "_arch_variant".
_ARCH_VARIANT_FLAGS = {
    "baseline":       [],
    "axis_pool":      ["--axis_pool"],
    "axis_cummax":    ["--axis_cummax"],
    "global_pool":    ["--global_pool"],
    "axis_global":    ["--axis_pool", "--global_pool"],
    "cummax_global":  ["--axis_cummax", "--global_pool"],
    "all":            ["--axis_pool", "--axis_cummax", "--global_pool"],
}


def generate_sweep_commands(
    preset: NCAWMSweepConfig,
    sweep_name: str,
) -> list[list[str]]:
    """Generate CLI arg lists from a preset's base_args + sweep_axes Cartesian product."""
    axis_names = list(preset.sweep_axes.keys())
    axis_values = [preset.sweep_axes[n] for n in axis_names]

    configs = []
    for combo in product(*axis_values):
        args = list(preset.base_args)
        for name, val in zip(axis_names, combo):
            # Special meta-axis: expand a variant string into multiple flags.
            if name == "_arch_variant":
                if val not in _ARCH_VARIANT_FLAGS:
                    raise ValueError(
                        f"Unknown _arch_variant {val!r}. "
                        f"Choose from: {sorted(_ARCH_VARIANT_FLAGS)}")
                args.extend(_ARCH_VARIANT_FLAGS[val])
                continue
            flag = f"--{name}"
            # Remove any existing occurrence from base_args
            if flag in args:
                idx = args.index(flag)
                if name in _STORE_TRUE_FLAGS:
                    args.pop(idx)
                else:
                    args.pop(idx)  # flag
                    if idx < len(args):
                        args.pop(idx)  # value

            if name in _STORE_TRUE_FLAGS:
                if val:
                    args.append(flag)
            else:
                args.extend([flag, str(val)])

        args.extend(["--sweep_name", sweep_name])
        configs.append(args)
    return configs


# ===================================================================
# Main
# ===================================================================

def main():
    p = argparse.ArgumentParser(
        description="Sweep launcher and cross-evaluation plotter for NCA world models"
    )
    p.add_argument("sweep_name",
                   help=f"Named sweep preset. Available: {', '.join(sorted(_NAMED_SWEEPS))}")
    p.add_argument("--mode", default="plot", choices=["plot", "train"],
                   help="'train' to launch runs, 'plot' (default) to analyze results")
    p.add_argument("--save_dir", default=None,
                   help="Directory for output plots (default: nca_wm/plots/{sweep_name})")
    p.add_argument("--per_level", action="store_true",
                   help="Also plot per-level heatmaps")
    p.add_argument("--slurm", action="store_true",
                   help="Submit sweep jobs via SLURM instead of running locally")
    args = p.parse_args()

    if args.sweep_name not in _NAMED_SWEEPS:
        known = ", ".join(sorted(_NAMED_SWEEPS))
        p.error(f"Unknown sweep_name={args.sweep_name!r}. Available: {known}")

    save_dir = args.save_dir or os.path.join(PLOTS_DIR, args.sweep_name)
    preset = _NAMED_SWEEPS[args.sweep_name]()

    # ------------------------------------------------------------------
    # Train mode: generate and launch all runs
    # ------------------------------------------------------------------
    if args.mode == "train":
        configs = generate_sweep_commands(preset, args.sweep_name)
        print(f"Sweep {args.sweep_name!r}: {len(configs)} configuration(s)")
        print(f"  Axes: { {k: v for k, v in preset.sweep_axes.items()} }")
        print(f"  Base: {' '.join(preset.base_args)}")
        for i, cmd_args in enumerate(configs):
            print(f"  [{i}] {' '.join(cmd_args)}")

        python = sys.executable
        for i, cmd_args in enumerate(configs):
            cmd = [python, _TRAIN_SCRIPT] + cmd_args
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(configs)}] {' '.join(cmd_args)}")
            print(f"{'='*60}")
            if args.slurm:
                subprocess.run([
                    "sbatch", "--job-name", f"nca-wm-{args.sweep_name}-{i}",
                    "--mem", "30G", "--gres", "gpu:1", "--time", "24:00:00",
                    "--account", os.environ.get("SLURM_ACCOUNT", ""),
                    "--wrap", " ".join(cmd),
                ])
            else:
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(f"Config [{i}] failed with exit code {proc.returncode}")
        return

    # ------------------------------------------------------------------
    # Plot mode: discover runs and generate comparison artifacts
    # ------------------------------------------------------------------
    runs = discover_runs(args.sweep_name)
    if not runs:
        print(f"No completed runs found for sweep {args.sweep_name!r} in {LOGS_DIR}/.")
        print(f"Run with --mode train first:\n  python sweep_nca_wm.py {args.sweep_name} --mode train")
        return

    print(f"Found {len(runs)} run(s) for sweep {args.sweep_name!r}:")
    for r in runs:
        cond = r.get("conditional", False)
        n_up = r.get("n_updates", "?")
        print(f"  {r['_dir']}  (conditional={cond}, n_updates={n_up})")

    varying = infer_varying_fields(runs)
    print(f"Varying fields: {varying}" if varying else "No varying fields detected.")

    os.makedirs(save_dir, exist_ok=True)
    plot_training_curves(runs, varying, save_dir)
    plot_per_game_training_curves(runs, varying, save_dir)
    plot_per_game_training_curves_by_game(runs, varying, save_dir)
    plot_summary_bar(runs, varying, save_dir)
    for metric in ("error_rate", "cell_error_rate", "mean_first_div"):
        plot_eval_heatmap(runs, varying, save_dir, metric=metric)
    plot_teacher_vs_autoregressive(runs, varying, save_dir)
    if args.per_level:
        plot_eval_per_level(runs, varying, save_dir)

    # Text + markdown summary
    groups = _group_runs(runs, varying)
    rows = []
    for key in sorted(groups, key=str):
        label = curve_label(key)
        losses, errs, cerrs, eval_errs = [], [], [], []
        for run in groups[key]:
            data = load_training_curves(run)
            if data:
                losses.append(data["losses"][-100:].mean())
                errs.append(1.0 - data["accs"][-100:].mean())
                cerrs.append(1.0 - data["change_accs"][-100:].mean())
            ev = load_eval_metrics(run)
            if ev:
                game_errs = _aggregate_eval_per_game(ev, "random")
                if game_errs:
                    eval_errs.append(np.mean(list(game_errs.values())))
        def _fmt(vals): return f"{np.mean(vals):.4e}" if vals else "N/A"
        rows.append((label, _fmt(losses), _fmt(errs), _fmt(cerrs), _fmt(eval_errs)))

    # Print to stdout
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Config':<40} {'Loss':>10} {'Err':>10} {'ChgErr':>10} {'EvalErr':>10}"
    print(header)
    print("-" * len(header))
    for label, loss, err, cerr, eerr in rows:
        print(f"{label:<40} {loss:>10} {err:>10} {cerr:>10} {eerr:>10}")

    # Save as markdown
    md_path = os.path.join(save_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# NCA World Model Sweep: {args.sweep_name}\n\n")
        f.write(f"Varying fields: {varying}\n\n")
        f.write("| Config | Loss | Error | Change Error | Eval Error (random) |\n")
        f.write("|--------|------|-------|--------------|---------------------|\n")
        for label, loss, err, cerr, eerr in rows:
            f.write(f"| {label} | {loss} | {err} | {cerr} | {eerr} |\n")
    print(f"\nSaved {md_path}")


if __name__ == "__main__":
    main()
