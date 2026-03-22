#!/usr/bin/env python3
"""Replay action sequences from JAX-generated LLM agent results in the JS engine.

The JAX and JS engines have transposed coordinate systems.  The canonical
mapping (from puzzlescript_jax/globals.py) is:
    JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]
Inverted for replay (JAX→JS): {0:1, 1:2, 2:3, 3:0, 4:4}

This script translates JAX action IDs to JS action IDs, replays them in the
canonical JS PuzzleScript engine, and reports whether the recorded wins hold up.

It also handles the "message level" offset: the JS engine includes message levels
in its level list, but JAX skips them. So JAX level N may correspond to a
different JS level index.

With --plot, generates the same heatmap suite as plot_llm_results.py using the
JS-validated win data.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Canonical action mapping from puzzlescript_jax/globals.py:
#   JS_TO_JAX_ACTIONS = [3, 0, 1, 2, 4]
# The two engines have transposed coordinate systems, so the mapping is NOT
# a simple semantic match.  Inverting the canonical map gives us:
#   JAX 0 -> JS 1,  JAX 1 -> JS 2,  JAX 2 -> JS 3,  JAX 3 -> JS 0,  JAX 4 -> JS 4
_JS_TO_JAX = [3, 0, 1, 2, 4]
JAX_TO_JS = {jax: js for js, jax in enumerate(_JS_TO_JAX)}

GAMES_DIR = "data/scraped_games"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLER_PATH = os.path.join(SCRIPT_DIR, "puzzlescript_nodejs", "puzzlescript", "single_step_controller.js")


# ---------------------------------------------------------------------------
# JS engine helpers
# ---------------------------------------------------------------------------

def probe_js_level_map(game_text: str) -> dict[int, int]:
    """Probe the JS engine to build a mapping from JAX level index to JS level index.

    JAX skips message levels; JS includes them.  We compile the game in a
    short-lived Node process that prints level metadata, then build the map.

    Returns {jax_level_i: js_level_i}.
    """
    node_script = """
    const engine = require('./puzzlescript_nodejs/puzzlescript/engine.js');
    const gameText = process.argv[1];
    engine.compile(['loadLevel', 0], gameText);
    const levels = engine.getState().levels;
    const out = [];
    for (let i = 0; i < levels.length; i++) {
        const isMessage = typeof levels[i] === 'string' ||
                          (levels[i] && typeof levels[i] === 'object' && 'message' in levels[i] && !('width' in levels[i]));
        out.push({i, msg: isMessage});
    }
    process.stdout.write(JSON.stringify(out));
    """
    try:
        result = subprocess.run(
            ["node", "-e", node_script, game_text],
            capture_output=True, text=True, timeout=15, cwd=SCRIPT_DIR,
        )
        if result.returncode != 0:
            return {}
        levels = json.loads(result.stdout)
    except Exception:
        return {}

    jax_to_js = {}
    jax_i = 0
    for entry in levels:
        if not entry["msg"]:
            jax_to_js[jax_i] = entry["i"]
            jax_i += 1
    return jax_to_js


class JSEngine:
    """Manages a single_step_controller.js subprocess."""

    def __init__(self):
        self.proc = subprocess.Popen(
            ["node", CONTROLLER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=SCRIPT_DIR,
        )
        self._alive = True

    def _send(self, msg: dict) -> dict:
        if not self._alive:
            raise RuntimeError("JS engine process is dead")
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self._alive = False
            raise RuntimeError("JS engine pipe broken")
        line = self.proc.stdout.readline()
        if not line:
            self._alive = False
            err = self.proc.stderr.read()
            raise RuntimeError(f"JS engine died: {err[:300]}")
        return json.loads(line)

    @property
    def alive(self):
        return self._alive

    def init(self, game_text: str, level_i: int) -> dict:
        return self._send({"cmd": "init", "gameText": game_text, "levelI": level_i})

    def step(self, action: int) -> dict:
        return self._send({"cmd": "step", "action": action})

    def reset(self) -> dict:
        return self._send({"cmd": "reset"})

    def close(self):
        if not self._alive:
            return
        try:
            self._send({"cmd": "close"})
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self._alive = False


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def find_game_text(game_name: str) -> str | None:
    """Find and read the game file for a given game name."""
    candidates = [
        os.path.join(GAMES_DIR, f"{game_name}.txt"),
        os.path.join(GAMES_DIR, f"{game_name.replace(' ', '_')}.txt"),
    ]
    for entry in os.listdir(GAMES_DIR):
        if entry.lower().replace(" ", "_") == game_name.lower().replace(" ", "_") + ".txt":
            candidates.append(os.path.join(GAMES_DIR, entry))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


def collect_result_files(results_dir: str, game_filter: str | None = None,
                         model_filter: str | None = None) -> list[str]:
    """Collect all result JSON files, optionally filtered by game/model."""
    files = []
    for model_dir in sorted(glob.glob(os.path.join(results_dir, "*/"))):
        dirname = os.path.basename(model_dir.rstrip("/"))
        if dirname == "analysis":
            continue
        if model_filter and model_filter.lower() != dirname.lower():
            continue
        for f in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
            if game_filter:
                basename = os.path.basename(f).lower()
                if game_filter.lower().replace(" ", "_") not in basename:
                    continue
            files.append(f)
    return files


def replay_one(engine: JSEngine, game_text: str, js_level_i: int,
               actions: list[int], original_win: bool) -> dict:
    """Replay a single action sequence in an already-running JS engine.

    Returns a dict with js_win, js_steps, original_win, error.
    """
    try:
        resp = engine.init(game_text, js_level_i)
        if not resp.get("ok"):
            return {"js_win": False, "js_steps": 0, "original_win": original_win,
                    "error": f"init failed: {resp}"}

        for i, jax_action in enumerate(actions):
            js_action = JAX_TO_JS.get(jax_action, jax_action)
            resp = engine.step(js_action)
            if not resp.get("ok"):
                return {"js_win": False, "js_steps": i, "original_win": original_win,
                        "error": f"step {i} failed: {resp.get('error', '?')[:120]}"}
            if resp.get("won"):
                return {"js_win": True, "js_steps": i + 1, "original_win": original_win,
                        "error": None}

        return {"js_win": False, "js_steps": len(actions), "original_win": original_win,
                "error": None}
    except Exception as e:
        return {"js_win": False, "js_steps": 0, "original_win": original_win,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


# ---------------------------------------------------------------------------
# Replay loop — returns a DataFrame of per-run results
# ---------------------------------------------------------------------------

def replay_all(result_files: list[str], *, verbose: bool = False) -> pd.DataFrame:
    """Replay all result files and return a DataFrame with both original and JS outcomes."""
    game_text_cache: dict[str, str | None] = {}
    level_map_cache: dict[str, dict[int, int]] = {}
    rows: list[dict] = []

    engine = JSEngine()

    for i, filepath in enumerate(result_files):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            continue

        if isinstance(result, dict) and result.get("status") == "running":
            continue
        if "action_sequence" not in result:
            continue

        game_name = result.get("game", "")
        model = result.get("model", "unknown")
        jax_level = result.get("level", 0)
        original_steps = result.get("state_data", {}).get("step", len(result["action_sequence"]))

        if not game_name:
            continue

        # Load game text
        if game_name not in game_text_cache:
            game_text_cache[game_name] = find_game_text(game_name)
        game_text = game_text_cache[game_name]
        if game_text is None:
            continue

        # Build level map (handles message levels)
        if game_name not in level_map_cache:
            level_map_cache[game_name] = probe_js_level_map(game_text)
            lm = level_map_cache[game_name]
            if lm and verbose:
                has_offset = any(jax_i != js_i for jax_i, js_i in lm.items())
                if has_offset:
                    print(f"  Level map for '{game_name}': {dict(list(lm.items())[:6])}{'...' if len(lm) > 6 else ''}")

        level_map = level_map_cache[game_name]
        js_level = level_map.get(jax_level, jax_level)

        # Respawn engine if dead
        if not engine.alive:
            engine = JSEngine()

        replay = replay_one(engine, game_text, js_level,
                            result["action_sequence"], result.get("win", False))

        if not engine.alive:
            engine = JSEngine()

        row = {
            "llm": model,
            "game": game_name,
            "level": jax_level,
            "original_win": bool(result.get("win", False)),
            "js_win": replay["js_win"],
            "original_steps": original_steps,
            "js_steps": replay["js_steps"],
            "error": replay["error"],
        }
        rows.append(row)

        if verbose:
            status = "MATCH" if row["original_win"] == row["js_win"] else "MISMATCH"
            orig = "W" if row["original_win"] else "L"
            js = "W" if row["js_win"] else "L"
            lvl_note = f" (jax={jax_level}->js={js_level})" if jax_level != js_level else ""
            err = f" ERR: {replay['error']}" if replay["error"] else ""
            print(f"  [{i+1}/{len(result_files)}] {status} {model}/{game_name}/L{jax_level}{lvl_note}: "
                  f"orig={orig} js={js} steps={replay['js_steps']}{err}")
        elif (i + 1) % 50 == 0:
            print(f"  ... processed {i+1}/{len(result_files)} files")

    engine.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting (mirrors plot_llm_results.py heatmaps)
# ---------------------------------------------------------------------------

def _format_game_name(game_name: str) -> str:
    from puzzlescript_jax.utils import game_names_remap
    game_name = game_names_remap.get(game_name, game_name)
    return ' '.join(word.capitalize() for word in game_name.replace('_', ' ').split())


def generate_plots(df: pd.DataFrame, output_dir: str, dataset: str = "priority") -> None:
    """Generate the same heatmap suite as plot_llm_results.py, side-by-side for
    original (JAX) and JS-corrected win data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from plot_llm_results import (
        get_search_game_order,
        build_expanded_heatmap_columns,
        draw_game_dividers,
        format_game_name_for_display,
    )
    from puzzlescript_jax.utils import get_list_of_games_for_testing

    os.makedirs(output_dir, exist_ok=True)

    # Filter out thinking models consistent with plot_llm_results.py
    df = df[~df["llm"].isin(["deepseek-r1", "llama"])].copy()

    # Normalize game names for display
    df["game"] = df["game"].apply(lambda x: "atlas_shrank" if "atlas" in x and "shrank" in x else x)
    df["game"] = df["game"].apply(format_game_name_for_display)

    # Filter to dataset
    dataset_games_display = set(format_game_name_for_display(g) for g in get_list_of_games_for_testing(dataset))
    df = df[df["game"].isin(dataset_games_display)]

    if df.empty:
        print("No data after filtering to dataset. Skipping plots.")
        return

    # Rename LLMs
    llm_name_mapping = {
        "4o-mini": "GPT 4o-mini",
        "deepseek": "Deepseek-chat",
        "qwen": "Qwen-plus",
        "gemini": "Gemini 2.0 flash exp",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
    }
    df["llm"] = df["llm"].replace(llm_name_mapping)

    search_order = get_search_game_order(dataset=dataset)

    # We produce two sets: original_win and js_win
    for win_col, label, suffix in [
        ("original_win", "Original (JAX)", "original"),
        ("js_win", "JS-Validated", "js_validated"),
    ]:
        _plot_heatmap_suite(df, win_col, label, suffix, output_dir, search_order, plt, sns,
                            build_expanded_heatmap_columns, draw_game_dividers)

    print(f"\nAll plots saved to {output_dir}")


def _plot_heatmap_suite(df, win_col, label, suffix, output_dir, search_order,
                        plt, sns, build_expanded_heatmap_columns, draw_game_dividers):
    """Generate per-LLM, LLM-vs-game, and expanded heatmaps for a given win column."""
    target_cell_height = 0.7
    target_cell_width = 1.0
    min_fig_w = 8.0

    # --- Heatmap 1: Average win rate per LLM ---
    llm_agg = df.groupby("llm").agg(win_rate=(win_col, "mean")).reset_index()

    heatmap_data = llm_agg.set_index("llm")[["win_rate"]]
    annot = [[f"{v:.0%}" if pd.notnull(v) else "" for v in row]
             for row in heatmap_data.to_numpy()]

    fig_h = len(heatmap_data) * target_cell_height + 2.0
    fig_w = max(1 * target_cell_width + 3.0, min_fig_w)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Average Win Rate", "shrink": 0.8, "pad": 0.01},
                annot_kws={"size": 10})
    plt.title(f"Average Win Rate per LLM ({label})")
    plt.ylabel("LLM Model")
    plt.xticks([])
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(output_dir, f"llm_win_rate_heatmap_{suffix}.png"))
    plt.close()
    print(f"  Saved llm_win_rate_heatmap_{suffix}.png")

    # --- Heatmap 2: LLM vs Game win rate ---
    llm_game_agg = df.groupby(["llm", "game"]).agg(win_rate=(win_col, "mean"))
    if llm_game_agg.empty:
        return
    pivot = llm_game_agg["win_rate"].unstack()

    present_games = set(pivot.columns)
    game_order = [g for g in search_order if g in present_games]
    extras = [g for g in pivot.columns if g not in game_order]
    game_order.extend(sorted(extras))
    pivot = pivot.reindex(columns=game_order)

    annot = [[f"{v:.0%}" if pd.notnull(v) else "" for v in row]
             for row in pivot.to_numpy()]

    fig_h = len(pivot.index) * target_cell_height + 2.5
    fig_w = max(len(pivot.columns) * target_cell_width + 3.0, min_fig_w)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(pivot,
                annot=pd.DataFrame(annot, index=pivot.index, columns=pivot.columns),
                fmt="", cmap="RdYlGn", vmin=0, vmax=1,
                cbar_kws={"label": "Average Win Rate", "shrink": 0.8, "pad": 0.01},
                annot_kws={"size": 9})
    plt.title(f"Average Win Rate: LLM vs. Game ({label})")
    plt.xlabel("Game", labelpad=10)
    plt.ylabel("LLM Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout(pad=1.2, rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(output_dir, f"llm_vs_game_win_rate_heatmap_{suffix}.png"))
    plt.close()
    print(f"  Saved llm_vs_game_win_rate_heatmap_{suffix}.png")

    # --- Heatmap 3: Expanded LLM vs Level across games ---
    expected_levels_by_game = {
        game: sorted(df.loc[df["game"] == game, "level"].dropna().astype(int).unique().tolist())
        for game in game_order
    }
    columns, game_spans = build_expanded_heatmap_columns(game_order, expected_levels_by_game)
    if not columns:
        return

    llm_order = list(dict.fromkeys(df["llm"]))
    column_keys = [f"{game}::level-{level}" for game, level in columns]
    heatmap_df = pd.DataFrame(index=llm_order, columns=column_keys, dtype=float)

    grouped = df.groupby(["llm", "game", "level"])[win_col].mean().reset_index()
    value_lookup = {
        (row.llm, row.game, int(row.level)): float(row[win_col])
        for _, row in grouped.iterrows()
    }
    for llm in llm_order:
        for game, level in columns:
            value = value_lookup.get((llm, game, level))
            if value is not None:
                heatmap_df.at[llm, f"{game}::level-{level}"] = value

    annot_data = None
    if len(columns) <= 40:
        annot_data = [
            [f"{val:.0%}" if pd.notnull(val) else "" for val in row]
            for row in heatmap_df.to_numpy(dtype=float)
        ]

    fig_w = max(len(columns) * 0.35 + 3.0, 12.0)
    fig_h = max(len(llm_order) * 0.8 + 2.5, 3.0)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        heatmap_df,
        annot=annot_data if annot_data is not None else False,
        fmt="",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Average Win Rate", "shrink": 0.8, "pad": 0.01},
        annot_kws={"size": 7},
        linewidths=0.25,
        linecolor="white",
    )
    draw_game_dividers(ax, game_spans, len(llm_order))
    ax.set_title(f"Average Win Rate: LLM vs. Level Across Games ({label})", pad=32)
    plt.xlabel("Level", labelpad=10)
    plt.ylabel("LLM Model")
    plt.yticks(rotation=0)
    ax.set_xticklabels([str(level) for _, level in columns], rotation=0, fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, f"llm_vs_game_win_rate_expanded_heatmap_{suffix}.png"))
    plt.close()
    print(f"  Saved llm_vs_game_win_rate_expanded_heatmap_{suffix}.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print comparison tables from the replay DataFrame."""
    stats = defaultdict(lambda: {"original_wins": 0, "js_wins": 0, "total": 0,
                                  "agree": 0, "disagree": 0, "errors": 0})
    for _, row in df.iterrows():
        key = (row["llm"], row["game"], row["level"])
        s = stats[key]
        s["total"] += 1
        if row["original_win"]:
            s["original_wins"] += 1
        if row["js_win"]:
            s["js_wins"] += 1
        if row["error"]:
            s["errors"] += 1
        if row["original_win"] == row["js_win"]:
            s["agree"] += 1
        else:
            s["disagree"] += 1

    print("\n" + "=" * 100)
    print(f"{'Model':<20} {'Game':<25} {'Lvl':>3} {'N':>4} {'Orig Win%':>10} {'JS Win%':>10} {'Match%':>8} {'Err':>4}")
    print("-" * 100)

    model_game_stats = defaultdict(lambda: {"original_wins": 0, "js_wins": 0, "total": 0,
                                             "agree": 0, "disagree": 0, "errors": 0})

    for (model, game, level), s in sorted(stats.items()):
        if s["total"] == 0:
            continue
        orig_wr = s["original_wins"] / s["total"] * 100
        js_wr = s["js_wins"] / s["total"] * 100
        match_pct = s["agree"] / s["total"] * 100
        print(f"{model:<20} {game:<25} {level:>3} {s['total']:>4} "
              f"{orig_wr:>9.1f}% {js_wr:>9.1f}% {match_pct:>7.1f}% {s['errors']:>4}")

        mg = model_game_stats[(model, game)]
        for k in ("original_wins", "js_wins", "total", "agree", "disagree", "errors"):
            mg[k] += s[k]

    print("\n" + "=" * 100)
    print(f"{'Model':<20} {'Game':<25} {'N':>4} {'Orig Win%':>10} {'JS Win%':>10} {'Match%':>8}")
    print("-" * 100)
    for (model, game), s in sorted(model_game_stats.items()):
        if s["total"] == 0:
            continue
        orig_wr = s["original_wins"] / s["total"] * 100
        js_wr = s["js_wins"] / s["total"] * 100
        match_pct = s["agree"] / s["total"] * 100
        print(f"{model:<20} {game:<25} {s['total']:>4} {orig_wr:>9.1f}% {js_wr:>9.1f}% {match_pct:>7.1f}%")

    grand = {"original_wins": 0, "js_wins": 0, "total": 0, "agree": 0, "disagree": 0, "errors": 0}
    for s in stats.values():
        for k in grand:
            grand[k] += s[k]

    if grand["total"] > 0:
        print(f"\nGrand total: {grand['total']} runs, "
              f"original win rate {grand['original_wins']/grand['total']*100:.1f}%, "
              f"JS win rate {grand['js_wins']/grand['total']*100:.1f}%, "
              f"agreement {grand['agree']/grand['total']*100:.1f}%, "
              f"errors {grand['errors']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay JAX LLM agent results in the JS PuzzleScript engine")
    parser.add_argument("--results_dir", type=str, default="llm_agent_results",
                        help="Root results directory (default: llm_agent_results)")
    parser.add_argument("--game", type=str, default="",
                        help="Filter to a specific game name (default: all games)")
    parser.add_argument("--model", type=str, default="",
                        help="Filter to a specific model folder (default: all models)")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max number of files to replay (0=unlimited)")
    parser.add_argument("--verbose", action="store_true", help="Print per-file results")
    parser.add_argument("--no_remap", action="store_true",
                        help="Don't remap actions (replay raw JAX action IDs in JS engine)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate heatmap plots (original + JS-validated)")
    parser.add_argument("--dataset", type=str, default="priority",
                        help="Dataset for game filtering/ordering in plots (default: priority)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Plot output directory (default: <results_dir>/analysis_js_replay)")
    parser.add_argument("--csv", type=str, default="",
                        help="Save replay results to a CSV file")
    args = parser.parse_args()

    if args.no_remap:
        global JAX_TO_JS
        JAX_TO_JS = {i: i for i in range(5)}
        print("WARNING: --no_remap mode: sending raw JAX action IDs to JS engine (no translation)\n")

    result_files = collect_result_files(
        args.results_dir,
        game_filter=args.game or None,
        model_filter=args.model or None,
    )

    if args.max_files > 0:
        result_files = result_files[:args.max_files]

    print(f"Found {len(result_files)} result files to replay")
    print(f"Action remapping (JAX->JS): {JAX_TO_JS}\n")

    df = replay_all(result_files, verbose=args.verbose)

    if df.empty:
        print("No results to report.")
        return

    print_summary(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nReplay results saved to {args.csv}")

    if args.plot:
        output_dir = args.output_dir or os.path.join(args.results_dir, "analysis_js_replay")
        generate_plots(df, output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
