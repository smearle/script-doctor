"""Plot an evolution run: 3 rows (early/mid/late champion) x 5 solution frames.

Reads the result JSON from an evolution run directory, picks 3 champion
generations evenly spaced through the run, replays their solutions, and
composites selected frames into a single figure.

Works with the C++, NodeJS, and JAX backends.

Usage:
    python plot_evolution.py --run_dir data/evolved_levels_cpp/Atlas_Shrank/<run>
    python plot_evolution.py --run_dir data/evolved_levels_cpp/Atlas_Shrank/<run> --n_rows 4 --n_cols 6
    python plot_evolution.py --run_dir data/evolved_levels_nodejs/Atlas_Shrank/<run> --backend nodejs
    python plot_evolution.py --run_dir data/evolved_levels/sokoban_basic/<run> --backend jax
"""

import argparse
import json
import os
import pickle
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

JAXTAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JAXtar")
if JAXTAR_DIR not in sys.path:
    sys.path.insert(0, JAXTAR_DIR)

MAX_AGAIN = 100


# ============================================================================
# Backend-agnostic replay
# ============================================================================

def _replay_frames_cpp(game: str, level_i: int, champion_dat: list,
                       actions: list[int], scale: int) -> list[np.ndarray]:
    """Replay actions on a C++ engine and return per-step frames."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_cpp._puzzlescript_cpp import LevelBackup
    from puzzlescript_jax.utils import init_ps_lark_parser

    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    backend.compile_game(parser, game)
    backend.cpp_engine.load_level(level_i)

    backup = LevelBackup(champion_dat,
                         backend.cpp_engine.width,
                         backend.cpp_engine.height)
    backend.cpp_engine._engine.restore_level(backup)
    renderer = backend._ensure_renderer()
    renderer.reset_viewport(backend.cpp_engine.width, backend.cpp_engine.height)

    frames = [backend.render_frame()]
    for action in actions:
        backend.process_input(int(action))
        again_steps = 0
        while backend.againing and again_steps < MAX_AGAIN:
            backend.process_input(-1)
            again_steps += 1
        frames.append(backend.render_frame())

    if scale > 1:
        frames = [
            np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
            for f in frames
        ]
    return frames


def _replay_frames_nodejs(game: str, level_i: int,
                          champion_game_text: str,
                          actions: list[int],
                          scale: int) -> list[np.ndarray]:
    """Replay actions via the NodeJS backend and return per-step frames."""
    from backends import NodeJSPuzzleScriptBackend

    backend = NodeJSPuzzleScriptBackend()
    backend.load_level(champion_game_text, level_i)

    frames = [backend.render_frame()]
    for action in actions:
        backend.process_input(int(action))
        again_steps = 0
        while backend.againing and again_steps < MAX_AGAIN:
            backend.process_input(-1)
            again_steps += 1
        frames.append(backend.render_frame())

    if scale > 1:
        frames = [
            np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
            for f in frames
        ]
    return frames


def _replay_frames_jax(game: str, level_i: int,
                       champion_multihot: np.ndarray,
                       actions: list[int],
                       scale: int) -> list[np.ndarray]:
    """Replay actions via the JAX/PuxleEnv backend and return per-step frames."""
    import jax
    import jax.numpy as jnp
    from puzzlescript_jax.wrappers import PuzzleJaxPuxleEnv, PuzzleJaxHeuristic
    from puzzlescript_jax.env import PJParams
    from helpers.visualization import build_path_steps_from_actions

    puzzle = PuzzleJaxPuxleEnv(game=game, level_i=level_i)
    heuristic = PuzzleJaxHeuristic(puzzle)
    env = puzzle.env

    params = PJParams(level=jnp.array(champion_multihot))
    _, pj_state = env.reset(jax.random.PRNGKey(0), params)
    init_state = puzzle._pj_to_state(pj_state)
    solve_config = puzzle.get_solve_config()

    path_steps = build_path_steps_from_actions(
        puzzle=puzzle,
        solve_config=solve_config,
        initial_state=init_state,
        actions=actions,
        heuristic=heuristic,
    )

    path_states = [step.state for step in path_steps]
    frames = []
    for idx, step in enumerate(path_steps):
        img = step.state.img(idx=idx, path=path_states, solve_config=solve_config)
        frames.append(np.asarray(img))

    if scale > 1:
        frames = [
            np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
            for f in frames
        ]
    return frames


# ============================================================================
# Load champion data from checkpoint
# ============================================================================

def _load_champion_at_gen(run_dir: str, target_gen: int, backend: str) -> dict:
    """Load champion state at a specific generation from the history.

    Returns dict with keys needed for replay: actions, gen, fitness, cost,
    and backend-specific state (champion_dat / champion_game_text / champion_multihot).
    """
    result_files = [f for f in os.listdir(run_dir) if f.endswith("_result.json")]
    if not result_files:
        raise FileNotFoundError(f"No result JSON in {run_dir}")
    with open(os.path.join(run_dir, result_files[0])) as f:
        result_data = json.load(f)

    history = result_data["history"]

    # Find the champion state at the target generation: the most recent
    # improvement at or before target_gen.
    best_entry = None
    best_fitness = -float("inf")
    for entry in history:
        if entry["gen"] <= target_gen and entry["fitness"] > best_fitness:
            best_fitness = entry["fitness"]
            best_entry = entry

    if best_entry is None:
        best_entry = history[0]

    return best_entry


def _load_checkpoint_champion_dat(run_dir: str) -> Optional[list]:
    """Load the final champion_dat from checkpoint.pkl (C++ backend)."""
    ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
    if not os.path.isfile(ckpt_path):
        return None
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt.get("champion_dat")


# ============================================================================
# Reconstruct champion states by re-running evolution mutations
# ============================================================================

def _reconstruct_champion_states_cpp(
    game: str, level_i: int, run_dir: str, history: list, target_gens: list[int],
) -> dict[int, list]:
    """Re-run the evolution to reconstruct champion dat at specific generations.

    Returns {gen: dat} for each target generation.
    """
    import json as json_mod
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_cpp._puzzlescript_cpp import LevelBackup, solve_astar, solve_bfs
    from puzzlescript_jax.utils import init_ps_lark_parser
    from evolve_level_cpp import LevelMutator, extract_tile_patterns, _to_i32

    # Parse run directory name for hyperparameters
    run_name = os.path.basename(run_dir)
    ckpt_path = os.path.join(run_dir, "checkpoint.pkl")

    # Load checkpoint to get the RNG seed and parameters
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    # We can't easily reconstruct intermediate states from the checkpoint alone.
    # Instead, we'll use the GIF files that already exist for champion generations.
    return {}


def _get_champion_improvement_gens(history: list) -> list[int]:
    """Return the generation numbers where the champion improved."""
    improvements = []
    best_fitness = -float("inf")
    for entry in history:
        if entry["fitness"] > best_fitness:
            best_fitness = entry["fitness"]
            improvements.append(entry["gen"])
    return improvements


# ============================================================================
# Main figure generation
# ============================================================================

def _composite_ghost_trail(frames_all: list[np.ndarray], current_idx: int,
                           n_ghosts: int = 3) -> np.ndarray:
    """Overlay previous frames as fading ghosts behind the current frame.

    For each ghost frame, we compute the pixels that *disappeared* between
    that frame and its successor (i.e. where something was but then moved
    away).  Only those delta pixels are blended onto the current frame, so
    static scenery and the current player position stay fully opaque.
    """
    current = frames_all[current_idx].astype(np.float32)
    if current.ndim == 3 and current.shape[2] == 4:
        current = current[:, :, :3]

    ghost_indices = list(range(max(0, current_idx - n_ghosts), current_idx))
    if not ghost_indices:
        return current.astype(np.uint8)

    canvas = current.copy()
    for i, gi in enumerate(ghost_indices):
        ghost = frames_all[gi].astype(np.float32)
        if ghost.ndim == 3 and ghost.shape[2] == 4:
            ghost = ghost[:, :, :3]
        if ghost.shape != current.shape:
            continue

        successor = frames_all[gi + 1].astype(np.float32)
        if successor.ndim == 3 and successor.shape[2] == 4:
            successor = successor[:, :, :3]

        # Pixels that changed between this ghost and its successor —
        # these are the "departures" we want to show as afterimages.
        delta_mask = np.any(ghost != successor, axis=-1, keepdims=True)

        # Also exclude pixels that are *already* occupied in the current
        # frame identically to the ghost (object didn't actually leave).
        still_there = np.all(ghost == current, axis=-1, keepdims=True)
        mask = delta_mask & ~still_there

        # Older ghosts are more transparent
        alpha = 0.12 + 0.08 * (i + 1)
        blend = canvas * (1 - alpha) + ghost * alpha
        canvas = np.where(mask, blend, canvas)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _select_evenly_spaced(items: list, n: int) -> tuple[list, list[int]]:
    """Select n items evenly spaced from a list, always including first and last.

    Returns (selected_items, selected_indices).
    """
    if len(items) <= n:
        return list(items), list(range(len(items)))
    if n == 1:
        return [items[-1]], [len(items) - 1]
    indices = [int(round(i * (len(items) - 1) / (n - 1))) for i in range(n)]
    return [items[idx] for idx in indices], indices


def replay_champion_frames(
    game: str, level_i: int, run_dir: str,
    gen: int, actions: list[int], backend: str,
    scale: int = 1,
) -> list[np.ndarray]:
    """Get rendered frames for a champion at a given generation.

    First tries to read frames from the existing GIF file. Falls back to
    re-rendering via the appropriate backend.
    """
    # Try reading from existing GIF
    gif_path = os.path.join(run_dir, f"evolved_lv{level_i}_gen{gen}.gif")
    if os.path.isfile(gif_path):
        import imageio
        reader = imageio.get_reader(gif_path)
        frames = [np.asarray(frame) for frame in reader]
        reader.close()
        if scale > 1:
            frames = [
                np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
                for f in frames
            ]
        return frames

    # Fall back: re-render using the backend
    result_files = [f for f in os.listdir(run_dir) if f.endswith("_result.json")]
    with open(os.path.join(run_dir, result_files[0])) as f:
        result_data = json.load(f)

    if backend == "cpp":
        ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        return _replay_frames_cpp(game, level_i, ckpt["champion_dat"],
                                  actions, scale)
    elif backend == "nodejs":
        game_text_path = os.path.join(
            run_dir, f"evolved_lv{level_i}_game.txt")
        with open(game_text_path, "r") as f:
            game_text = f.read()
        return _replay_frames_nodejs(game, level_i, game_text, actions, scale)
    elif backend == "jax":
        multihot_path = os.path.join(
            run_dir, f"evolved_lv{level_i}_multihot.npy")
        multihot = np.load(multihot_path)
        return _replay_frames_jax(game, level_i, multihot, actions, scale)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def plot_evolution(
    run_dir: str,
    backend: str = "cpp",
    n_rows: int = 3,
    n_cols: int = 5,
    scale: int = 10,
    out_path: Optional[str] = None,
    dpi: int = 150,
):
    """Generate an evolution progress figure.

    Args:
        run_dir: Path to the evolution run directory.
        backend: One of "cpp", "nodejs", "jax".
        n_rows: Number of champion generations to show (rows).
        n_cols: Number of solution frames per row.
        scale: Pixel scaling for rendered frames.
        out_path: Output image path. Defaults to <run_dir>/evolution_progress.png.
        dpi: Figure DPI.
    """
    # Load result JSON
    result_files = [f for f in os.listdir(run_dir) if f.endswith("_result.json")]
    if not result_files:
        raise FileNotFoundError(f"No result JSON found in {run_dir}")
    with open(os.path.join(run_dir, result_files[0])) as f:
        result_data = json.load(f)

    game = result_data["game"]
    level_i = result_data["level_i"]
    history = result_data["history"]

    # Find champion improvement generations
    improvement_gens = _get_champion_improvement_gens(history)
    print(f"Champion improvements at gens: {improvement_gens}")

    # Build a lookup: gen -> history entry
    best_at_gen = {}
    best_fitness = -float("inf")
    for entry in history:
        if entry["fitness"] > best_fitness:
            best_fitness = entry["fitness"]
            best_at_gen[entry["gen"]] = entry

    # Select n_rows evenly spaced champion generations
    selected_gens, _ = _select_evenly_spaced(improvement_gens, n_rows)
    print(f"Selected generations for plot: {selected_gens}")

    # Collect frames for each selected generation.
    # We separate the initial state (frame 0) from the play-trace frames.
    # rows_data: list of (gen, fitness, cost, init_frame, trace_frames, trace_timesteps)
    rows_data = []
    for gen in selected_gens:
        entry = best_at_gen[gen]
        actions = entry.get("actions", [])
        fitness = entry.get("fitness", 0)
        cost = entry.get("cost", "?")

        if not actions:
            print(f"  Gen {gen}: no actions, skipping")
            continue

        frames = replay_champion_frames(
            game, level_i, run_dir, gen, actions, backend, scale=scale,
        )
        if not frames:
            print(f"  Gen {gen}: no frames rendered, skipping")
            continue

        init_frame = frames[0]
        # Select n_cols trace frames from frames[1:] (post-action states)
        trace_frames_all = frames[1:]
        if trace_frames_all:
            _, trace_indices = _select_evenly_spaced(
                trace_frames_all, n_cols)
            # Apply ghost trail compositing using indices into the full
            # trace_frames_all list (so ghosts come from *all* intermediate
            # frames, not just the selected ones).
            trace_frames = [
                _composite_ghost_trail(trace_frames_all, idx, n_ghosts=3)
                for idx in trace_indices
            ]
            # timesteps are 1-indexed (action 1, action 2, ...)
            trace_timesteps = [idx + 1 for idx in trace_indices]
        else:
            trace_frames, trace_timesteps = [], []

        rows_data.append((gen, fitness, cost, init_frame,
                          trace_frames, trace_timesteps))
        print(f"  Gen {gen}: fitness={fitness}, cost={cost}, "
              f"{len(frames)} frames -> 1 init + {len(trace_frames)} trace")

    if not rows_data:
        print("No data to plot.")
        return

    # Build figure with gridspec: 1 init column | gap | n_cols trace columns
    actual_rows = len(rows_data)
    trace_cols = max(len(rd[4]) for rd in rows_data)
    total_cols = 1 + trace_cols  # init + trace

    sample_frame = rows_data[0][3]
    fh, fw = sample_frame.shape[:2]
    aspect = fw / fh
    cell_h_in = 1.8
    cell_w_in = cell_h_in * aspect
    gap_in = 0.25  # gap between init and trace columns

    fig_w = cell_w_in * total_cols + gap_in + 1.0
    fig_h = cell_h_in * actual_rows + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        actual_rows, total_cols,
        wspace=0.05, hspace=0.15,
        width_ratios=[1] + [1] * trace_cols,
    )

    for row_i, (gen, fitness, cost, init_frame,
                trace_frames, trace_timesteps) in enumerate(rows_data):
        gen_label = "initial" if gen == -1 else f"gen {gen}"

        # ---- Init column (column 0) ----
        ax_init = fig.add_subplot(gs[row_i, 0])
        frame = init_frame
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        ax_init.imshow(frame, interpolation="nearest", aspect="equal")
        ax_init.set_xticks([])
        ax_init.set_yticks([])
        # Visible border to delineate the initial level
        for spine in ax_init.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(1.5)
        ax_init.set_ylabel(
            f"{gen_label}\ncost={cost}",
            fontsize=9, rotation=0, labelpad=55, va="center",
        )
        if row_i == 0:
            ax_init.set_title("level", fontsize=9, fontweight="bold")

        # ---- Trace columns (columns 1..trace_cols) ----
        for col_i in range(trace_cols):
            ax = fig.add_subplot(gs[row_i, 1 + col_i])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_i < len(trace_frames):
                frame = trace_frames[col_i]
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                ax.imshow(frame, interpolation="nearest", aspect="equal")
                t = trace_timesteps[col_i]
                ax.set_xlabel(f"t={t}", fontsize=8)
            else:
                ax.axis("off")

            if row_i == 0 and col_i == 0:
                ax.set_title("solution trace", fontsize=9,
                             fontweight="bold", loc="left")

    fig.suptitle(f"{game} level {level_i} — evolution progress",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.03)

    if out_path is None:
        out_path = os.path.join(run_dir, "evolution_progress.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution progress: rows of champion generations, "
                    "columns of solution frames.",
    )
    parser.add_argument("--run_dir", required=True,
                        help="Path to an evolution run directory")
    parser.add_argument("--backend", default=None,
                        choices=["cpp", "nodejs", "jax"],
                        help="Backend used for the run (auto-detected from path if omitted)")
    parser.add_argument("--n_rows", type=int, default=3,
                        help="Number of champion generations to show")
    parser.add_argument("--n_cols", type=int, default=4,
                        help="Number of solution frames per row")
    parser.add_argument("--scale", type=int, default=10,
                        help="Pixel upscaling factor for rendered frames")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <run_dir>/evolution_progress.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Auto-detect backend from path
    backend = args.backend
    if backend is None:
        if "evolved_levels_cpp" in args.run_dir:
            backend = "cpp"
        elif "evolved_levels_nodejs" in args.run_dir:
            backend = "nodejs"
        elif "evolved_levels" in args.run_dir:
            backend = "jax"
        else:
            backend = "cpp"
        print(f"Auto-detected backend: {backend}")

    plot_evolution(
        run_dir=args.run_dir,
        backend=backend,
        n_rows=args.n_rows,
        n_cols=args.n_cols,
        scale=args.scale,
        out_path=args.out,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
