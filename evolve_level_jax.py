"""Evolve a PuzzleScript level to maximise A* search depth.

Given a game and an initial level, applies a (1+λ) evolutionary strategy that
mutates the tile layout and selects for levels that are still solvable but
induce the longest optimal solution (highest A* cost).

Runs are **resumable**: re-running the same command with more --gens will
pick up where the last run left off (checkpoint is saved in the output dir).

Usage examples:
    # Evolve from sokoban_basic level 0
    python evolve_level.py game=sokoban_basic level=0

    # Larger population, more generations
    python evolve_level.py game=sokoban_basic level=0 pop=8 gens=50

    # Resume the same run for 50 more generations
    python evolve_level.py game=sokoban_basic level=0 pop=8 gens=100

    # Use more search budget per evaluation
    python evolve_level.py game=sokoban_basic level=0 max_nodes=2000000

    # Save GIF of each new champion
    python evolve_level.py game=sokoban_basic level=0 render_gif=true

    # Allow the number of players to change during evolution
    python evolve_level.py game=sokoban_basic level=0 allow_player_change=true

Results are saved to data/evolved_levels/<GAME>/<run-subdir>/
"""

import copy
import json
import math
import os
import pickle
import sys
import time
import traceback
from typing import List, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
JAXTAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JAXtar")
if JAXTAR_DIR not in sys.path:
    sys.path.insert(0, JAXTAR_DIR)

from puzzlescript_jax.globals import DATA_DIR
from puzzlescript_jax.utils import init_ps_lark_parser, get_tree_from_txt
from puzzlescript_jax.env import PuzzleJaxEnv, PJParams, PJState
from puzzlescript_jax.wrappers import PuzzleJaxPuxleEnv, PuzzleJaxHeuristic

from JAxtar.stars.astar import astar_builder
from JAxtar.stars.search_base import SearchResult
from conf.config import EvolveLevelConfig
from helpers.visualization import PathStep, build_path_steps_from_actions

EVOLVED_DIR = os.path.join(DATA_DIR, "evolved_levels")


# ============================================================================
# Level helpers
# ============================================================================

def get_char_level(env: PuzzleJaxEnv, level_i: int) -> np.ndarray:
    """Return the character-grid for *level_i* as a 2-D numpy array of str."""
    raw = env.levels[level_i][0]  # list of lists of single-char strings
    return np.array(raw)


def char_level_to_multihot(env: PuzzleJaxEnv, char_level: np.ndarray) -> jnp.ndarray:
    """Convert a 2-D char array back to a multihot JAX array."""
    return jnp.array(env.char_level_to_multihot(char_level))


def multihot_to_char_level(env: PuzzleJaxEnv, multihot: np.ndarray) -> np.ndarray:
    """Convert a multihot level [n_objs, H, W] back to a 2-D char grid.

    For each cell we find the matching character in env.chars_to_idxs by
    looking up the object index of the *non-background* object present.
    Falls back to background character if nothing else is there.
    """
    # Build reverse maps
    idx_to_char = {}
    for ch, idx in env.chars_to_idxs.items():
        if idx not in idx_to_char:
            idx_to_char[idx] = ch

    bg_idx = env.objs_to_idxs.get("background", 0)
    bg_char = idx_to_char.get(bg_idx, ".")

    n_objs = env.n_objs
    multihot_np = np.asarray(multihot[:n_objs])  # [n_objs, H, W]
    H, W = multihot_np.shape[1], multihot_np.shape[2]
    char_level = np.full((H, W), bg_char, dtype="<U1")

    # For each cell, look at which atomic objects are present.  Try to find
    # a matching joint/compound character first; fall back to single objects.
    obj_vecs = env.obj_vecs  # includes joint objects as extra rows
    vec_to_char = {}
    for ch, idx in env.chars_to_idxs.items():
        if idx < len(obj_vecs):
            key = tuple(obj_vecs[idx][:n_objs].astype(int))
            vec_to_char[key] = ch

    for h in range(H):
        for w in range(W):
            cell_vec = tuple(multihot_np[:, h, w].astype(int))
            if cell_vec in vec_to_char:
                char_level[h, w] = vec_to_char[cell_vec]
            else:
                # Fall back: pick the first non-background object present
                for obj_i in range(n_objs):
                    if multihot_np[obj_i, h, w] and obj_i != bg_idx:
                        if obj_i in idx_to_char:
                            char_level[h, w] = idx_to_char[obj_i]
                            break
    return char_level


def get_placeable_chars(env: PuzzleJaxEnv) -> List[str]:
    """Return the list of legend characters that can appear in a level."""
    return list(env.chars_to_idxs.keys())


def get_layer_for_obj(env: PuzzleJaxEnv, obj_idx: int) -> int:
    """Return the collision layer index for an atomic object index."""
    for layer_i, mask in enumerate(env.layer_masks):
        if mask[obj_idx]:
            return layer_i
    return -1


# ============================================================================
# Mutations
# ============================================================================

def _count_players(env: PuzzleJaxEnv, multihot: np.ndarray) -> int:
    """Count the number of cells that contain a player object."""
    player_idxs = env.player_idxs
    player_present = np.zeros(multihot.shape[1:], dtype=bool)
    for pi in player_idxs:
        player_present |= np.asarray(multihot[pi]).astype(bool)
    return int(player_present.sum())


def mutate_swap(env: PuzzleJaxEnv, multihot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap the contents of two random cells."""
    H, W = multihot.shape[1], multihot.shape[2]
    h1, w1 = rng.integers(H), rng.integers(W)
    h2, w2 = rng.integers(H), rng.integers(W)
    if (h1, w1) == (h2, w2):
        return multihot
    m = multihot.copy()
    tmp = m[:, h1, w1].copy()
    m[:, h1, w1] = m[:, h2, w2]
    m[:, h2, w2] = tmp
    return m


def mutate_place(env: PuzzleJaxEnv, multihot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place a random non-background object at a random cell (respecting collision layers)."""
    n_objs = env.n_objs
    H, W = multihot.shape[1], multihot.shape[2]
    bg_idx = env.objs_to_idxs.get("background", 0)

    # Pick a random atomic non-background object
    candidates = [i for i in range(n_objs) if i != bg_idx]
    if not candidates:
        return multihot
    obj_idx = rng.choice(candidates)
    layer_i = get_layer_for_obj(env, obj_idx)

    h, w = rng.integers(H), rng.integers(W)
    m = multihot.copy()
    # Clear other objects on the same collision layer at this cell
    if layer_i >= 0:
        layer_mask = env.layer_masks[layer_i]
        for oi in range(n_objs):
            if layer_mask[oi]:
                m[oi, h, w] = False
    m[obj_idx, h, w] = True
    return m


def mutate_remove(env: PuzzleJaxEnv, multihot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remove a random non-background, non-player object from a random cell."""
    n_objs = env.n_objs
    H, W = multihot.shape[1], multihot.shape[2]
    bg_idx = env.objs_to_idxs.get("background", 0)
    player_set = set(int(p) for p in env.player_idxs)

    # Collect all (obj, h, w) where the object is present and removable
    candidates = []
    m_np = np.asarray(multihot[:n_objs])
    for oi in range(n_objs):
        if oi == bg_idx or oi in player_set:
            continue
        hs, ws = np.where(m_np[oi])
        for h, w in zip(hs, ws):
            candidates.append((oi, int(h), int(w)))
    if not candidates:
        return multihot
    oi, h, w = candidates[rng.integers(len(candidates))]
    m = multihot.copy()
    m[oi, h, w] = False
    return m


def mutate_move(env: PuzzleJaxEnv, multihot: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Move a random non-background object to a different cell."""
    n_objs = env.n_objs
    H, W = multihot.shape[1], multihot.shape[2]
    bg_idx = env.objs_to_idxs.get("background", 0)

    m_np = np.asarray(multihot[:n_objs])
    candidates = []
    for oi in range(n_objs):
        if oi == bg_idx:
            continue
        hs, ws = np.where(m_np[oi])
        for h, w in zip(hs, ws):
            candidates.append((oi, int(h), int(w)))
    if not candidates:
        return multihot

    oi, h_src, w_src = candidates[rng.integers(len(candidates))]
    h_dst, w_dst = rng.integers(H), rng.integers(W)
    if (h_src, w_src) == (h_dst, w_dst):
        return multihot

    m = multihot.copy()
    # Remove from source
    m[oi, h_src, w_src] = False
    # Clear same collision layer at destination
    layer_i = get_layer_for_obj(env, oi)
    if layer_i >= 0:
        layer_mask = env.layer_masks[layer_i]
        for oj in range(n_objs):
            if layer_mask[oj] and oj != oi:
                m[oj, h_dst, w_dst] = False
    m[oi, h_dst, w_dst] = True
    return m


MUTATIONS = [mutate_swap, mutate_place, mutate_remove, mutate_move]


def mutate(env: PuzzleJaxEnv, multihot: np.ndarray, rng: np.random.Generator,
           n_mutations: int = 1, required_player_count: Optional[int] = None) -> np.ndarray:
    """Apply *n_mutations* random mutations to a multihot level.

    Ensures at least one player tile remains after mutation.  When
    *required_player_count* is not None, also rejects any single-step
    mutation that would change the number of player-occupied cells.
    """
    m = multihot.copy()
    for _ in range(n_mutations):
        fn = rng.choice(MUTATIONS)
        candidate = fn(env, m, rng)
        # Safety: reject mutations that remove all players
        n_players = _count_players(env, candidate)
        if n_players < 1:
            continue
        if required_player_count is not None and n_players != required_player_count:
            continue
        m = candidate
    return m


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_level(
    puzzle: PuzzleJaxPuxleEnv,
    heuristic: PuzzleJaxHeuristic,
    multihot_level: jnp.ndarray,
    algo: str,
    max_nodes: int,
    batch_size: int,
    cost_weight: float,
    search_fn=None,
) -> dict:
    """Run A* on a custom level and return evaluation metrics.

    Returns dict with keys: solved, cost, generated_states, time, actions.
    """
    # Swap in the custom level via PJParams
    params = PJParams(level=multihot_level)
    _, pj_state = puzzle.env.reset(jax.random.PRNGKey(0), params)
    init_state = puzzle._pj_to_state(pj_state)
    solve_config = puzzle.get_solve_config()

    # Build search function if not provided (first call triggers JIT)
    if search_fn is None:
        effective_max_nodes = (max_nodes // batch_size) * batch_size or batch_size
        search_fn = astar_builder(
            puzzle, heuristic,
            batch_size=batch_size,
            max_nodes=effective_max_nodes,
            cost_weight=cost_weight,
        )

    start = time.time()
    search_result = search_fn(solve_config, init_state)
    solved = bool(search_result.solved.block_until_ready())
    elapsed = time.time() - start

    generated = int(search_result.generated_size)

    actions_list = []
    cost_val = float("inf")

    if solved:
        solved_idx = search_result.solved_idx
        cost_val = float(search_result.get_cost(solved_idx))
        try:
            path = search_result.get_solved_path()
            for node in path:
                action = getattr(node, "action", None)
                if action is not None:
                    actions_list.append(int(action))
        except Exception:
            pass

    return {
        "solved": solved,
        "cost": cost_val if math.isfinite(cost_val) else None,
        "generated_states": generated,
        "time": elapsed,
        "actions": actions_list,
        "search_fn": search_fn,
    }


# ============================================================================
# GIF saving (reused from search_jaxtar)
# ============================================================================

def save_gif(path_steps: List[PathStep], solve_config, gif_path: str,
             max_animation_time: float = 10.0):
    import imageio
    imgs = []
    path_states = [step.state for step in path_steps]
    for idx, step in enumerate(path_steps):
        img = step.state.img(idx=idx, path=path_states, solve_config=solve_config)
        imgs.append(img)
    if not imgs:
        return
    fps = max(4, len(imgs) / max_animation_time)
    imageio.mimsave(gif_path, imgs, fps=fps)


def render_champion_gif(
    puzzle: PuzzleJaxPuxleEnv,
    heuristic: PuzzleJaxHeuristic,
    multihot_level: jnp.ndarray,
    actions: List[int],
    gif_path: str,
):
    """Render a GIF of the champion solution."""
    params = PJParams(level=multihot_level)
    _, pj_state = puzzle.env.reset(jax.random.PRNGKey(0), params)
    init_state = puzzle._pj_to_state(pj_state)
    solve_config = puzzle.get_solve_config()

    try:
        path_steps = build_path_steps_from_actions(
            puzzle=puzzle,
            solve_config=solve_config,
            initial_state=init_state,
            actions=actions,
            heuristic=heuristic,
        )
        if path_steps:
            save_gif(path_steps, solve_config, gif_path)
    except Exception as e:
        print(f"  Warning: GIF rendering failed: {e}")


# ============================================================================
# Run-directory naming
# ============================================================================

def _make_run_dir_name(
    level_i: int, pop_size: int, n_mutations_min: int, n_mutations_max: int,
    max_nodes: int, batch_size: int, cost_weight: float, fitness_mode: str,
    seed: int, preserve_players: bool,
) -> str:
    """Build a deterministic subdirectory name from hyperparameters.

    ``gens`` is intentionally excluded so that resuming with more generations
    writes into the same directory.
    """
    parts = [
        f"lv{level_i}",
        f"pop{pop_size}",
        f"mut{n_mutations_min}-{n_mutations_max}",
        f"m{max_nodes}",
        f"b{batch_size}",
        f"w{cost_weight}",
        f"fit-{fitness_mode}",
        f"seed{seed}",
    ]
    if not preserve_players:
        parts.append("freeP")
    return "_".join(parts)


# ============================================================================
# Checkpoint persistence
# ============================================================================

_CHECKPOINT_FILE = "checkpoint.pkl"


def _save_evolution_state(
    out_dir: str,
    gen: int,
    champion_multihot: np.ndarray,
    champion_result: dict,
    champion_fitness: float,
    champion_gen: int,
    history: list,
    rng: np.random.Generator,
):
    """Persist everything needed to resume evolution."""
    state = {
        "gen": gen,
        "champion_multihot": champion_multihot,
        "champion_result": champion_result,
        "champion_fitness": champion_fitness,
        "champion_gen": champion_gen,
        "history": history,
        "rng_state": rng.__getstate__(),
    }
    path = os.path.join(out_dir, _CHECKPOINT_FILE)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)  # atomic on POSIX


def _load_evolution_state(out_dir: str) -> Optional[dict]:
    """Load a previously saved checkpoint, or return None."""
    path = os.path.join(out_dir, _CHECKPOINT_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Main evolution loop
# ============================================================================

def evolve(
    game: str,
    level_i: int,
    n_gens: int,
    pop_size: int,
    n_mutations_min: int,
    n_mutations_max: int,
    max_nodes: int,
    batch_size: int,
    cost_weight: float,
    render_gif: bool,
    seed: int,
    fitness_mode: str,
    preserve_players: bool,
):
    """Run the (1+λ) evolution loop.  Automatically resumes from checkpoint."""
    rng = np.random.default_rng(seed)

    # ---- Initialise environment & puzzle wrapper ----
    print(f"Initialising {game} level {level_i} ...")
    puzzle = PuzzleJaxPuxleEnv(game=game, level_i=level_i)
    heuristic = PuzzleJaxHeuristic(puzzle)
    env = puzzle.env

    # Get original level as multihot
    original_multihot = np.array(env.get_level(level_i))
    original_char = get_char_level(env, level_i)
    original_player_count = _count_players(env, original_multihot)
    required_player_count = original_player_count if preserve_players else None
    print(f"  Original player count: {original_player_count}"
          f"  (enforced={preserve_players})")

    # Output directory – includes hyperparam signature
    run_name = _make_run_dir_name(
        level_i, pop_size, n_mutations_min, n_mutations_max,
        max_nodes, batch_size, cost_weight, fitness_mode, seed,
        preserve_players,
    )
    out_dir = os.path.join(EVOLVED_DIR, game, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Warmup: evaluate original level to compile the search fn ----
    print("Warming up (JIT compile) on original level ...")
    warmup_start = time.time()
    result0 = evaluate_level(
        puzzle, heuristic, jnp.array(original_multihot),
        algo="astar", max_nodes=max_nodes, batch_size=batch_size,
        cost_weight=cost_weight,
    )
    warmup_time = time.time() - warmup_start
    search_fn = result0.pop("search_fn")
    print(f"  Warmup: {warmup_time:.2f}s  |  "
          f"solved={result0['solved']}  cost={result0['cost']}  "
          f"states={result0['generated_states']:,}")

    # ---- Second (timed) evaluation of original ----
    result0 = evaluate_level(
        puzzle, heuristic, jnp.array(original_multihot),
        algo="astar", max_nodes=max_nodes, batch_size=batch_size,
        cost_weight=cost_weight, search_fn=search_fn,
    )
    search_fn = result0.pop("search_fn")
    print(f"  Original: solved={result0['solved']}  cost={result0['cost']}  "
          f"states={result0['generated_states']:,}  time={result0['time']:.2f}s")

    def fitness(result: dict) -> float:
        """Compute fitness from an evaluation result dict.

        Higher is better.  Unsolvable levels **always** score below any
        solvable level – we use a large negative offset that can never be
        overcome by generated_states alone.
        """
        if not result["solved"]:
            # -1e18 ensures that no unsolved level can beat a solved one,
            # regardless of how many states were explored.
            return -1e18 + result["generated_states"]
        if fitness_mode == "cost":
            return result["cost"]
        elif fitness_mode == "states":
            return result["generated_states"]
        elif fitness_mode == "cost+states":
            # Weighted combination: primarily cost, tiebreak by states
            return result["cost"] * 1e6 + result["generated_states"]
        else:
            return result["cost"]

    # ---- Try to resume from checkpoint ----
    start_gen = 0
    ckpt = _load_evolution_state(out_dir)
    if ckpt is not None:
        start_gen = ckpt["gen"] + 1
        champion_multihot = ckpt["champion_multihot"]
        champion_result = ckpt["champion_result"]
        champion_fitness = ckpt["champion_fitness"]
        champion_gen = ckpt["champion_gen"]
        history = ckpt["history"]
        rng.__setstate__(ckpt["rng_state"])
        print(f"\nResumed from checkpoint at gen {ckpt['gen']}  "
              f"(fitness={champion_fitness:.0f})")
        if start_gen >= n_gens:
            print(f"Already completed {start_gen} generations (requested {n_gens}). "
                  f"Increase --gens to continue.")
            return
    else:
        # ---- Fresh evolution state ----
        champion_multihot = original_multihot.copy()
        champion_result = result0
        champion_fitness = fitness(result0)
        champion_gen = -1  # original level

        history = [{
            "gen": -1,
            "fitness": champion_fitness,
            **{k: v for k, v in result0.items() if k != "search_fn"},
        }]

        # Save GIF of the initial (unmodified) level for reference
        if render_gif and result0.get("solved") and result0.get("actions"):
            init_gif_path = os.path.join(out_dir, f"evolved_lv{level_i}_initial.gif")
            render_champion_gif(
                puzzle, heuristic, jnp.array(original_multihot),
                result0["actions"], init_gif_path,
            )
            print(f"  Initial level GIF saved to {init_gif_path}")

    print(f"\nStarting evolution: gens {start_gen}..{n_gens - 1}, λ={pop_size}")
    print(f"Fitness mode: {fitness_mode}")
    print(f"Current champion fitness: {champion_fitness}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    for gen in range(start_gen, n_gens):
        gen_start = time.time()
        best_child_fitness = -float("inf")
        best_child_multihot = None
        best_child_result = None

        for child_i in range(pop_size):
            # Determine number of mutations (anneal or fixed)
            n_mut = rng.integers(n_mutations_min, n_mutations_max + 1)
            child_multihot = mutate(
                env, champion_multihot, rng, n_mutations=n_mut,
                required_player_count=required_player_count,
            )

            # Evaluate
            child_result = evaluate_level(
                puzzle, heuristic, jnp.array(child_multihot),
                algo="astar", max_nodes=max_nodes, batch_size=batch_size,
                cost_weight=cost_weight, search_fn=search_fn,
            )
            _ = child_result.pop("search_fn")
            child_fit = fitness(child_result)

            status = "SOLVED" if child_result["solved"] else "UNSOLVED"
            print(f"  Gen {gen:3d} child {child_i}: {status}  "
                  f"cost={child_result['cost']}  states={child_result['generated_states']:,}  "
                  f"fit={child_fit:.0f}  ({child_result['time']:.2f}s)")

            if child_fit > best_child_fitness:
                best_child_fitness = child_fit
                best_child_multihot = child_multihot
                best_child_result = child_result

        # ---- Selection: keep the best of parent and children ----
        improved = False
        if best_child_fitness > champion_fitness:
            champion_multihot = best_child_multihot
            champion_result = best_child_result
            champion_fitness = best_child_fitness
            champion_gen = gen
            improved = True

        gen_elapsed = time.time() - gen_start
        marker = " ***NEW CHAMPION***" if improved else ""
        print(f"Gen {gen:3d}: champion fitness={champion_fitness:.0f}  "
              f"cost={champion_result['cost']}  "
              f"states={champion_result['generated_states']:,}  "
              f"({gen_elapsed:.2f}s){marker}")

        history.append({
            "gen": gen,
            "fitness": champion_fitness,
            **{k: v for k, v in champion_result.items() if k != "search_fn"},
        })

        # ---- Save on improvement ----
        if improved:
            _save_checkpoint(
                out_dir, game, level_i, gen, champion_multihot, champion_result,
                champion_fitness, history, env, puzzle, heuristic, render_gif,
            )
        # Always persist resumable state so we can pick up from any interruption
        _save_evolution_state(
            out_dir, gen, champion_multihot, champion_result,
            champion_fitness, champion_gen, history, rng,
        )

    print("\n" + "=" * 60)
    print(f"Evolution complete.  Best fitness: {champion_fitness:.0f}")
    print(f"  Champion from gen {champion_gen}")
    if champion_result["solved"]:
        print(f"  Solution cost: {champion_result['cost']}")
        print(f"  Solution length: {len(champion_result['actions'])} actions")
    print(f"  Generated states: {champion_result['generated_states']:,}")
    print(f"Results in {out_dir}")


def _save_checkpoint(
    out_dir, game, level_i, gen, multihot, result, fitness_val, history,
    env, puzzle, heuristic, render_gif,
):
    """Save the current champion to disk."""
    base = f"evolved_lv{level_i}"

    # Save multihot as numpy
    np.save(os.path.join(out_dir, f"{base}_multihot.npy"), np.array(multihot))

    # Save character-grid level
    try:
        char_level = multihot_to_char_level(env, multihot)
        char_path = os.path.join(out_dir, f"{base}_level.txt")
        with open(char_path, "w") as f:
            for row in char_level:
                f.write("".join(row) + "\n")
    except Exception as e:
        print(f"  Warning: could not save char level: {e}")

    # Save JSON results
    json_data = {
        "game": game,
        "level_i": level_i,
        "gen": gen,
        "fitness": fitness_val,
        "solved": result.get("solved"),
        "cost": result.get("cost"),
        "generated_states": result.get("generated_states"),
        "actions": result.get("actions", []),
        "time": result.get("time"),
        "history": history,
    }
    json_path = os.path.join(out_dir, f"{base}_result.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Render GIF of champion
    if render_gif:
        gif_path = os.path.join(out_dir, f"{base}_gen{gen}.gif")
        if result.get("solved") and result.get("actions"):
            render_champion_gif(puzzle, heuristic, jnp.array(multihot),
                                result["actions"], gif_path)
            print(f"  GIF saved to {gif_path}")
        else:
            # Render a single-frame GIF of the initial state for unsolved levels
            try:
                pj_state = PJState(
                    multihot_level=jnp.array(multihot),
                    win=jnp.array(False),
                    score=jnp.array(0, dtype=jnp.int32),
                    heuristic=jnp.array(0, dtype=jnp.int32),
                    restart=jnp.array(False),
                    init_heuristic=jnp.array(0, dtype=jnp.int32),
                    prev_heuristic=jnp.array(0, dtype=jnp.int32),
                    step_i=jnp.array(0, dtype=jnp.int32),
                    rng=jax.random.PRNGKey(0),
                    view_bounds=env._get_default_view_bounds(jnp.array(multihot).shape[1:]),
                )
                img = np.array(env.render(pj_state, cv2=False))
                import imageio
                imageio.mimsave(gif_path, [img], fps=1)
                print(f"  GIF saved to {gif_path} (unsolved – initial state only)")
            except Exception as e:
                print(f"  Warning: GIF rendering for unsolved level failed: {e}")


# ============================================================================
# Hydra entrypoint
# ============================================================================

def main(cfg: EvolveLevelConfig):
    if cfg.game is None:
        raise ValueError("`game` must be set, e.g. `python evolve_level.py game=sokoban_basic`.")
    if cfg.fitness not in {"cost", "states", "cost+states"}:
        raise ValueError(f"Unsupported fitness mode: {cfg.fitness}")
    evolve(
        game=cfg.game,
        level_i=cfg.level,
        n_gens=cfg.gens,
        pop_size=cfg["pop"],
        n_mutations_min=cfg.n_mutations_min,
        n_mutations_max=cfg.n_mutations_max,
        max_nodes=cfg.max_nodes,
        batch_size=cfg.batch_size,
        cost_weight=cfg.cost_weight,
        render_gif=cfg.render_gif,
        seed=cfg.seed,
        fitness_mode=cfg.fitness,
        preserve_players=not cfg.allow_player_change,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="evolve_level")
def main_launch(cfg: EvolveLevelConfig):
    main(cfg)


if __name__ == "__main__":
    main_launch()
