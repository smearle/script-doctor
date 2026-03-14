"""Evolve a PuzzleScript level using the C++ engine as evaluator.

Compiles the game once, then mutates levels at the bitfield representation
level and evaluates via C++ search — no per-mutant recompilation.

Uses a thread pool for parallel evaluation (C++ solvers release the GIL).
Designed so the thread pool can later be replaced by a C++ OpenMP batch solver.

Usage examples:
    python evolve_level_cpp.py game=sokoban_basic level=0
    python evolve_level_cpp.py game=sokoban_basic level=0 pop=8 gens=50
    python evolve_level_cpp.py game=sokoban_basic level=0 algo=bfs n_workers=8

Results are saved to data/evolved_levels_cpp/<GAME>/<run-subdir>/
"""

import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import hydra
import numpy as np

from conf.config import EvolveLevelCppConfig
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_cpp._puzzlescript_cpp import (
    Engine as _CppEngine,
    LevelBackup,
    MCTSOptions,
    solve_astar as _solve_astar,
    solve_bfs as _solve_bfs,
    solve_gbfs as _solve_gbfs,
    solve_mcts as _solve_mcts,
    solve_random as _solve_random,
)
from puzzlescript_jax.globals import DATA_DIR
from puzzlescript_jax.utils import init_ps_lark_parser

EVOLVED_DIR = os.path.join(DATA_DIR, "evolved_levels_cpp")
CHECKPOINT_FILE = "checkpoint.pkl"


# ============================================================================
# Level metadata extracted from compiled JSON
# ============================================================================

class LevelMutator:
    """Encapsulates game metadata and provides bitfield-level mutations."""

    def __init__(self, json_state: dict, width: int, height: int):
        self.n_objs = json_state["objectCount"]
        self.stride = json_state["STRIDE_OBJ"]
        self.width = width
        self.height = height
        self.n_tiles = width * height
        self.bg_idx = json_state.get("backgroundid", 0)

        # Decode player mask
        player_mask_words = json_state["playerMask"][1]
        self.player_indices = set()
        for word_i, word in enumerate(player_mask_words):
            for bit in range(32):
                if word & (1 << bit):
                    self.player_indices.add(word_i * 32 + bit)

        # Decode collision layers
        layer_mask_arrays = json_state["layerMasks"]
        self.layer_masks = layer_mask_arrays  # list of list[int], one per layer

        # Build obj -> layer mapping
        self.obj_to_layer = {}
        for layer_i, mask_words in enumerate(layer_mask_arrays):
            for word_i, word in enumerate(mask_words):
                for bit in range(32):
                    if word & (1 << bit):
                        obj_idx = word_i * 32 + bit
                        if obj_idx < self.n_objs:
                            self.obj_to_layer[obj_idx] = layer_i

        # Mutable objects (non-background)
        self.mutable_objs = [i for i in range(self.n_objs) if i != self.bg_idx]

    def has_obj(self, dat: list, tile: int, obj_idx: int) -> bool:
        word_i, bit = divmod(obj_idx, 32)
        return bool(dat[tile * self.stride + word_i] & (1 << bit))

    def set_obj(self, dat: list, tile: int, obj_idx: int):
        word_i, bit = divmod(obj_idx, 32)
        dat[tile * self.stride + word_i] |= (1 << bit)

    def clear_obj(self, dat: list, tile: int, obj_idx: int):
        word_i, bit = divmod(obj_idx, 32)
        dat[tile * self.stride + word_i] &= ~(1 << bit)

    def clear_layer_at_tile(self, dat: list, tile: int, layer_idx: int):
        """Clear all objects on a collision layer at a tile."""
        mask_words = self.layer_masks[layer_idx]
        base = tile * self.stride
        for i, mask_word in enumerate(mask_words):
            if mask_word:
                dat[base + i] &= ~mask_word

    def get_tile_words(self, dat: list, tile: int) -> list:
        base = tile * self.stride
        return dat[base:base + self.stride]

    def set_tile_words(self, dat: list, tile: int, words: list):
        base = tile * self.stride
        for i, w in enumerate(words):
            dat[base + i] = w

    def count_players(self, dat: list) -> int:
        count = 0
        for tile in range(self.n_tiles):
            for pi in self.player_indices:
                if self.has_obj(dat, tile, pi):
                    count += 1
                    break  # count tiles, not objects
        return count

    def get_present_nonbg_objs(self, dat: list, tile: int) -> list:
        """Return list of non-background object indices present at tile."""
        result = []
        base = tile * self.stride
        for obj_i in range(self.n_objs):
            if obj_i == self.bg_idx:
                continue
            word_i, bit = divmod(obj_i, 32)
            if dat[base + word_i] & (1 << bit):
                result.append(obj_i)
        return result

    # ---- Mutations ----

    def mutate_swap(self, dat: list, rng: np.random.Generator) -> list:
        """Swap the objects at two random tiles."""
        t1, t2 = rng.integers(self.n_tiles, size=2)
        if t1 == t2:
            return dat
        out = list(dat)
        w1 = self.get_tile_words(out, t1)
        w2 = self.get_tile_words(out, t2)
        self.set_tile_words(out, t1, w2)
        self.set_tile_words(out, t2, w1)
        return out

    def mutate_place(self, dat: list, rng: np.random.Generator) -> list:
        """Place a random non-background object at a random tile."""
        if not self.mutable_objs:
            return dat
        obj_idx = int(rng.choice(self.mutable_objs))
        tile = int(rng.integers(self.n_tiles))
        layer_i = self.obj_to_layer.get(obj_idx, -1)
        out = list(dat)
        if layer_i >= 0:
            self.clear_layer_at_tile(out, tile, layer_i)
        self.set_obj(out, tile, obj_idx)
        return out

    def mutate_remove(self, dat: list, rng: np.random.Generator) -> list:
        """Remove a random non-background, non-player object."""
        candidates = []
        for tile in range(self.n_tiles):
            for obj_i in self.get_present_nonbg_objs(dat, tile):
                if obj_i not in self.player_indices:
                    candidates.append((tile, obj_i))
        if not candidates:
            return dat
        tile, obj_i = candidates[int(rng.integers(len(candidates)))]
        out = list(dat)
        self.clear_obj(out, tile, obj_i)
        return out

    def mutate_move(self, dat: list, rng: np.random.Generator) -> list:
        """Move a random non-background object to a different tile."""
        candidates = []
        for tile in range(self.n_tiles):
            for obj_i in self.get_present_nonbg_objs(dat, tile):
                candidates.append((tile, obj_i))
        if not candidates:
            return dat
        src_tile, obj_i = candidates[int(rng.integers(len(candidates)))]
        dst_tile = int(rng.integers(self.n_tiles))
        if src_tile == dst_tile:
            return dat
        out = list(dat)
        self.clear_obj(out, src_tile, obj_i)
        layer_i = self.obj_to_layer.get(obj_i, -1)
        if layer_i >= 0:
            self.clear_layer_at_tile(out, dst_tile, layer_i)
        self.set_obj(out, dst_tile, obj_i)
        return out

    MUTATIONS = ("swap", "place", "remove", "move")

    def mutate(self, dat: list, rng: np.random.Generator,
               n_mutations: int = 1,
               required_player_count: Optional[int] = None) -> list:
        """Apply n_mutations random mutations, rejecting invalid results."""
        out = list(dat)
        fns = {
            "swap": self.mutate_swap,
            "place": self.mutate_place,
            "remove": self.mutate_remove,
            "move": self.mutate_move,
        }
        for _ in range(n_mutations):
            name = rng.choice(self.MUTATIONS)
            candidate = fns[name](out, rng)
            n_players = self.count_players(candidate)
            if n_players < 1:
                continue
            if required_player_count is not None and n_players != required_player_count:
                continue
            out = candidate
        return out


# ============================================================================
# Parallel evaluator pool
# ============================================================================

SOLVER_FNS = {
    "bfs": _solve_bfs,
    "astar": _solve_astar,
    "gbfs": _solve_gbfs,
}


def _evaluate_one(engine: _CppEngine, dat: list, width: int, height: int,
                  algo: str, max_iters: int, timeout_ms: int) -> dict:
    """Evaluate a single candidate level. Runs in a worker thread."""
    backup = LevelBackup(dat, width, height)
    engine.restore_level(backup)

    start = time.time()
    if algo == "mcts":
        opts = MCTSOptions()
        opts.max_iterations = max_iters
        result = _solve_mcts(engine, opts)
    elif algo == "random":
        result = _solve_random(engine, 100, max_iters, timeout_ms)
    else:
        solver_fn = SOLVER_FNS[algo]
        result = solver_fn(engine, max_iters, timeout_ms)
    elapsed = time.time() - start

    return {
        "solved": result.won,
        "cost": len(result.actions) if result.won else None,
        "generated_states": result.iterations,
        "time": elapsed,
        "actions": list(result.actions),
        "score": result.score,
        "timeout": result.timeout,
    }


class EvaluatorPool:
    """Pool of C++ engines for parallel level evaluation.

    Each worker thread gets its own Engine instance loaded from the same
    compiled JSON.  The pool can be replaced by a C++ OpenMP batch solver
    by changing evaluate_batch() internals.
    """

    def __init__(self, json_str: str, level_i: int, n_workers: int,
                 algo: str, max_iters: int, timeout_ms: int):
        self.n_workers = n_workers
        self.algo = algo
        self.max_iters = max_iters
        self.timeout_ms = timeout_ms

        # Create one engine per worker, each independently loaded
        self.engines: list[_CppEngine] = []
        for _ in range(n_workers):
            engine = _CppEngine()
            engine.load_from_json(json_str)
            engine.load_level(level_i)
            self.engines.append(engine)

        # Get level geometry from the first engine
        self.width = self.engines[0].get_width()
        self.height = self.engines[0].get_height()

        self.executor = ThreadPoolExecutor(max_workers=n_workers)

    def evaluate_batch(self, candidates: list[list]) -> list[dict]:
        """Evaluate a batch of candidate dat arrays in parallel.

        This is the abstraction boundary: swap internals for a C++ OpenMP
        batch solver without changing callers.
        """
        futures = []
        for i, dat in enumerate(candidates):
            engine = self.engines[i % self.n_workers]
            futures.append(self.executor.submit(
                _evaluate_one, engine, dat,
                self.width, self.height,
                self.algo, self.max_iters, self.timeout_ms,
            ))
        return [f.result() for f in futures]

    def evaluate_one(self, dat: list) -> dict:
        """Evaluate a single candidate (uses first engine, blocks)."""
        return _evaluate_one(
            self.engines[0], dat,
            self.width, self.height,
            self.algo, self.max_iters, self.timeout_ms,
        )

    def shutdown(self):
        self.executor.shutdown(wait=False)


# ============================================================================
# Fitness
# ============================================================================

def compute_fitness(result: dict, mode: str) -> float:
    if not result["solved"]:
        return -1e18 + result["generated_states"]
    if mode == "cost":
        return result["cost"]
    if mode == "states":
        return result["generated_states"]
    if mode == "cost+states":
        return result["cost"] * 1e6 + result["generated_states"]
    raise ValueError(f"Unknown fitness mode: {mode}")


# ============================================================================
# Run directory, checkpoint, champion persistence
# ============================================================================

def make_run_dir_name(
    level_i: int, pop_size: int, n_mutations_min: int, n_mutations_max: int,
    max_steps: int, algo: str, fitness_mode: str, seed: int,
    preserve_players: bool,
) -> str:
    parts = [
        f"lv{level_i}",
        f"pop{pop_size}",
        f"mut{n_mutations_min}-{n_mutations_max}",
        f"steps{max_steps}",
        f"algo-{algo}",
        f"fit-{fitness_mode}",
        f"seed{seed}",
    ]
    if not preserve_players:
        parts.append("freeP")
    return "_".join(parts)


def save_checkpoint(
    out_dir: str, gen: int, champion_dat: list, champion_result: dict,
    champion_fitness: float, champion_gen: int, history: list,
    rng: np.random.Generator,
) -> None:
    payload = {
        "gen": gen,
        "champion_dat": champion_dat,
        "champion_result": champion_result,
        "champion_fitness": champion_fitness,
        "champion_gen": champion_gen,
        "history": history,
        "rng_state": rng.bit_generator.state,
    }
    path = os.path.join(out_dir, CHECKPOINT_FILE)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_checkpoint(out_dir: str) -> Optional[dict]:
    path = os.path.join(out_dir, CHECKPOINT_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_champion(
    out_dir: str, game: str, level_i: int, gen: int,
    champion_dat: list, champion_result: dict, champion_fitness: float,
    history: list,
    backend: Optional[CppPuzzleScriptBackend] = None,
    render_gif: bool = False,
    gif_frame_duration: float = 0.05,
    gif_scale: int = 10,
) -> None:
    import json as json_mod
    base = f"evolved_lv{level_i}"

    # Save result JSON
    result_path = os.path.join(out_dir, f"{base}_result.json")
    with open(result_path, "w") as f:
        json_mod.dump({
            "game": game,
            "level_i": level_i,
            "gen": gen,
            "fitness": champion_fitness,
            **champion_result,
            "history": history,
        }, f, indent=2)

    # Render GIF of champion solution
    if render_gif and backend is not None and champion_result.get("solved"):
        actions = champion_result.get("actions", [])
        if actions:
            gif_path = os.path.join(out_dir, f"{base}_gen{gen}.gif")
            try:
                # Capture frames synchronously (needs the engine), then
                # write the GIF in a background thread.
                backend.cpp_engine.load_level(level_i)
                backup = LevelBackup(champion_dat,
                                     backend.cpp_engine.width,
                                     backend.cpp_engine.height)
                backend.cpp_engine.restore_level(backup)
                renderer = backend._ensure_renderer()
                renderer.reset_viewport(backend.cpp_engine.width,
                                        backend.cpp_engine.height)

                frames = [backend.render_frame()]
                MAX_AGAIN = 100
                for action in actions:
                    backend.process_input(int(action))
                    again_steps = 0
                    while backend.againing and again_steps < MAX_AGAIN:
                        backend.process_input(-1)
                        again_steps += 1
                    frames.append(backend.render_frame())

                if gif_scale > 1:
                    frames = [
                        np.repeat(np.repeat(frame, gif_scale, axis=0),
                                  gif_scale, axis=1)
                        for frame in frames
                    ]

                def _write_gif(path, frs, duration):
                    import imageio
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    imageio.mimsave(path, frs, duration=duration, loop=0)
                    print(f"  GIF saved to {path}")

                threading.Thread(
                    target=_write_gif,
                    args=(gif_path, frames, gif_frame_duration),
                    daemon=True,
                ).start()
            except Exception as e:
                print(f"  Warning: GIF rendering failed: {e}")


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
    max_steps: int,
    timeout_ms: int,
    algo: str,
    seed: int,
    fitness_mode: str,
    preserve_players: bool,
    render_gif: bool,
    gif_frame_duration: float,
    gif_scale: int,
    n_workers: int,
    depth_increase_threshold: float = 0.95,
) -> None:
    initial_max_steps = max_steps
    rng = np.random.default_rng(seed)

    # ---- Compile game once ----
    print(f"Compiling {game} ...")
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    backend.compile_game(parser, game)
    json_str = backend.cpp_engine._json_state
    # _json_state is the parsed dict; we need the raw string for worker engines
    # Re-serialize from the dict (CppPuzzleScriptEngine stores the parsed JSON)
    json_str_raw = json.dumps(json_str)

    # ---- Load target level and get template ----
    backend.cpp_engine.load_level(level_i)
    template_backup = backend.cpp_engine.backup_level()
    template_dat = list(template_backup.dat)
    width = template_backup.width
    height = template_backup.height
    print(f"  Level {level_i}: {width}x{height}, "
          f"{len(template_dat)} words (stride={json_str['STRIDE_OBJ']})")

    # ---- Build mutator ----
    mutator = LevelMutator(json_str, width, height)
    original_player_count = mutator.count_players(template_dat)
    required_player_count = original_player_count if preserve_players else None
    print(f"  Objects: {mutator.n_objs}, Players: {original_player_count} "
          f"(enforced={preserve_players})")

    # ---- Create evaluator pool ----
    pool = EvaluatorPool(
        json_str=json_str_raw, level_i=level_i,
        n_workers=min(n_workers, pop_size),
        algo=algo, max_iters=max_steps, timeout_ms=timeout_ms,
    )

    # ---- Output directory ----
    run_name = make_run_dir_name(
        level_i, pop_size, n_mutations_min, n_mutations_max,
        max_steps, algo, fitness_mode, seed, preserve_players,
    )
    out_dir = os.path.join(EVOLVED_DIR, game, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Evaluate original level ----
    print("Evaluating original level ...")
    original_result = pool.evaluate_one(template_dat)
    original_fitness = compute_fitness(original_result, fitness_mode)
    status = "SOLVED" if original_result["solved"] else "UNSOLVED"
    print(f"  Original: {status}  cost={original_result['cost']}  "
          f"states={original_result['generated_states']:,}  "
          f"fitness={original_fitness:.0f}  ({original_result['time']:.2f}s)")

    # ---- Resume or initialize ----
    ckpt = load_checkpoint(out_dir)
    if ckpt is not None:
        start_gen = ckpt["gen"] + 1
        champion_dat = ckpt["champion_dat"]
        champion_result = ckpt["champion_result"]
        champion_fitness = ckpt["champion_fitness"]
        champion_gen = ckpt["champion_gen"]
        history = ckpt["history"]
        rng.bit_generator.state = ckpt["rng_state"]
        print(f"\nResumed from generation {ckpt['gen']} "
              f"(fitness={champion_fitness:.0f})")
        if start_gen >= n_gens:
            print(f"Already completed {start_gen} generations. "
                  f"Increase gens to continue.")
            pool.shutdown()
            return
    else:
        champion_dat = list(template_dat)
        champion_result = original_result
        champion_fitness = original_fitness
        champion_gen = -1
        history = [{"gen": -1, "fitness": original_fitness, **original_result}]
        start_gen = 0

        if render_gif and original_result["solved"] and original_result["actions"]:
            save_champion(
                out_dir, game, level_i, -1, champion_dat, champion_result,
                champion_fitness, history,
                backend=backend, render_gif=True,
                gif_frame_duration=gif_frame_duration, gif_scale=gif_scale,
            )

    print(f"\nStarting evolution: gens {start_gen}..{n_gens - 1}, "
          f"lambda={pop_size}, workers={pool.n_workers}")
    print(f"Fitness mode: {fitness_mode}, Algo: {algo}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    for gen in range(start_gen, n_gens):
        gen_start = time.time()

        # ---- Generate children ----
        children_dat = []
        for _ in range(pop_size):
            n_mut = int(rng.integers(n_mutations_min, n_mutations_max + 1))
            child = mutator.mutate(
                champion_dat, rng, n_mutations=n_mut,
                required_player_count=required_player_count,
            )
            children_dat.append(child)

        # ---- Evaluate all children in parallel ----
        results = pool.evaluate_batch(children_dat)

        # ---- Find best child ----
        best_child_fitness = -float("inf")
        best_child_idx = -1
        for i, result in enumerate(results):
            child_fitness = compute_fitness(result, fitness_mode)
            status = "SOLVED" if result["solved"] else "UNSOLVED"
            print(f"  Gen {gen:3d} child {i}: {status}  "
                  f"cost={result['cost']}  "
                  f"states={result['generated_states']:,}  "
                  f"fit={child_fitness:.0f}  ({result['time']:.2f}s)")
            if child_fitness > best_child_fitness:
                best_child_fitness = child_fitness
                best_child_idx = i

        # ---- Selection ----
        improved = False
        if best_child_fitness > champion_fitness:
            champion_dat = children_dat[best_child_idx]
            champion_result = results[best_child_idx]
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
            **{k: v for k, v in champion_result.items()},
        })

        if improved:
            save_champion(
                out_dir, game, level_i, gen, champion_dat, champion_result,
                champion_fitness, history,
                backend=backend, render_gif=render_gif,
                gif_frame_duration=gif_frame_duration, gif_scale=gif_scale,
            )

        save_checkpoint(
            out_dir, gen, champion_dat, champion_result,
            champion_fitness, champion_gen, history, rng,
        )

        # ---- Adaptive depth increase ----
        if (depth_increase_threshold > 0
                and champion_result.get("solved")
                and champion_result["generated_states"] >= depth_increase_threshold * max_steps):
            max_steps += initial_max_steps
            pool.max_iters = max_steps
            print(f"  >> Depth increased: max_steps now {max_steps:,} "
                  f"(+{initial_max_steps:,})")

    pool.shutdown()
    print("\n" + "=" * 60)
    print(f"Evolution complete.  Best fitness: {champion_fitness:.0f}")
    print(f"  Champion from gen {champion_gen}")
    if champion_result["solved"]:
        print(f"  Solution cost: {champion_result['cost']}")
        print(f"  Solution length: {len(champion_result['actions'])} actions")
    print(f"  Generated states: {champion_result['generated_states']:,}")
    print(f"Results in {out_dir}")


# ============================================================================
# Hydra entrypoint
# ============================================================================

def main(cfg: EvolveLevelCppConfig) -> None:
    if cfg.game is None:
        raise ValueError(
            "`game` must be set, e.g. "
            "`python evolve_level_cpp.py game=sokoban_basic`."
        )
    if cfg.algo not in {"bfs", "astar", "gbfs", "mcts", "random"}:
        raise ValueError(f"Unsupported algo: {cfg.algo}")
    if cfg.fitness not in {"cost", "states", "cost+states"}:
        raise ValueError(f"Unsupported fitness mode: {cfg.fitness}")
    evolve(
        game=cfg.game,
        level_i=cfg.level,
        n_gens=cfg.gens,
        pop_size=cfg["pop"],
        n_mutations_min=cfg.n_mutations_min,
        n_mutations_max=cfg.n_mutations_max,
        max_steps=cfg.max_steps,
        timeout_ms=cfg.timeout * 1_000 if cfg.timeout > 0 else -1,
        algo=cfg.algo,
        seed=cfg.seed,
        fitness_mode=cfg.fitness,
        preserve_players=not cfg.allow_player_change,
        render_gif=cfg.render_gif,
        gif_frame_duration=cfg.gif_frame_duration,
        gif_scale=cfg.gif_scale,
        n_workers=cfg.n_workers,
        depth_increase_threshold=cfg.depth_increase_threshold,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="evolve_level_cpp")
def main_launch(cfg: EvolveLevelCppConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_launch()
