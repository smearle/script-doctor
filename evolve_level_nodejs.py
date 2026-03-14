"""Evolve a PuzzleScript level using the original NodeJS engine as evaluator."""

import json
import os
import pickle
import re
import time
from typing import Optional

import hydra
import numpy as np

from conf.config import EvolveLevelNodeJSConfig
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.globals import DATA_DIR, SIMPLIFIED_GAMES_DIR
from puzzlescript_jax.utils import init_ps_lark_parser, get_tree_from_txt


EVOLVED_DIR = os.path.join(DATA_DIR, "evolved_levels_nodejs")
CHECKPOINT_FILE = "checkpoint.pkl"
MUTATION_NAMES = ("swap", "place", "remove", "move")


def load_simplified_game_text(game: str) -> str:
    game_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
    if not os.path.isfile(game_path):
        parser = init_ps_lark_parser()
        get_tree_from_txt(parser, game, test_env_init=False, overwrite=True)
    with open(game_path, "r", encoding="utf-8") as f:
        return f.read()


def split_game_levels(game_text: str) -> tuple[str, list[np.ndarray]]:
    match = re.search(r"(^LEVELS\s*\n)([\s\S]*)$", game_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not find LEVELS section.")

    prefix = game_text[:match.start(2)]
    levels_blob = match.group(2).strip("\n")
    level_blocks = [block for block in re.split(r"\n\s*\n", levels_blob) if block.strip()]
    levels = [np.array([list(line) for line in block.splitlines()], dtype="<U1") for block in level_blocks]
    return prefix, levels


def join_game_levels(prefix: str, levels: list[np.ndarray]) -> str:
    blocks = ["\n".join("".join(row) for row in level) for level in levels]
    return prefix + "\n\n".join(blocks) + "\n"


def parse_legend_single_char_entries(game_text: str) -> dict[str, str]:
    legend_match = re.search(
        r"(^LEGEND\s*\n)([\s\S]*?)(?=^\w[\w ]*\s*$|^SOUNDS\s*$|^COLLISIONLAYERS\s*$|^RULES\s*$)",
        game_text,
        flags=re.MULTILINE,
    )
    if legend_match is None:
        return {}

    entries = {}
    for line in legend_match.group(2).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        if len(key) == 1:
            entries[key] = value.strip().lower()
    return entries


def get_placeable_chars(game_text: str, level: np.ndarray) -> list[str]:
    legend_entries = parse_legend_single_char_entries(game_text)
    level_chars = {char.lower() for char in level.flatten()}
    chars = set(legend_entries.keys()) | level_chars
    return sorted(chars)


def get_player_chars(game_text: str, level: np.ndarray) -> set[str]:
    legend_entries = parse_legend_single_char_entries(game_text)
    player_chars = {char for char, value in legend_entries.items() if "player" in value.split()}
    if player_chars:
        return player_chars
    # Fall back to common player tokens if legend parsing fails.
    fallback = {"p", "@", "0", "1", "2", "3", "4", "5", "6", ":"}
    return {char.lower() for char in level.flatten() if char.lower() in fallback}


def infer_background_char(level: np.ndarray, player_chars: set[str]) -> str:
    counts = {}
    for char in level.flatten():
        key = char.lower()
        if key in player_chars:
            continue
        counts[key] = counts.get(key, 0) + 1
    if "." in counts:
        return "."
    return max(counts, key=counts.get) if counts else "."


def count_players(level: np.ndarray, player_chars: set[str]) -> int:
    return sum(1 for char in level.flatten() if char.lower() in player_chars)


def mutate_swap(level: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = level.shape
    y1, x1 = rng.integers(h), rng.integers(w)
    y2, x2 = rng.integers(h), rng.integers(w)
    if (y1, x1) == (y2, x2):
        return level.copy()
    out = level.copy()
    out[y1, x1], out[y2, x2] = out[y2, x2], out[y1, x1]
    return out


def mutate_place(level: np.ndarray, rng: np.random.Generator, placeable_chars: list[str]) -> np.ndarray:
    h, w = level.shape
    y, x = rng.integers(h), rng.integers(w)
    out = level.copy()
    out[y, x] = rng.choice(placeable_chars)
    return out


def mutate_remove(level: np.ndarray, rng: np.random.Generator, background_char: str, player_chars: set[str]) -> np.ndarray:
    candidates = [
        (y, x) for y in range(level.shape[0]) for x in range(level.shape[1])
        if level[y, x].lower() != background_char and level[y, x].lower() not in player_chars
    ]
    if not candidates:
        return level.copy()
    y, x = candidates[rng.integers(len(candidates))]
    out = level.copy()
    out[y, x] = background_char
    return out


def mutate_move(level: np.ndarray, rng: np.random.Generator, background_char: str) -> np.ndarray:
    h, w = level.shape
    y1, x1 = rng.integers(h), rng.integers(w)
    y2, x2 = rng.integers(h), rng.integers(w)
    if (y1, x1) == (y2, x2):
        return level.copy()
    out = level.copy()
    out[y2, x2] = out[y1, x1]
    out[y1, x1] = background_char
    return out


def mutate_level(
    level: np.ndarray,
    rng: np.random.Generator,
    placeable_chars: list[str],
    player_chars: set[str],
    background_char: str,
    n_mutations: int,
    required_player_count: Optional[int],
) -> np.ndarray:
    out = level.copy()
    for _ in range(n_mutations):
        mutation = rng.choice(MUTATION_NAMES)
        if mutation == "swap":
            candidate = mutate_swap(out, rng)
        elif mutation == "place":
            candidate = mutate_place(out, rng, placeable_chars)
        elif mutation == "remove":
            candidate = mutate_remove(out, rng, background_char, player_chars)
        else:
            candidate = mutate_move(out, rng, background_char)

        n_players = count_players(candidate, player_chars)
        if n_players < 1:
            continue
        if required_player_count is not None and n_players != required_player_count:
            continue
        out = candidate
    return out


def evaluate_level(
    backend: NodeJSPuzzleScriptBackend,
    game_text: str,
    level_i: int,
    algo: str,
    max_steps: int,
    timeout_ms: int,
) -> dict:
    backend.engine.compile(["restart"], game_text)
    result = backend.run_search(
        algo,
        game_text=game_text,
        level_i=level_i,
        n_steps=max_steps,
        timeout_ms=timeout_ms,
        warmup=False,
    )
    cost = len(result.actions) if result.solved else None
    return {
        "solved": result.solved,
        "cost": cost,
        "generated_states": result.iterations,
        "time": result.time,
        "actions": list(result.actions),
        "score": result.score,
        "timeout": result.timeout,
    }


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


def make_run_dir_name(
    level_i: int,
    pop_size: int,
    n_mutations_min: int,
    n_mutations_max: int,
    max_steps: int,
    algo: str,
    fitness_mode: str,
    seed: int,
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
    out_dir: str,
    gen: int,
    champion_level: np.ndarray,
    champion_result: dict,
    champion_fitness: float,
    champion_game_text: str,
    history: list,
    rng: np.random.Generator,
) -> None:
    payload = {
        "gen": gen,
        "champion_level": champion_level,
        "champion_result": champion_result,
        "champion_fitness": champion_fitness,
        "champion_game_text": champion_game_text,
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
    out_dir: str,
    game: str,
    level_i: int,
    gen: int,
    champion_level: np.ndarray,
    champion_result: dict,
    champion_fitness: float,
    champion_game_text: str,
    history: list,
    backend: Optional[NodeJSPuzzleScriptBackend] = None,
    render_gif: bool = False,
    gif_frame_duration: float = 0.05,
    gif_scale: int = 10,
) -> None:
    base = f"evolved_lv{level_i}"
    level_path = os.path.join(out_dir, f"{base}_level.txt")
    with open(level_path, "w", encoding="utf-8") as f:
        for row in champion_level:
            f.write("".join(row) + "\n")

    game_text_path = os.path.join(out_dir, f"{base}_game.txt")
    with open(game_text_path, "w", encoding="utf-8") as f:
        f.write(champion_game_text)

    result_path = os.path.join(out_dir, f"{base}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "game": game,
                "level_i": level_i,
                "gen": gen,
                "fitness": champion_fitness,
                **champion_result,
                "history": history,
            },
            f,
            indent=2,
        )

    if render_gif:
        if backend is None:
            raise ValueError("backend is required when render_gif=True")
        gif_path = os.path.join(out_dir, f"{base}_gen{gen}.gif")
        backend.render_gif(
            game_text=champion_game_text,
            level_i=level_i,
            actions=champion_result["actions"],
            gif_path=gif_path,
            frame_duration_s=gif_frame_duration,
            scale=gif_scale,
        )


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
    depth_increase_threshold: float = 0.95,
) -> None:
    initial_max_steps = max_steps
    rng = np.random.default_rng(seed)
    backend = NodeJSPuzzleScriptBackend()
    game_text = load_simplified_game_text(game)
    game_prefix, levels = split_game_levels(game_text)
    original_level = levels[level_i].copy()

    placeable_chars = get_placeable_chars(game_text, original_level)
    player_chars = get_player_chars(game_text, original_level)
    background_char = infer_background_char(original_level, player_chars)
    original_player_count = count_players(original_level, player_chars)
    required_player_count = original_player_count if preserve_players else None

    run_name = make_run_dir_name(
        level_i=level_i,
        pop_size=pop_size,
        n_mutations_min=n_mutations_min,
        n_mutations_max=n_mutations_max,
        max_steps=max_steps,
        algo=algo,
        fitness_mode=fitness_mode,
        seed=seed,
        preserve_players=preserve_players,
    )
    out_dir = os.path.join(EVOLVED_DIR, game, run_name)
    os.makedirs(out_dir, exist_ok=True)

    ckpt = load_checkpoint(out_dir)
    if ckpt is not None:
        start_gen = ckpt["gen"] + 1
        champion_level = ckpt["champion_level"]
        champion_result = ckpt["champion_result"]
        champion_fitness = ckpt["champion_fitness"]
        champion_game_text = ckpt["champion_game_text"]
        history = ckpt["history"]
        rng.bit_generator.state = ckpt["rng_state"]
        print(f"Resumed from generation {ckpt['gen']}.")
    else:
        original_result = evaluate_level(
            backend=backend,
            game_text=game_text,
            level_i=level_i,
            algo=algo,
            max_steps=max_steps,
            timeout_ms=timeout_ms,
        )
        champion_level = original_level.copy()
        champion_result = original_result
        champion_fitness = compute_fitness(original_result, fitness_mode)
        champion_game_text = game_text
        history = [{
            "gen": -1,
            "fitness": champion_fitness,
            **original_result,
        }]
        start_gen = 0
        save_champion(
            out_dir, game, level_i, -1, champion_level, champion_result,
            champion_fitness, champion_game_text, history,
            backend=backend, render_gif=render_gif,
            gif_frame_duration=gif_frame_duration, gif_scale=gif_scale,
        )

    print(f"Starting NodeJS evolution for {game} level {level_i}")
    print(f"Output dir: {out_dir}")
    print(f"Player count: {original_player_count} (enforced={preserve_players})")

    for gen in range(start_gen, n_gens):
        gen_start = time.time()
        best_child_fitness = -float("inf")
        best_child_level = None
        best_child_result = None
        best_child_game_text = None

        for child_i in range(pop_size):
            n_mutations = rng.integers(n_mutations_min, n_mutations_max + 1)
            child_level = mutate_level(
                champion_level,
                rng=rng,
                placeable_chars=placeable_chars,
                player_chars=player_chars,
                background_char=background_char,
                n_mutations=n_mutations,
                required_player_count=required_player_count,
            )
            child_levels = [level.copy() for level in levels]
            child_levels[level_i] = child_level
            child_game_text = join_game_levels(game_prefix, child_levels)
            child_result = evaluate_level(
                backend=backend,
                game_text=child_game_text,
                level_i=level_i,
                algo=algo,
                max_steps=max_steps,
                timeout_ms=timeout_ms,
            )
            child_fitness = compute_fitness(child_result, fitness_mode)

            status = "SOLVED" if child_result["solved"] else "UNSOLVED"
            print(
                f"  Gen {gen:3d} child {child_i}: {status} "
                f"cost={child_result['cost']} states={child_result['generated_states']:,} "
                f"fit={child_fitness:.0f} ({child_result['time']:.2f}s)"
            )

            if child_fitness > best_child_fitness:
                best_child_fitness = child_fitness
                best_child_level = child_level
                best_child_result = child_result
                best_child_game_text = child_game_text

        improved = False
        if best_child_fitness > champion_fitness:
            champion_level = best_child_level
            champion_result = best_child_result
            champion_fitness = best_child_fitness
            champion_game_text = best_child_game_text
            levels[level_i] = champion_level.copy()
            improved = True

        gen_time = time.time() - gen_start
        marker = " ***NEW CHAMPION***" if improved else ""
        print(
            f"Gen {gen:3d}: champion fitness={champion_fitness:.0f} "
            f"cost={champion_result['cost']} states={champion_result['generated_states']:,} "
            f"({gen_time:.2f}s){marker}"
        )

        history.append({
            "gen": gen,
            "fitness": champion_fitness,
            **champion_result,
        })

        if improved:
            save_champion(
                out_dir, game, level_i, gen, champion_level, champion_result,
                champion_fitness, champion_game_text, history,
                backend=backend, render_gif=render_gif,
                gif_frame_duration=gif_frame_duration, gif_scale=gif_scale,
            )

        save_checkpoint(
            out_dir, gen, champion_level, champion_result,
            champion_fitness, champion_game_text, history, rng,
        )

        # ---- Adaptive depth increase ----
        if (depth_increase_threshold > 0
                and champion_result.get("solved")
                and champion_result["generated_states"] >= depth_increase_threshold * max_steps):
            max_steps += initial_max_steps
            print(f"  >> Depth increased: max_steps now {max_steps:,} "
                  f"(+{initial_max_steps:,})")

    print(f"Finished. Best fitness: {champion_fitness:.0f}")
    print(f"Results in {out_dir}")


def main(cfg: EvolveLevelNodeJSConfig) -> None:
    if cfg.game is None:
        raise ValueError("`game` must be set, e.g. `python evolve_level_nodejs.py game=sokoban_basic`.")
    if cfg.algo not in {"bfs", "astar", "gbfs", "mcts"}:
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
        depth_increase_threshold=cfg.depth_increase_threshold,
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="evolve_level_nodejs")
def main_launch(cfg: EvolveLevelNodeJSConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    main_launch()
