"""Train an NCA world model on a PuzzleScript game.

The NCA learns to predict the next game state given the current state and
player action: f(state_t, action_t) -> state_{t+1}.

Trajectories are collected from random rollouts and search (BFS/A*) via the
C++ PuzzleScript backend.

Usage:
    python train_nca_world_model.py --game pipe_bend
    python train_nca_world_model.py --game pipe_bend --serve
    python train_nca_world_model.py --game sokoban_basic --n_nca_steps 8 --lr 3e-4
"""
import argparse
import base64
import io
import json
import os
import pickle
import time

import flax.linen as nn
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser
from tokenize_game import tokenize_game, get_game_tree_from_js, VOCAB_SIZE

N_ACTIONS = 5


# ---------------------------------------------------------------------------
# 1. Data collection with per-game cache
# ---------------------------------------------------------------------------
#
# Cache layout:
#   rollout_data/{game}/level_{i}/random.npz
#   rollout_data/{game}/level_{i}/search_{algo}_{budget}_{timeout}.npz
#
# Random cache stores episodes contiguously with an episode boundary index.
# If an experiment needs more episodes than cached, only the deficit is collected
# and appended. Search caches are keyed by (algo, budget, timeout) — if the file
# exists the data is reused as-is.

ROLLOUT_CACHE_DIR = "rollout_data"


def _cache_dir(game_name: str, level_i: int) -> str:
    return os.path.join(ROLLOUT_CACHE_DIR, game_name, f"level_{level_i}")


def _load_npz_dict(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    cached = np.load(path)
    return {k: cached[k] for k in cached.files}


def _save_npz_dict(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)


def _dat_to_multihot(dat: list[int], n_objs: int, width: int, height: int) -> np.ndarray:
    """Convert bitpacked int32 state vector to (n_objs, H, W) uint8 multihot."""
    stride_obj = (n_objs + 31) // 32
    obs = np.zeros((n_objs, height, width), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            flat_idx = (x * height + y) * stride_obj
            for obj_i in range(n_objs):
                word = obj_i // 32
                bit = obj_i % 32
                if dat[flat_idx + word] & (1 << bit):
                    obs[obj_i, y, x] = 1
    return obs


def collect_unique_transitions(
    json_str: str,
    game_name: str,
    level_i: int = 0,
    max_iters: int = 100_000,
    timeout_ms: int = 60_000,
) -> dict:
    """Collect unique transitions via C++ BFS state-space exploration.

    Every (state, action, next_state) transition visited during BFS is returned.
    Much faster and more thorough than random rollouts.

    Cached at rollout_data/{game}/level_{i}/bfs_transitions.npz.
    """
    from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_bfs

    cache_path = os.path.join(
        _cache_dir(game_name, level_i),
        f"bfs_transitions_{max_iters}_{timeout_ms}.npz",
    )
    cached = _load_npz_dict(cache_path)
    if cached is not None and len(cached["states"]) > 0:
        n = len(cached["states"])
        print(f"  BFS transitions: {n} from cache")
        return {k: cached[k] for k in ("states", "actions", "next_states")}

    # Run C++ BFS transition collector
    engine = Engine()
    engine.load_from_json(json_str)
    engine.load_level(level_i)

    result = collect_transitions_bfs(engine, max_iters=max_iters, timeout_ms=timeout_ms)
    n_trans = len(result.actions)
    n_objs = len(result.id_dict)
    w, h = result.width, result.height

    print(f"  BFS transitions: {n_trans} from {result.iterations} iterations "
          f"({result.time:.3f}s, timeout={result.timeout})")

    if n_trans == 0:
        empty = {
            "states": np.empty((0, n_objs, h, w), dtype=np.uint8),
            "actions": np.empty((0,), dtype=np.int32),
            "next_states": np.empty((0, n_objs, h, w), dtype=np.uint8),
        }
        _save_npz_dict(cache_path, empty)
        return empty

    # Convert bitpacked states to multihot
    states = np.array([_dat_to_multihot(s, n_objs, w, h) for s in result.states],
                      dtype=np.uint8)
    next_states = np.array([_dat_to_multihot(s, n_objs, w, h) for s in result.next_states],
                           dtype=np.uint8)
    actions = np.array(result.actions, dtype=np.int32)

    data = {"states": states, "actions": actions, "next_states": next_states}
    _save_npz_dict(cache_path, data)
    print(f"    Cached -> {cache_path}")
    return data


def _collect_random_episodes(
    json_str: str, level_i: int, n_episodes: int, max_steps: int,
) -> tuple[dict, int]:
    """Collect random rollout transitions. Returns (data_dict, n_wins)."""
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    states, actions, next_states, ep_ends = [], [], [], []
    wins = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(max_steps):
            action = np.random.randint(N_ACTIONS)
            states.append(obs)
            actions.append(action)
            obs, _, done, truncated, info = env.step(action)
            next_states.append(obs)
            if done or truncated:
                wins += int(info.get("won", False))
                break
        ep_ends.append(len(states))
    return {
        "states": np.array(states, dtype=np.uint8),
        "actions": np.array(actions, dtype=np.int32),
        "next_states": np.array(next_states, dtype=np.uint8),
        "ep_ends": np.array(ep_ends, dtype=np.int32),
    }, wins


def collect_random_rollouts(
    json_str: str,
    game_name: str,
    level_i: int = 0,
    n_episodes: int = 500,
    max_steps: int = 200,
) -> dict:
    """Collect random transitions, using/extending the per-game cache."""
    cache_path = os.path.join(_cache_dir(game_name, level_i), "random.npz")
    cached = _load_npz_dict(cache_path)

    cached_eps = 0
    if cached is not None:
        cached_eps = len(cached["ep_ends"])

    if cached_eps >= n_episodes:
        # Take only what we need
        end_idx = int(cached["ep_ends"][n_episodes - 1])
        data = {k: cached[k][:end_idx] for k in ("states", "actions", "next_states")}
        data["ep_ends"] = cached["ep_ends"][:n_episodes]
        print(f"  Random: {n_episodes} eps from cache ({len(data['states'])} transitions)")
        return data

    # Need more episodes
    deficit = n_episodes - cached_eps
    print(f"  Random: {cached_eps} eps cached, collecting {deficit} more...")
    new_data, wins = _collect_random_episodes(json_str, level_i, deficit, max_steps)
    print(f"    Collected {deficit} eps, {len(new_data['states'])} transitions, {wins} wins")

    if cached is not None and len(cached["states"]) > 0:
        # Offset new ep_ends by existing transition count
        offset = len(cached["states"])
        merged = {
            "states": np.concatenate([cached["states"], new_data["states"]]),
            "actions": np.concatenate([cached["actions"], new_data["actions"]]),
            "next_states": np.concatenate([cached["next_states"], new_data["next_states"]]),
            "ep_ends": np.concatenate([cached["ep_ends"], new_data["ep_ends"] + offset]),
        }
    else:
        merged = new_data

    _save_npz_dict(cache_path, merged)
    print(f"    Cache updated: {len(merged['ep_ends'])} total eps -> {cache_path}")

    # Return only the requested amount
    end_idx = int(merged["ep_ends"][n_episodes - 1])
    data = {k: merged[k][:end_idx] for k in ("states", "actions", "next_states")}
    data["ep_ends"] = merged["ep_ends"][:n_episodes]
    return data


def collect_search_trajectories(
    json_str: str,
    game_name: str,
    level_i: int = 0,
    algos: list[str] = ("bfs", "astar"),
    n_steps: int = 100_000,
    timeout_ms: int = 60_000,
    max_episode_steps: int = 200,
) -> dict | None:
    """Collect search trajectories, using the per-game cache."""
    cdir = _cache_dir(game_name, level_i)
    all_states, all_actions, all_next_states = [], [], []

    for algo in algos:
        cache_path = os.path.join(cdir, f"search_{algo}_{n_steps}_{timeout_ms}.npz")
        cached = _load_npz_dict(cache_path)

        if cached is not None and len(cached["states"]) > 0:
            print(f"  {algo}: {len(cached['states'])} transitions from cache")
            all_states.append(cached["states"])
            all_actions.append(cached["actions"])
            all_next_states.append(cached["next_states"])
            continue

        # Run search
        print(f"  Running {algo} ({n_steps} nodes, {timeout_ms}ms timeout)...")
        backend = CppPuzzleScriptBackend()
        backend.load_from_json(json_str)
        env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_episode_steps)

        try:
            backend.load_level("", level_i)
            result = backend.run_search(
                algo, game_text="", level_i=level_i,
                n_steps=n_steps, timeout_ms=timeout_ms,
            )
        except Exception as e:
            print(f"    {algo} failed: {e}")
            # Cache empty result so we don't retry
            _save_npz_dict(cache_path, {
                "states": np.empty((0,), dtype=np.uint8),
                "actions": np.empty((0,), dtype=np.int32),
                "next_states": np.empty((0,), dtype=np.uint8),
            })
            continue

        sol_actions = result.actions
        print(f"    {algo}: solved={result.solved}, len={len(sol_actions)}, score={result.score}")

        if not sol_actions:
            _save_npz_dict(cache_path, {
                "states": np.empty((0,), dtype=np.uint8),
                "actions": np.empty((0,), dtype=np.int32),
                "next_states": np.empty((0,), dtype=np.uint8),
            })
            continue

        # Replay to get per-step states
        states, actions, next_states = [], [], []
        obs, _ = env.reset()
        for a in sol_actions:
            states.append(obs)
            actions.append(a)
            obs, _, done, _, _ = env.step(a)
            next_states.append(obs)
            if done:
                break

        search_data = {
            "states": np.array(states, dtype=np.uint8),
            "actions": np.array(actions, dtype=np.int32),
            "next_states": np.array(next_states, dtype=np.uint8),
        }
        _save_npz_dict(cache_path, search_data)
        print(f"    Cached {len(states)} transitions -> {cache_path}")

        all_states.append(search_data["states"])
        all_actions.append(search_data["actions"])
        all_next_states.append(search_data["next_states"])

    if not all_states:
        print("  No search trajectories collected.")
        return None

    total = sum(len(s) for s in all_states)
    print(f"  Search total: {total} transitions")
    return {
        "states": np.concatenate(all_states),
        "actions": np.concatenate(all_actions),
        "next_states": np.concatenate(all_next_states),
    }


def merge_datasets(random_data: dict, search_data: dict | None, search_weight: float) -> dict:
    """Merge random and search data, oversampling search via index duplication.

    Instead of materializing tiled arrays, creates a combined dataset with
    repeated *indices* into the search data to avoid memory blowup.
    """
    if search_data is None or len(search_data["states"]) == 0:
        return {k: random_data[k] for k in ("states", "actions", "next_states")}

    n_random = len(random_data["states"])
    n_search = len(search_data["states"])
    # Cap oversampled search at n_random to avoid memory blowup
    n_search_target = min(int(search_weight * n_random), n_random)
    # Sample with replacement rather than tiling
    rng = np.random.RandomState(0)
    search_idx = rng.randint(0, n_search, size=n_search_target)
    print(f"  Merging: {n_random} random + {n_search_target} search "
          f"(sampled from {n_search})")

    return {
        k: np.concatenate([
            random_data[k],
            search_data[k][search_idx],
        ], axis=0)
        for k in ("states", "actions", "next_states")
    }


# Preset game sets for multi-game training
MULTI_GAME_PRESETS = {
    "synthetic": [
        "push_sokoban_synthetic",    # standard push
        "swap_sokoban_synthetic",    # player-box swap
        "vanish_sokoban_synthetic",  # box vanishes on contact
    ],
    "small": [
        "nekopuzzle",              # 3 objs,  7x8
        "notsnake",                # 3 objs,  5x8
        "blocks",                  # 4 objs, 11x13
        "sokoban_basic",           # 5 objs,  7x6
        "sokoban_match3",          # 5 objs,  7x9
        "Zen_Puzzle_Garden",       # 6 objs, 12x12
        "Multi-word_Dictionary_Game",  # 7 objs,  7x9
        "kettle",                  # 8 objs, 13x15
        "Travelling_salesman",     # 9 objs,  5x5
    ],
}


def _pad_obs(obs: np.ndarray, target_C: int, target_H: int, target_W: int) -> np.ndarray:
    """Pad (N, C, H, W) observations to (N, target_C, target_H, target_W) with zeros."""
    N, C, H, W = obs.shape
    if C == target_C and H == target_H and W == target_W:
        return obs
    padded = np.zeros((N, target_C, target_H, target_W), dtype=obs.dtype)
    padded[:, :C, :H, :W] = obs
    return padded


def collect_multigame_dataset(
    game_names: list[str],
    ps_parser,
    level_i: int | None = None,
    n_random_episodes: int = 500,
    max_episode_steps: int = 200,
    search_algos: list[str] = ("bfs", "astar"),
    n_search_steps: int = 100_000,
    search_timeout_ms: int = 60_000,
    search_weight: float = 5.0,
    data_mode: str = "unique",
) -> tuple[dict, list[dict]]:
    """Collect padded transitions from multiple games.

    Args:
        level_i: If None (default), collect from all levels. If int, collect from that level only.
        data_mode: "unique" for deduplicated random exploration (default),
                   "random_search" for the old random+search+merge approach.

    Returns:
        dataset: merged dict with keys "states", "actions", "next_states", "game_ids"
            all padded to (max_n_objs, max_H, max_W).
        game_infos: list of per-game metadata dicts.
    """
    # First pass: compile all games and get shapes (max across all levels)
    game_infos = []
    for name in game_names:
        print(f"\nCompiling {name}...")
        backend = CppPuzzleScriptBackend()
        json_str = backend.compile_and_serialize(ps_parser, name)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        n_objs = env0.observation_shape[0]
        n_levels = env0.num_levels
        # Get max spatial dims across all levels
        game_max_H, game_max_W = 0, 0
        for li in range(n_levels):
            env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
            _, lH, lW = env_li.observation_shape
            game_max_H = max(game_max_H, lH)
            game_max_W = max(game_max_W, lW)
        # Tokenize game spec
        try:
            tree, canonical_ids = get_game_tree_from_js(ps_parser, name)
            token_ids = tokenize_game(tree, canonical_ids)
        except Exception as e:
            print(f"  WARNING: tokenization failed ({e}), using empty tokens")
            token_ids = []

        print(f"  n_objs={n_objs}, max_shape=({game_max_H}, {game_max_W}), "
              f"n_levels={n_levels}, n_tokens={len(token_ids)}")
        game_infos.append({
            "name": name,
            "json_str": json_str,
            "n_objs": n_objs,
            "H": game_max_H,
            "W": game_max_W,
            "n_levels": n_levels,
            "token_ids": token_ids,
        })

    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    print(f"\nPadded shape: ({max_C}, {max_H}, {max_W})")

    # Second pass: collect and pad transitions
    all_states, all_actions, all_next_states, all_game_ids = [], [], [], []

    for game_id, info in enumerate(game_infos):
        name = info["name"]
        json_str = info["json_str"]
        levels = [level_i] if level_i is not None else list(range(info["n_levels"]))
        print(f"\n[{game_id+1}/{len(game_infos)}] Collecting data for {name} "
              f"({len(levels)} level{'s' if len(levels) != 1 else ''})...")

        game_states, game_actions, game_next_states = [], [], []
        for li in levels:
            print(f"  Level {li}:")
            if data_mode == "unique":
                level_data = collect_unique_transitions(
                    json_str, name, level_i=li,
                    max_iters=n_search_steps,
                    timeout_ms=search_timeout_ms,
                )
            else:
                random_data = collect_random_rollouts(
                    json_str, name, level_i=li,
                    n_episodes=n_random_episodes,
                    max_steps=max_episode_steps,
                )
                search_data = collect_search_trajectories(
                    json_str, name, level_i=li,
                    algos=search_algos,
                    n_steps=n_search_steps,
                    timeout_ms=search_timeout_ms,
                    max_episode_steps=max_episode_steps,
                )
                level_data = merge_datasets(random_data, search_data, search_weight)
            # Pad each level's data to the global max before concatenating
            game_states.append(_pad_obs(level_data["states"], max_C, max_H, max_W))
            game_actions.append(level_data["actions"])
            game_next_states.append(_pad_obs(level_data["next_states"], max_C, max_H, max_W))

        states = np.concatenate(game_states)
        actions = np.concatenate(game_actions)
        next_states = np.concatenate(game_next_states)
        n_trans = len(states)
        info["n_transitions"] = n_trans

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_game_ids.append(np.full(n_trans, game_id, dtype=np.int32))

        changed = (states != next_states).any(axis=(1, 2, 3))
        print(f"  {name}: {n_trans} transitions, {changed.sum()} with state change "
              f"({100*changed.mean():.1f}%)")

    # Pad token sequences to common length and expand to per-transition
    max_tok_len = max(len(g["token_ids"]) for g in game_infos)
    max_tok_len = max(max_tok_len, 1)  # at least 1
    per_game_tokens = []
    per_game_masks = []
    for info in game_infos:
        tids = info["token_ids"]
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        per_game_tokens.append(padded)
        per_game_masks.append(mask)
    per_game_tokens = np.array(per_game_tokens)  # (n_games, max_tok_len)
    per_game_masks = np.array(per_game_masks)    # (n_games, max_tok_len)

    # Expand to per-transition using game_ids
    game_ids_all = np.concatenate(all_game_ids, axis=0)
    all_token_ids = per_game_tokens[game_ids_all]      # (N, max_tok_len)
    all_token_masks = per_game_masks[game_ids_all]      # (N, max_tok_len)

    merged = {
        "states": np.concatenate(all_states, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "next_states": np.concatenate(all_next_states, axis=0),
        "game_ids": game_ids_all,
        "game_tokens": all_token_ids,
        "game_masks": all_token_masks,
    }
    print(f"\nTotal multi-game dataset: {len(merged['states'])} transitions "
          f"from {len(game_infos)} games, padded to ({max_C}, {max_H}, {max_W}), "
          f"max_tokens={max_tok_len}")
    return merged, game_infos


# ---------------------------------------------------------------------------
# 2. NCA world model (Flax/JAX)
# ---------------------------------------------------------------------------

class NCAWorldModel(nn.Module):
    """Neural Cellular Automaton world model.

    Given multihot state (C, H, W) and a one-hot action (5,), predicts the
    next multihot state. The action is broadcast spatially and concatenated
    as extra input channels.

    The NCA applies `n_steps` shared-weight local update rules (3x3 conv),
    with skip connections from the input at each step.
    """
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1  # set to n_objs at init time
    return_intermediates: bool = False

    @nn.compact
    def __call__(self, state, action_onehot):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
        Returns:
            If return_intermediates is False:
                logits: (B, C, H, W) float32 logits for next state.
            If return_intermediates is True:
                (logits, intermediates) where intermediates is a dict with:
                    "hidden": list of (B, H, W, n_hid) per NCA step
                    "readouts": list of (B, C, H, W) logits per NCA step
        """
        B, C, H, W = state.shape
        # NHWC for Flax convolutions
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)

        # Broadcast action to spatial dims: (B, 5) -> (B, H, W, 5)
        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))

        # Input = state channels + action channels
        inp = jnp.concatenate([x, act], axis=-1)  # (B, H, W, C+5)

        # Embed to hidden
        h = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="embed")(inp)
        h = nn.relu(h)

        # Shared-weight NCA update steps
        nca_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name="nca_conv")
        nca_gate = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="nca_gate")
        readout_conv = nn.Conv(self.n_out, (1, 1), padding="SAME", name="readout")

        hidden_steps = []
        readout_steps = []

        for _ in range(self.n_steps):
            h_in = jnp.concatenate([h, inp], axis=-1)  # skip connection
            dh = nca_conv(h_in)
            dh = nn.relu(dh)
            dh = nca_gate(dh)
            h = h + dh  # residual update
            h = nn.relu(h)

            if self.return_intermediates:
                hidden_steps.append(h)
                step_logits = readout_conv(h).transpose(0, 3, 1, 2)
                readout_steps.append(step_logits)

        # Final readout
        logits = readout_conv(h)
        logits = logits.transpose(0, 3, 1, 2)

        if self.return_intermediates:
            return logits, {"hidden": hidden_steps, "readouts": readout_steps}
        return logits


# ---------------------------------------------------------------------------
# 2a. Conditional NCA world model (game-spec encoder + FiLM)
# ---------------------------------------------------------------------------

class GameSpecEncoder(nn.Module):
    """Transformer encoder: game token sequence -> latent z.

    Prepends a learnable [CLS] token. Output is the CLS representation
    projected to d_z dimensions.
    """
    vocab_size: int = 142       # VOCAB_SIZE + 1 for CLS
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192      # max tokens + 1 for CLS
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, mask, deterministic=True):
        """
        Args:
            token_ids: (B, S) int32 token IDs (without CLS, PAD=0).
            mask: (B, S) bool, True for real tokens.
        Returns:
            z: (B, d_z) float32 latent vector.
        """
        B, S = token_ids.shape
        # Token + positional embeddings
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_embed")
        pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_embed")

        # CLS token (position 0)
        cls_tok = jnp.full((B, 1), self.vocab_size - 1, dtype=jnp.int32)  # CLS token ID
        cls_mask = jnp.ones((B, 1), dtype=jnp.bool_)

        # Prepend CLS
        all_tokens = jnp.concatenate([cls_tok, token_ids], axis=1)  # (B, 1+S)
        all_mask = jnp.concatenate([cls_mask, mask], axis=1)        # (B, 1+S)

        L = all_tokens.shape[1]
        positions = jnp.arange(L)[None, :]  # (1, L)
        x = tok_emb(all_tokens) + pos_emb(positions)  # (B, L, d_model)

        # Attention mask: (B, 1, 1, L) for broadcast — 0 for real, -1e9 for pad
        attn_mask = jnp.where(all_mask[:, None, None, :], 0.0, -1e9)

        # Transformer encoder layers
        for i in range(self.n_layers):
            # Pre-norm self-attention
            y = nn.LayerNorm(name=f"ln1_{i}")(x)
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.d_model,
                name=f"attn_{i}",
            )(y, y, mask=attn_mask, deterministic=deterministic)
            x = x + y
            # Pre-norm FFN
            y = nn.LayerNorm(name=f"ln2_{i}")(x)
            y = nn.Dense(self.d_model * 4, name=f"ff1_{i}")(y)
            y = nn.gelu(y)
            y = nn.Dense(self.d_model, name=f"ff2_{i}")(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)

        # CLS output -> z
        cls_out = x[:, 0, :]  # (B, d_model)
        z = nn.Dense(self.d_z, name="z_proj")(cls_out)
        return z


class ConditionalNCAWorldModel(nn.Module):
    """NCA world model conditioned on a game specification via FiLM.

    The game spec (token sequence) is encoded by a transformer into a latent z.
    At each NCA step, z modulates the hidden state update via
    FiLM: dh = gamma(z) * dh + beta(z).
    """
    # NCA params
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1
    return_intermediates: bool = False
    # Encoder params
    vocab_size: int = 142
    d_model: int = 64
    n_heads: int = 4
    n_enc_layers: int = 2
    d_z: int = 64
    max_seq_len: int = 192

    @nn.compact
    def __call__(self, state, action_onehot, game_tokens, game_mask):
        """
        Args:
            state: (B, C, H, W) float32 multihot level.
            action_onehot: (B, 5) float32 one-hot action.
            game_tokens: (B, S) int32 tokenized game spec.
            game_mask: (B, S) bool mask (True for real tokens).
        Returns:
            logits or (logits, intermediates) depending on return_intermediates.
        """
        B, C, H, W = state.shape

        # --- Encode game spec → z ---
        z = GameSpecEncoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_enc_layers,
            d_z=self.d_z,
            max_seq_len=self.max_seq_len,
            name="game_encoder",
        )(game_tokens, game_mask)  # (B, d_z)

        # --- FiLM parameters from z (shared across NCA steps) ---
        # Initialize gamma near 1, beta near 0 for identity-like start
        gamma = nn.Dense(
            self.n_hid, name="film_gamma",
            kernel_init=nn.initializers.zeros,
        )(z) + 1.0  # (B, n_hid)
        beta = nn.Dense(
            self.n_hid, name="film_beta",
            kernel_init=nn.initializers.zeros,
        )(z)  # (B, n_hid)

        # Broadcast for spatial dims: (B, 1, 1, n_hid)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]

        # --- NCA forward (same as NCAWorldModel, with FiLM) ---
        x = state.transpose(0, 2, 3, 1)  # (B, H, W, C)

        act = action_onehot[:, None, None, :]
        act = jnp.broadcast_to(act, (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act], axis=-1)

        h = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="embed")(inp)
        h = nn.relu(h)

        nca_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name="nca_conv")
        nca_gate = nn.Conv(self.n_hid, (1, 1), padding="SAME", name="nca_gate")
        readout_conv = nn.Conv(self.n_out, (1, 1), padding="SAME", name="readout")

        hidden_steps = []
        readout_steps = []

        for _ in range(self.n_steps):
            h_in = jnp.concatenate([h, inp], axis=-1)
            dh = nca_conv(h_in)
            dh = nn.relu(dh)
            dh = nca_gate(dh)
            # FiLM modulation on the update
            dh = gamma * dh + beta
            h = h + dh
            h = nn.relu(h)

            if self.return_intermediates:
                hidden_steps.append(h)
                step_logits = readout_conv(h).transpose(0, 3, 1, 2)
                readout_steps.append(step_logits)

        logits = readout_conv(h)
        logits = logits.transpose(0, 3, 1, 2)

        if self.return_intermediates:
            return logits, {"hidden": hidden_steps, "readouts": readout_steps}
        return logits


# ---------------------------------------------------------------------------
# 2b. Activation visualization
# ---------------------------------------------------------------------------

def _make_channel_grid(activations, ncols=None, pad=1, normalize=True, scale=3):
    """Arrange (H, W, C) activations into a single image grid.

    Each channel cell is upscaled by `scale` for visibility.
    Returns an (grid_H, grid_W) float array suitable for colormap application.
    """
    H, W, C = activations.shape
    sH, sW = H * scale, W * scale
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(C)))
    nrows = int(np.ceil(C / ncols))
    grid = np.zeros((nrows * (sH + pad) - pad, ncols * (sW + pad) - pad), dtype=np.float32)

    for i in range(C):
        r, c = divmod(i, ncols)
        y0 = r * (sH + pad)
        x0 = c * (sW + pad)
        ch = np.array(activations[:, :, i], dtype=np.float32)
        if normalize:
            lo, hi = ch.min(), ch.max()
            ch = (ch - lo) / (hi - lo + 1e-8)
        # Upscale with nearest neighbor
        ch = np.repeat(np.repeat(ch, scale, axis=0), scale, axis=1)
        grid[y0:y0 + sH, x0:x0 + sW] = ch

    return grid


def _apply_colormap(gray, cmap_name="viridis"):
    """Convert (H, W) float in [0,1] to (H, W, 3) uint8 via matplotlib colormap."""
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(gray)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _labeled_channel_grid(logits_chw, obj_names, pad=2, scale=4):
    """Render per-object output channels as a labeled grid.

    Always shows ALL channels in order, labeled with object names.

    Args:
        logits_chw: (C, H, W) logits or probabilities.
        obj_names: list of C object name strings.
        scale: upscale each cell by this factor for readability.
    Returns:
        (grid_H, grid_W, 3) uint8 image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    C, H, W = logits_chw.shape
    logits_f = np.clip(np.array(logits_chw, dtype=np.float32), -50, 50)
    probs = 1.0 / (1.0 + np.exp(-logits_f))

    # Always show all channels in order
    ncols = min(10, C)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.5),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(C):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.imshow(probs[idx], vmin=0, vmax=1, cmap="magma",
                  interpolation="nearest", aspect="equal")
        name = obj_names[idx] if idx < len(obj_names) else f"ch{idx}"
        # Truncate long names
        if len(name) > 14:
            name = name[:12] + ".."
        ax.set_title(name, fontsize=5, pad=2)

    fig.tight_layout(pad=0.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


def visualize_nca_step(
    model: "NCAWorldModel",
    params,
    state_obs: np.ndarray,
    action: int,
    obj_names: list[str],
    backend: "CppPuzzleScriptBackend",
    grid_w: int,
    grid_h: int,
):
    """Run one NCA step with intermediates and return a visualization image.

    Returns a tall image with rows:
      - Input state (rendered)
      - For each NCA step: hidden activation grid + discrete output channels
      - Final predicted state (rendered)
    """
    # Create model variant that returns intermediates
    model_viz = NCAWorldModel(n_hid=model.n_hid, n_steps=model.n_steps,
                              n_out=model.n_out, return_intermediates=True)
    state_jnp = jnp.array(state_obs[None], dtype=jnp.float32)
    a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
    logits, intermediates = model_viz.apply(params, state_jnp, a_oh)

    sections = []

    # Input state rendered
    input_frame = backend.render_frame_from_objects(
        _multihot_to_objects(state_obs), grid_w, grid_h
    )
    sections.append(input_frame)

    # Per-step hidden activations + readouts
    for step_i, (h, readout) in enumerate(
        zip(intermediates["hidden"], intermediates["readouts"])
    ):
        h_np = np.array(h[0])   # (H, W, n_hid)
        r_np = np.array(readout[0])  # (C, H, W)

        # Hidden channel grid
        hid_grid = _make_channel_grid(h_np)
        hid_img = _apply_colormap(hid_grid)

        # Labeled output channels
        out_img = _labeled_channel_grid(r_np, obj_names)

        # Match widths for stacking
        target_w = max(hid_img.shape[1], out_img.shape[1], input_frame.shape[1])
        hid_img = _pad_to_width(hid_img, target_w)
        out_img = _pad_to_width(out_img, target_w)

        sections.append(hid_img)
        sections.append(out_img)

    # Final predicted state rendered
    pred_obs = np.array((jax.nn.sigmoid(logits[0]) > 0.5), dtype=np.uint8)
    pred_frame = backend.render_frame_from_objects(
        _multihot_to_objects(pred_obs), grid_w, grid_h
    )
    sections.append(pred_frame)

    # Match all widths
    max_w = max(s.shape[1] for s in sections)
    sections = [_pad_to_width(s, max_w) for s in sections]

    return np.concatenate(sections, axis=0)


def _pad_to_width(img, target_w):
    """Pad or resize image to target width, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w == target_w:
        return img
    if w < target_w:
        pad = np.zeros((h, target_w - w, 3), dtype=img.dtype)
        return np.concatenate([img, pad], axis=1)
    # Resize down
    import PIL.Image
    pil = PIL.Image.fromarray(img)
    new_h = int(h * target_w / w)
    pil = pil.resize((target_w, new_h), PIL.Image.NEAREST)
    return np.array(pil)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def make_train_step(model, optimizer, conditional=False):
    """Returns a JIT-compiled train step."""

    if conditional:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states,
                        game_tokens, game_masks):
            def loss_fn(params):
                logits = model.apply(params, states, action_onehots,
                                     game_tokens, game_masks)
                bce = optax.sigmoid_binary_cross_entropy(logits, next_states)
                loss = bce.mean()
                preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                acc = (preds == next_states).mean()
                changed = (states != next_states)
                n_changed = changed.sum()
                changed_correct = ((preds == next_states) & changed).sum()
                change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
                return loss, (acc, change_acc)

            (loss, (acc, change_acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss, acc, change_acc
    else:
        @jax.jit
        def train_step(params, opt_state, states, action_onehots, next_states):
            def loss_fn(params):
                logits = model.apply(params, states, action_onehots)
                bce = optax.sigmoid_binary_cross_entropy(logits, next_states)
                loss = bce.mean()
                preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                acc = (preds == next_states).mean()
                changed = (states != next_states)
                n_changed = changed.sum()
                changed_correct = ((preds == next_states) & changed).sum()
                change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
                return loss, (acc, change_acc)

            (loss, (acc, change_acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss, acc, change_acc

    return train_step


def train(
    model: NCAWorldModel,
    dataset: dict,
    lr: float = 1e-3,
    n_updates: int = 5000,
    batch_size: int = 64,
    seed: int = 0,
    log_interval: int = 100,
    save_dir: str = "logs_nca_wm",
    init_params=None,
    start_step: int = 0,
):
    """Train (or resume training) the NCA world model.

    If init_params is provided, resumes from those weights instead of
    initializing from scratch. start_step offsets the step counter for logging.
    Supports both conditional (ConditionalNCAWorldModel) and unconditional models.
    """
    conditional = "game_tokens" in dataset and isinstance(model, ConditionalNCAWorldModel)

    rng = jax.random.PRNGKey(seed)
    # Keep data on CPU; only send batches to GPU
    states_np = dataset["states"]        # (N, C, H, W) uint8
    actions_np = dataset["actions"]      # (N,) int32
    next_states_np = dataset["next_states"]  # (N, C, H, W) uint8
    if conditional:
        tokens_np = dataset["game_tokens"]     # (N, S) int32
        masks_np = dataset["game_masks"]       # (N, S) bool

    n_data = states_np.shape[0]
    n_objs, H, W = states_np.shape[1:]
    mode_str = "conditional" if conditional else "unconditional"
    print(f"Training ({mode_str}) on {n_data} transitions, obs=({n_objs},{H},{W}), "
          f"batch_size={batch_size}, lr={lr}")

    # Init or resume model
    rng, init_rng = jax.random.split(rng)
    dummy_state = jnp.zeros((1, n_objs, H, W), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)
    if init_params is not None:
        params = init_params
        print(f"Resuming from step {start_step}")
    else:
        if conditional:
            max_tok_len = tokens_np.shape[1]
            dummy_tokens = jnp.zeros((1, max_tok_len), dtype=jnp.int32)
            dummy_mask = jnp.zeros((1, max_tok_len), dtype=jnp.bool_)
            params = model.init(init_rng, dummy_state, dummy_action,
                                dummy_tokens, dummy_mask)
        else:
            params = model.init(init_rng, dummy_state, dummy_action)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model params: {n_params:,}")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer, conditional=conditional)

    os.makedirs(save_dir, exist_ok=True)
    losses, accs, change_accs = [], [], []
    t0 = time.time()
    np_rng = np.random.RandomState(seed + start_step)

    for step in range(n_updates):
        idx = np_rng.randint(0, n_data, size=batch_size)
        s = jnp.array(states_np[idx], dtype=jnp.float32)
        a_int = actions_np[idx]
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[a_int])
        ns = jnp.array(next_states_np[idx], dtype=jnp.float32)

        if conditional:
            gt = jnp.array(tokens_np[idx])
            gm = jnp.array(masks_np[idx])
            params, opt_state, loss, acc, change_acc = train_step(
                params, opt_state, s, a_oh, ns, gt, gm
            )
        else:
            params, opt_state, loss, acc, change_acc = train_step(
                params, opt_state, s, a_oh, ns
            )
        losses.append(float(loss))
        accs.append(float(acc))
        change_accs.append(float(change_acc))

        global_step = start_step + step + 1
        if global_step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:])
            avg_acc = np.mean(accs[-log_interval:])
            avg_cacc = np.mean(change_accs[-log_interval:])
            elapsed = time.time() - t0
            print(f"  step {global_step}/{start_step + n_updates}  loss={avg_loss:.4f}  "
                  f"acc={avg_acc:.4f}  change_acc={avg_cacc:.4f}  ({elapsed:.1f}s)")
            if wandb.run is not None:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "train/change_acc": avg_cacc,
                }, step=global_step)

    # Save checkpoint
    final_step = start_step + n_updates
    ckpt_path = os.path.join(save_dir, "params.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    # Also save the step count so we can resume later
    meta_path = os.path.join(save_dir, "train_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["total_steps"] = final_step
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved params to {ckpt_path} (total steps: {final_step})")

    return params, losses, accs, change_accs, final_step


# ---------------------------------------------------------------------------
# 4. Evaluation — multi-step rollout with the learned world model
# ---------------------------------------------------------------------------

def evaluate_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    level_i: int = 0,
    n_episodes: int = 10,
    max_steps: int = 50,
    save_dir: str | None = None,
):
    """Roll out the world model alongside the real env and measure divergence."""
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = jax.jit(model.apply)

    all_l1_errors = []
    for ep_i in range(n_episodes):
        real_obs, _ = env.reset()
        pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
        ep_errors = []

        for t in range(max_steps):
            action = np.random.randint(N_ACTIONS)
            a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

            # World model prediction
            logits = apply_fn(params, pred_state, a_oh)
            pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

            # Real env step
            real_obs, _, done, truncated, _ = env.step(action)
            real = jnp.array(real_obs[None], dtype=jnp.float32)

            l1 = float(jnp.abs(pred_state - real).sum())
            ep_errors.append(l1)
            if done or truncated:
                break

        all_l1_errors.append(ep_errors)

    # Report per-step average divergence
    max_len = max(len(e) for e in all_l1_errors)
    padded = np.full((n_episodes, max_len), np.nan)
    for i, e in enumerate(all_l1_errors):
        padded[i, :len(e)] = e
    mean_per_step = np.nanmean(padded, axis=0)
    print(f"Eval ({n_episodes} eps): step-1 L1={mean_per_step[0]:.1f}, "
          f"step-5 L1={mean_per_step[min(4, len(mean_per_step)-1)]:.1f}, "
          f"step-20 L1={mean_per_step[min(19, len(mean_per_step)-1)]:.1f}")

    if save_dir:
        np.savez(os.path.join(save_dir, "eval_divergence.npz"),
                 mean_per_step=mean_per_step, all_errors=padded)

    return mean_per_step


def _run_eval_rollout(
    apply_fn, params, json_str: str,
    level_i: int, n_objs: int,
    max_C: int, max_H: int, max_W: int,
    actions: list[int] | None = None,
    max_steps: int = 50,
    game_tokens: np.ndarray | None = None,
    game_mask: np.ndarray | None = None,
) -> dict:
    """Run a single eval rollout and return per-step metrics.

    Returns dict with:
        wrong_tiles: (T,) int — number of wrong tile bits per step
        tile_error_rate: (T,) float — wrong_tiles / total_tiles per step
        total_tiles: int — n_objs * H * W for this level
    """
    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])  # (1, S)
        gm = jnp.array(game_mask[None])    # (1, S)

    env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                             max_episode_steps=max_steps if actions is None else len(actions))
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape
    total_tiles = n_objs * H * W
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)

    n_steps = len(actions) if actions else max_steps
    wrong_tiles = []
    for t in range(n_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        if conditional:
            logits = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_obs, _, done, truncated, _ = env.step(action)

        # Count wrong tile bits on the game's actual extent
        pred_binary = np.array(pred_state[0, :n_objs, :H, :W] > 0.5, dtype=np.uint8)
        n_wrong = int((pred_binary != real_obs).sum())
        wrong_tiles.append(n_wrong)

        # Re-pad real obs for next step's prediction
        real_padded = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        # NCA runs open-loop (pred feeds into pred), but we track vs real

        if done or truncated:
            break

    wrong_tiles = np.array(wrong_tiles)
    return {
        "wrong_tiles": wrong_tiles,
        "tile_error_rate": wrong_tiles / total_tiles,
        "total_tiles": total_tiles,
    }


def evaluate_multigame(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser=None,
    n_random_episodes: int = 10,
    max_steps: int = 50,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = 60_000,
    save_dir: str | None = None,
):
    """Evaluate per game, per level, per rollout type (random + search).

    Reports tile discrepancy counts and error rates.
    """
    max_C = model.n_out
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    conditional = isinstance(model, ConditionalNCAWorldModel)
    apply_fn = jax.jit(model.apply)

    # Prepare padded token arrays for conditional eval
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_data(info):
        if not conditional:
            return {}, {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return padded, mask

    results = {}  # results[game][level_i][rollout_type] = dict of metrics

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        game_tokens, game_mask = _get_token_data(info)
        cond_kwargs = {}
        if conditional:
            cond_kwargs = {"game_tokens": game_tokens, "game_mask": game_mask}
        game_results = {}

        for level_i in range(n_levels):
            level_results = {}

            # --- Random rollouts ---
            random_errors = []
            for _ in range(n_random_episodes):
                r = _run_eval_rollout(
                    apply_fn, params, json_str, level_i, n_objs,
                    max_C, max_H, max_W, max_steps=max_steps,
                    **cond_kwargs,
                )
                random_errors.append(r["tile_error_rate"])

            # Pad to same length and average
            max_len = max(len(e) for e in random_errors)
            padded = np.full((n_random_episodes, max_len), np.nan)
            for i, e in enumerate(random_errors):
                padded[i, :len(e)] = e
            level_results["random"] = {
                "mean_error_rate": np.nanmean(padded, axis=0),
                "total_tiles": r["total_tiles"],
            }

            # --- Search rollouts ---
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                # Try cache first, then run search
                cache_path = os.path.join(
                    _cache_dir(name, level_i),
                    f"search_{algo}_{search_n_steps}_{search_timeout_ms}.npz"
                )
                cached = _load_npz_dict(cache_path)
                if cached is not None and len(cached["actions"]) > 0:
                    sol_actions = cached["actions"].tolist()
                else:
                    try:
                        backend_search.load_level("", level_i)
                        result = backend_search.run_search(
                            algo, game_text="", level_i=level_i,
                            n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                        )
                        if not result.actions:
                            continue
                        sol_actions = list(result.actions)
                    except Exception:
                        continue

                r = _run_eval_rollout(
                    apply_fn, params, json_str, level_i, n_objs,
                    max_C, max_H, max_W, actions=sol_actions,
                    **cond_kwargs,
                )
                level_results[algo] = {
                    "error_rate": r["tile_error_rate"],
                    "wrong_tiles": r["wrong_tiles"],
                    "total_tiles": r["total_tiles"],
                    "n_steps": len(sol_actions),
                }

            game_results[level_i] = level_results

        results[name] = game_results

        # Print summary for this game
        for level_i, level_results in game_results.items():
            for rtype, metrics in level_results.items():
                if rtype == "random":
                    er = metrics["mean_error_rate"]
                else:
                    er = metrics["error_rate"]
                s1 = er[0] if len(er) > 0 else float("nan")
                s5 = er[min(4, len(er)-1)] if len(er) > 0 else float("nan")
                sfinal = er[-1] if len(er) > 0 else float("nan")
                n = len(er)
                print(f"  {name} L{level_i} {rtype:<8} "
                      f"step1={s1:.3f}  step5={s5:.3f}  final={sfinal:.3f}  "
                      f"({n} steps, {metrics['total_tiles']} tiles)")

    # Log to wandb
    if wandb.run is not None:
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    er = metrics.get("mean_error_rate", metrics.get("error_rate"))
                    if er is not None and len(er) > 0:
                        wandb.log({
                            f"eval/{name}/L{level_i}/{rtype}/step1_err": float(er[0]),
                            f"eval/{name}/L{level_i}/{rtype}/step5_err": float(er[min(4, len(er)-1)]),
                            f"eval/{name}/L{level_i}/{rtype}/final_err": float(er[-1]),
                        }, commit=False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Flatten to saveable arrays
        save_dict = {}
        for name, game_results in results.items():
            for level_i, level_results in game_results.items():
                for rtype, metrics in level_results.items():
                    key = f"{name}_L{level_i}_{rtype}"
                    er = metrics.get("mean_error_rate", metrics.get("error_rate"))
                    if er is not None:
                        save_dict[f"{key}_error_rate"] = er
                    if "wrong_tiles" in metrics:
                        save_dict[f"{key}_wrong_tiles"] = metrics["wrong_tiles"]
        np.savez(os.path.join(save_dir, "eval_multigame.npz"), **save_dict)

    return results


def _render_rollout_frames(
    model, params, apply_fn, json_str, backend_render,
    level_i, n_objs, grid_h, grid_w, max_H, max_W,
    n_steps, actions=None, label="", obj_names=None,
    game_tokens=None, game_mask=None,
):
    """Run a rollout and return labeled frames.

    Each frame is composed vertically:
      - Banner with label
      - Game renders: real | NCA prediction
      - Hidden activation grid (final NCA step)
      - Per-object output channel predictions
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])
        gm = jnp.array(game_mask[None])

    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=n_steps)
    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, max_H, max_W)
    frames = []

    # Model variant with intermediates
    if conditional:
        model_viz = ConditionalNCAWorldModel(
            n_hid=model.n_hid, n_steps=model.n_steps, n_out=model.n_out,
            vocab_size=model.vocab_size, d_model=model.d_model,
            n_heads=model.n_heads, n_enc_layers=model.n_enc_layers,
            d_z=model.d_z, max_seq_len=model.max_seq_len,
            return_intermediates=True,
        )
    else:
        model_viz = NCAWorldModel(n_hid=model.n_hid, n_steps=model.n_steps,
                                  n_out=model.n_out, return_intermediates=True)
    viz_fn = jax.jit(model_viz.apply)

    # Object names for labeling channels (pad with generic names for extra channels)
    if obj_names is None:
        obj_names = [f"ch{i}" for i in range(model.n_out)]
    while len(obj_names) < model.n_out:
        obj_names.append(f"pad{len(obj_names)}")

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = PIL.ImageFont.load_default()

    max_steps = len(actions) if actions else n_steps
    for t in range(max_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # --- Game renders ---
        real_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        game_row = np.concatenate([real_frame, pred_frame], axis=1)

        # --- NCA internals ---
        if conditional:
            logits, intermediates = viz_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, intermediates = viz_fn(params, pred_state, a_oh)

        # Hidden activations from the final NCA step
        final_hidden = np.array(intermediates["hidden"][-1][0])  # (H, W, n_hid)
        hid_grid = _make_channel_grid(final_hidden)
        hid_img = _apply_colormap(hid_grid)

        # Per-object output logits from the final NCA step
        final_readout = np.array(intermediates["readouts"][-1][0])  # (C, H, W)
        obj_img = _labeled_channel_grid(final_readout, obj_names)

        # --- Compose frame vertically ---
        sections = [game_row, hid_img, obj_img]
        max_w = max(s.shape[1] for s in sections)

        # Add banner
        action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
        text = f"{label} t={t} {action_names.get(action, '?')}  (real | NCA)"
        text_bbox = font.getbbox(text)
        text_w = text_bbox[2] - text_bbox[0] + 8
        banner_h = 18
        banner_w = max(max_w, text_w)
        banner = PIL.Image.new("RGB", (banner_w, banner_h), (0, 0, 0))
        draw = PIL.ImageDraw.Draw(banner)
        draw.text((4, 2), text, fill=(255, 255, 255), font=font)
        banner_arr = np.array(banner)

        # Pad all sections to same width
        final_w = max(banner_w, max_w)
        parts = [banner_arr]
        for s in sections:
            if s.shape[1] < final_w:
                pad = np.zeros((s.shape[0], final_w - s.shape[1], 3), dtype=np.uint8)
                s = np.concatenate([s, pad], axis=1)
            parts.append(s)

        frame = np.concatenate(parts, axis=0)
        frames.append(frame)

        # Step the NCA (use logits already computed)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    return frames


def render_multigame_gifs(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser,
    n_steps_per_game: int = 30,
    save_dir: str = ".",
    step_label: int | None = None,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = 60_000,
):
    """Render a single combined GIF: for each game and level, random rollout then search rollout.

    The GIF filename includes the training step count for easy comparison across checkpoints.
    """
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    conditional = isinstance(model, ConditionalNCAWorldModel)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare padded token arrays for conditional rendering
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_kwargs(info):
        if not conditional:
            return {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return {"game_tokens": padded, "game_mask": mask}

    apply_fn = jax.jit(model.apply)
    all_frames = []

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        cond_kwargs = _get_token_kwargs(info)

        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, name)

        # Get object names for this game
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        obj_names = getattr(env0, "_canonical_ids", None)

        for level_i in range(n_levels):
            env_li = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10)
            _, grid_h, grid_w = env_li.observation_shape

            # Random rollout
            label = f"{name} L{level_i} random"
            print(f"  {label}")
            frames = _render_rollout_frames(
                model, params, apply_fn, json_str, backend_render,
                level_i, n_objs, grid_h, grid_w, max_H, max_W,
                n_steps_per_game, label=label, obj_names=obj_names,
                **cond_kwargs,
            )
            all_frames.extend(frames)

            # Search rollout(s)
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                try:
                    backend_search.load_level("", level_i)
                    result = backend_search.run_search(
                        algo, game_text="", level_i=level_i,
                        n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                    )
                    if not result.actions:
                        continue
                    label = f"{name} L{level_i} {algo} ({'win' if result.solved else 'no win'})"
                    print(f"  {label} ({len(result.actions)} steps)")
                    frames = _render_rollout_frames(
                        model, params, apply_fn, json_str, backend_render,
                        level_i, n_objs, grid_h, grid_w, max_H, max_W,
                        n_steps=len(result.actions), actions=result.actions,
                        label=label, obj_names=obj_names,
                        **cond_kwargs,
                    )
                    all_frames.extend(frames)
                except Exception as e:
                    print(f"    {algo} L{level_i} failed: {e}")

    if all_frames:
        # Pad all frames to the same size (games have different render sizes)
        max_fh = max(f.shape[0] for f in all_frames)
        max_fw = max(f.shape[1] for f in all_frames)
        padded_frames = []
        for f in all_frames:
            pf = np.zeros((max_fh, max_fw, 3), dtype=np.uint8)
            pf[:f.shape[0], :f.shape[1]] = f
            padded_frames.append(pf)

        tag = f"_step{step_label}" if step_label is not None else ""
        gif_path = os.path.join(save_dir, f"multigame_rollout{tag}.gif")
        imageio.mimsave(gif_path, padded_frames, duration=0.2, loop=0)
        print(f"Saved combined rollout GIF ({len(padded_frames)} frames) to {gif_path}")
        if wandb.run is not None:
            wandb.log({"eval/rollout_gif": wandb.Video(gif_path, fps=5, format="gif")})


def _multihot_to_objects(obs: np.ndarray) -> np.ndarray:
    """Convert (n_objs, H, W) multihot to flat objects array for CPP renderer."""
    n_objs, grid_h, grid_w = obs.shape
    stride_obj = (n_objs + 31) // 32
    # Use uint32 for bitwise ops, then view as int32 for C++ compatibility
    objects = np.zeros(grid_w * grid_h * stride_obj, dtype=np.uint32)
    for x in range(grid_w):
        for y in range(grid_h):
            flat_idx = (x * grid_h + y) * stride_obj
            for obj_i in range(n_objs):
                if obs[obj_i, y, x]:
                    word = obj_i // 32
                    bit = obj_i % 32
                    objects[flat_idx + word] |= np.uint32(1 << bit)
    return objects.view(np.int32)


def play_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    save_dir: str = "logs_nca_wm/play",
):
    """Interactive play: step through the NCA world model with keyboard input.

    Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit.
    Each step renders the predicted state as an image and saves a GIF at the end.
    """
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10000)
    apply_fn = jax.jit(model.apply)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
    frames = []
    os.makedirs(save_dir, exist_ok=True)

    key_to_action = {"w": 0, "a": 1, "s": 2, "d": 3, "x": 4}
    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
    step_i = 0

    def _render_and_save(pred_state, real_obs, step_i):
        """Render predicted vs real side by side."""
        pred_obs = np.array(pred_state[0] > 0.5, dtype=np.uint8)
        try:
            # Real: step the backend env to match, render from engine
            real_frame = backend.render_frame()
            # Predicted: convert multihot to objects array and render
            pred_objects = _multihot_to_objects(pred_obs)
            pred_frame = backend.render_frame_from_objects(pred_objects, grid_w, grid_h)
            # Side by side: real | predicted
            combined = np.concatenate([real_frame, pred_frame], axis=1)
            frames.append(combined)
            frame_path = os.path.join(save_dir, f"step_{step_i:04d}.png")
            imageio.imwrite(frame_path, combined)
            return frame_path
        except Exception as e:
            print(f"  (render error: {e})")
            return None

    # Keep backend engine in sync for rendering
    backend.load_level("", level_i)

    print("\n--- NCA World Model: Interactive Play ---")
    print("Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit")
    print("Left side = real engine, Right side = NCA prediction\n")

    frame_path = _render_and_save(pred_state, real_obs, step_i)
    if frame_path:
        print(f"  Step {step_i}: initial state -> {frame_path}")

    while True:
        try:
            key = input(f"Step {step_i}> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "q":
            break
        elif key == "r":
            real_obs, _ = env.reset()
            pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
            backend.load_level("", level_i)
            step_i = 0
            frame_path = _render_and_save(pred_state, real_obs, step_i)
            print(f"  Restarted -> {frame_path}")
            continue
        elif key not in key_to_action:
            print(f"  Unknown key '{key}'. Use w/a/s/d/x/r/q.")
            continue

        action = key_to_action[key]
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # NCA world model step
        logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        # Real env step (both gym env and backend engine for rendering)
        real_obs, _, done, _, info = env.step(action)
        backend.process_input(action)
        while backend.againing:
            backend.process_input(-1)
        step_i += 1

        # Divergence
        real_f = jnp.array(real_obs[None], dtype=jnp.float32)
        l1 = float(jnp.abs(pred_state - real_f).sum())

        frame_path = _render_and_save(pred_state, real_obs, step_i)
        status = f"  Step {step_i}: {action_names[action]}  L1={l1:.0f}"
        if info.get("won"):
            status += "  WIN!"
        if frame_path:
            status += f"  -> {frame_path}"
        print(status)

        if done:
            print("  Level complete!")

    # Save session as GIF
    if frames:
        gif_path = os.path.join(save_dir, "play_session.gif")
        imageio.mimsave(gif_path, frames, duration=0.3, loop=0)
        print(f"\nSaved play session GIF to {gif_path}")


def _pad_state_for_model(obs: np.ndarray, target_C: int,
                         target_H: int | None = None,
                         target_W: int | None = None) -> jnp.ndarray:
    """Pad a (C, H, W) observation to (1, target_C, target_H, target_W) for the model."""
    C, H, W = obs.shape
    tH = target_H or H
    tW = target_W or W
    if C == target_C and H == tH and W == tW:
        return jnp.array(obs[None], dtype=jnp.float32)
    padded = np.zeros((1, target_C, tH, tW), dtype=np.float32)
    padded[0, :C, :H, :W] = obs
    return jnp.array(padded)


def _unpad_pred(pred_state: jnp.ndarray, n_objs: int,
                H: int | None = None, W: int | None = None) -> np.ndarray:
    """Extract (n_objs, H, W) uint8 from padded (1, model_n_out, pad_H, pad_W) prediction."""
    cropped = pred_state[0, :n_objs]
    if H is not None:
        cropped = cropped[:, :H, :]
    if W is not None:
        cropped = cropped[:, :, :W]
    return np.array(cropped > 0.5, dtype=np.uint8)


def render_rollout_comparison(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    n_steps: int = 30,
    actions: list[int] | None = None,
    save_path: str = "nca_wm_rollout.gif",
    pad_H: int | None = None,
    pad_W: int | None = None,
):
    """Render a side-by-side GIF: real env (top) vs world model prediction (bottom).

    If `actions` is provided, replays that sequence. Otherwise uses random actions.
    pad_H, pad_W: if set, pad observations spatially to these dims for the model.
    """
    max_steps = len(actions) if actions else n_steps
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = jax.jit(model.apply)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, pad_H, pad_W)
    frames = []

    for t in range(max_steps):
        action = actions[t] if actions else np.random.randint(N_ACTIONS)
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # Render real
        real_frame = backend.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )

        # Render prediction (unpad to game's actual n_objs/spatial dims for rendering)
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )

        # Stack vertically: real on top, NCA on bottom
        combined = np.concatenate([real_frame, pred_frame], axis=0)
        frames.append(combined)

        # Step both
        logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            # Render final frame
            real_frame = backend.render_frame_from_objects(
                _multihot_to_objects(real_obs), grid_w, grid_h
            )
            pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
            pred_frame = backend.render_frame_from_objects(
                _multihot_to_objects(pred_obs), grid_w, grid_h
            )
            frames.append(np.concatenate([real_frame, pred_frame], axis=0))
            break

    if frames:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, frames, duration=0.2, loop=0)
        print(f"Saved rollout GIF ({len(frames)} frames) to {save_path}")


def render_post_training_gifs(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    search_data: dict | None,
    level_i: int = 0,
    save_dir: str = ".",
    n_random_steps: int = 50,
):
    """Render comparison GIFs after training: one random rollout + one per search algo."""
    # Random rollout
    print("Rendering random rollout comparison GIF...")
    render_rollout_comparison(
        model, params, json_str, backend, level_i=level_i,
        n_steps=n_random_steps,
        save_path=os.path.join(save_dir, "random_rollout.gif"),
    )

    # Search trajectories
    if search_data is not None and len(search_data["states"]) > 0:
        print("Rendering search trajectory comparison GIF...")
        search_actions = search_data["actions"].tolist()
        render_rollout_comparison(
            model, params, json_str, backend, level_i=level_i,
            actions=search_actions,
            save_path=os.path.join(save_dir, "search_rollout.gif"),
        )


# ---------------------------------------------------------------------------
# 5. Web server
# ---------------------------------------------------------------------------

def serve_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    port: int = 8000,
    host: str = "0.0.0.0",
    obj_names: list[str] | None = None,
):
    """Serve the NCA world model player as a web app."""
    from flask import Flask, jsonify, Response
    import PIL.Image

    app = Flask(__name__)
    apply_fn = jax.jit(model.apply)
    model_viz = NCAWorldModel(n_hid=model.n_hid, n_steps=model.n_steps,
                              n_out=model.n_out, return_intermediates=True)
    apply_viz = jax.jit(model_viz.apply)

    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10000)
    n_objs, grid_h, grid_w = env.observation_shape
    if obj_names is None:
        obj_names = [f"ch{i}" for i in range(n_objs)]

    # Mutable state
    state = {}

    def _reset():
        real_obs, _ = env.reset()
        backend.load_level("", level_i)
        state["real_obs"] = real_obs
        state["pred_state"] = jnp.array(real_obs[None], dtype=jnp.float32)
        state["step"] = 0
        state["diverged"] = False
        state["last_action"] = None

    def _render_obs(obs):
        objects = _multihot_to_objects(obs)
        frame = backend.render_frame_from_objects(objects, grid_w, grid_h)
        buf = io.BytesIO()
        PIL.Image.fromarray(frame).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _img_to_b64(img_arr):
        buf = io.BytesIO()
        PIL.Image.fromarray(img_arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    _reset()

    @app.route("/")
    def index():
        return HTML_PAGE

    @app.route("/api/state")
    def get_state():
        real_b64 = _render_obs(state["real_obs"])
        pred_obs = np.array(state["pred_state"][0] > 0.5, dtype=np.uint8)
        pred_b64 = _render_obs(pred_obs)
        real_f = jnp.array(state["real_obs"][None], dtype=jnp.float32)
        l1 = float(jnp.abs(state["pred_state"] - real_f).sum())
        return jsonify(real=real_b64, pred=pred_b64, step=state["step"],
                       l1=l1, diverged=state["diverged"])

    @app.route("/api/step/<int:action>")
    def step(action):
        if action < 0 or action >= N_ACTIONS:
            return jsonify(error="invalid action"), 400
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # NCA prediction
        logits = apply_fn(params, state["pred_state"], a_oh)
        state["pred_state"] = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        state["last_action"] = action

        # Real env
        state["real_obs"], _, done, _, info = env.step(action)
        backend.process_input(action)
        while backend.againing:
            backend.process_input(-1)
        state["step"] += 1

        real_f = jnp.array(state["real_obs"][None], dtype=jnp.float32)
        l1 = float(jnp.abs(state["pred_state"] - real_f).sum())

        real_b64 = _render_obs(state["real_obs"])
        pred_obs = np.array(state["pred_state"][0] > 0.5, dtype=np.uint8)
        pred_b64 = _render_obs(pred_obs)

        return jsonify(real=real_b64, pred=pred_b64, step=state["step"],
                       l1=l1, won=bool(info.get("won", False)), done=bool(done))

    @app.route("/api/step_dream/<int:action>")
    def step_dream(action):
        """Step only the NCA world model (no real env) — pure dreaming."""
        if action < 0 or action >= N_ACTIONS:
            return jsonify(error="invalid action"), 400
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        logits = apply_fn(params, state["pred_state"], a_oh)
        state["pred_state"] = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        state["step"] += 1
        state["diverged"] = True
        state["last_action"] = action

        pred_obs = np.array(state["pred_state"][0] > 0.5, dtype=np.uint8)
        pred_b64 = _render_obs(pred_obs)
        return jsonify(pred=pred_b64, step=state["step"])

    @app.route("/api/activations/<int:action>")
    def get_activations(action):
        """Run one NCA step with intermediates and return visualization images."""
        if action < 0 or action >= N_ACTIONS:
            return jsonify(error="invalid action"), 400
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        logits, intermediates = apply_viz(params, state["pred_state"], a_oh)

        result = {"steps": []}
        for step_i, (h, readout) in enumerate(
            zip(intermediates["hidden"], intermediates["readouts"])
        ):
            h_np = np.array(h[0])        # (H, W, n_hid)
            r_np = np.array(readout[0])   # (C, H, W)

            # Hidden channels grid
            hid_grid = _make_channel_grid(h_np)
            hid_img = _apply_colormap(hid_grid)

            # Labeled output channels
            out_img = _labeled_channel_grid(r_np, obj_names)

            result["steps"].append({
                "hidden": _img_to_b64(hid_img),
                "output": _img_to_b64(out_img),
            })

        return jsonify(result)

    @app.route("/api/reset")
    def reset():
        _reset()
        return get_state()

    print(f"\nServing NCA world model player at http://{host}:{port}")
    print("Use arrow keys / WASD to play. Press V to toggle dream mode.\n")
    app.run(host=host, port=port, debug=False)


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<title>NCA World Model Player</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee; font-family: monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  h1 { margin-bottom: 4px; font-size: 1.4em; color: #e94560; }
  .subtitle { color: #888; margin-bottom: 16px; font-size: 0.9em; }
  .game-row {
    display: flex; gap: 24px; align-items: flex-start;
    flex-wrap: wrap; justify-content: center;
  }
  .panel { text-align: center; }
  .panel h2 { font-size: 1em; margin-bottom: 8px; }
  .panel h2.real { color: #4ecca3; }
  .panel h2.pred { color: #e94560; }
  .panel img.game-img {
    image-rendering: pixelated;
    border: 2px solid #333;
    min-width: 256px; min-height: 200px;
    background: #111;
  }
  .info {
    margin-top: 16px; padding: 12px 20px;
    background: #16213e; border-radius: 8px;
    display: flex; gap: 24px; font-size: 0.95em;
  }
  .info .val { color: #4ecca3; font-weight: bold; }
  .info .warn { color: #e94560; }
  .controls {
    margin-top: 12px; color: #666; font-size: 0.85em;
    line-height: 1.6;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    color: #fff; font-size: 0.8em; margin-left: 8px;
    vertical-align: middle;
  }
  .badge.dream { background: #e94560; }
  .badge.act { background: #0f3460; }
  .badge.hidden { display: none; }
  /* Activations panel */
  #actPanel {
    margin-top: 20px; width: 100%; max-width: 1200px;
    display: none;
  }
  #actPanel.visible { display: block; }
  #actPanel h3 { color: #e94560; margin-bottom: 8px; font-size: 1em; }
  .act-step {
    margin-bottom: 16px; padding: 12px;
    background: #16213e; border-radius: 8px;
  }
  .act-step h4 { color: #4ecca3; margin-bottom: 6px; font-size: 0.9em; }
  .act-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }
  .act-row .act-label { color: #888; font-size: 0.8em; margin-bottom: 2px; }
  .act-row img { max-width: 100%; border: 1px solid #333; background: #111; }
</style>
</head>
<body>
  <h1>NCA World Model Player
    <span id="dreamBadge" class="badge dream hidden">DREAM</span>
    <span id="actBadge" class="badge act hidden">ACTIVATIONS</span>
  </h1>
  <p class="subtitle">Real game engine vs. learned NCA world model</p>
  <div class="game-row">
    <div class="panel">
      <h2 class="real">Real Engine</h2>
      <img id="realImg" class="game-img" src="" />
    </div>
    <div class="panel">
      <h2 class="pred">NCA Prediction</h2>
      <img id="predImg" class="game-img" src="" />
    </div>
  </div>
  <div class="info">
    <div>Step: <span class="val" id="stepVal">0</span></div>
    <div>L1 divergence: <span class="val" id="l1Val">0</span></div>
  </div>
  <div class="controls">
    Arrows / WASD = move &nbsp;|&nbsp; X = action &nbsp;|&nbsp;
    R = restart &nbsp;|&nbsp; V = dream mode &nbsp;|&nbsp;
    T = show activations
  </div>
  <div id="actPanel">
    <h3>NCA Internal Activations</h3>
    <div id="actContent"></div>
  </div>

<script>
const KEY_MAP = {
  ArrowUp: 0, ArrowLeft: 1, ArrowDown: 2, ArrowRight: 3,
  w: 0, a: 1, s: 2, d: 3, x: 4,
};
const ACTION_NAMES = ['up', 'left', 'down', 'right', 'action'];
let dreaming = false;
let showAct = false;
let busy = false;
let lastAction = 0;

async function fetchState() {
  const r = await fetch('/api/state');
  update(await r.json());
}

function update(d) {
  if (d.real) document.getElementById('realImg').src = 'data:image/png;base64,' + d.real;
  if (d.pred) document.getElementById('predImg').src = 'data:image/png;base64,' + d.pred;
  if (d.step !== undefined) document.getElementById('stepVal').textContent = d.step;
  if (d.l1 !== undefined) {
    const el = document.getElementById('l1Val');
    el.textContent = d.l1.toFixed(0);
    el.className = d.l1 > 20 ? 'val warn' : 'val';
  }
}

async function fetchActivations(action) {
  const r = await fetch('/api/activations/' + action);
  const d = await r.json();
  const container = document.getElementById('actContent');
  container.innerHTML = '';
  d.steps.forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'act-step';
    div.innerHTML = `
      <h4>NCA Step ${i + 1}</h4>
      <div class="act-row">
        <div><div class="act-label">Hidden channels (${ACTION_NAMES[action]})</div>
             <img src="data:image/png;base64,${s.hidden}" /></div>
        <div><div class="act-label">Output channels (per object)</div>
             <img src="data:image/png;base64,${s.output}" /></div>
      </div>`;
    container.appendChild(div);
  });
}

document.addEventListener('keydown', async (e) => {
  if (busy) return;
  const key = e.key;

  if (key === 'r' || key === 'R') {
    busy = true;
    dreaming = false;
    document.getElementById('dreamBadge').classList.add('hidden');
    update(await (await fetch('/api/reset')).json());
    if (showAct) await fetchActivations(lastAction);
    busy = false;
    return;
  }
  if (key === 'v' || key === 'V') {
    dreaming = !dreaming;
    document.getElementById('dreamBadge').classList.toggle('hidden', !dreaming);
    return;
  }
  if (key === 't' || key === 'T') {
    showAct = !showAct;
    document.getElementById('actPanel').classList.toggle('visible', showAct);
    document.getElementById('actBadge').classList.toggle('hidden', !showAct);
    if (showAct) await fetchActivations(lastAction);
    return;
  }

  const action = KEY_MAP[key];
  if (action === undefined) return;
  e.preventDefault();
  busy = true;
  lastAction = action;

  const endpoint = dreaming ? '/api/step_dream/' : '/api/step/';
  update(await (await fetch(endpoint + action)).json());
  if (showAct) await fetchActivations(action);
  busy = false;
});

fetchState();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train NCA world model on a PuzzleScript game")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--game", help="Single game name (e.g. pipe_bend, sokoban_basic)")
    g.add_argument("--games", help="Comma-separated game names, or a preset name (e.g. 'small')")
    p.add_argument("--level", type=int, default=None,
                   help="Train on a single level index. Default: all levels.")
    # Data collection
    p.add_argument("--n_random_episodes", type=int, default=500)
    p.add_argument("--n_search_steps", type=int, default=100_000)
    p.add_argument("--search_timeout_ms", type=int, default=60_000)
    p.add_argument("--search_algos", nargs="+", default=["bfs", "astar"])
    p.add_argument("--search_weight", type=float, default=5.0,
                   help="Oversample search data by this factor relative to random")
    p.add_argument("--max_episode_steps", type=int, default=200)
    p.add_argument("--data_mode", default="unique", choices=["unique", "random_search"],
                   help="Data collection mode: 'unique' (deduplicated exploration) "
                        "or 'random_search' (random rollouts + search)")
    # Architecture
    p.add_argument("--n_nca_steps", type=int, default=4,
                   help="Number of NCA update steps per forward pass")
    p.add_argument("--n_hid", type=int, default=128)
    # Conditional model
    p.add_argument("--conditional", action="store_true",
                   help="Use ConditionalNCAWorldModel with game-spec encoder")
    p.add_argument("--d_z", type=int, default=64, help="Latent dimension for game encoder")
    p.add_argument("--d_model", type=int, default=64, help="Transformer hidden dim")
    p.add_argument("--n_enc_layers", type=int, default=2, help="Transformer encoder layers")
    p.add_argument("--n_heads", type=int, default=4, help="Transformer attention heads")
    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_updates", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=100)
    # Output
    p.add_argument("--save_dir", default=None)
    p.add_argument("--render_gif", action="store_true", help="Render comparison GIF after training")
    p.add_argument("--load", default=None, metavar="DIR",
                   help="Load params from this dir instead of the default save_dir")
    p.add_argument("--play", action="store_true",
                   help="Interactive play mode (w/a/s/d/x keys)")
    p.add_argument("--serve", action="store_true",
                   help="Launch web server for browser-based play")
    p.add_argument("--port", type=int, default=8000)
    # Logging
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--wandb_project", default="nca-world-model", help="wandb project name")
    p.add_argument("--wandb_name", default=None, help="wandb run name (auto-generated if not set)")
    args = p.parse_args()

    ps_parser = init_ps_lark_parser()
    multigame = args.games is not None

    if multigame:
        # --- Multi-game path ---
        if args.games in MULTI_GAME_PRESETS:
            game_names = MULTI_GAME_PRESETS[args.games]
            preset_tag = args.games
        else:
            game_names = [g.strip() for g in args.games.split(",")]
            preset_tag = f"{len(game_names)}games"

        cond_tag = "cond" if args.conditional else "uncond"
        save_dir = (args.save_dir or
                    f"logs_nca_wm/multi_{preset_tag}_{cond_tag}_level-{args.level}"
                    f"_nca-{args.n_nca_steps}_hid-{args.n_hid}_lr-{args.lr}_s-{args.seed}")

        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"multi_{preset_tag}_nca{args.n_nca_steps}_h{args.n_hid}",
                config=vars(args),
                dir=save_dir,
                resume="allow",
            )

        load_dir = args.load or save_dir
        ckpt_path = os.path.join(load_dir, "params.pkl")
        infos_path = os.path.join(load_dir, "game_infos.pkl")

        # Load existing checkpoint + game_infos
        init_params = None
        start_step = 0
        if os.path.isfile(infos_path):
            with open(infos_path, "rb") as f:
                game_infos = pickle.load(f)
        else:
            game_infos = None

        if os.path.isfile(ckpt_path):
            print(f"Loading params from {ckpt_path}")
            with open(ckpt_path, "rb") as f:
                init_params = pickle.load(f)
            meta_path = os.path.join(load_dir, "train_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    start_step = json.load(f).get("total_steps", 0)
            print(f"  Resuming from step {start_step}")

        # Collect/load dataset only if we need to train
        remaining = args.n_updates - start_step
        needs_training = remaining > 0

        if needs_training:
            names_to_collect = [g["name"] for g in game_infos] if game_infos else game_names
            dataset, game_infos = collect_multigame_dataset(
                names_to_collect, ps_parser,
                level_i=args.level,
                n_random_episodes=args.n_random_episodes,
                max_episode_steps=args.max_episode_steps,
                search_algos=args.search_algos,
                n_search_steps=args.n_search_steps,
                search_timeout_ms=args.search_timeout_ms,
                search_weight=args.search_weight,
            )
            os.makedirs(save_dir, exist_ok=True)
            with open(infos_path, "wb") as f:
                pickle.dump(game_infos, f)
        elif game_infos is None:
            raise RuntimeError(
                f"No game_infos.pkl found at {infos_path} and no training to do. "
                "Run training first or provide --load pointing to a trained checkpoint."
            )

        max_C = max(g["n_objs"] for g in game_infos)
        if args.conditional:
            max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
            max_tok_len = max(max_tok_len, 1)
            model = ConditionalNCAWorldModel(
                n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C,
                vocab_size=VOCAB_SIZE + 1,  # +1 for CLS
                d_model=args.d_model, n_heads=args.n_heads,
                n_enc_layers=args.n_enc_layers, d_z=args.d_z,
                max_seq_len=max_tok_len + 1,  # +1 for CLS
            )
        else:
            model = NCAWorldModel(n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=max_C)

        if needs_training:
            params, losses, accs, change_accs, final_step = train(
                model, dataset,
                lr=args.lr,
                n_updates=remaining,
                batch_size=args.batch_size,
                seed=args.seed,
                log_interval=args.log_interval,
                save_dir=save_dir,
                init_params=init_params,
                start_step=start_step,
            )
            # Save training curves + config
            os.makedirs(save_dir, exist_ok=True)
            np.savez(
                os.path.join(save_dir, f"curves_step{final_step}.npz"),
                losses=np.array(losses),
                accs=np.array(accs),
                change_accs=np.array(change_accs),
            )
        else:
            print(f"Already at {start_step} steps (target {args.n_updates}), skipping training.")
            params = init_params
            final_step = start_step

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # Serve mode: skip eval, launch interactive server immediately
        if args.serve:
            # Serve the first game by default (or could add --serve_game flag)
            info = game_infos[0]
            print(f"\nServing {info['name']} (multi-game model, step {final_step})...")
            backend_render = CppPuzzleScriptBackend()
            backend_render.compile_game(ps_parser, info["name"])
            serve_world_model(
                model, params, info["json_str"], backend_render,
                level_i=args.level, port=args.port,
            )
            if wandb.run is not None:
                wandb.finish()
            return

        # Per-game evaluation
        print("\nEvaluating per-game (autoregressive rollout)...")
        evaluate_multigame(
            model, params, game_infos, ps_parser,
            search_algos=args.search_algos,
            search_n_steps=args.n_search_steps,
            search_timeout_ms=args.search_timeout_ms,
            save_dir=save_dir,
        )

        # Per-game GIFs (random + search, all levels)
        print("\nRendering per-game comparison GIFs...")
        render_multigame_gifs(
            model, params, game_infos, ps_parser,
            save_dir=save_dir, step_label=final_step,
            search_algos=args.search_algos,
            search_n_steps=args.n_search_steps,
            search_timeout_ms=args.search_timeout_ms,
        )

        if wandb.run is not None:
            wandb.finish()
        print("Done!")
        return

    # --- Single-game path (original) ---
    # Single-game path defaults to level 0 for backwards compat
    if args.level is None:
        args.level = 0
    save_dir = (args.save_dir or
                f"logs_nca_wm/{args.game}_level-{args.level}_nca-{args.n_nca_steps}"
                f"_hid-{args.n_hid}_lr-{args.lr}_s-{args.seed}")

    # Compile game
    print(f"Compiling {args.game}...")
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, args.game)
    env = CppPuzzleScriptEnv(json_str, level_i=args.level, max_episode_steps=args.max_episode_steps)
    n_objs, H, W = env.observation_shape
    print(f"  obs_shape=({n_objs}, {H}, {W}), num_levels={env.num_levels}")

    model = NCAWorldModel(n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=n_objs)

    # Load existing checkpoint if available, otherwise train
    load_dir = args.load or save_dir
    ckpt_path = os.path.join(load_dir, "params.pkl")

    if os.path.isfile(ckpt_path):
        print(f"Loading params from {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)
    else:
        # Collect data
        print("Collecting random rollouts...")
        random_data = collect_random_rollouts(
            json_str, args.game, level_i=args.level,
            n_episodes=args.n_random_episodes,
            max_steps=args.max_episode_steps,
        )

        print("Collecting search trajectories...")
        search_data = collect_search_trajectories(
            json_str, args.game, level_i=args.level,
            algos=args.search_algos,
            n_steps=args.n_search_steps,
            timeout_ms=args.search_timeout_ms,
            max_episode_steps=args.max_episode_steps,
        )

        dataset = merge_datasets(random_data, search_data, args.search_weight)
        print(f"Total dataset: {len(dataset['states'])} transitions")

        # How many transitions actually involve a state change?
        changed = (dataset["states"] != dataset["next_states"]).any(axis=(1, 2, 3))
        print(f"  Transitions with state change: {changed.sum()}/{len(changed)} "
              f"({100*changed.mean():.1f}%)")

        params, losses, accs, change_accs, _ = train(
            model, dataset,
            lr=args.lr,
            n_updates=args.n_updates,
            batch_size=args.batch_size,
            seed=args.seed,
            log_interval=args.log_interval,
            save_dir=save_dir,
        )

        # Save training curves
        np.savez(
            os.path.join(save_dir, "curves.npz"),
            losses=np.array(losses),
            accs=np.array(accs),
            change_accs=np.array(change_accs),
        )

        # Save config
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # Evaluate
        print("\nEvaluating world model (autoregressive rollout)...")
        evaluate_world_model(model, params, json_str, level_i=args.level, save_dir=save_dir)

        # Render post-training comparison GIFs
        print("\nRendering comparison GIFs...")
        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, args.game)
        render_post_training_gifs(
            model, params, json_str, backend_render,
            search_data=search_data, level_i=args.level,
            save_dir=save_dir,
        )

    # Object names for activation visualization
    obj_names = env._canonical_ids

    # Modes that need the renderer
    need_renderer = args.play or args.serve or args.render_gif
    if need_renderer:
        # May already exist from post-training GIF rendering; create if not
        try:
            backend_render
        except NameError:
            backend_render = CppPuzzleScriptBackend()
            backend_render.compile_game(ps_parser, args.game)

    if args.serve:
        serve_world_model(
            model, params, json_str, backend_render,
            level_i=args.level, port=args.port,
            obj_names=obj_names,
        )
    elif args.play:
        play_dir = os.path.join(save_dir, "play")
        play_world_model(
            model, params, json_str, backend_render,
            level_i=args.level, save_dir=play_dir,
        )
    elif args.render_gif:
        gif_path = os.path.join(save_dir, "rollout_comparison.gif")
        render_rollout_comparison(
            model, params, json_str, backend_render,
            level_i=args.level, save_path=gif_path,
        )

    print("Done!")


if __name__ == "__main__":
    main()
