"""Expert Iteration (ExIt) training loop for PuzzleScript games via JAXtar.

This implements a MuZero-esque alternating search-and-train loop for learning
neural heuristics on PuzzleScript games where the goal state is **unknown**
(win conditions are defined by rules, not a target state).

The loop:
    1. Run batched A* search on puzzle levels using the current neural heuristic
    2. Extract training data from the search graph:
       - Solved puzzles: exact h*(s) = optimal_cost - g(s) for all expanded nodes
       - Unsolved puzzles: use g-values as distance-from-start supervision
    3. Train the neural heuristic on this data (+ replay buffer)
    4. Repeat — improved heuristic → better search → better training data

Usage:
    python exit_train.py --game blocks --level 0 --iterations 50
    python exit_train.py --game sokoban_basic --level 0 --iterations 100 -m 100000 -b 1000
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import deque
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

# Ensure JAXtar is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JAXTAR_DIR = os.path.join(SCRIPT_DIR, "JAXtar")
if JAXTAR_DIR not in sys.path:
    sys.path.insert(0, JAXTAR_DIR)

from puzzlejax.wrappers import PuzzleJaxPuxleEnv, PuzzleJaxHeuristic
from JAxtar.stars.astar import astar_builder
from JAxtar.stars.search_base import SearchResult
from heuristic.heuristic_base import Heuristic
from heuristic.neuralheuristic.neuralheuristic_base import NeuralHeuristicBase, HeuristicBase
from neural_util.modules import DTYPE, HEAD_DTYPE, get_norm_fn, get_activation_fn, get_resblock_fn
from neural_util.param_manager import save_params_with_metadata, load_params_with_metadata
from train_util.optimizer import setup_optimizer, get_eval_params, get_learning_rate
from helpers.visualization import PathStep, build_path_steps_from_actions


# ---------------------------------------------------------------------------
# Extract the best frontier path using the *environment* heuristic
# ---------------------------------------------------------------------------

def extract_best_env_f_path(
    puzzle: PuzzleJaxPuxleEnv,
    search_result: SearchResult,
    solve_config,
    init_state,
    rule_heuristic: PuzzleJaxHeuristic = None,
    render: bool = True,
) -> Optional[dict]:
    """Find the expanded node closest to the goal by environment heuristic.

    Picks the node with the lowest h_env(s) among all expanded nodes, i.e.
    the one the environment considers nearest to a win.  Ties are broken by
    lowest f_env = g + h_env, then lowest g.

    Reconstructs the path from start to that node by tracing parent pointers,
    and optionally builds PathSteps for GIF rendering.

    Returns a dict with:
        - "best_env_h": float, environment heuristic at best node
        - "best_g": float, g-value (cost-to-reach) of best node
        - "best_env_f": float, g + h_env
        - "best_actions": list[int], action sequence from start to best node
        - "path_steps": list[PathStep] or None (if render=True)
        - "n_expanded": int, number of expanded nodes considered
    or None if no expanded nodes exist.
    """
    from xtructure import HashIdx
    from JAxtar.stars.search_base import Parent, Current
    from JAxtar.annotate import ACTION_DTYPE

    ACTION_PAD = int(np.iinfo(np.dtype(ACTION_DTYPE)).max)

    # --- JAX-ified f_env computation (no Python loop) ---
    costs = search_result.cost  # g-values; inf for unexpanded slots
    # state.heuristic is stored negative (e.g. -manhattan_dist); negate it.
    env_h_all = -search_result.hashtable.table.heuristic.astype(jnp.float32)

    # Only consider nodes that were actually *expanded* (popped from PQ),
    # because only they have valid parent pointers for path reconstruction.
    # Discovered-but-not-expanded nodes have finite cost but default (sentinel)
    # parent entries, which breaks the parent-chain traversal.
    expanded_mask = search_result.pop_generation >= 0
    finite_mask = jnp.logical_and(jnp.isfinite(costs), expanded_mask)
    n_expanded = int(jnp.sum(finite_mask))
    if n_expanded < 1:
        return None

    # Primary key: lowest h_env (closest to goal).
    # Secondary: lowest f_env = g + h_env.  Use a combined sort key so that
    # argmin picks the most-progressed node rather than the start state.
    f_env = jnp.where(finite_mask, costs + env_h_all, jnp.inf)
    sort_key = jnp.where(
        finite_mask,
        env_h_all * 1e6 + f_env,   # h_env dominates, f_env breaks ties
        jnp.inf,
    )
    best_idx = int(jnp.argmin(sort_key))
    best_g = float(costs[best_idx])
    best_env_h = float(env_h_all[best_idx])
    best_env_f = float(f_env[best_idx])

    # --- Path reconstruction (mirrors get_solved_path logic) ---
    hash_idx = HashIdx(index=jnp.array(best_idx, dtype=jnp.int32))
    current_node = Current(hashidx=hash_idx, cost=jnp.array(best_g, dtype=jnp.float32))

    path_nodes = [current_node]
    visited = {best_idx}
    parent = search_result.get_parent(current_node)
    while True:
        # Compare at JAX level so uint32(-1) matches Python -1 correctly.
        if bool(parent.hashidx.index == -1):
            break
        idx = int(parent.hashidx.index)
        if idx in visited:
            break
        visited.add(idx)
        path_nodes.append(parent)
        parent = search_result.get_parent(parent)
    path_nodes.reverse()

    # Extract actions — Parent nodes carry .action, Current does not.
    actions_list = []
    for node in path_nodes:
        action = getattr(node, "action", None)
        if action is not None:
            action_val = int(action)
            if action_val != ACTION_PAD:
                actions_list.append(action_val)

    # Build PathSteps for GIF rendering
    path_steps = None
    if render and actions_list:
        try:
            path_steps = build_path_steps_from_actions(
                puzzle=puzzle,
                solve_config=solve_config,
                initial_state=init_state,
                actions=actions_list,
                heuristic=rule_heuristic,
            )
        except Exception as e:
            print(f"  Warning: Failed to build path steps for best-f: {e}")
            path_steps = None

    return {
        "best_env_h": best_env_h,
        "best_g": best_g,
        "best_env_f": best_env_f,
        "best_actions": actions_list,
        "path_steps": path_steps,
        "n_expanded": n_expanded,
    }


# ---------------------------------------------------------------------------
# Neural heuristic for PuzzleScript games
# ---------------------------------------------------------------------------

class PuzzleScriptNeuralHeuristic(NeuralHeuristicBase):
    """Neural heuristic for PuzzleScript games.

    Pre-processes the multihot board representation into a flat float vector.
    Since there is no known target state, we only encode the current state.
    """

    def __init__(self, puzzle: PuzzleJaxPuxleEnv, path: str = None, **kwargs):
        # Compute input dimension from the multihot shape
        dummy_state = puzzle.State.default()
        self.multihot_shape = dummy_state.multihot_level.shape
        self.input_dim = int(np.prod(self.multihot_shape))

        # Use smaller defaults suitable for PuzzleScript (much smaller than Rubik's cube)
        kwargs.setdefault("initial_dim", 512)
        kwargs.setdefault("hidden_dim", 256)
        kwargs.setdefault("Res_N", 2)
        kwargs.setdefault("hidden_N", 1)

        super().__init__(puzzle, path=path, **kwargs)

    def pre_process(self, solve_config, current) -> jnp.ndarray:
        """Flatten multihot board to a float vector in [-1, 1]."""
        board = current.multihot_level.astype(jnp.float32)
        flat = jnp.reshape(board, (-1,))
        return ((flat - 0.5) * 2.0).astype(DTYPE)

    def post_process(self, x: jnp.ndarray) -> jnp.ndarray:
        """Ensure heuristic output is non-negative."""
        return jnp.maximum(x.squeeze(-1), 0.0)


# ---------------------------------------------------------------------------
# Heuristic wrapper that delegates to a neural net with updatable params
# ---------------------------------------------------------------------------

class NeuralHeuristicWrapper(Heuristic):
    """Wraps a NeuralHeuristicBase so that A* can use it, with swappable params.

    We also blend the neural heuristic with the built-in rule-based heuristic 
    from PuzzleScript to provide a safety net.
    """

    def __init__(
        self,
        neural_heuristic: PuzzleScriptNeuralHeuristic,
        rule_heuristic: PuzzleJaxHeuristic,
        blend_alpha: float = 0.5,
    ):
        self.neural = neural_heuristic
        self.rule = rule_heuristic
        self.blend_alpha = blend_alpha
        self.puzzle = neural_heuristic.puzzle

    def distance(self, solve_config, current) -> float:
        nn_h = self.neural.distance(solve_config, current)
        rule_h = float(current.heuristic)
        return self.blend_alpha * nn_h + (1.0 - self.blend_alpha) * rule_h

    def batched_distance(self, solve_config, current):
        nn_h = self.neural.batched_distance(solve_config, current)
        rule_h = current.heuristic.astype(jnp.float32)
        return self.blend_alpha * nn_h + (1.0 - self.blend_alpha) * rule_h


# ---------------------------------------------------------------------------
# Extract training data from a completed A* search
# ---------------------------------------------------------------------------

def extract_training_data(
    puzzle: PuzzleJaxPuxleEnv,
    search_result: SearchResult,
    solve_config,
    max_samples: int = 50000,
) -> Optional[dict]:
    """Extract (state, heuristic_target) pairs from a completed A* search.

    For solved puzzles:
        h*(s) = optimal_cost - g(s) for every expanded node on the optimal path.
        For other expanded nodes: h_target(s) = max(optimal_cost - g(s), 0) as an
        upper-bound target (not exact, but useful supervision).

    For unsolved puzzles:
        We still have useful g-values (cost-to-reach). We use the minimum
        f-value in the open set as an estimate of the optimal cost, giving
        h_target(s) = f_min - g(s) for expanded nodes.

    Returns dict with:
        - "multihot_levels": [N, *multihot_shape] bool array 
        - "targets": [N] float array of heuristic targets
        - "weights": [N] float array of sample weights
        - "solved": bool
    """
    solved = bool(search_result.solved)
    generated = int(search_result.generated_size)

    if generated < 2:
        return None

    # Get the cost array — finite entries are expanded nodes
    costs = np.array(search_result.cost)
    finite_mask = np.isfinite(costs)
    n_expanded = int(np.sum(finite_mask))

    if n_expanded < 1:
        return None

    # Get all expanded states from the hashtable
    finite_indices = np.where(finite_mask)[0]

    # Subsample if too many
    if n_expanded > max_samples:
        rng = np.random.default_rng(42)
        finite_indices = rng.choice(finite_indices, max_samples, replace=False)
        n_expanded = max_samples

    g_values = costs[finite_indices]

    if solved:
        # Optimal cost is the cost of the solved state
        solved_idx = search_result.solved_idx
        optimal_cost = float(search_result.get_cost(solved_idx))
        h_targets = optimal_cost - g_values
        h_targets = np.maximum(h_targets, 0.0)

        # Higher weight for nodes with higher g (closer to start, more reliable)
        # and for nodes with lower h (closer to goal, rarer to see)
        weights = np.ones(n_expanded, dtype=np.float32)

        # Nodes on the optimal path get extra weight
        # Extract path nodes
        try:
            path = search_result.get_solved_path()
            path_indices = set()
            for node in path:
                idx = int(node.hashidx.index)
                if idx >= 0:
                    path_indices.add(idx)
            # Boost path nodes
            for i, idx in enumerate(finite_indices):
                if idx in path_indices:
                    weights[i] = 3.0
        except Exception:
            pass
    else:
        # Unsolved: use heuristic stored in dist array as a fallback target
        dist_values = np.array(search_result.dist)
        h_targets_from_dist = dist_values[finite_indices]
        h_targets_from_dist = np.where(
            np.isfinite(h_targets_from_dist),
            np.maximum(h_targets_from_dist, 0.0),
            0.0
        )

        # Also compute f_min from the open set as an upper bound estimate
        f_values = costs + dist_values
        f_values_finite = f_values[np.isfinite(f_values)]
        if len(f_values_finite) > 0:
            f_min = float(np.min(f_values_finite))
        else:
            f_min = float(np.max(g_values)) + 1.0

        # h_target = max(f_min - g, rule_heuristic, 0)
        h_targets = np.maximum(f_min - g_values, h_targets_from_dist)
        h_targets = np.maximum(h_targets, 0.0)
        weights = np.ones(n_expanded, dtype=np.float32) * 0.5  # Lower weight for unsolved

    # Extract states
    from xtructure import HashIdx
    multihot_levels = []
    for idx in finite_indices:
        hash_idx = HashIdx(index=jnp.array(idx, dtype=jnp.int32))
        state = search_result.hashtable[hash_idx]
        multihot_levels.append(np.array(state.multihot_level))

    multihot_levels = np.stack(multihot_levels, axis=0)
    h_targets = np.array(h_targets, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    return {
        "multihot_levels": multihot_levels,
        "targets": h_targets,
        "weights": weights,
        "solved": solved,
        "n_expanded": n_expanded,
    }


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple replay buffer that stores (state, target, weight) tuples."""

    def __init__(self, max_size: int = 200000):
        self.max_size = max_size
        self.multihot_levels = None
        self.targets = None
        self.weights = None
        self.size = 0

    def add(self, data: dict):
        """Add a batch of data to the buffer."""
        ml = data["multihot_levels"]
        tgt = data["targets"]
        w = data["weights"]

        if self.multihot_levels is None:
            self.multihot_levels = ml
            self.targets = tgt
            self.weights = w
        else:
            self.multihot_levels = np.concatenate([self.multihot_levels, ml], axis=0)
            self.targets = np.concatenate([self.targets, tgt], axis=0)
            self.weights = np.concatenate([self.weights, w], axis=0)

        # Trim if over capacity (keep most recent)
        if self.multihot_levels.shape[0] > self.max_size:
            self.multihot_levels = self.multihot_levels[-self.max_size:]
            self.targets = self.targets[-self.max_size:]
            self.weights = self.weights[-self.max_size:]

        self.size = self.multihot_levels.shape[0]

    def sample(self, batch_size: int, rng: np.random.Generator = None) -> dict:
        """Sample a minibatch from the buffer."""
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(self.size, min(batch_size, self.size), replace=False)
        return {
            "multihot_levels": self.multihot_levels[indices],
            "targets": self.targets[indices],
            "weights": self.weights[indices],
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "multihot_levels": self.multihot_levels,
                "targets": self.targets,
                "weights": self.weights,
            }, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.multihot_levels = data["multihot_levels"]
            self.targets = data["targets"]
            self.weights = data["weights"]
            self.size = self.multihot_levels.shape[0] if self.multihot_levels is not None else 0
            return True
        return False


# ---------------------------------------------------------------------------
# Training step (pure JAX)
# ---------------------------------------------------------------------------

def build_train_step(
    model: HeuristicBase,
    optimizer: optax.GradientTransformation,
    multihot_shape: tuple,
):
    """Build a JIT-compiled training step function."""

    @jax.jit
    def train_step(params, opt_state, batch_multihot, batch_targets, batch_weights):
        """Single gradient step.

        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch_multihot: [B, *multihot_shape] bool array
            batch_targets: [B] float targets
            batch_weights: [B] float sample weights
        """

        def loss_fn(params):
            # Pre-process: flatten and scale to [-1, 1]
            x = batch_multihot.astype(jnp.float32)
            x = jnp.reshape(x, (x.shape[0], -1))
            x = ((x - 0.5) * 2.0).astype(DTYPE)

            pred, variable_updates = model.apply(
                params, x, training=True, mutable=["batch_stats"]
            )
            pred = jnp.maximum(pred.squeeze(-1), 0.0)

            diff = batch_targets - pred
            per_sample_loss = jnp.square(diff)  # MSE
            weighted_loss = jnp.mean(per_sample_loss * batch_weights)

            # Merge batch_stats back
            if "batch_stats" in variable_updates:
                new_params = {**params, "batch_stats": variable_updates["batch_stats"]}
            else:
                new_params = params
            return weighted_loss, (new_params, pred, diff)

        (loss, (new_params, pred, diff)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params=params)
        params_new = optax.apply_updates(new_params, updates)

        metrics = {
            "loss": loss,
            "mean_pred": jnp.mean(pred),
            "mean_target": jnp.mean(batch_targets),
            "mean_abs_diff": jnp.mean(jnp.abs(diff)),
        }
        return params_new, opt_state_new, metrics

    return train_step


# ---------------------------------------------------------------------------
# Main ExIt loop
# ---------------------------------------------------------------------------

def run_exit_training(
    game: str,
    level_i: int,
    n_iterations: int = 50,
    max_nodes: int = 100_000,
    batch_size: int = 1000,
    cost_weight: float = 0.6,
    train_steps_per_iter: int = 200,
    train_batch_size: int = 256,
    lr: float = 1e-3,
    blend_alpha: float = 0.7,
    replay_max_size: int = 200_000,
    save_dir: str = None,
    resume: bool = False,
    # Neural net architecture
    initial_dim: int = 512,
    hidden_dim: int = 256,
    res_n: int = 2,
):
    """Run the ExIt training loop.

    Args:
        game: PuzzleScript game name
        level_i: Level index
        n_iterations: Number of search-train iterations
        max_nodes: Maximum A* search nodes
        batch_size: A* batch size
        cost_weight: A* cost weight (f = g + w*h)
        train_steps_per_iter: Gradient steps per search iteration
        train_batch_size: Minibatch size for training
        lr: Learning rate
        blend_alpha: Weight of neural heuristic vs rule-based (0=all rule, 1=all neural)
        replay_max_size: Replay buffer capacity
        save_dir: Directory to save checkpoints
        resume: Whether to resume from a checkpoint
        initial_dim: Neural net initial dimension
        hidden_dim: Neural net hidden dimension
        res_n: Number of residual blocks
    """
    if save_dir is None:
        save_dir = os.path.join(SCRIPT_DIR, "data", "exit_training", f"{game}_level{level_i}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"ExIt Training: {game} level {level_i}")
    print(f"{'='*70}")
    print(f"  Max nodes: {max_nodes:,}  Batch size: {batch_size}")
    print(f"  cost_weight: {cost_weight}  blend_alpha: {blend_alpha}")
    print(f"  Train steps/iter: {train_steps_per_iter}  Train batch: {train_batch_size}")
    print(f"  Network: initial_dim={initial_dim}, hidden_dim={hidden_dim}, Res_N={res_n}")
    print(f"  Save dir: {save_dir}")
    print()

    # ---- Build puzzle ----
    puzzle = PuzzleJaxPuxleEnv(game=game, level_i=level_i)
    rule_heuristic = PuzzleJaxHeuristic(puzzle)

    # ---- Build neural heuristic ----
    model_path = os.path.join(save_dir, "heuristic.pkl")
    neural_heuristic = PuzzleScriptNeuralHeuristic(
        puzzle,
        path=model_path if resume and os.path.exists(model_path) else None,
        initial_dim=initial_dim,
        hidden_dim=hidden_dim,
        Res_N=res_n,
    )

    # ---- Build combined heuristic for search ----
    combined_heuristic = NeuralHeuristicWrapper(
        neural_heuristic, rule_heuristic, blend_alpha=blend_alpha,
    )

    # ---- Replay buffer ----
    replay = ReplayBuffer(max_size=replay_max_size)
    replay_path = os.path.join(save_dir, "replay_buffer.pkl")
    if resume:
        replay.load(replay_path)
        print(f"  Resumed replay buffer: {replay.size} samples")

    # ---- Optimizer ----
    # Use a simple adam optimizer with cosine schedule
    total_train_steps = n_iterations * train_steps_per_iter
    warmup_steps = min(500, total_train_steps // 10)
    warmup_schedule = optax.linear_schedule(init_value=1e-6, end_value=lr, transition_steps=warmup_steps)
    decay_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max(1, total_train_steps - warmup_steps))
    lr_schedule = optax.join_schedules([warmup_schedule, decay_schedule], boundaries=[warmup_steps])
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(neural_heuristic.params)

    # ---- Build JIT-compiled train step ----
    train_step = build_train_step(neural_heuristic.model, optimizer, neural_heuristic.multihot_shape)

    # ---- Tracking ----
    history = []
    best_cost = float("inf")
    best_iter = -1
    global_train_step = 0

    # ---- Load checkpoint if resuming ----
    checkpoint_path = os.path.join(save_dir, "checkpoint.json")
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        best_cost = ckpt.get("best_cost", float("inf"))
        best_iter = ckpt.get("best_iter", -1)
        global_train_step = ckpt.get("global_train_step", 0)
        history = ckpt.get("history", [])
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"  Resumed from iteration {start_iter}, best_cost={best_cost}, buffer={replay.size}")
    else:
        start_iter = 0

    # ---- Get initial state ----
    solve_config, init_state = puzzle.get_inits(jax.random.PRNGKey(0))

    # ---- Main ExIt loop ----
    for iteration in range(start_iter, n_iterations):
        iter_start = time.time()
        print(f"\n{'─'*70}")
        print(f"Iteration {iteration}/{n_iterations}")
        print(f"{'─'*70}")

        # ==================================================================
        # Phase 1: SEARCH with current heuristic
        # ==================================================================
        print("  [Search] Running A* ...", end=" ", flush=True)

        # Update neural heuristic params for search
        neural_heuristic.params = neural_heuristic.params  # already up to date

        # Build fresh A* with current heuristic
        # We rebuild each iteration because the heuristic closure captures params
        search_fn = astar_builder(
            puzzle,
            combined_heuristic,
            batch_size=batch_size,
            max_nodes=max_nodes,
            cost_weight=cost_weight,
        )

        search_start = time.time()
        search_result = search_fn(solve_config, init_state)
        solved = bool(search_result.solved.block_until_ready())
        search_time = time.time() - search_start
        generated = int(search_result.generated_size)
        states_per_sec = generated / search_time if search_time > 0 else 0

        status = "SOLVED" if solved else "UNSOLVED"
        cost_str = f"cost={float(search_result.get_cost(search_result.solved_idx)):.1f}" if solved else ""
        print(f"{status} {cost_str} | {generated:,} states | {search_time:.2f}s | {states_per_sec:,.0f} st/s")

        # ==================================================================
        # Phase 2: EXTRACT training data from search
        # ==================================================================
        print("  [Extract] Mining training data ...", end=" ", flush=True)
        data = extract_training_data(puzzle, search_result, solve_config)
        if data is not None:
            replay.add(data)
            print(f"{data['n_expanded']} samples (buffer: {replay.size})")
        else:
            print("No data extracted")

        # Track solution quality
        if solved:
            sol_cost = float(search_result.get_cost(search_result.solved_idx))
            if sol_cost < best_cost:
                best_cost = sol_cost
                best_iter = iteration
                # Save best model
                neural_heuristic.save_model(os.path.join(save_dir, "heuristic_best.pkl"))

        # ==================================================================
        # Phase 2b: BEST-F PATH (env heuristic) for diagnostics
        # ==================================================================
        print("  [Best-f] Extracting best env-f path ...", end=" ", flush=True)
        best_f_info = extract_best_env_f_path(
            puzzle, search_result, solve_config, init_state,
            rule_heuristic=rule_heuristic, render=True,
        )
        if best_f_info is not None:
            print(f"g={best_f_info['best_g']:.1f}  h_env={best_f_info['best_env_h']:.1f}  "
                  f"f_env={best_f_info['best_env_f']:.1f}  "
                  f"path_len={len(best_f_info['best_actions'])}")
            # Render GIF of best-f path
            if best_f_info["path_steps"] is not None and len(best_f_info["path_steps"]) > 0:
                try:
                    import imageio
                    gif_dir = os.path.join(save_dir, "gifs")
                    os.makedirs(gif_dir, exist_ok=True)
                    gif_path = os.path.join(gif_dir, f"best_f_iter{iteration:04d}.gif")
                    imgs = []
                    path_states = [step.state for step in best_f_info["path_steps"]]
                    for idx_s, step in enumerate(best_f_info["path_steps"]):
                        img = step.state.img(idx=idx_s, path=path_states, solve_config=solve_config)
                        imgs.append(img)
                    if imgs:
                        fps = max(4, len(imgs) / 10.0)
                        imageio.mimsave(gif_path, imgs, fps=fps)
                        print(f"           GIF saved: {gif_path}")
                except Exception as e:
                    print(f"           GIF render failed: {e}")
        else:
            print("No expanded nodes")

        # ==================================================================
        # Phase 3: TRAIN neural heuristic
        # ==================================================================
        if replay.size < train_batch_size:
            print(f"  [Train] Buffer too small ({replay.size} < {train_batch_size}), skipping")
        else:
            print(f"  [Train] Training for {train_steps_per_iter} steps ...", end=" ", flush=True)
            train_start = time.time()
            rng = np.random.default_rng(iteration)
            epoch_losses = []

            params = neural_heuristic.params
            for step in range(train_steps_per_iter):
                batch = replay.sample(train_batch_size, rng)
                batch_ml = jnp.array(batch["multihot_levels"])
                batch_tgt = jnp.array(batch["targets"])
                batch_w = jnp.array(batch["weights"])
                # Normalize weights
                batch_w = batch_w / (jnp.mean(batch_w) + 1e-8)

                params, opt_state, metrics = train_step(
                    params, opt_state, batch_ml, batch_tgt, batch_w,
                )
                epoch_losses.append(float(metrics["loss"]))
                global_train_step += 1

            neural_heuristic.params = params
            train_time = time.time() - train_start
            mean_loss = np.mean(epoch_losses)
            print(f"loss={mean_loss:.4f} | {train_time:.2f}s")
            print(f"           mean_pred={float(metrics['mean_pred']):.2f}  "
                  f"mean_target={float(metrics['mean_target']):.2f}  "
                  f"mean_abs_diff={float(metrics['mean_abs_diff']):.2f}")

        # ==================================================================
        # Phase 4: LOG and SAVE
        # ==================================================================
        iter_time = time.time() - iter_start
        record = {
            "iteration": iteration,
            "solved": solved,
            "cost": float(search_result.get_cost(search_result.solved_idx)) if solved else None,
            "generated_states": generated,
            "search_time": search_time,
            "states_per_sec": states_per_sec,
            "buffer_size": replay.size,
            "loss": float(mean_loss) if replay.size >= train_batch_size else None,
            "iter_time": iter_time,
            "blend_alpha": blend_alpha,
            "best_cost": best_cost if best_cost < float("inf") else None,
            # Best-f diagnostics (environment heuristic)
            "best_env_h": best_f_info["best_env_h"] if best_f_info else None,
            "best_env_g": best_f_info["best_g"] if best_f_info else None,
            "best_env_f": best_f_info["best_env_f"] if best_f_info else None,
            "best_env_path_len": len(best_f_info["best_actions"]) if best_f_info else None,
        }
        history.append(record)

        # Periodic save
        if iteration % 5 == 0 or iteration == n_iterations - 1:
            neural_heuristic.save_model(model_path)
            replay.save(replay_path)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "iteration": iteration,
                    "best_cost": best_cost if best_cost < float("inf") else None,
                    "best_iter": best_iter,
                    "global_train_step": global_train_step,
                    "history": history,
                }, f, indent=2)

        # Gradually increase neural heuristic influence
        if iteration > 0 and iteration % 10 == 0:
            old_alpha = combined_heuristic.blend_alpha
            new_alpha = min(0.95, old_alpha + 0.05)
            combined_heuristic.blend_alpha = new_alpha
            if new_alpha != old_alpha:
                print(f"  [Schedule] blend_alpha: {old_alpha:.2f} → {new_alpha:.2f}")

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print(f"ExIt Training Complete: {game} level {level_i}")
    print(f"{'='*70}")
    n_solved = sum(1 for r in history if r["solved"])
    print(f"  Solved in {n_solved}/{len(history)} iterations")
    if best_cost < float("inf"):
        print(f"  Best solution cost: {best_cost:.1f} (iteration {best_iter})")
    print(f"  Final buffer size: {replay.size}")
    print(f"  Checkpoints saved to: {save_dir}")

    # Save final model and history
    neural_heuristic.save_model(model_path)
    replay.save(replay_path)
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ExIt training for PuzzleScript games (neural heuristic + A* search)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--game", type=str, required=True, help="PuzzleScript game name")
    parser.add_argument("--level", type=int, default=0, help="Level index (default: 0)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of ExIt iterations")
    parser.add_argument("-m", "--max_nodes", type=int, default=100_000, help="Max A* search nodes")
    parser.add_argument("-b", "--batch_size", type=int, default=1000, help="A* batch size")
    parser.add_argument("-w", "--cost_weight", type=float, default=0.6, help="A* cost weight")
    parser.add_argument("--train_steps_per_iter", type=int, default=200,
                        help="Gradient steps per ExIt iteration")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--blend_alpha", type=float, default=0.5,
                        help="Initial blend weight (0=rule only, 1=neural only)")
    parser.add_argument("--replay_max_size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory")
    # Architecture
    parser.add_argument("--initial_dim", type=int, default=512, help="Neural net initial dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Neural net hidden dim")
    parser.add_argument("--res_n", type=int, default=2, help="Number of residual blocks")

    args = parser.parse_args()

    run_exit_training(
        game=args.game,
        level_i=args.level,
        n_iterations=args.iterations,
        max_nodes=args.max_nodes,
        batch_size=args.batch_size,
        cost_weight=args.cost_weight,
        train_steps_per_iter=args.train_steps_per_iter,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        blend_alpha=args.blend_alpha,
        replay_max_size=args.replay_max_size,
        save_dir=args.save_dir,
        resume=args.resume,
        initial_dim=args.initial_dim,
        hidden_dim=args.hidden_dim,
        res_n=args.res_n,
    )


if __name__ == "__main__":
    main()
