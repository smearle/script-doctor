"""Expert Iteration (ExIt) training loop for PuzzleScript games via PyTorch.

This mirrors the JAX ExIt pipeline more closely than the PPO trainers:
  1. Run weighted A* using a learned neural heuristic blended with the
     backend's native heuristic score.
  2. Extract min-descendant-f targets from the search tree.
  3. Train the heuristic network on those targets with replay.

Search expansion uses a single backend env with explicit state snapshots so the
Python A* loop stays strictly best-first without batched frontier pops.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from exit_train_backend_common import build_backend_runtime, save_rollout_gif, write_json
from exit_training_config import build_run_config, run_subdir_name
from puzzlescript_cpp import CppPuzzleScriptEnv
from puzzlescript_nodejs.rl_env import NodeJSEnvState, NodeJSPuzzleEnv


EXIT_TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exit_training_pytorch")
N_ACTIONS = 5


def _job_root_dir(backend: str, game: str, level_i: int, save_dir: Optional[str] = None) -> str:
    if save_dir is not None:
        return save_dir
    return os.path.join(EXIT_TRAINING_DIR, backend, f"{game}_level{level_i}")


def _resolve_run_dir(run_root_dir: str, run_config: dict[str, object]) -> str:
    return os.path.join(run_root_dir, run_subdir_name(run_config))


def _write_run_config(save_dir: str, run_config: dict[str, object]) -> None:
    write_json(os.path.join(save_dir, "run_config.json"), run_config)


def layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + residual)


class PuzzleScriptHeuristicNet(nn.Module):
    def __init__(self, input_dim: int, initial_dim: int, hidden_dim: int, res_n: int) -> None:
        super().__init__()
        self.fc_in = layer_init(nn.Linear(input_dim, initial_dim))
        self.fc_hidden = layer_init(nn.Linear(initial_dim, hidden_dim))
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim) for _ in range(res_n))
        self.fc_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().flatten(1)
        x = (x - 0.5) * 2.0
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc_hidden(x))
        for block in self.blocks:
            x = block(x)
        return F.relu(self.fc_out(x).squeeze(-1))


@dataclass
class SearchNode:
    obs: np.ndarray
    key: bytes
    g: float
    env_h: float
    nn_h: float
    combined_h: float
    parent: int
    action_from_parent: int
    depth: int
    expanded: bool = False
    solved: bool = False
    children: list[int] = field(default_factory=list)


@dataclass
class SearchRunResult:
    nodes: list[SearchNode]
    solved_idx: Optional[int]
    generated_size: int
    expanded_size: int
    search_time: float

    @property
    def solved(self) -> bool:
        return self.solved_idx is not None


class ReplayBuffer:
    def __init__(self, max_size: int, expected_obs_shape: tuple[int, ...]) -> None:
        self.max_size = int(max_size)
        self.expected_obs_shape = tuple(expected_obs_shape)
        self.obs: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.size = 0
        self.write_ptr = 0

    def _ensure_storage(self, obs: np.ndarray) -> None:
        if self.obs is not None:
            return
        if tuple(obs.shape[1:]) != self.expected_obs_shape:
            raise ValueError(
                f"Replay sample shape {tuple(obs.shape[1:])} does not match expected "
                f"{self.expected_obs_shape}."
            )
        self.obs = np.empty((self.max_size, *obs.shape[1:]), dtype=obs.dtype)
        self.targets = np.empty((self.max_size,), dtype=np.float32)
        self.weights = np.empty((self.max_size,), dtype=np.float32)

    def add(self, data: dict[str, np.ndarray]) -> None:
        obs = np.asarray(data["multihot_levels"])
        targets = np.asarray(data["targets"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        if obs.shape[0] == 0:
            return

        self._ensure_storage(obs)
        n = obs.shape[0]
        if n >= self.max_size:
            self.obs[...] = obs[-self.max_size :]
            self.targets[...] = targets[-self.max_size :]
            self.weights[...] = weights[-self.max_size :]
            self.size = self.max_size
            self.write_ptr = 0
            return

        end_ptr = self.write_ptr + n
        if end_ptr <= self.max_size:
            self.obs[self.write_ptr:end_ptr] = obs
            self.targets[self.write_ptr:end_ptr] = targets
            self.weights[self.write_ptr:end_ptr] = weights
        else:
            first = self.max_size - self.write_ptr
            second = n - first
            self.obs[self.write_ptr:] = obs[:first]
            self.targets[self.write_ptr:] = targets[:first]
            self.weights[self.write_ptr:] = weights[:first]
            self.obs[:second] = obs[first:]
            self.targets[:second] = targets[first:]
            self.weights[:second] = weights[first:]

        self.write_ptr = (self.write_ptr + n) % self.max_size
        self.size = min(self.max_size, self.size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.choice(self.size, min(batch_size, self.size), replace=False)
        return {
            "multihot_levels": self.obs[indices],
            "targets": self.targets[indices],
            "weights": self.weights[indices],
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.savez_compressed(
                f,
                obs=self.obs,
                targets=self.targets,
                weights=self.weights,
                size=np.asarray(self.size, dtype=np.int64),
                write_ptr=np.asarray(self.write_ptr, dtype=np.int64),
                max_size=np.asarray(self.max_size, dtype=np.int64),
            )

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with np.load(path, allow_pickle=False) as data:
            self.obs = data["obs"] if "obs" in data else None
            self.targets = data["targets"] if "targets" in data else None
            self.weights = data["weights"] if "weights" in data else None
            self.size = int(data["size"]) if "size" in data else 0
            self.write_ptr = int(data["write_ptr"]) if "write_ptr" in data else 0
        return self.obs is not None


def create_search_env(runtime):
    if runtime.backend_name == "cpp":
        return CppPuzzleScriptEnv(
            runtime.json_str,
            level_i=runtime.level_i,
            max_episode_steps=runtime.max_episode_steps,
        )
    if runtime.backend_name == "nodejs":
        return NodeJSPuzzleEnv(
            runtime.game,
            runtime.level_i,
            runtime.max_episode_steps,
            game_text=runtime.game_text,
        )
    raise ValueError(f"Unsupported backend: {runtime.backend_name}")


def reset_search_env(runtime, env) -> tuple[np.ndarray, float, bool, object]:
    if runtime.backend_name == "cpp":
        obs, info = env.reset()
        return np.asarray(obs, dtype=np.uint8), float(info["score"]), bool(info["won"]), None

    obs, state = env.reset()
    return np.asarray(obs.multihot_level, dtype=np.uint8), float(state.score), bool(state.won), state


def step_search_env(runtime, env, state: object, action: int) -> tuple[np.ndarray, bool, bool, dict[str, object], object]:
    if runtime.backend_name == "cpp":
        obs, _reward, done, truncated, info = env.step(action)
        return np.asarray(obs, dtype=np.uint8), bool(done), bool(truncated), info, None

    obs, next_state, _reward, done, info = env.step(state, action)
    truncated = bool(done and not info["won"])
    return np.asarray(obs.multihot_level, dtype=np.uint8), bool(info["won"]), truncated, info, next_state


def capture_search_snapshot(runtime, env, state: object) -> object:
    if runtime.backend_name == "cpp":
        return (env._engine.backup_level(), env._steps, env._prev_score)

    assert isinstance(state, NodeJSEnvState)
    return NodeJSEnvState(
        score=float(state.score),
        won=bool(state.won),
        steps=int(state.steps),
        level=state.level,
        objs=list(state.objs),
    )


def restore_search_snapshot(runtime, env, snapshot: object) -> object:
    if runtime.backend_name == "cpp":
        level_backup, steps, prev_score = snapshot
        env._engine.restore_level(level_backup)
        env._steps = int(steps)
        env._prev_score = float(prev_score)
        return None

    assert isinstance(snapshot, NodeJSEnvState)
    env.backend.engine.restoreLevel(snapshot.level)
    return NodeJSEnvState(
        score=float(snapshot.score),
        won=bool(snapshot.won),
        steps=int(snapshot.steps),
        level=snapshot.level,
        objs=list(snapshot.objs),
    )


def get_initial_state(runtime) -> tuple[np.ndarray, float, bool]:
    if runtime.backend_name == "cpp":
        env = CppPuzzleScriptEnv(
            runtime.json_str,
            level_i=runtime.level_i,
            max_episode_steps=runtime.max_episode_steps,
        )
        try:
            obs, info = env.reset()
            return np.asarray(obs, dtype=np.uint8), float(info["score"]), bool(info["won"])
        finally:
            if hasattr(env, "close"):
                env.close()

    env = NodeJSPuzzleEnv(
        runtime.game,
        runtime.level_i,
        runtime.max_episode_steps,
        game_text=runtime.game_text,
    )
    try:
        obs, state = env.reset()
        return np.asarray(obs.multihot_level, dtype=np.uint8), float(state.score), bool(state.won)
    finally:
        if hasattr(env, "close"):
            env.close()


def predict_heuristic(model: nn.Module, device: torch.device, obs_batch: np.ndarray, infer_batch_size: int = 4096) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(obs_batch), infer_batch_size):
            chunk = torch.from_numpy(obs_batch[start:start + infer_batch_size]).to(device)
            preds.append(model(chunk).cpu().numpy())
    if not preds:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)


def reconstruct_actions(nodes: list[SearchNode], idx: int) -> list[int]:
    actions: list[int] = []
    while idx >= 0 and nodes[idx].parent != -1:
        actions.append(nodes[idx].action_from_parent)
        idx = nodes[idx].parent
    actions.reverse()
    return actions


def extract_training_data(search_result: SearchRunResult, max_samples: int = 50000) -> Optional[dict[str, np.ndarray | bool | int]]:
    expanded_indices = [i for i, node in enumerate(search_result.nodes) if node.expanded]
    if not expanded_indices:
        return None

    min_desc_f = {idx: search_result.nodes[idx].g + search_result.nodes[idx].env_h for idx in expanded_indices}
    for idx in sorted(expanded_indices, key=lambda i: search_result.nodes[i].g, reverse=True):
        parent = search_result.nodes[idx].parent
        if parent >= 0 and search_result.nodes[parent].expanded:
            min_desc_f[parent] = min(min_desc_f[parent], min_desc_f[idx])

    if len(expanded_indices) > max_samples:
        rng = np.random.default_rng(42)
        expanded_indices = list(rng.choice(np.asarray(expanded_indices), max_samples, replace=False))

    global_min_f = min(min_desc_f[idx] for idx in expanded_indices)
    solved = search_result.solved
    obs = []
    targets = []
    weights = []
    for idx in expanded_indices:
        node = search_result.nodes[idx]
        target = max(0.0, min_desc_f[idx] - node.g)
        path_quality = min_desc_f[idx] - global_min_f
        if solved:
            weight = 3.0 if path_quality < 1.0 else 1.0
        else:
            weight = 1.5 if path_quality < 1.0 else 0.5
        obs.append(node.obs)
        targets.append(target)
        weights.append(weight)

    return {
        "multihot_levels": np.stack(obs, axis=0),
        "targets": np.asarray(targets, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "solved": solved,
        "n_expanded": len(expanded_indices),
    }


def extract_best_env_f_path(search_result: SearchRunResult) -> Optional[dict[str, object]]:
    if search_result.solved_idx is not None:
        best_idx = int(search_result.solved_idx)
    else:
        expanded_indices = [i for i, node in enumerate(search_result.nodes) if node.expanded]
        if not expanded_indices:
            return None
        best_idx = min(
            expanded_indices,
            key=lambda idx: (
                search_result.nodes[idx].env_h,
                search_result.nodes[idx].g + search_result.nodes[idx].env_h,
                search_result.nodes[idx].g,
            ),
        )
    node = search_result.nodes[best_idx]
    return {
        "best_env_h": float(node.env_h),
        "best_g": float(node.g),
        "best_env_f": float(node.g + node.env_h),
        "best_actions": reconstruct_actions(search_result.nodes, best_idx),
        "n_expanded": sum(1 for search_node in search_result.nodes if search_node.expanded),
    }


def run_astar_search(
    *,
    runtime,
    search_env,
    model: nn.Module,
    device: torch.device,
    max_nodes: int,
    cost_weight: float,
    blend_alpha: float,
) -> SearchRunResult:
    start_time = time.time()

    root_obs, root_env_h, root_solved, _root_state = reset_search_env(runtime, search_env)
    root_nn_h = float(predict_heuristic(model, device, root_obs[None])[0])
    root_combined_h = blend_alpha * root_nn_h + (1.0 - blend_alpha) * root_env_h
    root = SearchNode(
        obs=root_obs,
        key=root_obs.tobytes(),
        g=0.0,
        env_h=root_env_h,
        nn_h=root_nn_h,
        combined_h=root_combined_h,
        parent=-1,
        action_from_parent=-1,
        depth=0,
        solved=root_solved,
    )
    nodes = [root]
    seen = {root.key}
    frontier: list[tuple[float, float, int, int]] = []
    heapq.heappush(frontier, (cost_weight * root_combined_h, root_env_h, 0, 0))
    counter = 1
    expanded_size = 0

    if root_solved:
        return SearchRunResult(nodes=nodes, solved_idx=0, generated_size=1, expanded_size=0, search_time=time.time() - start_time)

    while frontier and len(nodes) < max_nodes:
        _f, _h, _ctr, idx = heapq.heappop(frontier)
        if nodes[idx].expanded or nodes[idx].solved:
            continue

        parent = nodes[idx]
        path = reconstruct_actions(nodes, idx)
        _obs, _env_h, _solved, state = reset_search_env(runtime, search_env)
        for action in path:
            _obs, done, truncated, _info, state = step_search_env(runtime, search_env, state, action)
            if done or truncated:
                raise RuntimeError("Search path replay hit a terminal state before expansion completed.")

        parent_snapshot = capture_search_snapshot(runtime, search_env, state)
        parent.expanded = True
        expanded_size += 1

        solved_idx: Optional[int] = None
        child_obs_batch = []
        child_meta: list[tuple[int, float, bool]] = []
        for action in range(N_ACTIONS):
            branch_state = restore_search_snapshot(runtime, search_env, parent_snapshot)
            child_obs, child_done, child_truncated, info, _next_state = step_search_env(
                runtime,
                search_env,
                branch_state,
                action,
            )
            child_obs_batch.append(child_obs)
            child_meta.append((action, float(info["score"]), bool(child_done and not child_truncated)))

        nn_scores = predict_heuristic(model, device, np.stack(child_obs_batch, axis=0))

        for action, child_obs, nn_h, (meta_action, env_h, child_solved) in zip(
            range(N_ACTIONS),
            child_obs_batch,
            nn_scores,
            child_meta,
            strict=True,
        ):
            if action != meta_action:
                raise RuntimeError("Action metadata mismatch during child expansion.")
            child_key = child_obs.tobytes()
            if child_key in seen:
                continue
            seen.add(child_key)

            child_g = parent.g + 1.0
            combined_h = blend_alpha * float(nn_h) + (1.0 - blend_alpha) * env_h
            child = SearchNode(
                obs=child_obs,
                key=child_key,
                g=child_g,
                env_h=env_h,
                nn_h=float(nn_h),
                combined_h=combined_h,
                parent=idx,
                action_from_parent=action,
                depth=parent.depth + 1,
                solved=child_solved,
            )
            child_idx = len(nodes)
            nodes.append(child)
            parent.children.append(child_idx)

            if child.solved:
                solved_idx = child_idx
                break

            f_score = child_g + cost_weight * combined_h
            heapq.heappush(frontier, (f_score, env_h, counter, child_idx))
            counter += 1
            if len(nodes) >= max_nodes:
                break

        if solved_idx is not None or len(nodes) >= max_nodes:
            return SearchRunResult(
                nodes=nodes,
                solved_idx=solved_idx,
                generated_size=len(nodes),
                expanded_size=expanded_size,
                search_time=time.time() - start_time,
            )

    return SearchRunResult(
        nodes=nodes,
        solved_idx=None,
        generated_size=len(nodes),
        expanded_size=expanded_size,
        search_time=time.time() - start_time,
    )


def run_exit_training(
    backend: str,
    game: str,
    level_i: int,
    n_iterations: int = 50,
    max_nodes: int = 100_000,
    cost_weight: float = 0.6,
    train_steps_per_iter: int = 200,
    train_batch_size: int = 256,
    lr: float = 1e-3,
    blend_alpha: float = 0.7,
    replay_max_size: int = 200_000,
    save_dir: str | None = None,
    resume: bool = False,
    initial_dim: int = 512,
    hidden_dim: int = 256,
    res_n: int = 2,
    max_episode_steps: int = 200,
) -> list[dict[str, object]]:
    run_config = build_run_config(
        game=game,
        level_i=level_i,
        n_iterations=n_iterations,
        max_nodes=max_nodes,
        batch_size=1,
        cost_weight=cost_weight,
        train_steps_per_iter=train_steps_per_iter,
        train_batch_size=train_batch_size,
        lr=lr,
        blend_alpha=blend_alpha,
        replay_max_size=replay_max_size,
        initial_dim=initial_dim,
        hidden_dim=hidden_dim,
        res_n=res_n,
    )
    save_dir = _resolve_run_dir(_job_root_dir(backend, game, level_i, save_dir=save_dir), run_config)
    os.makedirs(save_dir, exist_ok=True)
    _write_run_config(save_dir, run_config)

    runtime = build_backend_runtime(
        game=game,
        level_i=level_i,
        backend_name=backend,
        max_episode_steps=max_episode_steps,
    )
    root_obs, _root_env_h, _root_solved = get_initial_state(runtime)
    obs_shape = tuple(root_obs.shape)
    input_dim = int(np.prod(obs_shape))
    replay = ReplayBuffer(replay_max_size, expected_obs_shape=obs_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PuzzleScriptHeuristicNet(input_dim, initial_dim, hidden_dim, res_n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    search_env = create_search_env(runtime)

    model_path = os.path.join(save_dir, "heuristic.pt")
    replay_path = os.path.join(save_dir, "replay_buffer.npz")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    history_path = os.path.join(save_dir, "history.json")

    history: list[dict[str, object]] = []
    best_cost = float("inf")
    best_iter = -1
    global_train_step = 0
    start_iter = 0

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = list(ckpt.get("history", []))
        best_cost = float(ckpt.get("best_cost", float("inf")))
        best_iter = int(ckpt.get("best_iter", -1))
        global_train_step = int(ckpt.get("global_train_step", 0))
        start_iter = int(ckpt.get("iteration", -1)) + 1
    if resume:
        replay.load(replay_path)
        if os.path.exists(history_path) and not history:
            with open(history_path, "r") as f:
                history = json.load(f)

    print(f"{'='*70}")
    print(f"ExIt Training (PyTorch): {game} level {level_i}")
    print(f"{'='*70}")
    print(f"  Max nodes: {max_nodes:,}")
    print(f"  cost_weight: {cost_weight}  blend_alpha: {blend_alpha}")
    print(f"  Train steps/iter: {train_steps_per_iter}  Train batch: {train_batch_size}")
    print(f"  Network: initial_dim={initial_dim}, hidden_dim={hidden_dim}, Res_N={res_n}")
    print(f"  Device: {device}")
    print(f"  Save dir: {save_dir}")
    print()

    try:
        for iteration in range(start_iter, n_iterations):
            iter_start = time.time()
            print(f"\n{'─'*70}")
            print(f"Iteration {iteration}/{n_iterations}")
            print(f"{'─'*70}")

            print("  [Search] Running A* ...", end=" ", flush=True)
            search_result = run_astar_search(
                runtime=runtime,
                search_env=search_env,
                model=model,
                device=device,
                max_nodes=max_nodes,
                cost_weight=cost_weight,
                blend_alpha=blend_alpha,
            )
            solved = search_result.solved
            search_time = float(search_result.search_time)
            generated = int(search_result.generated_size)
            states_per_sec = generated / search_time if search_time > 0 else 0.0
            cost = search_result.nodes[search_result.solved_idx].g if solved else None
            status = "SOLVED" if solved else "UNSOLVED"
            cost_str = f"cost={cost:.1f}" if cost is not None else ""
            print(f"{status} {cost_str} | {generated:,} states | {search_time:.2f}s | {states_per_sec:,.0f} st/s")

            print("  [Extract] Mining training data ...", end=" ", flush=True)
            data = extract_training_data(search_result)
            if data is not None:
                replay.add(data)
                print(f"{data['n_expanded']} samples (buffer: {replay.size})")
            else:
                print("No data extracted")

            if solved and cost is not None and cost < best_cost:
                best_cost = cost
                best_iter = iteration
                torch.save(model.state_dict(), os.path.join(save_dir, "heuristic_best.pt"))

            print("  [Best-f] Extracting best env-f path ...", end=" ", flush=True)
            best_f_info = extract_best_env_f_path(search_result)
            if best_f_info is not None:
                print(
                    f"g={best_f_info['best_g']:.1f}  h_env={best_f_info['best_env_h']:.1f}  "
                    f"f_env={best_f_info['best_env_f']:.1f}  path_len={len(best_f_info['best_actions'])}"
                )
                if best_f_info["best_actions"]:
                    try:
                        gif_dir = os.path.join(save_dir, "gifs")
                        os.makedirs(gif_dir, exist_ok=True)
                        gif_path = os.path.join(gif_dir, f"best_f_iter{iteration:04d}.gif")
                        save_rollout_gif(
                            runtime,
                            list(best_f_info["best_actions"]),
                            gif_path,
                            frame_duration_s=0.08,
                        )
                        print(f"           GIF saved: {gif_path}")
                    except Exception as exc:
                        print(f"           GIF render failed: {exc}")
            else:
                print("No expanded nodes")

            mean_loss = None
            metrics = None
            if replay.size < train_batch_size:
                print(f"  [Train] Buffer too small ({replay.size} < {train_batch_size}), skipping")
            else:
                print(f"  [Train] Training for {train_steps_per_iter} steps ...", end=" ", flush=True)
                train_start = time.time()
                rng = np.random.default_rng(iteration)
                epoch_losses = []
                epoch_mean_preds = []
                epoch_mean_targets = []
                epoch_mean_diffs = []

                model.train()
                for _ in range(train_steps_per_iter):
                    batch = replay.sample(train_batch_size, rng)
                    obs_t = torch.from_numpy(batch["multihot_levels"]).to(device)
                    targets_t = torch.from_numpy(batch["targets"]).to(device)
                    weights_t = torch.from_numpy(batch["weights"]).to(device)
                    weights_t = weights_t / weights_t.mean().clamp(min=1e-8)

                    preds = model(obs_t)
                    diff = targets_t - preds
                    loss = torch.mean((diff * diff) * weights_t)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_losses.append(float(loss.item()))
                    epoch_mean_preds.append(float(preds.mean().item()))
                    epoch_mean_targets.append(float(targets_t.mean().item()))
                    epoch_mean_diffs.append(float(diff.abs().mean().item()))
                    global_train_step += 1

                model.eval()
                mean_loss = float(np.mean(epoch_losses))
                metrics = {
                    "mean_pred": float(np.mean(epoch_mean_preds)),
                    "mean_target": float(np.mean(epoch_mean_targets)),
                    "mean_abs_diff": float(np.mean(epoch_mean_diffs)),
                }
                train_time = time.time() - train_start
                print(f"loss={mean_loss:.4f} | {train_time:.2f}s")
                print(
                    f"           mean_pred={metrics['mean_pred']:.2f}  "
                    f"mean_target={metrics['mean_target']:.2f}  "
                    f"mean_abs_diff={metrics['mean_abs_diff']:.2f}"
                )

            iter_time = time.time() - iter_start
            record = {
                "iteration": iteration,
                "solved": solved,
                "cost": cost,
                "generated_states": generated,
                "expanded_states": search_result.expanded_size,
                "search_time": search_time,
                "states_per_sec": states_per_sec,
                "buffer_size": replay.size,
                "loss": mean_loss,
                "iter_time": iter_time,
                "blend_alpha": blend_alpha,
                "best_cost": best_cost if best_cost < float("inf") else None,
                "best_env_h": best_f_info["best_env_h"] if best_f_info else None,
                "best_env_g": best_f_info["best_g"] if best_f_info else None,
                "best_env_f": best_f_info["best_env_f"] if best_f_info else None,
                "best_env_path_len": len(best_f_info["best_actions"]) if best_f_info else None,
            }
            if metrics is not None:
                record.update(metrics)
            history.append(record)

            if iteration % 5 == 0 or iteration == n_iterations - 1:
                torch.save(model.state_dict(), model_path)
                replay.save(replay_path)
                torch.save(
                    {
                        "iteration": iteration,
                        "best_cost": best_cost,
                        "best_iter": best_iter,
                        "global_train_step": global_train_step,
                        "history": history,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "run_config": run_config,
                    },
                    checkpoint_path,
                )
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

            if iteration > 0 and iteration % 10 == 0:
                old_alpha = blend_alpha
                blend_alpha = min(0.95, blend_alpha + 0.05)
                if blend_alpha != old_alpha:
                    print(f"  [Schedule] blend_alpha: {old_alpha:.2f} -> {blend_alpha:.2f}")

        torch.save(model.state_dict(), model_path)
        replay.save(replay_path)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        return history
    finally:
        if hasattr(search_env, "close"):
            search_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExIt training for PuzzleScript games (PyTorch neural heuristic + Python A*)",
    )
    parser.add_argument("--backend", choices=("cpp", "nodejs"), default="cpp", help="Backend to execute search expansions")
    parser.add_argument("--game", type=str, required=True, help="PuzzleScript game name")
    parser.add_argument("--level", type=int, default=0, help="Level index")
    parser.add_argument("--iterations", type=int, default=200, help="Number of ExIt iterations")
    parser.add_argument("-m", "--max_nodes", type=int, default=100_000, help="Max A* search nodes")
    parser.add_argument("-w", "--cost_weight", type=float, default=0.6, help="A* cost weight")
    parser.add_argument("--train_steps_per_iter", type=int, default=200, help="Gradient steps per iteration")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Training minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--blend_alpha", type=float, default=0.5, help="Initial blend weight")
    parser.add_argument("--replay_max_size", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume")
    parser.add_argument("--save_dir", type=str, default=None, help="Override save directory")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="Max episode steps for backend envs")
    parser.add_argument("--initial_dim", type=int, default=512, help="Neural net initial dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Neural net hidden dim")
    parser.add_argument("--res_n", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--overwrite", action="store_true", help="Delete the save dir before running")
    args = parser.parse_args()

    if args.overwrite:
        root = _resolve_run_dir(
            _job_root_dir(args.backend, args.game, args.level, save_dir=args.save_dir),
            build_run_config(
                game=args.game,
                level_i=args.level,
                n_iterations=args.iterations,
                max_nodes=args.max_nodes,
                batch_size=1,
                cost_weight=args.cost_weight,
                train_steps_per_iter=args.train_steps_per_iter,
                train_batch_size=args.train_batch_size,
                lr=args.lr,
                blend_alpha=args.blend_alpha,
                replay_max_size=args.replay_max_size,
                initial_dim=args.initial_dim,
                hidden_dim=args.hidden_dim,
                res_n=args.res_n,
            ),
        )
        if os.path.isdir(root):
            shutil.rmtree(root)

    run_exit_training(
        backend=args.backend,
        game=args.game,
        level_i=args.level,
        n_iterations=args.iterations,
        max_nodes=args.max_nodes,
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
        max_episode_steps=args.max_episode_steps,
    )


if __name__ == "__main__":
    main()
