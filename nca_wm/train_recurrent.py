"""Train the recurrent ("memory" / Option B) NCA world model over ordered
transition trajectories.

This is the apples-to-apples counterpart to the ``--history`` feature in
``train.py``: it reuses the EXACT same n_per_rule game selection +
``collect_multigame_dataset`` merged-dataset disk cache (so no re-collection),
but instead of stacking a fixed history window into the model input, it carries
a persistent hidden grid across env ticks (BPTT over a length-L trajectory).

A length-k backward path (sampled with ``sample_backward_paths``) read FORWARD
is a valid teacher-forcing trajectory, because ``next_states[row_i] ==
states[row_{i+1}]`` by construction of the predecessor graph. The last tick is
the "current" transition; we report held-out final-tick change_err there, which
is directly comparable to the history 2x2's per-transition change_err.

Usage (smoke test, CPU):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python3 -m nca_wm.train_recurrent \
    --n_games 4 --n_hid 32 --n_steps 4 --k 3 --n_updates 60 \
    --eval_interval 20 --batch_size 8 --save_dir nca_wm/logs/_recurrent_smoke
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp
import optax

from nca_wm.models import RecurrentNCAWorldModel, N_ACTIONS
from nca_wm.data_collection import (
    collect_multigame_dataset,
    build_predecessor_adjacency,
    sample_backward_paths,
)
from nca_wm.state_ops import _unpack_states

# These are fixed to keep the disk cache key identical to the history 2x2 run
# (torch_history_2x2.sbatch / run_2x2_final.sh): dedup_pool universe, sorted
# strategy, search astar/100k/60s, val_frac/ancestor_closed/max_area threaded
# through. Only the model + trajectory construction differ here.
N_PER_RULE_UNIVERSE = "dedup_pool"
N_PER_RULE_MAX_AREA = 30
N_PER_RULE_STRATEGY = "sorted"
N_PER_RULE_INCLUDE_RANDOM = False
SEARCH_ALGO = "astar"
N_SEARCH_STEPS = 100_000


# ---------------------------------------------------------------------------
# n_per_rule game selection (replica of train.py main(), lines ~2370-2518)
# ---------------------------------------------------------------------------
def select_n_per_rule_games(n_games: int, max_area: int = N_PER_RULE_MAX_AREA):
    """Reproduce train.py's ``--n_per_rule_games`` selection EXACTLY so the
    merged-dataset cache built by the history runs is reused verbatim.

    Returns the ordered list of game names (sorted strategy, dedup_pool).
    """
    heldout_names: set[str] = set()
    heldout_path = _REPO_ROOT / "data" / "heldout_v4_n30.json"
    if heldout_path.is_file():
        heldout_names = {h["name"] for h in
                         json.loads(heldout_path.read_text())["heldout"]}

    meta_path = _REPO_ROOT / "data" / "games_metadata.json"
    meta = json.loads(meta_path.read_text())

    def _meta_for(g: str) -> dict | None:
        for cand in (g + ".txt", g.replace(' ', '_') + ".txt",
                     g.lower() + ".txt"):
            if cand in meta:
                return meta[cand]
        return None

    ranked: list[tuple[int, str]] = []
    # dedup_pool universe (v3 preferred, v2 fallback).
    v3 = _REPO_ROOT / "data" / "dedup_candidates_v3.json"
    v2 = _REPO_ROOT / "data" / "dedup_candidates_v2.json"
    dedup_path = v3 if v3.is_file() else v2
    print(f"  [n_per_rule] universe pool: {dedup_path.name}")
    pool = json.loads(dedup_path.read_text())["candidates"]
    for c in pool:
        name = c["name"]
        if name in heldout_names:
            continue
        n_rules = int(c.get("n_rules", -1))
        if n_rules < 0:
            continue
        area = int(c.get("max_level_area", 999))
        if area > max_area:
            continue
        if not N_PER_RULE_INCLUDE_RANDOM:
            m = _meta_for(name)
            if m is not None and m.get("has_randomness", False):
                continue
        ranked.append((n_rules, name))

    ranked.sort(key=lambda x: (x[0], x[1]))

    # Canvas double-check on cached games (metadata max_level_area is 1-D).
    cap = max_area
    ranked_filtered: list[tuple[int, str]] = []
    for n_rules, name in ranked:
        files = sorted(_glob.glob(
            f"rollout_data/{name}/level_*/astar_transitions_*.npz"))
        bad = False
        for fp in files:
            try:
                with np.load(fp, allow_pickle=True) as d:
                    H = int(d['states'].shape[1])
                    W = int(d['W'])
                    if H > cap or W > cap:
                        bad = True
                        break
            except Exception:
                pass
        if bad:
            continue
        ranked_filtered.append((n_rules, name))
    ranked = ranked_filtered

    game_names = [g for _, g in ranked[:n_games]]
    if not game_names:
        raise RuntimeError(
            "n_per_rule selection produced zero games (empty universe or all "
            "filtered out)")
    print(f"[n_per_rule n_games={n_games} universe={N_PER_RULE_UNIVERSE} "
          f"max_area={max_area} strategy={N_PER_RULE_STRATEGY}]: "
          f"{len(game_names)} games selected from {len(ranked)} eligible")
    return game_names


# ---------------------------------------------------------------------------
# Per-game trajectory infra
# ---------------------------------------------------------------------------
class GameData:
    """Holds the unpacked (or lazily-unpacked) per-game arrays + predecessor
    adjacency + train/val row partition for trajectory sampling."""

    def __init__(self, g, dataset, info):
        self.g = g
        self.name = info["name"]
        self.n_objs = int(info["n_objs"])
        self.H = int(info["H"])
        self.W = int(info["W"])
        self.states_packed = dataset["per_game_states"][g]
        self.next_packed = dataset["per_game_next_states"][g]
        self.actions = np.asarray(dataset["per_game_actions"][g], dtype=np.int64)
        self.n = len(self.states_packed)
        # Predecessor adjacency over PACKED rows (hashes raw bytes).
        self.pred_lists = build_predecessor_adjacency(
            self.states_packed, self.next_packed)
        # Train/val partition: val_idx are held-out rows.
        val_idx = np.asarray(dataset["per_game_val_idx"][g], dtype=np.int64) \
            if dataset.get("per_game_val_idx") is not None else np.empty(0, np.int64)
        is_val = np.zeros(self.n, dtype=bool)
        is_val[val_idx] = True
        self.val_rows = np.nonzero(is_val)[0]
        self.train_rows = np.nonzero(~is_val)[0]

    def unpack(self, packed_rows: np.ndarray) -> np.ndarray:
        """Unpack a (M, C, H, ceil(W/8)) packed array -> (M, C, H, W) float32."""
        return _unpack_states(packed_rows, self.W).astype(np.float32)


def build_trajectory_batch(gd: GameData, target_rows: np.ndarray, k: int,
                           rng, max_C: int, max_H: int, max_W: int):
    """Construct a teacher-forced trajectory batch ending at ``target_rows``.

    Returns (states, actions_onehot, targets, valid):
      states:        (B, L, max_C, max_H, max_W) float32 input frames
      actions_onehot:(B, L, N_ACTIONS) float32
      targets:       (B, L, max_C, max_H, max_W) float32 next-state targets
      valid:         (B, L) bool tick-validity (False at -1 holes)
    L = k + 1 (k history ticks + the current transition).
    """
    B = len(target_rows)
    L = k + 1
    hist_rows, _miss = sample_backward_paths(gd.pred_lists, target_rows, k, rng)
    rows_seq = np.concatenate([hist_rows, target_rows[:, None]], axis=1)  # (B, L)
    valid = rows_seq != -1                                                # (B, L)

    states = np.zeros((B, L, max_C, max_H, max_W), dtype=np.float32)
    targets = np.zeros((B, L, max_C, max_H, max_W), dtype=np.float32)
    actions = np.zeros((B, L), dtype=np.int64)

    C, H, W = gd.n_objs, gd.H, gd.W
    # Unpack only the rows we need (flatten unique rows for one unpack call).
    flat_rows = rows_seq.reshape(-1)
    real = flat_rows >= 0
    uniq = np.unique(flat_rows[real])
    s_unp = gd.unpack(gd.states_packed[uniq])      # (U, C, H, W)
    n_unp = gd.unpack(gd.next_packed[uniq])
    row_to_idx = {int(r): i for i, r in enumerate(uniq)}
    for b in range(B):
        for t in range(L):
            r = int(rows_seq[b, t])
            if r < 0:
                continue
            idx = row_to_idx[r]
            states[b, t, :C, :H, :W] = s_unp[idx][:C, :H, :W]
            targets[b, t, :C, :H, :W] = n_unp[idx][:C, :H, :W]
            actions[b, t] = gd.actions[r]

    actions_onehot = np.eye(N_ACTIONS, dtype=np.float32)[actions]   # (B, L, 5)
    actions_onehot = actions_onehot * valid[..., None].astype(np.float32)
    return (jnp.asarray(states), jnp.asarray(actions_onehot),
            jnp.asarray(targets), jnp.asarray(valid))


# ---------------------------------------------------------------------------
# Loss + metric
# ---------------------------------------------------------------------------
def _real_cell_mask(states):
    """Per-(B,L,H,W) real-cell mask: a cell is real iff >=1 channel set in the
    input state (padded cells are all-zero). Returns (B, L, 1, H, W)."""
    return (states.sum(axis=2, keepdims=True) > 0).astype(jnp.float32)


def loss_fn(params, model, states, actions_onehot, targets, valid):
    """Masked sigmoid-BCE over real cells x valid ticks. Returns (loss, aux)."""
    logits, _win_logits, _sprite = model.apply(params, states, actions_onehot)
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)  # (B,L,C,H,W)
    cell_mask = _real_cell_mask(states)                        # (B,L,1,H,W)
    L = logits.shape[1]
    # With truncated BPTT, only the last bptt_window ticks carry a loss term
    # (their activations are the only ones retained for backward); context
    # ticks are forward-only and stop-gradient'd inside the model.
    w = model.bptt_window if model.bptt_window else L
    win = (jnp.arange(L) >= (L - w)).astype(bce.dtype)         # (L,)
    tick_mask = valid[:, :, None, None, None].astype(bce.dtype) \
        * win[None, :, None, None, None]
    mask = cell_mask * tick_mask                               # broadcast over C
    msum = jnp.maximum((mask * jnp.ones_like(bce)).sum(), 1.0)
    loss = (bce * mask).sum() / msum
    return loss, logits


def compute_change_err(logits, states, targets, valid, final_only=True):
    """change_err matching train.py _heads_loss: preds=logits>0;
    changed=(input!=target)&real_cell; err = 1 - mean over changed of
    (preds==target). Optionally restrict to the FINAL tick only.

    Also returns overall masked cell error. All numpy in/out.
    """
    logits = np.asarray(logits)
    states = np.asarray(states)
    targets = np.asarray(targets)
    valid = np.asarray(valid)
    preds = (logits > 0)
    cell_mask = (states.sum(axis=2, keepdims=True) > 0)        # (B,L,1,H,W)
    cell_mask = np.broadcast_to(cell_mask, preds.shape)
    tick_mask = valid[:, :, None, None, None]
    full_mask = cell_mask & tick_mask
    changed = (states != targets) & full_mask
    correct = (preds == (targets > 0.5))

    if final_only:
        sel = np.zeros_like(full_mask)
        sel[:, -1] = True
        ch = changed & sel
        fm = full_mask & sel
    else:
        ch, fm = changed, full_mask

    n_ch = ch.sum()
    change_err = float(1.0 - (correct[ch].mean())) if n_ch > 0 else 0.0
    n_cells = fm.sum()
    cell_err = float(1.0 - (correct[fm].mean())) if n_cells > 0 else 0.0
    return change_err, cell_err, int(n_ch)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, default=800)
    ap.add_argument("--games_list_file", type=str, default=None,
                    help="newline-separated game-name file. When set, use this "
                         "list directly (skips n_per_rule selection).")
    ap.add_argument("--heldout_game_frac", type=float, default=0.1,
                    help="fraction of GAMES held out for OOD eval (eval-only, "
                         "never trained on).")
    ap.add_argument("--k", type=int, default=4, help="history depth; L = k+1")
    ap.add_argument("--n_hid", type=int, default=288)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--n_updates", type=int, default=50_000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="linear warmup to --lr before cosine decay (0=off).")
    ap.add_argument("--bptt_window", type=int, default=0,
                    help="truncated BPTT: backprop only the last N ticks while "
                    "carrying context forward (0=full BPTT). Lets k be large.")
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--eval_interval", type=int, default=2500)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--max_transitions_per_game", type=int, default=20_000)
    ap.add_argument("--max_grid_dim", type=int, default=30)
    ap.add_argument("--search_timeout_ms", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_eval_rows", type=int, default=2048,
                    help="cap on val rows sampled for eval (per game balanced)")
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[train_recurrent] devices: {jax.devices()}")

    # --- Game selection + dataset (reuses history-run merged cache) ---
    from puzzlescript_jax.utils import init_ps_lark_parser
    ps_parser = init_ps_lark_parser()
    if args.games_list_file is not None:
        list_path = Path(args.games_list_file)
        game_names = [ln.strip() for ln in list_path.read_text().splitlines()
                      if ln.strip()]
        if not game_names:
            raise RuntimeError(f"empty games_list_file: {args.games_list_file}")
        print(f"[train_recurrent] games_list_file={args.games_list_file}: "
              f"{len(game_names)} games")
    else:
        game_names = select_n_per_rule_games(args.n_games, args.max_grid_dim)

    t0 = time.time()
    dataset, game_infos = collect_multigame_dataset(
        game_names, ps_parser,
        level_i=None,
        n_search_steps=N_SEARCH_STEPS,
        search_timeout_ms=args.search_timeout_ms,
        search_algo=SEARCH_ALGO,
        encode_sprites=False,
        max_transitions_per_game=(args.max_transitions_per_game or None),
        val_frac=args.val_frac,
        history=0,
        ancestor_closed=True,
        max_grid_dim=args.max_grid_dim,
    )
    print(f"[train_recurrent] dataset ready in {time.time()-t0:.1f}s; "
          f"{len(game_infos)} games")

    # --- Per-game wrappers + global pad dims ---
    games = []
    for g, info in enumerate(game_infos):
        gd = GameData(g, dataset, info)
        if gd.n > 0:
            games.append(gd)
    if not games:
        raise RuntimeError("No non-empty games in dataset")
    max_C = max(gd.n_objs for gd in games)
    max_H = max(gd.H for gd in games)
    max_W = max(gd.W for gd in games)

    # --- Hold out whole GAMES for OOD eval (deterministic, seeded) ---
    # Held-out games keep their full GameData (predecessor adjacency etc.) so
    # their trajectories can be sampled for OOD eval; they are simply never
    # sampled for TRAINING.
    split_rng = np.random.default_rng(args.seed)
    order = split_rng.permutation(len(games))
    n_heldout = int(round(args.heldout_game_frac * len(games)))
    n_heldout = min(max(n_heldout, 0), len(games))
    heldout_set = set(int(i) for i in order[len(games) - n_heldout:]) \
        if n_heldout > 0 else set()
    train_game_objs = [games[i] for i in range(len(games))
                       if i not in heldout_set]
    heldout_game_objs = [games[i] for i in range(len(games))
                         if i in heldout_set]
    print(f"[train_recurrent] train games={len(train_game_objs)}, "
          f"heldout games={len(heldout_game_objs)}")

    n_train_games = sum(1 for gd in train_game_objs if len(gd.train_rows) > 0)
    n_val_games = sum(1 for gd in train_game_objs if len(gd.val_rows) > 0)
    print(f"[train_recurrent] {len(games)} games; max C/H/W = "
          f"{max_C}/{max_H}/{max_W}; train-games={n_train_games} "
          f"val-games={n_val_games}")

    # --- Model + init ---
    model = RecurrentNCAWorldModel(
        n_hid=args.n_hid, n_steps=args.n_steps, n_out=max_C,
        axis_pool=True, axis_cummax=True, global_pool=True, input_skip=True,
        bptt_window=args.bptt_window,
    )
    L = args.k + 1
    key = jax.random.PRNGKey(args.seed)
    dummy_states = jnp.zeros((1, L, max_C, max_H, max_W), jnp.float32)
    dummy_actions = jnp.zeros((1, L, N_ACTIONS), jnp.float32)
    params = model.init(key, dummy_states, dummy_actions)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"[train_recurrent] params: {n_params:,}")

    # --- Optimizer ---
    if args.warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=max(args.warmup_steps + 1, args.n_updates),
            end_value=1e-7)
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=args.lr, decay_steps=max(1, args.n_updates),
            alpha=(1e-7 / args.lr))
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, states, actions_onehot, targets, valid):
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model, states, actions_onehot, targets, valid)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_forward(params, states, actions_onehot):
        logits, _w, _s = model.apply(params, states, actions_onehot)
        return logits

    train_games = [gd for gd in train_game_objs if len(gd.train_rows) > 0]
    if not train_games:
        raise RuntimeError("No train games with train rows after holdout split")

    def sample_train_batch():
        gd = train_games[rng.integers(len(train_games))]
        sel = rng.choice(gd.train_rows, size=args.batch_size,
                         replace=len(gd.train_rows) < args.batch_size)
        return build_trajectory_batch(gd, sel, args.k, rng, max_C, max_H, max_W)

    def _eval_over_games(eval_games, rows_attr):
        """Count-weighted final-tick change_err over a set of games, sampling
        ``rows_attr`` rows (balanced + capped by args.max_eval_rows) as eval
        targets. Returns (change_err, cell_err)."""
        eval_games = [gd for gd in eval_games
                      if len(getattr(gd, rows_attr)) > 0]
        if not eval_games:
            return float("nan"), float("nan")
        tot_ch_num, tot_ch_den = 0.0, 0
        tot_cell_num, tot_cell_den = 0.0, 0
        cap_per_game = max(1, args.max_eval_rows // len(eval_games))
        bs = args.batch_size
        for gd in eval_games:
            rows = getattr(gd, rows_attr)
            if len(rows) > cap_per_game:
                rows = rng.choice(rows, size=cap_per_game, replace=False)
            for i in range(0, len(rows), bs):
                chunk = rows[i:i + bs]
                states, a_oh, targets, valid = build_trajectory_batch(
                    gd, chunk, args.k, rng, max_C, max_H, max_W)
                logits = eval_forward(params, states, a_oh)
                ce, cle, n_ch = compute_change_err(
                    logits, states, targets, valid, final_only=True)
                if n_ch > 0:
                    tot_ch_num += ce * n_ch
                    tot_ch_den += n_ch
                st = np.asarray(states)
                vl = np.asarray(valid)
                cm = (st.sum(axis=2) > 0)[:, -1]  # (B,H,W) final tick
                cm = cm & vl[:, -1][:, None, None]
                n_cells = int(cm.sum())
                if n_cells > 0:
                    tot_cell_num += cle * n_cells
                    tot_cell_den += n_cells
        change_err = tot_ch_num / tot_ch_den if tot_ch_den > 0 else float("nan")
        cell_err = tot_cell_num / tot_cell_den if tot_cell_den > 0 else float("nan")
        return change_err, cell_err

    # --- Eval helpers ---
    # in-dist: change_err on val ROWS of TRAIN games.
    def run_eval():
        return _eval_over_games(train_game_objs, "val_rows")

    # OOD: change_err on HELD-OUT games (use ALL their rows as eval targets).
    def run_ood_eval():
        if not heldout_game_objs:
            return float("nan"), float("nan")
        # Use every row (train+val) of each held-out game; the model has never
        # seen any of these games during training.
        for gd in heldout_game_objs:
            if not hasattr(gd, "all_rows"):
                gd.all_rows = np.arange(gd.n, dtype=np.int64)
        return _eval_over_games(heldout_game_objs, "all_rows")

    # --- Config dump ---
    config = dict(
        recurrent=True,
        n_games=args.n_games, n_games_used=len(games), k=args.k, L=L,
        games_list_file=args.games_list_file,
        heldout_game_frac=args.heldout_game_frac,
        n_train_games=len(train_game_objs),
        n_heldout_games=len(heldout_game_objs),
        n_hid=args.n_hid, n_steps=args.n_steps, n_out=max_C,
        n_updates=args.n_updates, batch_size=args.batch_size,
        lr=args.lr, lr_schedule="cosine", lr_min=1e-7, grad_clip=args.grad_clip,
        eval_interval=args.eval_interval, val_frac=args.val_frac,
        max_transitions_per_game=args.max_transitions_per_game,
        max_grid_dim=args.max_grid_dim, search_timeout_ms=args.search_timeout_ms,
        search_algo=SEARCH_ALGO, n_search_steps=N_SEARCH_STEPS,
        seed=args.seed, max_C=max_C, max_H=max_H, max_W=max_W,
        axis_pool=True, axis_cummax=True, global_pool=True, input_skip=True,
        n_per_rule_universe=N_PER_RULE_UNIVERSE,
        n_per_rule_strategy=N_PER_RULE_STRATEGY,
        n_per_rule_max_area=N_PER_RULE_MAX_AREA,
    )
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Training loop ---
    # Best checkpoint is selected by OOD change_err (held-out games). When
    # there are no held-out games (heldout_game_frac=0), fall back to in-dist.
    best_ood = float("inf")
    recent_losses = []
    t_start = time.time()
    for step in range(1, args.n_updates + 1):
        states, a_oh, targets, valid = sample_train_batch()
        params, opt_state, loss = train_step(
            params, opt_state, states, a_oh, targets, valid)
        recent_losses.append(float(loss))

        if step % max(1, args.eval_interval) == 0 or step == 1 \
                or step == args.n_updates:
            avg_loss = float(np.mean(recent_losses[-args.eval_interval:]))
            id_ch, _id_cell = run_eval()
            ood_ch, _ood_cell = run_ood_eval()
            elapsed = time.time() - t_start
            print(f"[step {step:>7}/{args.n_updates}] loss={avg_loss:.4e}  "
                  f"in_dist_change_err={id_ch:.4e}  "
                  f"ood_change_err={ood_ch:.4e}  ({elapsed:.0f}s)", flush=True)
            sel_metric = ood_ch if np.isfinite(ood_ch) else id_ch
            if np.isfinite(sel_metric) and sel_metric < best_ood:
                best_ood = sel_metric
                with open(os.path.join(args.save_dir, "params_best.pkl"),
                          "wb") as f:
                    pickle.dump(params, f)

    with open(os.path.join(args.save_dir, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    # Make sure a best checkpoint exists even if no eval improved it.
    best_path = os.path.join(args.save_dir, "params_best.pkl")
    if not os.path.isfile(best_path):
        with open(best_path, "wb") as f:
            pickle.dump(params, f)
    # Final metrics.
    final_id_ch, _ = run_eval()
    final_ood_ch, _ = run_ood_eval()
    config["best_ood_change_err"] = (None if not np.isfinite(best_ood)
                                     else best_ood)
    config["final_in_dist_change_err"] = (None if not np.isfinite(final_id_ch)
                                          else final_id_ch)
    config["final_ood_change_err"] = (None if not np.isfinite(final_ood_ch)
                                      else final_ood_ch)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[train_recurrent] done. best ood change_err={best_ood:.4e}; "
          f"saved to {args.save_dir}")


if __name__ == "__main__":
    main()
