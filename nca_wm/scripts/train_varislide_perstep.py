"""Per-step engine-aligned supervision for multi-grid varislide.

Test the algorithmic-alignment hypothesis: the model can't learn the slide
rule's `again`-iteration from final-step BCE alone, but CAN learn it when
each NCA step is supervised against the corresponding engine intermediate
state.

Engine intermediate states for varislide are computed by hand (one rule):
starting from the input, at each iteration k, every (Player, no-Wall-to-right)
pair moves the player one cell right; iterate until convergence.

Loss: sum_k BCE(NCA_step_k_logits, engine_state_at_iter_k), with steps
beyond convergence supervised against the converged final state.

This isolates whether the architecture *can* learn iteration with the right
gradient signal — if it does, that motivates extending the per-step target
generation to arbitrary games via the engine in a follow-up.

Usage:
    .venv/bin/python3 nca_wm/scripts/train_varislide_perstep.py \\
        --save_dir nca_wm/logs_canary/varislide_perstep_h128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel  # noqa: E402
from nca_wm.train import _unpack_states  # noqa: E402
from nca_wm.tokenize_game import (  # noqa: E402
    VOCAB_SIZE_EXT,
    get_game_tree_from_js,
    tokenize_game,
)
from puzzlescript_cpp import CppPuzzleScriptBackend  # noqa: E402
from puzzlescript_jax.utils import init_ps_lark_parser  # noqa: E402

ROLLOUT_CACHE_DIR = os.path.join(_REPO, "rollout_data")
N_ACTIONS = 5
RIGHT_ACTION = 3
PLAYER, WALL, BG = 2, 1, 0  # varislide channel indices


def simulate_varislide_iterations(state: np.ndarray, action: int,
                                   n_max_iters: int,
                                   active_H: int | None = None,
                                   active_W: int | None = None) -> np.ndarray:
    """Return (n_max_iters + 1, C, H, W) intermediate states.

    state[0] = input, state[k] = state after k applications of the slide rule,
    padded with the converged state for steps past convergence.

    If active_H/active_W are passed, the rule only fires within the [0:active_H,
    0:active_W] region (matches engine, which sees the level at native size).
    """
    C, H, W = state.shape
    aH = H if active_H is None else min(active_H, H)
    aW = W if active_W is None else min(active_W, W)
    out = np.zeros((n_max_iters + 1, C, H, W), dtype=state.dtype)
    out[0] = state
    if action != RIGHT_ACTION:
        for k in range(1, n_max_iters + 1):
            out[k] = state
        return out
    cur = state.copy()
    for k in range(1, n_max_iters + 1):
        new = cur.copy()
        changed = False
        for r in range(aH):
            for c in range(aW - 1):
                # rule: [> Player | no Wall] -> [ | > Player]
                if cur[PLAYER, r, c] > 0.5 and cur[WALL, r, c + 1] < 0.5:
                    new[PLAYER, r, c] = 0.0
                    new[PLAYER, r, c + 1] = 1.0
                    changed = True
        cur = new
        out[k] = cur
        if not changed:
            for kk in range(k + 1, n_max_iters + 1):
                out[kk] = cur
            return out
    return out


def load_dataset(game: str = "varislide", train_seed: int = 0,
                 widths: set[int] | None = None):
    """Load all multi-grid synth caches for the game.

    Returns (states_padded, actions, next_states_padded, native_lH, native_lW, max_H, max_W)
    where native_l* gives each transition's pre-padding spatial size, so the
    intermediate-state simulator can iterate within the active region only.
    """
    pat = os.path.join(ROLLOUT_CACHE_DIR, game,
                       f"synthetic_*x*",
                       f"seed{train_seed}_n*_v*_mode-*solv0*.npz")
    caches = sorted(glob.glob(pat))
    if not caches:
        raise RuntimeError(f"no synth caches for {game}; run train.py first to populate")
    all_states, all_actions, all_next, all_lH, all_lW = [], [], [], [], []
    grid_max_H, grid_max_W = 0, 0
    for c in caches:
        dirname = os.path.basename(os.path.dirname(c))
        if widths is not None:
            try:
                wh = dirname.removeprefix("synthetic_")
                w = int(wh.split("x", 1)[0])
            except Exception:
                w = -1
            if w not in widths:
                continue
        z = np.load(c)
        s_raw, a, n_raw = z["states"], z["actions"], z["next_states"]
        if "W" in z.files:
            lW = int(z["W"])
            s = _unpack_states(s_raw, lW).astype(np.float32)
            n = _unpack_states(n_raw, lW).astype(np.float32)
        else:
            s = s_raw.astype(np.float32)
            n = n_raw.astype(np.float32)
        all_states.append(s); all_actions.append(a); all_next.append(n)
        all_lH.append(np.full(len(s), s.shape[2], dtype=np.int32))
        all_lW.append(np.full(len(s), s.shape[3], dtype=np.int32))
        grid_max_H = max(grid_max_H, s.shape[2])
        grid_max_W = max(grid_max_W, s.shape[3])
    if not all_states:
        width_msg = f" matching widths={sorted(widths)}" if widths is not None else ""
        raise RuntimeError(f"no synth caches for {game} seed={train_seed}{width_msg}")
    pad_states, pad_next = [], []
    for s, n in zip(all_states, all_next):
        ph, pw = grid_max_H - s.shape[2], grid_max_W - s.shape[3]
        s_p = np.pad(s, [(0, 0), (0, 0), (0, ph), (0, pw)])
        n_p = np.pad(n, [(0, 0), (0, 0), (0, ph), (0, pw)])
        pad_states.append(s_p); pad_next.append(n_p)
    states = np.concatenate(pad_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    next_states = np.concatenate(pad_next, axis=0)
    native_lH = np.concatenate(all_lH, axis=0)
    native_lW = np.concatenate(all_lW, axis=0)
    return states, actions, next_states, native_lH, native_lW, grid_max_H, grid_max_W


def precompute_intermediate_states(states: np.ndarray, actions: np.ndarray,
                                    native_lH: np.ndarray, native_lW: np.ndarray,
                                    n_steps: int) -> np.ndarray:
    """For each transition, generate the (n_steps+1, C, H, W) sequence of engine
    states. Simulation is restricted to the original (lH, lW) active region so
    the player can't escape into zero-padding.

    Returns (N, n_steps+1, C, H, W).
    """
    N = len(states)
    C, H, W = states.shape[1], states.shape[2], states.shape[3]
    out = np.zeros((N, n_steps + 1, C, H, W), dtype=np.float32)
    for i in range(N):
        out[i] = simulate_varislide_iterations(
            states[i], int(actions[i]), n_steps,
            active_H=int(native_lH[i]), active_W=int(native_lW[i]))
    return out


def build_game_info(game: str, states: np.ndarray, max_H: int, max_W: int,
                    encode_sprites: bool = False) -> tuple[list[dict], dict]:
    """Compile/tokenize the game when no prior run dir is available."""
    ps_parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, game)
    tree, canonical_ids = get_game_tree_from_js(ps_parser, game)
    token_ids = tokenize_game(tree, canonical_ids, encode_sprites=encode_sprites)
    max_token_id = max(token_ids) if token_ids else 0
    cfg = {
        "vocab_size": max(max_token_id + 1, VOCAB_SIZE_EXT + 1),
        "d_model": 64,
        "n_enc_layers": 2,
        "d_slot": 64,
        "n_heads": 4,
        "n_app_slots": 0,
    }
    info = {
        "name": game,
        "json_str": json_str,
        "n_objs": int(states.shape[1]),
        "H": int(max_H),
        "W": int(max_W),
        "n_levels": 0,
        "token_ids": token_ids,
    }
    return [info], cfg


def make_model(n_hid: int, n_steps: int, n_out: int, n_slots: int,
               src_cfg: dict, max_seq_len: int, input_skip: bool,
               use_vq: bool = False, vq_codebook_size: int = 512,
               vq_commitment_weight: float = 0.25):
    return RuleAttnNCAWorldModel(
        n_hid=n_hid, n_steps=n_steps, n_out=n_out,
        vocab_size=src_cfg["vocab_size"] + 1,
        enc_d_model=src_cfg["d_model"], enc_n_self_layers=src_cfg["n_enc_layers"],
        n_slots=n_slots, n_app_slots=src_cfg.get("n_app_slots", 0),
        d_slot=src_cfg["d_slot"],
        n_attn_heads=src_cfg["n_heads"], max_seq_len=max_seq_len,
        axis_pool=True, axis_cummax=True, global_pool=True,
        use_vq=use_vq,
        vq_codebook_size=vq_codebook_size,
        vq_commitment_weight=vq_commitment_weight,
        use_layernorm=False, input_skip=input_skip,
        n_repeats=n_steps,           # fully shared body
        adaptive_halt=True,          # to expose per-step logits
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_hid", type=int, default=128)
    p.add_argument("--n_steps", type=int, default=16)
    p.add_argument("--n_slots", type=int, default=16)
    p.add_argument("--input_skip", action="store_true")
    p.add_argument("--change_loss_weight", type=float, default=5.0)
    p.add_argument("--per_step_loss_weight", type=float, default=1.0,
                    help="Weight on per-step supervision loss vs final-step.")
    p.add_argument("--final_only", action="store_true",
                    help="Disable per-step loss (control: same training, only final-step BCE).")
    p.add_argument("--n_updates", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=250)
    p.add_argument("--game", default="varislide")
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--widths", default=None,
                    help="Optional comma-separated synthetic widths to load, "
                         "useful when old caches exist for a stale game spec.")
    p.add_argument("--game_info_from", default=None,
                    help="Optional prior run dir to borrow game_infos.pkl + config. "
                         "If omitted, compile/tokenize --game directly.")
    p.add_argument("--vq_codebook", action="store_true",
                    help="Quantize rule-attention slots with a VQ codebook.")
    p.add_argument("--vq_codebook_size", type=int, default=512)
    p.add_argument("--vq_commitment_weight", type=float, default=0.25)
    p.add_argument("--vq_loss_weight", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[load] dataset...")
    widths = ({int(x) for x in args.widths.split(",") if x.strip()}
              if args.widths else None)
    states, actions, next_states, native_lH, native_lW, H, W = load_dataset(
        args.game, train_seed=args.train_seed, widths=widths)
    print(f"[load] {len(states)} transitions, max grid={H}x{W}")
    print(f"[load] right-action transitions: {(actions == RIGHT_ACTION).sum()}")
    print(f"[load] native widths in dataset: {sorted(set(native_lW.tolist()))}")

    if args.game_info_from:
        print(f"[load] borrowing game_infos from {args.game_info_from}")
        gi = pickle.load(open(os.path.join(args.game_info_from, "game_infos.pkl"), "rb"))
        src_cfg = json.load(open(os.path.join(args.game_info_from, "config.json")))
    else:
        print(f"[load] compiling/tokenizing {args.game}")
        gi, src_cfg = build_game_info(args.game, states, H, W)
    g = gi[0]
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)
    # Save game_infos so the inspector can find it
    with open(os.path.join(args.save_dir, "game_infos.pkl"), "wb") as f:
        pickle.dump(gi, f)
    C_pad = src_cfg.get("C_pad")
    if C_pad is None:
        if args.game_info_from:
            # Read from borrowed params
            src_params = pickle.load(open(os.path.join(args.game_info_from, "params_best.pkl"), "rb"))
            C_pad = src_params["params"]["embed"]["kernel"].shape[0] - N_ACTIONS
        else:
            C_pad = states.shape[1]
    args.vocab_size = int(src_cfg["vocab_size"])
    args.C_pad = int(C_pad)
    json.dump(vars(args), open(os.path.join(args.save_dir, "config.json"), "w"), indent=2)
    # Pad states to C_pad channels for the model
    if states.shape[1] != C_pad:
        ph_c = C_pad - states.shape[1]
        states_padded = np.pad(states, [(0, 0), (0, ph_c), (0, 0), (0, 0)])
        next_padded = np.pad(next_states, [(0, 0), (0, ph_c), (0, 0), (0, 0)])
    else:
        states_padded = states
        next_padded = next_states

    print(f"[precompute] intermediate states for {len(states)} transitions, n_steps={args.n_steps}...")
    t0 = time.time()
    inter = precompute_intermediate_states(states_padded, actions, native_lH, native_lW, args.n_steps)
    print(f"[precompute] done in {time.time()-t0:.1f}s, shape={inter.shape}, "
          f"size={inter.nbytes/1e6:.1f} MB")
    # Sanity: compare final intermediate state vs cached next_state for action=right
    if (actions == RIGHT_ACTION).any():
        ridx = np.where(actions == RIGHT_ACTION)[0][:5]
        for ix in ridx:
            sim_final = inter[ix, args.n_steps]
            cached_next = next_padded[ix]
            diff = float(np.abs(sim_final - cached_next).mean())
            if diff > 1e-3:
                print(f"  [SANITY WARN] transition {ix}: simulated final differs from cached next by mean abs {diff:.4f}")

    # Sanity: for action=right, intermediate states should differ from state_0
    right_mask = (actions == RIGHT_ACTION)
    if right_mask.any():
        idx = np.where(right_mask)[0][0]
        diffs_per_step = [(inter[idx, k] != inter[idx, 0]).any() for k in range(args.n_steps + 1)]
        print(f"[sanity] right-action transition {idx}: per-step differs-from-input mask = {diffs_per_step[:8]}...")

    # Build model
    print(f"[build] model: n_hid={args.n_hid}, n_steps={args.n_steps}, "
          f"n_slots={args.n_slots}, input_skip={args.input_skip}, "
          f"vq={args.vq_codebook}")
    model = make_model(args.n_hid, args.n_steps, C_pad, args.n_slots,
                        src_cfg, eff_len + 1, args.input_skip,
                        use_vq=args.vq_codebook,
                        vq_codebook_size=args.vq_codebook_size,
                        vq_commitment_weight=args.vq_commitment_weight)

    rng = jax.random.PRNGKey(args.seed)
    rng, subkey = jax.random.split(rng)
    # Initialize with a dummy batch
    B = args.batch_size
    dummy_s = jnp.zeros((B, C_pad, H, W))
    dummy_a = jnp.zeros((B, N_ACTIONS))
    toks_padded_b = np.zeros((B, eff_len), dtype=np.int32)
    if toks_np.shape[0] > 0:
        toks_padded_b[:, :toks_np.shape[0]] = toks_np
    dummy_t = jnp.asarray(toks_padded_b)
    gmask = np.zeros((B, eff_len), dtype=bool)
    if toks_np.shape[0] > 0:
        gmask[:, :toks_np.shape[0]] = True
    dummy_m = jnp.asarray(gmask)
    params = model.init(subkey, dummy_s, dummy_a, dummy_t, dummy_m)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[build] {n_params:,} params")

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, s_b, a_b, n_b, inter_b, t_b, m_b):
        def loss_fn(params):
            out = model.apply(params, s_b, a_b, t_b, m_b,
                              return_vq_aux=args.vq_codebook)
            # adaptive_halt=True returns (logits, win_logit, sprite_logits, halt_aux)
            # halt_aux = (logits_per_step, win_per_step, halt_logits_per_step)
            final_logits = out[0]                 # (B, C, H, W)
            halt_aux = out[-1]
            if args.vq_codebook:
                vq_cb_loss, vq_commit_loss, vq_indices = out[-2]
                vq_util = jnp.count_nonzero(jnp.bincount(
                    vq_indices.reshape(-1),
                    length=args.vq_codebook_size,
                ))
            else:
                vq_cb_loss = jnp.asarray(0.0)
                vq_commit_loss = jnp.asarray(0.0)
                vq_util = jnp.asarray(0)
            per_step_logits = halt_aux[0]          # (T, B, C, H, W)
            T = per_step_logits.shape[0]
            # Per-step targets: inter_b is (B, n_steps+1, C, H, W); we want
            # the targets for steps 1..n_steps inclusive (the model has
            # n_steps NCA steps producing T outputs at steps 1..T).
            targets = inter_b[:, 1:T+1].transpose(1, 0, 2, 3, 4)  # (T, B, C, H, W)

            # Per-step BCE with change-loss-weighting against engine states.
            bce = optax.sigmoid_binary_cross_entropy(per_step_logits, targets)
            # Weight: per-step changed cells (between consecutive engine states)
            # get up-weighted. Use change wrt the original input as a proxy
            # consistent with train.py's weighting.
            states_b_T = jnp.broadcast_to(inter_b[:, 0:1].transpose(1,0,2,3,4),
                                           per_step_logits.shape)  # (T,B,C,H,W) = state_0
            changed = (targets != states_b_T).astype(bce.dtype)
            weight = 1.0 + args.change_loss_weight * changed
            per_step_loss = (bce * weight).sum(axis=(1,2,3,4)) / (weight.sum(axis=(1,2,3,4)) + 1e-8)
            # mean over T
            per_step_loss_mean = per_step_loss.mean()

            # Final-step loss against ground-truth next_state (target at step n_steps)
            final_target = inter_b[:, args.n_steps]                # (B, C, H, W)
            final_bce = optax.sigmoid_binary_cross_entropy(final_logits, final_target)
            final_changed = (final_target != s_b).astype(final_bce.dtype)
            final_weight = 1.0 + args.change_loss_weight * final_changed
            final_loss = (final_bce * final_weight).sum() / (final_weight.sum() + 1e-8)

            if args.final_only:
                total = final_loss
            else:
                total = final_loss + args.per_step_loss_weight * per_step_loss_mean
            if args.vq_codebook:
                total = total + args.vq_loss_weight * (
                    vq_cb_loss + args.vq_commitment_weight * vq_commit_loss)
            # Diagnostics
            preds_final = (jax.nn.sigmoid(final_logits) > 0.5).astype(jnp.float32)
            chg = (s_b != final_target)
            n_chg = chg.sum()
            chg_correct = ((preds_final == final_target) & chg).sum()
            change_acc = jnp.where(n_chg > 0, chg_correct / n_chg, 1.0)
            return total, (final_loss, per_step_loss_mean, change_acc,
                           vq_cb_loss, vq_commit_loss, vq_util)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # Build per-batch token + mask once (constant for varislide single game)
    toks_b = jnp.asarray(toks_padded_b)
    mask_b = jnp.asarray(gmask)

    print(f"[train] starting; final_only={args.final_only}, "
          f"per_step_weight={args.per_step_loss_weight}")
    rng_np = np.random.default_rng(args.seed)
    N = len(states_padded)
    best_loss = float("inf")
    t_start = time.time()
    for step in range(1, args.n_updates + 1):
        idx = rng_np.choice(N, size=args.batch_size, replace=False)
        s_b = jnp.asarray(states_padded[idx])
        a_b = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[actions[idx]])
        n_b = jnp.asarray(next_padded[idx])
        inter_b = jnp.asarray(inter[idx])
        params, opt_state, loss, aux = train_step(params, opt_state, s_b, a_b, n_b, inter_b, toks_b, mask_b)
        loss_v = float(loss)
        if loss_v < best_loss:
            best_loss = loss_v
            with open(os.path.join(args.save_dir, "params_best.pkl"), "wb") as f:
                pickle.dump(params, f)
        if step % args.log_interval == 0 or step == 1:
            final_loss, per_step_loss, chg_acc, vq_cb, vq_commit, vq_util = [
                float(x) for x in aux
            ]
            elapsed = time.time() - t_start
            vq_bit = (f"  vq_cb={vq_cb:.2e}  vq_commit={vq_commit:.2e}"
                      f"  vq_util={vq_util:.0f}"
                      if args.vq_codebook else "")
            print(f"  step {step:>5d}/{args.n_updates}  loss={loss_v:.4e}  "
                  f"final={final_loss:.4e}  per_step={per_step_loss:.4e}  "
                  f"change_acc={chg_acc:.4f}{vq_bit}  ({elapsed:.0f}s)")
    with open(os.path.join(args.save_dir, "params.pkl"), "wb") as f:
        pickle.dump(params, f)
    print(f"[done] best_loss={best_loss:.4e}; saved to {args.save_dir}")


if __name__ == "__main__":
    main()
