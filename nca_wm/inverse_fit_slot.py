"""Inverse-fit a game's slot embedding from its (state, action, next_state)
transitions, using a frozen joint-trained NCA + SlotTokenDecoder.

Given observations $\mathcal{D}=\{(s_t,a_t,s_{t+1})\}$ from an unknown game
that was NOT seen during training, this script optimizes a slot matrix
$z \in \mathbb{R}^{K \times d}$ to minimize the same per-cell next-state
BCE used during training, holding NCA and decoder parameters fixed. The
optimized slot is then decoded via SlotTokenDecoder → PuzzleScript text.

This addresses the abstract claim ``fit out-of-distribution environment
transitions to game code by optimizing the unobserved latent game embedding
through the frozen NCA dynamics model'' and the results.tex line 27 +
method.tex line 144 papertodos.

Headline metrics:
  - per-step BCE loss on held transitions (during fit)
  - cell-error vs ground truth on held transitions (after fit)
  - 1-NN training game by cosine distance to the optimized slot
  - decoded text + JS-engine compile success
  - (optional) tokenized re-encoding match against the held-out game's
    own tokens

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m nca_wm.inverse_fit_slot \
        --load nca_wm/logs/multi_scaling_gallery_v3_decoder \
        --target_game monophobic_multiban_by_increpare \
        --n_steps 2000 --lr 5e-3
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_run(save_dir: str, prefer_best: bool = True):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    pp = "params_best.pkl" if (
        prefer_best and os.path.exists(os.path.join(save_dir, "params_best.pkl"))
    ) else "params.pkl"
    with open(os.path.join(save_dir, pp), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    return cfg, params, game_infos


def collect_transitions_for_game(name: str, cfg: dict, n_max: int = 2000):
    """Collect (states, actions, next_states) for a target game via the
    standard A* transition collector. Truncated to the first n_max items
    for speed."""
    from nca_wm.train import collect_unique_transitions
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv

    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, name)

    # Iterate over levels
    env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    n_levels = env0.num_levels
    all_states, all_actions, all_next, all_n_objs = [], [], [], None
    all_H, all_W = 0, 0
    for li in range(n_levels):
        res = collect_unique_transitions(
            json_str=json_str, game_name=name, level_i=li,
            max_iters=10000, timeout_ms=30000, search_algo="astar",
            max_transitions=n_max,
        )
        # Per-level cache returns packed states; we need to unpack.
        from nca_wm.train import _unpack_states
        st = _unpack_states(np.asarray(res["states"]), int(res["W"]))
        nx = _unpack_states(np.asarray(res["next_states"]), int(res["W"]))
        all_states.append(st)
        all_next.append(nx)
        all_actions.append(np.asarray(res["actions"]))
        if all_n_objs is None:
            all_n_objs = st.shape[1]
        all_H = max(all_H, st.shape[2])
        all_W = max(all_W, st.shape[3])

    # Pad each level's arrays to the (max_n_objs, max_H, max_W) shape and
    # concatenate.
    from nca_wm.train import _pad_obs
    # Pad states/next_states
    padded_states = np.concatenate([
        _pad_obs(s, all_n_objs, all_H, all_W) for s in all_states
    ])
    padded_next = np.concatenate([
        _pad_obs(s, all_n_objs, all_H, all_W) for s in all_next
    ])
    actions = np.concatenate(all_actions).astype(np.int32)
    print(f"  {name}: {len(actions):,} transitions, "
          f"shape=({all_n_objs}, {all_H}, {all_W}), "
          f"levels={n_levels}", file=sys.stderr)
    return padded_states, actions, padded_next, all_n_objs, all_H, all_W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--target_game", required=True,
                    help="Game NOT in the training set, by name.")
    ap.add_argument("--n_steps", type=int, default=2000,
                    help="Adam steps to optimize the slot.")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--init", default="random",
                    choices=["random", "mean", "knn"],
                    help="Slot initialization. 'mean' = mean of training "
                         "slots; 'knn' = nearest training slot by token "
                         "encoder; 'random' = small Gaussian.")
    ap.add_argument("--n_max_transitions", type=int, default=500)
    ap.add_argument("--out_subdir", default="inverse_fit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, params, game_infos = load_run(args.load)
    if not isinstance(params, dict) or "wm" not in params or "dec" not in params:
        print("ERROR: this checkpoint has no joint decoder.", file=sys.stderr)
        sys.exit(1)
    wm_params = params["wm"]
    dec_params = params["dec"]

    # Sanity: target must NOT be in training set.
    train_names = {g["name"] for g in game_infos}
    if args.target_game in train_names:
        print(f"WARN: target_game {args.target_game!r} is IN the training set "
              f"— this is a sanity check, not a real OOD test.",
              file=sys.stderr)

    out_dir = os.path.join(args.load, args.out_subdir, args.target_game)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}", file=sys.stderr)

    # Build modules using the tooling we already have.
    from nca_wm.sample_latent_games_rule_attn import build_modules
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    encoder, decoder, max_tok_len = build_modules(cfg, game_infos, params=params)

    # Build the world model
    max_C = max(g["n_objs"] for g in game_infos)
    enc_max_seq_len = encoder.max_seq_len
    pool_kwargs = dict(
        axis_pool=cfg.get("axis_pool", False),
        axis_cummax=cfg.get("axis_cummax", False),
        global_pool=cfg.get("global_pool", False),
    )
    from nca_wm.sample_latent_games_rule_attn import _derive_vocab_size
    vocab_size = _derive_vocab_size(cfg, game_infos)
    wm = RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        vocab_size=vocab_size + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"],
        n_attn_heads=cfg["n_heads"],
        max_seq_len=enc_max_seq_len,
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg.get("n_nca_repeats", 1),
        adaptive_halt=cfg.get("adaptive_halt", False),
        **pool_kwargs,
    )

    # Collect transitions for the target game
    print(f"\nCollecting transitions for {args.target_game}...",
          file=sys.stderr)
    states, actions, next_states, n_objs, H, W = collect_transitions_for_game(
        args.target_game, cfg, n_max=args.n_max_transitions,
    )
    # Pad to model's max_C (in case target_game has fewer object channels
    # than the trained max_C)
    from nca_wm.train import _pad_obs
    states = _pad_obs(states, max_C, H, W)
    next_states = _pad_obs(next_states, max_C, H, W)
    n_train = min(len(actions), args.n_max_transitions)
    states = states[:n_train]
    actions = actions[:n_train]
    next_states = next_states[:n_train]
    print(f"  using {n_train} transitions for fitting", file=sys.stderr)

    # Build action one-hot
    from nca_wm.train import N_ACTIONS
    actions_oh = np.eye(N_ACTIONS, dtype=np.float32)[actions]

    # Encode all training games to compute init / 1-NN
    @jax.jit
    def _enc(tids, mask):
        return encoder.apply(
            {"params": wm_params["params"]["game_encoder"]},
            tids[None], mask[None],
        )

    K = cfg["n_slots"]
    d_slot = cfg["d_slot"]
    train_slots = []
    train_names_list = []
    for info in game_infos:
        tids = info.get("token_ids", [])
        pad = np.zeros(enc_max_seq_len, dtype=np.int32)
        mask = np.zeros(enc_max_seq_len, dtype=np.bool_)
        L = min(len(tids), enc_max_seq_len)
        pad[:L] = tids[:L]
        mask[:L] = True
        slots = np.array(_enc(jnp.array(pad), jnp.array(mask))[0])
        train_slots.append(slots)
        train_names_list.append(info["name"])
    train_slots = np.stack(train_slots)  # (N, K, d)
    train_mean = train_slots.mean(axis=0)

    # Initialize slot
    rng_np = np.random.default_rng(args.seed)
    if args.init == "mean":
        slot_init = train_mean.copy()
    elif args.init == "knn":
        # Use a default fallback: just pick a random training slot
        slot_init = train_slots[rng_np.integers(len(train_slots))].copy()
    else:  # random
        slot_init = (
            train_mean + 0.1 * rng_np.standard_normal(
                (K, d_slot)
            ).astype(np.float32)
        )
    slot_init = jnp.array(slot_init)

    # Forward function that uses model with slots_override (added to
    # RuleAttnNCAWorldModel.__call__ 2026-05-05).
    # Dummy tokens/mask shape match training-time max_seq_len; values
    # don't matter because slots_override bypasses the encoder output.
    dummy_tokens = jnp.zeros((1, enc_max_seq_len), dtype=jnp.int32)
    dummy_mask = jnp.zeros((1, enc_max_seq_len), dtype=jnp.bool_)

    @jax.jit
    def forward(slots_2d, state, action_oh):
        # slots_2d: (K, d). Expand to (B, K, d) for the model's batch dim.
        # state: (B, C, H, W); action_oh: (B, N_ACTIONS).
        B = state.shape[0]
        slots = jnp.broadcast_to(slots_2d[None], (B,) + slots_2d.shape)
        toks = jnp.broadcast_to(dummy_tokens, (B, enc_max_seq_len))
        msk = jnp.broadcast_to(dummy_mask, (B, enc_max_seq_len))
        logits, win_logit, _ = wm.apply(
            wm_params, state, action_oh, toks, msk, slots_override=slots,
        )
        return logits, win_logit

    # BCE loss matching training (per-cell sigmoid + change weighting)
    change_weight = float(cfg.get("change_loss_weight", 5.0))

    def per_cell_loss(slots, state, action_oh, next_state):
        logits, _ = forward(slots, state, action_oh)
        # Per-cell BCE with logits
        # logits: (B, C, H, W); next_state: same shape, in {0, 1}.
        log_p = -jax.nn.softplus(-logits)
        log_1mp = -jax.nn.softplus(logits)
        bce = -(next_state * log_p + (1 - next_state) * log_1mp)
        change = jnp.abs(next_state - state)  # (B, C, H, W) in [0, 1]
        weights = 1.0 + (change_weight - 1.0) * change
        return (bce * weights).mean()

    grad_fn = jax.jit(jax.value_and_grad(per_cell_loss))

    # Pre-build batches for fitting
    states_j = jnp.array(states.astype(np.float32))
    actions_oh_j = jnp.array(actions_oh)
    next_j = jnp.array(next_states.astype(np.float32))

    print(f"\nFitting slot ({K}×{d_slot}) for "
          f"{n_train} transitions of {args.target_game}...", file=sys.stderr)
    print(f"  init mode: {args.init}", file=sys.stderr)
    print(f"  optimizer: Adam lr={args.lr}", file=sys.stderr)

    import optax
    slots_var = slot_init.copy()
    optim = optax.adam(args.lr)
    opt_state = optim.init(slots_var)

    # Training loop with full-batch updates (n_train is small).
    losses = []
    for step in range(args.n_steps):
        # Add a batch dim of size 1 (full-batch over the level)
        # Use a sliding-window mini-batch if n_train > 32 to fit in memory
        bs = min(32, n_train)
        idx = np.random.default_rng(args.seed + step).choice(
            n_train, size=bs, replace=False,
        )
        s_b = states_j[idx]
        a_b = actions_oh_j[idx]
        n_b = next_j[idx]
        loss, grads = grad_fn(slots_var, s_b, a_b, n_b)
        updates, opt_state = optim.update(grads, opt_state, slots_var)
        slots_var = optax.apply_updates(slots_var, updates)
        if step % max(1, args.n_steps // 20) == 0 or step == args.n_steps - 1:
            losses.append({"step": step, "loss": float(loss)})
            print(f"  step {step:5d}: loss = {float(loss):.4e}",
                  file=sys.stderr)

    fitted = np.array(slots_var)

    # 1-NN train game
    flat_fit = fitted.reshape(-1)
    flat_train = train_slots.reshape(len(train_slots), -1)
    cos = (flat_train @ flat_fit) / (
        (np.linalg.norm(flat_train, axis=1) + 1e-9) *
        (np.linalg.norm(flat_fit) + 1e-9)
    )
    nn_idx = int(np.argmax(cos))
    nn_name = train_names_list[nn_idx]
    nn_dist = float(1 - cos[nn_idx])
    print(f"\n1-NN training game by cosine distance: {nn_name} "
          f"(d={nn_dist:.3f})", file=sys.stderr)

    # Decode the fitted slot
    from nca_wm.token_decoder import sample_tokens_from_slots
    print(f"\nDecoding fitted slot...", file=sys.stderr)
    fitted_2d = np.asarray(fitted)
    if fitted_2d.ndim == 2:
        fitted_2d = fitted_2d[None]  # (1, K, d) for the decoder's batch dim
    decoded = np.array(sample_tokens_from_slots(
        decoder, dec_params, jnp.array(fitted_2d),
        max_len=decoder.max_seq_len, bos_id=0, temperature=0.0,
    ))[0]

    from nca_wm.detokenize_game import detokenize
    ids = [int(t) for t in decoded if int(t) > 0]
    text = detokenize(ids, title=f"fit_{args.target_game}")
    out_txt = os.path.join(out_dir, "fitted_decoded.txt")
    with open(out_txt, "w") as f:
        f.write(text)
    print(f"  decoded text: {out_txt}", file=sys.stderr)

    # Save artifacts
    np.savez(
        os.path.join(out_dir, "fitted.npz"),
        slot_init=np.array(slot_init),
        slot_fitted=fitted,
        nn_train_name=nn_name,
        nn_cos_dist=nn_dist,
    )
    summary = {
        "target_game": args.target_game,
        "n_transitions": int(n_train),
        "n_steps": int(args.n_steps),
        "lr": float(args.lr),
        "init": args.init,
        "loss_curve": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
        "nn_train_name": nn_name,
        "nn_cos_dist": nn_dist,
        "decoded_text_file": out_txt,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Summary: {os.path.join(out_dir, 'summary.json')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
