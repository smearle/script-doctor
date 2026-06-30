"""Information-gain adapter -- the optional q1 head from the Notion note
("Neural world models + Active learning").

The base world model gives q0(o | s, a) = p(o | s, a) (here: NCAWorldModel).
This adapter gives q1(o' | s, a, o) = p(o' | s, a, o): predict a SECOND
observation o', conditioned on the (state, action) AND an already-sampled first
observation o. Per the haoo' objective, the Bayes-optimal q1 is the counterfactual
posterior predictive  p(o'|s,a,o) = sum_theta p(theta|s,a,o) p(o'|s,a,theta).

It is trained SEPARATELY, after the base model, on the same transitions: o and o'
are two samples of the next state; in a DETERMINISTIC environment o == o' == the
true next state, so the adapter's job is simply to copy o (conditioned on s, a) --
it should learn that perfectly and fast. (Both Mario worlds here are deterministic.)

Why it matters: it lets us estimate information gain without latents,
    IG(s, a, o) = log q1(o | s, a, o) - log q0(o | s, a),
which is ~0 on confidently-predicted transitions and large on the genuinely
ambiguous ones (where q0 ~ 0.5 but, having "observed" o, q1 is confident).

History / recurrence (h) is deferred: this is the markov (s, a, o) version.

    python -m nca_wm.adapter --games mario,mario_breakable \
        --n_updates 3000 --save_dir nca_wm/logs/adapter_mario
"""
from __future__ import annotations

import argparse
import glob
import pickle
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from nca_wm.models import N_ACTIONS
from nca_wm.state_ops import _unpack_states


class AdapterHead(nn.Module):
    """q1: (state, action, o) -> o' logits. A small NCA conditioned on o.

    Mirrors NCAWorldModel's masked conv/residual body, but the input also
    contains o, so copying o (the deterministic-case optimum) is easy while the
    body retains capacity for the multimodal posterior-predictive in general.
    """
    n_hid: int = 128
    n_steps: int = 4
    n_out: int = 1          # set to C at init
    input_skip: bool = True

    @nn.compact
    def __call__(self, state, action_onehot, o):
        B, C, H, W = state.shape
        x = state.transpose(0, 2, 3, 1)                     # (B,H,W,C)
        oo = o.transpose(0, 2, 3, 1)                        # (B,H,W,C)
        act = jnp.broadcast_to(action_onehot[:, None, None, :], (B, H, W, N_ACTIONS))
        inp = jnp.concatenate([x, act, oo], axis=-1)        # condition on s, a, AND o
        h_inp = nn.Dense(self.n_hid, name="embed")(inp)
        h = h_inp
        mask = (x.sum(-1, keepdims=True) > 0).astype(jnp.float32)
        h = h * mask
        for i in range(self.n_steps):
            conv_in = jnp.concatenate([h, h_inp], -1) if self.input_skip else h
            h_conv = nn.Conv(self.n_hid, (3, 3), padding="SAME", name=f"conv_{i}")(conv_in)
            h = h + nn.Dense(self.n_hid, name=f"out_{i}")(jax.nn.gelu(h_conv))
            h = h * mask
        logits = nn.Dense(self.n_out, name="readout")(h)    # (B,H,W,C)
        return logits.transpose(0, 3, 1, 2)                 # (B,C,H,W) o' logits


def load_games(games, cap):
    out = []
    for g in games:
        f = glob.glob(f"rollout_data/{g}/level_0/bfs_transitions_v5_*_capall.npz")[0]
        d = np.load(f, allow_pickle=True)
        W = int(d["W"])
        Sp, Nsp, A = d["states"], d["next_states"], np.asarray(d["actions"], np.int64)
        if cap and len(Sp) > cap:
            idx = np.sort(np.random.RandomState(0).choice(len(Sp), cap, replace=False))
            Sp, Nsp, A = Sp[idx], Nsp[idx], A[idx]
        out.append((Sp, Nsp, A, W))
        print(f"  {g}: {len(Sp):,} transitions", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="mario,mario_breakable")
    ap.add_argument("--cap_per_game", type=int, default=0)        # 0 = all
    ap.add_argument("--n_hid", type=int, default=128)
    ap.add_argument("--n_steps", type=int, default=4)
    ap.add_argument("--n_updates", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_dir", default="nca_wm/logs/adapter_mario")
    args = ap.parse_args()

    games = args.games.split(",")
    data = load_games(games, args.cap_per_game or None)
    s0 = _unpack_states(data[0][0][:1], data[0][3])
    C, H, W = s0.shape[1], s0.shape[2], s0.shape[3]
    print(f"C/H/W = {C}/{H}/{W}", flush=True)

    model = AdapterHead(n_hid=args.n_hid, n_steps=args.n_steps, n_out=C)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, C, H, W)), jnp.zeros((1, N_ACTIONS)),
                        jnp.zeros((1, C, H, W)))
    nparam = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"adapter params: {nparam:,}", flush=True)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    def loss_fn(p, s, a, o, tgt):
        logits = model.apply(p, s, a, o)
        bce = optax.sigmoid_binary_cross_entropy(logits, tgt)
        m = (s.sum(1, keepdims=True) > 0).astype(jnp.float32)
        return (bce * m).sum() / jnp.maximum(m.sum() * C, 1.0)

    @jax.jit
    def update(p, os_, s, a, o, tgt):
        loss, g = jax.value_and_grad(loss_fn)(p, s, a, o, tgt)
        upd, os_ = opt.update(g, os_)
        return optax.apply_updates(p, upd), os_, loss

    rng = np.random.default_rng(0)

    def batch():
        Sp, Nsp, A, Wg = data[rng.integers(len(data))]
        idx = rng.integers(len(Sp), size=args.batch_size)
        s = _unpack_states(Sp[idx], Wg).astype(np.float32)
        nx = _unpack_states(Nsp[idx], Wg).astype(np.float32)
        a = np.eye(N_ACTIONS, dtype=np.float32)[A[idx]]
        return s, a, nx, nx                     # o = o' = next (deterministic)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for u in range(args.n_updates + 1):
        s, a, o, tgt = batch()
        params, opt_state, loss = update(params, opt_state, jnp.asarray(s),
                                         jnp.asarray(a), jnp.asarray(o), jnp.asarray(tgt))
        if u % 100 == 0:
            logits = np.asarray(model.apply(params, jnp.asarray(s), jnp.asarray(a), jnp.asarray(o)))
            m = (s.sum(1, keepdims=True) > 0)
            acc = float(((logits > 0) == (tgt > 0.5))[np.broadcast_to(m, logits.shape)].mean())
            print(f"step {u:5d}/{args.n_updates}  loss={float(loss):.6e}  copy_acc={acc:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    pickle.dump({"params": jax.device_get(params),
                 "cfg": {"n_hid": args.n_hid, "n_steps": args.n_steps, "n_out": C}},
                open(f"{args.save_dir}/adapter_params.pkl", "wb"))
    print(f"saved adapter -> {args.save_dir}/adapter_params.pkl", flush=True)


if __name__ == "__main__":
    main()
