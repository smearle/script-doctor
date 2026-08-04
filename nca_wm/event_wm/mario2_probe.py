"""The mario2 ambiguity probe MARIO2_ARCH_COMPARISON.md called for.

The two worlds (`mario` vs `mario_breakable`) differ in exactly ~12
ambiguous ``(state, action)`` keys (all UP = the platform-break jumps).
Global metrics are blind to them. This probe scores, for every ambiguous
key and natural sampled histories from each world:

  * P(break outcome | history) and P(stay outcome | history) — EXACT under
    the event WM (teacher-forced product over each candidate delta's
    canonical token sequence; no sampling error);
  * grouped by history evidence: a window containing a prior UP-at-platform
    step that survived (only possible in `mario`) identifies the world, so
    P(stay) should -> 1 there; windows with no such evidence are genuinely
    ambiguous and a calibrated model should sit near the training-data
    branch frequency (reported as reference).

Usage::

    python -m nca_wm.event_wm.mario2_probe --ckpt nca_wm/logs/mario2_event_wm/ckpt_best \
        [--per-key 8] [--out probe.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from nca_wm.recurrent_data import (
    GameData,
    build_trajectory_batch,
    load_dataset_from_caches,
)

from .model import EventWorldModel, dec_positions, legality_mask, masked_logits
from .tokenizer import Geom, encode_step
from .train import load_checkpoint

LN2 = float(np.log(2.0))


def _keys(gd: GameData):
    """(state_bytes, action) -> {next_bytes: [row, ...]} over all rows."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in range(gd.n):
        k = (gd.states_packed[r].tobytes(), int(gd.actions[r]))
        out[k][gd.next_packed[r].tobytes()].append(r)
    return out


def find_ambiguous(g0: GameData, g1: GameData):
    """Shared (s,a) keys whose next-state differs across the two games."""
    k0, k1 = _keys(g0), _keys(g1)
    amb = []
    for k in set(k0) & set(k1):
        n0, n1 = set(k0[k]), set(k1[k])
        if n0 != n1:
            amb.append((k, k0[k], k1[k]))
    return amb


def _score_step(net, params, cfg, geom, states, actions, valid, delta_tokens):
    """Exact -log2 P(delta at final step | history) for one trajectory.

    ``delta_tokens`` is the canonical [e1..ek, EOF] sequence of the candidate
    outcome. Returns bits (float).
    """
    L = states.shape[0]
    l_dec = len(delta_tokens) + 1
    dec_in = np.full((1, l_dec), geom.pad, np.int32)
    dec_tgt = np.full((1, l_dec), geom.pad, np.int32)
    dec_in[0, 0] = geom.bos
    dec_in[0, 1:len(delta_tokens)] = delta_tokens[:-1]
    dec_tgt[0, :len(delta_tokens)] = delta_tokens
    dec_step = np.where(dec_tgt != geom.pad, L - 1, 255).astype(np.int32)
    dec_pos = dec_positions(dec_step)
    obs = states[None].astype(np.float32)
    act = actions[None].astype(np.int32)
    sv = valid[None]
    logits = net.apply(params, jnp.asarray(obs), jnp.asarray(act),
                       jnp.asarray(dec_in), jnp.asarray(dec_step),
                       jnp.asarray(dec_pos), jnp.asarray(sv))
    mask = legality_mask(cfg, jnp.asarray(obs), jnp.asarray(dec_in),
                         jnp.asarray(dec_step))
    logp = jax.nn.log_softmax(masked_logits(logits, mask), -1)
    tgt = jnp.asarray(dec_tgt)
    ok = tgt != geom.pad
    t_safe = jnp.where(ok, tgt, 0)
    nll = -jnp.take_along_axis(logp, t_safe[..., None], -1)[..., 0]
    return float((nll * ok).sum()) / LN2


def _nca_score_fn(ckpt_dir: str):
    """Scorer for a train_recurrent RecurrentNCA checkpoint: factorized
    joint P(outcome) = product of per-cell Bernoullis at the final step."""
    import pickle
    import os

    from nca_wm.models import RecurrentNCAWorldModel, N_ACTIONS

    with open(os.path.join(ckpt_dir, "config.json")) as f:
        c = json.load(f)
    path = os.path.join(ckpt_dir, "params_best.pkl")
    if not os.path.exists(path):
        path = os.path.join(ckpt_dir, "params.pkl")
    with open(path, "rb") as f:
        params = pickle.load(f)
    model = RecurrentNCAWorldModel(
        n_hid=c["n_hid"], n_steps=c["n_steps"], n_out=c["n_out"],
        axis_pool=c["axis_pool"], axis_cummax=c["axis_cummax"],
        global_pool=c["global_pool"], input_skip=c["input_skip"])
    dims = (c["max_C"], c["max_H"], c["max_W"])

    @jax.jit
    def _final_logits(prm, states, acts_onehot):
        logits, _, _ = model.apply(prm, states, acts_onehot)
        return logits[:, -1]

    def score(states, actions, valid, outcome, geom):
        a_onehot = jax.nn.one_hot(actions, N_ACTIONS, dtype=jnp.float32)
        lg = _final_logits(params, jnp.asarray(states[None], jnp.float32),
                           a_onehot[None])[0]
        p = jax.nn.sigmoid(lg)
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        t = jnp.asarray(outcome)
        ll = t * jnp.log(p) + (1 - t) * jnp.log(1 - p)
        return -float(ll.sum()) / LN2

    return score, dims


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", choices=["event", "nca"], default="event")
    p.add_argument("--games", nargs=2, default=["mario", "mario_breakable"])
    p.add_argument("--max-transitions", type=int, default=200_000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--per-key", type=int, default=8,
                   help="sampled histories per (key, game)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)

    if a.arch == "event":
        cfg, net, params, step = load_checkpoint(a.ckpt)
        geom = Geom(cfg.c_chan, cfg.h, cfg.w)

        def score_fn(states, actions, valid, outcome, geom_):
            return _score_step(net, params, cfg, geom_, states, actions,
                               valid, encode_step(geom_, states[-1], outcome))
        t_len = cfg.t_len
        print(f"[probe] event ckpt step {step} "
              f"geom=({cfg.c_chan},{cfg.h},{cfg.w})")
    else:
        score_fn, (mc, mh, mw) = _nca_score_fn(a.ckpt)

        class _C:  # minimal shim for the loop below
            c_chan, h, w, t_len, seed = mc, mh, mw, 9, a.seed
        cfg = _C()
        geom = Geom(mc, mh, mw)
        t_len = cfg.t_len
        print(f"[probe] nca ckpt geom=({mc},{mh},{mw}) L={t_len}")

    dataset, infos = load_dataset_from_caches(
        a.games, a.max_transitions, a.val_frac,
        ancestor_closed=True, max_grid_dim=32, seed=cfg.seed)
    g0, g1 = [GameData(g, dataset, i) for g, i in enumerate(infos)]
    amb = find_ambiguous(g0, g1)
    print(f"[probe] ambiguous (s,a) keys: {len(amb)} "
          f"(actions: {sorted(set(k[0][1] for k in amb))})")

    # empirical branch frequency over ambiguous rows (training reference)
    n_stay = sum(len(r) for _, k0rows, _ in amb for r in k0rows.values())
    n_break = sum(len(r) for _, _, k1rows in amb for r in k1rows.values())
    prior_break = n_break / max(n_stay + n_break, 1)
    print(f"[probe] empirical branch freq: stay {n_stay} rows, "
          f"break {n_break} rows -> P(break) = {prior_break:.3f}")

    rng = np.random.default_rng(a.seed)
    amb_keys = set(k for k, _, _ in amb)
    groups = defaultdict(list)          # (game, evidence) -> [dict]
    C, H, W = cfg.c_chan, cfg.h, cfg.w

    def _hist_key(gd: GameData, frame, action):
        """Pack a (padded) history frame back to the cache byte key."""
        crop = frame[:gd.n_objs, :gd.H, :gd.W].astype(np.uint8)
        return (np.packbits(crop, axis=-1).tobytes(), int(action))

    for key, rows0, rows1 in amb:
        # the two candidate outcomes for this key (one row from each game)
        outs = []
        for gd, rows_g in ((g0, rows0), (g1, rows1)):
            row = next(iter(rows_g.values()))[0]
            nxt = gd.unpack(gd.next_packed[np.asarray([row])])[0]
            full = np.zeros((C, H, W), np.float32)
            cc, hh, ww = nxt.shape
            full[:cc, :hh, :ww] = nxt
            outs.append(full)
        for gi, gd, rows_g in ((0, g0, rows0), (1, g1, rows1)):
            all_rows = [r for rl in rows_g.values() for r in rl]
            pick = rng.choice(all_rows, size=min(a.per_key, len(all_rows)),
                              replace=False)
            S, A, T, V = build_trajectory_batch(
                gd, np.asarray(pick), cfg.t_len - 1, rng, C, H, W)
            acts = A.argmax(-1)
            for b in range(len(pick)):
                # evidence: any prior in-window step whose (s,a) is itself an
                # ambiguous key — its survived outcome identifies the world
                ev = any(
                    bool(V[b, t]) and
                    _hist_key(gd, S[b, t], acts[b, t]) in amb_keys
                    for t in range(cfg.t_len - 1))
                bits = [score_fn(S[b], acts[b], V[b], out, geom)
                        for out in outs]
                p_stay = 2.0 ** -bits[0]
                p_break = 2.0 ** -bits[1]
                groups[(a.games[gi], "ev+" if ev else "ev-")].append({
                    "p_stay": p_stay, "p_break": p_break,
                    "bits_realized": bits[gi],
                    "leak": max(0.0, 1 - p_stay - p_break),
                })

    report = {"prior_break": prior_break, "n_keys": len(amb), "groups": {}}
    for (game, ev), lst in sorted(groups.items()):
        arr = {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}
        arr["n"] = len(lst)
        report["groups"][f"{game}/{ev}"] = arr
        print(f"[probe] {game:16s} ev={ev}: n={len(lst)} "
              f"P(stay)={arr['p_stay']:.3f} P(break)={arr['p_break']:.3f} "
              f"bits(realized)={arr['bits_realized']:.3f} "
              f"leak={arr['leak']:.4f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[probe] wrote {a.out}")


if __name__ == "__main__":
    main()
