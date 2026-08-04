"""EventWM: drop-in world model for the active-learning tree growth.

Implements the exact numpy-facing interface of ``tree_growth.WM`` (bits /
eig / train / tf_err, bucket-padded) on top of the event-tokenized
transformer (``event_wm.model``) with ``t_len=1``: PuzzleScript grids are
full state and the target games are deterministic, so no history is needed —
the comparison isolates the *joint AR likelihood head* against the NCA's
factorized per-cell Bernoulli head.

ONE CURRENCY, upgraded: ``bits(S, A, T)`` is the exactly normalized joint
-log2 P(T | S, A) (chain rule over the canonical event sequence), not a sum
of independent per-cell BCEs. ``eig`` is the predictive entropy of the same
joint, estimated by Monte-Carlo: draw ``k`` event-sequences with the
training-time legality mask and average their -log2 P (an unbiased entropy
estimator; deterministic learned dynamics concentrate it fast).

``cell_mask`` handling: add-events into non-real (padding) cells are removed
from the legality mask, so no probability mass leaks off the board.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import EWMConfig, EventWorldModel, masked_logits, NEG
from .tokenizer import Geom, encode_step

LN2 = float(np.log(2.0))


def _bucket(n):
    b = 8
    while b < n:
        b *= 2
    return b


class EventWM:
    """Same public surface as ``tree_growth.WM``; single-step episodes."""

    def __init__(self, n_obj, hp, wp, seed, lr, d_enc=192, enc_layers=2,
                 enc_heads=6, d_dec=160, dec_layers=3, dec_heads=5,
                 conv_channels=(32, 32), max_step_tokens=48, mc_samples=4,
                 cell_mask=None):
        cfg = EWMConfig(c_chan=n_obj, h=hp, w=wp, t_len=1,
                        max_len=max_step_tokens, d_enc=d_enc,
                        enc_layers=enc_layers, enc_heads=enc_heads,
                        d_dec=d_dec, dec_layers=dec_layers,
                        dec_heads=dec_heads, conv_channels=conv_channels,
                        max_step_tokens=max_step_tokens, lr=lr)
        self.cfg = cfg
        self.geom = Geom(n_obj, hp, wp)
        self.l_dec = max_step_tokens
        self.mc_samples = mc_samples
        self.model = EventWorldModel(cfg)
        # legality add-mask restricted to real cells (per channel)
        cm = np.ones((hp, wp), np.float32) if cell_mask is None else \
            np.asarray(cell_mask, np.float32)
        self._real_flat = jnp.asarray(
            np.tile(cm.reshape(-1) > 0, n_obj))            # (n_cell,)

        l = self.l_dec
        self.params = self.model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((1, 1, n_obj, hp, wp), jnp.float32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, l), jnp.int32),
            jnp.zeros((1, l), jnp.int32),
            jnp.zeros((1, l), jnp.int32),
            jnp.ones((1, 1), bool))
        self.tx = optax.adam(lr)
        self.opt = self.tx.init(self.params)

        n = self.geom.n_cell
        eof, bos, pad = self.geom.eof, self.geom.bos, self.geom.pad
        real = self._real_flat

        def _legality(S_flat, dec_in, real_b):
            """[B, L, vocab] legality; S_flat [B, n] occupancy, real_b [B, n]."""
            b, l_ = dec_in.shape
            cur = S_flat[:, None, :].repeat(l_, axis=1) > 0     # [B,L,n]
            frame_ok = jnp.concatenate(
                [cur, (~cur) & real_b[:, None, :]], axis=-1)
            prev_rank = jnp.where(dec_in == bos, -1, dec_in)
            ids = jnp.arange(2 * n)
            order_ok = ids[None, None, :] > prev_rank[:, :, None]
            ev_ok = frame_ok & order_ok
            one = jnp.ones((b, l_, 1), bool)
            no = jnp.zeros((b, l_, 1), bool)
            return jnp.concatenate([ev_ok, one, no, no], axis=-1)

        def _dec_logits(params, S, A, dec_in, real_b):
            """Teacher-forced logits [B, L, vocab] for single-step episodes."""
            b, l_ = dec_in.shape
            enc = self.model.apply(params, S[:, None], A[:, None].astype(
                jnp.int32), jnp.ones((b, 1), bool),
                method=EventWorldModel.encode)
            step = jnp.where(dec_in != pad, 0, 255)
            pos = jnp.arange(l_)[None].repeat(b, 0)
            lg = self.model.apply(params, enc, dec_in, step, pos,
                                  jnp.ones((b, 1), bool),
                                  method=EventWorldModel.decode)
            return masked_logits(lg, _legality(
                S.reshape(b, -1), dec_in, real_b))

        def bits_fn(params, S, A, dec_in, dec_tgt, real_b):
            logp = jax.nn.log_softmax(
                _dec_logits(params, S, A, dec_in, real_b), -1)
            valid = dec_tgt != pad
            tgt = jnp.where(valid, dec_tgt, 0)
            nll = -jnp.take_along_axis(logp, tgt[..., None], -1)[..., 0]
            return (nll * valid).sum(-1) / LN2                 # bits/example

        def train_fn(params, opt, S, A, dec_in, dec_tgt, W, real_b):
            def loss_fn(prm):
                logp = jax.nn.log_softmax(
                    _dec_logits(prm, S, A, dec_in, real_b), -1)
                valid = dec_tgt != pad
                tgt = jnp.where(valid, dec_tgt, 0)
                nll = -jnp.take_along_axis(logp, tgt[..., None], -1)[..., 0]
                per_ex = (nll * valid).sum(-1)                 # nats
                loss = (per_ex * W).sum() / jnp.maximum(W.sum(), 1e-8)
                return loss, per_ex / LN2
            (loss, per_bits), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(params)
            upd, opt = self.tx.update(grads, opt, params)
            return optax.apply_updates(params, upd), opt, loss, per_bits

        def rollout_fn(params, S, A, rng, greedy, real_b):
            """AR decode one step per lane: returns (tokens [B,L], nll_bits
            [B], next_frame_flat [B, n]).

            Iterative full-decode per position (L small); frame state is
            updated per emitted token so the legality mask stays exact.
            """
            b = S.shape[0]
            enc = self.model.apply(params, S[:, None],
                                   A[:, None].astype(jnp.int32),
                                   jnp.ones((b, 1), bool),
                                   method=EventWorldModel.encode)
            frame = S.reshape(b, -1) > 0
            toks = jnp.full((b, self.l_dec), pad, jnp.int32)
            dec_in = toks.at[:, 0].set(bos)
            done = jnp.zeros(b, bool)
            nll = jnp.zeros(b, jnp.float32)
            prev = jnp.full(b, bos, jnp.int32)

            def body(i, carry):
                frame, dec_in, done, nll, prev, rng = carry
                step = jnp.where(dec_in != pad, 0, 255)
                pos = jnp.arange(self.l_dec)[None].repeat(b, 0)
                lg = self.model.apply(params, enc, dec_in, step, pos,
                                      jnp.ones((b, 1), bool),
                                      method=EventWorldModel.decode)[:, i]
                cur = frame
                frame_ok = jnp.concatenate(
                    [cur, (~cur) & real_b], axis=-1)
                prev_rank = jnp.where(prev == bos, -1, prev)
                order_ok = jnp.arange(2 * n)[None, :] > prev_rank[:, None]
                ok = jnp.concatenate(
                    [frame_ok & order_ok,
                     jnp.ones((b, 1), bool),
                     jnp.zeros((b, 2), bool)], axis=-1)
                lg = jnp.where(ok, lg, NEG)
                logp = jax.nn.log_softmax(lg, -1)
                rng, sub = jax.random.split(rng)
                tok = jnp.where(
                    greedy, jnp.argmax(logp, -1),
                    jax.random.categorical(sub, logp))
                tok_lp = jnp.take_along_axis(
                    logp, tok[:, None], -1)[:, 0]
                nll = nll - jnp.where(done, 0.0, tok_lp) / LN2
                is_rm = tok < n
                is_add = (tok >= n) & (tok < 2 * n)
                cell = jnp.where(is_rm, tok, tok - n)
                onehot = jax.nn.one_hot(cell, n, dtype=bool)
                frame = jnp.where(
                    (is_rm & ~done)[:, None] & onehot, False, frame)
                frame = jnp.where(
                    (is_add & ~done)[:, None] & onehot, True, frame)
                emit = jnp.where(done, pad, tok)
                dec_in = dec_in.at[:, i + 1].set(
                    jnp.where(done | (tok == eof), pad, tok))
                done = done | (tok == eof)
                prev = jnp.where(done, bos, jnp.where(
                    (is_rm | is_add), tok, prev))
                del emit
                return frame, dec_in, done, nll, prev, rng

            frame, dec_in, done, nll, prev, rng = jax.lax.fori_loop(
                0, self.l_dec - 1, body,
                (frame, dec_in, done, nll, prev, rng))
            return dec_in, nll, frame

        self._bits = jax.jit(bits_fn)
        self._train = jax.jit(train_fn)
        self._rollout = jax.jit(rollout_fn, static_argnames=("greedy",))
        self._rng = jax.random.PRNGKey(seed + 1)

    # -- tokenization (host side) ------------------------------------------

    def _tokenize(self, S, T):
        b = len(S)
        dec_in = np.full((b, self.l_dec), self.geom.pad, np.int32)
        dec_tgt = np.full((b, self.l_dec), self.geom.pad, np.int32)
        for i in range(b):
            ev = encode_step(self.geom, S[i], T[i])
            k = len(ev) - 1
            if k + 1 > self.l_dec:      # delta longer than the decode cap
                ev = ev[:self.l_dec - 1] + [self.geom.eof]
                k = self.l_dec - 2
            dec_in[i, 0] = self.geom.bos
            dec_in[i, 1:1 + k] = ev[:k]
            dec_tgt[i, :k + 1] = ev
        return dec_in, dec_tgt

    def _pad(self, arrs, n):
        b = _bucket(n)
        out = []
        for a in arrs:
            pad = np.zeros((b - n,) + a.shape[1:], a.dtype)
            out.append(np.concatenate([a, pad], axis=0))
        return out, b

    # -- public surface (mirrors tree_growth.WM) ---------------------------

    def _real(self, M, n):
        if M is None:
            r = np.tile(np.asarray(self._real_flat)[None], (n, 1))
        else:
            r = np.asarray(M).reshape(n, -1) > 0
        return r

    def bits(self, S, A, T, M=None):
        n = len(S)
        di, dt = self._tokenize(S, T)
        R = self._real(M, n)
        (S, A, di, dt, R), _ = self._pad(
            [np.asarray(S, np.float32), np.asarray(A, np.int32),
             di, dt, R], n)
        return np.asarray(self._bits(self.params, S, A, di, dt, R))[:n]

    def eig(self, S, A, M=None):
        """MC predictive entropy in bits: mean over k sampled sequences of
        -log2 P(sequence)."""
        n = len(S)
        k = self.mc_samples
        Sr = np.repeat(np.asarray(S, np.float32), k, axis=0)
        Ar = np.repeat(np.asarray(A, np.int32), k, axis=0)
        Rr = np.repeat(self._real(M, n), k, axis=0)
        (Sr, Ar, Rr), _ = self._pad([Sr, Ar, Rr], n * k)
        self._rng, sub = jax.random.split(self._rng)
        _, nll, _ = self._rollout(self.params, Sr, Ar, sub, False, Rr)
        return np.asarray(nll)[:n * k].reshape(n, k).mean(axis=1)

    def train(self, S, A, T, M=None, W=None):
        n = len(S)
        di, dt = self._tokenize(S, T)
        W = np.ones(n, np.float32) if W is None else np.asarray(W, np.float32)
        R = self._real(M, n)
        (S, A, di, dt, W, R), _ = self._pad(
            [np.asarray(S, np.float32), np.asarray(A, np.int32),
             di, dt, W, R], n)
        self.params, self.opt, loss, per_bits = self._train(
            self.params, self.opt, S, A, di, dt, W, R)
        return float(loss), np.asarray(per_bits)[:n]

    def tf_err(self, S, A, T, M=None, chunk=256):
        """Greedy-decoded next frame vs truth, per-cell error rate."""
        wrong = tot = 0.0
        real = np.asarray(self._real_flat)
        for i in range(0, len(S), chunk):
            sl = slice(i, min(i + chunk, len(S)))
            n = sl.stop - sl.start
            Rc0 = self._real(M[sl] if M is not None else None, n)
            (Sc, Ac, Rc), _ = self._pad(
                [np.asarray(S[sl], np.float32),
                 np.asarray(A[sl], np.int32), Rc0], n)
            self._rng, sub = jax.random.split(self._rng)
            _, _, frame = self._rollout(self.params, Sc, Ac, sub, True, Rc)
            pred = np.asarray(frame)[:n]
            tgt = np.asarray(T[sl]).reshape(n, -1) > 0.5
            wrong += float(((pred != tgt) * Rc0).sum())
            tot += float(Rc0.sum())
        return wrong / max(tot, 1.0)
