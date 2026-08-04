"""Online IG-seeking tree growth for a PuzzleScript world model.

Port of neural-world-model's ``mario_variants/train_wm_active.py`` loop onto
the C++ PuzzleScript engine, radically simplified by two facts about
PuzzleScript worlds: the grid IS the full state (nothing hidden), and the
target games are deterministic. So there is no recurrence, no hidden-state
walkers, no beliefs/q1 adapter — the world model is the plain feedforward
``NCAWorldModel`` and every recorded edge's bits can be re-measured EXACTLY
with one batched forward pass.

ONE CURRENCY: BITS. When the model assigns probability p to the next frame
that actually happens, -log2(p) is both the training loss on that transition
and the information gained by witnessing it.

THREE PERSISTENT OBJECTS (one game level):
  * engine   -- the live C++ engine, with backup_level()/restore_level()
                snapshots and full-state dedup on the multihot grid bytes.
  * graph    -- every recorded (state, action, next_state) edge with its most
                recently measured bits; states carry their first-discovery
                parent so any state (e.g. a WIN) yields a root path for free.
  * frontier -- a lazy best-first heap over unexpanded states. Scores are
                re-measured at pop time (fresh expected-IG = predictive
                entropy of the current model; sunk-cost-free "future" score,
                the champion recipe from the mario A/Bs).

THE LOOP (every ``--expand_every`` updates; also as warmup):
  1. SEARCH: pop the best frontier states, re-score them with fresh EIG,
     expand the top k in the real engine (all enabled actions), dedup
     children, record realized bits on the new edges.
  2. TRAIN:  sample edges with probability rising in (bits - lam + V[next]),
     one gradient step of masked per-cell BCE; the fresh per-edge loss is
     written back onto the sampled edges (training re-measures the map).
  3. VALUE:  V = bits-still-reachable, a few Bellman sweeps on the graph.

No hand-authored reward anywhere. Ground-truth metrics only:
  * mechanics-witnessed = distinct engine rule indices fired on recorded
    edges (the engine tracks rules fired natively);
  * wins = edges landing on is_winning() states; each win's root action path
    is extracted from first-discovery parents and verified by replay.

Heldout eval: teacher-forced one-step per-cell error on random-rollout
transitions from OTHER levels of the same game (the generalization axis on
which offline BFS/A* collection plateaued at ~23-28% for heroes).

    .venv/bin/python -m nca_wm.active_learning.tree_growth \
        --game heroes_of_sokoban --levels 0-4 --n_updates 3000 --out /tmp/tg_heroes
"""
from __future__ import annotations

import argparse
import heapq
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax
import jax.numpy as jnp
import optax

from nca_wm.models import NCAWorldModel, N_ACTIONS

ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT", "ACTION", "TICK"]
LN2 = float(np.log(2.0))
EPS = 1e-6


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------
def compile_game(name: str):
    """Locate <name>.txt, materialize it, compile → (json_str, n_levels)."""
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser

    roots = [
        _REPO / "script_doctor/puzzlescript_deprecated_hack/demo",
        _REPO / "custom_games",
    ]
    src = next((r / f"{name}.txt" for r in roots if (r / f"{name}.txt").exists()), None)
    if src is None:
        raise FileNotFoundError(f"{name}.txt not found under {[str(r) for r in roots]}")

    scratch = _REPO / "nca_wm" / "active_learning" / "_tree_growth_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(scratch)
    gc._materialize_game(name, src.read_text())
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, name)
    return json_str


def new_engine(json_str, level_i):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine()
    e.load_from_json(json_str)
    e.load_level(level_i)
    e.set_track_rules_fired(True)
    e.seed_rng("0")
    return e


def enabled_actions(json_str) -> list[int]:
    """Mirror of data_collection._enabled_actions (solver.cpp actionsForEngine)."""
    meta = json.loads(json_str).get("metadata", {})
    acts = [0, 1, 2, 3]
    if "noaction" not in meta:
        acts.append(4)
    if "realtime_interval" in meta:
        acts.append(5)
    return acts


def engine_step(eng, a, max_again=50):
    eng.clear_rules_fired()
    eng.process_input(a)
    n = 0
    while eng.is_againing() and n < max_again:
        eng.process_input(-1)
        n += 1


def read_obs(eng, n_obj, hp, wp):
    """(n_obj, hp, wp) float32 multihot, channels in id_dict order."""
    a = np.asarray(eng.get_objects_2d())              # (W, H, stride)
    cell = a[:, :, 0].T.astype(np.int64)              # (H, W) bitmask
    h, w = cell.shape
    g = np.zeros((n_obj, hp, wp), dtype=np.float32)
    hh, ww = min(h, hp), min(w, wp)
    for c in range(n_obj):
        g[c, :hh, :ww] = ((cell >> c) & 1).astype(np.float32)[:hh, :ww]
    return g


def pack_obs(obs):
    return np.packbits(obs.astype(np.uint8), axis=None).tobytes()


# ---------------------------------------------------------------------------
# The growing graph
# ---------------------------------------------------------------------------
class Graph:
    """States (deduped on grid bytes) + edges (with live bits) + frontier."""

    def __init__(self, eng, n_obj, hp, wp, acts):
        self.eng = eng
        self.n_obj, self.hp, self.wp = n_obj, hp, wp
        self.acts = acts
        self.key2id: dict[bytes, int] = {}
        self.obs: list[np.ndarray] = []          # per state (C,hp,wp) float32
        self.backup: list = []                   # engine backup objects
        self.depth: list[int] = []
        self.parent: list[tuple[int, int]] = []  # (parent_sid, action) at discovery
        self.expanded: set[int] = set()
        # edges (parallel lists)
        self.e_s: list[int] = []
        self.e_a: list[int] = []
        self.e_s2: list[int] = []
        self.e_bits: list[float] = []
        self.e_won: list[bool] = []
        # frontier: (-score, tiebreak, sid); lazily rescored at pop
        self.heap: list[tuple[float, int, int]] = []
        self._tie = 0
        self.env_steps = 0
        self.rules_witnessed: dict[int, int] = {}   # rule idx -> env_step first seen
        self.win_states: list[int] = []

        o = read_obs(eng, n_obj, hp, wp)
        self._add_state(o, eng.backup_level(), 0, (-1, -1))

    def _add_state(self, o, bak, depth, parent) -> tuple[int, bool]:
        k = pack_obs(o)
        sid = self.key2id.get(k)
        if sid is not None:
            return sid, False
        sid = len(self.obs)
        self.key2id[k] = sid
        self.obs.append(o)
        self.backup.append(bak)
        self.depth.append(depth)
        self.parent.append(parent)
        return sid, True

    def push(self, sid, score):
        self._tie += 1
        heapq.heappush(self.heap, (-score, self._tie, sid))

    def pop_candidates(self, m):
        out = []
        while self.heap and len(out) < m:
            _, _, sid = heapq.heappop(self.heap)
            if sid not in self.expanded:
                out.append(sid)
        return out

    def expand(self, sid) -> list[tuple[int, int, int, bool]]:
        """Expand all enabled actions from sid. Returns new (eid range) edges
        as (sid, a, s2id, is_new_state). Win states are recorded but NOT
        expandable (the level ends there in the real game; expanding past
        them floods the frontier with post-win near-duplicates)."""
        recs = []
        for a in self.acts:
            self.eng.restore_level(self.backup[sid])
            engine_step(self.eng, a)
            self.env_steps += 1
            o2 = read_obs(self.eng, self.n_obj, self.hp, self.wp)
            won = bool(self.eng.is_winning())
            for r in self.eng.get_rules_fired():
                self.rules_witnessed.setdefault(int(r), self.env_steps)
            s2, is_new = self._add_state(
                o2, self.eng.backup_level(), self.depth[sid] + 1, (sid, a))
            if won and s2 not in self.win_states:
                self.win_states.append(s2)
            self.e_s.append(sid)
            self.e_a.append(a)
            self.e_s2.append(s2)
            self.e_bits.append(0.0)   # measured in batch by caller
            self.e_won.append(won)
            recs.append((sid, a, s2, is_new, won))
        self.expanded.add(sid)
        return recs

    def root_path(self, sid) -> list[int]:
        acts = []
        while self.parent[sid][0] >= 0:
            p, a = self.parent[sid]
            acts.append(a)
            sid = p
        return acts[::-1]

    @property
    def n_states(self):
        return len(self.obs)

    @property
    def n_edges(self):
        return len(self.e_s)


# ---------------------------------------------------------------------------
# Model wrappers (bucketed batching to bound XLA recompiles)
# ---------------------------------------------------------------------------
def _bucket(n):
    b = 8
    while b < n:
        b *= 2
    return b


class WM:
    def __init__(self, n_obj, hp, wp, n_hid, n_steps, seed, lr):
        self.model = NCAWorldModel(n_hid=n_hid, n_steps=n_steps, n_out=n_obj,
                                   input_skip=True)
        dummy_s = jnp.zeros((1, n_obj, hp, wp), jnp.float32)
        dummy_a = jnp.zeros((1, N_ACTIONS), jnp.float32)
        self.params = self.model.init(jax.random.PRNGKey(seed), dummy_s, dummy_a)
        self.tx = optax.adam(lr)
        self.opt = self.tx.init(self.params)

        def logits_fn(params, S, A):
            return self.model.apply(params, S, jax.nn.one_hot(A, N_ACTIONS))[0]

        def bits_fn(params, S, A, T, M):
            """Per-example realized bits: masked BCE(sigmoid(logits), T) in bits."""
            lg = logits_fn(params, S, A)
            p = jnp.clip(jax.nn.sigmoid(lg), EPS, 1 - EPS)
            nll = -(T * jnp.log(p) + (1 - T) * jnp.log(1 - p))     # nats/cell
            return (nll * M).sum(axis=(1, 2, 3)) / LN2             # bits/transition

        def eig_fn(params, S, A, M):
            """Per-example expected bits = Bernoulli predictive entropy."""
            lg = logits_fn(params, S, A)
            p = jnp.clip(jax.nn.sigmoid(lg), EPS, 1 - EPS)
            h = -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))
            return (h * M).sum(axis=(1, 2, 3)) / LN2

        def train_step(params, opt, S, A, T, M, W):
            def loss_fn(prm):
                lg = logits_fn(prm, S, A)
                p = jnp.clip(jax.nn.sigmoid(lg), EPS, 1 - EPS)
                nll = -(T * jnp.log(p) + (1 - T) * jnp.log(1 - p))
                per_ex = (nll * M).sum(axis=(1, 2, 3))             # nats
                loss = (per_ex * W).sum() / jnp.maximum(W.sum(), 1e-8)
                return loss, per_ex / LN2
            (loss, per_bits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            upd, opt = self.tx.update(grads, opt, params)
            return optax.apply_updates(params, upd), opt, loss, per_bits

        def tf_err_fn(params, S, A, T, M):
            """Teacher-forced per-cell error (threshold 0.5), sweep-comparable."""
            lg = logits_fn(params, S, A)
            wrong = ((lg > 0) != (T > 0.5)).astype(jnp.float32) * M
            return wrong.sum(), M.sum()

        self._bits = jax.jit(bits_fn)
        self._eig = jax.jit(eig_fn)
        self._train = jax.jit(train_step)
        self._tf_err = jax.jit(tf_err_fn)

    # -- padded/bucketed numpy-facing wrappers --
    def _pad(self, arrs, n):
        b = _bucket(n)
        out = []
        for a in arrs:
            pad = np.zeros((b - n,) + a.shape[1:], a.dtype)
            out.append(np.concatenate([a, pad], axis=0))
        return out, b

    def bits(self, S, A, T, M):
        n = len(S)
        (S, A, T, M), _ = self._pad([S, A, T, M], n)
        return np.asarray(self._bits(self.params, S, A, T, M))[:n]

    def eig(self, S, A, M):
        n = len(S)
        (S, A, M), _ = self._pad([S, A, M], n)
        return np.asarray(self._eig(self.params, S, A, M))[:n]

    def train(self, S, A, T, M, W):
        self.params, self.opt, loss, per_bits = self._train(
            self.params, self.opt, S, A, T, M, W)
        return float(loss), np.asarray(per_bits)

    def tf_err(self, S, A, T, M, chunk=256):
        wrong = tot = 0.0
        for i in range(0, len(S), chunk):
            sl = slice(i, min(i + chunk, len(S)))
            n = sl.stop - sl.start
            (Sc, Ac, Tc, Mc), _ = self._pad([S[sl], A[sl], T[sl], M[sl]], n)
            w, t = self._tf_err(self.params, Sc, Ac, Tc, Mc)
            wrong += float(w); tot += float(t)
        return wrong / max(tot, 1.0)


# ---------------------------------------------------------------------------
# Search round + value iteration
# ---------------------------------------------------------------------------
def frontier_eig(g: Graph, wm: WM, sids, cell_mask):
    """Fresh expected IG per state = max over enabled actions of predictive
    entropy (bits). One batched forward pass."""
    if not sids:
        return np.zeros(0)
    A = len(g.acts)
    S = np.repeat(np.stack([g.obs[s] for s in sids]), A, axis=0)
    Aa = np.tile(np.asarray(g.acts, np.int32), len(sids))
    M = np.repeat(cell_mask[None], len(S), axis=0)
    e = wm.eig(S, Aa, M).reshape(len(sids), A)
    return e.max(axis=1)


def search_round(g: Graph, wm: WM, cell_mask, pop_m, expand_k, lam):
    cands = g.pop_candidates(pop_m)
    if not cands:
        return 0
    scores = frontier_eig(g, wm, cands, cell_mask)
    scores = scores - lam * np.asarray([g.depth[s] for s in cands])
    order = np.argsort(-scores)
    chosen = [cands[i] for i in order[:expand_k]]
    for i in order[expand_k:]:
        g.push(cands[i], float(scores[i]))

    e0 = g.n_edges
    new_sids = []
    for sid in chosen:
        for _, _, s2, is_new, won in g.expand(sid):
            if is_new and not won:    # win states are terminal: never queued
                new_sids.append(s2)
    # measure realized bits on the new edges in one batch
    idx = list(range(e0, g.n_edges))
    if idx:
        S = np.stack([g.obs[g.e_s[i]] for i in idx])
        A = np.asarray([g.e_a[i] for i in idx], np.int32)
        T = np.stack([g.obs[g.e_s2[i]] for i in idx])
        M = np.repeat(cell_mask[None], len(idx), axis=0)
        b = wm.bits(S, A, T, M)
        for j, i in enumerate(idx):
            g.e_bits[i] = float(b[j])
    # enqueue new states at their parent-edge realized bits (optimistic;
    # rescored with fresh EIG when popped)
    for s2 in new_sids:
        g.push(s2, max(float(g.e_bits[i]) for i in idx if g.e_s2[i] == s2))
    return len(chosen)


def value_iteration(g: Graph, lam, n_sweeps=10):
    """V[s] = bits still reachable from s (Bellman backup on current bits)."""
    V = np.zeros(g.n_states, np.float32)
    if g.n_edges == 0:
        return V
    es = np.asarray(g.e_s); es2 = np.asarray(g.e_s2)
    eb = np.asarray(g.e_bits, np.float32)
    for _ in range(n_sweeps):
        q = np.maximum(eb - lam + V[es2], 0.0)
        V2 = np.zeros_like(V)
        np.maximum.at(V2, es, q)
        if np.allclose(V2, V, atol=1e-4):
            V = V2
            break
        V = V2
    return V


# ---------------------------------------------------------------------------
# Heldout set (random rollouts on other levels)
# ---------------------------------------------------------------------------
def build_heldout(json_str, levels, n_obj, hp, wp, acts, n_eps, ep_len, seed):
    rng = np.random.default_rng(seed)
    S, A, T, M = [], [], [], []
    for li in levels:
        eng = new_engine(json_str, li)
        w, h = eng.get_width(), eng.get_height()
        m = np.zeros((hp, wp), np.float32)
        m[:min(h, hp), :min(w, wp)] = 1.0
        m3 = np.repeat(m[None], n_obj, axis=0)
        for _ in range(n_eps):
            eng.restart()
            o = read_obs(eng, n_obj, hp, wp)
            for _ in range(ep_len):
                a = int(rng.choice(acts))
                engine_step(eng, a)
                o2 = read_obs(eng, n_obj, hp, wp)
                S.append(o); A.append(a); T.append(o2); M.append(m3)
                o = o2
    return (np.stack(S), np.asarray(A, np.int32), np.stack(T), np.stack(M))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_levels(spec: str, n_levels: int) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [li for li in out if li < n_levels]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="heroes_of_sokoban")
    p.add_argument("--game_json", default=None,
                   help="path to a precompiled game JSON (skips the PS "
                        "parser entirely — lets the run happen on hosts with "
                        "only the C++ engine + jax installed)")
    p.add_argument("--levels", default="0",
                   help="train levels, e.g. '0-9' or '0,2,5' (one graph per "
                        "level, round-robin search, one WM on the union)")
    p.add_argument("--n_updates", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_hid", type=int, default=128)
    p.add_argument("--n_nca_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wm", choices=["nca", "event"], default="nca",
                   help="world-model architecture: the feedforward NCA or "
                        "the event-tokenized transformer (event_wm.al_wm."
                        "EventWM, exact joint likelihood + MC-entropy EIG)")
    p.add_argument("--mc_samples", type=int, default=4,
                   help="event WM only: MC samples for the EIG estimate")
    p.add_argument("--lam", type=float, default=0.0,
                   help="bits/step depth tax on frontier + value backup")
    p.add_argument("--tau", type=float, default=1.0,
                   help="edge-sampling temperature over (bits - lam + V[next])")
    p.add_argument("--expand_every", type=int, default=25)
    p.add_argument("--pop_m", type=int, default=24)
    p.add_argument("--expand_k", type=int, default=8)
    p.add_argument("--warmup_rounds", type=int, default=8)
    p.add_argument("--value_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--heldout_levels", type=int, default=6)
    p.add_argument("--heldout_eps", type=int, default=4)
    p.add_argument("--heldout_len", type=int, default=24)
    p.add_argument("--max_wins_per_level", type=int, default=5,
                   help="report/save at most this many win paths per level")
    p.add_argument("--env_step_budget", type=int, default=0,
                   help="if >0, stop expanding (but keep training) once total "
                        "env steps reach this — for matched-budget comparisons "
                        "against offline collection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/tmp/tree_growth")
    args = p.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.game_json:
        json_str = Path(args.game_json).read_text()
    else:
        json_str = compile_game(args.game)
    acts = enabled_actions(json_str)
    probe = new_engine(json_str, 0)
    n_obj = probe.get_object_count()
    n_levels = probe.get_num_levels()
    levels = parse_levels(args.levels, n_levels)
    hl = [li for li in range(n_levels) if li not in levels][:args.heldout_levels]
    # pad to max grid over train + heldout levels
    hp = wp = 0
    for li in levels + hl:
        e = new_engine(json_str, li)
        hp = max(hp, e.get_height()); wp = max(wp, e.get_width())
    print(f"[game] {args.game}  train levels {levels}  n_obj={n_obj} "
          f"pad=({hp},{wp}) actions={[ACTION_NAMES[a] for a in acts]}  "
          f"heldout levels {hl}", flush=True)

    graphs: list[Graph] = []
    masks: list[np.ndarray] = []
    for li in levels:
        eng = new_engine(json_str, li)
        tw, th = eng.get_width(), eng.get_height()
        m = np.zeros((n_obj, hp, wp), np.float32)
        m[:, :min(th, hp), :min(tw, wp)] = 1.0
        graphs.append(Graph(eng, n_obj, hp, wp, acts))
        masks.append(m)

    if args.wm == "event":
        from nca_wm.event_wm.al_wm import EventWM
        wm = EventWM(n_obj, hp, wp, args.seed, args.lr,
                     mc_samples=args.mc_samples)
    else:
        wm = WM(n_obj, hp, wp, args.n_hid, args.n_nca_steps, args.seed,
                args.lr)
    heldout = build_heldout(json_str, hl, n_obj, hp, wp, acts,
                            args.heldout_eps, args.heldout_len, args.seed + 1)
    print(f"[heldout] {len(heldout[0])} transitions from {len(hl)} levels", flush=True)

    # warmup expansion so training has edges
    for g, m in zip(graphs, masks):
        g.push(0, 1e9)
        for _ in range(args.warmup_rounds):
            search_round(g, wm, m, args.pop_m, args.expand_k, args.lam)
    print("[warmup] " + "  ".join(
        f"L{li}:{g.n_states}s/{g.n_edges}e/f{len(g.heap)}"
        for li, g in zip(levels, graphs)), flush=True)

    Vs = [value_iteration(g, args.lam) for g in graphs]
    wins_reported: set[tuple[int, int]] = set()
    metrics = []
    t0 = time.time()
    loss_ema = None

    def _fix_V(gi):
        if len(Vs[gi]) < graphs[gi].n_states:
            Vs[gi] = np.concatenate(
                [Vs[gi], np.zeros(graphs[gi].n_states - len(Vs[gi]), np.float32)])

    for u in range(1, args.n_updates + 1):
        if u % args.expand_every == 0 and not (
                args.env_step_budget
                and sum(g.env_steps for g in graphs) >= args.env_step_budget):
            # expand every level whose frontier is non-empty (mirrors the
            # mario trainer's all-variants-per-round expansion)
            for gi, g in enumerate(graphs):
                if g.heap:
                    search_round(g, wm, masks[gi],
                                 args.pop_m, args.expand_k, args.lam)
        if u % args.value_every == 0:
            Vs = [value_iteration(g, args.lam) for g in graphs]

        # -- sample edges across the union ∝ (bits - lam + V[next])^(1/tau).
        # Proportional prioritized replay (not softmax): early-training bits
        # are O(1e3)/transition, which under/overflows any exp() weighting. --
        pris, owners = [], []
        for gi, g in enumerate(graphs):
            if g.n_edges == 0:
                continue
            _fix_V(gi)
            eb = np.asarray(g.e_bits, np.float32)
            pris.append(np.clip(eb - args.lam + Vs[gi][np.asarray(g.e_s2)], 0.0, None))
            owners.append(np.full(g.n_edges, gi, np.int32))
        pri = np.concatenate(pris); own = np.concatenate(owners)
        offs = np.concatenate([[0], np.cumsum([len(p) for p in pris])])[:-1]
        w = pri ** (1.0 / max(args.tau, 1e-6)) + 1e-3
        w /= w.sum()
        flat = rng.choice(len(w), size=args.batch_size, replace=True, p=w)
        S, A, T, M, picked = [], [], [], [], []
        for f in flat:
            gi = int(own[f])
            ei = int(f - offs[np.searchsorted(offs, f, side="right") - 1])
            g = graphs[gi]
            S.append(g.obs[g.e_s[ei]]); A.append(g.e_a[ei])
            T.append(g.obs[g.e_s2[ei]]); M.append(masks[gi])
            picked.append((gi, ei))
        loss, per_bits = wm.train(np.stack(S), np.asarray(A, np.int32),
                                  np.stack(T), np.stack(M),
                                  np.ones(len(S), np.float32))
        for j, (gi, ei) in enumerate(picked):   # write fresh bits back
            graphs[gi].e_bits[ei] = float(per_bits[j])
        loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss

        # -- win reporting: extract + verify root path for new win states --
        for gi, g in enumerate(graphs):
            for sid in g.win_states:
                if (gi, sid) in wins_reported:
                    continue
                n_lvl = sum(1 for (gj, _) in wins_reported if gj == gi)
                if n_lvl >= args.max_wins_per_level:
                    wins_reported.add((gi, sid))   # count it, stay silent
                    continue
                path = g.root_path(sid)
                e2 = new_engine(json_str, levels[gi])
                for a in path:
                    engine_step(e2, a)
                ok = bool(e2.is_winning())
                print(f"[WIN] L{levels[gi]} state {sid} depth {len(path)} "
                      f"verified={ok}  "
                      f"actions={'.'.join(ACTION_NAMES[a] for a in path)}",
                      flush=True)
                (out / f"win_L{levels[gi]}_{sid}.json").write_text(json.dumps(
                    {"level": levels[gi], "state": sid, "verified": ok,
                     "actions": [ACTION_NAMES[a] for a in path],
                     "env_steps_at_discovery": g.env_steps}))
                wins_reported.add((gi, sid))

        if u % args.log_every == 0:
            ns = sum(g.n_states for g in graphs)
            ne = sum(g.n_edges for g in graphs)
            nf = sum(len(g.heap) for g in graphs)
            es = sum(g.env_steps for g in graphs)
            rules = set().union(*(g.rules_witnessed for g in graphs))
            nw = sum(len(g.win_states) for g in graphs)
            print(f"u={u:6d}  loss={loss_ema:.4f}  states={ns:6d} "
                  f"edges={ne:6d} frontier={nf:5d} env_steps={es:6d} "
                  f"rules={len(rules):2d} wins={nw}  {time.time()-t0:.0f}s",
                  flush=True)
        if u % args.eval_every == 0:
            err = wm.tf_err(*heldout)
            # train-fit: TF error on a sample of recorded edges (union)
            Ss, As, Ts, Ms = [], [], [], []
            for gi, g in enumerate(graphs):
                k = min(512 // len(graphs) + 1, g.n_edges)
                for i in rng.choice(g.n_edges, size=k, replace=False):
                    Ss.append(g.obs[g.e_s[i]]); As.append(g.e_a[i])
                    Ts.append(g.obs[g.e_s2[i]]); Ms.append(masks[gi])
            errt = wm.tf_err(np.stack(Ss), np.asarray(As, np.int32),
                             np.stack(Ts), np.stack(Ms))
            rules = set().union(*(g.rules_witnessed for g in graphs))
            print(f"  [eval u={u}] heldout_tf_err={err:.4f}  "
                  f"train_tf_err={errt:.4f}  rules_witnessed={sorted(rules)}",
                  flush=True)
            metrics.append(dict(
                u=u, heldout_tf_err=err, train_tf_err=errt,
                states=sum(g.n_states for g in graphs),
                edges=sum(g.n_edges for g in graphs),
                env_steps=sum(g.env_steps for g in graphs),
                rules=len(rules),
                wins=sum(len(g.win_states) for g in graphs),
                loss_ema=loss_ema))

    # -- save --
    with open(out / "params.pkl", "wb") as f:
        pickle.dump(wm.params, f)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for li, g in zip(levels, graphs):
        np.savez_compressed(
            out / f"graph_L{li}.npz",
            e_s=np.asarray(g.e_s), e_a=np.asarray(g.e_a),
            e_s2=np.asarray(g.e_s2), e_bits=np.asarray(g.e_bits),
            e_won=np.asarray(g.e_won), depth=np.asarray(g.depth),
            parent=np.asarray(g.parent),
            rules_witnessed=np.asarray(sorted(g.rules_witnessed)),
            env_steps=g.env_steps)
    rules = set().union(*(g.rules_witnessed for g in graphs))
    print(f"[done] {sum(g.n_states for g in graphs)} states, "
          f"{sum(g.n_edges for g in graphs)} edges, "
          f"{len(rules)} rules witnessed, "
          f"{sum(len(g.win_states) for g in graphs)} wins. Saved to {out}",
          flush=True)


if __name__ == "__main__":
    main()
