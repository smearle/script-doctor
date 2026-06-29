"""Two-world Mario belief WM for the information-gain experiment.

Worlds: custom_games/autumn/mario.txt (base) and mario_breakable.txt (variant).
They share level/objects/obs0 and differ in EXACTLY one transition: rising into a
Step from below leaves it intact (base) vs. removes that cell (variant). So the
world is hidden and the sole disambiguating action is a jump into a platform from
underneath.

This module:
  - compiles both worlds into `Game` objects (cached under _mario_worlds/),
  - a `mario_policy` biased toward the disambiguating jump (sparse under random),
  - haoo' trajectory batches (reuses multigame_data sampler internals),
  - trains the attn-belief model and gates on per-world held-out NLL,
  - a break-jump IG probe (fresh vs. known).

    .venv/bin/python -u -m nca_wm.active_learning.mario_belief --verify
    .venv/bin/python -u -m nca_wm.active_learning.mario_belief --train --updates 6000
"""
from __future__ import annotations

import argparse
import collections
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from nca_wm.active_learning import multigame_data as MD
from nca_wm.active_learning.multigame_data import Game, _engine, _masks, _perm, _read_padded
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig

ROOT = Path(__file__).resolve().parents[2]
_CACHE = Path(__file__).resolve().parent / "_mario_worlds"
CMAX, HMAX, WMAX = 24, 18, 16
WORLDS = [("mario", "custom_games/autumn/mario.txt"),
          ("mario_breakable", "custom_games/autumn/mario_breakable.txt")]

# Mario is realtime: the action index == the engine input id, and id 5 is the
# no-op REALTIME TICK on which the world's autonomous dynamics advance (gravity
# pulls the player down; bullets rise). Enemy patrol fires on every input, but
# gravity ONLY on the tick, so the world only evolves correctly if ticks are
# issued. (Matches _enabled_actions() / num_actions for realtime_interval games.)
ACTIONS = ["UP", "LEFT", "DOWN", "RIGHT", "ACTION", "TICK"]
N_ACT = len(ACTIONS)
TICK = 5


def _step(eng, a, seed=None):
    """One engine frame: action index a is the engine input id (5 = realtime tick)."""
    if seed is not None:
        eng.seed_rng(seed)
    eng.process_input(a)
    k = 0
    while eng.is_againing() and k < 50:
        eng.process_input(-1); k += 1


# ----------------------------- world building -----------------------------
def _compile(name, txt_rel):
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    _CACHE.mkdir(exist_ok=True)
    mat = _CACHE / "_scratch"; mat.mkdir(exist_ok=True)
    gc._set_materialize_dir(mat)
    parser = init_ps_lark_parser()
    code = (ROOT / txt_rel).read_text()
    gc._materialize_game(name, code)
    return CppPuzzleScriptBackend().compile_and_serialize(parser, name)


def build_worlds(force=False):
    games = []
    for name, rel in WORLDS:
        jf = _CACHE / f"{name}.json"
        if force or not jf.exists():
            _CACHE.mkdir(exist_ok=True)
            jf.write_text(_compile(name, rel))
        js = jf.read_text()
        e = _engine(js, 0)
        C, H, Wd = e.get_object_count(), e.get_height(), e.get_width()
        games.append(Game(name, js, C, H, Wd, e.get_num_levels()))
    return games


class _MG:
    """Minimal MultiGameSet shim for multigame_data.sample_trajectory."""
    def __init__(self, games):
        self.games = games
        self.cmax, self.hmax, self.wmax = CMAX, HMAX, WMAX


# ----------------------------- biased policy -----------------------------
def _bit(eng, name):
    low = [str(x).lower() for x in eng.get_id_dict()]
    return low.index(name.lower())


def _grid(eng):
    a = np.asarray(eng.get_objects_2d())          # (W,H,stride)
    return a[:, :, 0].T.astype(np.int64)          # (H,W)


def make_mario_policy(eng0, eps=0.3, jump_p=0.85):
    """Realtime-aware break-seeking policy with eps random coverage.

    - airborne -> TICK (let gravity bring Mario down a cell),
    - grounded under a break column -> UP (jump = the disambiguating action),
    - otherwise walk toward the nearest break column,
    - eps of the time take a random action (weighted toward TICK so the world
      actually advances; includes ACTION/shoot and movement for coverage).
    Break-reachable column c: scanning up <=4 rows, the first SOLID cell is a Step.
    """
    pbit = _bit(eng0, "Player"); sbit = _bit(eng0, "Step"); fbit = _bit(eng0, "Floor")

    def solid(cell):
        return bool(cell & ((1 << sbit) | (1 << fbit)))

    def policy(eng, rng):
        g = _grid(eng); H, Wd = g.shape
        pp = np.argwhere((g >> pbit) & 1)
        if len(pp) == 0:
            return TICK
        pr, pc = int(pp[0][0]), int(pp[0][1])
        grounded = pr + 1 < H and solid(g[pr + 1, pc])
        if not grounded:
            return TICK
        if rng.random() < eps:
            return rng.choices(range(N_ACT), weights=[2, 2, 1, 2, 1, 4])[0]
        targets = []
        for c in range(Wd):
            for dr in range(1, 5):
                rr = pr - dr
                if rr < 0:
                    break
                cell = g[rr, c]
                if solid(cell):
                    if (cell >> sbit) & 1:
                        targets.append(c)
                    break
        if not targets:
            return TICK
        if pc in targets:
            return 0 if rng.random() < jump_p else TICK     # 0 = UP / jump
        nearest = min(targets, key=lambda c: abs(c - pc))
        return 3 if nearest > pc else 1                     # 3 = RIGHT, 1 = LEFT

    return policy


class Explorer:
    """Novelty-seeking coverage policy (frontier search that yields trajectories).

    One-step lookahead over all 6 actions; take the one whose resulting state is
    least-visited (random tie-break), with eps random. Persistent visit counts
    drive broad coverage of the reachable state space the greedy break-seeker never
    reaches: collecting coins (-> HUD ammo), shooting (ACTION -> Bullet), killing the
    enemy, falling, etc. The state key is player-centric (player/coins/cursor/enemy/
    bullets) so coverage targets the gameplay-relevant variables, not enemy phase."""

    def __init__(self, eng0, eps=0.2):
        self.visits = collections.Counter()
        self.bits = {n: _bit(eng0, n) for n in
                     ("Player", "Coin", "Cursor", "Bullet", "EnemyL", "EnemyR")}
        self.eps = eps

    def _feat(self, eng):
        g = _grid(eng); b = self.bits
        def cells(k):
            return tuple(map(tuple, np.argwhere((g >> b[k]) & 1).tolist()))
        en = np.argwhere(((g >> b["EnemyL"]) & 1) | ((g >> b["EnemyR"]) & 1))
        encol = int(en[:, 1].min()) if len(en) else -1
        nb = int(((g >> b["Bullet"]) & 1).sum())
        return (cells("Player"), cells("Coin"), cells("Cursor"), encol, nb)

    def __call__(self, eng, rng):
        bak = eng.backup_level()
        cand = []
        for a in range(N_ACT):
            eng.restore_level(bak); eng.process_input(a)
            k = 0
            while eng.is_againing() and k < 50:
                eng.process_input(-1); k += 1
            f = self._feat(eng)
            cand.append((self.visits[f], a, f))
        eng.restore_level(bak)
        if rng.random() < self.eps:
            _, a, f = rng.choice(cand)
        else:
            mn = min(c[0] for c in cand)
            _, a, f = rng.choice([c for c in cand if c[0] <= mn])
        self.visits[f] += 1
        return a


def collect_start_states(game, n_pool=30000, cap=90000, seed=0):
    """BFS the reachable state graph (6 actions, full-grid state hash) and
    reservoir-sample `n_pool` diverse states as engine LevelBackup tokens. Training
    trajectories start from these so the WM sees the WHOLE reachable space (high
    ammo, enemy-in-line, deep platform positions, broken platforms) — not just the
    sliver near the initial state that short from-start rollouts reach."""
    rng = random.Random(seed)
    e = _engine(game.json_str, 0)

    def hsh():
        return np.asarray(e.get_objects_2d())[:, :, 0].tobytes()

    seen = {hsh()}
    q = collections.deque([e.backup_level()])
    pool = []
    nseen = 0
    while q and len(seen) < cap:
        bak = q.popleft()
        for a in range(N_ACT):
            e.restore_level(bak); _step(e, a)
            h = hsh()
            if h not in seen:
                seen.add(h); b = e.backup_level(); q.append(b); nseen += 1
                if len(pool) < n_pool:
                    pool.append(b)
                else:
                    j = rng.randrange(nseen)
                    if j < n_pool:
                        pool[j] = b
    return pool


def sample_traj(game, n_steps, rng, policy, start=None):
    """Roll one trajectory on ONE world, optionally from a search-discovered `start`
    state (LevelBackup). Both Mario worlds are DETERMINISTIC, so the haoo' resample
    equals the realized next state (o'==o); we set R = states[1:]."""
    perm = _perm(game.n_obj, CMAX, rng)
    eng = _engine(game.json_str, 0)
    if start is not None:
        eng.restore_level(start)

    def rd():
        return _read_padded(eng, game.n_obj, perm, CMAX, HMAX, WMAX)

    grids = [rd()]; acts = []
    for _ in range(n_steps):
        ai = rng.randrange(N_ACT) if policy is None else policy(eng, rng)
        _step(eng, ai)
        grids.append(rd()); acts.append(ai)
    cell, chan = _masks(game.n_obj, game.H, game.W, perm, CMAX, HMAX, WMAX)
    grids = np.stack(grids)
    return grids, np.asarray(acts, np.int64), grids[1:], cell, chan


def mario_batch(games, n, n_steps, rng, policies, start_pools=None, p_initial=0.3):
    """One per trajectory: with prob p_initial start at the INITIAL state with the
    break-seeker (canonical disambiguation-from-scratch); otherwise start from a
    random search-discovered state with the mixed policy (broad coverage)."""
    pol_list = policies if isinstance(policies, (list, tuple)) else [policies]
    O, A, R, CM, CH = [], [], [], [], []
    for _ in range(n):
        g = rng.choice(games)
        if start_pools and rng.random() > p_initial:
            start, policy = rng.choice(start_pools[g.gist]), rng.choice(pol_list)
        else:
            start, policy = None, pol_list[0]
        o, a, r, cell, chan = sample_traj(g, n_steps, rng, policy, start=start)
        O.append(o); A.append(a); R.append(r); CM.append(cell); CH.append(chan)
    t = torch.from_numpy
    return (t(np.stack(O)), t(np.stack(A)).long(), t(np.stack(R)),
            t(np.stack(CM)), t(np.stack(CH)))


def make_random_policy():
    """Cheap weighted-random rollout policy (favors TICK so the world advances)."""
    def policy(eng, rng):
        return rng.choices(range(N_ACT), weights=[2, 2, 1, 2, 1, 4])[0]
    return policy


def make_policies(games, eps=0.3, use_explorer=False):
    """Collection policy mix. With search (diverse BFS start states), broad coverage
    comes from the START states, so a cheap break-seeker + weighted-random rollout
    suffices. Without search, add the novelty Explorer to cover from the initial
    state (slower: 1-step lookahead)."""
    e0 = _engine(games[0].json_str, 0)
    pols = [make_mario_policy(e0, eps=eps), make_random_policy()]
    if use_explorer:
        ex = Explorer(e0)
        pols += [ex, ex]
    return pols


# ----------------------------- verification -----------------------------
def verify():
    games = build_worlds()
    for g in games:
        print(f"world {g.gist:16s} C={g.n_obj} H={g.H} W={g.W} levels={g.n_levels}")
    rng = random.Random(0)
    e0 = _engine(games[0].json_str, 0)
    policies = make_policies(games)
    sbit, ebL, ebR = _bit(e0, "Step"), _bit(e0, "EnemyL"), _bit(e0, "EnemyR")
    cbit, abit, blbit = _bit(e0, "Coin"), _bit(e0, "Ammo"), _bit(e0, "Bullet")

    # real determinism check (justifies R==next): same action from same state twice
    # gives the identical next state, across all actions, over a random walk.
    e = _engine(games[1].json_str, 0); det_ok = True
    for _ in range(40):
        a = rng.randrange(N_ACT); bak = e.backup_level()
        _step(e, a); n1 = _grid(e).copy()
        e.restore_level(bak); _step(e, a); n2 = _grid(e)
        if not np.array_equal(n1, n2):
            det_ok = False
    print(f"  determinism (o'==o for all actions): {det_ok}")

    # SEARCH pipeline: small BFS pool, roll trajectories from pool states; confirm
    # broad coverage (coins/shots/breaks) the from-initial rollouts never reached.
    pools = {g.gist: collect_start_states(g, 4000, 12000, 0) for g in games}
    print(f"  BFS pools: {', '.join(f'{k}={len(v)}' for k, v in pools.items())}")
    for g in games:
        n_break = enemy = coin = shot = tot = 0
        for _ in range(80):
            start = rng.choice(pools[g.gist])
            o, a, r, cell, chan = sample_traj(g, 16, rng, rng.choice(policies), start=start)
            spf = (o.astype(np.int64)[:, sbit] > 0).sum(axis=(1, 2))
            if (np.diff(spf) < 0).any():
                n_break += 1
            em = (o[:, ebL] + o[:, ebR]) > 0
            cols = [np.argwhere(f)[:, 1].mean() if f.any() else -1 for f in em]
            if len(set(np.round(cols, 1))) > 1:
                enemy += 1
            if (np.diff((o[:, abit] > 0).sum((1, 2))) > 0).any():     # ammo went up = coin collected
                coin += 1
            if (o[:, blbit] > 0).any():                              # a bullet present = shooting
                shot += 1
            tot += 1
        print(f"  {g.gist:16s} break {n_break}/{tot}  enemy-moved {enemy}/{tot}  "
              f"coin-pickup {coin}/{tot}  bullets {shot}/{tot}")


# ----------------------------- train + gate -----------------------------
@torch.no_grad()
def per_world_nll(model, games, device, n=120, n_steps=10, seed=7, start_pools=None):
    rng = random.Random(seed)
    policies = make_policies(games)
    out = {}
    for g in games:
        sp = {g.gist: start_pools[g.gist]} if start_pools else None
        O, A, R, CM, CH = mario_batch([g], n, n_steps, rng, policies, sp)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        q0, _ = model.forward_traj(O, A, R, CM, CH)
        out[g.gist] = (q0.mean().item(), q0.mean(0).tolist())
    return out


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    games = build_worlds()
    pol = make_policies(games, eps=args.eps, use_explorer=not args.search)
    start_pools = None
    if args.search:
        start_pools = {}
        for g in games:
            t = time.time()
            start_pools[g.gist] = collect_start_states(g, args.pool, args.bfs_cap, args.seed)
            print(f"BFS start-states {g.gist}: pool {len(start_pools[g.gist])} "
                  f"({time.time()-t:.0f}s)", flush=True)
    cfg = AttnConfig(n_obj=CMAX, n_act=N_ACT, d=args.d, d_model=args.d_model,
                     n_layer=args.n_layer, d_cond=args.d_cond, max_T=max(18, args.n_steps + 2))
    model = AttnBeliefModel(cfg).to(device)
    print(f"attn-belief params {sum(q.numel() for q in model.parameters()):,} | "
          f"cfg d={cfg.d} d_model={cfg.d_model} L={cfg.n_layer}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed); t0 = time.time(); model.train()
    for step in range(args.updates):
        O, A, R, CM, CH = mario_batch(games, args.batch_size, args.n_steps, rng, pol, start_pools)
        O, A, R, CM, CH = [x.to(device) for x in (O, A, R, CM, CH)]
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1))))
        q0, q1 = model.forward_traj(O, A, R, CM, CH)
        loss = q0.mean() + q1.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.3f}  upd/s "
                  f"{(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            model.eval()
            nw = per_world_nll(model, games, device, start_pools=start_pools)
            for k, (m, curve) in nw.items():
                print(f"  [{step}] {k:16s} q0NLL mean {m:.3f}  per-step "
                      f"{' '.join(f'{x:.2f}' for x in curve)}", flush=True)
            model.train()
            out = Path(__file__).resolve().parent / "ckpts" / "mario_belief.pt"
            out.parent.mkdir(exist_ok=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                        "worlds": [w[0] for w in WORLDS]}, out)
    model.eval()
    nw = per_world_nll(model, games, device, start_pools=start_pools)
    print("\nFINAL per-world q0 NLL:")
    for k, (m, curve) in nw.items():
        print(f"  {k:16s} mean {m:.4f}  per-step {' '.join(f'{x:.2f}' for x in curve)}")
    out = Path(__file__).resolve().parent / "ckpts" / "mario_belief.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                "worlds": [w[0] for w in WORLDS]}, out)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--updates", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=5)
    p.add_argument("--d-cond", type=int, default=160)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--search", action="store_true",
                   help="BFS-discover reachable states and start trajectories from them")
    p.add_argument("--pool", type=int, default=30000)
    p.add_argument("--bfs-cap", type=int, default=90000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    if args.verify:
        verify()
    if args.train:
        train(args)
    if not (args.verify or args.train):
        verify()


if __name__ == "__main__":
    main()
