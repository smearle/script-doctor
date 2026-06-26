#!/usr/bin/env python3
"""Per-object-CHANNEL representation (vs the discrete color grid).

`render_all` returns objects keyed by type, so a cell can hold several objects
(Mario *under* a coin). The color grid collapses that to one color (the coin),
making Mario "disappear". Here each object type is its own channel and a cell is
MULTI-HOT, so overlaps are preserved. Multi-hot => sigmoid + BCE (not softmax/CE).

Trains a Mario object-channel WM and verifies the overlap no longer hides Mario.

Usage:
    python -m nca_wm.autumn.objects --game mario --updates 4000
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.autumn.collect import AutumnGame, _suppress
from nca_wm.autumn.model import AutumnNCA, N_ATYPES, ATYPE_IDX, action_to_fields
from nca_wm.autumn.train import pick_device


def render_objects(env, vocab):
    """Multi-hot (C,H,W) uint8 over object-type channels; grows vocab in place."""
    d = json.loads(env.itp.render_all())
    gs = env.grid_size
    keys = [k for k in d if k != "GRID_SIZE"]
    for k in keys:
        if k not in vocab:
            vocab[k] = len(vocab)
    grid = np.zeros((max(len(vocab), 1), gs, gs), dtype=np.uint8)
    for k in keys:
        for c in d[k]:
            x, y = c["position"]["x"], c["position"]["y"]
            if 0 <= x < gs and 0 <= y < gs:
                grid[vocab[k], y, x] = 1
    return grid


def collect_objects(game, rollouts, length, seed, profile="agent", keep_prev=False):
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    vocab = {}
    arrow = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    aw = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def act():
        if profile == "agent":
            menu = [("noop", -1, -1)] + arrow + [("click", int(rng.integers(gs)), int(rng.integers(gs)))]
            w = np.array([0.15] + [aw[a[0]] for a in arrow] + [0.20])
        else:
            menu = [("noop", -1, -1), ("click", int(rng.integers(gs)), int(rng.integers(gs)))]
            w = np.array([0.2, 0.8])
        return menu[rng.choice(len(menu), p=w / w.sum())]

    # first pass discovers the full vocab; we pad all states to final C afterwards
    seen = {}
    t0 = time.time()
    for r in range(rollouts):
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", int(rng.integers(1 << 30)))
        s = render_objects(env, vocab)
        prev = s
        for t in range(length):
            a = act()
            env.apply(a)
            ns = render_objects(env, vocab)
            key = (prev.tobytes(), s.tobytes(), a) if keep_prev else (s.tobytes(), a, s.shape[0])
            if key not in seen:
                ai, cx, cy = action_to_fields(a)
                seen[key] = (prev, s, ai, cx, cy, ns)
            prev = s; s = ns
    C = len(vocab)

    def pad(g):
        if g.shape[0] == C:
            return g
        out = np.zeros((C, g.shape[1], g.shape[2]), dtype=np.uint8)
        out[:g.shape[0]] = g
        return out

    prevs = np.stack([pad(v[0]) for v in seen.values()])
    states = np.stack([pad(v[1]) for v in seen.values()])
    ai = np.array([v[2] for v in seen.values()], dtype=np.uint8)
    cx = np.array([v[3] for v in seen.values()], dtype=np.int16)
    cy = np.array([v[4] for v in seen.values()], dtype=np.int16)
    nexts = np.stack([pad(v[5]) for v in seen.values()])
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    overlap = int((states.sum(1) > 1).sum())
    print(f"[{game}/objects] {len(seen)} transitions in {time.time()-t0:.1f}s | "
          f"channels({C})={inv} | multi-object cells in data: {overlap}")
    out = dict(states=states, next_states=nexts, action_type=ai, click_x=cx,
               click_y=cy, vocab=np.array(inv), grid_size=np.int32(gs), game=np.str_(game))
    if keep_prev:
        out["prev_states"] = prevs
    return out


def collect_object_sequences(game, episodes, length, seed, profile="agent"):
    """Ordered multi-hot episodes for recurrent BPTT. states (E,L+1,C,H,W)."""
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    vocab = {}
    arrow = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    aw = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def rand_act():
        menu = [("noop", -1, -1)] + arrow + [("click", int(rng.integers(gs)), int(rng.integers(gs)))]
        w = np.array([0.15] + [aw[a[0]] for a in arrow] + [0.20])
        return menu[rng.choice(len(menu), p=w / w.sum())]

    # mario objects heuristic: the RANDOM policy rarely triggers real jumps (only fire when
    # GROUNDED) or real fires (only after collecting a coin), so those mechanics are undercovered
    # and diverge autoregressively. Deliberately exercise them: walk under the nearest coin, jump
    # to collect it (gaining a bullet), then fire bursts (bullets travel up). vocab: mario/steps/
    # coins/enemy/bullets channels.
    mario_fire_left = [0]

    def mario_act():
        g = render_objects(env, vocab)  # (C,H,W) in current vocab order
        idx = {k: i for i, k in vocab.items()}
        mi, ci = idx.get("mario"), idx.get("coins")
        if mi is None or mi >= g.shape[0]:
            return rand_act()
        ys, xs = np.where(g[mi] == 1)
        if not len(xs):
            return rand_act()
        mx, my = int(xs[0]), int(ys[0])
        if mario_fire_left[0] > 0:                         # fire burst (bullets travel up)
            mario_fire_left[0] -= 1
            return ("click", mx, my)
        if rng.random() < 0.20:                            # fire opportunistically (uses a bullet
            mario_fire_left[0] = int(rng.integers(2, 5))    # if mario has any -> bullet travel)
            return ("click", mx, my)
        # CLIMBER: pursue the nearest coin -- walk to its column, jump to climb platforms toward it
        # (collecting a coin grants a bullet, enabling real fires). This makes the jump AND
        # bullet mechanics -- the two AR-divergent ones -- actually fire in the data.
        if ci is not None and ci < g.shape[0]:
            cys, cxs = np.where(g[ci] == 1)
            if len(cxs):
                j = int(np.argmin(np.abs(cxs - mx) + 0.3 * np.abs(cys - my)))
                tx, ty = int(cxs[j]), int(cys[j])
                if abs(mx - tx) >= 2:
                    return ("right", -1, -1) if mx < tx else ("left", -1, -1)
                if ty < my:                                # coin is above -> jump to climb
                    return ("up", -1, -1)
                return ("right", -1, -1) if mx < tx else (("left", -1, -1) if mx > tx else ("up", -1, -1))
        return ("up", -1, -1) if rng.random() < 0.5 else arrow[rng.integers(4)]

    act = mario_act if game == "mario" else rand_act

    all_s, all_a = [], []
    t0 = time.time()
    for e in range(episodes):
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", int(rng.integers(1 << 30)))
        st = [render_objects(env, vocab)]
        ac = []
        for t in range(length):
            a = act(); env.apply(a)
            st.append(render_objects(env, vocab)); ac.append(list(action_to_fields(a)))
        all_s.append(st); all_a.append(np.array(ac, dtype=np.int16))
    C = len(vocab)

    def pad(g):
        if g.shape[0] == C:
            return g
        o = np.zeros((C, g.shape[1], g.shape[2]), dtype=np.uint8); o[:g.shape[0]] = g; return o

    states = np.stack([np.stack([pad(g) for g in ep]) for ep in all_s]).astype(np.uint8)
    actions = np.stack(all_a)
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    print(f"[{game}/obj-seq] {episodes}x{length} in {time.time()-t0:.1f}s | channels({C})={inv}")
    return dict(states=states, actions=actions, vocab=np.array(inv),
                grid_size=np.int32(gs), game=np.str_(game))


def _obj_mario_act(env, vocab, rng, fire_state):
    """Climber heuristic over object channels: pursue+collect coins (-> bullets), fire bursts,
    jump to climb -- exercises the jump+bullet mechanics that diverge. Used to SEED and BIAS the
    object search toward heuristically-valuable states."""
    g = render_objects(env, vocab)
    idx = dict(vocab)  # vocab is {name: channel}; use it directly (the old inversion gave {channel: name}, so mario/coins were never found and the climber degenerated to random nav)
    mi, ci = idx.get("mario"), idx.get("coins")
    gs = env.grid_size
    if mi is None or mi >= g.shape[0] or not len(np.where(g[mi] == 1)[0]):
        return ("noop", -1, -1) if rng.random() < 0.3 else (["up", "down", "left", "right"][rng.integers(4)],)
    ys, xs = np.where(g[mi] == 1); mx, my = int(xs[0]), int(ys[0])
    if fire_state[0] > 0:
        fire_state[0] -= 1; return ("click", mx, my)
    if rng.random() < 0.20:
        fire_state[0] = int(rng.integers(2, 5)); return ("click", mx, my)
    if ci is not None and ci < g.shape[0]:
        cys, cxs = np.where(g[ci] == 1)
        # only coins within a single jump of the floor are collectable (mario jumps up 4 from the
        # floor, then gravity pulls it back). Targeting a coin straight overhead that it can never
        # reach makes it bounce forever and never exercise the coin->ammo->fire chain.
        reach = cys >= (gs - 5)
        if reach.any():
            cxs, cys = cxs[reach], cys[reach]
        if len(cxs):
            j = int(np.argmin(np.abs(cxs - mx) + 0.3 * np.abs(cys - my))); tx, ty = int(cxs[j]), int(cys[j])
            if abs(mx - tx) >= 2: return ("right", -1, -1) if mx < tx else ("left", -1, -1)
            if ty < my: return ("up", -1, -1)
            return ("right", -1, -1) if mx < tx else (("left", -1, -1) if mx > tx else ("up", -1, -1))
    return ("up", -1, -1) if rng.random() < 0.5 else (["up", "down", "left", "right"][rng.integers(4)],)


def collect_object_search(game, n_seed=300, seed_len=40, n_cont=4, cont_len=14, heur_frac=0.55,
                          target_unique=400000, max_seconds=900, max_seqs=14000, seed=0,
                          seed_human=True, verbose=True):
    """Heuristic-GUIDED object-state search. Two phases:
      1. seed rollouts using the climber heuristic -> trajectories through heuristically-valuable
         states (jumps/fires/collects); every unique object-state becomes a frontier node (with its
         replayable seed+path).
      2. from frontier nodes, branch CONTINUATION rollouts under a mix of heuristic (explore valuable
         states) + random (novelty/coverage); dedup by object-state bytes so kept transitions are
         unique. With enough budget this covers the reachable object-state space.
    Returns object-sequences (E,L+1,C,H,W) for the recurrent trainer."""
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    vocab = {}
    fire = [0]

    def policy(use_heur):
        if game == "mario" and use_heur:
            return _obj_mario_act(env, vocab, rng, fire)
        menu = [("noop", -1, -1), ("up", -1, -1), ("down", -1, -1), ("left", -1, -1),
                ("right", -1, -1), ("click", int(rng.integers(gs)), int(rng.integers(gs)))]
        return menu[rng.integers(len(menu))]

    seen = set(); seqs = []; frontier = []
    t0 = time.time()

    def add_rollout(ep_seed, path_in, record_seq=True):
        """Replay (ep_seed, path_in) in the objects env; add unique object-states to the frontier
        and (optionally) the trajectory as a training sequence."""
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", int(ep_seed))
        fire[0] = 0; states = [render_objects(env, vocab)]; built = []
        for a in path_in:
            if a[0] == "click" and (a[1] < 0 or a[2] < 0):
                a = ("noop", -1, -1)
            env.apply(a); states.append(render_objects(env, vocab)); built.append(a)
            k = states[-1].tobytes()
            if k not in seen:
                seen.add(k); frontier.append((int(ep_seed), list(built)))
        if record_seq and built:
            seqs.append((states, built))

    n_human = 0
    if seed_human:                                            # phase 0: HUMAN playtraces seed the frontier
        try:
            from nca_wm.autumn.human_replay import collect_human
            from nca_wm.autumn.search_collect import _fields_to_action
            eps, meta = collect_human(game, ["black"])        # palette irrelevant -- we use actions+seeds
            hseeds = meta.get("seeds", [])
            for i, (S, A) in enumerate(eps):
                if i >= len(hseeds) or int(hseeds[i]) < 0:
                    continue
                path = [_fields_to_action(*a) for a in A][:seed_len * 2]
                add_rollout(hseeds[i], path)
                n_human += 1
        except Exception as e:
            print(f"  (human seeding skipped: {e})")

    for e in range(n_seed):                                   # phase 1: heuristic seed rollouts
        ep_seed = int(rng.integers(1 << 30))
        with _suppress():
            env.itp.run_script(env.prog, env._stdlib, "", ep_seed)
        fire[0] = 0; states = [render_objects(env, vocab)]; path = []
        for t in range(seed_len):
            a = policy(True); env.apply(a); states.append(render_objects(env, vocab)); path.append(a)
            k = states[-1].tobytes()
            if k not in seen:
                seen.add(k); frontier.append((ep_seed, list(path)))
        seqs.append((states, path))
    n_seed_states = len(seen)
    if verbose and seed_human:
        print(f"  seeded from {n_human} human traces + {n_seed} heuristic rollouts -> {len(frontier)} frontier nodes")
    t_seed = time.time() - t0
    rng.shuffle(frontier)
    for ep_seed, path in frontier:                            # phase 2: heuristic-biased continuations
        if len(seen) >= target_unique or time.time() - t0 > max_seconds or len(seqs) >= max_seqs:
            break
        for m in range(n_cont):
            with _suppress():
                env.itp.run_script(env.prog, env._stdlib, "", ep_seed)
            fire[0] = 0; states = [render_objects(env, vocab)]
            for a in path:
                env.apply(a); states.append(render_objects(env, vocab))
            cpath = list(path)
            for t in range(cont_len):
                a = policy(rng.random() < heur_frac)
                env.apply(a); states.append(render_objects(env, vocab)); cpath.append(a)
                seen.add(states[-1].tobytes())
            seqs.append((states, cpath))
    C = len(vocab)

    def pad(gg):
        if gg.shape[0] == C:
            return gg
        o = np.zeros((C, gg.shape[1], gg.shape[2]), np.uint8); o[:gg.shape[0]] = gg; return o

    L = max(len(p) for _, p in seqs)
    E = len(seqs)
    states = np.zeros((E, L + 1, C, gs, gs), np.uint8); actions = np.zeros((E, L, 3), np.int64)
    for i, (st, pa) in enumerate(seqs):
        l = len(pa)
        for t in range(l + 1):
            states[i, t] = pad(st[t])
        for t in range(l + 1, L + 1):
            states[i, t] = pad(st[l])
        for t in range(l):
            actions[i, t] = action_to_fields(pa[t])
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    if verbose:
        cur = states[:, :-1]; nxt = states[:, 1:]
        uniq = len(seen)
        print(f"[{game}/obj-search] {E} seqs (maxL={L}), {n_seed_states} seed-states -> {uniq} UNIQUE "
              f"object-states in {time.time()-t0:.0f}s | channels({C})={inv}")
    return dict(states=states, actions=actions, vocab=np.array(inv),
                grid_size=np.int32(gs), game=np.str_(game))


def _apply_action(itp, a):
    """Apply an action tuple to the interpreter + step."""
    t = a[0]
    if t == "noop":
        itp.step()
    elif t == "click":
        itp.click(int(a[1]), int(a[2])); itp.step()
    else:
        getattr(itp, t)(); itp.step()


def collect_object_search_snap(game, n_seed=300, seed_len=40, click_stride=3, max_states=200000,
                               max_seconds=600, seed_human=True, seed=0, verbose=True):
    """SNAPSHOT-based BFS object search (uses the engine's save_state/load_state -- ~47x faster
    than replay-from-seed, O(1) restore instead of re-walking each path). Seeds the frontier from
    human + heuristic rollouts (snapshotting every visited state), then BFS-expands by restoring a
    node's snapshot and trying each action -- novelty-dedup'd on the object-state so every kept
    transition is unique. Reconstructs training sequences via parent pointers."""
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed); itp = env.itp; gs = env.grid_size
    if not hasattr(itp, "save_state"):
        raise RuntimeError("engine lacks save_state/load_state -- rebuild MARA with the snapshot binding")
    vocab = {}
    fire = [0]

    def render():
        return render_objects(env, vocab)

    def policy(use_heur):
        if game == "mario" and use_heur:
            return _obj_mario_act(env, vocab, rng, fire)
        menu = [("noop",), ("up",), ("down",), ("left",), ("right",),
                ("click", int(rng.integers(gs)), int(rng.integers(gs)))]
        return menu[rng.integers(len(menu))]

    action_set = [("noop",), ("up",), ("down",), ("left",), ("right",)]
    for y in range(0, gs, click_stride):
        for x in range(0, gs, click_stride):
            action_set.append(("click", x, y))

    # node = dict(snap, grid, parent, act_from_parent, depth)
    nodes = []
    seen = {}
    frontier = []
    t0 = time.time()

    def add_node(grid, parent, act):
        k = grid.tobytes()
        if k in seen:
            return None
        seen[k] = len(nodes)
        nodes.append(dict(snap=itp.save_state(), grid=grid.copy(), parent=parent, act=act,
                          depth=(0 if parent is None else nodes[parent]["depth"] + 1)))
        frontier.append(len(nodes) - 1)
        return len(nodes) - 1

    # --- seed phase: human + heuristic rollouts; snapshot every visited state into the frontier ---
    n_human = 0
    seed_specs = []
    if seed_human:
        try:
            from nca_wm.autumn.human_replay import collect_human
            from nca_wm.autumn.search_collect import _fields_to_action
            eps, meta = collect_human(game, ["black"])
            hs = meta.get("seeds", [])
            for i, (S, A) in enumerate(eps):
                if i < len(hs) and int(hs[i]) >= 0:
                    seed_specs.append((int(hs[i]), [_fields_to_action(*a) for a in A][:seed_len * 2]))
                    n_human += 1
        except Exception as e:
            print(f"  (human seeding skipped: {e})")
    for _ in range(n_seed):
        seed_specs.append((int(rng.integers(1 << 30)), None))   # heuristic rollout
    for ep_seed, path in seed_specs:
        with _suppress():
            itp.run_script(env.prog, env._stdlib, "", ep_seed)
        fire[0] = 0
        prev = add_node(render(), None, None)
        steps = path if path is not None else range(seed_len)
        for st in steps:
            a = st if path is not None else policy(True)
            if a[0] == "click" and (a[1] < 0 or a[2] < 0):
                a = ("noop",)
            try:
                _apply_action(itp, a)
            except RuntimeError:
                break  # interpreter died (e.g. mario.sexp crashes when a bullet kills the enemy -> prev enemy undefined); abort this rollout
            nid = add_node(render(), prev, a)
            prev = nid if nid is not None else seen[render().tobytes()]
    n_seed_states = len(nodes)

    # --- BFS expansion via snapshot restore (no replay) ---
    fi = 0
    while fi < len(frontier) and len(nodes) < max_states and time.time() - t0 < max_seconds:
        nid = frontier[fi]; fi += 1
        node = nodes[nid]
        for a in action_set:
            itp.load_state(node["snap"])         # O(1) restore to this node (no replay!)
            try:
                _apply_action(itp, a)
            except RuntimeError:
                continue  # this action crashes the interpreter (enemy-death edge case); skip it, snapshot reload keeps the node clean
            add_node(render(), nid, a)
            if len(nodes) >= max_states or time.time() - t0 > max_seconds:
                break

    # --- reconstruct sequences via parent pointers (root -> node paths) ---
    C = len(vocab)

    def pad(g):
        if g.shape[0] == C:
            return g
        o = np.zeros((C, g.shape[1], g.shape[2]), np.uint8); o[:g.shape[0]] = g; return o

    # emit a sequence for each node that is a leaf or at the frontier tail (sampled for diversity)
    leaf = np.ones(len(nodes), bool)
    for n in nodes:
        if n["parent"] is not None:
            leaf[n["parent"]] = False
    seqs = []
    for nid, n in enumerate(nodes):
        if not leaf[nid] or n["depth"] == 0:
            continue
        chain = []
        cur = nid
        while cur is not None:
            chain.append(cur); cur = nodes[cur]["parent"]
        chain.reverse()
        states = [pad(nodes[c]["grid"]) for c in chain]
        acts = [nodes[c]["act"] for c in chain[1:]]
        seqs.append((states, acts))
    L = max((len(a) for _, a in seqs), default=1)
    E = len(seqs)
    S = np.zeros((E, L + 1, C, gs, gs), np.uint8); Aout = np.zeros((E, L, 3), np.int64)
    for i, (st, ac) in enumerate(seqs):
        l = len(ac)
        for t in range(l + 1):
            S[i, t] = st[t]
        for t in range(l + 1, L + 1):
            S[i, t] = st[l]
        for t in range(l):
            Aout[i, t] = action_to_fields(ac[t])
    inv = [None] * C
    for k, i in vocab.items():
        inv[i] = k
    if verbose:
        print(f"[{game}/obj-search-snap] {n_human} human + {n_seed} heuristic seeds -> {len(nodes)} UNIQUE "
              f"states, {E} seqs in {time.time()-t0:.0f}s | channels({C})={inv}")
    return dict(states=S, actions=Aout, vocab=np.array(inv), grid_size=np.int32(gs), game=np.str_(game))


def encode(states, at, cx, cy, device, hist=None):
    """states (B,C,H,W) multi-hot float; returns (state, atype_oh, click_map, hist|None)."""
    s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    B, C, H, W = s.shape
    a = torch.as_tensor(np.asarray(at), dtype=torch.long, device=device)
    atype_oh = F.one_hot(a, N_ATYPES).float()
    click = torch.zeros(B, 1, H, W, device=device)
    cxx = torch.as_tensor(np.asarray(cx), dtype=torch.long, device=device)
    cyy = torch.as_tensor(np.asarray(cy), dtype=torch.long, device=device)
    idx = torch.nonzero(a == ATYPE_IDX["click"], as_tuple=True)[0]
    if idx.numel():
        click[idx, 0, cyy[idx], cxx[idx]] = 1.0
    hist_t = None
    if hist is not None:
        hist_t = torch.as_tensor(np.asarray(hist), dtype=torch.float32, device=device)
    return s, atype_oh, click, hist_t


@torch.no_grad()
def evaluate(model, data, idx, device, bs=256):
    model.eval()
    s, ns = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    prev = data.get("prev_states")
    cell_corr = cell_tot = ch_chg = ch_chg_corr = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        st, ao, cm, ht = encode(s[b], at[b], cx[b], cy[b], device,
                                hist=prev[b] if prev is not None else None)
        pred = (torch.sigmoid(model(st, ao, cm, ht)) > 0.5).float().cpu().numpy().astype(np.uint8)
        tgt = ns[b]
        cell_corr += int((pred == tgt).sum()); cell_tot += pred.size
        chg = tgt != s[b]
        ch_chg += int(chg.sum()); ch_chg_corr += int(((pred == tgt) & chg).sum())
    model.train()
    return cell_corr / cell_tot, ch_chg_corr / max(ch_chg, 1)


def run_recurrent(args, device):
    """Object channels + recurrent hidden grid -> tracks bullet counter too."""
    from nca_wm.autumn.model import RecurrentAutumnNCA
    save_dir = getattr(args, "save_dir", "") or f"nca_wm/autumn/runs/{args.game}_objects_recurrent"
    os.makedirs(save_dir, exist_ok=True)
    if getattr(args, "data", "") and os.path.exists(args.data):
        d = dict(np.load(args.data, allow_pickle=True))   # pre-collected large dataset
        print(f"loaded {args.data}: states {d['states'].shape}")
    else:
        d = collect_object_sequences(args.game, args.rollouts, min(args.length, 40), args.seed)
    s, acts = d["states"], d["actions"]
    E, Lp1, C, H, W = s.shape
    L = acts.shape[1]
    vocab = list(d["vocab"]); BUL = vocab.index("bullets") if "bullets" in vocab else -1
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(E); nv = max(1, int(0.1 * E)); val, train = perm[:nv], perm[nv:]
    model = RecurrentAutumnNCA(C, n_hid=args.n_hid, n_micro=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # per-channel positive weight (inverse frequency, capped) so ultra-sparse
    # channels like `bullets` aren't drowned out by always-0 BCE.
    pos = s[:, 1:].reshape(-1, C, H, W).mean((0, 2, 3))  # positive rate per channel
    pw = np.clip((1 - pos) / np.clip(pos, 1e-6, None), 1, args.pw_cap).astype(np.float32)
    if args.bullet_pw >= 0 and BUL >= 0:
        pw[BUL] = args.bullet_pw   # bullets get a dedicated (low) weight to stop the over-spawn
    pos_weight = torch.tensor(pw, device=device)[None, None, :, None, None]  # (1,1,C,1,1)
    print(f"obj-recurrent channels={C} E={E} L={L} pw_cap={args.pw_cap} bullet_pw={args.bullet_pw} select={args.select} pos_weight={pw.round(1)}")

    def unroll(bs_states, bs_acts, ss_p=0.0):
        h = model.init_hidden(len(bs_states), H, W, device)
        outs = []
        prev_pred = None
        for t in range(L):
            # encode gives the action maps (ao, cm); the state is GT or, under scheduled
            # sampling, the model's OWN previous prediction (binarized) -- so it learns to
            # recover from its own drift, attacking the AR exposure-bias gap (TF 99% vs AR 77%)
            st_gt, ao, cm, _ = encode(bs_states[:, t], bs_acts[:, t, 0], bs_acts[:, t, 1],
                                      bs_acts[:, t, 2], device)
            if ss_p > 0 and t > 0 and prev_pred is not None and float(torch.rand(())) < ss_p:
                st = prev_pred
            else:
                st = st_gt
            logits, h = model.step(st, ao, cm, h)
            prev_pred = (torch.sigmoid(logits) > 0.5).float().detach()
            outs.append(logits)
        return torch.stack(outs, 1)  # (B,L,C,H,W)

    best = -1
    for step in range(1, args.updates + 1):
        ss_p = args.sched_samp * min(1.0, step / max(1, args.updates // 2))  # ramp 0->max
        b = rng.choice(train, size=args.batch_size)
        logits = unroll(s[b], acts[b], ss_p)
        tgt = torch.as_tensor(s[b, 1:], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                ft = hit = pred_fire = 0
                chg_cells = chg_ok = 0
                for i in range(0, len(val), 8):
                    vb = val[i:i + 8]
                    pr = (torch.sigmoid(unroll(s[vb], acts[vb])) > 0.5).cpu().numpy().astype(np.uint8)
                    cur_a = s[vb, :-1]; tgt_a = s[vb, 1:]                  # (b,L,C,H,W)
                    changed = cur_a != tgt_a
                    chg_cells += int(changed.sum()); chg_ok += int(((pr == tgt_a) & changed).sum())
                    if BUL >= 0:
                        cur = (cur_a[:, :, BUL] == 1).sum((2, 3)); nxt = (tgt_a[:, :, BUL] == 1).sum((2, 3))
                        pn = (pr[:, :, BUL] == 1).sum((2, 3)); sp = nxt > cur
                        ft += int(sp.sum()); hit += int(((pn > cur) & sp).sum())
                        pred_fire += int((pn > cur).sum())   # every step the model spawns a bullet (real OR phantom)
                frec = hit / max(ft, 1); chg_acc = chg_ok / max(chg_cells, 1)
                fprec = hit / max(pred_fire, 1)              # of the model's spawns, fraction that were real
                ff1 = 2 * fprec * frec / max(fprec + frec, 1e-9)
            print(f"[{step}] loss={loss.item():.4f} changed_acc={chg_acc:.3f} fire_events={ft} "
                  f"bullet_recall={frec:.3f} bullet_prec={fprec:.3f} bullet_f1={ff1:.3f}")
            # combined selection: chg_acc + bullet_{recall|f1}. recall alone picks the most-firing
            # checkpoint (ignores false positives -> AR phantom bullets); f1 is precision-aware.
            bullet_term = ff1 if args.select == "f1" else frec
            score = chg_acc + bullet_term
            if score > best:
                best = score
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
    json.dump(dict(vocab=vocab, grid_size=int(d["grid_size"]), n_colors=C, n_hid=args.n_hid,
                   n_steps=args.n_steps, n_micro=3, global_pool=True, history=0, multihot=True,
                   recurrent=True, game=str(d["game"]), buttons={}),
              open(os.path.join(save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {save_dir} (best bullet_recall={best:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mario")
    ap.add_argument("--rollouts", type=int, default=400)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_hid", type=int, default=96)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--history", action="store_true",
                    help="add 1-frame history (object channels + history -> overlap AND enemy fixed)")
    ap.add_argument("--recurrent", action="store_true",
                    help="object channels + recurrent hidden grid (tracks bullet counter)")
    ap.add_argument("--sched_samp", type=float, default=0.0,
                    help="max scheduled-sampling prob (ramped 0->this): feed the model its own "
                         "predictions during the unroll to fix AR exposure-bias drift")
    ap.add_argument("--data", default="",
                    help="pre-collected object-sequence npz to train from (skips inline collection)")
    ap.add_argument("--save_dir", default="",
                    help="override the output run dir (default nca_wm/autumn/runs/{game}_objects_recurrent)")
    ap.add_argument("--pw_cap", type=float, default=100.0,
                    help="cap on per-channel inverse-freq pos_weight.")
    ap.add_argument("--bullet_pw", type=float, default=-1.0,
                    help="override pos_weight for the bullets channel specifically (-1 = use capped "
                         "inverse-freq like the rest). The inverse-freq weight for ultra-sparse bullets "
                         "is ~100, which makes a missed bullet cost 100x a spurious one -> the model "
                         "over-fires (AR phantom bullets). Set ~2-5 to stop the over-spawn without "
                         "lowering the denser coins/enemy channels.")
    ap.add_argument("--select", default="recall", choices=["recall", "f1"],
                    help="checkpoint selection: chg_acc + bullet_{recall|f1}. recall picks the most-firing "
                         "checkpoint (ignores false positives -> phantoms); f1 is precision-aware.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    if args.recurrent:
        run_recurrent(args, device); return
    tag = "_objects_hist" if args.history else "_objects"
    save_dir = f"nca_wm/autumn/runs/{args.game}{tag}"
    os.makedirs(save_dir, exist_ok=True)

    data = collect_objects(args.game, args.rollouts, args.length, args.seed, keep_prev=args.history)
    C = data["states"].shape[1]
    N = len(data["states"])
    prev = data.get("prev_states")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N); nval = int(0.1 * N)
    val, train = perm[:nval], perm[nval:]
    model = AutumnNCA(C, n_hid=args.n_hid, n_steps=args.n_steps,
                      history=1 if args.history else 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    s, ns = data["states"], data["next_states"]
    at, cx, cy = data["action_type"], data["click_x"], data["click_y"]
    print(f"channels={C} N={N} history={args.history} params={sum(p.numel() for p in model.parameters()):,}")
    best = -1
    for step in range(1, args.updates + 1):
        b = rng.choice(train, size=args.batch_size)
        st, ao, cm, ht = encode(s[b], at[b], cx[b], cy[b], device,
                                hist=prev[b] if prev is not None else None)
        tgt = torch.as_tensor(ns[b], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(model(st, ao, cm, ht), tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == 1:
            acc, chg = evaluate(model, data, val, device)
            print(f"[{step}] loss={loss.item():.4f} cell_acc={acc:.4f} changed_acc={chg:.3f}")
            if chg > best:
                best = chg
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
    json.dump(dict(vocab=list(data["vocab"]), grid_size=int(data["grid_size"]), n_colors=C,
                   n_hid=args.n_hid, n_steps=args.n_steps, global_pool=True,
                   history=1 if args.history else 0,
                   multihot=True, game=str(data["game"]), buttons={}),
              open(os.path.join(save_dir, "config.json"), "w"), indent=2)
    print(f"saved -> {save_dir} (best changed_acc={best:.3f})")


if __name__ == "__main__":
    main()
