#!/usr/bin/env python3
"""Collect (state, action, next_state) transitions from an Autumn program.

Discrete color-grid representation; actions are {noop, click(x,y), up/down/left/right}.
Collection is randomized deduped exploration ("search"): many rollouts under a
coverage policy, recording every real engine transition and deduping by
(state, action). All next-states come from the engine (the oracle).

Coverage fixes for CA-like games (clustered seeding + density-adaptive policy +
periodic re-seeding) ensure local neighborhoods span 0..8 live neighbors AND that
global actions (a board-clear / global-step button) are sampled from DENSE boards,
not just the sparse boards a random walk usually settles into.

Usage:
    python -m nca_wm.autumn.collect --game gameOfLife --rollouts 300 --rollout_len 200 \
        --out nca_wm/autumn/data/gameOfLife.npz
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np

from nca_wm.autumn.model import action_to_fields

MARA = "/home/jupyter-smearle/mara/MARA"
TESTS = f"{MARA}/domains/autumnbench/Autumn.wasm/tests"


@contextlib.contextmanager
def _suppress():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old1, 1); os.dup2(old2, 2)
        os.close(devnull); os.close(old1); os.close(old2)


class AutumnGame:
    """Thin wrapper around the pybind Autumn interpreter for one program."""

    def __init__(self, game, seed=0):
        from MARA.autumn_cpp.interpreter_module import Interpreter
        from MARA.autumn_cpp.autumnstdlib import autumnstdlib
        self.prog = open(os.path.join(TESTS, f"{game}.sexp")).read()
        self._stdlib = autumnstdlib
        self.itp = Interpreter()
        self.itp.set_verbose(False)
        with _suppress():
            self.itp.run_script(self.prog, self._stdlib, "", seed)
        d = json.loads(self.itp.render_all())
        self.grid_size = d["GRID_SIZE"]
        self.bg = self.itp.get_background() or "black"
        self.button_pos = {}
        for key in d:
            if key in ("buttonNext", "buttonReset"):
                for c in d[key]:
                    p = c["position"]
                    self.button_pos[key] = (p["x"], p["y"])

    def reseed(self, seed):
        with _suppress():
            self.itp.run_script(self.prog, self._stdlib, "", int(seed))

    def color_grid(self):
        d = json.loads(self.itp.render_all())
        gs = self.grid_size
        g = [[self.bg] * gs for _ in range(gs)]
        keys = [k for k in d if k != "GRID_SIZE"]
        for k in sorted(keys, key=lambda k: k in ("buttonNext", "buttonReset")):
            for c in d[k]:
                p = c["position"]
                x, y = p["x"], p["y"]
                if 0 <= x < gs and 0 <= y < gs:
                    g[y][x] = c.get("color", "black")
        return g

    def apply(self, action):
        with _suppress():
            t = action[0]
            if t == "noop":
                self.itp.step()
            elif t == "click":
                self.itp.click(int(action[1]), int(action[2])); self.itp.step()
            else:  # up/down/left/right
                getattr(self.itp, t)(); self.itp.step()

    def reset_board(self):
        if "buttonReset" in self.button_pos:
            self.apply(("click", *self.button_pos["buttonReset"]))


def _grid_to_idx(g, palette):
    gs = len(g)
    out = np.zeros((gs, gs), dtype=np.uint8)
    for y in range(gs):
        for x in range(gs):
            c = g[y][x]
            if c not in palette:
                palette[c] = len(palette)
            out[y, x] = palette[c]
    return out


def _seed_clusters(env, rng, gs, light=False):
    """Place dense blobs (rich local-neighborhood coverage for CA rules).
    light=True places a single small blob (cheap mid-rollout re-seed)."""
    n_blobs = 1 if light else int(rng.integers(1, 6))
    for _ in range(n_blobs):
        cx0, cy0 = int(rng.integers(gs)), int(rng.integers(gs))
        rad = int(rng.integers(1, 3)) if light else int(rng.integers(1, 4))
        dens = rng.uniform(0.3, 0.9)
        for dy in range(-rad, rad + 1):
            for dx in range(-rad, rad + 1):
                x, y = cx0 + dx, cy0 + dy
                if 0 <= x < gs and 0 <= y < gs and rng.random() < dens:
                    env.apply(("click", x, y))


def collect(game, rollouts, rollout_len, seed, profile="ca", arrows=True,
            reseed_every=12, keep_prev=False):
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    palette = {env.bg: 0}  # index 0 = the game's true background color
    bnext = env.button_pos.get("buttonNext")
    breset = env.button_pos.get("buttonReset")
    arrow_acts = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]

    seen = {}
    visited = set()
    t0 = time.time()

    def grid_idx():
        return _grid_to_idx(env.color_grid(), palette)

    arrow_w = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def sample_action(density):
        menu, w = [], []
        if profile == "spamclick":
            # adversarial: heavy clicking at varied positions (empty/occupied/edges) +
            # occasional arrows — covers spawn/placement dynamics the agent profile misses
            # (fixes click-spawn divergences; see DIVERGENCES.md).
            for _ in range(3):
                menu.append(("click", int(rng.integers(gs)), int(rng.integers(gs)))); w.append(0.25)
            menu.append(("click", int(rng.integers(gs)), 0)); w.append(0.10)   # top row (buttons live here)
            for a in arrow_acts:
                menu.append(a); w.append(0.04)
            menu.append(("noop", -1, -1)); w.append(0.05)
            w = np.array(w); w /= w.sum()
            return menu[rng.choice(len(menu), p=w)]
        if profile == "agent":
            # arrow/agent-driven (e.g. mario): no seeding, arrows + occasional click(shoot)
            menu.append(("noop", -1, -1)); w.append(0.15)
            for a in arrow_acts:
                menu.append(a); w.append(arrow_w[a[0]])
            menu.append(("click", int(rng.integers(gs)), int(rng.integers(gs)))); w.append(0.10)
        else:
            menu.append(("noop", -1, -1)); w.append(0.10)
            if arrows:
                for a in arrow_acts:
                    menu.append(a); w.append(0.02)
            menu.append(("click", int(rng.integers(gs)), int(rng.integers(gs)))); w.append(0.40)
            dense = density > 0.06
            if bnext is not None:
                menu.append(("click", *bnext)); w.append(0.30 * (2.5 if dense else 1.0))
            if breset is not None:
                menu.append(("click", *breset)); w.append(0.15 * (2.8 if dense else 1.0))
        w = np.array(w); w /= w.sum()
        return menu[rng.choice(len(menu), p=w)]

    def record(prev_s, s_, act, ns_):
        key = (prev_s.tobytes(), s_.tobytes(), act) if keep_prev else (s_.tobytes(), act)
        if key not in seen:
            ai_, cx_, cy_ = action_to_fields(act)
            seen[key] = (prev_s, s_, ai_, cx_, cy_, ns_)

    for r in range(rollouts):
        env.reseed(int(rng.integers(1 << 30)))
        if profile == "ca":
            env.reset_board()
            _seed_clusters(env, rng, gs)
            # Record a guaranteed dense-source reset from a UNIFORM-density board
            # (matches a hand-drawn board; the walk thins out fast and otherwise
            # starves dense-board clear coverage). Then re-seed for the walk.
            if breset is not None:
                env.reset_board()
                d = rng.uniform(0.1, 0.5)
                for yy in range(gs):
                    for xx in range(gs):
                        if rng.random() < d and (xx, yy) not in (bnext, breset):
                            env.apply(("click", xx, yy))
                s_dense = grid_idx()
                env.apply(("click", *breset))
                record(s_dense, s_dense, ("click", *breset), grid_idx())
                _seed_clusters(env, rng, gs)
        s = grid_idx()
        prev = s  # first frame has no predecessor -> treat as stationary
        for step in range(rollout_len):
            visited.add(s.tobytes())
            density = float((s != 0).mean())
            if profile == "ca" and reseed_every and step and step % reseed_every == 0:
                _seed_clusters(env, rng, gs, light=True)
                s = grid_idx(); prev = s
                density = float((s != 0).mean())
            act = sample_action(density)
            env.apply(act)
            ns = grid_idx()
            record(prev, s, act, ns)
            prev = s
            s = ns

    prev_states = np.stack([v[0] for v in seen.values()])
    states = np.stack([v[1] for v in seen.values()])
    ai = np.array([v[2] for v in seen.values()], dtype=np.uint8)
    cxs = np.array([v[3] for v in seen.values()], dtype=np.int16)
    cys = np.array([v[4] for v in seen.values()], dtype=np.int16)
    next_states = np.stack([v[5] for v in seen.values()])
    inv = [None] * len(palette)
    for name, i in palette.items():
        inv[i] = name

    from nca_wm.autumn.model import ACTION_TYPES
    dt = time.time() - t0
    n_changed = int((states != next_states).any(axis=(1, 2)).sum())
    dens = (states != 0).mean(axis=(1, 2))
    print(f"[{game}] {len(seen)} unique transitions / {len(visited)} unique states in {dt:.1f}s")
    print(f"  palette ({len(inv)}): {inv}")
    counts = {ACTION_TYPES[k]: int((ai == k).sum()) for k in np.unique(ai)}
    print(f"  action types: {counts}")
    print(f"  changing transitions: {n_changed} ({100*n_changed/len(seen):.1f}%)")
    print(f"  state density: min={dens.min():.3f} mean={dens.mean():.3f} max={dens.max():.3f} "
          f"| frac dense(>0.1): {(dens>0.1).mean():.2f}")
    if breset is not None:
        rsel = (ai == 1) & (cxs == breset[0]) & (cys == breset[1])
        if rsel.any():
            rd = dens[rsel]
            print(f"  reset transitions={int(rsel.sum())} dense-source(>0.1)={(rd>0.1).mean():.2f}")
    out = dict(states=states, next_states=next_states, action_type=ai,
               click_x=cxs, click_y=cys, palette=np.array(inv),
               grid_size=np.int32(gs), game=np.str_(game),
               button_pos_json=np.str_(json.dumps(env.button_pos)))
    if keep_prev:
        out["prev_states"] = prev_states
    return out


def collect_sequences(game, episodes, length, seed, profile="agent"):
    """Collect ORDERED episodes (from episode start, where hidden state is known)
    for recurrent BPTT training. Returns states (E, L+1, H, W) + actions (E, L, 3)."""
    rng = np.random.default_rng(seed)
    env = AutumnGame(game, seed=seed)
    gs = env.grid_size
    palette = {env.bg: 0}
    bnext = env.button_pos.get("buttonNext")
    breset = env.button_pos.get("buttonReset")
    arrow_acts = [("up", -1, -1), ("down", -1, -1), ("left", -1, -1), ("right", -1, -1)]
    arrow_w = {"up": 0.2, "down": 0.05, "left": 0.25, "right": 0.25}

    def sample_action():
        menu, w = [("noop", -1, -1)], [0.12]
        if profile == "agent":
            for a in arrow_acts:
                menu.append(a); w.append(arrow_w[a[0]])
            menu.append(("click", int(rng.integers(gs)), int(rng.integers(gs)))); w.append(0.20)
        else:
            menu.append(("click", int(rng.integers(gs)), int(rng.integers(gs)))); w.append(0.5)
            if bnext: menu.append(("click", *bnext)); w.append(0.25)
            if breset: menu.append(("click", *breset)); w.append(0.13)
        w = np.array(w); w /= w.sum()
        return menu[rng.choice(len(menu), p=w)]

    def grid_idx():
        return _grid_to_idx(env.color_grid(), palette)

    # --- Mario heuristic: walk under a coin (gravity lands Mario on the step =
    # collect), then FIRE IN BURSTS so the WM sees fire-then-empty -> decrement.
    # Fixes the recurrent model's over-firing (random data had ~0 multi-fire episodes).
    last_mario = [gs // 2, gs - 1]
    burst = [0]

    def mario_heuristic(grid):
        red = palette.get("red"); gold = palette.get("gold")
        if red is not None:
            ys, xs = np.where(grid == red)
            if len(xs):
                last_mario[0], last_mario[1] = int(xs[0]), int(ys[0])
        mx, my = last_mario
        if burst[0] > 0:                                # mid fire-burst
            burst[0] -= 1
            return ("click", int(rng.integers(gs)), int(rng.integers(gs)))
        if rng.random() < 0.30:                         # start a burst of 2-6 fires
            burst[0] = int(rng.integers(1, 6))
            return ("click", int(rng.integers(gs)), int(rng.integers(gs)))
        if gold is not None:
            cys, cxs = np.where(grid == gold)
            if len(cxs):                                # walk under nearest coin (gravity collects)
                i = int(np.argmin(np.abs(cxs - mx)))
                cx = int(cxs[i])
                if mx < cx: return ("right", -1, -1)
                if mx > cx: return ("left", -1, -1)
                return ("up", -1, -1)                   # aligned but coin above -> try jump
        return arrow_acts[rng.integers(4)]

    def snake_heuristic(grid):
        # steer the snake (green centroid) toward the food (pink) so it eats
        # frequently -> many random respawns (tests aleatoric uncertainty).
        green = palette.get("green"); pink = palette.get("pink")
        if green is None or pink is None:
            return arrow_acts[rng.integers(4)]
        gy, gx = np.where(grid == green); fy, fx = np.where(grid == pink)
        if not len(gx) or not len(fx):
            return arrow_acts[rng.integers(4)]
        dx = int(fx[0]) - int(gx.mean()); dy = int(fy[0]) - int(gy.mean())
        if abs(dx) >= abs(dy):
            return ("right", -1, -1) if dx > 0 else ("left", -1, -1)
        return ("down", -1, -1) if dy > 0 else ("up", -1, -1)

    # --- waterplug heuristic: random data places the DEFAULT mode (vessel) ~96% of
    # the time because the agent rarely presses the plug/water buttons. Cycle modes
    # explicitly — press a button, then place several cells in that mode — so the
    # WM sees button->mode->placement and can learn the hidden currentParticle.
    wp_btn = {0: (2, 0), 1: (5, 0), 2: (8, 0)}   # vessel / plug / water buttons
    wp_mode = [0]
    wp_place_left = [0]

    def waterplug_heuristic(grid):
        if wp_place_left[0] <= 0:                 # switch mode: press its button
            wp_mode[0] = int(rng.integers(3))
            wp_place_left[0] = int(rng.integers(3, 8))
            bx, by = wp_btn[wp_mode[0]]
            return ("click", bx, by)
        wp_place_left[0] -= 1
        if rng.random() < 0.2:                    # interleave noops -> water flow
            return ("noop", -1, -1)
        return ("click", int(rng.integers(gs)), int(rng.integers(1, gs)))  # avoid row-0 buttons

    # --- coins heuristic: the agent (red) moving ONTO a coin (gold) is a rare event
    # (~0.6% of random-walk transitions) that the WM underfits -> it predicts the gold
    # cell stays gold ("agent disappears behind the coin"). Steer the agent straight onto
    # the nearest coin every step so onto-coin (gold->red) transitions are dense, and fire
    # occasionally so the hidden numBullets counter (coin pickup -> can shoot) is exercised.
    coins_fire_noops = [0]   # after a fire, hold (noop) so the bullet travels up off the agent
    coins_wall_drive = [0, 0]  # [steps_left, dir_idx] -- pin the agent against a wall

    def coins_heuristic(grid):
        red = palette.get("red"); gold = palette.get("gold")
        # CRITICAL: mix in noop (agent must PERSIST when stationary -- a pure steer-to-coin
        # policy never shows the agent sitting still, so the WM learns "red always leaves its
        # cell on tick 1" and deletes a stationary agent) and random wandering (agent away
        # from coins) alongside the steer-onto-coin moves that cover the occlusion.
        if coins_fire_noops[0] > 0:                            # let the just-fired bullet travel
            coins_fire_noops[0] -= 1                           # up while the agent stays put -- a
            return ("noop", -1, -1)                            # bullet renders OVER the agent, so
            #                                                    when it moves off, the agent (red)
            #                                                    must reappear (2nd occlusion layer)
        if coins_wall_drive[0] > 0:                            # drive INTO and hold against a wall:
            coins_wall_drive[0] -= 1                           # covers boundary-clamp states (the
            return arrow_acts[coins_wall_drive[1]]             # agent pinned at an edge) that long
            #                                                    forced-direction free-play hits
        r = rng.random()
        if r < 0.10:                                           # start a wall-pin burst
            coins_wall_drive[0] = int(rng.integers(8, 16)); coins_wall_drive[1] = int(rng.integers(4))
            return arrow_acts[coins_wall_drive[1]]
        if r < 0.22:
            return ("noop", -1, -1)                            # persistence: agent stays put
        if r < 0.42:
            return arrow_acts[rng.integers(4)]                 # wander (not coin-directed)
        if red is None:
            return arrow_acts[rng.integers(4)]
        ys, xs = np.where(grid == red)
        if not len(xs):
            return arrow_acts[rng.integers(4)]
        mx, my = int(xs[0]), int(ys[0])
        if gold is not None and rng.random() < 0.20:          # spend a pickup: fire, then hold
            coins_fire_noops[0] = int(rng.integers(3, 7))     # so the bullet travels off-agent
            return ("click", mx, my)
        if gold is not None:
            cys, cxs = np.where(grid == gold)
            if len(cxs):                                       # head straight for nearest coin
                i = int(np.argmin(np.abs(cxs - mx) + np.abs(cys - my)))
                tx, ty = int(cxs[i]), int(cys[i])
                if mx < tx: return ("right", -1, -1)
                if mx > tx: return ("left", -1, -1)
                if my < ty: return ("down", -1, -1)
                if my > ty: return ("up", -1, -1)
        return arrow_acts[rng.integers(4)]                     # no coins left: wander

    # --- masters_logic heuristic: the core mechanic is a click-to-CYCLE peg
    # (col -> (col+1)%7: grey->red->green->blue->yellow->purple->orange) at the 4 guess
    # cells (3..6, row 1). Random/agent data almost never clicks the same peg twice, so the
    # cycle past red is never seen (palette is even missing blue/yellow/purple/orange) and
    # the WM predicts pegs never change. Click the 4 pegs hard to exercise the full cycle;
    # press enter (0,0) occasionally to submit (row-shift). Hints are aleatoric -> not the
    # target; keep enter rare so deterministic cycle transitions dominate.
    ml_pegs = [(3, 1), (4, 1), (5, 1), (6, 1)]

    def masters_logic_heuristic(grid):
        if rng.random() < 0.12:
            return ("click", 0, 0)                                # enter/submit
        px, py = ml_pegs[rng.integers(len(ml_pegs))]
        return ("click", px, py)

    # --- paint heuristic: hidden currColor (init "red", UP cycles red->gold->green->blue->
    # purple). The recurrent WM mispredicts WHICH color a click paints, especially the first
    # clicks of an episode (wrong cold-start prior at h=0). Interleave UP-cycles with bursts
    # of canvas paints so the model must track currColor across UP presses AND from h=0.
    pt_left = [0]

    def paint_heuristic(grid):
        if pt_left[0] <= 0:                                       # cycle color, then paint a burst
            pt_left[0] = int(rng.integers(2, 7))
            return ("up", -1, -1)
        pt_left[0] -= 1
        return ("click", int(rng.integers(gs)), int(rng.integers(gs)))

    heuristics = {"mario_heuristic": mario_heuristic, "snake_heuristic": snake_heuristic,
                  "waterplug_heuristic": waterplug_heuristic, "coins_heuristic": coins_heuristic,
                  "masters_logic_heuristic": masters_logic_heuristic, "paint_heuristic": paint_heuristic}

    all_states, all_actions, all_seeds = [], [], []
    t0 = time.time()
    for e in range(episodes):
        ep_seed = int(rng.integers(1 << 30))
        env.reseed(ep_seed)
        if profile == "ca":
            env.reset_board(); _seed_clusters(env, rng, gs)
        states = [grid_idx()]
        actions = []
        for t in range(length):
            act = heuristics[profile](states[-1]) if profile in heuristics else sample_action()
            env.apply(act)
            states.append(grid_idx())
            actions.append(list(action_to_fields(act)))
        all_states.append(np.stack(states))
        all_actions.append(np.array(actions, dtype=np.int16))
        all_seeds.append(ep_seed)
    states = np.stack(all_states).astype(np.uint8)        # (E, L+1, H, W)
    actions = np.stack(all_actions)                       # (E, L, 3)
    inv = [None] * len(palette)
    for name, i in palette.items():
        inv[i] = name
    print(f"[{game}] {episodes} episodes x {length} steps in {time.time()-t0:.1f}s | palette {inv}")
    # seeds enable replaying any episode state as a search-frontier seed (profile 'ca' reseeds
    # the board after reset, so those states are not pure-seed-replayable -> seed=-1 there)
    seeds = np.array([-1 if profile == "ca" else s for s in all_seeds], dtype=np.int64)
    return dict(states=states, actions=actions, palette=np.array(inv),
                grid_size=np.int32(gs), game=np.str_(game), seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="gameOfLife")
    ap.add_argument("--rollouts", type=int, default=300)
    ap.add_argument("--rollout_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="ca",
                    choices=["ca", "generic", "agent", "spamclick", "mario_heuristic", "snake_heuristic",
                             "waterplug_heuristic", "coins_heuristic", "masters_logic_heuristic",
                             "paint_heuristic"])
    ap.add_argument("--no_arrows", action="store_true")
    ap.add_argument("--reseed_every", type=int, default=12)
    ap.add_argument("--keep_prev", action="store_true",
                    help="store previous frame + dedup by (prev,state,action) for history models")
    ap.add_argument("--sequences", type=int, default=0,
                    help="collect N ordered episodes (for recurrent BPTT) instead of deduped transitions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.sequences:
        data = collect_sequences(args.game, args.sequences, args.rollout_len,
                                 args.seed, profile=args.profile)
    else:
        data = collect(args.game, args.rollouts, args.rollout_len, args.seed,
                       profile=args.profile, arrows=not args.no_arrows,
                       reseed_every=args.reseed_every, keep_prev=args.keep_prev)
    out = args.out or f"nca_wm/autumn/data/{args.game}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
