"""Information-gain-seeking BEAM SEARCH over the JAX NCA world model + adapter,
executed in the REAL PuzzleScript engine, with GIF rendering of the best rollout.

The planner maximizes CUMULATIVE information gain over a fixed-horizon rollout:
    IG(s,a) = E_{o~q0(s,a)}[ sum_realcells( log q1(o|s,a,o) - log q0(o|s,a) ) ]
where q0 = sigmoid(base NCAWorldModel(s,a)) and q1 = sigmoid(adapter(s,a,o)).

Beam search (a bounded best-first): keep the top-K action sequences by cumulative
IG. The engine is deterministic, so a beam path == an action sequence and every
leaf is reproduced by replaying the actions from a fresh engine. At each depth we
re-simulate every beam path to its leaf, read the obs in the model's channel
order, score all 6 actions by IG (n_samples Bernoulli draws of o), expand, and
keep the top-K children. We do this SEPARATELY for `mario` and `mario_breakable`
(the model can't tell the worlds apart, so it sees high IG at "jump under a
breakable platform" in BOTH worlds), then render the REAL engine frames of each
world's best (max cumulative IG) rollout to a GIF.

    export PATH=$HOME/.nvm/versions/node/v24.15.0/bin:$PATH
    .venv/bin/python -u -m nca_wm.active_learning.ig_search

Channel order (id_dict, identity perm): floor=5, step=6, player=7, dead=14.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import jax
import jax.numpy as jnp

from nca_wm.models import NCAWorldModel, N_ACTIONS
from nca_wm.adapter import AdapterHead
from nca_wm.state_ops import _multihot_to_objects
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine, _read_padded

CMAX = 20                       # model's trained channel dim (== mario n_obj)
BASE = "nca_wm/logs/mario_uncond_fulldata"
ADAPT = "nca_wm/logs/adapter_mario_skip/adapter_params.pkl"
ACTIONS = MB.ACTIONS            # ["UP","LEFT","DOWN","RIGHT","ACTION","TICK"]
STEP_CH, PLAYER_CH, DEAD_CH, FLOOR_CH = 6, 7, 14, 5
EPS = 1e-6
PERM = list(range(CMAX))        # identity channel map (in-distribution objective)


# ---------------------------------------------------------------------------
# Model loading (lifted from active_learning.mario_nca_serve)
# ---------------------------------------------------------------------------
def load_base():
    params = pickle.load(open(f"{BASE}/params_best.pkl", "rb"))
    cfg = json.load(open(f"{BASE}/config.json"))
    gi = pickle.load(open(f"{BASE}/game_infos.pkl", "rb"))
    max_C = max(g["n_objs"] for g in gi)
    assert max_C == CMAX, f"max_C {max_C} != CMAX {CMAX}"
    model = NCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        input_skip=cfg["input_skip"], n_repeats=cfg["n_nca_repeats"],
        history=cfg["history"], axis_pool=cfg["axis_pool"],
        axis_cummax=cfg["axis_cummax"], global_pool=cfg["global_pool"])

    @jax.jit
    def q0(s, a):
        logits = model.apply(params, s, jax.nn.one_hot(a, N_ACTIONS))[0]
        return jax.nn.sigmoid(logits)

    print(f"[base] loaded {BASE} (n_hid={cfg['n_hid']} n_steps={cfg['n_nca_steps']} "
          f"C={max_C})", flush=True)
    return q0


def load_adapter():
    ad = pickle.load(open(ADAPT, "rb"))
    c = ad["cfg"]
    adapter = AdapterHead(n_hid=c["n_hid"], n_steps=c["n_steps"], n_out=c["n_out"])
    aparams = ad["params"]

    @jax.jit
    def q1(s, a, o):
        return jax.nn.sigmoid(adapter.apply(aparams, s, jax.nn.one_hot(a, N_ACTIONS), o))

    print(f"[adapter] loaded {ADAPT} (n_hid={c['n_hid']} n_steps={c['n_steps']} "
          f"C={c['n_out']})", flush=True)
    return q1


def build_render_backends():
    """Compile each world with a sprite-capable C++ backend (for rendering)."""
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    MB._CACHE.mkdir(exist_ok=True)
    gc._set_materialize_dir(MB._CACHE / "_scratch")
    (MB._CACHE / "_scratch").mkdir(exist_ok=True)
    parser = init_ps_lark_parser()
    games = MB.build_worlds()
    backends = {}
    for name, rel in MB.WORLDS:
        gc._materialize_game(name, (MB.ROOT / rel).read_text())
        b = CppPuzzleScriptBackend()
        b.compile_game(parser, name)
        b.cpp_engine.load_level(0)
        backends[name] = b
    return {g.gist: g for g in games}, backends


# ---------------------------------------------------------------------------
# Engine helpers (deterministic; a path == an action sequence)
# ---------------------------------------------------------------------------
def make_engine(game):
    eng = _engine(game.json_str, 0)
    eng.set_track_rules_fired(True)
    eng.seed_rng("0")
    return eng


def engine_step(eng, a):
    eng.clear_rules_fired()
    eng.process_input(a)                       # action idx == engine input id (5 = tick)
    k = 0
    while eng.is_againing() and k < 50:
        eng.process_input(-1)
        k += 1


def read_obs(eng, game):
    # (CMAX,H,W) float multihot, channels in trained (id_dict) order.
    return _read_padded(eng, game.n_obj, PERM, CMAX, game.H, game.W)


def stepcount(eng):
    g = MB._grid(eng)
    return int(((g >> STEP_CH) & 1).sum())


def replay(game, actions):
    """Fresh engine, replay the action sequence; return (eng, leaf_obs, stuck)."""
    eng = make_engine(game)
    obs = read_obs(eng, game)
    prev = None
    for a in actions:
        prev = obs
        engine_step(eng, a)
        obs = read_obs(eng, game)
    stuck = prev is not None and np.array_equal(prev, obs)
    return eng, obs, stuck


def render_frame(eng, game, backend):
    obs = read_obs(eng, game)
    crop = (obs[:game.n_obj, :game.H, :game.W] > 0.5).astype(np.uint8)
    objs = _multihot_to_objects(crop)
    return np.asarray(backend.render_frame_from_objects(objs, game.W, game.H))


def is_terminal(eng, obs, stuck):
    dead = float(obs[DEAD_CH].sum()) > 0
    return bool(dead) or bool(eng.is_winning()) or bool(stuck)


# ---------------------------------------------------------------------------
# Information gain
# ---------------------------------------------------------------------------
def ig_batch(obs_list, q0, q1, n_samples, seed=0):
    """IG for all 6 actions from each obs in obs_list. Returns (N,6) array."""
    N = len(obs_list)
    C, H, W = obs_list[0].shape
    S = np.empty((N * N_ACTIONS, C, H, W), dtype=np.float32)
    for i, o in enumerate(obs_list):
        S[i * N_ACTIONS:(i + 1) * N_ACTIONS] = o[None]
    A = np.tile(np.arange(N_ACTIONS), N)
    Sj, Aj = jnp.asarray(S), jnp.asarray(A)
    P0 = np.asarray(q0(Sj, Aj)).clip(EPS, 1 - EPS)         # (N*6,C,H,W)
    m = (S.sum(1, keepdims=True) > 0)                      # real (non-pad) cells
    rng = np.random.default_rng(seed)
    ig = np.zeros(N * N_ACTIONS)
    for _ in range(n_samples):
        o = (rng.random(P0.shape) < P0).astype(np.float32)
        P1 = np.asarray(q1(Sj, Aj, jnp.asarray(o))).clip(EPS, 1 - EPS)
        lq0 = o * np.log(P0) + (1 - o) * np.log(1 - P0)
        lq1 = o * np.log(P1) + (1 - o) * np.log(1 - P1)
        ig += ((lq1 - lq0) * m).sum(axis=(1, 2, 3))
    ig /= n_samples
    return ig.reshape(N, N_ACTIONS)


def ig_action(obs, a, q0, q1, n_samples, n_seeds=8):
    """Stabilized IG for a single action from a single obs (per-step reporting).

    The per-cell IG estimator is a high-variance Bernoulli average: a single seed
    occasionally produces spurious ~0.1-0.3 spikes on genuinely-uninformative
    states (verified by multi-seed probing). The beam SEARCH runs on the cheap
    single-seed estimate (whose variance doubles as exploration noise), but for an
    HONEST per-step readout we average over several seeds, which collapses the
    spurious intermediate IG toward 0 and leaves the real jump-under-platform
    signal (~0.3-0.6) intact."""
    return float(np.mean([ig_batch([obs], q0, q1, n_samples, s)[0, a]
                          for s in range(n_seeds)]))


# ---------------------------------------------------------------------------
# Beam search (bounded best-first; maximize cumulative IG)
# ---------------------------------------------------------------------------
def beam_search(game, q0, q1, K=12, horizon=30, n_samples=48, verbose=True):
    # node = (actions_tuple, cum_ig, per_step_ig_list)
    root = ((), 0.0, [])
    beam = [root]
    best = root
    for depth in range(horizon):
        expand, obs_list = [], []
        for node in beam:
            actions, cum, _ = node
            eng, obs, stuck = replay(game, actions)
            if cum > best[1]:
                best = node
            if is_terminal(eng, obs, stuck):
                continue
            expand.append(node)
            obs_list.append(obs)
        if not obs_list:
            break
        igs_mat = ig_batch(obs_list, q0, q1, n_samples)     # (N,6)
        children = []
        for node, igs in zip(expand, igs_mat):
            actions, cum, igl = node
            for a in range(N_ACTIONS):
                children.append((actions + (a,), cum + float(igs[a]), igl + [float(igs[a])]))
        children.sort(key=lambda x: -x[1])
        beam = children[:K]
        if verbose:
            top = beam[0]
            print(f"  [{game.gist}] depth {depth+1:2d}/{horizon}  "
                  f"frontier={len(children):4d}  best_cumIG={top[1]:8.3f}  "
                  f"last_action={ACTIONS[top[0][-1]]}", flush=True)
    for node in beam:
        if node[1] > best[1]:
            best = node
    return best


# ---------------------------------------------------------------------------
# Render + report the best rollout in the REAL engine
# ---------------------------------------------------------------------------
def render_rollout(game, backend, actions, q0, q1, n_samples):
    eng = make_engine(game)
    frames = [render_frame(eng, game, backend)]
    sc = stepcount(eng)
    steps_info = []
    for i, a in enumerate(actions):
        obs_before = read_obs(eng, game)
        ig = ig_action(obs_before, a, q0, q1, n_samples)
        sc_before = sc
        engine_step(eng, a)
        sc = stepcount(eng)
        broke = sc < sc_before
        frames.append(render_frame(eng, game, backend))
        steps_info.append(dict(i=i, a=a, ig=ig, broke=broke,
                               dead=float(read_obs(eng, game)[DEAD_CH].sum()) > 0,
                               won=bool(eng.is_winning())))
    return frames, steps_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=12)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--n_samples", type=int, default=48)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--out", default="/tmp/ig_rollouts")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q0 = load_base()
    q1 = load_adapter()
    games, backends = build_render_backends()

    summary = {}
    for world in ("mario", "mario_breakable"):
        game, backend = games[world], backends[world]
        print(f"\n=== BEAM SEARCH: {world} "
              f"(K={args.K} horizon={args.horizon} n_samples={args.n_samples}) ===",
              flush=True)
        best = beam_search(game, q0, q1, K=args.K, horizon=args.horizon,
                           n_samples=args.n_samples)
        actions, cum_ig, _ = best
        print(f"\n--- BEST ROLLOUT [{world}]  cumulative IG = {cum_ig:.3f}  "
              f"(beam search objective, single-seed est.)  len={len(actions)} ---\n"
              f"    per-step IG below is the STABILIZED (multi-seed) readout; "
              f"intermediate steps are ~0 (noise), the jump is the real signal:",
              flush=True)

        frames, steps_info = render_rollout(game, backend, actions, q0, q1, args.n_samples)
        n_break = 0
        for s in steps_info:
            flag = ""
            if s["broke"]:
                n_break += 1
                flag = "  <-- PLATFORM BROKE (step present->absent)"
            if s["dead"]:
                flag += "  [DEAD]"
            if s["won"]:
                flag += "  [WON]"
            print(f"  step {s['i']:2d}: {ACTIONS[s['a']]:6s}  IG={s['ig']:8.3f}{flag}",
                  flush=True)
        print(f"  total platform breaks in best rollout: {n_break}", flush=True)

        gif_path = out / f"{world}_ig.gif"
        imageio.mimsave(gif_path, frames, fps=args.fps)
        print(f"  saved GIF -> {gif_path}  ({len(frames)} frames)", flush=True)

        # a couple of key still frames
        imageio.imwrite(out / f"{world}_first.png", frames[0])
        imageio.imwrite(out / f"{world}_last.png", frames[-1])
        brk_idx = next((s["i"] for s in steps_info if s["broke"]), None)
        if brk_idx is not None:
            imageio.imwrite(out / f"{world}_break.png", frames[brk_idx + 1])

        summary[world] = dict(cum_ig=cum_ig, n_actions=len(actions),
                              n_break=n_break, gif=str(gif_path),
                              actions=[ACTIONS[a] for a in actions])

    print("\n=== SUMMARY ===", flush=True)
    for world, s in summary.items():
        print(f"  {world:16s}  cumIG={s['cum_ig']:8.3f}  steps={s['n_actions']:2d}  "
              f"breaks={s['n_break']}  gif={s['gif']}", flush=True)


if __name__ == "__main__":
    main()
