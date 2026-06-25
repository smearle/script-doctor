"""Neural game engine trained on GENERATED games (end-to-end microcosm).

Closes the loop: the synthesized GP games (shared Player/Wall/ObjA/ObjB/ObjC
vocab, fixed 8x8) are a multi-game distribution; we train ONE NCA-belief world
model to simulate them. Each game is a "world"; the spatial belief state infers
that game's local rewrite rules in-context from observed transitions. Eval on
HELD-OUT games shows (a) the model predicts their transitions far better than an
identity baseline, and (b) in-context improvement — prediction sharpens as the
belief observes more of a held-out game's transitions.

    .venv/bin/python -u -m game_synth.engine_train --updates 6000
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel
from nca_wm.active_learning.nca_belief_train import roll_loss

OBJ6 = ["background", "wall", "player", "obja", "objb", "objc"]
B2I = {n: i for i, n in enumerate(OBJ6)}
N_ACT = 5  # up,left,down,right,action


def compile_games(paths, parser, want_h=8, want_w=8, limit=None):
    """Compile generated games; keep 8x8, vocab subset of OBJ6. Returns {name: json}."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    gc._set_materialize_dir(_REPO / "game_synth" / "_engine_scratch")
    (_REPO / "game_synth" / "_engine_scratch").mkdir(parents=True, exist_ok=True)
    jsons, ok = {}, 0
    for p in paths:
        if limit and ok >= limit:
            break
        name = Path(p).stem
        code = Path(p).read_text(errors="ignore")
        try:
            gc._materialize_game(name, code)
            js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
            e = Engine(); e.load_from_json(js); e.load_level(0)
            idd = [s.lower() for s in e.get_id_dict()]
            if any(n not in OBJ6 for n in idd):
                continue
            a = np.asarray(e.get_objects_2d())
            if a.shape[0] != want_w or a.shape[1] != want_h or a.shape[2] != 1:
                continue
            jsons[name] = (js, idd)
            ok += 1
        except Exception:
            continue
    return jsons


def _engine(js):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine(); e.load_from_json(js); e.load_level(0)
    return e


def grid_of(engine, idd) -> np.ndarray:
    """(6,H,W) multihot in canonical OBJ6 order."""
    a = np.asarray(engine.get_objects_2d())[:, :, 0]      # (W,H) bitmask over engine ids
    W, H = a.shape
    g = np.zeros((len(OBJ6), H, W), np.float32)
    for ei, nm in enumerate(idd):
        if nm in B2I:
            mask = (a >> ei) & 1                          # (W,H)
            g[B2I[nm]] = mask.T                           # -> (H,W)
    return g


def _step(engine, a, seed, max_again=40):
    engine.seed_rng(seed); engine.process_input(a)
    n = 0
    while engine.is_againing() and n < max_again:
        engine.process_input(-1); n += 1


def sample_traj(jsons, name, T, rng):
    js, idd = jsons[name]
    e = _engine(js)
    grids = [grid_of(e, idd)]; acts = []; resamps = []
    obs = grids[0]
    for _ in range(T):
        a = rng.randrange(N_ACT)
        s1, s2 = str(rng.getrandbits(40)), str(rng.getrandbits(40))
        bak = e.backup_level()
        _step(e, a, s1); o1 = grid_of(e, idd)
        e.restore_level(bak); _step(e, a, s2); o2 = grid_of(e, idd)
        e.restore_level(bak); _step(e, a, s1)
        grids.append(o1); acts.append(a); resamps.append(o2)
    return np.stack(grids), np.asarray(acts, np.int64), np.stack(resamps)


def batch(jsons, names, n, T, rng):
    O, A, R = [], [], []
    for _ in range(n):
        o, a, r = sample_traj(jsons, rng.choice(names), T, rng)
        O.append(o); A.append(a); R.append(r)
    return (torch.from_numpy(np.stack(O)), torch.from_numpy(np.stack(A)),
            torch.from_numpy(np.stack(R)))


_DELTA = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}  # up,left,down,right (action=4: none)


def norule_next(grid: torch.Tensor, action: int) -> torch.Tensor:
    """Default PuzzleScript dynamics with NO custom rules: the player moves one
    cell in the action direction unless blocked (all objects share one collision
    layer here, so any wall/obj/player blocks). Everything else unchanged.
    grid: (1,6,H,W). Returns predicted next grid."""
    g = grid.clone()
    pl = B2I["player"]
    if action == 4:
        return g
    dy, dx = _DELTA[action]
    H, W = g.shape[-2:]
    for (y, x) in (g[0, pl] > 0.5).nonzero():
        y, x = int(y), int(x)
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and not (g[0, 1:, ny, nx] > 0.5).any():
            g[0, pl, y, x] = 0.0
            g[0, pl, ny, nx] = 1.0
    return g


@torch.no_grad()
def eval_in_context(model, jsons, names, device, n_per=4, T=8, seed=7):
    """Per-step q0 NLL on held-out games (by belief-step) + identity & no-rule NLL."""
    rng = random.Random(seed)
    step_nll = [[] for _ in range(T)]
    ident, norule = [], []
    z1 = torch.zeros(1, 1, device=device)
    for name in names:
        for _ in range(n_per):
            o, a, _ = sample_traj(jsons, name, T, rng)
            o = torch.from_numpy(o[None]).to(device); a = torch.from_numpy(a[None]).to(device)
            B = model.init_belief(o[:, 0])
            for t in range(T):
                tgt = o[:, t + 1]
                l0, p0 = model.q0_logits(B, a[:, t])
                step_nll[t].append(model.mixture_nll(l0, p0, tgt).item())
                # identity: predict next == current; no-rule: default movement only
                ident.append(model.mixture_nll((o[:, t] * 12 - 6)[:, None], z1, tgt).item())
                nr = norule_next(o[:, t], int(a[0, t].item()))
                norule.append(model.mixture_nll((nr * 12 - 6)[:, None], z1, tgt).item())
                B = model.update_belief(B, tgt, a[:, t])
    return ([float(np.mean(s)) for s in step_nll],
            float(np.mean(ident)), float(np.mean(norule)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-glob", default="game_synth/fixed_s*/games/*.txt")
    ap.add_argument("--max-games", type=int, default=250)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    paths = sorted(glob.glob(str(_REPO / args.games_glob)))
    random.Random(args.seed).shuffle(paths)
    print(f"compiling up to {args.max_games} generated games...", flush=True)
    jsons = compile_games(paths, parser, limit=args.max_games)
    names = list(jsons.keys())
    random.Random(args.seed).shuffle(names)
    holdout = names[:args.holdout]; train = names[args.holdout:]
    print(f"usable games: {len(names)} | train {len(train)} | holdout {len(holdout)}", flush=True)

    model = NCABeliefModel(BeliefConfig(n_obj=len(OBJ6), n_act=N_ACT)).to(device)
    print(f"belief engine params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed)
    t0 = time.time(); model.train()
    for step in range(args.updates):
        o, a, r = batch(jsons, train, args.batch_size, args.T, rng)
        o, a, r = o.to(device), a.to(device), r.to(device)
        loss = roll_loss(model, o, a, r, usage_w=0.02)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}",
                  flush=True)

    model.eval()
    for label, gset in [("TRAINING games (held-out trajectories)", train[:args.holdout]),
                        ("HELD-OUT games (unseen rules; generalization)", holdout)]:
        step_nll, ident, norule = eval_in_context(model, jsons, gset, device, T=args.T)
        print(f"\n=== {label}: per-belief-step q0 NLL (lower=better) ===", flush=True)
        for t, v in enumerate(step_nll):
            print(f"  after {t} observed transitions: NLL {v:.4f}", flush=True)
        print(f"  baselines: identity(nothing-moves) {ident:.4f} | "
              f"no-rule(default movement+collision) {norule:.4f}", flush=True)
        beat = "BEATS" if step_nll[-1] < norule else "ABOVE"
        print(f"  -> model in-context {step_nll[0]:.2f}->{step_nll[-1]:.2f}; "
              f"{beat} the no-rule baseline ({norule:.2f}) "
              f"[= learned rule effects beyond movement]", flush=True)
    torch.save({"model_state": model.state_dict(),
                "cfg": dict(n_obj=len(OBJ6), n_act=N_ACT)},
               "game_synth/neural_engine.pt")
    print("saved game_synth/neural_engine.pt", flush=True)


if __name__ == "__main__":
    main()
