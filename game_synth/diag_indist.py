"""In-distribution test: does the WM learning LIVE games drive their excess-loss
below that of DEAD games (making selection prefer dead games)?

Build a fixed pool of dead (activation=0) + live (activation>0) games, train ONE
CodeCondWorldModel on them, and track per-group excess_loss across training. If
the collapse is "the WM learns too effectively," we expect: live games start with
HIGH excess (unlearned rules) but, as the WM masters them, their excess drops
BELOW the dead games' excess -- whose floor is set by norule_NLL~0 (the no-rule
baseline is exactly right for a dead game, so excess = residual model error >= ~0
and cannot go negative). That inversion is the selection attractor.

    .venv/bin/python -m game_synth.diag_indist --n-dead 40 --n-live 25 --updates 3000
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth.code_cond_train import CodeCondWorldModel, mixture_nll, pad_tokens
from game_synth.engine_train import OBJ6, _engine, norule_next, sample_traj
from game_synth.gp_evolve import train_batch
from game_synth.render_games import coverage_walk
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024


def load_game(name, code, parser):
    from puzzlescript_cpp import CppPuzzleScriptBackend
    gc._materialize_game(name, code)
    js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
    e = _engine(js)
    if np.asarray(e.get_objects_2d()).shape != (8, 8, 1):
        return None
    idd = [s.lower() for s in e.get_id_dict()]
    if any(s not in OBJ6 for s in idd):
        return None
    tree, ids = get_game_tree_from_js(parser, name)
    tok = tokenize_game(tree, ids, encode_sprites=False, include_levels=False)[:256]
    if not tok or max(tok) >= VOCAB:
        return None
    et = _engine(js); et.set_track_rules_fired(True)
    _, fired = coverage_walk(et, idd, 5, 16, random.Random(0))
    return {"name": name, "json": js, "idd": idd, "tok": tok, "activated": len(fired)}


@torch.no_grad()
def excess(model, e, device, n=8, seed=0):
    rng = random.Random(seed)
    jsons = {e["name"]: (e["json"], e["idd"])}
    tk, mask = pad_tokens([e["tok"]], device)
    z1 = torch.zeros(1, 1, device=device)
    m, nr = [], []
    for _ in range(n):
        o, a, _ = sample_traj(jsons, e["name"], 5, rng)
        for t in range(len(a)):
            s = torch.from_numpy(o[t][None]).to(device)
            nx = torch.from_numpy(o[t + 1][None]).to(device)
            lo, lp = model.logits(s, torch.tensor([a[t]], device=device), tk, mask)
            m.append(mixture_nll(lo, lp, nx).item())
            nr.append(mixture_nll((norule_next(s, int(a[t])) * 12 - 6)[:, None], z1, nx).item())
    return float(np.mean(m)) - float(np.mean(nr)), float(np.mean(m)), float(np.mean(nr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="game_synth/dataset/games/*.txt")
    ap.add_argument("--n-dead", type=int, default=40)
    ap.add_argument("--n-live", type=int, default=25)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)

    out = _REPO / "game_synth" / "diag_out"; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()

    paths = sorted(glob.glob(str(_REPO / args.glob)))
    random.Random(args.seed).shuffle(paths)
    dead, live = [], []
    for p in paths:
        if len(dead) >= args.n_dead and len(live) >= args.n_live:
            break
        try:
            e = load_game(Path(p).stem, Path(p).read_text(errors="ignore"), parser)
        except Exception:
            e = None
        if e is None:
            continue
        if e["activated"] == 0 and len(dead) < args.n_dead:
            dead.append(e)
        elif e["activated"] > 0 and len(live) < args.n_live:
            live.append(e)
    pop = dead + live
    print(f"pool: {len(dead)} dead + {len(live)} live "
          f"(live act mean {np.mean([e['activated'] for e in live]):.1f})", flush=True)

    model = CodeCondWorldModel(vocab=VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(args.seed)

    def snapshot(step):
        model.eval()
        de = np.array([excess(model, e, device)[0] for e in dead])
        li = np.array([excess(model, e, device)[0] for e in live])
        print(f"  step {step:5d} | excess  dead {de.mean():+7.3f}  live {li.mean():+7.3f}  "
              f"| live>dead? {'LIVE WINS' if li.mean()>de.mean() else 'DEAD WINS (attractor!)'} "
              f"| frac live with excess<dead-mean: {(li < de.mean()).mean():.0%}", flush=True)
        return de.mean(), li.mean()

    print("=== excess_loss (GP fitness) by group, across training ===", flush=True)
    traj = [(0, *snapshot(0))]
    checkpoints = sorted(set([200, 500, 1000, 2000, args.updates]))
    model.train()
    nxt = 0
    for step in range(1, args.updates + 1):
        loss = train_batch(model, pop, device, rng, bs=48)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in checkpoints:
            traj.append((step, *snapshot(step)))
            model.train()

    # final detailed components
    model.eval()
    dde = np.array([excess(model, e, device) for e in dead])
    lle = np.array([excess(model, e, device) for e in live])
    print(f"\n=== FINAL (step {args.updates}) components ===", flush=True)
    print(f"  dead: excess {dde[:,0].mean():+.3f}  model_NLL {dde[:,1].mean():.3f}  norule_NLL {dde[:,2].mean():.3f}")
    print(f"  live: excess {lle[:,0].mean():+.3f}  model_NLL {lle[:,1].mean():.3f}  norule_NLL {lle[:,2].mean():.3f}")
    print(f"  -> under pure WM-loss selection, the higher-excess group is preferred: "
          f"{'DEAD' if dde[:,0].mean()>lle[:,0].mean() else 'LIVE'}", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        steps = [t[0] for t in traj]
        ax.plot(steps, [t[1] for t in traj], "-o", color="tab:red", label="dead games (act=0)")
        ax.plot(steps, [t[2] for t in traj], "-o", color="tab:green", label="live games (act>0)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("WM training updates", fontsize=13)
        ax.set_ylabel("excess loss (GP fitness)", fontsize=13)
        ax.set_title("WM learns live games -> their fitness falls below dead games", fontsize=13)
        ax.legend(fontsize=12); ax.grid(alpha=0.3)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(_REPO / "nca_wm" / "figures" / f"excess_attractor.{e}",
                        dpi=140, bbox_inches="tight")
        print("saved nca_wm/figures/excess_attractor.png", flush=True)
    except Exception as ex:
        print(f"plot skipped: {ex}", flush=True)


if __name__ == "__main__":
    main()
