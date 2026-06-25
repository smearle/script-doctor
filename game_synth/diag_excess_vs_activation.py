"""Diagnose WHY WM-excess-loss selection collapses to dead games.

excess_loss(g) = model_NLL(g) - norule_NLL(g) is the GP fitness. If the collapse
were because "the WM learns everything too well," excess would be ~0 everywhere
and uncorrelated with mechanics. If instead the metric is mis-specified, dead
games (activation 0) should retain HIGHER excess than mechanical games -- because
for a dead game the no-rule baseline is exactly correct (norule_NLL ~ 0) so any
residual WM movement-error makes excess positive, while a learned mechanical game
has model_NLL < norule_NLL (excess negative). We measure model_NLL, norule_NLL,
excess, and activation per game and correlate.

    .venv/bin/python -m game_synth.diag_excess_vs_activation \
        --wm game_synth/evolve_w0/wm.pt --glob 'game_synth/dataset/games/*.txt' --n 200
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth.code_cond_train import CodeCondWorldModel, mixture_nll, pad_tokens
from game_synth.engine_train import OBJ6, _engine, norule_next, sample_traj
from game_synth.render_games import coverage_walk
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024
OBJS = ["ObjA", "ObjB", "ObjC"]


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
    return js, idd, tok, len(fired)


@torch.no_grad()
def components(model, name, js, idd, tok, device, n=8, seed=0):
    """Return (model_NLL, norule_NLL) averaged over sampled transitions."""
    rng = random.Random(seed)
    jsons = {name: (js, idd)}
    tk, mask = pad_tokens([tok], device)
    z1 = torch.zeros(1, 1, device=device)
    m, nr = [], []
    for _ in range(n):
        o, a, _ = sample_traj(jsons, name, 5, rng)
        for t in range(len(a)):
            s = torch.from_numpy(o[t][None]).to(device)
            nx = torch.from_numpy(o[t + 1][None]).to(device)
            lo, lp = model.logits(s, torch.tensor([a[t]], device=device), tk, mask)
            m.append(mixture_nll(lo, lp, nx).item())
            nr.append(mixture_nll((norule_next(s, int(a[t])) * 12 - 6)[:, None], z1, nx).item())
    return float(np.mean(m)), float(np.mean(nr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="game_synth/evolve_w0/wm.pt")
    ap.add_argument("--glob", default="game_synth/dataset/games/*.txt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--tag", default="w0")
    args = ap.parse_args()
    device = torch.device(args.device)

    out = _REPO / "game_synth" / "diag_out"; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()

    model = CodeCondWorldModel(vocab=VOCAB).to(device)
    sd = torch.load(_REPO / args.wm, map_location=device)
    model.load_state_dict(sd["model_state"]); model.eval()
    print(f"loaded WM {args.wm}", flush=True)

    paths = sorted(glob.glob(str(_REPO / args.glob)))
    random.Random(args.seed).shuffle(paths)
    rows = []
    for p in paths:
        if len(rows) >= args.n:
            break
        try:
            r = load_game(Path(p).stem, Path(p).read_text(errors="ignore"), parser)
            if r is None:
                continue
            js, idd, tok, act = r
            mnll, nrnll = components(model, Path(p).stem, js, idd, tok, device)
            rows.append((act, mnll, nrnll, mnll - nrnll))
        except Exception:
            continue
    print(f"measured {len(rows)} games", flush=True)
    act = np.array([r[0] for r in rows]); mnll = np.array([r[1] for r in rows])
    nrnll = np.array([r[2] for r in rows]); exc = np.array([r[3] for r in rows])

    dead = act == 0; live = act > 0
    print(f"\n=== {args.tag}: dead (act=0, n={dead.sum()}) vs live (act>0, n={live.sum()}) ===")
    print(f"  excess_loss   dead {exc[dead].mean():+.3f}   live {exc[live].mean():+.3f}")
    print(f"  model_NLL     dead {mnll[dead].mean():.3f}   live {mnll[live].mean():.3f}")
    print(f"  norule_NLL    dead {nrnll[dead].mean():.3f}   live {nrnll[live].mean():.3f}")
    if len(rows) > 2:
        print(f"  corr(excess, activation) = {np.corrcoef(exc, act)[0,1]:+.3f}")
    print(f"  fraction with excess>0 (would be 'selected'): "
          f"dead {(exc[dead]>0).mean():.0%}  live {(exc[live]>0).mean():.0%}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(act + np.random.uniform(-0.15, 0.15, len(act)), exc, alpha=0.5, s=18)
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].set_xlabel("activated mechanics"); ax[0].set_ylabel("excess loss (GP fitness)")
        ax[0].set_title(f"{args.tag}: fitness vs mechanics"); ax[0].grid(alpha=0.3)
        ax[1].scatter(act + np.random.uniform(-0.15, 0.15, len(act)), mnll, alpha=0.5, s=18, label="model NLL")
        ax[1].scatter(act + np.random.uniform(-0.15, 0.15, len(act)), nrnll, alpha=0.5, s=18, label="no-rule NLL")
        ax[1].set_xlabel("activated mechanics"); ax[1].set_ylabel("NLL")
        ax[1].set_title("model vs no-rule NLL"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(_REPO / "nca_wm" / "figures" / f"excess_vs_activation_{args.tag}.{e}",
                        dpi=140, bbox_inches="tight")
        print(f"saved nca_wm/figures/excess_vs_activation_{args.tag}.png", flush=True)
    except Exception as ex:
        print(f"plot skipped: {ex}", flush=True)


if __name__ == "__main__":
    main()
