"""Transfer eval: does a GP-trained code-conditioned WM predict HUMAN-authored
PuzzleScript transitions better than trivial baselines?

The WM is specialized to the OBJ6 domain (<=6 object channels). We map any human
game with <=6 objects into that domain BY ROLE, reproducing the exact channel
convention the GP WM learned (state order [bg, o1, player, o2, o3, o4]; token
order [bg, player, o1, o2, o3, o4] -- the GP player/wall swap), so the
code-conditioning stays aligned WITHOUT rewriting the game. Roles come from the
compiled JSON (backgroundid, playerMask); remaining objects fill wall/objA-C.

Per game we roll out random trajectories and compare, per transition:
  model_NLL   : the WM's prediction given the game's tokenized code
  norule_NLL  : default player movement + collision only
  ident_NLL   : nothing changes
Transfer = the WM beats norule (it captured rule effects beyond movement) on
games it never trained on, from a totally different (human) distribution.

    .venv/bin/python -m game_synth.human_transfer --wm game_synth/evolve_lossw0/wm.pt \
        --glob 'gallery_games/*.txt' --n 40
"""
from __future__ import annotations

import argparse
import glob
import json
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
from game_synth.engine_train import norule_next, _step
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024
N_ACT = 5


def role_map(js):
    """Return (state_names, token_names, name2bit) or None if game can't fit 6 chans.
    state order  = [bg, o1, player, o2, o3, o4]   (grid_of / GP state convention)
    token order  = [bg, player, o1, o2, o3, o4]   (tokenize / GP token convention)
    where o1..o4 are the non-bg/non-player objects (o1 takes the 'wall' slot)."""
    d = json.loads(js)
    idd = [s.lower() for s in d["idDict"]]
    name2bit = {nm: i for i, nm in enumerate(idd)}
    bg = idd[d.get("backgroundid", 0)]
    # playerMask = [hasMov, [maskvals...]] where each maskval is a BITMASK over
    # object indices (value 2^k => object index k), NOT an index itself.
    pmask = d.get("playerMask", [False, []])
    vals = pmask[1] if isinstance(pmask, list) and len(pmask) > 1 else []
    player_bits = []
    for v in (vals if isinstance(vals, list) else [vals]):
        b = 0
        while v:
            if v & 1:
                player_bits.append(b)
            v >>= 1; b += 1
    player_bits = [b for b in player_bits if b < len(idd)]
    if not player_bits:
        return None
    player = idd[player_bits[0]]
    others = [nm for nm in idd if nm != bg and nm != player]
    if len(others) > 4 or len(others) == 0:
        return None
    o = others + [None] * (4 - len(others))          # pad to 4 slots
    state_names = [bg, o[0], player, o[1], o[2], o[3]]
    token_names = [bg, player, o[0], o[1], o[2], o[3]]
    return state_names, token_names, name2bit


def grid_of(engine, state_names, name2bit):
    """(6,H,W) multihot in the role-mapped channel order; None slots stay zero."""
    a = np.asarray(engine.get_objects_2d())[:, :, 0]      # (W,H) bitmask over idDict bits
    W, H = a.shape
    g = np.zeros((6, H, W), np.float32)
    for ch, nm in enumerate(state_names):
        if nm is None:
            continue
        ei = name2bit[nm]
        g[ch] = ((a >> ei) & 1).T
    return g


def rollout(js, state_names, name2bit, T, rng):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine(); e.load_from_json(js); e.load_level(0)
    grids = [grid_of(e, state_names, name2bit)]; acts = []
    for _ in range(T):
        a = rng.randrange(N_ACT)
        _step(e, a, str(rng.getrandbits(40)))
        grids.append(grid_of(e, state_names, name2bit)); acts.append(a)
    return np.stack(grids), np.asarray(acts, np.int64)


@torch.no_grad()
def eval_game(model, js, tok, state_names, name2bit, device, n=6, T=5, seed=0):
    rng = random.Random(seed)
    tk, mask = pad_tokens([tok], device)
    z1 = torch.zeros(1, 1, device=device)
    m, nr, ident = [], [], []
    for _ in range(n):
        o, a = rollout(js, state_names, name2bit, T, rng)
        for t in range(len(a)):
            s = torch.from_numpy(o[t][None]).to(device)
            nx = torch.from_numpy(o[t + 1][None]).to(device)
            lo, lp = model.logits(s, torch.tensor([a[t]], device=device), tk, mask)
            m.append(mixture_nll(lo, lp, nx).item())
            nrp = norule_next(s, int(a[t]))
            nr.append(mixture_nll((nrp * 12 - 6)[:, None], z1, nx).item())
            ident.append(mixture_nll((s * 12 - 6)[:, None], z1, nx).item())
    return float(np.mean(m)), float(np.mean(nr)), float(np.mean(ident))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="game_synth/evolve_lossw0/wm.pt")
    ap.add_argument("--glob", default="gallery_games/*.txt")
    ap.add_argument("--n", type=int, default=40, help="how many compatible games to eval")
    ap.add_argument("--scan", type=int, default=400, help="how many files to scan for compatible games")
    ap.add_argument("--max-cells", type=int, default=400, help="grid-size cap (W*H); use ~100 for size-matched to 8x8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)

    out = _REPO / "game_synth" / "transfer_out"; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    parser = init_ps_lark_parser()

    model = CodeCondWorldModel(vocab=VOCAB).to(device)
    model.load_state_dict(torch.load(_REPO / args.wm, map_location=device)["model_state"])
    model.eval()
    print(f"loaded WM {args.wm}", flush=True)

    paths = sorted(glob.glob(str(_REPO / args.glob)))
    random.Random(args.seed).shuffle(paths)
    rows = []
    for p in paths[:args.scan]:
        if len(rows) >= args.n:
            break
        name = Path(p).stem
        try:
            code = Path(p).read_text(errors="ignore")
            gc._materialize_game(name, code)
            js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
            e = Engine(); e.load_from_json(js); e.load_level(0)
            a0 = np.asarray(e.get_objects_2d())
            if a0.shape[2] != 1 or a0.shape[0] * a0.shape[1] > args.max_cells:   # <=32 objs, size cap
                continue
            rm = role_map(js)
            if rm is None:
                continue
            state_names, token_names, name2bit = rm
            tree, _ = get_game_tree_from_js(parser, name)
            canon = [n for n in token_names if n is not None]
            tok = tokenize_game(tree, canon, encode_sprites=False, include_levels=False)[:256]
            if not tok or max(tok) >= VOCAB:
                continue
            mnll, nrnll, idnll = eval_game(model, js, tok, state_names, name2bit, device)
        except Exception:
            continue
        rows.append({"name": name, "model": mnll, "norule": nrnll, "ident": idnll,
                     "excess": mnll - nrnll})
        print(f"  {name:34s} model {mnll:7.3f}  norule {nrnll:7.3f}  ident {idnll:7.3f}  "
              f"{'BEATS norule' if mnll < nrnll else 'above norule'}", flush=True)

    if not rows:
        print("no compatible human games found", flush=True); return
    M = np.array([r["model"] for r in rows]); NR = np.array([r["norule"] for r in rows])
    ID = np.array([r["ident"] for r in rows])
    print(f"\n=== transfer over {len(rows)} compatible human games ({args.glob}) ===", flush=True)
    print(f"  mean model_NLL {M.mean():.3f} | norule {NR.mean():.3f} | identity {ID.mean():.3f}")
    print(f"  WM beats norule on {(M < NR).sum()}/{len(rows)} games "
          f"({(M < NR).mean():.0%}); beats identity on {(M < ID).sum()}/{len(rows)}")
    print(f"  mean excess (model-norule) {(M-NR).mean():+.3f} (negative => WM transfers)")
    (out / "transfer.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out}/transfer.jsonl", flush=True)


if __name__ == "__main__":
    main()
