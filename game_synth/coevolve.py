"""Co-evolution microcosm: generation + world-model training over a growing pool.

Two optimization processes run together over a shared, ever-growing pool of
DISTINCT environments:
  - GENERATION: GP search proposes games -> validate (compile) -> token-dedup ->
    add to the pool.
  - LEARNING: a code-conditioned world model trains on transitions from the whole
    pool by gradient descent.
Each round we track every environment's LEARNABILITY w.r.t. the current WM:
  excess_loss(env) = model_NLL(env) - norule_NLL(env)
i.e. how much of the env's rule-effects (beyond default movement) the WM has yet
to capture. excess_loss->0 = mastered; high-but-decreasing = being learned (the
open-ended frontier); high-flat = hard/under-sampled. This is the signal that
would steer generation toward learnable-but-unlearned games.

    .venv/bin/python -u -m game_synth.coevolve --rounds 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth import gp_generate as GP
from game_synth.dedup import token_key
from game_synth.engine_train import B2I, N_ACT, OBJ6, _engine, norule_next, sample_traj
from game_synth.code_cond_train import (CodeCondWorldModel, mixture_nll, pad_tokens)
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024  # generous fixed embedding vocab (GP token ids are bounded)


def _validate_and_load(name, code, parser):
    """Compile, check 8x8 + OBJ6 vocab, tokenize. Returns (json, tokens) or None."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    try:
        gc._materialize_game(name, code)
        js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
        e = _engine(js)
        a = np.asarray(e.get_objects_2d())
        if a.shape != (8, 8, 1):
            return None
        idd = [s.lower() for s in e.get_id_dict()]
        if any(s not in OBJ6 for s in idd):
            return None
        tree, ids = get_game_tree_from_js(parser, name)
        tok = tokenize_game(tree, ids, encode_sprites=False, include_levels=False)[:256]
        if not tok or max(tok) >= VOCAB:
            return None
        return js, idd, tok          # (engine json, id_dict, code tokens)
    except Exception:
        return None


def transitions(pool, names, n, device, rng):
    S, A, NX, codes = [], [], [], []
    for _ in range(n):
        name = rng.choice(names)
        o, a, _ = sample_traj(pool["json"], name, rng.randint(2, 6), rng)
        t = rng.randrange(len(a))
        S.append(o[t]); A.append(a[t]); NX.append(o[t + 1]); codes.append(pool["tok"][name])
    tok, mask = pad_tokens(codes, device)
    return (torch.from_numpy(np.stack(S)).to(device), torch.tensor(A, device=device),
            torch.from_numpy(np.stack(NX)).to(device), tok, mask)


@torch.no_grad()
def excess_loss(model, pool, name, device, n=6, seed=0):
    rng = random.Random(seed + hash(name) % 9973)
    tok, mask = pad_tokens([pool["tok"][name]], device)
    z1 = torch.zeros(1, 1, device=device)
    m, nr = [], []
    for _ in range(n):
        o, a, _ = sample_traj(pool["json"], name, 5, rng)
        for t in range(len(a)):
            s = torch.from_numpy(o[t][None]).to(device)
            nx = torch.from_numpy(o[t + 1][None]).to(device)
            act = torch.tensor([a[t]], device=device)
            lo, lp = model.logits(s, act, tok, mask)
            m.append(mixture_nll(lo, lp, nx).item())
            nr.append(mixture_nll((norule_next(s, int(a[t])) * 12 - 6)[:, None], z1, nx).item())
    return float(np.mean(m)) - float(np.mean(nr)), float(np.mean(nr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed-games", type=int, default=40)
    ap.add_argument("--gen-per-round", type=int, default=6)
    ap.add_argument("--updates-per-round", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seed-glob", default="game_synth/fixed_s*/games/*.txt")
    ap.add_argument("--out", default="game_synth/coevolve_run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    out = _REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    # GP fixed geometry/vocab so games match the WM's 8x8 / OBJ6 setup.
    GP.FIXED.update(w=8, h=8, n_levels=1, objs=["ObjA", "ObjB", "ObjC"])

    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    rng = random.Random(args.seed)

    pool = {"json": {}, "tok": {}, "sig": set(), "added": {}}

    def add(name, code, rnd):
        sig = token_key(code)[0]
        if sig in pool["sig"]:
            return False
        r = _validate_and_load(name, code, parser)
        if r is None:
            return False
        pool["json"][name] = (r[0], r[1]); pool["tok"][name] = r[2]   # (json,id_dict), tokens
        pool["sig"].add(sig); pool["added"][name] = rnd
        return True

    # seed pool from already-generated fixed games
    seeds = sorted((_REPO).glob(args.seed_glob)); random.Random(1).shuffle(seeds)
    si = 0
    while len(pool["json"]) < args.seed_games and si < len(seeds):
        p = seeds[si]; si += 1
        add(p.stem, p.read_text(errors="ignore"), 0)
    print(f"seeded pool: {len(pool['json'])} envs", flush=True)

    model = CodeCondWorldModel(vocab=VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"code-conditioned WM params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    metrics = []
    log = (out / "learnability.jsonl").open("w")
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        # --- GENERATION: propose, validate, dedup, grow pool ---
        added = tries = 0
        while added < args.gen_per_round and tries < args.gen_per_round * 8:
            tries += 1
            try:
                code = GP.sample_game(rng)
            except Exception:
                continue
            if add(f"co_{rnd:03d}_{tries:02d}", code, rnd):
                added += 1
        names = list(pool["json"].keys())
        # --- LEARNING: train WM on the whole pool ---
        model.train()
        for _ in range(args.updates_per_round):
            s, a, nx, tok, mask = transitions(pool, names, args.batch_size, device, rng)
            lo, lp = model.logits(s, a, tok, mask)
            loss = mixture_nll(lo, lp, nx).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # --- TRACK: per-env learnability ---
        row = {"round": rnd, "pool_size": len(names), "t": time.time() - t0}
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            model.eval()
            per = {n: excess_loss(model, pool, n, device)[0] for n in names}
            xs = np.array(list(per.values()))
            row.update(mean_excess=float(xs.mean()), frac_mastered=float((xs < 0.1).mean()),
                       new_env_excess=float(np.mean([per[n] for n in names
                                                     if pool["added"][n] == rnd]) if added else float("nan")))
            for n, v in per.items():
                log.write(json.dumps({"round": rnd, "env": n, "excess": v,
                                      "added": pool["added"][n]}) + "\n")
            log.flush()
            print(f"round {rnd:2d} | pool {len(names):3d} | mean_excess {row['mean_excess']:+.3f} "
                  f"| mastered {row['frac_mastered']:.0%} | new-env excess {row['new_env_excess']:+.3f} "
                  f"| {row['t']:.0f}s", flush=True)
        metrics.append(row)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.close()
    torch.save({"model_state": model.state_dict()}, out / "wm.pt")

    # figure: pool growth + WM mean excess-loss (learning the growing pool)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ev = [m for m in metrics if "mean_excess" in m]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot([m["round"] for m in metrics], [m["pool_size"] for m in metrics], "-o")
        ax[0].set_xlabel("round"); ax[0].set_ylabel("distinct envs in pool"); ax[0].set_title("Pool growth")
        ax[1].plot([m["round"] for m in ev], [m["mean_excess"] for m in ev], "-o", label="mean excess-loss")
        ax[1].plot([m["round"] for m in ev], [m["new_env_excess"] for m in ev], "-s", label="newly-added envs")
        ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("round")
        ax[1].set_ylabel("excess loss vs no-rule (learnability)")
        ax[1].set_title("WM learning the growing pool"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(f"nca_wm/figures/coevolve_pool_learnability.{e}", dpi=140, bbox_inches="tight")
        print("saved nca_wm/figures/coevolve_pool_learnability.{png,pdf}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)
    print(f"\nDONE | final pool {len(pool['json'])} envs | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
