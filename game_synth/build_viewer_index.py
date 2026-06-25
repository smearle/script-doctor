"""Build a viewer_index.json for a gp_evolve run so its games are playable in
PuzzleScript/src/viewer.html, sortable by generation / recency / fitness /
mechanics.

Scans <run>/_scratch/*.txt (every materialized game), keeps the valid 8x8/OBJ6
ones, and records per game: gen + ctr (parsed from the ev_<gen>_<ctr> name),
activated mechanics (engine rule-firing telemetry), solvable, and -- if a trained
WM checkpoint is given -- the WM loss (model_NLL) as a fitness proxy.

INCREMENTAL: re-running only processes games not already in the existing index,
so it is cheap to call repeatedly (see auto_index.py for a live watcher that lets
the viewer show the latest generations on refresh).

    .venv/bin/python -m game_synth.build_viewer_index --run game_synth/big_loss --source gp
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth.engine_train import OBJ6, _engine
from game_synth.render_games import coverage_walk, game_quality
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024
_NAME = re.compile(r"ev_(\d+)_(\d+)")


def _fitness(model, device, name, js, idd, tok):
    import torch
    from game_synth.code_cond_train import mixture_nll, pad_tokens
    from game_synth.engine_train import sample_traj
    r = random.Random(0); tk, mask = pad_tokens([tok], device); m = []
    with torch.no_grad():
        for _ in range(6):
            o, a, _ = sample_traj({name: (js, idd)}, name, 5, r)
            for t in range(len(a)):
                s = torch.from_numpy(o[t][None]).to(device)
                nx = torch.from_numpy(o[t + 1][None]).to(device)
                lo, lp = model.logits(s, torch.tensor([a[t]], device=device), tk, mask)
                m.append(mixture_nll(lo, lp, nx).item())
    return float(np.mean(m))


def build_index(run_rel, parser, *, source="gp", model=None, device=None,
                limit=8000, out=None, verbose=True):
    """Incrementally (re)build <run>/viewer_index.json. Returns (#total, #new)."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    run = _REPO / run_rel
    out = Path(out) if out else run / "viewer_index.json"
    vbuild = run / "_viewer_build"; vbuild.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(vbuild)

    rows, seen = [], set()
    if out.exists():
        try:
            rows = json.loads(out.read_text())
            seen = {r["title"] for r in rows}
        except Exception:
            rows, seen = [], set()

    paths = sorted(glob.glob(str(run / "_scratch" / "*.txt")))[:limit]
    new = 0
    for p in paths:
        name = Path(p).stem
        if name in seen:
            continue
        mm = _NAME.match(name)
        if not mm:
            continue
        gen, ctr = int(mm.group(1)), int(mm.group(2))
        try:
            code = Path(p).read_text(errors="ignore")
            gc._materialize_game(name, code)
            js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
            e = _engine(js)
            if np.asarray(e.get_objects_2d()).shape != (8, 8, 1):
                continue
            idd = [s.lower() for s in e.get_id_dict()]
            if any(s not in OBJ6 for s in idd):
                continue
            tree, ids = get_game_tree_from_js(parser, name)
            tok = tokenize_game(tree, ids, encode_sprites=False, include_levels=False)[:256]
            if not tok or max(tok) >= VOCAB:
                continue
            et = _engine(js); et.set_track_rules_fired(True)
            _, fired = coverage_walk(et, idd, 5, 16, random.Random(0))
            q = game_quality(js)
        except Exception:
            continue
        row = {"id": len(rows),
               "file": f"/{run_rel.rstrip('/')}/_scratch/{name}.txt",
               "source": source, "title": name,
               "solvable": q["solvable"], "mechanics": len(fired),
               "sol_len": q["sol_len"], "states": q["n_states"],
               "effect": round(q["effect_rate"], 2),
               "gen": gen, "recency": ctr}
        if model is not None:
            row["fitness"] = round(_fitness(model, device, name, js, idd, tok), 3)
        rows.append(row); seen.add(name); new += 1
    # write atomically so the viewer never reads a half-written file
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows)); tmp.replace(out)
    if verbose and rows:
        act = np.array([r["mechanics"] for r in rows])
        print(f"{run_rel}: {len(rows)} games (+{new} new) | activated mean {act.mean():.2f} "
              f"max {act.max()} | gens {min(r['gen'] for r in rows)}-{max(r['gen'] for r in rows)}",
              flush=True)
    return len(rows), new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="gp_evolve run dir (contains _scratch/)")
    ap.add_argument("--source", default="gp")
    ap.add_argument("--wm", default="", help="optional wm.pt for per-game fitness (model_NLL)")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    model = device = None
    if args.wm and (_REPO / args.wm).exists():
        import torch
        from game_synth.code_cond_train import CodeCondWorldModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CodeCondWorldModel(vocab=VOCAB).to(device)
        model.load_state_dict(torch.load(_REPO / args.wm, map_location=device)["model_state"])
        model.eval()
        print(f"loaded WM {args.wm} for fitness", flush=True)

    build_index(args.run, parser, source=args.source, model=model, device=device,
                limit=args.limit, out=(args.out or None))


if __name__ == "__main__":
    main()
