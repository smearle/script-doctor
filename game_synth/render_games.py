"""Render generated PuzzleScript games to GIFs + measure ACTIVATED mechanics.

For each game we compile it, then walk a trajectory chosen to surface as many
rules as possible (greedy rule-coverage: at each step try every action and take
the one that fires the most not-yet-seen rules). The engine's rule-firing
telemetry (set_track_rules_fired / get_rules_fired) gives the count of DISTINCT
rules that actually fire during play -- the "activated mechanics" -- which is a
far better complexity signal than the raw rule count in the source (many GP/LLM
rules are dead, unsatisfiable, or shadowed).

    .venv/bin/python -m game_synth.render_games --glob 'game_synth/dataset/games/*.txt' \
        --n 24 --sort-by activated --out game_synth/render_out

Outputs: <out>/<name>.gif per game + <out>/mechanics.jsonl + a histogram figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from game_synth.engine_train import OBJ6, B2I, grid_of, _step

N_ACT = 5  # up,left,down,right,action

# fixed palette matching the GP games' declared colors (RGB)
PALETTE = {
    "background": (255, 255, 255),
    "wall": (90, 90, 90),
    "player": (40, 90, 220),
    "obja": (220, 50, 50),
    "objb": (40, 180, 70),
    "objc": (240, 210, 40),
}
# z-order: later = drawn on top (player/objects over background/wall)
ZORDER = ["background", "wall", "obja", "objb", "objc", "player"]


def _engine_tracked(js):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine(); e.load_from_json(js); e.load_level(0)
    e.set_track_rules_fired(True)
    return e


def render_frame(grid_chw, scale=24, grid_lines=True):
    """(6,H,W) multihot -> (H*scale, W*scale, 3) uint8 with z-ordered colors."""
    C, H, W = grid_chw.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = PALETTE["background"]
    for nm in ZORDER:
        ci = B2I.get(nm)
        if ci is None or ci >= C:
            continue
        mask = grid_chw[ci] > 0.5
        img[mask] = PALETTE[nm]
    big = np.repeat(np.repeat(img, scale, 0), scale, 1)
    if grid_lines:
        big[::scale, :, :] = 200
        big[:, ::scale, :] = 200
    return big


def coverage_walk(e, idd, n_act, max_steps, rng, max_again=40):
    """Greedy rule-coverage walk: each step pick the action firing the most
    not-yet-seen rules (ties broken randomly), to surface activated mechanics.
    Returns (frames, fired_set)."""
    def _apply(a):
        e.clear_rules_fired()
        e.seed_rng(str(rng.getrandbits(40)))
        e.process_input(a)
        k = 0
        while e.is_againing() and k < max_again:
            e.process_input(-1); k += 1
        return set(e.get_rules_fired())

    frames = [grid_of(e, idd)]
    fired_all = set()
    for _ in range(max_steps):
        step_bak = e.backup_level()                  # start-of-step state S
        best_a, best_new = None, -1
        for a in rng.sample(range(n_act), n_act):    # try every action from S
            new = len(_apply(a) - fired_all)
            e.restore_level(step_bak)                # always rewind to S
            if new > best_new:
                best_new, best_a = new, a
        fired_all |= _apply(best_a)                  # commit the chosen action
        frames.append(grid_of(e, idd))
        if e.is_winning():
            break
    return frames, fired_all


def game_quality(js, budget=2000, timeout_ms=1500):
    """Non-triviality metrics from BFS search over a compiled game:
      solvable / sol_len  : shortest BFS solution (sol_len<=1 => trivially winnable)
      n_states            : distinct reachable states BFS discovered (state-space size)
      effect_rate         : fraction of explored (s,a) where the action CHANGES the
                            state (low => player actions are mostly no-ops)
    All from a bounded search so it is cheap enough to gate every candidate."""
    import numpy as np
    import puzzlescript_cpp._puzzlescript_cpp as m
    e = m.Engine(); e.load_from_json(js); e.load_level(0)
    sr = m.solve_bfs(e, budget, timeout_ms)
    solvable = bool(sr.won)
    sol_len = len(sr.actions) if solvable else -1
    e2 = m.Engine(); e2.load_from_json(js); e2.load_level(0)
    td = m.collect_transitions_bfs(e2, budget, timeout_ms, False)
    S = np.asarray(td.states); NS = np.asarray(td.next_states)
    n_trans = len(S)
    if n_trans:
        chg = np.fromiter(((S[i] != NS[i]).any() for i in range(n_trans)), bool, n_trans)
        effect_rate = float(chg.mean())
        distinct = len({S[i].tobytes() for i in range(n_trans)}
                       | {NS[i].tobytes() for i in range(n_trans)})
    else:
        effect_rate, distinct = 0.0, 0
    return {"solvable": solvable, "sol_len": sol_len, "n_states": distinct,
            "n_trans": n_trans, "effect_rate": effect_rate, "iters": int(td.iterations)}


def process(path, parser, rng, scale, max_steps, out_dir, save_gif=True):
    import imageio
    from puzzlescript_cpp import CppPuzzleScriptBackend
    name = Path(path).stem
    code = Path(path).read_text(errors="ignore")
    try:
        gc._materialize_game(name, code)
        js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
        e = _engine_tracked(js)
        idd = [s.lower() for s in e.get_id_dict()]
        if any(s not in OBJ6 for s in idd):
            return None
        rc = e.get_rule_count()
        frames, fired = coverage_walk(e, idd, N_ACT, max_steps, rng)
        won = e.is_winning()
    except Exception as ex:
        return None
    rec = {"name": name, "rule_count": int(rc), "activated": len(fired),
           "won": bool(won), "steps": len(frames) - 1, "path": str(path)}
    if save_gif and len(frames) > 1:
        imgs = [render_frame(f, scale) for f in frames]
        gif = out_dir / f"{name}.gif"
        imageio.mimsave(gif, imgs, duration=0.25, loop=0)
        rec["gif"] = str(gif)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="game_synth/gp_evolve_run/_scratch/*.txt")
    ap.add_argument("--n", type=int, default=24, help="how many games to render GIFs for")
    ap.add_argument("--scan", type=int, default=0,
                    help="if >0, measure mechanics on this many games (no GIF) to rank, then GIF top --n")
    ap.add_argument("--sort-by", default="activated", choices=["activated", "rule_count", "none"])
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--scale", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="game_synth/render_out")
    args = ap.parse_args()

    out = _REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    rng = random.Random(args.seed)

    paths = sorted(glob.glob(str(_REPO / args.glob)))
    random.Random(args.seed).shuffle(paths)
    print(f"found {len(paths)} games matching glob", flush=True)

    # Phase 1: scan (measure mechanics, no GIF) if requested, else just process --n
    recs = []
    if args.scan > 0:
        scan_paths = paths[:args.scan]
        for i, p in enumerate(scan_paths):
            r = process(p, parser, rng, args.scale, args.max_steps, out, save_gif=False)
            if r:
                recs.append(r)
            if (i + 1) % 25 == 0:
                print(f"  scanned {i+1}/{len(scan_paths)} | usable {len(recs)}", flush=True)
        if args.sort_by != "none":
            recs.sort(key=lambda r: r[args.sort_by], reverse=True)
        top = recs[:args.n]
        print(f"scanned {len(recs)} | rendering GIFs for top {len(top)} by {args.sort_by}", flush=True)
        for r in top:
            rr = process(r["path"], parser, rng, args.scale, args.max_steps, out, save_gif=True)
            if rr:
                r["gif"] = rr.get("gif")
    else:
        for p in paths[:args.n]:
            r = process(p, parser, rng, args.scale, args.max_steps, out, save_gif=True)
            if r:
                recs.append(r)
        if args.sort_by != "none":
            recs.sort(key=lambda r: r[args.sort_by], reverse=True)

    (out / "mechanics.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    if recs:
        act = np.array([r["activated"] for r in recs])
        rc = np.array([r["rule_count"] for r in recs])
        print(f"\n=== activated mechanics over {len(recs)} games ===", flush=True)
        print(f"  activated: mean {act.mean():.2f}  median {np.median(act):.0f}  "
              f"max {act.max()}  | dist {np.bincount(act).tolist()}", flush=True)
        print(f"  rule_count: mean {rc.mean():.2f}  max {rc.max()}", flush=True)
        print(f"  solvable(won): {sum(r['won'] for r in recs)}/{len(recs)}", flush=True)
        # histogram figure
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            mx = max(act.max(), rc.max())
            bins = np.arange(0, mx + 2) - 0.5
            ax.hist(rc, bins=bins, alpha=0.5, label="rules in source", color="tab:gray")
            ax.hist(act, bins=bins, alpha=0.7, label="activated (fire in play)", color="tab:green")
            ax.set_xlabel("# rules"); ax.set_ylabel("# games")
            ax.set_title("Source rules vs. activated mechanics"); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            for e in ("png", "pdf"):
                fig.savefig(out / f"mechanics_hist.{e}", dpi=140, bbox_inches="tight")
            print(f"saved {out}/mechanics_hist.png", flush=True)
        except Exception as ex:
            print(f"plot skipped: {ex}", flush=True)
        # show top games
        for r in sorted(recs, key=lambda r: r["activated"], reverse=True)[:12]:
            print(f"  {r['name']:24s} activated {r['activated']}/{r['rule_count']} "
                  f"won={r['won']} steps={r['steps']}", flush=True)
    print(f"\nGIFs + mechanics.jsonl in {out}", flush=True)


if __name__ == "__main__":
    main()
