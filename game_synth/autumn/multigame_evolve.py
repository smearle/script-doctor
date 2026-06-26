"""Multi-game Autumn ELM: seed the archive with all authored environments and
evolve with two operators — single-program mutation and cross-game crossover
(LLM-mediated; includes level/setting swap). Crossover children re-enter the
archive, so mechanics compound across games over time.

Hard cumulative budget cap for paid backends (persisted in cost log). Resumable.

Run: PYTHONPATH=game_synth/autumn .venv/bin/python3 game_synth/autumn/multigame_evolve.py \
       --iters 60 --backend anthropic --p-cross 0.5 --budget 5.0 \
       --out game_synth/autumn/runs/multigame
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

import crossover
import mutate
import rollout
from sexp import mutable_targets

TESTS = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"


def _load_dotenv(path="/home/jupyter-smearle/script-doctor/.env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def seed_archive(include_stochastic, probe_seeds, probe_steps, max_seeds=0):
    archive, seen, skipped = [], set(), []
    files = sorted(glob.glob(os.path.join(TESTS, "*.sexp")))
    for f in files:
        if max_seeds and len(archive) >= max_seeds:
            break
        name = os.path.basename(f)[:-4]
        prog = open(f).read()
        if not include_stochastic and "uniformChoice" in prog:
            continue
        sig = rollout.evaluate_multi(prog, probe_seeds, probe_steps)
        if not sig.ok:
            skipped.append((name, sig.error))
            continue
        archive.append({"id": name, "prog": prog, "name": name})
        seen.add(sig.key())
    return archive, seen, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--backend", default="anthropic", choices=["vllm", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--p-cross", type=float, default=0.5, help="prob of crossover vs mutation")
    ap.add_argument("--include-stochastic", action="store_true")
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--probe-steps", type=int, default=120)
    ap.add_argument("--max-seeds", type=int, default=0, help="cap authored seeds (0=all; for testing)")
    ap.add_argument("--budget", type=float, default=5.0, help="cumulative USD cap (paid backends)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--resume-cost", action="store_true",
                    help="reload prior children + accumulate prior spend from the manifest")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "multigame"))
    args = ap.parse_args()

    _load_dotenv()
    rng = random.Random(args.rng_seed)
    palette = mutate.load_palette()
    seeds = tuple(range(args.probe_seeds))
    served = args.model or ("claude-sonnet-4-6" if args.backend == "anthropic" else "qwen3-4b")
    pin, pout = mutate.PRICING.get(served, (0.0, 0.0))

    os.makedirs(os.path.join(args.out, "variants"), exist_ok=True)
    man_path = os.path.join(args.out, "manifest.jsonl")

    archive, seen, skipped = seed_archive(args.include_stochastic, seeds, args.probe_steps, args.max_seeds)
    print(f"seeded {len(archive)} authored games ({len(skipped)} skipped: {skipped[:3]}...)")

    n_child = 0
    spent = 0.0
    if args.resume_cost and os.path.exists(man_path):
        for line in open(man_path):
            r = json.loads(line)
            if r.get("result") == "accept" and "traj_key" in r:
                vp = os.path.join(args.out, "variants", f"{r['id']}.sexp")
                if os.path.exists(vp):
                    archive.append({"id": r["id"], "prog": open(vp).read(), "name": r["id"]})
                    seen.add(tuple(tuple(t) if isinstance(t, list) else t for t in r["traj_key"]))
                    n_child += 1
            spent += r.get("cost_usd", 0.0)
        print(f"resumed: +{n_child} children, prior spend ${spent:.4f}")

    man = open(man_path, "a")
    stats = {"mutate": 0, "cross": 0, "invalid": 0, "dead": 0, "dup": 0, "parse_fail": 0, "accept": 0}
    tok_in = tok_out = 0
    t0 = time.time()

    for it in range(args.iters):
        if pin and spent >= args.budget:
            print(f"BUDGET REACHED (${spent:.4f}); stopping at iter {it}")
            break
        do_cross = rng.random() < args.p_cross and len(archive) >= 2
        rec = {"iter": it}
        if do_cross:
            a, b = rng.sample(archive, 2)
            mode = rng.choice(["inject", "levelswap"])
            stats["cross"] += 1
            cm = crossover.propose_crossover(a["prog"], b["prog"], a["name"], b["name"],
                                             mode=mode, backend=args.backend, model=args.model)
            tok_in += cm.in_tok; tok_out += cm.out_tok
            cost = (cm.in_tok / 1e6 * pin) + (cm.out_tok / 1e6 * pout)
            spent += cost
            rec.update(op="cross", mode=mode, parent_a=a["name"], parent_b=b["name"],
                       backend=args.backend, model=served, cost_usd=round(cost, 5))
            if not cm.ok:
                stats["parse_fail"] += 1
                rec.update(result="parse_fail", error=cm.error)
                man.write(json.dumps(rec) + "\n"); man.flush()
                print(f"[{it:3d}] xover parse_fail {a['name']}x{b['name']} {cm.error}")
                continue
            rec["kind"] = cm.kind
            child = cm.program
        else:
            p = rng.choice(archive)
            tg = rng.choice(mutable_targets(p["prog"]))
            stats["mutate"] += 1
            mt = mutate.propose(p["prog"], tg, palette, backend=args.backend, model=args.model)
            tok_in += mt.in_tok; tok_out += mt.out_tok
            cost = (mt.in_tok / 1e6 * pin) + (mt.out_tok / 1e6 * pout)
            spent += cost
            rec.update(op="mutate", parent=p["name"], target=tg.label,
                       backend=args.backend, model=served, cost_usd=round(cost, 5))
            if not mt.ok:
                stats["parse_fail"] += 1
                rec.update(result="parse_fail", error=mt.error)
                man.write(json.dumps(rec) + "\n"); man.flush()
                print(f"[{it:3d}] mut parse_fail {p['name']} {mt.error}")
                continue
            rec["kind"] = mt.kind
            child = mutate.apply(p["prog"], tg, mt)

        sig = rollout.evaluate_multi(child, seeds, args.probe_steps)
        op = rec["op"]
        if not sig.ok:
            stats["invalid"] += 1
            rec.update(result="invalid", error=sig.error)
            print(f"[{it:3d}] {op:5s} invalid   {rec.get('kind','')[:28]:28s} {sig.error[:40]}")
        elif sig.dead:
            stats["dead"] += 1
            rec.update(result="dead")
            print(f"[{it:3d}] {op:5s} dead      {rec.get('kind','')[:28]:28s}")
        elif sig.key() in seen:
            stats["dup"] += 1
            rec.update(result="dup")
            print(f"[{it:3d}] {op:5s} dup       {rec.get('kind','')[:28]:28s}")
        else:
            stats["accept"] += 1
            n_child += 1
            vid = f"{'x' if op == 'cross' else 'm'}{n_child:03d}"
            open(os.path.join(args.out, "variants", f"{vid}.sexp"), "w").write(child)
            archive.append({"id": vid, "prog": child, "name": vid})
            seen.add(sig.key())
            rec.update(result="accept", id=vid, covered=list(sig.covered),
                       n_states=sig.n_states_max, traj_key=list(sig.key()))
            print(f"[{it:3d}] {op:5s} ACCEPT {vid} {rec.get('kind','')[:28]:28s} cum ${spent:.3f}")
        man.write(json.dumps(rec) + "\n"); man.flush()

    man.close()
    summary = {
        "iters": args.iters, "backend": args.backend, "model": served,
        "n_seed_games": len(archive) - stats["accept"], **stats,
        "accept_rate": round(stats["accept"] / max(1, stats["mutate"] + stats["cross"]), 3),
        "tok_in": tok_in, "tok_out": tok_out, "cost_usd": round(spent, 4),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
