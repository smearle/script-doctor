"""ELM loop: evolve novel-but-valid variants of an Autumn seed program.

Each step: sample a parent from the archive (seed + accepted variants), pick a
random mutable target (whole on-handler or initnext NEXT-clause), ask the LLM to
rewrite just that span, splice, validate in a subprocess, fingerprint the
rollout, and accept iff valid, non-dead, and behaviorally novel (unseen
trajectory+handler-set signature). Layout stays fixed by construction (only
dynamics spans are mutable).

Run:  PYTHONPATH=game_synth/autumn .venv/bin/python3 game_synth/autumn/evolve.py \
        --seed-game mario --iters 20 --backend vllm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import mutate
import rollout
from sexp import mutable_targets

TESTS = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"


def _load_dotenv(path="/home/jupyter-smearle/script-doctor/.env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-game", default="mario")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--probe-steps", type=int, default=120)
    ap.add_argument("--probe-seeds", type=int, default=3,
                    help="number of random-probe seeds; novelty = any trajectory differs")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reload the existing pool in --out and keep appending")
    args = ap.parse_args()

    _load_dotenv()
    rng = random.Random(args.rng_seed)
    palette = mutate.load_palette()

    seeds = tuple(range(args.probe_seeds))
    seed_prog = open(os.path.join(TESTS, f"{args.seed_game}.sexp")).read()
    seed_sig = rollout.evaluate_multi(seed_prog, seeds, args.probe_steps)
    assert seed_sig.ok, f"seed program failed to evaluate: {seed_sig.error}"

    tag = f"{args.seed_game}_{args.backend}_{args.model or 'default'}".replace("/", "-")
    out = args.out or os.path.join(os.path.dirname(__file__), "runs", tag)
    os.makedirs(os.path.join(out, "variants"), exist_ok=True)

    # archive entries: dict(prog, sig, id)
    archive = [{"id": "seed", "prog": seed_prog, "sig": seed_sig}]
    seen = {seed_sig.key()}
    stats = {"parse_fail": 0, "invalid": 0, "dead": 0, "dup": 0, "accept": 0}
    man_path = os.path.join(out, "manifest.jsonl")

    if args.resume and os.path.exists(man_path):
        # reload accepted variants (prog from disk; signature key from manifest,
        # or re-evaluate for older manifests that predate traj_key)
        for line in open(man_path):
            r = json.loads(line)
            if r.get("result") != "accept":
                continue
            vp = os.path.join(out, "variants", f"{r['id']}.sexp")
            if not os.path.exists(vp):
                continue
            prog = open(vp).read()
            if "traj_key" in r:
                key = tuple(tuple(t) if isinstance(t, list) else t for t in r["traj_key"])
            else:
                s = rollout.evaluate_multi(prog, seeds, args.probe_steps)
                if not s.ok:
                    continue
                key = s.key()
            archive.append({"id": r["id"], "prog": prog, "sig": None})
            seen.add(key)
            stats["accept"] += 1
        print(f"resumed: {stats['accept']} variants, {len(seen)} signatures")
    reloaded = stats["accept"]
    man = open(man_path, "a" if args.resume else "w")
    tok_in = tok_out = 0
    t0 = time.time()

    for it in range(args.iters):
        parent = rng.choice(archive)
        targets = mutable_targets(parent["prog"])
        tgt = rng.choice(targets)
        mut = mutate.propose(parent["prog"], tgt, palette, backend=args.backend,
                             model=args.model, temperature=args.temperature)
        tok_in += mut.in_tok
        tok_out += mut.out_tok
        rec = {"iter": it, "parent": parent["id"], "target": tgt.label,
               "target_kind": tgt.kind, "backend": args.backend,
               "model": args.model or ("claude-sonnet-4-6" if args.backend == "anthropic" else "qwen3-4b")}
        if not mut.ok:
            stats["parse_fail"] += 1
            rec.update(result="parse_fail", error=mut.error)
            man.write(json.dumps(rec) + "\n"); man.flush()
            print(f"[{it:3d}] parse_fail   {tgt.label[:40]:40s} {mut.error}")
            continue
        rec["kind"] = mut.kind
        cand = mutate.apply(parent["prog"], tgt, mut)
        sig = rollout.evaluate_multi(cand, seeds, args.probe_steps)
        if not sig.ok:
            stats["invalid"] += 1
            rec.update(result="invalid", error=sig.error)
            print(f"[{it:3d}] invalid      {mut.kind[:30]:30s} {sig.error[:50]}")
        elif sig.dead:
            stats["dead"] += 1
            rec.update(result="dead")
            print(f"[{it:3d}] dead         {mut.kind[:30]:30s}")
        elif sig.key() in seen:
            stats["dup"] += 1
            rec.update(result="dup")
            print(f"[{it:3d}] dup          {mut.kind[:30]:30s} covered={sig.covered}")
        else:
            stats["accept"] += 1
            vid = f"v{stats['accept']:03d}"
            path = os.path.join(out, "variants", f"{vid}.sexp")
            open(path, "w").write(cand)
            archive.append({"id": vid, "prog": cand, "sig": sig})
            seen.add(sig.key())
            rec.update(result="accept", id=vid, covered=list(sig.covered),
                       n_states=sig.n_states_max, traj_key=list(sig.key()))
            print(f"[{it:3d}] ACCEPT {vid}  {mut.kind[:30]:30s} covered={sig.covered} states={sig.n_states_max}")
        man.write(json.dumps(rec) + "\n"); man.flush()

    man.close()
    dt = time.time() - t0
    n = args.iters
    this_run_accept = stats["accept"] - reloaded
    served_model = args.model or ("claude-sonnet-4-6" if args.backend == "anthropic" else "qwen3-4b")
    pin, pout = mutate.PRICING.get(served_model, (0.0, 0.0))
    cost = tok_in / 1e6 * pin + tok_out / 1e6 * pout
    summary = {
        "seed_game": args.seed_game, "backend": args.backend,
        "model": args.model, "iters": n, "reloaded": reloaded, **stats,
        "valid_rate": round((n - stats["parse_fail"] - stats["invalid"]) / n, 3),
        "accept_rate": round(this_run_accept / n, 3),
        "n_variants": stats["accept"], "seconds": round(dt, 1),
        "tok_in": tok_in, "tok_out": tok_out, "cost_usd": round(cost, 4),
    }
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"variants in {out}/variants/")


if __name__ == "__main__":
    main()
