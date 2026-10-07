"""Sample mechanics for held-out levels from a level-first generator (stage 3) and score them
with the reference engine.

Candidate levels (rule pre-registered in protocol.md, stage 3): held-out (val and test) rows
of level_prompts.jsonl with height and width in [--min-side, --max-side], at least
--min-active objects that the level's own rules use, at most --max-objects objects, and an
own game of at most --max-rules rules (a game far bigger than DT can take); one per mechanics (its level with the most active objects, then the most objects, then the
first id); the --n-levels with the most active objects, then the most objects, then id.

For each candidate and temperature, --samples continuations of the level's prompt
(level_first.prompt_of) are drawn, and each is scored:
  format     level_first.to_standard() accepts it;
  engine     ps_check.js --dynamics on the standard text: compiles, one playable level, a
             200-step seeded random rollout (state changes, wins) and a 20k-node BFS;
             a puzzle is dynamic, not won by the rollout and BFS-solved in >= 5 moves;
  mechanics  the canonical mechanics_key (ps_extract.js, canonicalize.from_engine): distinct
             mechanics, copies of a train mechanics, the level's own held-out mechanics,
             and rule and object counts;
  behaviour  ps_probe.js fingerprints under 16 shared random action sequences: playable
             samples with equal fingerprints form one behaviour class (classes and the
             plug-in entropy of the class distribution); stochastic games are flagged.
The level's own (human) mechanics is scored the same way, as the reference.

Post-hoc diagnostic options (not pre-registered): --select random draws the candidates as a
seeded sample of held-out levels (one per mechanics) instead of the richest ones, and
--prompt-until RULES extends each prompt with the level's own flags, objects, legend and
collision layers, so only the rules and win conditions are sampled. --rescore recomputes the
scores of the samples already in --out with the current checkers. With --prompt-until RULES,
--constrain names|full samples under constrained.py's masks (exact sampling of the masked
distribution; per-sample rejection counts are recorded), and --deterministic also masks the
words random and randomdir. Outputs:
level_eval_report.json, samples.jsonl, candidates.jsonl and candidates.png (each candidate
level drawn under its human mechanics).

    python level_eval.py --data DIR --run DIR --out DIR --engine-dir DIR
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from canonicalize import CanonError, from_engine, mechanics_key
from check_games import check_texts
from level_first import DELIM, FormatError, to_standard
from model import GPT, GPTConfig
from prepare_level_first import active_objects


def prompt_until(text, header):
    """The document up to and including its `header` line: DELIM, or a mechanics section."""
    lines = text.split("\n")
    k = lines.index(DELIM)
    if header != DELIM:
        k = lines.index(header, k + 1)
    return "\n".join(lines[:k + 1]) + "\n"


def select_candidates(rows, args):
    def rank(r):
        return (-r["n_active"], -r["n_objects"], r["id"])

    if args.select == "random":
        one = {}
        for r in sorted(rows, key=lambda r: r["id"]):
            one.setdefault(r["mechanics_key"], r)
        pool = sorted(one.values(), key=lambda r: r["id"])
        return list(np.random.default_rng(args.seed).choice(pool, size=min(args.n_levels, len(pool)), replace=False))
    ok = [r for r in rows if args.min_side <= r["height"] <= args.max_side
          and args.min_side <= r["width"] <= args.max_side
          and r["n_active"] >= args.min_active and r["n_objects"] <= args.max_objects
          and r["n_rules"] <= args.max_rules]
    best = {}
    for r in sorted(ok, key=rank):
        best.setdefault(r["mechanics_key"], r)
    return sorted(best.values(), key=rank)[:args.n_levels]


def score(items, args, train_keys):
    """Engine, mechanics and behaviour scores for items with "id" and "standard" (None when
    the format check failed); fills each item in place."""
    std = [it for it in items if it["standard"] is not None]
    for it, c in zip(std, check_texts([{"id": it["id"], "text": it["standard"]} for it in std],
                                      args.engine_dir, workers=args.workers, dynamics=True, stall_s=90.0,
                                      progress_s=300)):
        it["check"] = c
    play = [it for it in std if it["check"].get("ok")]
    ext = check_texts([{"id": it["id"], "text": it["standard"]} for it in play], args.engine_dir,
                      workers=args.workers, script="ps_extract.js", stall_s=90.0, progress_s=300)
    probes = check_texts([{"id": it["id"], "text": it["standard"]} for it in play], args.engine_dir,
                         workers=args.workers, script="ps_probe.js", stall_s=90.0, progress_s=300)
    for it, e, p in zip(play, ext, probes):
        it["probe"] = {k: p.get(k) for k in ("ok", "probes", "distinct_states", "won", "again_capped",
                                             "stochastic", "timeout", "exception")}
        try:
            c = from_engine(e) if e.get("ok") else None
        except CanonError:
            c = None
        if c is not None:
            it["mechanics_key"] = mechanics_key(c)
            it["n_rules"] = len(c.rules)
            it["n_objects"] = len([o for o in c.objects if o != "background"])
            it["n_active"] = len(active_objects(c))
            it["train_copy"] = it["mechanics_key"] in train_keys


def summarize(items, own_key):
    n = len(items)
    play = [it for it in items if it.get("check", {}).get("ok")]
    roll = lambda it: it["check"].get("rollout", {})
    bfs = lambda it: it["check"].get("bfs", {})
    dyn = [it for it in play if roll(it).get("changed", 0) > 0]
    puzzle = [it for it in dyn if not roll(it).get("won") and bfs(it).get("solved")
              and (bfs(it).get("sol_len") or 0) >= 5]
    keyed = [it for it in play if "mechanics_key" in it]
    probed = [it for it in play if it.get("probe", {}).get("ok")]
    classes = Counter(tuple(it["probe"]["probes"]) for it in probed)
    entropy = -sum(k / len(probed) * math.log2(k / len(probed)) for k in classes.values()) if probed else None
    frac = lambda part, whole: len(part) / len(whole) if whole else None
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"n": n, "format_ok": frac([it for it in items if it["standard"] is not None], items),
            "playable": frac(play, items), "dynamic_if_playable": frac(dyn, play),
            "random_won_if_playable": frac([it for it in play if roll(it).get("won")], play),
            "puzzle_if_playable": frac(puzzle, play), "n_puzzles": len(puzzle),
            "n_distinct_puzzle_mechanics": len({it.get("mechanics_key") for it in puzzle} - {None}),
            "n_distinct_mechanics": len({it["mechanics_key"] for it in keyed}),
            "train_copy_if_keyed": frac([it for it in keyed if it["train_copy"]], keyed),
            "own_mechanics_if_keyed": frac([it for it in keyed if it["mechanics_key"] == own_key], keyed),
            "behaviour_classes": len(classes), "behaviour_entropy_bits": entropy,
            "n_probed": len(probed),
            "stochastic_if_probed": frac([it for it in probed if it["probe"]["stochastic"]], probed),
            "again_capped_if_probed": frac([it for it in probed if it["probe"]["again_capped"]], probed),
            "mean_rules_if_keyed": mean([it["n_rules"] for it in keyed]),
            "mean_objects_if_keyed": mean([it["n_objects"] for it in keyed]),
            "mean_active_if_keyed": mean([it["n_active"] for it in keyed]),
            "mean_distinct_states_if_probed": mean([it["probe"]["distinct_states"] for it in probed])}


def contact_sheet(cands, human, levels_report, t, args):
    from gallery import TILE, fit, render, sheet
    imgs = render([{"id": c["id"], "text": human[c["id"]]["standard"]} for c in cands], args.engine_dir, "node")
    tiles = []
    for c in cands:
        s = levels_report[c["id"]][t]
        tiles.append((fit(imgs.get(c["id"]), TILE),
                      f"{c['id'][:12]} {c['height']}x{c['width']} act {c['n_active']} | T{t}: "
                      f"play {s['playable']:.0%} cls {s['behaviour_classes']} puz {s['n_puzzles']}"))
    im = sheet(tiles, f"candidate levels (under their human mechanics); stats of {args.samples} samples at T={t}",
               4, TILE + 160)
    im.save(args.out / "candidates.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="prepare_level_first.py output")
    ap.add_argument("--run", type=Path, required=True, help="training output dir with best.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--n-levels", type=int, default=24)
    ap.add_argument("--min-side", type=int, default=5)
    ap.add_argument("--max-side", type=int, default=12)
    ap.add_argument("--min-active", type=int, default=5)
    ap.add_argument("--max-objects", type=int, default=16)
    ap.add_argument("--max-rules", type=int, default=64)
    ap.add_argument("--select", choices=["rich", "random"], default="rich", help="diagnostic: random held-out levels")
    ap.add_argument("--prompt-until", choices=[DELIM, "RULES"], default=DELIM,
                    help="diagnostic: RULES also gives the level's own flags, objects, legend and layers")
    ap.add_argument("--samples", type=int, default=64, help="per level and temperature")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.8, 1.0])
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--constrain", choices=["none", "names", "full"], default="none",
                    help="masks for rules-only sampling (constrained.py); needs --prompt-until RULES")
    ap.add_argument("--deterministic", action="store_true", help="with --constrain: also mask random and randomdir")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the samples already in --out with the current checkers (no sampling)")
    args = ap.parse_args()
    if (args.constrain != "none" or args.deterministic) and args.prompt_until != "RULES":
        ap.error("--constrain and --deterministic apply to rules-only sampling (--prompt-until RULES)")
    if args.deterministic and args.constrain == "none":
        ap.error("--deterministic needs --constrain")
    args.out.mkdir(parents=True, exist_ok=True)
    train_keys = {json.loads(line)["mechanics_key"] for line in open(args.data / "train_docs.jsonl")}
    if args.rescore:
        report, cands, items, human = load_for_rescore(args)
    else:
        report, cands, items, human = sample(args)
    t0 = time.time()
    score(items + list(human.values()), args, train_keys)
    print(f"scored ({time.time() - t0:.0f}s)", flush=True)

    temps = sorted({it["temp"] for it in items}, key=float)
    by = defaultdict(list)
    for it in items:
        by[(it["level"], it["temp"])].append(it)
    levels_report = {}
    for c in cands:
        own = c["mechanics_key"]
        levels_report[c["id"]] = {t: summarize(by[(c["id"], t)], own) for t in temps}
        levels_report[c["id"]]["human"] = summarize([human[c["id"]]], own)
    report["levels"] = levels_report
    report["overall"] = {t: summarize([it for it in items if it["temp"] == t], None) for t in temps}
    report["mean_over_levels"] = {
        t: {k: float(np.mean([levels_report[c["id"]][t][k] for c in cands if levels_report[c["id"]][t][k] is not None]))
            for k in ("playable", "dynamic_if_playable", "puzzle_if_playable", "n_distinct_mechanics",
                      "train_copy_if_keyed", "behaviour_classes", "behaviour_entropy_bits", "stochastic_if_probed")}
        for t in temps}
    report["format_errors"] = Counter(it["format_error"].split(":")[0][:60] for it in items
                                      if "format_error" in it).most_common(10)
    if any("constrain" in it for it in items):
        report["constrain_stats"] = {t: {
            "rejections_per_token": sum(it["constrain"]["rejections"] for it in items if it["temp"] == t)
            / max(1, sum(it["n_new_tokens"] for it in items if it["temp"] == t)),
            "fallbacks": sum(it["constrain"]["fallbacks"] for it in items if it["temp"] == t),
            "dead_ends": sum(it["constrain"]["dead_end"] for it in items if it["temp"] == t),
            "hit_eos": sum(it["hit_eos"] for it in items if it["temp"] == t)} for t in temps}
    with open(args.out / "samples.jsonl", "w") as f:
        for it in items + list(human.values()):
            f.write(json.dumps(it) + "\n")
    (args.out / "level_eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"overall": report["overall"], "mean_over_levels": report["mean_over_levels"]}, indent=1),
          flush=True)
    contact_sheet(cands, human, levels_report, temps[-1], args)
    print("done", flush=True)


SCORES = ("check", "probe", "mechanics_key", "n_rules", "n_objects", "n_active", "train_copy")


def load_for_rescore(args):
    """The report header, candidates and samples of an earlier run in --out, scores removed."""
    old = json.loads((args.out / "level_eval_report.json").read_text())
    report = {k: old[k] for k in ("run", "best_step", "val_loss", "test_loss_nats_per_token", "args", "candidates")}
    report["rescored"] = {"unix": time.time(), "note": "scores recomputed by the current checkers; samples unchanged"}
    cands = [json.loads(line) for line in open(args.out / "candidates.jsonl")]
    items, human = [], {}
    for it in map(json.loads, open(args.out / "samples.jsonl")):
        for k in SCORES:
            it.pop(k, None)
        if it["temp"] == "human":
            human[it["level"]] = it
        else:
            items.append(it)
    return report, cands, items, human


def sample(args):
    """Select the candidate levels and sample mechanics for them."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    prep = json.loads((args.data / "prep_report.json").read_text())
    bos, eos = prep["bos"], prep["eos"]
    ck = torch.load(args.run / "best.pt", map_location=args.device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    from train_lm import evaluate, load_split
    arr, off, lens = load_split(args.data, "test", cfg.max_seq_len)
    test_loss, _, _ = evaluate(model, arr, off, lens, cfg.max_seq_len, prep["pad"], args.device, 131072)
    report = {"run": str(args.run), "best_step": ck["step"], "val_loss": ck["val_loss"],
              "test_loss_nats_per_token": test_loss, "args": {k: str(v) for k, v in vars(args).items()}}

    texts = {}
    for split in ("val", "test"):
        for line in open(args.data / f"{split}_texts.jsonl"):
            t = json.loads(line)
            texts[t["id"]] = t["text"]
    # level_prompts.jsonl is written before over-long documents are dropped from the splits
    rows = [r for r in map(json.loads, open(args.data / "level_prompts.jsonl")) if r["id"] in texts]
    cands = select_candidates(rows, args)
    report["candidates"] = {"rule": {k: getattr(args, k) for k in ("n_levels", "min_side", "max_side",
                                                                    "min_active", "max_objects", "max_rules",
                                                                    "select", "prompt_until")},
                            "n_eligible_rows": len(rows), "n_selected": len(cands)}
    with open(args.out / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps({k: v for k, v in c.items() if k != "prompt"}) + "\n")
    print(f"{len(cands)} candidate levels", flush=True)

    if args.constrain != "none":
        from constrained import RulesChecker, generate_constrained, token_texts
        texts_of_tokens = token_texts(args.data / "tokenizer.json")
        report["constrain"] = {"mode": args.constrain, "deterministic": args.deterministic}
    items, t0 = [], time.time()
    for c in cands:
        prompt_text = prompt_until(texts[c["id"]], args.prompt_until)
        ids = tok.encode(prompt_text).ids
        max_new = min(args.max_new, cfg.max_seq_len - 1 - len(ids))
        for temp in args.temps:
            prompt = torch.tensor([[bos] + ids] * args.samples, dtype=torch.long, device=args.device)
            stats = [None] * args.samples
            if args.constrain == "none":
                outs, done = model.generate(prompt, max_new, eos, temperature=temp, generator=gen)
            else:
                checkers = [RulesChecker(prompt_text, args.constrain, texts_of_tokens, eos, args.deterministic)
                            for _ in range(args.samples)]
                outs, done, stats = generate_constrained(model, prompt, max_new, eos, checkers,
                                                         temperature=temp, generator=gen)
            for k, (o, d) in enumerate(zip(outs, done)):
                text = tok.decode(ids + o)
                it = {"id": f"{c['id']}-t{temp:g}-{k:03d}", "level": c["id"], "temp": f"{temp:g}", "text": text,
                      "n_new_tokens": len(o), "hit_eos": bool(d), "standard": None}
                if stats[k] is not None:
                    it["constrain"] = stats[k]
                try:
                    it["standard"] = to_standard(text)
                except (FormatError, CanonError) as ex:
                    it["format_error"] = str(ex)[:200]
                items.append(it)
        print(f"sampled {c['id']}: {len(items)} samples ({time.time() - t0:.0f}s)", flush=True)
    # human reference: each candidate's own held-out mechanics
    human = {c["id"]: {"id": "human-" + c["id"], "level": c["id"], "temp": "human", "text": texts[c["id"]],
                       "standard": to_standard(texts[c["id"]])} for c in cands}
    return report, cands, items, human

if __name__ == "__main__":
    main()
