"""Canonical-space readouts for generated PuzzleScript (the arms of sample_eval.py).

Every playable text is compiled again and canonicalized (ps_extract.js +
canonicalize.from_engine), so texts that differ only in names, art, sounds, messages or
glyphs have the same mechanics key. Per arm:
  distinct mechanics among playable texts, and the most common mechanics' share;
  memorized: share of playable texts whose mechanics equal a training game's;
  levels_t1.0: per sample with the source game's mechanics, the share of its levels that
  are copies of one of the source game's levels.
Each of these is also given for the quality subsets of playable texts: dynamic (the state
changes in the 200-step random rollout) and puzzle (dynamic, not won by the random
rollout, and solved by the BFS in >= 5 moves).
The training reference is the set of mechanics keys of the training split: the train
documents of the canonical corpus, or with --raw-train-docs (a model trained on raw
sources) the canonical keys of those raw training games.

    python canonical_eval.py --eval DIR --data DIR --canonical DIR --engine-dir DIR --out FILE
        [--raw-train-docs FILE]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from canonicalize import CanonError, from_engine, level_key, mechanics_key
from check_games import check_texts

ARMS = ("uncond_t1.0", "uncond_t0.8", "levels_t1.0", "human_test")


def canon_keys(items, args):
    """id -> (mechanics key, [level keys]) or None, for {"id", "text"} items."""
    out = {}
    for it, e in zip(items, check_texts(items, args.engine_dir, workers=args.workers, stall_s=90.0,
                                        script="ps_extract.js")):
        try:
            c = from_engine(e) if e.get("ok") else None
        except CanonError:
            c = None
        out[it["id"]] = (mechanics_key(c), [level_key(lv) for lv in c.levels]) if c else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="the data dir the evaluated model used")
    ap.add_argument("--canonical", type=Path, required=True, help="prepare_canonical.py output")
    ap.add_argument("--raw-train-docs", type=Path, default=None)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.raw_train_docs:
        key_of = {r["id"]: r["mechanics_key"] for r in map(json.loads, open(args.canonical / "canonical.jsonl"))}
        raw_ids = [json.loads(line)["id"] for line in open(args.raw_train_docs)]
        train_keys = {key_of[g] for g in raw_ids if g in key_of}
        ref = {"train_reference": "canonical keys of raw train games", "raw_train_games": len(raw_ids),
               "raw_train_games_verified": sum(g in key_of for g in raw_ids)}
    else:
        train_keys = {json.loads(line)["id"] for line in open(args.canonical / "train_docs.jsonl")}
        ref = {"train_reference": "canonical train documents"}
    ref["train_mechanics"] = len(train_keys)
    report = {"eval": str(args.eval), **ref}

    sources = {t["id"]: t["text"] for t in map(json.loads, open(args.data / "test_texts.jsonl"))}
    for name in ARMS:
        recs = [json.loads(line) for line in open(args.eval / f"{name}.jsonl")]
        playable = [r for r in recs if r["check"].get("ok")]
        keys = canon_keys([{"id": r["id"], "text": r["text"]} for r in playable], args)

        def stats(subset):
            good = [keys[r["id"]] for r in subset if keys[r["id"]]]
            mech = Counter(k for k, _ in good)
            return {"n": len(subset), "canonicalized": len(good), "distinct_mechanics": len(mech),
                    "top_mechanics_share": (mech.most_common(1)[0][1] / len(good)) if good else None,
                    "memorized_mechanics_share": (sum(k in train_keys for k, _ in good) / len(good)) if good else None,
                    "distinct_unseen_mechanics": sum(k not in train_keys for k in mech)}

        def dynamic(r):
            return r["check"].get("rollout", {}).get("changed", 0) > 0

        def puzzle(r):
            b = r["check"].get("bfs") or {}
            return (dynamic(r) and not r["check"]["rollout"].get("won") and bool(b.get("solved"))
                    and (b.get("sol_len") or 0) >= 5)

        s = {"n": len(recs), "playable": len(playable), **{k: v for k, v in stats(playable).items() if k != "n"},
             "dynamic": stats([r for r in playable if dynamic(r)]),
             "puzzle": stats([r for r in playable if puzzle(r)])}
        if name == "levels_t1.0":
            src_ids = sorted({r["source_game"] for r in playable})
            src = canon_keys([{"id": g, "text": sources[g]} for g in src_ids], args)
            same, copied = 0, []
            for r in playable:
                k, s_ = keys[r["id"]], src.get(r["source_game"])
                if k and s_ and k[0] == s_[0]:
                    same += 1
                    copied.append(sum(lk in set(s_[1]) for lk in k[1]) / len(k[1]))
            s["same_mechanics_as_source"] = same
            s["level_copy_share_if_same_mechanics"] = (sum(copied) / len(copied)) if copied else None
            s["samples_with_a_new_level"] = sum(c < 1 for c in copied)
        report[name] = s
        print(name, json.dumps(s), flush=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
