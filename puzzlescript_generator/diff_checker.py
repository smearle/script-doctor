"""Differential test of constrained.py against a reference copy of the checker (a performance
rewrite must not change one decision).

    python diff_checker.py --ref REF.py --samples SAMPLES.jsonl [...] --tokenizer TOKENIZER.json [--every K]

For every sample (level_eval.py samples.jsonl: rules-only completions, scored or not) and each
mode (names, full) and deterministic flag:
  - check_text (one character at a time) gives the same verdict and first rejected index;
  - walking the completion one character at a time, every intermediate state is the same, and
    (on all samples, or a seeded --token-subset) at every K-th position both checkers allow
    the same tokens and reach the same state for each of them (try_token over the whole
    vocabulary, the end-of-text token included), so the incremental path is compared on the
    lines sampling actually extends.
Exit status 1 on any difference; the report lists the first ones.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import constrained as new  # noqa: E402


def load_ref(path):
    spec = importlib.util.spec_from_file_location("constrained_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["constrained_ref"] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def same(x, y):
    """Equal decisions: both refused, or the same state (the two modules' classes differ)."""
    if x is None or y is None:
        return x is None and y is None
    return dataclasses.astuple(x) == dataclasses.astuple(y)


def split(text):
    k = text.index("\nRULES\n") + len("\nRULES\n")
    return text[:k], text[k:]


def run(job):
    ref_path, tok_path, items, every, eos, token_ids = job
    ref = load_ref(ref_path)
    texts = new.token_texts(tok_path)
    diffs, positions, decisions = [], 0, 0
    for it in items:
        prompt, completion = split(it["text"])
        for mode in ("names", "full"):
            for det in (False, True):
                a = ref.check_text(prompt, completion, mode, det)
                b = new.check_text(prompt, completion, mode, det)
                if a != b:
                    diffs.append(dict(id=it["id"], mode=mode, det=det, kind="check_text", ref=a, new=b))
                ck_r = ref.RulesChecker(prompt, mode, texts, eos, det)
                ck_n = new.RulesChecker(prompt, mode, texts, eos, det)
                st_r, st_n = ck_r.state, ck_n.state
                for i, ch in enumerate(completion):
                    if every and it["id"] in token_ids and i % every == 0:
                        ck_r.state, ck_n.state = st_r, st_n
                        positions += 1
                        for t in range(len(texts)):
                            x, y = ck_r.try_token(t), ck_n.try_token(t)
                            decisions += 1
                            if not same(x, y):
                                diffs.append(dict(id=it["id"], mode=mode, det=det, kind="token", pos=i, token=t,
                                                  ref=str(x)[:200], new=str(y)[:200]))
                                break
                    nr, nn = ck_r.advance(st_r, ch), ck_n.advance(st_n, ch)
                    if not same(nr, nn):
                        diffs.append(dict(id=it["id"], mode=mode, det=det, kind="advance", pos=i,
                                          ref=str(nr)[:200], new=str(nn)[:200]))
                        break
                    if nr is None:
                        break
                    st_r, st_n = nr, nn
                if len(diffs) > 20:
                    return diffs, positions, decisions
    return diffs, positions, decisions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--samples", type=Path, nargs="+", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--every", type=int, default=29)
    ap.add_argument("--eos", type=int, default=1, help="the tokenizer's end-of-text id (prep_report.json)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--token-subset", type=int, default=0,
                    help="compare whole-vocabulary decisions on this many samples (seeded draw); 0 = all")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    items = [json.loads(line) for path in a.samples for line in open(path)]
    items = [it for it in items if "\nRULES\n" in it["text"]]
    if a.limit:
        items = items[:a.limit]
    t0 = time.time()
    ids = [it["id"] for it in items]
    if a.token_subset:
        import random
        ids = random.Random(0).sample(ids, min(a.token_subset, len(ids)))
    token_ids = frozenset(ids)
    chunks = [items[i::a.workers] for i in range(a.workers)]
    with Pool(a.workers) as pool:
        results = pool.map(run, [(str(a.ref), str(a.tokenizer), c, a.every, a.eos, token_ids) for c in chunks])
    diffs = [d for r in results for d in r[0]]
    report = dict(samples=len(items), token_level_samples=len(token_ids), modes=["names", "full"],
                  deterministic=[False, True], every=a.every,
                  positions=sum(r[1] for r in results), token_decisions=sum(r[2] for r in results),
                  differences=len(diffs), first=diffs[:20], seconds=time.time() - t0)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "first"}))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
