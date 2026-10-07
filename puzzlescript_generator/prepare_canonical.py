"""Canonical, deduplicated PuzzleScript corpus (user request 2026-10-06: deduplicate
functionally equivalent mechanics and levels; drop sprites, names and other aesthetics).

1. extract: every dedup representative of the pinned dataset revision is compiled by the
   reference engine, which dumps its own parse (ps_extract.js). ok = the verdict of
   prepare_data.py (compiles cleanly, >= 1 playable level).
2. canonicalize (canonicalize.py), then verify against the original with ps_equiv.js:
   the same initial cells in every playable level, and the same cells and win flag after
   each of 60 seeded random actions on the first two levels (same engine RNG seed;
   message output is a no-op in both). Only verified games are kept.
3. group by mechanics_key: one document per distinct mechanics. Its levels are the
   distinct levels (level_key) of its member games, members in id order; above
   --max-levels a seeded subset keeps their relative order. Levels that would push the
   document past --max-chars or past the glyph alphabet are skipped.
4. every document is compiled again (ps_check.js); failures are dropped and counted.
5. split: documents are linked when their member games share a normalized title or a
   dataset mechanics_hash; each component goes wholly to train, val or test
   (prepare_data.split_of, same fractions and cap, counted in documents).
6. BPE and token bins as in prepare_data.py, so train_lm.py and sample_eval.py run unchanged.

    python prepare_canonical.py --out DIR --revision SHA --engine-dir DIR [--limit N]

Outputs (in --out): extract.jsonl, equiv.jsonl, doc_check.jsonl (stage caches, reused
when made from identical inputs), canonical.jsonl (per verified game: id, mechanics_key, level_keys, canonical
text), doc_check.jsonl, the token files of prepare_data.py and prep_report.json.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from canonicalize import CanonError, from_engine, level_key, mechanics_key, mechanics_text, render
from check_games import check_texts
from prepare_data import SPECIALS, UnionFind, norm_title, normalize, split_of


@functools.lru_cache(maxsize=None)
def engine_fingerprint(engine_dir: str) -> str:
    """sha256 of the engine's JavaScript: the Node wrapper and PuzzleScript/src/js."""
    root = Path(engine_dir)
    files = [root / "puzzlescript_nodejs/puzzlescript/engine.js"] + sorted((root / "PuzzleScript/src/js").rglob("*.js"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def cached_check(path: Path, items, script, args):
    """Run `script` over items, reusing path when it was made from identical inputs (the
    items, the script's and the engine loader's source, and the engine's JavaScript)."""
    here = Path(__file__).resolve().parent
    sources = [(here / name).read_text() for name in (script, "ps_engine.js")]
    digest = hashlib.sha256(json.dumps([script, sources, engine_fingerprint(str(args.engine_dir)), items])
                            .encode()).hexdigest()
    stamp = path.with_suffix(".inputs-sha256")
    if path.exists() and stamp.exists() and stamp.read_text() == digest:
        print(f"reusing {path.name}")
        return [json.loads(line) for line in path.open()]
    t0 = time.time()
    res = check_texts(items, args.engine_dir, workers=args.workers, stall_s=args.stall_s, script=script,
                      progress_s=300)
    with path.open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    stamp.write_text(digest)
    print(f"{script}: {len(items)} games in {time.time() - t0:.0f}s")
    return res


def select_levels(c, pooled, key, max_levels, max_chars):
    if len(pooled) > max_levels:
        keep = sorted(random.Random(f"levels:{key}").sample(range(len(pooled)), max_levels))
        pooled = [pooled[i] for i in keep]
    sel = []
    for lv in pooled:
        try:
            text = render(c, sel + [lv])
        except CanonError:  # more distinct cells than glyphs
            continue
        if sel and len(text) > max_chars:
            continue
        sel.append(lv)
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="smearle/puzzlescript-gists")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--stall-s", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="seeded subset of representatives (tests)")
    ap.add_argument("--max-levels", type=int, default=32)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--val-frac", type=float, default=0.025)
    ap.add_argument("--test-frac", type=float, default=0.025)
    ap.add_argument("--max-heldout-component", type=int, default=40)
    ap.add_argument("--no-tokenize", action="store_true", help="stop after the documents (tests)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    args.out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(args.repo, "data/puzzlescript_games.jsonl", repo_type="dataset",
                           revision=args.revision)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    reps = [r for r in rows if r["is_dedup_representative"] and r["content"].strip()]
    for r in reps:
        r["content"] = normalize(r["content"])
    if args.limit:
        reps = random.Random(0).sample(reps, args.limit)
    by_id = {r["id"]: r for r in reps}
    report = {"repo": args.repo, "revision": args.revision, "n_rows": len(rows), "n_representatives": len(reps),
              "limit": args.limit, "max_levels": args.max_levels, "max_chars": args.max_chars}

    # 1-2. extract, canonicalize, verify
    ext = cached_check(args.out / "extract.jsonl", [{"id": r["id"], "text": r["content"]} for r in reps],
                       "ps_extract.js", args)
    report["extract"] = {"ok": sum(bool(e.get("ok")) for e in ext),
                         "timeouts": sum(bool(e.get("timeout")) for e in ext),
                         "crashes": sum("crash" in e for e in ext),
                         "exceptions": sum("exception" in e for e in ext)}
    canon, items, canon_err = {}, [], Counter()
    for r, e in zip(reps, ext):
        if not e.get("ok"):
            continue
        try:
            c = from_engine(e)
            text = render(c)
        except CanonError as ex:
            canon_err[str(ex).split(":")[0]] += 1
            continue
        canon[r["id"]] = c
        items.append({"id": r["id"], "orig": r["content"], "canon": text,
                      "map": {o: c.rename[o] for o in e["objects"]}})
    report["canonicalize_errors"] = dict(canon_err)
    eq = cached_check(args.out / "equiv.jsonl", items, "ps_equiv.js", args)
    outcome = Counter()
    verified = []
    for it, q in zip(items, eq):
        if q.get("timeout") or "crash" in q:
            outcome["checker timeout/crash"] += 1
        elif "exception" in q:
            outcome["exception"] += 1
        elif not q.get("canon_compiled"):
            outcome["canonical does not compile"] += 1
        elif not q.get("init_ok"):
            outcome["initial cells differ"] += 1
        elif q.get("nondet"):
            outcome["original does not replay itself"] += 1
        elif not q.get("dyn_ok"):
            outcome["dynamics differ"] += 1
        else:
            outcome["equivalent"] += 1
            verified.append(it["id"])
    report["equivalence"] = dict(outcome)
    print("canonicalize errors", dict(canon_err), "| equivalence", dict(outcome))
    vset = set(verified)
    with open(args.out / "canonical.jsonl", "w") as f:
        for it in items:
            if it["id"] in vset:
                c = canon[it["id"]]
                f.write(json.dumps({"id": it["id"], "mechanics_key": mechanics_key(c),
                                    "level_keys": [level_key(lv) for lv in c.levels],
                                    "text": it["canon"]}) + "\n")

    # 3. one document per distinct mechanics, with the pooled distinct levels
    groups = defaultdict(list)
    for gid in verified:
        groups[mechanics_key(canon[gid])].append(gid)
    docs = []
    for key, members in groups.items():
        members.sort()
        pooled, seen = [], set()
        for gid in members:
            for lv in canon[gid].levels:
                k = level_key(lv)
                if k not in seen:
                    seen.add(k)
                    pooled.append(lv)
        c0 = canon[members[0]]
        sel = select_levels(c0, pooled, key, args.max_levels, args.max_chars)
        docs.append({"id": key, "text": render(c0, sel), "members": members,
                     "n_levels_pooled": len(pooled), "n_levels_kept": len(sel)})
    sizes = Counter(len(d["members"]) for d in docs)
    top = sorted(docs, key=lambda d: -len(d["members"]))[:8]
    report["groups"] = {
        "n_verified_games": len(verified), "n_mechanics": len(docs),
        "games_in_groups_of_size": {f"{lo}-{hi}": sum(n * k for n, k in sizes.items() if lo <= n <= hi)
                                    for lo, hi in ((1, 1), (2, 4), (5, 19), (20, 99), (100, 10 ** 9))},
        "largest": [{"key": d["id"], "members": len(d["members"]), "levels_pooled": d["n_levels_pooled"],
                     "titles": Counter(norm_title(by_id[g]["content"]) for g in d["members"]).most_common(3),
                     "rules": mechanics_text(canon[d["members"][0]]).split("RULES\n", 1)[1][:400]}
                    for d in top],
        "levels_pooled": sum(d["n_levels_pooled"] for d in docs),
        "levels_kept": sum(d["n_levels_kept"] for d in docs)}
    print(json.dumps(report["groups"], indent=1)[:3000])

    # 4. every document must compile and be playable
    chk = cached_check(args.out / "doc_check.jsonl", [{"id": d["id"], "text": d["text"]} for d in docs],
                       "ps_check.js", args)
    bad = [d["id"] for d, k in zip(docs, chk) if not k.get("ok")]
    report["documents"] = {"n": len(docs), "fail_engine": len(bad), "fail_examples": bad[:5]}
    docs = [d for d, k in zip(docs, chk) if k.get("ok")]

    # 5. split by components of shared titles / dataset mechanics hashes
    uf = UnionFind()
    for d in docs:
        node = f"k:{d['id']}"
        uf.find(node)
        for gid in d["members"]:
            t = norm_title(by_id[gid]["content"])
            if t:
                uf.union(node, f"t:{t}")
            if by_id[gid].get("mechanics_hash"):
                uf.union(node, f"m:{by_id[gid]['mechanics_hash']}")
    comp_of = {d["id"]: uf.find(f"k:{d['id']}") for d in docs}
    comp_size = Counter(comp_of.values())
    splits = defaultdict(list)
    for d in docs:
        c = comp_of[d["id"]]
        splits[split_of(c, comp_size[c], args.val_frac, args.test_frac, args.max_heldout_component)].append(d)
    for s in splits.values():
        s.sort(key=lambda d: d["id"])
    report["split_rule"] = {"link": ["normalized title", "dataset mechanics_hash"], "val_frac": args.val_frac,
                            "test_frac": args.test_frac, "max_heldout_component": args.max_heldout_component}
    report["largest_components"] = comp_size.most_common(5)
    print({k: len(v) for k, v in splits.items()}, "largest components:", comp_size.most_common(3))
    if args.no_tokenize:
        (args.out / "prep_report.json").write_text(json.dumps(report, indent=2))
        return

    # 6. tokenizer and bins (as prepare_data.py --pretok lines)
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="merged_with_previous"),
         pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator((d["text"] for d in splits["train"]), trainer=trainer)
    tok.save(str(args.out / "tokenizer.json"))
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    assert tok.get_vocab_size() <= 65535
    report.update({"pretok": "lines", "vocab_size": tok.get_vocab_size(), "bos": bos, "eos": eos,
                   "pad": tok.token_to_id("<|pad|>"), "splits": {}})
    for name in ("train", "val", "test"):
        sd = splits[name]
        texts = [d["text"] for d in sd]
        with open(args.out / f"{name}_texts.jsonl", "w") as f:
            for d in sd:
                f.write(json.dumps({"id": d["id"], "text": d["text"]}) + "\n")
        encs = tok.encode_batch(texts)
        lens = np.array([len(e.ids) + 2 for e in encs], dtype=np.int64)
        offsets = np.zeros(len(sd) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lens)
        arr = np.empty(offsets[-1], dtype=np.uint16)
        for i, e in enumerate(encs):
            arr[offsets[i]] = bos
            arr[offsets[i] + 1:offsets[i + 1] - 1] = e.ids
            arr[offsets[i + 1] - 1] = eos
        arr.tofile(args.out / f"{name}.bin")
        np.save(args.out / f"{name}_offsets.npy", offsets)
        n_bytes = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
        with open(args.out / f"{name}_docs.jsonl", "w") as f:
            for d, nb, nt in zip(sd, n_bytes, lens):
                f.write(json.dumps({"id": d["id"], "component": comp_of[d["id"]], "members": d["members"],
                                    "n_levels_pooled": d["n_levels_pooled"], "n_levels_kept": d["n_levels_kept"],
                                    "n_bytes": int(nb), "n_tokens": int(nt)}) + "\n")
        report["splits"][name] = {
            "n_docs": len(sd), "n_components": len({comp_of[d["id"]] for d in sd}),
            "n_member_games": sum(len(d["members"]) for d in sd),
            "n_tokens": int(offsets[-1]), "n_bytes": int(n_bytes.sum()),
            "bytes_per_token": float(n_bytes.sum() / max(1, (lens - 2).sum())),
            "tokens_pct": {str(q): float(np.percentile(lens, q)) for q in (50, 90, 95, 99)},
            "frac_docs_over_8192": float((lens > 8192).mean())}
        print(name, report["splits"][name])
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
