"""Level-first PuzzleScript corpus: one document per (mechanics, level), level first, so a
model learns to write mechanics for a given level (format: level_first.py).

The stages mirror prepare_canonical.py, whose functions and stage files are reused:
1-2. extract, canonicalize and verify every dedup representative exactly as stage 2, with
   its stage files and items (extract.jsonl, equiv.jsonl); only games equivalent to their
   original are kept.
3. group by mechanics_key. Each mechanics keeps stage 2's level selection (select_levels on
   its members' pooled distinct levels: at most --max-levels, within the --max-chars budget
   of stage 2's multi-level document), and each kept level becomes one document,
   level_first() of the first member's canonical game. --max-levels (default 8, stage 2
   kept 32) caps how much a many-level game outweighs a one-level game.
4. verify: (a) doc_check: every document's to_standard() text compiles and its level is
   playable (ps_check.js); (b) equivalence on a seeded subset of --equiv-sample documents
   (0: all): ps_equiv.js compares render(c, [level]), the canonical game restricted to that
   level, with to_standard(document), under the level-first renaming. Documents that fail
   either are dropped and counted.
5. split: stage 2's rule per mechanics (mechanics are linked when member games share a
   normalized title or a dataset mechanics_hash; component sizes count mechanics), and
   every document goes to its mechanics' split, so held-out mechanics stay held out.
6. level_prompts.jsonl: one row per val/test document with its prompt (prompt_of) and the
   level's height, width, distinct objects, active objects (those that a rule or win
   condition names, directly or through legend names) and the game's rule count.
7. BPE and token bins as in prepare_canonical.py, so train_lm.py runs unchanged. Documents
   longer than --max-tokens are dropped (counted): train_lm.py would cut their mechanics.

    python prepare_level_first.py --out DIR --revision SHA --engine-dir DIR [--limit N]
        [--equiv-sample N] [--no-tokenize]

Outputs (in --out): extract.jsonl and equiv.jsonl (stage 2's caches: copy them in from a
prepare_canonical.py run to reuse them; their stamps include the engine, so a cache made by
another engine version is recomputed), doc_check.jsonl and doc_equiv.jsonl (caches),
{split}_texts.jsonl, {split}_docs.jsonl, level_prompts.jsonl, prep_report.json, and unless
--no-tokenize tokenizer.json, {split}.bin and {split}_offsets.npy.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from canonicalize import CanonError, from_engine, level_key, mechanics_key, render
from level_first import FormatError, level_first, prompt_of, to_standard
from prepare_canonical import cached_check, select_levels
from prepare_data import SPECIALS, UnionFind, norm_title, normalize, split_of


def equiv_outcome(q) -> str:
    """prepare_canonical.py's verdict on a ps_equiv.js record."""
    if q.get("timeout") or "crash" in q:
        return "checker timeout/crash"
    if "exception" in q:
        return "exception"
    if not q.get("canon_compiled"):
        return "canonical does not compile"
    if not q.get("init_ok"):
        return "initial cells differ"
    if q.get("nondet"):
        return "original does not replay itself"
    if not q.get("dyn_ok"):
        return "dynamics differ"
    return "equivalent"


def verified_games(reps, args, report) -> dict:
    """Stages 1-2 of prepare_canonical.py: id -> canonical game, for the games whose
    canonical form is equivalent to their original."""
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
    verdicts = [equiv_outcome(q) for q in eq]
    report["equivalence"] = dict(Counter(verdicts))
    print("canonicalize errors", dict(canon_err), "| equivalence", report["equivalence"])
    return {it["id"]: canon[it["id"]] for it, v in zip(items, verdicts) if v == "equivalent"}


def active_objects(c) -> set:
    """Objects that a rule or win condition names, directly or through legend names."""
    defs = {name: members for name, _, members in c.legend}
    objs = set(c.objects)
    out, seen = set(), set()
    todo = [t for toks in c.rules + c.wins for t in toks if t in objs or t in defs]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            if n in objs:
                out.add(n)
            else:
                todo += defs[n]
    return out


def build_documents(verified, args):
    """One level-first document per (mechanics, selected level), and the members of each
    mechanics."""
    groups = defaultdict(list)
    for gid, c in verified.items():
        groups[mechanics_key(c)].append(gid)
    docs, errors = [], Counter()
    for key, members in groups.items():
        members.sort()
        pooled, source = [], {}
        for gid in members:
            for lv in verified[gid].levels:
                k = level_key(lv)
                if k not in source:
                    source[k] = gid
                    pooled.append(lv)
        c0 = verified[members[0]]
        active = active_objects(c0)
        for j, lv in enumerate(select_levels(c0, pooled, key, args.max_levels, args.max_chars)):
            text, rename = level_first(c0, lv)
            try:
                standard = to_standard(text)
            except (FormatError, CanonError) as ex:  # a bug in level_first.py: count and show it
                errors[str(ex)[:120]] += 1
                continue
            objs = frozenset().union(*(cell for row in lv for cell in row))
            k = level_key(lv)
            docs.append({"id": f"{key}-{j:02d}", "mechanics_key": key, "level_index": j, "level_key": k,
                         "source_game": source[k], "height": len(lv), "width": len(lv[0]),
                         "n_objects": len(objs), "n_active": len(objs & active), "n_rules": len(c0.rules),
                         "text": text, "standard": standard, "map": rename, "game": c0, "level": lv})
    return docs, dict(groups), errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="smearle/puzzlescript-gists")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--stall-s", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0, help="seeded subset of representatives (tests)")
    ap.add_argument("--max-levels", type=int, default=8,
                    help="levels per mechanics (each is a document, so this caps its weight)")
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--equiv-sample", type=int, default=0, help="documents checked by ps_equiv.js (0: all)")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--val-frac", type=float, default=0.025)
    ap.add_argument("--test-frac", type=float, default=0.025)
    ap.add_argument("--max-heldout-component", type=int, default=40)
    ap.add_argument("--no-tokenize", action="store_true", help="stop before the tokenizer (tests)")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="drop longer documents (with BOS/EOS): train_lm.py would cut their mechanics")
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

    # 1-2. extract, canonicalize, verify (stage 2)
    verified = verified_games(reps, args, report)

    # 3. one document per (mechanics, level)
    docs, groups, errors = build_documents(verified, args)
    if errors:
        print("to_standard failed on level_first output:", dict(errors))
    report["groups"] = {"n_verified_games": len(verified), "n_mechanics": len(groups),
                        "n_documents": len(docs), "to_standard_errors": dict(errors)}

    # 4. (a) every document compiles and is playable; (b) seeded subset equivalent to its source
    chk = cached_check(args.out / "doc_check.jsonl", [{"id": d["id"], "text": d["standard"]} for d in docs],
                       "ps_check.js", args)
    bad = {d["id"] for d, k in zip(docs, chk) if not k.get("ok")}
    n = args.equiv_sample
    sample = sorted(random.Random("equiv").sample(range(len(docs)), n)) if 0 < n < len(docs) else range(len(docs))
    items = [{"id": docs[i]["id"], "orig": render(docs[i]["game"], [docs[i]["level"]]),
              "canon": docs[i]["standard"], "map": docs[i]["map"]} for i in sample]
    eq = cached_check(args.out / "doc_equiv.jsonl", items, "ps_equiv.js", args)
    verdicts = [equiv_outcome(q) for q in eq]
    fails = [(it["id"], v) for it, v in zip(items, verdicts) if v != "equivalent"]
    bad |= {i for i, _ in fails}
    kept = [d for d in docs if d["id"] not in bad]
    report["documents"] = {
        "n": len(docs), "doc_check_fail": sum(not k.get("ok") for k in chk),
        "doc_check_fail_examples": [d["id"] for d, k in zip(docs, chk) if not k.get("ok")][:5],
        "equiv_sampled": len(items), "equivalence": dict(Counter(verdicts)), "equiv_fail_examples": fails[:5],
        "dropped": len(docs) - len(kept),
        "mechanics_dropped": len(groups) - len({d["mechanics_key"] for d in kept})}
    print(json.dumps(report["documents"]))

    # 5. split by components of shared titles / dataset mechanics hashes, per mechanics
    uf = UnionFind()
    keys = sorted({d["mechanics_key"] for d in kept})
    for key in keys:
        node = f"k:{key}"
        uf.find(node)
        for gid in groups[key]:
            t = norm_title(by_id[gid]["content"])
            if t:
                uf.union(node, f"t:{t}")
            if by_id[gid].get("mechanics_hash"):
                uf.union(node, f"m:{by_id[gid]['mechanics_hash']}")
    comp_of = {key: uf.find(f"k:{key}") for key in keys}
    comp_size = Counter(comp_of.values())
    split_of_key = {key: split_of(comp_of[key], comp_size[comp_of[key]], args.val_frac, args.test_frac,
                                  args.max_heldout_component) for key in keys}
    splits = defaultdict(list)
    for d in sorted(kept, key=lambda d: d["id"]):
        splits[split_of_key[d["mechanics_key"]]].append(d)
    report["split_rule"] = {"link": ["normalized title", "dataset mechanics_hash"], "unit": "mechanics",
                            "val_frac": args.val_frac, "test_frac": args.test_frac,
                            "max_heldout_component": args.max_heldout_component}
    report["largest_components"] = comp_size.most_common(5)
    print({k: len(v) for k, v in splits.items()}, "largest components:", comp_size.most_common(3))

    # 6. prompts of the held-out documents
    meta = ("id", "mechanics_key", "level_index", "level_key", "source_game", "height", "width", "n_objects",
            "n_active", "n_rules")
    with open(args.out / "level_prompts.jsonl", "w") as f:
        for name in ("val", "test"):
            for d in splits[name]:
                f.write(json.dumps({**{k: d[k] for k in meta}, "split": name, "prompt": prompt_of(d["text"])})
                        + "\n")

    # 7. tokenizer and bins (as prepare_canonical.py)
    tok = None
    if not args.no_tokenize:
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
                       "pad": tok.token_to_id("<|pad|>")})
    report["splits"] = {}
    for name in ("train", "val", "test"):
        sd = splits[name]
        texts = [d["text"] for d in sd]
        lens, n_long = None, 0
        if tok:
            encs = tok.encode_batch(texts)
            lens = np.array([len(e.ids) + 2 for e in encs], dtype=np.int64)
            keep = lens <= args.max_tokens
            n_long = int((~keep).sum())
            sd = [d for d, k in zip(sd, keep) if k]
            texts = [t for t, k in zip(texts, keep) if k]
            encs = [e for e, k in zip(encs, keep) if k]
            lens = lens[keep]
        with open(args.out / f"{name}_texts.jsonl", "w") as f:
            for d in sd:
                f.write(json.dumps({"id": d["id"], "text": d["text"]}) + "\n")
        n_bytes = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
        s = {"n_docs": len(sd), "n_mechanics": len({d["mechanics_key"] for d in sd}),
             "n_components": len({comp_of[d["mechanics_key"]] for d in sd}),
             "n_member_games": sum(len(groups[k]) for k in {d["mechanics_key"] for d in sd}),
             "n_bytes": int(n_bytes.sum()), "n_dropped_over_max_tokens": n_long}
        if tok:
            offsets = np.zeros(len(sd) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(lens)
            arr = np.empty(offsets[-1], dtype=np.uint16)
            for i, e in enumerate(encs):
                arr[offsets[i]] = bos
                arr[offsets[i] + 1:offsets[i + 1] - 1] = e.ids
                arr[offsets[i + 1] - 1] = eos
            arr.tofile(args.out / f"{name}.bin")
            np.save(args.out / f"{name}_offsets.npy", offsets)
            s.update({"n_tokens": int(offsets[-1]),
                      "bytes_per_token": float(n_bytes.sum() / max(1, (lens - 2).sum())),
                      "tokens_pct": {str(q): float(np.percentile(lens, q)) for q in (50, 90, 95, 99)}})
        with open(args.out / f"{name}_docs.jsonl", "w") as f:
            for i, (d, nb) in enumerate(zip(sd, n_bytes)):
                rec = {**{k: d[k] for k in meta}, "component": comp_of[d["mechanics_key"]],
                       "members": groups[d["mechanics_key"]], "n_bytes": int(nb)}
                if lens is not None:
                    rec["n_tokens"] = int(lens[i])
                f.write(json.dumps(rec) + "\n")
        report["splits"][name] = s
        print(name, s)
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
