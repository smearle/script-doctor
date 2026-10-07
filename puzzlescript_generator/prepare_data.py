"""Build the token corpus for the PuzzleScript game generator.

Source: the public HF dataset smearle/puzzlescript-gists, pinned to one revision.
Kept rows: one representative per dedup cluster that the reference PuzzleScript engine
compiles without errors and that has at least one playable level (ps_check.js). The
dataset's Lark parse_status is only recorded for comparison: it disagrees with the
engine in both directions.

Split: games are linked when they share a normalized title or a mechanics fingerprint
(name/art-invariant hash of the rules), and each connected component goes wholly to
train, val or test. This keeps template derivatives ("Simple Block Pushing Game") and
an author's successive drafts of one game in a single split, so val/test measure
generation of unseen mechanics rather than recall of near-copies.

Tokenizer: byte-level BPE trained on the train split only. --pretok none: no pre-split
(merges may cross lines; slow to train because every game is one BPE word); --pretok
lines: split after each newline, so a token never spans two lines (grid rows still merge).

Outputs (in --out):
    engine_check.jsonl (one engine result per representative), tokenizer.json,
    {train,val,test}.bin (uint16 tokens), {split}_offsets.npy (int64, n_docs+1),
    {split}_docs.jsonl (id, component, n_bytes, n_tokens), {split}_texts.jsonl, prep_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from check_games import check_texts

TITLE_RE = re.compile(r"(?im)^\s*title\s+(.+?)\s*$")
SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>"]


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def norm_title(text: str):
    m = TITLE_RE.search(text)
    if not m:
        return None
    t = re.sub(r"\s+", " ", m.group(1).strip().lower())
    return t or None


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def split_of(component: str, size: int, val_frac: float, test_frac: float, max_heldout: int) -> str:
    if size > max_heldout:  # giant template components always train
        return "train"
    u = int(hashlib.sha256(component.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="smearle/puzzlescript-gists")
    ap.add_argument("--revision", default=None, help="dataset commit sha (default: current main)")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--pretok", choices=["none", "lines"], default="none")
    ap.add_argument("--val-frac", type=float, default=0.025)
    ap.add_argument("--test-frac", type=float, default=0.025)
    ap.add_argument("--max-heldout-component", type=int, default=40)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    args.out.mkdir(parents=True, exist_ok=True)
    revision = args.revision or HfApi().dataset_info(args.repo).sha
    path = hf_hub_download(args.repo, "data/puzzlescript_games.jsonl", repo_type="dataset",
                           revision=revision)
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    reps = [r for r in rows if r["is_dedup_representative"] and r["content"].strip()]
    for r in reps:
        r["content"] = normalize(r["content"])
    t0 = time.time()
    checks = check_texts([{"id": r["id"], "text": r["content"]} for r in reps], args.engine_dir,
                         workers=args.workers, dynamics=False, stall_s=60.0)
    with open(args.out / "engine_check.jsonl", "w") as f:
        for c in checks:
            f.write(json.dumps(c) + "\n")
    lark_vs_engine = Counter((r["parse_status"] == "ok", bool(c.get("ok"))) for r, c in zip(reps, checks))
    kept = [r for r, c in zip(reps, checks) if c.get("ok")]
    print(f"revision {revision}: {len(rows)} rows, {len(reps)} representatives, {len(kept)} pass "
          f"the engine ({time.time() - t0:.0f}s); (lark_ok, engine_ok) counts {dict(lark_vs_engine)}")

    uf = UnionFind()
    for r in kept:
        node = f"g:{r['id']}"
        uf.find(node)
        if r.get("mechanics_hash"):
            uf.union(node, f"m:{r['mechanics_hash']}")
        t = norm_title(r["content"])
        if t:
            uf.union(node, f"t:{t}")
    comp_of = {r["id"]: uf.find(f"g:{r['id']}") for r in kept}
    comp_size = Counter(comp_of.values())
    splits = defaultdict(list)
    for r in kept:
        c = comp_of[r["id"]]
        splits[split_of(c, comp_size[c], args.val_frac, args.test_frac,
                        args.max_heldout_component)].append(r)
    for s in splits.values():
        s.sort(key=lambda r: r["id"])
    print({k: len(v) for k, v in splits.items()}, "largest components:",
          comp_size.most_common(3))

    tok = Tokenizer(models.BPE())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.pre_tokenizer = byte_level if args.pretok == "none" else pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="merged_with_previous"), byte_level])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.time()
    tok.train_from_iterator((r["content"] for r in splits["train"]), trainer=trainer)
    tok.save(str(args.out / "tokenizer.json"))
    bos, eos = tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")
    assert tok.get_vocab_size() <= 65535
    print(f"tokenizer trained in {time.time() - t0:.0f}s, vocab {tok.get_vocab_size()}")

    report = {"repo": args.repo, "revision": revision, "pretok": args.pretok, "n_rows": len(rows),
              "n_representatives": len(reps), "n_kept": len(kept),
              "engine_filter": {"timeouts": sum(bool(c.get("timeout")) for c in checks),
                                "crashes": sum("crash" in c for c in checks),
                                "lark_ok_vs_engine_ok": {f"lark={a},engine={b}": n
                                                         for (a, b), n in lark_vs_engine.items()}},
              "vocab_size": tok.get_vocab_size(), "bos": bos, "eos": eos,
              "pad": tok.token_to_id("<|pad|>"), "split_rule": {
                  "link": ["normalized title", "mechanics_hash"], "val_frac": args.val_frac,
                  "test_frac": args.test_frac,
                  "max_heldout_component": args.max_heldout_component},
              "largest_components": comp_size.most_common(5), "splits": {}}
    for name in ("train", "val", "test"):
        docs = splits[name]
        texts = [r["content"] for r in docs]
        with open(args.out / f"{name}_texts.jsonl", "w") as f:
            for r in docs:
                f.write(json.dumps({"id": r["id"], "text": r["content"]}) + "\n")
        encs = tok.encode_batch(texts)
        lens = np.array([len(e.ids) + 2 for e in encs], dtype=np.int64)
        offsets = np.zeros(len(docs) + 1, dtype=np.int64)
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
            for r, nb, nt in zip(docs, n_bytes, lens):
                f.write(json.dumps({"id": r["id"], "component": comp_of[r["id"]],
                                    "mechanics_hash": r.get("mechanics_hash"),
                                    "n_bytes": int(nb), "n_tokens": int(nt)}) + "\n")
        report["splits"][name] = {
            "n_docs": len(docs), "n_components": len({comp_of[r["id"]] for r in docs}),
            "n_tokens": int(offsets[-1]), "n_bytes": int(n_bytes.sum()),
            "bytes_per_token": float(n_bytes.sum() / max(1, (lens - 2).sum())),
            "tokens_pct": {str(q): float(np.percentile(lens, q)) for q in (50, 90, 95, 99)},
            "frac_docs_over_8192": float((lens > 8192).mean())}
        print(name, report["splits"][name])
    (args.out / "prep_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
