"""Sample PuzzleScript games from a trained generator and score them with the reference engine.

Arms:
  uncond_t1.0 / uncond_t0.8  whole games sampled from <|bos|>
  levels_t1.0                new LEVELS sections for held-out test games: the prompt is
                             the test game up to its LEVELS header, so the rules are
                             fixed and only the initial conditions are generated
  human_test                 the held-out human test games themselves (reference)

Every text gets ps_check.js --dynamics: compiles without errors, >= 1 playable level, a
200-step seeded random rollout on the first level (state changes, distinct states, win)
and a 20k-node BFS. Novelty is measured against the train split: exact copies, verbatim
RULES sections, and the fraction of a text's token 32-grams that occur in train; the
human test games give the reference value for each.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from check_games import check_texts
from model import GPT, GPTConfig

SECTION_RE = re.compile(r"(?im)^\s*(OBJECTS|LEGEND|SOUNDS|COLLISIONLAYERS|RULES|WINCONDITIONS|LEVELS)\s*$")
NGRAM = 32
HASH_B = np.uint64(1000003)


def sections(text: str) -> dict:
    marks = [(m.start(), m.end(), m.group(1).upper()) for m in SECTION_RE.finditer(text)]
    out = {}
    for k, (s, e, name) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        out.setdefault(name, text[e:end])
    return out


def norm_block(block: str) -> str:
    lines = [re.sub(r"\s+", " ", ln.strip().lower()) for ln in block.splitlines()]
    return "\n".join(ln for ln in lines if ln and not set(ln) <= {"="})


def content_key(text: str) -> str:
    lines = [ln.rstrip() for ln in text.split("\n")]
    return hashlib.sha1("\n".join(lines).strip("\n").encode()).hexdigest()


def ngram_hashes(tokens: np.ndarray) -> np.ndarray:
    n = len(tokens) - NGRAM + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    t = tokens.astype(np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(NGRAM):
            h = h * HASH_B + t[k:k + n]
    return h


def levels_cut(text: str):
    """Character offset just past the LEVELS header (and its ===== line), or None."""
    m = re.search(r"(?im)^\s*LEVELS\s*$", text)
    if not m:
        return None
    eq = re.match(r"\n\s*=+\s*\n", text[m.end():])  # the usual ===== line under the header
    return m.end() + (eq.end() if eq else 0)


def levels_prompt_ids(tok, text: str):
    """Prompt = the game's own training tokenization up to the first token boundary at or
    after the LEVELS header, so the model continues from a split it saw in training."""
    cut = levels_cut(text)
    if cut is None:
        return None
    enc = tok.encode(text)
    k = next((i for i, (a, _) in enumerate(enc.offsets) if a >= cut), None)
    return None if not k else enc.ids[:k]


def summarize(recs, checks, novelty):
    n = len(recs)
    ok = [c for c in checks if c.get("ok")]
    roll = [c["rollout"] for c in ok if "rollout" in c]
    bfs = [c["bfs"] for c in ok if "bfs" in c]
    s = {"n": n,
         "compile_ok": sum(bool(c.get("compiled")) for c in checks) / n,
         "playable": len(ok) / n,
         "hit_eos": (sum(bool(r.get("hit_eos", True)) for r in recs) / n),
         "timeouts": sum(bool(c.get("timeout")) for c in checks),
         "median_levels_if_playable": float(np.median([c["n_levels"] for c in ok])) if ok else None,
         "dynamic_if_playable": (sum(r["changed"] > 0 for r in roll) / len(roll)) if roll else None,
         "median_distinct_states_200_steps": float(np.median([r["distinct"] for r in roll])) if roll else None,
         "random_win_if_playable": (sum(r["won"] for r in roll) / len(roll)) if roll else None,
         "bfs_solved_if_playable": (sum(b["solved"] for b in bfs) / len(bfs)) if bfs else None,
         "bfs_solved_nontrivial_if_playable":
             (sum(b["solved"] and (b["sol_len"] or 0) >= 5 for b in bfs) / len(bfs)) if bfs else None,
         "playable_dynamic_rate": sum(bool(c.get("ok")) and c.get("rollout", {}).get("changed", 0) > 0
                                      for c in checks) / n}
    s.update({k: float(np.mean(v)) if v else None for k, v in novelty.items()})
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="training output dir with best.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--n-uncond", type=int, default=512)
    ap.add_argument("--n-uncond-cold", type=int, default=256)
    ap.add_argument("--n-level-prompts", type=int, default=64)
    ap.add_argument("--level-samples", type=int, default=4)
    ap.add_argument("--level-max-new", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-human", type=int, default=None, help="cap the human_test reference arm")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.data / "tokenizer.json"))
    prep = json.loads((args.data / "prep_report.json").read_text())
    bos, eos = prep["bos"], prep["eos"]
    dev = args.device
    ck = torch.load(args.run / "best.pt", map_location=dev, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    # test-split loss in nats/token and bits/byte over games that fit the context
    from train_lm import evaluate, load_split
    te_arr, te_off, te_lens = load_split(args.data, "test", cfg.max_seq_len)
    test_loss, _, _ = evaluate(model, te_arr, te_off, te_lens, cfg.max_seq_len, prep["pad"], dev, 131072)
    docs = [json.loads(line) for line in open(args.data / "test_docs.jsonl")]
    fit = np.array([d["n_tokens"] <= cfg.max_seq_len + 1 for d in docs])
    idx = np.where(fit)[0]
    _, fit_nats, _ = evaluate(model, te_arr, te_off[idx], te_lens[idx], cfg.max_seq_len,
                              prep["pad"], dev, 131072)
    report = {"run": str(args.run), "best_step": ck["step"], "val_loss": ck["val_loss"],
              "test_loss_nats_per_token": test_loss,
              "test_bits_per_byte_fitting_games": fit_nats / math.log(2) /
              sum(docs[i]["n_bytes"] for i in idx),
              "test_frac_fit_context": float(fit.mean())}

    arms = {}
    t0 = time.time()
    for name, n, temp in (("uncond_t1.0", args.n_uncond, 1.0), ("uncond_t0.8", args.n_uncond_cold, 0.8)):
        recs = []
        for b0 in range(0, n, args.batch):
            bsz = min(args.batch, n - b0)
            prompt = torch.full((bsz, 1), bos, dtype=torch.long, device=dev)
            outs, done = model.generate(prompt, cfg.max_seq_len - 1, eos, temperature=temp, generator=gen)
            for o, d in zip(outs, done):
                recs.append({"id": f"{name}-{len(recs):05d}", "text": tok.decode(o), "n_tokens": len(o),
                             "hit_eos": bool(d), "tokens": o})
            print(f"{name}: {len(recs)}/{n} ({time.time() - t0:.0f}s)", flush=True)
        arms[name] = recs

    test_texts = [json.loads(line) for line in open(args.data / "test_texts.jsonl")]
    rng = np.random.default_rng(args.seed)
    prompts = []
    for i in rng.permutation(len(test_texts)):
        ids = levels_prompt_ids(tok, test_texts[i]["text"])
        if ids and len(ids) + 1 + args.level_max_new < cfg.max_seq_len:
            prompts.append((test_texts[i]["id"], ids))
        if len(prompts) == args.n_level_prompts:
            break
    recs = []
    for gid, ids in prompts:
        prompt = torch.tensor([[bos] + ids] * args.level_samples, dtype=torch.long, device=dev)
        outs, done = model.generate(prompt, args.level_max_new, eos, temperature=1.0, generator=gen)
        for k, (o, d) in enumerate(zip(outs, done)):
            recs.append({"id": f"levels-{gid}-{k}", "source_game": gid, "text": tok.decode(ids + o),
                         "prompt_tokens": len(ids), "n_tokens": len(o), "hit_eos": bool(d),
                         "tokens": ids + o})
    arms["levels_t1.0"] = recs
    print(f"levels: {len(recs)} samples ({time.time() - t0:.0f}s)", flush=True)
    arms["human_test"] = [{"id": t["id"], "text": t["text"], "tokens": tok.encode(t["text"]).ids}
                          for t in test_texts[:args.max_human]]

    # novelty references from the train split
    train_texts = [json.loads(line) for line in open(args.data / "train_texts.jsonl")]
    train_keys = {content_key(t["text"]) for t in train_texts}
    train_rules = {norm_block(sections(t["text"]).get("RULES", "")) for t in train_texts}
    train_rules.discard("")
    tr = np.fromfile(args.data / "train.bin", dtype=np.uint16)
    train_ngrams = np.unique(ngram_hashes(tr))
    print(f"novelty refs: {len(train_keys)} texts, {len(train_rules)} rule blocks, "
          f"{len(train_ngrams)} distinct {NGRAM}-grams", flush=True)

    for name, recs in arms.items():
        checks = check_texts([{"id": r["id"], "text": r["text"]} for r in recs], args.engine_dir,
                             workers=args.workers, dynamics=True, stall_s=90.0)
        novelty = defaultdict(list)
        for r, c in zip(recs, checks):
            r["check"] = c
            r["exact_train_copy"] = content_key(r["text"]) in train_keys
            rules = norm_block(sections(r["text"]).get("RULES", ""))
            r["rules_copied"] = bool(rules) and rules in train_rules
            h = ngram_hashes(np.asarray(r["tokens"], dtype=np.uint16))
            r["frac_ngrams_in_train"] = (float(np.isin(h, train_ngrams).mean()) if len(h) else None)
            novelty["exact_train_copy"].append(r["exact_train_copy"])
            if c.get("ok"):
                novelty["rules_copied_if_playable"].append(r["rules_copied"])
            if r["frac_ngrams_in_train"] is not None:
                novelty["frac_ngrams_in_train"].append(r["frac_ngrams_in_train"])
        report[name] = summarize(recs, checks, novelty)
        with open(args.out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "tokens"}) + "\n")
        print(name, json.dumps(report[name]), flush=True)

    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
