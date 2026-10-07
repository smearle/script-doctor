"""Regression of a new reference-engine version against the current one (engine_regress.js).

Runs every dedup representative of the pinned dataset revision through both engines and
counts compile-verdict changes and, for games both accept, differences in the parse, the
compiled initial levels and 60-step seeded dynamics on the first two levels.

    python regress_engines.py --old DIR --new DIR --revision SHA --out DIR [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from check_games import check_texts
from prepare_data import normalize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--repo", default="smearle/puzzlescript-gists")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(args.repo, "data/puzzlescript_games.jsonl", repo_type="dataset", revision=args.revision)
    reps = [r for r in map(json.loads, open(path, encoding="utf-8"))
            if r["is_dedup_representative"] and r["content"].strip()]
    if args.limit:
        reps = random.Random(0).sample(reps, args.limit)
    os.environ["PS_NEW_ENGINE"] = str(args.new.resolve())
    res = check_texts([{"id": r["id"], "text": normalize(r["content"])} for r in reps], args.old,
                      workers=args.workers, stall_s=120.0, script="engine_regress.js", progress_s=300)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "engine_regress.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    verdict = Counter()
    both = Counter()
    for r in res:
        if r.get("timeout") or "crash" in r:
            verdict["checker timeout/crash"] += 1
        elif "exception" in r:
            verdict["exception"] += 1
        else:
            verdict[f"old_ok={r['ok'][0]} new_ok={r['ok'][1]}"] += 1
            if all(r["ok"]):
                for k in ("same_parse", "same_init", "same_dyn"):
                    both[f"{k}={r[k]}"] += 1
    report = {"old": str(args.old), "new": str(args.new), "revision": args.revision, "n": len(res),
              "verdicts": dict(verdict), "both_ok": dict(both)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
