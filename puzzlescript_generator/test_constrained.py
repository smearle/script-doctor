"""Checks of constrained.py against texts the reference engine has judged.

    python test_constrained.py --data DATA --samples SAMPLES.jsonl [--docs N] [--out REPORT.json]

- Every engine-accepted text must pass in both modes, character by character: the human
  originals and the playable generated samples of a level_eval.py run (samples.jsonl), and N
  level-first corpus documents from val and test (DATA/{val,test}_texts.jsonl) cut at RULES.
- Samples the engine rejected are reported by mode: the share the checker stops, and the
  engine's first fatal error class for those it lets through.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constrained import check_text  # noqa: E402

HEADER = "\nRULES\n"


def split(text):
    i = text.index(HEADER) + len(HEADER)
    return text[:i], text[i:]


def first_error(errors):
    for e in errors or []:
        if "Successful" in e or "warning" in e.lower():
            continue
        return re.sub(r'"[^"]*"', '"X"', re.sub(r"^line \d+ : ", "", e))[:70]
    return "none recorded"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-failed", action="store_true", help="only the engine-accepted texts")
    args = ap.parse_args()
    report, failures = {}, []

    accepted = []
    rows = [json.loads(l) for l in open(args.samples)]
    for r in rows:
        if r.get("check", {}).get("ok"):
            accepted.append((r["id"], r["text"]))
    docs = []
    for split_name in ("val", "test"):
        docs += [json.loads(l) for l in open(args.data / f"{split_name}_texts.jsonl")]
    random.Random(args.seed).shuffle(docs)
    accepted += [(d["id"], d["text"]) for d in docs[:args.docs]]
    for mode in ("names", "full"):
        bad = []
        for id_, text in accepted:
            prompt, completion = split(text)
            ok, at = check_text(prompt, completion, mode)
            if not ok:
                line_start = completion.rfind("\n", 0, at) + 1 if at is not None else len(completion)
                line_end = completion.find("\n", line_start)
                bad.append(dict(id=id_, at=at, line=completion[line_start:line_end if line_end >= 0 else None][:160]))
        report[f"engine_accepted_rejected_{mode}"] = dict(n=len(accepted), rejected=len(bad), cases=bad)
        failures += bad

    failed = [] if args.skip_failed else [r for r in rows if "hit_eos" in r and not r.get("check", {}).get("ok")]
    for mode in ("names", "full"):
        stopped, through = 0, Counter()
        for r in failed:
            prompt, completion = split(r["text"])
            ok, _ = check_text(prompt, completion, mode)
            if ok:
                through[first_error(r.get("check", {}).get("errors")) if "check" in r else
                        "format: " + r.get("format_error", "")[:50]] += 1
            else:
                stopped += 1
        report[f"engine_rejected_{mode}"] = dict(n=len(failed), stopped=stopped,
                                                 let_through_by_first_error=through.most_common(12))
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
