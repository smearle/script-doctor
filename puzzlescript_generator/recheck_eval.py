"""Re-score saved sample_eval.py arms with the current ps_check.js (2026-10-07 fix).

Bug: an in-level message left the headless engine in text mode, after which no win was
checked, so the random-rollout and BFS columns undercounted wins for games with messages.
ps_check.js now makes message output a no-op. Samples, compile and playable verdicts and
the novelty fields are unaffected; this re-runs only the engine checks of every arm and
recomputes the summary with sample_eval.summarize.

    python recheck_eval.py --eval OLD_EVAL_DIR --out NEW_DIR --engine-dir DIR
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from check_games import check_texts
from sample_eval import summarize

ARMS = ("uncond_t1.0", "uncond_t0.8", "levels_t1.0", "human_test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.eval / "eval_report.json").read_text())
    report = {k: v for k, v in old.items() if k not in ARMS}
    report["rechecked_from"] = str(args.eval)
    changed = {}
    for name in ARMS:
        recs = [json.loads(line) for line in open(args.eval / f"{name}.jsonl")]
        checks = check_texts([{"id": r["id"], "text": r["text"]} for r in recs], args.engine_dir,
                             workers=args.workers, dynamics=True, stall_s=90.0)
        novelty = defaultdict(list)
        n_diff = 0
        for r, c in zip(recs, checks):
            n_diff += (c.get("ok") != r["check"].get("ok") or c.get("rollout") != r["check"].get("rollout")
                       or (c.get("bfs") or {}).get("solved") != (r["check"].get("bfs") or {}).get("solved"))
            r["check"] = c
            novelty["exact_train_copy"].append(r["exact_train_copy"])
            if c.get("ok"):
                novelty["rules_copied_if_playable"].append(r["rules_copied"])
            if r.get("frac_ngrams_in_train") is not None:
                novelty["frac_ngrams_in_train"].append(r["frac_ngrams_in_train"])
        report[name] = summarize(recs, checks, novelty)
        changed[name] = n_diff
        with open(args.out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(name, "changed records:", n_diff, json.dumps(report[name]), flush=True)
    report["records_changed_by_recheck"] = changed
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
