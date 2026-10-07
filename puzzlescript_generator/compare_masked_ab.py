"""Stage 3b readout: masked rules-only sampling (diag-names, diag-full) against the unmasked control
on file (stage 3's diag-rules), on the same 24 levels, model, prompts and seeds; plus the stage-4
decision quantities at the loop's level (protocol.md, "Pre-fit decisions").

    python3 compare_masked_ab.py --control DIR --arms DIR [DIR ...] --level ID --out FILE
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

KEYS = ("format_ok", "playable", "dynamic_if_playable", "puzzle_if_playable", "n_puzzles", "n_distinct_mechanics",
        "behaviour_classes", "behaviour_entropy_bits", "stochastic_if_probed", "train_copy_if_keyed",
        "mean_rules_if_keyed")


def usable(it):
    """Playable, changes the level in the 200-move rollout, and not stochastic under the probes."""
    c = it.get("check") or {}
    return bool(c.get("ok")) and c.get("rollout", {}).get("changed", 0) > 0 and not (it.get("probe") or {}).get("stochastic")


def level_stats(items):
    probed = [it for it in items if (it.get("check") or {}).get("ok") and (it.get("probe") or {}).get("ok")]
    good = [it for it in items if usable(it)]
    return dict(n=len(items), usable=len(good), usable_share=len(good) / len(items) if items else None,
                behaviour_classes=len(Counter(tuple(it["probe"]["probes"]) for it in probed)),
                usable_behaviour_classes=len(Counter(tuple(it["probe"]["probes"]) for it in good if it.get("probe", {}).get("ok"))))


def arm(path, level):
    report = json.loads((path / "level_eval_report.json").read_text())
    items = [json.loads(line) for line in open(path / "samples.jsonl")]
    items = [it for it in items if it["temp"] != "human"]
    by = defaultdict(list)
    for it in items:
        by[it["temp"]].append(it)
    out = {"overall": {t: {k: report["overall"][t].get(k) for k in KEYS} for t in sorted(report["overall"])},
           "usable_share": {t: level_stats(v)["usable_share"] for t, v in sorted(by.items())},
           "level": {t: level_stats([it for it in v if it["level"] == level]) for t, v in sorted(by.items())}}
    if "constrain_stats" in report:
        out["constrain_stats"] = report["constrain_stats"]
        out["constrain"] = report.get("constrain")
    ntok = defaultdict(list)
    for it in items:
        ntok[it["temp"]].append(it["n_new_tokens"])
    out["mean_new_tokens"] = {t: sum(v) / len(v) for t, v in sorted(ntok.items())}
    out["no_eos_share"] = {t: sum(not it["hit_eos"] for it in v) / len(v) for t, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--arms", type=Path, nargs="+", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = {"control": {"path": str(a.control), **arm(a.control, a.level)}}
    for p in a.arms:
        result[p.name] = {"path": str(p), **arm(p, a.level)}
    # stage-4 decision (protocol.md, "Pre-fit decisions")
    full, ctrl = result.get("diag-full"), result["control"]
    if full is not None:
        temps = sorted(full["level"], key=float)
        better = {t: full["level"][t]["usable_share"] > ctrl["level"][t]["usable_share"] for t in temps}
        decision = {"full_beats_control_at_level": better}

        def pick(stats):
            best = max(temps, key=lambda t: (stats[t]["behaviour_classes"], float(t)))
            return best
        sampler_full = {t: better[t] for t in temps}
        # the sampler is decided at the chosen temperature; the temperature by behaviour classes under that
        # sampler. Resolve both orders and require them to agree.
        t_full = pick(full["level"])
        t_none = pick(ctrl["level"])
        if sampler_full[t_full]:
            decision.update(sampler="full", temperature=float(t_full))
        elif not sampler_full[t_none]:
            decision.update(sampler="none", temperature=float(t_none))
        else:
            decision.update(sampler="ambiguous", note="the rule's two orders disagree; see the protocol")
        result["stage4_decision"] = decision
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (v if k == "stage4_decision" else {"overall": v["overall"], "level": v["level"]})
                      for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
