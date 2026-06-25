"""Export the co-evolution pool for the viewer + chart pool stats over rounds.

Reads coevolve_run/learnability.jsonl (per-round pool membership + per-env excess
+ added-round) and the pool games in coevolve_run/_scratch/, then:
  - writes game_synth/coevolve_pool/{games,viewer_index.json} (loadable in viewer),
  - charts mean #rules (mechanics) per game and pool size vs round.

    .venv/bin/python -u -m game_synth.coevolve_report
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from game_synth.dedup import strip_noop_rules

_HEADERS = {"OBJECTS", "LEGEND", "SOUNDS", "COLLISIONLAYERS", "RULES", "WINCONDITIONS", "LEVELS"}


def rule_count(code: str) -> int:
    """Number of effective (non-no-op) rules in the RULES section."""
    code = strip_noop_rules(code)
    in_r, n = False, 0
    for line in code.split("\n"):
        u = line.strip().upper()
        if u in _HEADERS:
            in_r = (u == "RULES")
            continue
        if in_r and "->" in line and not line.lstrip().startswith("("):
            n += 1
    return n


def main():
    run = _REPO / "game_synth" / "coevolve_run"
    scratch = run / "_scratch"
    rows = [json.loads(l) for l in (run / "learnability.jsonl").read_text().splitlines()]
    rounds = sorted({r["round"] for r in rows})
    by_round = {}
    for r in rows:
        by_round.setdefault(r["round"], []).append(r)
    final = {r["env"]: r for r in by_round[rounds[-1]]}

    codes, rules = {}, {}
    for env in final:
        p = scratch / f"{env}.txt"
        if p.exists():
            codes[env] = p.read_text(errors="ignore")
            rules[env] = rule_count(codes[env])

    # export pool for the viewer
    pooldir = _REPO / "game_synth" / "coevolve_pool"
    (pooldir / "games").mkdir(parents=True, exist_ok=True)
    idx = []
    for i, (env, meta) in enumerate(sorted(final.items(), key=lambda kv: kv[1]["added"])):
        if env not in codes:
            continue
        fn = f"env_{i:04d}_{env}.txt"
        (pooldir / "games" / fn).write_text(codes[env])
        m = re.search(r"(?im)^\s*title\s+(.+)$", codes[env])
        idx.append({"id": i, "file": "/game_synth/coevolve_pool/games/" + fn, "source": "gp",
                    "title": (m.group(1).strip() if m else env)[:60], "solvable": False,
                    "mechanics": rules[env], "excess": round(meta["excess"], 2),
                    "added_round": meta["added"]})
    (pooldir / "viewer_index.json").write_text(json.dumps(idx))

    # per-round stats
    xs, mean_rules, psize, mean_excess = [], [], [], []
    for rd in rounds:
        envs = [r["env"] for r in by_round[rd] if r["env"] in rules]
        if not envs:
            continue
        xs.append(rd)
        mean_rules.append(sum(rules[e] for e in envs) / len(envs))
        psize.append(len(by_round[rd]))
        mean_excess.append(sum(r["excess"] for r in by_round[rd]) / len(by_round[rd]))
    print("pool exported:", len(idx), "games ->", pooldir / "games")
    print("mean rules/game by round:", list(zip(xs, [round(m, 2) for m in mean_rules])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(xs, mean_rules, "-o", color="tab:blue")
    ax[0].set_xlabel("co-evolution round"); ax[0].set_ylabel("mean #rules (mechanics) / game")
    ax[0].set_title("Pool complexity over rounds"); ax[0].grid(alpha=0.3)
    a2 = ax[0].twinx(); a2.plot(xs, psize, "-s", color="tab:green", alpha=0.7)
    a2.set_ylabel("pool size (distinct envs)", color="tab:green")
    # mechanics-count histogram of the final pool
    from collections import Counter
    c = Counter(rules.values())
    ks = sorted(c)
    ax[1].bar(ks, [c[k] for k in ks], color="tab:purple")
    ax[1].set_xlabel("#rules (mechanics) per game"); ax[1].set_ylabel("count")
    ax[1].set_title(f"Final pool ({len(idx)} envs): mechanics distribution"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"nca_wm/figures/coevolve_pool_mechanics.{e}", dpi=140, bbox_inches="tight")
    print("chart saved nca_wm/figures/coevolve_pool_mechanics.{png,pdf}")


if __name__ == "__main__":
    main()
