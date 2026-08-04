"""Do WM-IG-guided trees illuminate trajectories classical search misses?

Compares SEARCH PRODUCTS (no WM training anywhere in this script) at matched
engine-step budgets:

  * active — a saved tree_growth run: graph_L*.npz edge arrays are appended
             in discovery order, so the first B edges = the budget-B search.
             States are reconstructed exactly by replaying the parent
             spanning tree in the engine (restore parent snapshot -> step).
  * bfs / astar — collect_transitions_{bfs,astar} dumps: row order is
             expansion order, so the first-B-rows prefix = a budget-B run.
             (astar's priority = the WINCONDITIONS-derived heuristic, i.e.
             the fixed goal-directed competitor.)

Per (level, budget) and aggregated: unique states, states found ONLY by
active (vs bfs∪astar), rules witnessed (engine telemetry; dumps replayed via
raw LevelBackup injection), wins reached, depth stats.

    python -m nca_wm.active_learning.compare_search_products \
        --game_json .../heroes_of_sokoban.json --levels 0-13 \
        --active_dir /workspace/nwm_logs/tree_growth_heroes \
        --budgets 30000,100000,300000,978525
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm.active_learning.tree_growth import (
    engine_step, new_engine, parse_levels, read_obs)


def state_key(obs):
    return np.packbits(obs.astype(np.uint8), axis=None).tobytes()


# ---------------------------------------------------------------------------
# Active arm: reconstruct states + per-edge rules by replaying the graph
# ---------------------------------------------------------------------------
def reconstruct_active(npz_path, json_str, li, n_obj, hp, wp):
    """Replay the saved graph; return per-edge (step-ordered) records:
    (s2_key, rules_fired frozenset, won, depth_of_s2) plus root key."""
    g = np.load(npz_path)
    e_s, e_a, e_s2 = g["e_s"], g["e_a"], g["e_s2"]
    parent = g["parent"]                      # (n_states, 2): (parent_sid, action)
    n_states = len(parent)
    eng = new_engine(json_str, li)
    baks = {0: eng.backup_level()}
    keys = {0: state_key(read_obs(eng, n_obj, hp, wp))}
    # children of each state in the spanning tree, replayed DFS
    kids = defaultdict(list)
    for sid in range(1, n_states):
        kids[int(parent[sid, 0])].append(sid)
    stack = [0]
    while stack:
        sid = stack.pop()
        for c in kids[sid]:
            eng.restore_level(baks[sid])
            engine_step(eng, int(parent[c, 1]))
            baks[c] = eng.backup_level()
            keys[c] = state_key(read_obs(eng, n_obj, hp, wp))
            stack.append(c)
    # per-edge rules: restore source, step action
    depth = g["depth"]
    recs = []
    for i in range(len(e_s)):
        eng.restore_level(baks[int(e_s[i])])
        eng.clear_rules_fired()
        engine_step(eng, int(e_a[i]))
        rules = frozenset(int(r) for r in eng.get_rules_fired())
        won = bool(eng.is_winning())
        recs.append((keys[int(e_s2[i])], rules, won, int(depth[int(e_s2[i])])))
    return keys[0], recs


# ---------------------------------------------------------------------------
# Dump arms: per-row rules via LevelBackup injection; depths via graph BFS
# ---------------------------------------------------------------------------
def annotate_dump(json_str, game, li, algo, max_iters, n_obj):
    """Return step-ordered records (s2_key, rules, won, None) + state keys
    of s and s2 per row (for depth computation)."""
    from nca_wm.data_collection import collect_unique_transitions
    from puzzlescript_cpp._puzzlescript_cpp import LevelBackup
    d = collect_unique_transitions(json_str, game, level_i=li,
                                   max_iters=max_iters, timeout_ms=300_000,
                                   search_algo=algo)
    W = int(d["W"])
    S = np.unpackbits(d["states"], axis=-1)[..., :W]
    T = np.unpackbits(d["next_states"], axis=-1)[..., :W]
    A = np.asarray(d["actions"], np.int32)
    won = np.asarray(d["wons"]).astype(bool)
    eng = new_engine(json_str, li)
    h, w = S.shape[2], S.shape[3]
    recs, skeys = [], []
    for i in range(len(S)):
        cell = np.zeros((h, w), np.int64)
        for c in range(S.shape[1]):
            cell |= S[i, c].astype(np.int64) << c
        bak = LevelBackup(cell.T.flatten().astype(np.int32).tolist(), w, h)
        eng.restore_level(bak)
        eng.clear_rules_fired()
        engine_step(eng, int(A[i]))
        rules = frozenset(int(r) for r in eng.get_rules_fired())
        sk = S[i].tobytes()      # dedup key within this dump (same layout)
        tk = T[i].tobytes()
        recs.append((tk, rules, bool(won[i]), None))
        skeys.append((sk, tk))
    return recs, skeys


def dump_state_keys_padded(json_str, li, keys_raw, n_obj, hp, wp, shape):
    """Dump keys use native (C,H,W) bytes; active keys use packed padded
    obs. Re-key dump states in the active format so sets are comparable."""
    C, h, w = shape
    out = {}
    for kr in keys_raw:
        arr = np.frombuffer(kr, dtype=np.uint8).reshape(C, h, w)
        g = np.zeros((n_obj, hp, wp), np.float32)
        g[:C, :h, :w] = arr
        out[kr] = state_key(g)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="heroes_of_sokoban")
    p.add_argument("--game_json", required=True)
    p.add_argument("--levels", default="0-13")
    p.add_argument("--active_dir", required=True)
    p.add_argument("--budgets", default="30000,100000,300000,978525")
    p.add_argument("--max_iters", type=int, default=100_000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    json_str = Path(args.game_json).read_text()
    probe = new_engine(json_str, 0)
    n_obj = probe.get_object_count()
    n_levels = probe.get_num_levels()
    levels = parse_levels(args.levels, n_levels)
    budgets = [int(b) for b in args.budgets.split(",")]
    hp = wp = 0
    for li in range(n_levels):
        e = new_engine(json_str, li)
        hp = max(hp, e.get_height()); wp = max(wp, e.get_width())

    # ---- gather step-ordered records per arm, per level ----
    arms: dict[str, dict[int, list]] = {"active": {}, "bfs": {}, "astar": {}}
    for li in levels:
        npz = Path(args.active_dir) / f"graph_L{li}.npz"
        root_key, recs = reconstruct_active(npz, json_str, li, n_obj, hp, wp)
        arms["active"][li] = recs
        for algo in ("bfs", "astar"):
            recs_d, skeys = annotate_dump(json_str, args.game, li, algo,
                                          args.max_iters, n_obj)
            # re-key into active (padded) format
            eng = new_engine(json_str, li)
            h, w = eng.get_height(), eng.get_width()
            raw = {k for pair in skeys for k in pair}
            remap = dump_state_keys_padded(json_str, li, raw, n_obj, hp, wp,
                                           (n_obj, h, w))
            arms[algo][li] = [(remap[tk], r, wn, None)
                              for (tk, r, wn, _), (sk, tk2) in zip(recs_d, skeys)]
        print(f"L{li}: active={len(arms['active'][li])} "
              f"bfs={len(arms['bfs'][li])} astar={len(arms['astar'][li])} "
              f"rows annotated", flush=True)

    # per-level budget share mirrors the active runs (round-robin => equal)
    share = {b: b // len(levels) for b in budgets}
    results = []
    for b in budgets:
        row = {"budget": b}
        per_arm_states = {}
        for arm in arms:
            states, rules, wins, win_lvls = set(), set(), 0, set()
            for li in levels:
                pre = arms[arm][li][:share[b]]
                for tk, rr, wn, _ in pre:
                    states.add((li, tk)); rules |= rr
                    if wn:
                        wins += 1; win_lvls.add(li)
            per_arm_states[arm] = states
            row[arm] = dict(states=len(states), rules=len(rules),
                            win_rows=wins, levels_won=sorted(win_lvls))
        classical = per_arm_states["bfs"] | per_arm_states["astar"]
        only_active = per_arm_states["active"] - classical
        inter = per_arm_states["active"] & classical
        row["active_only_states"] = len(only_active)
        row["active_frac_novel"] = (len(only_active) /
                                    max(len(per_arm_states["active"]), 1))
        row["jaccard_active_vs_classical"] = (
            len(inter) / max(len(per_arm_states["active"] | classical), 1))
        results.append(row)
        print(f"\n=== budget {b:,} (per level {share[b]:,}) ===")
        for arm in ("active", "bfs", "astar"):
            r = row[arm]
            print(f"  {arm:7s} states={r['states']:>8,}  rules={r['rules']:>2}  "
                  f"win_rows={r['win_rows']:>4}  levels_won={r['levels_won']}")
        print(f"  active-only states: {row['active_only_states']:,} "
              f"({row['active_frac_novel']:.1%} of active)  "
              f"jaccard(active, bfs∪astar)={row['jaccard_active_vs_classical']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
