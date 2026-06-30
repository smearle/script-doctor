"""Probe 2: what does the matched-BFS TRAINING DATA actually say about the
under-the-platform jump?

For each world, over all cached transitions with action==UP where a Player has a
Step directly above it, tally whether that Step STAYS (present in next_state) or
BREAKS (absent). If the data is ~50/50 across the two worlds for the same input,
the model's collapse to P(step)=0 is miscalibration; if the data is itself
break-dominant at this state, the model is just reproducing the labels.

    .venv/bin/python -u -m nca_wm.active_learning.mario_label_check
"""
from __future__ import annotations

import numpy as np

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine
from nca_wm.active_learning.mario_explore import UP
from nca_wm.active_learning.mario_nca_belief import load_dataset_for_algo, GAMES_LIST
from nca_wm.state_ops import _unpack_states


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-ch", type=int, default=None)
    ap.add_argument("--step-ch", type=int, default=None)
    a = ap.parse_args()
    game_names = [ln.strip() for ln in GAMES_LIST.read_text().splitlines() if ln.strip()]
    if a.player_ch is not None and a.step_ch is not None:
        pb, sb = a.player_ch, a.step_ch   # skip the slow build_worlds/node bridge
    else:
        worlds = {w.gist: w for w in MB.build_worlds()}
        eng = _engine(worlds["mario"].json_str, 0)
        pb, sb = MB._bit(eng, "Player"), MB._bit(eng, "Step")
    print(f"UP action idx = {UP}   Player channel = {pb}   Step channel = {sb}")

    dataset, infos = load_dataset_for_algo(
        game_names, ["bfs"], None, 0.0, 30, 0, cap_tag="all")
    CH = 50000
    for g, info in enumerate(infos):
        states_packed = dataset["per_game_states"][g]
        next_packed = dataset["per_game_next_states"][g]
        actions = np.asarray(dataset["per_game_actions"][g], dtype=np.int64)
        W = int(info["W"])
        unpack = lambda p: _unpack_states(p, W).astype(np.float32)
        # sanity: mean active cells/frame for the Player & Step channels
        s0 = unpack(states_packed[:200])
        print(f"\n  [{info['name']}] sanity: mean Player cells/frame="
              f"{(s0[:, pb] > 0.5).mean() * s0.shape[2] * s0.shape[3]:.2f}  "
              f"mean Step cells/frame="
              f"{(s0[:, sb] > 0.5).mean() * s0.shape[2] * s0.shape[3]:.2f}")
        up = np.where(actions == UP)[0]
        stay = brk = 0
        # global: does the Step channel EVER go present->absent (a break) anywhere,
        # under ANY action? (sanity that ch=sb is a breakable object, not Floor)
        n_all = len(states_packed)
        global_breaks = global_step_cells = 0
        for i in range(0, n_all, CH):
            sl = slice(i, i + CH)
            S = unpack(states_packed[sl]); Ns = unpack(next_packed[sl])
            sp = S[:, sb] > 0.5; nsp = Ns[:, sb] > 0.5
            global_breaks += int((sp & ~nsp).sum())       # cell was Step, now gone
            global_step_cells += int(sp.sum())
        for i in range(0, len(up), CH):
            ch = up[i:i + CH]
            S = unpack(states_packed[ch])
            Ns = unpack(next_packed[ch])
            player = S[:, pb] > 0.5
            step = S[:, sb] > 0.5
            nstep = Ns[:, sb] > 0.5
            hit = player[:, 1:, :] & step[:, :-1, :]      # player row r, step at r-1
            ns_above = nstep[:, :-1, :]
            stay += int(ns_above[hit].sum())
            brk += int((~ns_above[hit]).sum())
        print(f"  {info['name']:16s} GLOBAL: Step-cell present->absent events="
              f"{global_breaks} of {global_step_cells} step-cell-instances "
              f"(any action, anywhere)")
        tot = stay + brk
        print(f"  {info['name']:16s} UP-transitions={len(up):>7}  "
              f"jump-into-Step events={tot:>6}  "
              f"STAY={stay:>6} ({stay/max(tot,1):.3f})  "
              f"BREAK={brk:>6} ({brk/max(tot,1):.3f})")


if __name__ == "__main__":
    main()
