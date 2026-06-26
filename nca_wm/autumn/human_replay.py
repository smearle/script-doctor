#!/usr/bin/env python3
"""Replay AutumnBench *human* gameplay action sequences through the engine to make
(state, action, next_state) transitions in the model's palette.

The human data (data/autumn_human.json, 517 users x AutumnBench tasks) records only
ACTIONS + the reset SEED, not states. We re-execute each user's action sequence in the
real Autumn interpreter to recover the states. These are the purposeful, targeted,
mode-switching inputs a person actually makes -- exactly the out-of-distribution poking
the divergence audit found agent/random-trained models fail on.

A task id is e.g. 'gravity_3_planning' / '..._change_detection' / '..._masked_frame_prediction';
the base game name is the part before the last of those suffixes. Each task has an
interactive_phase and a test_phase, each beginning with a 'reset' (seed). We replay each
phase as one ordered episode.

Output mirrors collect_sequences(): states (E, L+1, H, W) uint8, actions (E, L, 3) int.
Encoding uses a SUPPLIED palette (the target model's) so indices align; colors outside it
map to a sentinel index (n_colors) and are counted -- an out-of-palette color is itself a
coverage divergence.
"""
import argparse, json, os, sys
import numpy as np

from nca_wm.autumn.collect import AutumnGame, MARA
from nca_wm.autumn.model import action_to_fields

HUMAN = os.path.join(os.path.dirname(__file__), "data", "autumn_human.json")
SUFFIXES = ("_planning", "_change_detection", "_masked_frame_prediction")
# grid actions we can replay; everything else (start_task, button_click, submission) is UI
GRID_ACTS = {"noop", "up", "down", "left", "right", "click"}


def base_game(task_id):
    for s in SUFFIXES:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id


def encode_grid(g, pal_index):
    """color grid (list of lists of names) -> idx grid; unknown color -> len(pal) sentinel."""
    gs = len(g)
    sentinel = len(pal_index)
    out = np.full((gs, gs), 0, dtype=np.uint8)
    oop = 0
    for y in range(gs):
        row = g[y]
        for x in range(gs):
            i = pal_index.get(row[x])
            if i is None:
                out[y, x] = sentinel
                oop += 1
            else:
                out[y, x] = i
    return out, oop


def replay_phase(env, events, pal_index, gs):
    """Replay one phase's events. Returns (states list[H,W], actions list[3], oop_total)."""
    states, actions, oop_total = [], [], 0
    # find the reset to set the seed; if none, keep env as-is
    started = False
    cur = None
    ep_seed = -1               # the reset seed -> lets the search replay this trajectory
    for e in events:
        at = e.get("actionType")
        if at == "reset":
            seed = e.get("seed", 0)
            try:
                env.reseed(int(seed)); ep_seed = int(seed)
            except Exception:
                env.reseed(0); ep_seed = 0
            cur, o = encode_grid(env.color_grid(), pal_index); oop_total += o
            started = True
            continue
        if not started:
            continue
        if at not in GRID_ACTS:
            continue  # UI action: ends/!affects grid episode boundary-wise; just skip
        # build action tuple in collect convention
        if at == "click":
            x, y = int(e.get("x", 0)), int(e.get("y", 0))
            if not (0 <= x < gs and 0 <= y < gs):
                continue
            act = ("click", x, y)
        else:
            act = (at,)
        env.apply(act)
        nxt, o = encode_grid(env.color_grid(), pal_index); oop_total += o
        states.append(cur)
        actions.append(action_to_fields(act))
        cur = nxt
    # states list are the PRE-states; append final to make L+1 frames
    if states:
        states.append(cur)
    return states, actions, oop_total, ep_seed


def collect_human(game, palette, max_episodes=0, phases=("interactive_phase", "test_phase")):
    pal_index = {c: i for i, c in enumerate(palette)}
    data = json.load(open(HUMAN))
    env = AutumnGame(game, seed=0)
    gs = env.grid_size
    episodes = []         # list of (states (L+1,H,W), actions (L,3))
    seeds = []            # per-episode reset seed (parallel to episodes) for search replay
    n_oop = n_users = 0
    n_changing = n_steps = 0
    for u in data["users"]:
        for task_id, task in u["tasks"].items():
            if base_game(task_id) != game:
                continue
            for ph in phases:
                evs = task.get(ph)
                if not evs:
                    continue
                states, actions, oop, ep_seed = replay_phase(env, evs, pal_index, gs)
                n_oop += oop
                if len(actions) >= 1:
                    S = np.stack(states).astype(np.uint8)       # (L+1,H,W)
                    A = np.array(actions, dtype=np.int64)        # (L,3)
                    episodes.append((S, A))
                    seeds.append(ep_seed)
                    n_steps += len(actions)
                    n_changing += int((S[1:] != S[:-1]).any(axis=(1, 2)).sum())
            n_users += 1
            if max_episodes and len(episodes) >= max_episodes:
                break
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes, dict(grid_size=gs, n_oop=n_oop, n_user_tasks=n_users,
                          n_steps=n_steps, n_changing=n_changing, seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max_episodes", type=int, default=0)
    args = ap.parse_args()
    best = json.load(open(os.path.join(os.path.dirname(__file__), "best.json")))
    run = best.get(args.game)
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "runs", run, "config.json")))
    palette = list(cfg["palette"])
    eps, meta = collect_human(args.game, palette, args.max_episodes)
    L = max((a.shape[0] for _, a in eps), default=0)
    print(f"game={args.game} palette={palette}")
    print(f"  episodes={len(eps)} user_tasks={meta['n_user_tasks']} max_len={L} "
          f"steps={meta['n_steps']} changing={meta['n_changing']} "
          f"({100*meta['n_changing']/max(meta['n_steps'],1):.1f}%) out_of_palette_cells={meta['n_oop']}")
    if args.out and eps:
        # pad to (E, Lmax+1, H, W) with edge-repeat; store true lengths
        H = W = meta["grid_size"]
        E = len(eps)
        states = np.zeros((E, L + 1, H, W), dtype=np.uint8)
        actions = np.zeros((E, L, 3), dtype=np.int64)
        lengths = np.zeros(E, dtype=np.int64)
        for i, (S, A) in enumerate(eps):
            l = A.shape[0]
            states[i, : l + 1] = S
            states[i, l + 1:] = S[-1]
            actions[i, :l] = A
            lengths[i] = l
        np.savez_compressed(args.out, states=states, actions=actions, lengths=lengths,
                            palette=np.array(palette), grid_size=np.int32(H),
                            game=np.str_(args.game))
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
