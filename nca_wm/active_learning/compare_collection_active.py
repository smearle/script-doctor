"""Task #2: the LEARNED IG planner as an online data collector.

Unlike the navigate heuristic (which uses privileged seed positions), the IG
planner chooses actions purely from its learned world model's predicted
information gain. On the directional-trigger family only one action (moving RIGHT
into a seed) is informative, so a greedy depth-1 IG planner should DISCOVER it.

Pipeline:
  1. Train a competent reference WM on mixed-policy data (knows the dynamics).
  2. Verify greedy IG ranks the informative action (RIGHT) highest on fresh worlds.
  3. Budget sweep: train from-scratch WMs on data collected by
     {random, ig_greedy, navigate}, eval on a shared held-out informative set.
  Expect ig_greedy ~ navigate >> random — the learned active learner matches the
  oracle and beats passive collection.

    .venv/bin/python -m nca_wm.active_learning.compare_collection_active \
        --budgets 64,128,256,512,1024 --updates 1500
"""
from __future__ import annotations

import argparse
import random

import torch

from nca_wm.active_learning import collect as C
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.compare_collection import (eval_loss,
                                                       make_eval_set,
                                                       train_on_dataset)
from nca_wm.active_learning.inference import estimate_information_gain


def train_reference(family, device, updates, seed=0):
    """Reference WM trained on a 50/50 random+navigate mix (sees full dynamics)."""
    half = 3000
    data = (C.build_dataset(family, C.random_policy, half, seed=seed)
            + C.build_dataset(family, C.navigate_policy, half, seed=seed + 1))
    random.Random(seed).shuffle(data)
    return train_on_dataset(data, device, updates, seed=seed)


def verify_ig_ranking(model, family, device, n=24, seed=7):
    """Mean predicted IG per action on fresh worlds (RIGHT should win)."""
    rng = random.Random(seed)
    sums = {a: 0.0 for a in V.ACTIONS}
    for _ in range(n):
        mech = rng.choice(family.mechanisms)
        li = rng.randrange(family.n_layouts)
        eng = W._new_engine(family.jsons[mech], li)
        id2b = W._engine_id_to_canon_bit(eng)
        hist = V.encode([V.BOS] + V.serialize_obs(W.read_obs(eng, id2b)))
        for a in V.ACTIONS:
            sums[a] += estimate_information_gain(model, hist, a, device, n_samples=8)
    return {a: sums[a] / n for a in V.ACTIONS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=str, default="64,128,256,512,1024")
    p.add_argument("--updates", type=int, default=1500)
    p.add_argument("--ref-updates", type=int, default=3000)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--n-layouts", type=int, default=64)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--grid-h", type=int, default=3)
    p.add_argument("--grid-w", type=int, default=5)
    p.add_argument("--ig-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    budgets = [int(b) for b in args.budgets.split(",")]

    family = W.build_family(n_layouts=args.n_layouts, n_seeds=args.n_seeds,
                            seed=args.seed, form="dir_adjacency",
                            grid_h=args.grid_h, grid_w=args.grid_w,
                            style="left_of_cluster")
    family.activate_geometry()
    print(f"family {family.mechanisms} geom {args.grid_h}x{args.grid_w} dir_adjacency",
          flush=True)

    print("training reference WM...", flush=True)
    ref = train_reference(family, device, args.ref_updates, seed=args.seed)
    ig_rank = verify_ig_ranking(ref, family, device)
    print("  mean IG per action (RIGHT should be highest): "
          + ", ".join(f"{a}={v:+.3f}" for a, v in ig_rank.items()), flush=True)

    ig_greedy = C.make_ig_policy(ref, device, greedy=True, n_chance=args.ig_samples,
                                 epsilon=0.1)
    policies = {"random": C.random_policy, "navigate": C.navigate_policy,
                "ig_greedy": ig_greedy}

    eval_set = make_eval_set(family, args.eval_n, seed=999)
    print(f"eval_n={len(eval_set)} budgets={budgets} updates={args.updates}", flush=True)
    print(f"{'budget':>7} | {'policy':>9} | {'full':>8} | {'resample':>8}", flush=True)

    results = {pol: [] for pol in policies}
    for B in budgets:
        for pol, fn in policies.items():
            data = C.build_dataset(family, fn, B, min_prefix_steps=4,
                                   max_prefix_steps=10, seed=2000 + B)
            model = train_on_dataset(data, device, args.updates, seed=args.seed)
            full, rs = eval_loss(model, eval_set, device)
            results[pol].append(rs)
            print(f"{B:>7} | {pol:>9} | {full:>8.4f} | {rs:>8.4f}", flush=True)

    print("\nResample-block loss (lower=better):", flush=True)
    for pol in policies:
        print(f"  {pol:>9}: " + "  ".join(f"{B}:{v:.3f}" for B, v in zip(budgets, results[pol])))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for pol in policies:
            ax.plot(budgets, results[pol], marker="o", label=pol)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("collection budget (# sequences)")
        ax.set_ylabel("held-out outcome NLL")
        ax.set_title("Learned IG planner vs navigate-oracle vs random collection")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"nca_wm/figures/active_collection_ig_planner.{ext}",
                        dpi=140, bbox_inches="tight")
        print("\nsaved nca_wm/figures/active_collection_ig_planner.{png,pdf}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
