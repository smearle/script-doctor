"""Step 4: online-vs-offline data collection on sokoban, NCA-belief model.

Train a fresh NCA-belief WM on data gathered by each collection policy at matched
budget; eval held-out predictive NLL (q0 at the identified final transition of
held-out navigate trajectories). Expect active (navigate / ig_planner) >= passive
(random) on this sparse-push family, sustained up to a large budget.

    .venv/bin/python -u -m nca_wm.active_learning.sokoban_collect_compare \
        --budgets 64,256,1024 --updates 1000 --seeds 2
"""
from __future__ import annotations

import argparse
import random
import statistics

import torch

from nca_wm.active_learning import collect as C
from nca_wm.active_learning import grid_data as G
from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning import belief_planner as BP
from nca_wm.active_learning.collect import navigate_policy, random_policy
from nca_wm.active_learning.nca_belief_model import BeliefConfig, NCABeliefModel
from nca_wm.active_learning.nca_belief_train import roll_loss

N_STEPS = 8


def collect(family, kind, n, seed, device, ref=None):
    rng = random.Random(seed)
    if kind in ("random", "navigate"):
        pol = random_policy if kind == "random" else navigate_policy
        return G.batch(family, n, N_STEPS, rng, pol)
    if kind == "ig_planner":
        return _collect_planner(family, ref, n, rng, device)
    raise ValueError(kind)


@torch.no_grad()
def _collect_planner(family, ref, n, rng, device):
    """Trajectories whose actions come from the belief expectimax planner."""
    O, A, R = [], [], []
    for _ in range(n):
        mech = rng.choice(family.mechanisms)
        eng = W._new_engine(family.jsons[mech], rng.randrange(family.n_layouts))
        id2b = W._engine_id_to_canon_bit(eng)
        obs = W.read_obs(eng, id2b)
        Bel = ref.init_belief(torch.from_numpy(G.obs_to_grid(obs))[None].to(device))
        grids = [G.obs_to_grid(obs)]; acts = []; resamps = []
        for _ in range(N_STEPS):
            if rng.random() < 0.1:
                a = rng.choice(V.ACTIONS)
            else:
                a, _ = BP.plan_action(ref, Bel, device, depth=3, n_chance=1, n_ig=4)
            ai = V.ACTIONS.index(a)
            s1, s2 = str(rng.getrandbits(40)), str(rng.getrandbits(40))
            bak = eng.backup_level()
            W.step_engine(eng, a, seed=s1); o1 = W.read_obs(eng, id2b)
            eng.restore_level(bak); W.step_engine(eng, a, seed=s2); o2 = W.read_obs(eng, id2b)
            eng.restore_level(bak); W.step_engine(eng, a, seed=s1)
            obs = o1
            grids.append(G.obs_to_grid(o1)); acts.append(ai); resamps.append(G.obs_to_grid(o2))
            Bel = ref.update_belief(Bel, torch.from_numpy(grids[-1])[None].to(device),
                                    torch.tensor([ai], device=device))
        import numpy as np
        O.append(np.stack(grids)); A.append(np.asarray(acts)); R.append(np.stack(resamps))
    import numpy as np
    return (torch.from_numpy(np.stack(O)), torch.from_numpy(np.stack(A)).long(),
            torch.from_numpy(np.stack(R)))


def train_fresh(trajs, cfg, updates, device, seed, batch=32, usage_w=0.02):
    torch.manual_seed(seed)
    O, A, Rr = trajs
    model = NCABeliefModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    rng = random.Random(seed)
    N = O.shape[0]
    model.train()
    for _ in range(updates):
        idx = [rng.randrange(N) for _ in range(min(batch, N))]
        loss = roll_loss(model, O[idx].to(device), A[idx].to(device), Rr[idx].to(device), usage_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def eval_nll(model, eval_trajs, device):
    O, A, _ = eval_trajs
    O, A = O.to(device), A.to(device)
    B = model.init_belief(O[:, 0])
    for t in range(O.shape[1] - 2):
        B = model.update_belief(B, O[:, t + 1], A[:, t])
    l0, p0 = model.q0_logits(B, A[:, -1])
    return model.mixture_nll(l0, p0, O[:, -1]).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=str, default="64,256,1024")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--eval-n", type=int, default=256)
    p.add_argument("--policies", type=str, default="random,navigate")
    p.add_argument("--ref", type=str, default="nca_wm/active_learning/ckpts/nca_belief_sokoban.pt")
    p.add_argument("--n-layouts", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    device = torch.device(args.device)
    budgets = [int(b) for b in args.budgets.split(",")]
    policies = args.policies.split(",")

    family = W.build_family(n_layouts=args.n_layouts, seed=args.seed, form="sokoban",
                            grid_h=7, grid_w=8, style="box_pushable")
    family.activate_geometry()
    cfg = BeliefConfig(n_obj=V.N_CANON, n_act=len(V.ACTIONS))

    ref = None
    if "ig_planner" in policies:
        ck = torch.load(args.ref, map_location=device)
        ref = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
        ref.load_state_dict(ck["model_state"]); ref.eval()

    eval_trajs = G.batch(family, args.eval_n, N_STEPS, random.Random(999), navigate_policy)
    print(f"family {family.mechanisms} | policies {policies} | budgets {budgets} | "
          f"updates {args.updates} seeds {args.seeds}", flush=True)
    print(f"{'budget':>7} | {'policy':>10} | {'NLL mean':>9} | {'std':>6}", flush=True)

    results = {pol: [] for pol in policies}
    for B in budgets:
        for pol in policies:
            vals = []
            for s in range(args.seeds):
                trajs = collect(family, pol, B, 1000 + B + 7919 * s, device, ref=ref)
                model = train_fresh(trajs, cfg, args.updates, device, seed=s)
                vals.append(eval_nll(model, eval_trajs, device))
            m = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            results[pol].append(m)
            print(f"{B:>7} | {pol:>10} | {m:>9.4f} | {sd:>6.4f}", flush=True)

    print("\nHeld-out NLL by policy (lower=better):", flush=True)
    for pol in policies:
        print(f"  {pol:>10}: " + "  ".join(f"{B}:{v:.3f}" for B, v in zip(budgets, results[pol])))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for pol in policies:
            ax.plot(budgets, results[pol], marker="o", label=pol)
        ax.set_xscale("log", base=2); ax.set_xlabel("collection budget (# trajectories)")
        ax.set_ylabel("held-out NLL"); ax.set_title("Sokoban: online vs offline collection (NCA-belief)")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"nca_wm/figures/sokoban_collect_compare.{ext}", dpi=140, bbox_inches="tight")
        print("\nsaved nca_wm/figures/sokoban_collect_compare.{png,pdf}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
