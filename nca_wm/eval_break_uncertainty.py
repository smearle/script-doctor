"""Break-uncertainty eval for the unconditional jax NCAWorldModel.

Question: on a jump where Mario is <=4 cells below a Step platform (so he can
reach it), action UP, does the model predict P(step present in next) ~ 0.5
(correctly uncertain -- the SAME pre-state stays in mario but breaks in
mario_breakable), or does it collapse to a confident 0/1?

Self-validating: confirms the model reconstructs cache transitions near-perfectly
(=> per-game cache channel order matches the model), identifies the Step + Player
channels from behavior, then reads P(step present|UP) at break cells that have a
Player within 4 rows below.
"""
import glob
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

from nca_wm.models import NCAWorldModel, N_ACTIONS
from nca_wm.state_ops import _unpack_states

SAVE = "nca_wm/logs/mario_uncond_full"
UP = 0
MAX_BELOW = 4   # Mario must be within this many rows below the platform to break it


def load_game(name, n, seed=0):
    f = glob.glob(f"rollout_data/{name}/level_0/bfs_transitions_v5_200000_-1_capall.npz")[0]
    d = np.load(f, allow_pickle=True)
    W = int(d["W"])
    Sp, Nsp, A = d["states"], d["next_states"], np.asarray(d["actions"], np.int64)
    if n < len(Sp):                                   # subsample PACKED rows first (memory!)
        idx = np.sort(np.random.RandomState(seed).choice(len(Sp), n, replace=False))
        Sp, Nsp, A = Sp[idx], Nsp[idx], A[idx]
    S = _unpack_states(Sp, W).astype(np.float32)
    Ns = _unpack_states(Nsp, W).astype(np.float32)
    return S, Ns, A


def main():
    params = pickle.load(open(f"{SAVE}/params_best.pkl", "rb"))
    cfg = json.load(open(f"{SAVE}/config.json"))
    gi = pickle.load(open(f"{SAVE}/game_infos.pkl", "rb"))
    max_C = max(g["n_objs"] for g in gi)
    print(f"max_C={max_C} n_hid={cfg['n_hid']} n_steps={cfg['n_nca_steps']} "
          f"history={cfg['history']} conditional={cfg['conditional']}")

    model = NCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        input_skip=cfg["input_skip"], n_repeats=cfg["n_nca_repeats"],
        history=cfg["history"], axis_pool=cfg["axis_pool"],
        axis_cummax=cfg["axis_cummax"], global_pool=cfg["global_pool"])

    @jax.jit
    def _pred(states, actions):
        logits, _w, _s = model.apply(params, states, jax.nn.one_hot(actions, N_ACTIONS))
        return jax.nn.sigmoid(logits)

    def P(S, A, chunk=1024):
        out = []
        for i in range(0, len(S), chunk):
            out.append(np.asarray(_pred(jnp.asarray(S[i:i+chunk]), jnp.asarray(A[i:i+chunk]))))
        return np.concatenate(out)

    def padC(X):
        return np.pad(X, ((0, 0), (0, max_C - X.shape[1]), (0, 0), (0, 0))) if X.shape[1] < max_C else X

    Sb, Nsb, Ab = load_game("mario_breakable", 12000)
    Sb, Nsb = padC(Sb), padC(Nsb)
    Sm, Nsm, Am = load_game("mario", 12000)
    Sm, Nsm = padC(Sm), padC(Nsm)

    # 1. self-validate alignment + "perfection"
    pb = P(Sb, Ab)
    acc = float(((pb > 0.5) == (Nsb > 0.5)).mean())
    chg = (Sb > 0.5) != (Nsb > 0.5)
    cacc = float(((pb > 0.5) == (Nsb > 0.5))[chg].mean()) if chg.any() else float("nan")
    print(f"\n[self-val] breakable cell-acc={acc:.5f}  changed-cell-acc={cacc:.5f}  "
          "(~1.0 => aligned & ~perfect)")

    # 2. identify Step + Player channels.  Step = breaks in breakable but ~NEVER
    # in mario (a high-churn channel breaks in BOTH -> not the platform).
    pa_b = ((Sb > 0.5) & ~(Nsb > 0.5)).sum(axis=(0, 2, 3))
    pa_m = ((Sm > 0.5) & ~(Nsm > 0.5)).sum(axis=(0, 2, 3))
    per_frame = (Sb > 0.5).sum(axis=(2, 3)).mean(axis=0)
    print("[per-channel] (ch: pres->abs breakable / mario, mean cells/frame)")
    for c in range(max_C):
        if pa_b[c] or pa_m[c] or per_frame[c] > 0.01:
            print(f"   ch{c:2d}: {int(pa_b[c]):6d} / {int(pa_m[c]):6d}   {per_frame[c]:.2f}")
    # auto: channel that breaks in breakable with ~0 breaks in mario
    auto = [(pa_b[c], c) for c in range(max_C) if pa_b[c] > 0 and pa_m[c] <= max(2, 0.02 * pa_b[c])]
    step_auto = int(max(auto)[1]) if auto else -1
    # verified mapping from mario_label_check on this same BFS cache:
    step_ch, player_ch = 6, 7
    print(f"[channels] using Step={step_ch}, Player={player_ch} (verified)  "
          f"| auto-detected Step={step_auto}  "
          f"| Step6: breaks {int(pa_b[6])}x breakable / {int(pa_m[6])}x mario, "
          f"Player7 mean={per_frame[7]:.2f}/frame")

    def below_player(S, h, w, ti):
        # is there a Player within MAX_BELOW rows below (h,w), same column?
        lo, hi = h + 1, min(h + MAX_BELOW, S.shape[2] - 1)
        return bool(S[ti, player_ch, lo:hi + 1, w].any())

    # 2b. FILTER VALIDATION: does {Step present, Mario<=4 below, UP} correspond
    # EXACTLY to {that Step cell disappears next} in mario_breakable?
    is_up = Ab == UP
    broke = (Sb[:, step_ch] > 0.5) & ~(Nsb[:, step_ch] > 0.5)
    f_hit = f_break = b_total = b_in_filt = 0
    for ti in np.where(is_up)[0]:
        for (h, w) in np.argwhere(Sb[ti, step_ch] > 0.5):          # present Step cells
            if below_player(Sb, h, w, ti):
                f_hit += 1
                if broke[ti, h, w]:
                    f_break += 1
        for (h, w) in np.argwhere(broke[ti]):                      # actual breaks
            b_total += 1
            if below_player(Sb, h, w, ti):
                b_in_filt += 1
    print(f"\n[filter-validation, mario_breakable, UP only]")
    print(f"  precision: {f_break}/{f_hit} (Step w/ Mario<=4below) actually break = {f_break/max(f_hit,1):.3f}")
    print(f"  recall:    {b_in_filt}/{b_total} (actual breaks) have Mario<=4below = {b_in_filt/max(b_total,1):.3f}")
    print("  -> both ~1.0 confirms the filter isolates exactly the platform-bonk mechanic")

    # 3. break cells in breakable with Mario <=4 below, model P(step present|UP)
    is_up = Ab == UP
    broke = (Sb[:, step_ch] > 0.5) & ~(Nsb[:, step_ch] > 0.5)
    probs, probs_nofilter = [], []
    for ti in np.where(is_up)[0]:
        for (h, w) in np.argwhere(broke[ti]):
            probs_nofilter.append(float(pb[ti, step_ch, h, w]))
            if below_player(Sb, h, w, ti):
                probs.append(float(pb[ti, step_ch, h, w]))
    probs, probs_nofilter = np.array(probs), np.array(probs_nofilter)
    print(f"\n[BREAK cells, UP] n_all={len(probs_nofilter)}  n_with_Mario<=4below={len(probs)}")
    for tag, pr in [("all break cells", probs_nofilter), ("Mario<=4 below", probs)]:
        if len(pr):
            print(f"  [{tag}] P(step present|UP): mean={pr.mean():.3f} median={np.median(pr):.3f}  "
                  f"hist[0..1]={np.histogram(pr, bins=np.linspace(0,1,11))[0].tolist()}")
    print("  -> ~0.5 correctly uncertain | ~0 collapsed to BREAK | ~1 collapsed to STAY")

    # 3b. EXACT-MATCH ambiguity test: does a break pre-state (breakable) have an
    # identical (Player+Step) config in mario, under UP? If mario then STAYS, the
    # transition is genuinely ambiguous and the confident model has collapsed.
    PS = [player_ch, step_ch]
    def keyset(S, A):
        d = {}
        for ti in np.where(A == UP)[0]:
            d.setdefault((S[ti, PS] > 0.5).tobytes(), []).append(ti)
        return d
    mario_keys = keyset(Sm, Am)
    full_keys = {}                                  # full-state (all channels) keys in mario
    for ti in np.where(Am == UP)[0]:
        full_keys.setdefault((Sm[ti] > 0.5).tobytes(), True)
    n_ps_match = n_ps_match_stay = n_full_match = 0
    for ti in np.where(is_up)[0]:
        if not broke[ti].any():
            continue
        k = (Sb[ti, PS] > 0.5).tobytes()
        if k in mario_keys:
            n_ps_match += 1
            # in mario, does the Step stay at the cell that broke in breakable?
            bh, bw = np.argwhere(broke[ti])[0]
            if any(Nsm[mti, step_ch, bh, bw] > 0.5 for mti in mario_keys[k]):
                n_ps_match_stay += 1
        if (Sb[ti] > 0.5).tobytes() in full_keys:
            n_full_match += 1
    nbrk = int((is_up[:, None] & broke.reshape(len(Sb), -1).any(1)[:, None]).sum())
    print(f"\n[exact-match ambiguity test] over {len(np.where(is_up & broke.reshape(len(Sb),-1).any(1))[0])} breakable break-jumps:")
    print(f"  (Player+Step) config also seen in mario under UP: {n_ps_match}")
    print(f"     ...and mario's Step STAYS there (=> genuine ambiguity): {n_ps_match_stay}")
    print(f"  FULL-state (all channels) also seen in mario under UP: {n_full_match}")
    print("  -> match_stay>0 means identical observable states diverge (model collapsed);")
    print("     0 means the break is observably determined (no ambiguity -> confident is correct)")

    # 4. mario sanity: UP with Mario<=4 below a Step that STAYS -> truth present
    pm = P(Sm, Am)
    stays = (Sm[:, step_ch] > 0.5) & (Nsm[:, step_ch] > 0.5)
    pst = []
    for ti in np.where(Am == UP)[0]:
        for (h, w) in np.argwhere(stays[ti]):
            if below_player(Sm, h, w, ti):
                pst.append(float(pm[ti, step_ch, h, w]))
    pst = np.array(pst)
    if len(pst):
        print(f"\n[mario STAY cells, Mario<=4 below, truth=present] P(step present): "
              f"mean={pst.mean():.3f} median={np.median(pst):.3f} n={len(pst)}")


if __name__ == "__main__":
    main()
