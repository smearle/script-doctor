import json, pickle, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from nca_wm.serve_wm import _build_wm, _unwrap_wm
from nca_wm.train import _run_eval_rollouts_jax

d = os.path.join(REPO, "nca_wm/logs/tsm_pool_diag/pool_on_skip_off")
cfg = json.load(open(f"{d}/config.json"))
params = _unwrap_wm(pickle.load(open(f"{d}/params.pkl", "rb")))
gi = pickle.load(open(f"{d}/game_infos.pkl", "rb")); info = gi[0]
model, mtl = _build_wm(cfg, gi)
maxC = model.n_out; maxH = max(g["H"] for g in gi); maxW = max(g["W"] for g in gi)
tids = info.get("token_ids", []); gt = np.zeros(mtl, np.int32); gm = np.zeros(mtl, bool)
gt[:len(tids)] = tids; gm[:len(tids)] = True


def drift(level, n_act, seed=0, neps=10, steps=50):
    g = np.random.default_rng(seed).integers(0, n_act, size=(neps, steps), dtype=np.int32)
    r = _run_eval_rollouts_jax(model, params, info["json_str"], level, info["n_objs"],
                               maxC, maxH, maxW, n_episodes=neps, max_steps=steps,
                               actions_2d=g, return_both=True, game_tokens=gt, game_mask=gm)
    arf = np.nan_to_num(r["ar_wrong_cells_grid"][:, -1])
    tfmax = np.nanmax(r["tf_wrong_cells_grid"])
    return float(arf.mean()), float(arf.max()), float(tfmax)


print("lvl | 5-action(0-4): ar_mean ar_max tf_max | 4-action(0-3): ar_mean ar_max tf_max", flush=True)
for lvl in [0, 1, 7, 11]:
    a5 = drift(lvl, 5); a4 = drift(lvl, 4)
    print(f"{lvl:>3} |   {a5[0]:7.2f} {a5[1]:6.0f} {a5[2]:6.0f}            |   "
          f"{a4[0]:7.2f} {a4[1]:6.0f} {a4[2]:6.0f}", flush=True)
print("done", flush=True)
