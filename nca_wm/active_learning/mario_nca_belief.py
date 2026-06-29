"""Belief Recurrent-NCA on the MATURE data pipeline — the NCA counterpart of
``mario_transformer_baseline.py``.

Trains the spatial-recurrent ``NCABeliefModel`` (same discrete-latent mixture +
q0/q1/IG heads as the ``AttnBeliefModel`` Belief Transformer) on the EXACT SAME
data the Belief Transformer (``mario2_transformer``) was trained on: the
``train_recurrent`` predecessor-chain sampler over the cached A* transitions
(``load_dataset_from_caches`` / ``build_trajectory_batch``), same train/val
split, no new online collection. The ONLY difference vs ``mario2_transformer`` is
the belief backbone (recurrent spatial NCA vs causal Transformer), so the two are
an apples-to-apples, parameter-matched comparison (~5.3M params each).

  build_trajectory_batch -> (states, actions_onehot, targets, valid)  [L=k+1]
  -> _to_attn_batch -> (O,A,R,CM,CH,valid)
  -> roll the spatial belief over O, per-tick q0/q1 mixture NLL (valid-masked).

Run JAX on CPU (loader only) so torch owns the GPU:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -u -m nca_wm.active_learning.mario_nca_belief --train \
        --updates 50000 --batch-size 32 --k 8 --d 128 --d-b 272 --d-cond 168 \
        --max-transitions-per-game 60000 --save-dir nca_wm/active_learning/ckpts/mario2_nca_belief

    .venv/bin/python -u -m nca_wm.active_learning.mario_nca_belief --probe \
        --ckpt nca_wm/active_learning/ckpts/mario2_nca_belief/params_best.pkl
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from nca_wm.train_recurrent import (
    GameData, N_ACTIONS, build_trajectory_batch, load_dataset_from_caches,
)
from nca_wm.active_learning.mario_transformer_baseline import _to_attn_batch
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig

GAMES_LIST = Path(__file__).resolve().parent / "_mario_recurrent_games.txt"


def load_dataset_for_algo(game_names, algos, cap, val_frac, max_grid_dim, seed,
                          cap_tag="all"):
    """Like train_recurrent.load_dataset_from_caches but for arbitrary search algo
    cache(s) ('astar' / 'bfs' / both). BFS from the shared start yields IDENTICAL
    pre-break context windows across the two Mario worlds (they're identical until
    a break), so the belief can't disambiguate shared states from the path -> it's
    forced to the marginal -> calibrated IG. A* explores the worlds differently
    (it exploits breaking), giving a spurious path cue that collapses the belief."""
    import glob as _glob
    import numpy as _np
    from nca_wm.state_ops import _pack_states, _unpack_states
    from nca_wm.data_collection import ancestor_closed_subsample
    rng = _np.random.default_rng(seed)
    per_states, per_next, per_actions, per_val, game_infos = [], [], [], [], []
    for name in game_names:
        files = []
        for algo in algos:
            # Pin the cap tag (default "all") so we load the matched cross-world
            # cache and never silently concat a stale differently-capped one
            # (e.g. mario_breakable's old cap1000000). Both worlds MUST be the
            # same algo + cap so the belief can't infer the world from a
            # collection-distribution cue instead of an observed break.
            files += sorted(_glob.glob(
                f"rollout_data/{name}/level_*/{algo}_transitions_*_cap{cap_tag}.npz"))
        metas = []
        for f in files:
            with _np.load(f, allow_pickle=True) as d:
                sh = d["states"].shape
                W = int(d["W"])
            if sh[0] > 0:
                metas.append((f, sh[1], sh[2], W, sh[0]))
        if not metas:
            continue
        gC = max(m[1] for m in metas); gH = max(m[2] for m in metas); gW = max(m[3] for m in metas)
        if max(gH, gW) > max_grid_dim:
            continue
        uniform = all((m[1], m[2], m[3]) == (gC, gH, gW) for m in metas)
        Sp, Np, A = [], [], []
        for f, C, H, W, N in metas:
            with _np.load(f, allow_pickle=True) as d:
                s = d["states"]; ns = d["next_states"]; a = _np.asarray(d["actions"], dtype=_np.int64)
            if not uniform:
                pad = ((0, 0), (0, gC - C), (0, gH - H), (0, gW - W))
                s = _pack_states(_np.pad(_unpack_states(s, W), pad).astype(_np.uint8))
                ns = _pack_states(_np.pad(_unpack_states(ns, W), pad).astype(_np.uint8))
            Sp.append(s); Np.append(ns); A.append(a)
        Sp = _np.concatenate(Sp); Np = _np.concatenate(Np); A = _np.concatenate(A)
        if cap and len(Sp) > cap:
            keep = _np.sort(ancestor_closed_subsample(Sp, Np, cap, seed))
            Sp, Np, A = Sp[keep], Np[keep], A[keep]
        n = len(Sp); n_val = int(round(val_frac * n))
        val_idx = _np.sort(rng.permutation(n)[:n_val]).astype(_np.int64)
        per_states.append(Sp); per_next.append(Np); per_actions.append(A); per_val.append(val_idx)
        game_infos.append({"name": name, "n_objs": gC, "H": gH, "W": gW})
        print(f"[load_algo {'+'.join(algos)}] {name}: {n} transitions "
              f"(C/H/W={gC}/{gH}/{gW})", flush=True)
    dataset = {"per_game_states": per_states, "per_game_next_states": per_next,
               "per_game_actions": per_actions, "per_game_val_idx": per_val}
    return dataset, game_infos


def _vmask(CM, CH):
    return CH[:, :, None, None] * CM[:, None, :, :]            # (N,C,H,W)


def roll_loss(model, O, A, R, CM, CH, valid):
    """Per-tick q0/q1 mixture NLL over a trajectory, weighted by `valid` (B,T).

    Belief B starts from frame 0; at tick t the head predicts O[t+1] from the
    belief over frames 0..t-1 + action A[t] (BEFORE ingesting O[t+1]), then the
    belief ingests (O[t+1], A[t]). Mirrors AttnBeliefModel.forward_traj but with
    the recurrent spatial-NCA belief."""
    cm = CM
    vmask = _vmask(CM, CH)
    B = model.init_belief(O[:, 0]) * cm[:, None]
    T = A.shape[1]
    vn = valid.sum().clamp_min(1.0)
    q0_tot = q1_tot = 0.0
    for t in range(T):
        a, onext, oresamp, w = A[:, t], O[:, t + 1], R[:, t], valid[:, t]
        l0, p0 = model.q0_logits(B, a)
        l1, p1 = model.q1_logits(B, a, onext)
        q0_tot = q0_tot + (model.mixture_nll(l0, p0, onext, vmask) * w).sum()
        q1_tot = q1_tot + (model.mixture_nll(l1, p1, oresamp, vmask) * w).sum()
        B = model.update_belief(B, onext, a, cell_mask=cm)
    return (q0_tot + q1_tot) / vn, q0_tot / vn


@torch.no_grad()
def _final_tick_q0_logits(model, O, A, CM):
    """MAP-latent q0 logits at the FINAL tick: (N,C,H,W)."""
    cm = CM
    B = model.init_belief(O[:, 0]) * cm[:, None]
    T = A.shape[1]
    for t in range(T - 1):
        B = model.update_belief(B, O[:, t + 1], A[:, t], cell_mask=cm)
    l0, p0 = model.q0_logits(B, A[:, T - 1])                   # (N,K,C,H,W),(N,K)
    kstar = p0.argmax(-1)
    N = O.shape[0]
    return l0[torch.arange(N, device=O.device), kstar]        # (N,C,H,W)


@torch.no_grad()
def per_world_eval(model, games, device, k, max_rows, max_C, max_H, max_W,
                   batch=64, seed=0):
    """Final-tick change_err + q0 NLL on each game's held-out val rows.
    Identical metric/harness to mario_transformer_baseline.per_world_eval."""
    model.eval()
    out = {}
    for gd in games:
        rows = gd.val_rows
        if len(rows) == 0:
            out[gd.name] = (float("nan"), float("nan"), 0)
            continue
        rng = np.random.default_rng(seed)
        if len(rows) > max_rows:
            rows = rng.choice(rows, size=max_rows, replace=False)
        ch_err_num = q0_sum = q0_n = 0.0
        n_ch_cnt = 0
        for i in range(0, len(rows), batch):
            tr = rows[i:i + batch]
            sb = build_trajectory_batch(gd, tr, k, rng, max_C, max_H, max_W)
            O, A, R, CM, CH, _v = [x.to(device) for x in _to_attn_batch(*sb, gd.n_objs)]
            logits = _final_tick_q0_logits(model, O, A, CM)       # (B,C,H,W)
            preds = (logits > 0)
            inp, tgt = O[:, -2], O[:, -1]
            cell_mask = (inp.sum(1, keepdim=True) > 0).expand_as(preds)
            changed = (inp != tgt) & cell_mask
            correct = (preds == (tgt > 0.5))
            nch = int(changed.sum().item())
            if nch:
                ch_err_num += float((~correct[changed]).sum().item())
                n_ch_cnt += nch
            # mixture q0 NLL at final tick
            B = model.init_belief(O[:, 0]) * CM[:, None]
            T = A.shape[1]
            for t in range(T - 1):
                B = model.update_belief(B, O[:, t + 1], A[:, t], cell_mask=CM)
            l0, p0 = model.q0_logits(B, A[:, T - 1])
            q0_sum += float(model.mixture_nll(l0, p0, O[:, -1], _vmask(CM, CH)).sum().item())
            q0_n += O.shape[0]
        ch_err = ch_err_num / n_ch_cnt if n_ch_cnt else 0.0
        out[gd.name] = (ch_err, q0_sum / max(q0_n, 1), n_ch_cnt)
    model.train()
    return out


# ----------------------------- disambiguation IG probe -----------------------
@torch.no_grad()
def ig_probe(model, device, n_obj, n_samples=12, seed=0):
    """Disambiguating-jump IG (fresh vs known) on the two live worlds. Reads at
    the model's n_obj (=20, identity raw->canonical) so it matches training."""
    from nca_wm.active_learning import mario_belief as MB
    from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded
    from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, ACTIONS, NA, UP, TICK
    C, H, W = n_obj, 18, 16

    class Ctx:
        def __init__(self, game, rng):
            self.game = game
            self.perm = _perm(game.n_obj, C, rng)
            cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, C, H, W)
            self.cellT = torch.from_numpy(cell)[None].to(device)
            self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
            self.eng = _engine(game.json_str, 0)
            self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
            self.B = model.init_belief(self._read()) * self.cellT[:, None]
            self.last_steps = self._sc(); self.n_break = 0; self.n_disambig_jump = 0

        def _read(self):
            return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                                 C, H, W))[None].to(device)

        def grid(self):
            return MB._grid(self.eng)

        def _sc(self):
            return int(((self.grid() >> self.sb) & 1).sum())

        def ig(self, ai):
            return model.information_gain(self.B, torch.tensor([ai], device=device),
                                          n_samples=n_samples, vmask=self.vmask)

        def is_disambig_jump(self, ai):
            cols, info = break_cols(self.grid(), self.pb, self.sb, self.fb)
            return ai == UP and info is not None and info[2] and info[1] in cols

        def step(self, ai):
            dis = self.is_disambig_jump(ai)
            self.eng.process_input(ai)
            k = 0
            while self.eng.is_againing() and k < 50:
                self.eng.process_input(-1); k += 1
            sc = self._sc()
            if sc < self.last_steps:
                self.n_break += 1
            self.last_steps = sc
            if dis:
                self.n_disambig_jump += 1
            self.B = model.update_belief(self.B, self._read(),
                                         torch.tensor([ai], device=device), cell_mask=self.cellT)

    print(f"\nNCA-belief IG probe (n_samples={n_samples})  [UP=jump]")
    for game in MB.build_worlds():
        rng = random.Random(seed)
        ctx = Ctx(game, rng)
        cols0, _ = break_cols(ctx.grid(), ctx.pb, ctx.sb, ctx.fb)
        ig_air = ctx.ig(UP)
        ok = navigate_to_break(ctx)
        per_act = {ACTIONS[a]: ctx.ig(a) for a in range(NA)} if ok else {}
        if ok:
            ctx.step(UP)
        for _ in range(3):
            ctx.step(TICK)
        ok2 = navigate_to_break(ctx)
        ig_known = ctx.ig(UP) if ok2 else None
        print(f"  world {game.gist}")
        print(f"    open-air UP IG (control)   : {ig_air:+.3f}  [break-cols elsewhere={cols0}]")
        if ok:
            print(f"    fresh per-act              : " +
                  "  ".join(f"{kk} {vv:+.3f}" for kk, vv in per_act.items()))
        print(f"    known UP IG (after 1 jump) : "
              f"{ig_known:+.3f}" if ig_known is not None else "    known UP IG: (no 2nd break col)")
        print(f"    breaks={ctx.n_break} disambig-jumps={ctx.n_disambig_jump}")


# ----------------------------- train -----------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    game_names = [ln.strip() for ln in GAMES_LIST.read_text().splitlines() if ln.strip()]
    print(f"[nca_belief] games={game_names}", flush=True)
    t0 = time.time()
    if args.algo == "astar":
        dataset, game_infos = load_dataset_from_caches(
            game_names, (args.max_transitions_per_game or None), args.val_frac,
            ancestor_closed=True, max_grid_dim=args.max_grid_dim, seed=args.seed)
    else:
        algos = ["astar", "bfs"] if args.algo == "both" else [args.algo]
        dataset, game_infos = load_dataset_for_algo(
            game_names, algos, (args.max_transitions_per_game or None),
            args.val_frac, args.max_grid_dim, args.seed, cap_tag=args.cap_tag)
    games = [GameData(g, dataset, info) for g, info in enumerate(game_infos)]
    games = [gd for gd in games if gd.n > 0]
    max_C = max(gd.n_objs for gd in games)
    max_H = max(gd.H for gd in games)
    max_W = max(gd.W for gd in games)
    print(f"[nca_belief] dataset ready in {time.time()-t0:.1f}s; {len(games)} games; "
          f"max C/H/W={max_C}/{max_H}/{max_W}", flush=True)
    for gd in games:
        print(f"  {gd.name}: n={gd.n} train={len(gd.train_rows)} val={len(gd.val_rows)}", flush=True)

    cfg = BeliefConfig(n_obj=max_C, n_act=N_ACTIONS, d=args.d, d_b=args.d_b,
                       d_cond=args.d_cond, K=args.K, nca_steps=args.nca_steps)
    model = NCABeliefModel(cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[nca_belief] params: {nparam:,} | d={cfg.d} d_b={cfg.d_b} "
          f"nca_steps={cfg.nca_steps} K={cfg.K}", flush=True)

    use_wandb = getattr(args, "wandb", False)
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=(args.wandb_name or Path(args.save_dir).name),
                   config={**vars(args), **cfg.__dict__, "nparam": nparam})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    train_games = [gd for gd in games if len(gd.train_rows) > 0]
    t0 = time.time(); model.train(); best = float("inf")
    for step in range(args.updates):
        gd = train_games[rng.integers(len(train_games))]
        tr = rng.choice(gd.train_rows, size=args.batch_size,
                        replace=len(gd.train_rows) < args.batch_size)
        sb = build_trajectory_batch(gd, tr, args.k, rng, max_C, max_H, max_W)
        O, A, R, CM, CH, valid = [x.to(device) for x in _to_attn_batch(*sb, gd.n_objs)]
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * min(step / args.updates, 1)))
        loss, _q0 = roll_loss(model, O, A, R, CM, CH, valid)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            print(f"step {step:6d}/{args.updates}  loss {loss.item():.4f}  "
                  f"upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/loss": float(loss.item()),
                           "train/q0_nll": float(_q0.item()),
                           "train/q1_nll": float(loss.item() - _q0.item()),
                           "train/lr": opt.param_groups[0]["lr"],
                           "perf/upd_per_s": (step + 1) / max(time.time() - t0, 1e-9)},
                          step=step)
        if step > 0 and step % args.eval_every == 0:
            ev = per_world_eval(model, games, device, args.k, args.max_eval_rows,
                                max_C, max_H, max_W, seed=args.seed)
            mean_ce = float(np.mean([v[0] for v in ev.values()]))
            for name, (ce, nll, nch) in ev.items():
                print(f"  [{step}] {name:16s} change_err {ce:.4f}  q0NLL {nll:.3f}  "
                      f"(n_changed={nch})", flush=True)
            if use_wandb:
                import wandb
                from nca_wm.active_learning.mario_belief_viz import ig_metrics, wm_vs_engine_gifs
                logd = {"val/mean_change_err": mean_ce}
                for name, (ce, nll, nch) in ev.items():
                    logd[f"val/{name}/change_err"] = ce
                    logd[f"val/{name}/q0_nll"] = nll
                # disambiguation-IG calibration curve (the whole point of the WM)
                logd.update(ig_metrics(model, device, max_C, n_samples=args.n_samples, seed=args.seed))
                # side-by-side ENGINE vs WM-dream GIFs every --gif-every evals
                eval_idx = step // args.eval_every
                if args.gif_every and eval_idx % args.gif_every == 0:
                    try:
                        gifs = wm_vs_engine_gifs(model, device, max_C, save_dir / "gifs",
                                                 seed=args.seed)
                        for g, p in gifs.items():
                            logd[f"gif/{g}"] = wandb.Video(p, fps=4, format="gif")
                    except Exception as e:
                        print(f"[wandb] gif render failed: {e}", flush=True)
                wandb.log(logd, step=step)
            ckpt = {"model_state": model.state_dict(), "cfg": cfg.__dict__,
                    "step": step, "games": game_names}
            torch.save(ckpt, save_dir / "params.pkl")
            # Save EVERY eval checkpoint so the calibration-vs-training curve can be
            # OBSERVED post-hoc (NOT for IG-based selection — that's unscalable).
            torch.save(ckpt, save_dir / f"params_step{step}.pkl")
            if mean_ce < best:
                best = mean_ce
                torch.save(ckpt, save_dir / "params_best.pkl")
    ev = per_world_eval(model, games, device, args.k, args.max_eval_rows,
                        max_C, max_H, max_W, seed=args.seed)
    print("\nFINAL per-world (Belief Recurrent-NCA, mature data):", flush=True)
    for name, (ce, nll, nch) in ev.items():
        print(f"  {name:16s} change_err {ce:.4f}  q0NLL {nll:.3f}", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                "step": args.updates, "games": game_names}, save_dir / "params.pkl")
    ig_probe(model, device, max_C, args.n_samples, args.seed)
    if use_wandb:
        import wandb
        wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/mario2_nca_belief/params_best.pkl")
    ap.add_argument("--updates", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="history depth; L=k+1")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--d-b", type=int, default=272)
    ap.add_argument("--d-cond", type=int, default=168)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--nca-steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--algo", choices=["astar", "bfs", "both"], default="astar",
                    help="search-cache source. bfs gives matched cross-world "
                         "contexts (calibration); astar's differ (collapse).")
    ap.add_argument("--cap-tag", default="all",
                    help="transition-cache cap tag to load for --algo bfs/both "
                         "(e.g. 'all'). Pins a single matched cache per world so "
                         "both games use identical collection (same algo+cap).")
    ap.add_argument("--max-transitions-per-game", type=int, default=60_000)
    ap.add_argument("--max-grid-dim", type=int, default=30)
    ap.add_argument("--max-eval-rows", type=int, default=2048)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir", default="nca_wm/active_learning/ckpts/mario2_nca_belief")
    ap.add_argument("--wandb", action="store_true", help="log curves/val/IG + WM-vs-engine GIFs to W&B")
    ap.add_argument("--wandb-project", default="mario-belief-wm")
    ap.add_argument("--wandb-name", default=None, help="run name (default: save-dir basename)")
    ap.add_argument("--gif-every", type=int, default=1,
                    help="render side-by-side WM-vs-engine GIFs every N evals (0=off)")
    args = ap.parse_args()
    if args.train:
        train(args)
    elif args.probe:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.ckpt, map_location=device)
        model = NCABeliefModel(BeliefConfig(**ck["cfg"])).to(device)
        model.load_state_dict(ck["model_state"]); model.eval()
        for q in model.parameters():
            q.requires_grad_(False)
        ig_probe(model, device, ck["cfg"]["n_obj"], args.n_samples, args.seed)
    else:
        print("pass --train or --probe")


if __name__ == "__main__":
    main()
