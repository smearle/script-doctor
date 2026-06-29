"""Parameter-matched Transformer baseline on the MATURE data pipeline.

This trains the same :class:`AttnBeliefModel` (causal-transformer belief WM,
incl. its q0/q1/IG heads) used by ``mario_belief.py``, but strips ALL of the
active-learning machinery: there is no heuristic explorer, no BFS start-state
pool, no enumeration, no IG-biased collection. Instead it consumes the exact
same trajectory batches the RecurrentNCA trains on (``train_recurrent``'s
predecessor-chain sampler over the A* transition caches), with the same
train/val split. The ONLY difference vs ``nca_wm/logs/mario2_recurrent`` is the
architecture (Transformer belief vs Recurrent NCA), so the two are an
apples-to-apples, parameter-matched comparison.

Data flow per step:
  build_trajectory_batch -> (states, actions_onehot, targets, valid)  [L=k+1]
  -> O (B,L+1,C,H,W) = [states ; targets[:,-1]]   (deterministic chain)
     A (B,L) action indices,  R (B,L) = O[:,1:]   (q1 resample target),
     CM (B,H,W) real-cell mask, CH (B,C) real-channel mask.

Run JAX on CPU (loader only) so torch owns the GPU:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 \
        -m nca_wm.active_learning.mario_transformer_baseline --train \
        --updates 50000 --batch-size 32 --k 8 --d 208 --d-model 272 \
        --n-layer 5 --d-cond 168 --max-transitions-per-game 60000 \
        --save-dir nca_wm/active_learning/ckpts/mario2_transformer
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig
from nca_wm.train_recurrent import (
    GameData,
    N_ACTIONS,
    build_trajectory_batch,
    load_dataset_from_caches,
)

GAMES_LIST = Path(__file__).resolve().parent / "_mario_recurrent_games.txt"


def _to_attn_batch(states, actions_onehot, targets, valid, n_objs):
    """train_recurrent batch -> AttnBeliefModel (O, A, R, CM, CH) torch tensors."""
    states = np.asarray(states, dtype=np.float32)        # (B,L,C,H,W)
    actions_onehot = np.asarray(actions_onehot)          # (B,L,A)
    targets = np.asarray(targets, dtype=np.float32)      # (B,L,C,H,W)
    valid = np.asarray(valid)                            # (B,L) bool
    B, L, C, H, W = states.shape
    # O has L+1 frames: the L observed inputs plus the final next-state target.
    O = np.concatenate([states, targets[:, -1:]], axis=1)        # (B,L+1,C,H,W)
    A = actions_onehot.argmax(-1).astype(np.int64)               # (B,L)
    R = O[:, 1:]                                                 # (B,L,C,H,W)
    CM = (states.sum(axis=(1, 2)) > 0).astype(np.float32)        # (B,H,W)
    CH = np.zeros((B, C), np.float32)
    CH[:, :n_objs] = 1.0
    return (torch.from_numpy(O), torch.from_numpy(A), torch.from_numpy(R),
            torch.from_numpy(CM), torch.from_numpy(CH),
            torch.from_numpy(valid.astype(np.float32)))


@torch.no_grad()
def _final_tick_q0_logits(model, O, A):
    """MAP-latent q0 logits at the FINAL tick: (B,C,H,W)."""
    N, Tp1, C, H, W = O.shape
    T = Tp1 - 1
    flat = model.enc(O.reshape(N * Tp1, C, H, W))
    spatial = flat.view(N, Tp1, -1, H, W)
    CM = (O[:, :T].sum(dim=(1, 2)) > 0).float()
    cmf = CM[:, None].expand(-1, Tp1, -1, -1).reshape(N * Tp1, H, W)
    from nca_wm.active_learning.attn_belief_model import masked_pool
    pooled = masked_pool(flat, cmf).view(N, Tp1, -1)
    beliefs = model._beliefs(pooled, A)                          # (N,T,d_cond)
    t = T - 1
    cb0 = beliefs[:, t] + model.emb_a(A[:, t])
    l0 = model._dec_all_k(model.dec0, model.emb_z0, spatial[:, t], cb0)  # (N,K,C,H,W)
    p0 = torch.log_softmax(model.prior0(cb0), -1)                # (N,K)
    kstar = p0.argmax(-1)                                        # (N,)
    return l0[torch.arange(N, device=O.device), kstar]          # (N,C,H,W)


@torch.no_grad()
def per_world_eval(model, games, device, k, max_rows, max_C, max_H, max_W,
                   batch=64, seed=0):
    """Final-tick change_err + q0 NLL on each game's held-out val rows."""
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
        n_ch_tot = ch_err_num = q0_sum = q0_n = 0.0
        n_ch_cnt = 0
        for i in range(0, len(rows), batch):
            tr = rows[i:i + batch]
            sb = build_trajectory_batch(gd, tr, k, rng, max_C, max_H, max_W)
            O, A, R, CM, CH, _v = [x.to(device) for x in
                                   _to_attn_batch(*sb, gd.n_objs)]
            logits = _final_tick_q0_logits(model, O, A)          # (B,C,H,W)
            preds = (logits > 0)
            inp = O[:, -2]
            tgt = O[:, -1]
            cell_mask = (inp.sum(1, keepdim=True) > 0)
            cell_mask = cell_mask.expand_as(preds)
            changed = (inp != tgt) & cell_mask
            correct = (preds == (tgt > 0.5))
            nch = int(changed.sum().item())
            if nch:
                ch_err_num += float((~correct[changed]).sum().item())
                n_ch_cnt += nch
            # q0 NLL at final tick (mixture)
            q0, _q1 = model.forward_traj(O, A, R, CM, CH)
            q0_sum += float(q0[:, -1].sum().item())
            q0_n += q0.shape[0]
        ch_err = ch_err_num / n_ch_cnt if n_ch_cnt else 0.0
        out[gd.name] = (ch_err, q0_sum / max(q0_n, 1), n_ch_cnt)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--updates", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="history depth; L=k+1")
    ap.add_argument("--d", type=int, default=208)
    ap.add_argument("--d-model", type=int, default=272)
    ap.add_argument("--n-layer", type=int, default=5)
    ap.add_argument("--d-cond", type=int, default=168)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-transitions-per-game", type=int, default=60_000)
    ap.add_argument("--max-grid-dim", type=int, default=30)
    ap.add_argument("--max-eval-rows", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir",
                    default="nca_wm/active_learning/ckpts/mario2_transformer")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    game_names = [ln.strip() for ln in GAMES_LIST.read_text().splitlines()
                  if ln.strip()]
    print(f"[transformer_baseline] games={game_names}", flush=True)
    t0 = time.time()
    dataset, game_infos = load_dataset_from_caches(
        game_names, (args.max_transitions_per_game or None), args.val_frac,
        ancestor_closed=True, max_grid_dim=args.max_grid_dim, seed=args.seed)
    games = [GameData(g, dataset, info) for g, info in enumerate(game_infos)]
    games = [gd for gd in games if gd.n > 0]
    max_C = max(gd.n_objs for gd in games)
    max_H = max(gd.H for gd in games)
    max_W = max(gd.W for gd in games)
    print(f"[transformer_baseline] dataset ready in {time.time()-t0:.1f}s; "
          f"{len(games)} games; max C/H/W={max_C}/{max_H}/{max_W}", flush=True)
    for gd in games:
        print(f"  {gd.name}: n={gd.n} train={len(gd.train_rows)} "
              f"val={len(gd.val_rows)}", flush=True)

    cfg = AttnConfig(n_obj=max_C, n_act=N_ACTIONS, d=args.d, d_model=args.d_model,
                     n_layer=args.n_layer, d_cond=args.d_cond,
                     max_T=args.k + 3)
    model = AttnBeliefModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[transformer_baseline] params: {n_params:,} | d={cfg.d} "
          f"d_model={cfg.d_model} L={cfg.n_layer} d_cond={cfg.d_cond}", flush=True)
    if not args.train:
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    train_games = [gd for gd in games if len(gd.train_rows) > 0]
    t0 = time.time()
    model.train()
    best = float("inf")
    for step in range(args.updates):
        gd = train_games[rng.integers(len(train_games))]
        tr = rng.choice(gd.train_rows, size=args.batch_size,
                        replace=len(gd.train_rows) < args.batch_size)
        sb = build_trajectory_batch(gd, tr, args.k, rng, max_C, max_H, max_W)
        O, A, R, CM, CH, valid = [x.to(device) for x in
                                  _to_attn_batch(*sb, gd.n_objs)]
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(
                math.pi * min(step / args.updates, 1)))
        q0, q1 = model.forward_traj(O, A, R, CM, CH)             # (B,T) each
        vn = valid.sum().clamp_min(1.0)
        loss = (q0 * valid).sum() / vn + (q1 * valid).sum() / vn
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"step {step:6d}/{args.updates}  loss {loss.item():.4f}  "
                  f"upd/s {(step+1)/max(time.time()-t0,1e-9):.1f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            ev = per_world_eval(model, games, device, args.k, args.max_eval_rows,
                                max_C, max_H, max_W, seed=args.seed)
            mean_ce = np.mean([v[0] for v in ev.values()])
            for name, (ce, nll, nch) in ev.items():
                print(f"  [{step}] {name:16s} change_err {ce:.4f}  "
                      f"q0NLL {nll:.3f}  (n_changed={nch})", flush=True)
            torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                        "step": step, "games": game_names},
                       save_dir / "params.pkl")
            if mean_ce < best:
                best = mean_ce
                torch.save({"model_state": model.state_dict(),
                            "cfg": cfg.__dict__, "step": step,
                            "games": game_names}, save_dir / "params_best.pkl")
    ev = per_world_eval(model, games, device, args.k, args.max_eval_rows,
                        max_C, max_H, max_W, seed=args.seed)
    print("\nFINAL per-world (mature-data Transformer baseline):", flush=True)
    for name, (ce, nll, nch) in ev.items():
        print(f"  {name:16s} change_err {ce:.4f}  q0NLL {nll:.3f}", flush=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg.__dict__,
                "step": args.updates, "games": game_names},
               save_dir / "params.pkl")


if __name__ == "__main__":
    main()
