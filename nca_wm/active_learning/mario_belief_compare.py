"""Head-to-head: Belief Transformer vs Belief Recurrent-NCA (mature-data) on the
disambiguating-jump information-gain probe.

Both belief models are trained on the IDENTICAL mature A* pipeline and param-
matched (~5.3M). This runs the SAME probe on both: navigate under a Step, measure
IG of the disambiguating UP-jump when the world is unresolved (fresh) vs after one
jump has revealed base-vs-breakable (known). A belief WM that supports active
learning shows fresh-UP IG >> known-UP IG and >> open-air control / other actions.

    .venv/bin/python -u -m nca_wm.active_learning.mario_belief_compare \
        --tf nca_wm/active_learning/ckpts/mario2_transformer/params_best.pkl \
        --nca nca_wm/active_learning/ckpts/mario2_nca_belief/params_best.pkl
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded
from nca_wm.active_learning.mario_explore import break_cols, navigate_to_break, ACTIONS, NA, UP, TICK
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig, masked_pool
from nca_wm.active_learning.nca_belief_model import NCABeliefModel, BeliefConfig

H, W = 18, 16


class _BaseCtx:
    """Engine + disambiguation bookkeeping; navigate_to_break drives it via
    grid()/pb/sb/fb/step(). Subclasses implement the belief + ig."""
    def __init__(self, game, device, C, rng):
        self.game, self.dev, self.C = game, device, C
        self.perm = _perm(game.n_obj, C, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, C, H, W)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
        self.eng = _engine(game.json_str, 0)
        self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
        self.last_steps = self._sc(); self.n_break = 0; self.n_disambig_jump = 0

    def _obs(self):
        return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                             self.C, H, W))[None].to(self.dev)

    def grid(self):
        return MB._grid(self.eng)

    def _sc(self):
        return int(((self.grid() >> self.sb) & 1).sum())

    def is_disambig_jump(self, ai):
        cols, info = break_cols(self.grid(), self.pb, self.sb, self.fb)
        return ai == UP and info is not None and info[2] and info[1] in cols

    def _advance(self, ai):
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


class TfCtx(_BaseCtx):
    def __init__(self, wm, game, device, rng):
        super().__init__(game, device, wm.cfg.n_obj, rng)
        self.wm = wm
        self.sp = wm.encode_frame(self._obs())
        self.ps = [masked_pool(self.sp, self.cellT)]; self.pa = [NA]

    @torch.no_grad()
    def _belief(self):
        return self.wm.belief_now(torch.stack(self.ps, 1),
                                  torch.tensor([self.pa], device=self.dev))

    @torch.no_grad()
    def ig(self, ai):
        return self.wm.information_gain(self._belief(), self.sp,
                                        torch.tensor([ai], device=self.dev),
                                        self.cellT, self.vmask, n_samples=self.ns)

    @torch.no_grad()
    def step(self, ai):
        self._advance(ai)
        self.sp = self.wm.encode_frame(self._obs())
        self.ps.append(masked_pool(self.sp, self.cellT)); self.pa.append(ai)


class NcaCtx(_BaseCtx):
    def __init__(self, wm, game, device, rng):
        super().__init__(game, device, wm.cfg.n_obj, rng)
        self.wm = wm
        self.B = wm.init_belief(self._obs()) * self.cellT[:, None]

    @torch.no_grad()
    def ig(self, ai):
        return self.wm.information_gain(self.B, torch.tensor([ai], device=self.dev),
                                        n_samples=self.ns, vmask=self.vmask)

    @torch.no_grad()
    def step(self, ai):
        self._advance(ai)
        self.B = self.wm.update_belief(self.B, self._obs(),
                                       torch.tensor([ai], device=self.dev), cell_mask=self.cellT)


@torch.no_grad()
def probe(make_ctx, label, games, device, n_samples, seed):
    print(f"\n=== {label} ===")
    rows = []
    for game in games:
        ctx = make_ctx(game, random.Random(seed)); ctx.ns = n_samples
        ig_air = ctx.ig(UP)
        ok = navigate_to_break(ctx)
        per_act = {ACTIONS[a]: ctx.ig(a) for a in range(NA)} if ok else {}
        ig_fresh = per_act.get("UP")
        if ok:
            ctx.step(UP)
        for _ in range(3):
            ctx.step(TICK)
        ok2 = navigate_to_break(ctx)
        ig_known = ctx.ig(UP) if ok2 else None
        print(f"  {game.gist}")
        print(f"    open-air UP (control) : {ig_air:+.3f}")
        if ok:
            print(f"    fresh per-act         : " +
                  "  ".join(f"{k} {v:+.3f}" for k, v in per_act.items()))
        kn = f"{ig_known:+.3f}" if ig_known is not None else "(no 2nd break col)"
        gap = f"{ig_fresh - ig_known:+.3f}" if (ig_fresh is not None and ig_known is not None) else "n/a"
        print(f"    fresh UP {('%+.3f'%ig_fresh) if ig_fresh is not None else 'n/a':>8}  "
              f"known UP {kn:>8}  fresh-known {gap}")
        rows.append((game.gist, ig_fresh, ig_known))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf", default="nca_wm/active_learning/ckpts/mario2_transformer/params_best.pkl")
    p.add_argument("--nca", default="nca_wm/active_learning/ckpts/mario2_nca_belief/params_best.pkl")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    games = MB.build_worlds()

    ck = torch.load(args.tf, map_location=device)
    tf = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device); tf.load_state_dict(ck["model_state"]); tf.eval()
    print(f"[TF]  {args.tf} n_obj={ck['cfg']['n_obj']} step={ck.get('step')}")
    ckn = torch.load(args.nca, map_location=device)
    nca = NCABeliefModel(BeliefConfig(**ckn["cfg"])).to(device); nca.load_state_dict(ckn["model_state"]); nca.eval()
    print(f"[NCA] {args.nca} n_obj={ckn['cfg']['n_obj']} step={ckn.get('step')}")

    probe(lambda g, r: TfCtx(tf, g, device, r), "Belief Transformer IG", games, device, args.n_samples, args.seed)
    probe(lambda g, r: NcaCtx(nca, g, device, r), "Belief Recurrent-NCA IG", games, device, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
