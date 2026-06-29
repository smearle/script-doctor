"""W&B visualization helpers for the belief WM (NCABeliefModel).

Two things, used by ``mario_nca_belief.train`` when ``--wandb`` is on:

  ig_metrics(model, device, n_obj)  -> flat dict of disambiguation-IG numbers
      (open-air / fresh / known UP IG per world, and calibration gaps), so the
      calibration-vs-training curve is visible in W&B, not just stdout.

  wm_vs_engine_gifs(model, device, n_obj, out_dir) -> {world: gif_path}
      Side-by-side ENGINE (ground truth) vs WM dream (autoregressive q0 rollout)
      over a scripted "walk under a Step and jump" sequence -- the transition the
      WM must learn. Mirrors mario_compare_serve's dream stepping but at the
      model's own n_obj (=20), like the canonical ig_probe Ctx.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded
from nca_wm.active_learning.mario_explore import break_cols, ACTIONS, NA, UP, LEFT, RIGHT, TICK
from nca_wm.state_ops import _multihot_to_objects

H, W = 18, 16
_BACKENDS = None


def get_backends():
    """Compile the two PuzzleScript worlds' sprite backends once (cached)."""
    global _BACKENDS
    if _BACKENDS is not None:
        return _BACKENDS
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    MB._CACHE.mkdir(exist_ok=True); (MB._CACHE / "_scratch").mkdir(exist_ok=True)
    gc._set_materialize_dir(MB._CACHE / "_scratch")
    parser = init_ps_lark_parser(); MB.build_worlds()
    bk = {}
    for name, rel in MB.WORLDS:
        gc._materialize_game(name, (MB.ROOT / rel).read_text())
        b = CppPuzzleScriptBackend(); b.compile_game(parser, name); b.cpp_engine.load_level(0)
        bk[name] = b
    _BACKENDS = bk
    return bk


class _Ctx:
    """Engine + carried belief at the model's n_obj; supports dream prediction."""
    def __init__(self, model, game, device, n_obj, rng):
        self.m, self.dev, self.game = model, device, game
        self.perm = _perm(game.n_obj, n_obj, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, n_obj, H, W)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmaskT = torch.from_numpy((chan[:, None, None] * cell[None]).astype(np.float32)).to(device)[None]
        self.eng = _engine(game.json_str, 0)
        self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
        self.B = model.init_belief(self._obs()) * self.cellT[:, None]

    def _obs(self):
        return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                             self.m.cfg.n_obj, H, W))[None].to(self.dev)

    def grid(self):
        return MB._grid(self.eng)

    def _advance_engine(self, a):
        self.eng.process_input(a)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1); k += 1

    @torch.no_grad()
    def predict(self, a):
        """q0 mixture-marginal next-frame prediction from the carried belief."""
        l0, p0 = self.m.q0_logits(self.B, torch.tensor([a], device=self.dev))
        w = F.softmax(p0, -1)
        prob = (w[..., None, None, None] * torch.sigmoid(l0)).sum(1)        # (1,C,H,W)
        return (prob * self.vmaskT > 0.5).float()

    @torch.no_grad()
    def step_dream(self, a):
        """Predict next frame, advance engine (truth), feed the WM its OWN pred."""
        pred = self.predict(a)
        self._advance_engine(a)
        eng = self._obs()
        self.B = self.m.update_belief(self.B, pred, torch.tensor([a], device=self.dev),
                                      cell_mask=self.cellT)
        return pred, eng


def _script_action(grid, pb, sb, fb):
    """Walk toward nearest break column; jump (UP) when grounded under a Step."""
    cols, info = break_cols(grid, pb, sb, fb)
    if info is None:
        return TICK
    pr, pc, grounded = info
    if not grounded:
        return TICK
    if cols and pc in cols:
        return UP
    if cols:
        tgt = min(cols, key=lambda c: abs(c - pc))
        return RIGHT if tgt > pc else LEFT
    return TICK


def _render(backend, obs_chw, game):
    crop = (obs_chw[:game.n_obj, :game.H, :game.W] > 0.5).astype(np.uint8)
    return backend.render_frame_from_objects(_multihot_to_objects(crop), game.W, game.H)


@torch.no_grad()
def wm_vs_engine_gifs(model, device, n_obj, out_dir, n_steps=20, seed=7, scale=12):
    """Per world, render ENGINE | WM-dream side by side over a scripted rollout."""
    import PIL.Image
    model.eval()
    backends = get_backends()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for game in MB.build_worlds():
        bk = backends[game.gist]
        ctx = _Ctx(model, game, device, n_obj, random.Random(seed))
        frames = []
        # initial frame: engine truth on both sides (WM starts synced)
        eng_np = ctx._obs()[0].cpu().numpy()
        wm_cur = eng_np
        for t in range(n_steps + 1):
            ef = _render(bk, eng_np, game)
            wf = _render(bk, wm_cur, game)
            gap = np.full((ef.shape[0], 6, 3), 60, np.uint8)
            combo = np.concatenate([ef, gap, wf], axis=1)
            img = PIL.Image.fromarray(combo).resize(
                (combo.shape[1] * scale, combo.shape[0] * scale), PIL.Image.NEAREST)
            frames.append(img)
            if t == n_steps:
                break
            a = _script_action(ctx.grid(), ctx.pb, ctx.sb, ctx.fb)
            pred, eng = ctx.step_dream(a)
            eng_np = eng[0].cpu().numpy()
            wm_cur = pred[0].cpu().numpy()
        p = out_dir / f"wm_vs_engine_{game.gist}.gif"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=300, loop=0)
        paths[game.gist] = str(p)
    model.train()
    return paths


class _IGCtx:
    def __init__(self, model, game, device, n_obj, n_samples, rng):
        self.m, self.dev, self.game, self.ns = model, device, game, n_samples
        self.perm = _perm(game.n_obj, n_obj, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, n_obj, H, W)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmaskT = torch.from_numpy((chan[:, None, None] * cell[None]).astype(np.float32)).to(device)[None]
        self.eng = _engine(game.json_str, 0)
        self.pb, self.sb, self.fb = (MB._bit(self.eng, n) for n in ("Player", "Step", "Floor"))
        self.B = model.init_belief(self._obs()) * self.cellT[:, None]

    def _obs(self):
        return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                             self.m.cfg.n_obj, H, W))[None].to(self.dev)

    def grid(self):
        return MB._grid(self.eng)

    @torch.no_grad()
    def ig(self, a):
        return self.m.information_gain(self.B, torch.tensor([a], device=self.dev),
                                       n_samples=self.ns, vmask=self.vmaskT)

    @torch.no_grad()
    def step(self, a):
        self.eng.process_input(a)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1); k += 1
        self.B = self.m.update_belief(self.B, self._obs(),
                                      torch.tensor([a], device=self.dev), cell_mask=self.cellT)


@torch.no_grad()
def ig_metrics(model, device, n_obj, n_samples=12, seed=0):
    """Disambiguation-IG numbers per world for W&B. The KEY health signals:
      fresh_up        : IG of the disambiguating jump when the world is unresolved
      open_air_up     : IG of an open-air (uninformative) jump -- the noisy-TV control
      selectivity     : fresh_up - open_air_up   (>0 == jump beats noise: GOOD)
      resolve_gap     : fresh_up - known_up       (>0 == belief sharpens after a jump)
    """
    from nca_wm.active_learning.mario_explore import navigate_to_break
    model.eval()
    out = {}
    for game in MB.build_worlds():
        ctx = _IGCtx(model, game, device, n_obj, n_samples, random.Random(seed))
        open_air = ctx.ig(UP)
        ok = navigate_to_break(ctx)
        fresh = ctx.ig(UP) if ok else float("nan")
        fresh_max = max((ctx.ig(a) for a in range(NA)), default=float("nan")) if ok else float("nan")
        if ok:
            ctx.step(UP)
        for _ in range(3):
            ctx.step(TICK)
        ok2 = navigate_to_break(ctx)
        known = ctx.ig(UP) if ok2 else float("nan")
        g = game.gist
        out[f"ig/{g}/open_air_up"] = open_air
        out[f"ig/{g}/fresh_up"] = fresh
        out[f"ig/{g}/fresh_max_action"] = fresh_max
        out[f"ig/{g}/known_up"] = known
        out[f"ig/{g}/selectivity"] = fresh - open_air
        out[f"ig/{g}/resolve_gap"] = fresh - known
    model.train()
    return out
