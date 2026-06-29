"""NCA-WM-Player-style viewer for the two-world Mario belief WM.

Left = real engine (ground truth). Right = the belief WM, rolled out
AUTOREGRESSIVELY from its own previous prediction (NOT re-fed the engine state).
Both are driven by the same action, so you can watch the WM track — or drift from
— the engine. Real PuzzleScript sprites (rendered by the C++ backend). Arrow keys
drive it; "re-sync WM" snaps the WM state back to the engine's current frame.

  arrows = UP(jump)/DOWN(wait)/LEFT/RIGHT, space = ACTION(shoot)

    .venv/bin/python -u -m nca_wm.active_learning.mario_serve --port 8770 --device cuda
Then port-forward and open http://localhost:8770
"""
from __future__ import annotations

import argparse
import base64
import io

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request

from nca_wm.state_ops import _multihot_to_objects
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig, masked_pool
from nca_wm.active_learning.mario_explore import break_cols, NA, ACTIONS
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded

app = Flask(__name__)
S = {}


def build_render_backends():
    """Compile each world with a sprite-capable backend (for render_frame_from_objects)."""
    from nca_wm import game_curriculum as gc
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser
    MB._CACHE.mkdir(exist_ok=True)
    gc._set_materialize_dir(MB._CACHE / "_scratch")
    (MB._CACHE / "_scratch").mkdir(exist_ok=True)
    parser = init_ps_lark_parser()
    games = MB.build_worlds()
    backends = {}
    for name, rel in MB.WORLDS:
        gc._materialize_game(name, (MB.ROOT / rel).read_text())
        b = CppPuzzleScriptBackend()
        b.compile_game(parser, name)
        b.cpp_engine.load_level(0)
        backends[name] = b
    return {g.gist: g for g in games}, backends


class CompareCtx:
    """Engine ground truth + autoregressive WM rollout, both stepped by one action."""
    def __init__(self, wm, game, backend, device, rng):
        self.wm, self.dev, self.game, self.backend = wm, device, game, backend
        self.perm = _perm(game.n_obj, MB.CMAX, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, MB.CMAX, MB.HMAX, MB.WMAX)
        self.cellT = torch.from_numpy(cell)[None].to(device)
        self.vmask = torch.from_numpy(chan[:, None, None] * cell[None]).to(device)[None]
        self.eng = _engine(game.json_str, 0); self.eng.set_track_rules_fired(True)
        self.sb = MB._bit(self.eng, "Step")
        self.step_i = 0; self.n_break = 0; self.last_steps = self._stepcount()
        self.mode = "dream"        # "dream" = autoregressive; "predict" = one-step teacher-forced
        self.resync()

    def _engine_obs(self):
        return torch.from_numpy(_read_padded(self.eng, self.game.n_obj, self.perm,
                                             MB.CMAX, MB.HMAX, MB.WMAX))[None].to(self.dev)

    def _stepcount(self):
        return int(((MB._grid(self.eng) >> self.sb) & 1).sum())

    @torch.no_grad()
    def resync(self):
        self.wm_board = self._engine_obs().clone()
        self.sp = self.wm.encode_frame(self.wm_board)
        self.ps = [masked_pool(self.sp, self.cellT)]; self.pa = [NA]

    @torch.no_grad()
    def _belief(self):
        return self.wm.belief_now(torch.stack(self.ps, 1),
                                  torch.tensor([self.pa], device=self.dev))

    @torch.no_grad()
    def step(self, a):
        # WM prediction of the next frame (q0 mixture marginal, thresholded, off-grid masked)
        # from the CURRENT input encoding self.sp + belief-so-far + action.
        cb = self._belief() + self.wm.emb_a(torch.tensor([a], device=self.dev))
        l0 = self.wm._dec_all_k(self.wm.dec0, self.wm.emb_z0, self.sp, cb)
        p0 = F.softmax(self.wm.prior0(cb), -1)
        prob = (p0[..., None, None, None] * torch.sigmoid(l0)).sum(1)
        pred_next = (prob * self.vmask > 0.5).float()
        self.wm_board = pred_next                       # displayed prediction
        # engine ground truth
        self.eng.clear_rules_fired()
        self.eng.process_input(a)            # action index == engine input id (5 = realtime tick)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1); k += 1
        sc = self._stepcount()
        if sc < self.last_steps:
            self.n_break += 1
        self.last_steps = sc
        self.step_i += 1
        # next input + belief frame: own prediction (dream) or true observation (predict)
        nxt = pred_next if self.mode == "dream" else self._engine_obs()
        self.sp = self.wm.encode_frame(nxt)
        self.ps.append(masked_pool(self.sp, self.cellT)); self.pa.append(a)

    @torch.no_grad()
    def ig_all(self, n):
        bel = self._belief()
        return [round(self.wm.information_gain(bel, self.sp,
                torch.tensor([a], device=self.dev), self.cellT, self.vmask, n), 3)
                for a in range(NA)]

    def _png(self, obs):
        crop = (obs[:self.game.n_obj, :self.game.H, :self.game.W] > 0.5).astype(np.uint8)
        objs = _multihot_to_objects(crop)
        frame = self.backend.render_frame_from_objects(objs, self.game.W, self.game.H)
        buf = io.BytesIO(); PIL.Image.fromarray(frame).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def payload(self, nsamp, with_ig=True):
        eng_obs = self._engine_obs()[0].cpu().numpy()
        wm_obs = self.wm_board[0].cpu().numpy()
        H, Wd = self.game.H, self.game.W
        e = eng_obs[:self.game.n_obj, :H, :Wd] > 0.5
        w = wm_obs[:self.game.n_obj, :H, :Wd] > 0.5
        cols, info = break_cols(MB._grid(self.eng), MB._bit(self.eng, "Player"),
                                self.sb, MB._bit(self.eng, "Floor"))
        under = info is not None and info[2] and info[1] in cols
        return dict(real=self._png(eng_obs), wm=self._png(wm_obs),
                    step=self.step_i, l1=int((e != w).sum()), diff_cells=int((e != w).any(0).sum()),
                    n_break=self.n_break, under_platform=bool(under), mode=self.mode,
                    ig=(self.ig_all(nsamp) if with_ig else None), actions=ACTIONS, world=self.game.gist)


def new_ctx(world):
    import random
    return CompareCtx(S["wm"], S["games"][world], S["backends"][world], S["dev"], random.Random(0))


@app.route("/")
def index():
    return HTML


@app.route("/reset")
def reset():
    S["ctx"] = new_ctx(request.args.get("world", "mario"))
    return jsonify(S["ctx"].payload(S["nsamp"]))


@app.route("/step")
def step():
    S["ctx"].step(int(request.args.get("a", 0)))
    # realtime auto-ticks pass fast=1 to skip the (slow) 6-action IG so ticking stays smooth
    with_ig = request.args.get("fast", "0") != "1"
    return jsonify(S["ctx"].payload(S["nsamp"], with_ig=with_ig))


@app.route("/resync")
def resync():
    S["ctx"].resync()
    return jsonify(S["ctx"].payload(S["nsamp"]))


@app.route("/mode")
def set_mode():
    S["ctx"].mode = request.args.get("m", "dream")
    return jsonify(S["ctx"].payload(S["nsamp"]))


@app.route("/reload")
def reload_ckpt():
    ck = torch.load(S["ckpt"], map_location=S["dev"])
    S["wm"].load_state_dict(ck["model_state"]); S["wm"].eval()
    S["ctx"] = new_ctx(S["ctx"].game.gist)
    return jsonify(S["ctx"].payload(S["nsamp"]))


HTML = """
<!doctype html><html><head><meta charset=utf-8><title>Mario belief WM player</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#1a1a2e;color:#eee}
h2{margin:2px 0;color:#e94560}.sub{color:#888;font-size:13px;margin-bottom:10px}
.row{display:flex;gap:34px;align-items:flex-start}
.panel h3{margin:4px 0}.real{color:#4ecca3}.pred{color:#e94560}
img{width:320px;image-rendering:pixelated;border:1px solid #333;background:#000}
button{margin:2px;padding:6px 10px;background:#0f3460;color:#eee;border:1px solid #444;border-radius:4px;cursor:pointer}
button:hover{background:#1b5a8a}#info{margin:10px 0;font-size:14px}
.val{color:#4ecca3;font-weight:bold}.warn{color:#e94560;font-weight:bold}
.bars{margin-top:6px}.barrow{margin:3px 0;font-size:13px}.k{display:inline-block;width:56px}
.bar{height:14px;background:#4a90e2;display:inline-block;vertical-align:middle}.hot{background:#e2a04a}
</style></head><body>
<h2>Mario two-world belief WM — engine vs. autoregressive WM</h2>
<div class=sub>Mario is realtime: ▶ play advances frames (gravity pulls Mario down, enemy patrols). arrows = ←/→ move, ↑ jump, x/space = shoot (needs ammo — collect a coin first), t = single tick, r = reset. The WM rolls forward on its OWN predictions (dream); "re-sync" snaps it back to the engine.</div>
<div>
 <button onclick="reset('mario')">reset BASE</button>
 <button onclick="reset('mario_breakable')">reset BREAKABLE</button>
 <button id=playbtn onclick="togglePlay()">▶ play realtime</button>
 <button onclick="step(5)">tick (1 frame)</button>
 <button onclick="resync()">re-sync WM → engine</button>
 <button onclick="reloadCkpt()">reload checkpoint</button>
 <button id=modebtn onclick="toggleMode()">mode: dream (autoregressive)</button>
</div>
<div id=info></div>
<div class=row>
 <div class=panel><h3 class=real>engine (ground truth)</h3><img id=real></div>
 <div class=panel><h3 class=pred>belief WM (autoregressive)</h3><img id=wm></div>
 <div class=panel><h3>information gain / action</h3><div id=bars class=bars></div></div>
</div>
<script>
var igMax=0.01, lastIg=null, lastActs=null;
function drawBars(ig,acts){if(!ig)return; lastIg=ig; lastActs=acts;
 igMax=Math.max(igMax,...ig.map(v=>Math.abs(v)));        // running max over the whole session
 let b=document.getElementById('bars');b.innerHTML='';let best=ig.indexOf(Math.max(...ig));
 for(let i=0;i<ig.length;i++){let r=document.createElement('div');r.className='barrow';
  let w=Math.max(0,ig[i])/igMax*150;
  r.innerHTML='<span class=k>'+acts[i]+'</span><span class="bar'+(i==best?' hot':'')+'" style="width:'+w+'px"></span> '+ig[i].toFixed(3);
  b.appendChild(r);}}
function render(d){
 document.getElementById('real').src='data:image/png;base64,'+d.real;
 document.getElementById('wm').src='data:image/png;base64,'+d.wm;
 if(d.ig) drawBars(d.ig,d.actions);                     // ticks omit IG; keep the last bars
 let agree=d.l1==0?'<span class=val>WM matches engine exactly</span>':
   '<span class=warn>WM differs: '+d.diff_cells+' cells ('+d.l1+' channel-bits)</span>';
 document.getElementById('info').innerHTML='world <b>'+d.world+'</b> | step '+d.step+
  ' | breaks '+d.n_break+' | under breakable platform: <b>'+d.under_platform+'</b> | '+agree;
 curMode=d.mode; curWorld=d.world;
 document.getElementById('modebtn').innerText='mode: '+(d.mode=='dream'?'dream (autoregressive)':'predict (one-step, obs-fed)');}
var curMode='dream', curWorld='mario';
// --- input/realtime loop: queue user keypresses (never dropped), fill gaps with ticks ---
var busy=false, playing=true, queue=[];
function send(a,fast){busy=true;
 fetch('/step?a='+a+(fast?'&fast=1':'')).then(r=>r.json()).then(d=>{render(d);busy=false;pump();});}
function pump(){if(busy)return;
 if(queue.length){send(queue.shift(),false);return;}     // user inputs first
 if(playing)setTimeout(()=>{if(!busy&&!queue.length&&playing)send(5,false);},30);}  // tick WITH IG
function step(a){if(busy){if(queue.length<8)queue.push(a);}else send(a,false);}
function setPlayBtn(){document.getElementById('playbtn').innerText=playing?'⏸ pause realtime':'▶ play realtime';}
function togglePlay(){playing=!playing;setPlayBtn();if(playing)pump();}
function afterReset(d){render(d);setPlayBtn();if(playing&&!busy)pump();}
function toggleMode(){let m=curMode=='dream'?'predict':'dream';fetch('/mode?m='+m).then(r=>r.json()).then(render);}
function reset(w){igMax=0.01;fetch('/reset?world='+w).then(r=>r.json()).then(afterReset);}
function resync(){fetch('/resync').then(r=>r.json()).then(render);}
function reloadCkpt(){fetch('/reload').then(r=>r.json()).then(render);}
document.addEventListener('keydown',e=>{let m={ArrowUp:0,ArrowLeft:1,ArrowDown:2,ArrowRight:3,' ':4,x:4,t:5};
 if(e.key in m){e.preventDefault();step(m[e.key]);}
 else if(e.key=='r'){e.preventDefault();reset(curWorld);}});
reset('mario');
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="nca_wm/active_learning/ckpts/mario_belief.pt")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--device", default="cpu")
    p.add_argument("--nsamp", type=int, default=4)
    args = p.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=dev)
    wm = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(dev)
    wm.load_state_dict(ck["model_state"]); wm.eval()
    for q in wm.parameters():
        q.requires_grad_(False)
    games, backends = build_render_backends()
    S.update(wm=wm, dev=dev, games=games, backends=backends, ckpt=args.ckpt, nsamp=args.nsamp)
    S["ctx"] = new_ctx("mario")
    print(f"serving {args.ckpt} (cfg d={ck['cfg']['d']}) on :{args.port}  worlds={list(games)}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
