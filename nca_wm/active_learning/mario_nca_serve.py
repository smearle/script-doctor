"""Three-panel viewer for the JAX NCA world model + information-gain adapter.

  LEFT   = live PuzzleScript ENGINE frame (ground truth, real sprites).
  MIDDLE = the JAX base world model's predicted next frame
           (q0 = sigmoid(NCAWorldModel(s, a)) > 0.5, real sprites).
           "predict" = teacher-forced (feed the true engine obs each tick);
           "dream"   = autoregressive (feed the model its own prediction).
  RIGHT  = per-action INFORMATION GAIN, computed from the CURRENT observed state:
             q0(o|s,a) = sigmoid(base(s,a));   q1(o'|s,a,o) = sigmoid(adapter(s,a,o))
             IG(s,a) = E_{o~q0}[ sum_realcells( log q1(o|s,a,o) - log q0(o|s,a) ) ]
           (Bernoulli per cell; averaged over ~16 samples of o.) Highlights argmax.

This mirrors ``mario_compare_serve.py`` (its stdlib web server, sprite rendering
via the C++ backend, dream/predict toggle, IG bar panel) but replaces the two
torch belief models with the SINGLE jax base model + the jax adapter head.

CRITICAL: the engine observation is fed to the model in the SAME channel order it
was trained on, via ``_read_padded`` / identity ``_perm`` (the in-distribution
objective uses no slot permutation). In "predict" mode the MIDDLE panel should
match the engine's next frame almost exactly (cell-acc ~0.99999).

    export PATH=$HOME/.nvm/versions/node/v24.15.0/bin:$PATH
    .venv/bin/python -u -m nca_wm.active_learning.mario_nca_serve --port 8772
Then on the Mac: ssh -L 8772:localhost:8772 210  and open http://localhost:8772
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import PIL.Image
import jax
import jax.numpy as jnp

from nca_wm.models import NCAWorldModel, N_ACTIONS
from nca_wm.adapter import AdapterHead
from nca_wm.state_ops import _multihot_to_objects
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.mario_explore import break_cols
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded

CMAX = 20                       # model's trained channel dim (max_C across both worlds)
BASE = "nca_wm/logs/mario_uncond_fulldata"
ADAPT = "nca_wm/logs/adapter_mario_skip/adapter_params.pkl"
ACTIONS = MB.ACTIONS            # ["UP","LEFT","DOWN","RIGHT","ACTION","TICK"]
TICK = MB.TICK                  # 5  (no-op realtime tick: gravity advances here)
EPS = 1e-6

S = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_base():
    params = pickle.load(open(f"{BASE}/params_best.pkl", "rb"))
    cfg = json.load(open(f"{BASE}/config.json"))
    gi = pickle.load(open(f"{BASE}/game_infos.pkl", "rb"))
    max_C = max(g["n_objs"] for g in gi)
    assert max_C == CMAX, f"max_C {max_C} != CMAX {CMAX}"
    model = NCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        input_skip=cfg["input_skip"], n_repeats=cfg["n_nca_repeats"],
        history=cfg["history"], axis_pool=cfg["axis_pool"],
        axis_cummax=cfg["axis_cummax"], global_pool=cfg["global_pool"])

    @jax.jit
    def q0(s, a):
        logits = model.apply(params, s, jax.nn.one_hot(a, N_ACTIONS))[0]
        return jax.nn.sigmoid(logits)

    print(f"[base] loaded {BASE} (n_hid={cfg['n_hid']} n_steps={cfg['n_nca_steps']} "
          f"C={max_C})", flush=True)
    return q0


def load_adapter():
    ad = pickle.load(open(ADAPT, "rb"))
    c = ad["cfg"]
    adapter = AdapterHead(n_hid=c["n_hid"], n_steps=c["n_steps"], n_out=c["n_out"])
    aparams = ad["params"]

    @jax.jit
    def q1(s, a, o):
        return jax.nn.sigmoid(adapter.apply(aparams, s, jax.nn.one_hot(a, N_ACTIONS), o))

    print(f"[adapter] loaded {ADAPT} (n_hid={c['n_hid']} n_steps={c['n_steps']} "
          f"C={c['n_out']})", flush=True)
    return q1


def build_render_backends():
    """Compile each world with a sprite-capable C++ backend (for rendering)."""
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


# ---------------------------------------------------------------------------
# One engine + one autoregressive JAX-WM rollout + per-action IG
# ---------------------------------------------------------------------------
class NCACtx:
    def __init__(self, q0, q1, game, backend, rng):
        self.q0, self.q1, self.game, self.backend = q0, q1, game, backend
        self.H, self.W, self.n_obj = game.H, game.W, game.n_obj
        # identity perm + masks; feed the model EXACTLY game.H x game.W (no H/W pad)
        # so the obs reproduces the training distribution (only C is padded to CMAX).
        self.perm = _perm(self.n_obj, CMAX, rng)
        self.eng = _engine(game.json_str, 0)
        if hasattr(self.eng, "set_track_rules_fired"):   # newer engine builds only
            self.eng.set_track_rules_fired(True)
        self.pb = MB._bit(self.eng, "Player")
        self.sb = MB._bit(self.eng, "Step")
        self.fb = MB._bit(self.eng, "Floor")
        self.rng = np.random.default_rng(0)
        self.mode = "dream"            # "dream" = autoregressive; "predict" = obs-fed
        self.step_i = 0
        self.n_break = 0
        self.last_steps = self._stepcount()
        obs = self._engine_obs()
        self.wm_board = obs.copy()     # state the model conditions on next
        self.wm_disp = obs.copy()      # what the MIDDLE panel shows

    # --- engine helpers ---
    def _engine_obs(self):
        # (CMAX, H, W) float multihot, channels in trained (id_dict) order.
        return _read_padded(self.eng, self.n_obj, self.perm, CMAX, self.H, self.W)

    def _grid(self):
        return MB._grid(self.eng)

    def _stepcount(self):
        return int(((self._grid() >> self.sb) & 1).sum())

    def _engine_step(self, a):
        if hasattr(self.eng, "clear_rules_fired"):
            self.eng.clear_rules_fired()
        self.eng.process_input(a)              # action idx == engine input id (5 = tick)
        k = 0
        while self.eng.is_againing() and k < 50:
            self.eng.process_input(-1)
            k += 1
        sc = self._stepcount()
        if sc < self.last_steps:
            self.n_break += 1
        self.last_steps = sc

    # --- world-model one-step prediction ---
    def _predict(self, board, a):
        """Threshold q0(board, a) to a multihot next frame (CMAX,H,W)."""
        p0 = np.asarray(self.q0(jnp.asarray(board[None]), jnp.asarray([a])))[0]
        pred = (p0 > 0.5).astype(np.float32)
        pred[self.n_obj:] = 0.0                # zero padded channels (already ~0)
        return pred

    def step(self, a):
        pred = self._predict(self.wm_board, a)
        self.wm_disp = pred
        self._engine_step(a)
        self.step_i += 1
        obs = self._engine_obs()
        self.wm_board = pred if self.mode == "dream" else obs

    def reset_wm(self):
        """Snap the model's conditioning board back to the live engine frame."""
        obs = self._engine_obs()
        self.wm_board = obs.copy()
        self.wm_disp = obs.copy()

    # --- information gain (q0/q1) from the CURRENT observed engine state ---
    def _ig(self, a, n_samples):
        S0 = self._engine_obs()[None]                          # (1,C,H,W)
        A = np.asarray([a])
        Sj, Aj = jnp.asarray(S0), jnp.asarray(A)
        P0 = np.asarray(self.q0(Sj, Aj)).clip(EPS, 1 - EPS)
        m = (S0.sum(1, keepdims=True) > 0)                     # real (non-pad) cells
        ig = 0.0
        for _ in range(n_samples):
            o = (self.rng.random(P0.shape) < P0).astype(np.float32)
            P1 = np.asarray(self.q1(Sj, Aj, jnp.asarray(o))).clip(EPS, 1 - EPS)
            lq0 = o * np.log(P0) + (1 - o) * np.log(1 - P0)
            lq1 = o * np.log(P1) + (1 - o) * np.log(1 - P1)
            ig += float(((lq1 - lq0) * m).sum())
        return ig / n_samples

    def ig_all(self, n_samples=16):
        return [round(self._ig(a, n_samples), 3) for a in range(N_ACTIONS)]

    # --- rendering ---
    def _png(self, obs):
        crop = (obs[:self.n_obj, :self.H, :self.W] > 0.5).astype(np.uint8)
        objs = _multihot_to_objects(crop)
        frame = self.backend.render_frame_from_objects(objs, self.W, self.H)
        buf = io.BytesIO()
        PIL.Image.fromarray(frame).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _diff(self, wm_obs):
        e = self._engine_obs()[:self.n_obj, :self.H, :self.W] > 0.5
        w = wm_obs[:self.n_obj, :self.H, :self.W] > 0.5
        return int((e != w).sum()), int((e != w).any(0).sum())

    def payload(self, with_ig=True):
        eng_obs = self._engine_obs()
        wm_l1, wm_diff = self._diff(self.wm_disp)
        cols, info = break_cols(self._grid(), self.pb, self.sb, self.fb)
        under = info is not None and info[2] and info[1] in cols
        return dict(real=self._png(eng_obs), wm=self._png(self.wm_disp),
                    step=self.step_i, wm_l1=wm_l1, wm_diff=wm_diff,
                    n_break=self.n_break, under_platform=bool(under),
                    ig=(self.ig_all() if with_ig else None),
                    mode=self.mode, actions=ACTIONS, world=self.game.gist)


def new_ctx(world):
    import random
    return NCACtx(S["q0"], S["q1"], S["games"][world], S["backends"][world],
                  random.Random(0))


def dispatch(path, q):
    if path == "/":
        return "text/html", HTML.encode()
    with_ig = True
    if path == "/reset":
        S["ctx"] = new_ctx(q.get("world", ["mario"])[0])
    elif path == "/step":
        S["ctx"].step(int(q.get("a", ["0"])[0]))
        with_ig = q.get("fast", ["0"])[0] != "1"   # realtime ticks skip the (slow) IG
    elif path == "/reset_wm":
        S["ctx"].reset_wm()
    elif path == "/mode":
        S["ctx"].mode = q.get("m", ["dream"])[0]
    else:
        return "text/plain", b"not found"
    return "application/json", json.dumps(S["ctx"].payload(with_ig)).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        try:
            ctype, body = dispatch(u.path, parse_qs(u.query))
            code = 200
        except Exception as e:
            import traceback
            traceback.print_exc()
            ctype, body, code = "text/plain", str(e).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


HTML = """
<!doctype html><html><head><meta charset=utf-8><title>Mario JAX-NCA world model + info-gain</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#1a1a2e;color:#eee}
h2{margin:2px 0;color:#e94560}.sub{color:#888;font-size:13px;margin-bottom:10px;max-width:980px}
.row{display:flex;gap:30px;align-items:flex-start}
.panel h3{margin:4px 0}.real{color:#4ecca3}.wm{color:#e2a04a}.ig{color:#9b8cff}
img{width:300px;image-rendering:pixelated;border:1px solid #333;background:#000}
button{margin:2px;padding:6px 10px;background:#0f3460;color:#eee;border:1px solid #444;border-radius:4px;cursor:pointer}
button:hover{background:#1b5a8a}#info{margin:10px 0;font-size:14px}
.val{color:#4ecca3;font-weight:bold}.warn{color:#e94560;font-weight:bold}
.stat{font-size:13px;margin-top:4px}
.bars{margin-top:6px}.barrow{margin:4px 0;font-size:13px}.k{display:inline-block;width:62px}
.bar{height:15px;background:#4a90e2;display:inline-block;vertical-align:middle}.hot{background:#e2a04a}
</style></head><body>
<h2>Mario JAX-NCA world model &mdash; engine vs. prediction vs. per-action info-gain</h2>
<div class=sub>LEFT = live engine (ground truth). MIDDLE = base WM next-frame prediction
q0 = sigmoid(NCA(s,a))&gt;0.5. RIGHT = info-gain per action IG(s,a)=E_o[log q1 - log q0]
from the CURRENT observed state (elevated where the next obs is genuinely ambiguous,
e.g. a jump into a breakable platform). <b>Controls:</b> &larr;/&rarr; = move,
&uarr; = jump, <b>x = shoot (ACTION)</b>, <b>space = no-op tick</b>, <b>r = reset</b>.
<b>predict</b> mode teacher-forces the engine obs each tick (the WM panel should match
the engine exactly); <b>dream</b> feeds the model its own prediction.
<b>snap WM &rarr; engine</b> re-anchors the dream board to the live frame.</div>
<div>
 <button onclick="reset('mario')">reset BASE (mario)</button>
 <button onclick="reset('mario_breakable')">reset BREAKABLE</button>
 <button id=playbtn onclick="togglePlay()">&#9654; play realtime</button>
 <button onclick="step(5)">no-op tick (space)</button>
 <button onclick="snap()">snap WM &rarr; engine</button>
 <button id=modebtn onclick="toggleMode()">mode: dream (autoregressive)</button>
</div>
<div id=info></div>
<div class=row>
 <div class=panel><h3 class=real>engine (ground truth)</h3><img id=real></div>
 <div class=panel><h3 class=wm>world model (q0 next-frame)</h3><img id=wm><div class=stat id=wmstat></div></div>
 <div class=panel><h3 class=ig>info-gain / action</h3><div id=bars class=bars></div></div>
</div>
<script>
function agreeTxt(l1,diff){return l1==0?'<span class=val>matches engine exactly</span>':
   '<span class=warn>differs: '+diff+' cells ('+l1+' bits)</span>';}
var igMax=0.01;
function drawBars(ig,acts){if(!ig)return;
 igMax=Math.max(igMax,...ig.map(v=>Math.abs(v)));
 let b=document.getElementById('bars');b.innerHTML='';let best=ig.indexOf(Math.max(...ig));
 for(let i=0;i<ig.length;i++){let r=document.createElement('div');r.className='barrow';
  let w=Math.max(0,ig[i])/igMax*150;
  r.innerHTML='<span class=k>'+acts[i]+'</span><span class="bar'+(i==best?' hot':'')+'" style="width:'+w+'px"></span> '+ig[i].toFixed(3);
  b.appendChild(r);}}
function render(d){
 document.getElementById('real').src='data:image/png;base64,'+d.real;
 document.getElementById('wm').src='data:image/png;base64,'+d.wm;
 document.getElementById('wmstat').innerHTML=agreeTxt(d.wm_l1,d.wm_diff);
 if(d.ig) drawBars(d.ig,d.actions);
 document.getElementById('info').innerHTML='world <b>'+d.world+'</b> | step '+d.step+
  ' | breaks '+d.n_break+' | under breakable platform: <b>'+d.under_platform+'</b>';
 curMode=d.mode;
 document.getElementById('modebtn').innerText='mode: '+(d.mode=='dream'?'dream (autoregressive)':'predict (one-step, obs-fed)');}
var curMode='dream', lastWorld='mario';
var busy=false, playing=false, queue=[], tickDue=false;
const RT_MS=120;
setInterval(()=>{if(playing){tickDue=true;pump();}},RT_MS);
function send(a,fast){busy=true;
 fetch('/step?a='+a+(fast?'&fast=1':'')).then(r=>r.json()).then(d=>{render(d);busy=false;pump();});}
function pump(){if(busy)return;
 if(tickDue){tickDue=false;send(5,true);return;}
 if(queue.length){send(queue.shift(),false);return;}}
function step(a){if(busy||tickDue){if(queue.length<8)queue.push(a);pump();}else send(a);}
function setPlayBtn(){document.getElementById('playbtn').innerText=playing?'⏸ pause realtime':'▶ play realtime';}
function togglePlay(){playing=!playing;setPlayBtn();if(playing)pump();}
function afterReset(d){render(d);setPlayBtn();if(playing&&!busy)pump();}
function toggleMode(){let m=curMode=='dream'?'predict':'dream';fetch('/mode?m='+m).then(r=>r.json()).then(render);}
function reset(w){lastWorld=w;fetch('/reset?world='+w).then(r=>r.json()).then(afterReset);}
function snap(){fetch('/reset_wm').then(r=>r.json()).then(render);}
document.addEventListener('keydown',e=>{let m={ArrowUp:0,ArrowLeft:1,ArrowDown:2,ArrowRight:3,x:4,' ':5,t:5};
 if(e.key=='r'){e.preventDefault();reset(lastWorld);return;}
 if(e.key in m){e.preventDefault();step(m[e.key]);}});
reset('mario');
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8772)
    args = p.parse_args()
    q0 = load_base()
    q1 = load_adapter()
    games, backends = build_render_backends()
    S.update(q0=q0, q1=q1, games=games, backends=backends)
    S["ctx"] = new_ctx("mario")
    print(f"serving on :{args.port}  worlds={list(games)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
