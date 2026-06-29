"""Side-by-side viewer for the two PARAMETER-MATCHED Mario world models.

Engine (ground truth) vs. the Recurrent-NCA WM vs. the Transformer-belief WM,
all three stepped by the SAME action so you can watch each WM track — or drift
from — the engine. The two WMs have ~equal parameter counts (NCA n_hid=164 ≈
5.39M; Transformer d=208/d_model=272 ≈ 5.36M) and were trained on the IDENTICAL
A*-transition trajectory data (``train_recurrent``'s predecessor-chain sampler),
so this is an apples-to-apples architecture comparison.

  * NCA   = JAX/Flax ``RecurrentNCAWorldModel`` (hidden grid carried across ticks)
            from ``nca_wm/logs/mario2_recurrent``.
  * TF    = PyTorch ``AttnBeliefModel`` (causal-transformer belief) from
            ``nca_wm/active_learning/ckpts/mario2_transformer``.

Both roll out AUTOREGRESSIVELY ("dream": feed own prediction) over a fixed
L=k+1 context window (the training context); "predict" mode teacher-forces the
engine observation each tick. Real PuzzleScript sprites (C++ backend renderer).

  arrows = ←/→ move, ↑ jump, space = shoot (ACTION), t = single tick.

    .venv/bin/python -u -m nca_wm.active_learning.mario_compare_serve \
        --port 8771 --device cpu
Then port-forward and open http://localhost:8771
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
import torch
import torch.nn.functional as F

from nca_wm.state_ops import _multihot_to_objects
from nca_wm.active_learning import mario_belief as MB
from nca_wm.active_learning.attn_belief_model import AttnBeliefModel, AttnConfig, masked_pool
from nca_wm.active_learning.mario_explore import break_cols, NA, ACTIONS
from nca_wm.active_learning.multigame_data import _engine, _masks, _perm, _read_padded

# The two mario worlds both compile to exactly n_obj=20 with an IDENTITY
# raw->canonical channel map (verified), so _read_padded's raw-bit channels
# already match the canonical channel order the cached training data uses. Pad
# dims are the models' trained dims (max C/H/W across the two games).
CMAX, HMAX, WMAX = 20, 18, 16

S = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_transformer(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    wm = AttnBeliefModel(AttnConfig(**ck["cfg"])).to(device)
    wm.load_state_dict(ck["model_state"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    nparam = sum(p.numel() for p in wm.parameters())
    print(f"[TF] loaded {ckpt_path} (n_obj={ck['cfg']['n_obj']}, "
          f"step={ck.get('step')}, params={nparam:,})", flush=True)
    return wm


def load_nca(run_dir):
    """Return (apply_fn, L) where apply_fn(states (1,L,C,H,W), acts_oh (1,L,A))
    -> next-state logits (1,L,C,H,W). JIT-compiled at fixed L."""
    import jax
    import jax.numpy as jnp
    from nca_wm.models import RecurrentNCAWorldModel, N_ACTIONS
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    with open(run_dir / "params_best.pkl", "rb") as f:
        params = pickle.load(f)
    model = RecurrentNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_steps"], n_out=cfg["n_out"],
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"], input_skip=cfg["input_skip"])
    L = int(cfg["L"])
    nparam = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))

    @jax.jit
    def _apply(states, acts_oh):
        logits, _win, _spr = model.apply(params, states, acts_oh)
        return logits

    print(f"[NCA] loaded {run_dir} (n_hid={cfg['n_hid']}, n_steps={cfg['n_steps']}, "
          f"L={L}, params={nparam:,})", flush=True)
    return _apply, L, N_ACTIONS


def build_render_backends():
    """Compile each world with a sprite-capable backend (for rendering)."""
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
# Comparison context: one engine + two autoregressive WM rollouts
# ---------------------------------------------------------------------------
class CompareCtx:
    def __init__(self, tf_wm, nca_apply, nca_L, nca_A, game, backend, device, rng):
        self.tf = tf_wm
        self.nca_apply, self.nca_L, self.nca_A = nca_apply, nca_L, nca_A
        self.dev, self.game, self.backend = device, game, backend
        self.perm = _perm(game.n_obj, CMAX, rng)
        cell, chan = _masks(game.n_obj, game.H, game.W, self.perm, CMAX, HMAX, WMAX)
        self.cellT = torch.from_numpy(cell)[None].to(device)            # (1,H,W)
        vmask = (chan[:, None, None] * cell[None]).astype(np.float32)   # (C,H,W)
        self.vmaskT = torch.from_numpy(vmask).to(device)[None]          # (1,C,H,W)
        self.vmask_np = vmask
        self.eng = _engine(game.json_str, 0)
        self.eng.set_track_rules_fired(True)
        self.sb = MB._bit(self.eng, "Step")
        self.step_i = 0
        self.n_break = 0
        self.last_steps = self._stepcount()
        self.mode = "dream"            # "dream" = autoregressive; "predict" = obs-fed
        self.resync()

    # --- engine helpers ---
    def _engine_obs_np(self):
        return _read_padded(self.eng, self.game.n_obj, self.perm,
                            CMAX, HMAX, WMAX)                            # (C,H,W) float32

    def _stepcount(self):
        return int(((MB._grid(self.eng) >> self.sb) & 1).sum())

    def _engine_step(self, a):
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

    # --- (re)initialise both WM rollouts to the engine's current frame ---
    def resync(self):
        obs = self._engine_obs_np()
        # Transformer state (mirrors mario_serve.CompareCtx)
        self.tf_board = torch.from_numpy(obs)[None].to(self.dev)
        self.tf_sp = self.tf.encode_frame(self.tf_board)
        self.tf_ps = [masked_pool(self.tf_sp, self.cellT)]
        self.tf_pa = [NA]
        # NCA rolling-window state: frames[t] is an input frame, acts[t] the
        # action applied to it (acts has one fewer entry than frames).
        self.nca_frames = [obs.copy()]
        self.nca_acts = []
        self.nca_board = obs.copy()

    # --- per-model one-step prediction ---
    @torch.no_grad()
    def _tf_predict(self, a):
        bel = self.tf.belief_now(torch.stack(self.tf_ps, 1),
                                 torch.tensor([self.tf_pa], device=self.dev))
        cb = bel + self.tf.emb_a(torch.tensor([a], device=self.dev))
        l0 = self.tf._dec_all_k(self.tf.dec0, self.tf.emb_z0, self.tf_sp, cb)
        p0 = F.softmax(self.tf.prior0(cb), -1)
        prob = (p0[..., None, None, None] * torch.sigmoid(l0)).sum(1)       # (1,C,H,W)
        pred = (prob * self.vmaskT > 0.5).float()
        return pred                                                         # (1,C,H,W)

    def _nca_predict(self, a):
        import jax.numpy as jnp
        L = self.nca_L
        frames = self.nca_frames
        acts_full = self.nca_acts + [a]            # action aligned to each frame
        # Take the last L; front-pad (repeat oldest frame + TICK) to fixed L so
        # the JIT shape is constant. Padding re-observes a static frame, which
        # warms the hidden grid harmlessly when the episode is < L ticks old.
        fr = frames[-L:]
        ac = acts_full[-L:]
        while len(fr) < L:
            fr = [frames[0]] + fr
            ac = [5] + ac                          # 5 = TICK (no-op advance)
        states = np.stack(fr)[None].astype(np.float32)                      # (1,L,C,H,W)
        acts_oh = np.zeros((1, L, self.nca_A), np.float32)
        for t, ai in enumerate(ac):
            acts_oh[0, t, ai] = 1.0
        logits = self.nca_apply(jnp.asarray(states), jnp.asarray(acts_oh))
        last = np.asarray(logits[0, -1])                                    # (C,H,W)
        pred = ((1.0 / (1.0 + np.exp(-last))) * self.vmask_np > 0.5).astype(np.float32)
        return pred                                                        # (C,H,W)

    @torch.no_grad()
    def step(self, a):
        # 1) both WM predictions of the next frame from current inputs
        tf_pred = self._tf_predict(a)                  # (1,C,H,W) torch
        nca_pred = self._nca_predict(a)                # (C,H,W) numpy
        self.tf_board = tf_pred
        self.nca_board = nca_pred
        # 2) advance the engine (ground truth)
        self._engine_step(a)
        self.step_i += 1
        engine_obs = self._engine_obs_np()             # (C,H,W) numpy
        # 3) feed back: own prediction (dream) or true observation (predict)
        tf_next = tf_pred if self.mode == "dream" else \
            torch.from_numpy(engine_obs)[None].to(self.dev)
        self.tf_sp = self.tf.encode_frame(tf_next)
        self.tf_ps.append(masked_pool(self.tf_sp, self.cellT))
        self.tf_pa.append(a)
        nca_next = nca_pred if self.mode == "dream" else engine_obs
        self.nca_frames.append(nca_next.copy())
        self.nca_acts.append(a)
        # bound buffers (only the last L window is ever used)
        cap = self.nca_L + 1
        if len(self.nca_frames) > cap:
            self.nca_frames = self.nca_frames[-cap:]
            self.nca_acts = self.nca_acts[-(cap - 1):]

    # --- rendering ---
    def _png(self, obs):
        crop = (obs[:self.game.n_obj, :self.game.H, :self.game.W] > 0.5).astype(np.uint8)
        objs = _multihot_to_objects(crop)
        frame = self.backend.render_frame_from_objects(objs, self.game.W, self.game.H)
        buf = io.BytesIO()
        PIL.Image.fromarray(frame).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _diff(self, wm_obs):
        e = self._engine_obs_np()[:self.game.n_obj, :self.game.H, :self.game.W] > 0.5
        w = wm_obs[:self.game.n_obj, :self.game.H, :self.game.W] > 0.5
        return int((e != w).sum()), int((e != w).any(0).sum())

    def payload(self):
        eng_obs = self._engine_obs_np()
        nca_obs = self.nca_board
        tf_obs = self.tf_board[0].cpu().numpy()
        nca_l1, nca_diff = self._diff(nca_obs)
        tf_l1, tf_diff = self._diff(tf_obs)
        cols, info = break_cols(MB._grid(self.eng), MB._bit(self.eng, "Player"),
                                self.sb, MB._bit(self.eng, "Floor"))
        under = info is not None and info[2] and info[1] in cols
        return dict(real=self._png(eng_obs), nca=self._png(nca_obs),
                    tf=self._png(tf_obs), step=self.step_i,
                    nca_l1=nca_l1, nca_diff=nca_diff, tf_l1=tf_l1, tf_diff=tf_diff,
                    n_break=self.n_break, under_platform=bool(under),
                    mode=self.mode, actions=ACTIONS, world=self.game.gist)


def new_ctx(world):
    import random
    return CompareCtx(S["tf"], S["nca_apply"], S["nca_L"], S["nca_A"],
                      S["games"][world], S["backends"][world], S["dev"],
                      random.Random(0))


def dispatch(path, q):
    """Route a GET request -> (content_type, body_bytes). Stdlib server, no Flask
    (this venv's pinned Flask 1.1.2 is incompatible with its Jinja2/Werkzeug)."""
    if path == "/":
        return "text/html", HTML.encode()
    if path == "/reset":
        S["ctx"] = new_ctx(q.get("world", ["mario"])[0])
    elif path == "/step":
        S["ctx"].step(int(q.get("a", ["0"])[0]))
    elif path == "/resync":
        S["ctx"].resync()
    elif path == "/mode":
        S["ctx"].mode = q.get("m", ["dream"])[0]
    else:
        return "text/plain", b"not found"
    return "application/json", json.dumps(S["ctx"].payload()).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        try:
            ctype, body = dispatch(u.path, parse_qs(u.query))
            code = 200
        except Exception as e:  # surface errors to the browser instead of hanging
            import traceback
            traceback.print_exc()
            ctype, body, code = "text/plain", str(e).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet the per-request stderr spam
        pass


HTML = """
<!doctype html><html><head><meta charset=utf-8><title>Mario WM compare — NCA vs Transformer</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#1a1a2e;color:#eee}
h2{margin:2px 0;color:#e94560}.sub{color:#888;font-size:13px;margin-bottom:10px;max-width:900px}
.row{display:flex;gap:30px;align-items:flex-start}
.panel h3{margin:4px 0}.real{color:#4ecca3}.nca{color:#e2a04a}.tf{color:#9b8cff}
img{width:300px;image-rendering:pixelated;border:1px solid #333;background:#000}
button{margin:2px;padding:6px 10px;background:#0f3460;color:#eee;border:1px solid #444;border-radius:4px;cursor:pointer}
button:hover{background:#1b5a8a}#info{margin:10px 0;font-size:14px}
.val{color:#4ecca3;font-weight:bold}.warn{color:#e94560;font-weight:bold}
.stat{font-size:13px;margin-top:4px}
</style></head><body>
<h2>Mario world models — engine vs. parameter-matched NCA &amp; Transformer</h2>
<div class=sub>Both WMs (~5.4M params each, same training data) roll forward on their OWN predictions (dream). Mario is realtime: ▶ play fires a no-op tick every 0.12s on the wall clock (gravity pulls Mario down, enemy patrols) — interleaved with your keypresses, just like the PuzzleScript web player.
<b>Controls:</b> ← / → = move, ↑ = jump, <b>x = shoot (ACTION)</b>, <b>space = no-op tick</b>, <b>r = reset episode</b> (current world). "re-sync" snaps both WMs back to the engine; "predict" mode teacher-forces the engine frame each tick.</div>
<div>
 <button onclick="reset('mario')">reset BASE</button>
 <button onclick="reset('mario_breakable')">reset BREAKABLE</button>
 <button id=playbtn onclick="togglePlay()">▶ play realtime</button>
 <button onclick="step(5)">no-op tick (space)</button>
 <button onclick="resync()">re-sync WMs → engine</button>
 <button id=modebtn onclick="toggleMode()">mode: dream (autoregressive)</button>
</div>
<div id=info></div>
<div class=row>
 <div class=panel><h3 class=real>engine (ground truth)</h3><img id=real></div>
 <div class=panel><h3 class=nca>Recurrent NCA</h3><img id=nca><div class=stat id=ncastat></div></div>
 <div class=panel><h3 class=tf>Transformer belief</h3><img id=tf><div class=stat id=tfstat></div></div>
</div>
<script>
function agreeTxt(l1,diff){return l1==0?'<span class=val>matches engine exactly</span>':
   '<span class=warn>differs: '+diff+' cells ('+l1+' bits)</span>';}
function render(d){
 document.getElementById('real').src='data:image/png;base64,'+d.real;
 document.getElementById('nca').src='data:image/png;base64,'+d.nca;
 document.getElementById('tf').src='data:image/png;base64,'+d.tf;
 document.getElementById('ncastat').innerHTML=agreeTxt(d.nca_l1,d.nca_diff);
 document.getElementById('tfstat').innerHTML=agreeTxt(d.tf_l1,d.tf_diff);
 document.getElementById('info').innerHTML='world <b>'+d.world+'</b> | step '+d.step+
  ' | breaks '+d.n_break+' | under breakable platform: <b>'+d.under_platform+'</b>';
 curMode=d.mode;
 document.getElementById('modebtn').innerText='mode: '+(d.mode=='dream'?'dream (autoregressive)':'predict (one-step, obs-fed)');}
var curMode='dream', lastWorld='mario';
// Realtime model (faithful to PuzzleScript realtime_interval): a steady wall-clock
// timer makes a no-op tick DUE every RT_MS; the tick takes priority over user
// keypresses so gravity keeps firing even while you walk (a stationary-Player
// turn lands every interval). User keys fill the gaps between ticks.
var busy=false, playing=false, queue=[], tickDue=false;
const RT_MS=120;                                  // realtime_interval 0.12s
setInterval(()=>{if(playing){tickDue=true;pump();}},RT_MS);
function send(a){busy=true;
 fetch('/step?a='+a).then(r=>r.json()).then(d=>{render(d);busy=false;pump();});}
function pump(){if(busy)return;
 if(tickDue){tickDue=false;send(5);return;}       // realtime no-op tick has priority
 if(queue.length){send(queue.shift());return;}}
function step(a){if(busy||tickDue){if(queue.length<8)queue.push(a);pump();}else send(a);}
function setPlayBtn(){document.getElementById('playbtn').innerText=playing?'⏸ pause realtime':'▶ play realtime';}
function togglePlay(){playing=!playing;setPlayBtn();if(playing)pump();}
function afterReset(d){render(d);setPlayBtn();if(playing&&!busy)pump();}
function toggleMode(){let m=curMode=='dream'?'predict':'dream';fetch('/mode?m='+m).then(r=>r.json()).then(render);}
function reset(w){lastWorld=w;fetch('/reset?world='+w).then(r=>r.json()).then(afterReset);}
function resync(){fetch('/resync').then(r=>r.json()).then(render);}
// ←/→ move, ↑ jump, x = shoot (ACTION), space = no-op tick, r = reset episode.
document.addEventListener('keydown',e=>{let m={ArrowUp:0,ArrowLeft:1,ArrowDown:2,ArrowRight:3,x:4,' ':5,t:5};
 if(e.key=='r'){e.preventDefault();reset(lastWorld);return;}
 if(e.key in m){e.preventDefault();step(m[e.key]);}});
reset('mario');
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf_ckpt",
                   default="nca_wm/active_learning/ckpts/mario2_transformer/params_best.pkl")
    p.add_argument("--nca_run", default="nca_wm/logs/mario2_recurrent")
    p.add_argument("--port", type=int, default=8771)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    tf_wm = load_transformer(args.tf_ckpt, dev)
    nca_apply, nca_L, nca_A = load_nca(args.nca_run)
    games, backends = build_render_backends()
    S.update(tf=tf_wm, nca_apply=nca_apply, nca_L=nca_L, nca_A=nca_A,
             dev=dev, games=games, backends=backends)
    S["ctx"] = new_ctx("mario")
    print(f"serving on :{args.port}  worlds={list(games)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
