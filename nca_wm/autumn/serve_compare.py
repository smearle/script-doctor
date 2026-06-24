#!/usr/bin/env python3
"""Interactive side-by-side: real Autumn engine vs. trained NCA world model.

Your input drives BOTH. The engine is ground truth; the WM predicts the next
board autoregressively from its own previous prediction. Disagreeing cells are
outlined. A dropdown switches between all trained models under --runs_dir.

Run:
    python -m nca_wm.autumn.serve_compare --runs_dir nca_wm/autumn/runs --port 8766
Then port-forward and open http://localhost:8766
"""
import argparse
import glob
import json
import os

from flask import Flask, jsonify, request

from nca_wm.autumn import infer

app = Flask(__name__)
S = {"runs": {}, "cur": None, "model": None, "cfg": None,
     "env": None, "pal_index": None, "wm_board": None, "frame": 0, "device": "cuda:0"}


def discover_runs(runs_dir):
    runs = {}
    for cfg in glob.glob(os.path.join(runs_dir, "*", "config.json")):
        d = os.path.dirname(cfg)
        runs[os.path.basename(d)] = d
    return dict(sorted(runs.items()))


def load_best(runs_dir, runs):
    """best.json maps game -> best run (the lever-fixed model). Returns
    {game: run} filtered to runs that actually exist."""
    p = os.path.join(runs_dir, "best.json")
    if not os.path.exists(p):
        return {}
    try:
        best = json.load(open(p))
    except Exception:
        return {}
    return {g: r for g, r in sorted(best.items()) if r in runs}


def load_model(name):
    model, cfg = infer.load_run(S["runs"][name], device=S["device"])
    pal_index = {} if cfg.get("multihot") else {n: i for i, n in enumerate(cfg["palette"])}
    S.update(cur=name, model=model, cfg=cfg, pal_index=pal_index)
    reset_all()


def _init_episode():
    # (Re)build engine + WM at the CURRENT seed (no increment), frame 0.
    env = infer.make_engine(S["cfg"]["game"], seed=S["seed_ctr"])
    S["env"] = env
    if S["cfg"].get("multihot"):
        C = S["cfg"]["n_colors"]
        S["wm_board"] = infer.object_engine_board(env, C, list(S["cfg"]["vocab"]))
    else:
        S["wm_board"] = infer.engine_board(env, S["pal_index"])
    S["wm_prev"] = S["wm_board"].copy()
    S["h"] = None  # recurrent hidden state (re-init at episode start)
    S["frame"] = 0


def reset_all():
    # NB: seed 0 is degenerate in the Autumn interpreter (randomPositions -> (0,0),
    # e.g. ants food always top-left); use a fresh non-zero seed each reset.
    S["seed_ctr"] = S.get("seed_ctr", 0) + 1
    S["actlog"] = []           # actions since reset — replayed for undo
    _init_episode()


def apply_step(act, smp=False):
    infer.engine_apply(S["env"], act)
    if S["cfg"].get("multihot"):
        nxt, S["h"] = infer.object_step(S["model"], S["cfg"], S["wm_board"], act, S["wm_prev"], S["h"], sample=smp)
    elif S["cfg"].get("recurrent"):
        nxt, S["h"] = infer.recurrent_step(S["model"], S["cfg"], S["wm_board"], act, S["h"])
    else:
        nxt = infer.wm_step(S["model"], S["cfg"], S["wm_board"], act, prev_board=S["wm_prev"], sample=smp)
    S["wm_prev"], S["wm_board"] = S["wm_board"], nxt
    S["frame"] += 1


def undo():
    # Undo by replay: rebuild engine + WM from the same seed and replay all but the
    # last action. The Autumn interpreter is deterministic given (seed, actions) —
    # verified to reproduce exactly, including seeded-random spawns (ants food etc).
    # Cheap: only the small action log is stored (no hidden-state tensors). The one
    # caveat is "sample WM" mode, which re-samples on replay (engine stays exact).
    if not S.get("actlog"):
        return
    log = S["actlog"][:-1]
    _init_episode()
    for a in log:
        apply_step(a)
    S["actlog"] = log


def snapshot():
    cfg = S["cfg"]
    if cfg.get("multihot"):
        C = cfg["n_colors"]
        eng = infer.object_engine_board(S["env"], C, list(cfg["vocab"]))
        wm = S["wm_board"]
        disagree = int((eng != wm).any(0).sum())   # cells where any object channel differs
        eng_disp = infer.object_display(eng, cfg)
        wm_disp = infer.object_display(wm, cfg)
    else:
        eng = infer.engine_board(S["env"], S["pal_index"])
        wm = S["wm_board"]
        pal = cfg["palette"]
        disagree = int((eng != wm).sum())
        eng_disp = [[pal[c] for c in row] for row in eng.tolist()]
        wm_disp = [[pal[c] for c in row] for row in wm.tolist()]
    return {
        "models": list(S["runs"]), "best": S.get("best", {}), "current": S["cur"], "game": cfg["game"],
        "grid_size": cfg["grid_size"],
        "engine": eng_disp, "wm": wm_disp,
        "disagree": disagree, "frame": S["frame"],
    }


@app.route("/api/state")
def state():
    return jsonify(snapshot())


@app.route("/api/load_model")
def load_model_route():
    name = request.args.get("name")
    if name in S["runs"]:
        load_model(name)
    return jsonify(snapshot())


@app.route("/api/action")
def action():
    a = request.args.get("a", "noop")
    if a == "reset":
        reset_all(); return jsonify(snapshot())
    if a == "undo":
        undo(); return jsonify(snapshot())
    if a == "resync":
        S["wm_board"] = infer.engine_board(S["env"], S["pal_index"])
        S["wm_prev"] = S["wm_board"].copy()
        return jsonify(snapshot())
    if a == "click":
        act = ("click", int(request.args["x"]), int(request.args["y"]))
    elif a in ("up", "down", "left", "right"):
        act = (a,)
    else:
        act = ("noop",)
    smp = request.args.get("sample") == "1"
    S.setdefault("actlog", []).append(act)
    apply_step(act, smp)
    return jsonify(snapshot())


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Engine vs World Model</title>
<style>
 body{font-family:monospace;background:#111;color:#ddd;margin:16px}
 .grids{display:flex;gap:28px;margin-top:10px}
 table{border-collapse:collapse} td{width:20px;height:20px;border:1px solid #333;padding:0}
 td.bad{outline:2px solid #f33;outline-offset:-2px}
 h4{margin:4px 0} .key{color:#9cf}
 select,button{font-family:monospace;font-size:14px;margin-right:6px}
</style></head><body>
<h3 id=hud>Engine vs World Model</h3>
<div>
 <b>best per game:</b> <select id=best onchange="loadBest()"><option value="">— pick a game —</option></select>
 &nbsp;&nbsp; all models: <select id=model onchange="loadModel()"></select>
 <button id=play onclick="togglePlay()">&#9208; Pause</button>
 speed: <select id=speed onchange="setSpeed()">
   <option value=1000>slow</option>
   <option value=400 selected>normal</option>
   <option value=150>fast</option>
 </select>
 <button onclick="act('reset')">reset (R)</button>
 <button onclick="act('undo')">&#8630; undo (Z)</button>
 <button onclick="act('resync')">resync WM&rarr;engine</button>
 <button onclick="act('noop')">step once</button>
 <label title="sample the WM from its predicted distribution instead of taking the most-likely state (shows random outcomes like food spawns)">
   <input type=checkbox id=samp> sample WM</label>
</div>
<div class=grids>
 <div><h4>Engine (truth)</h4><table id=eng></table></div>
 <div><h4 id=wmh>World Model</h4><table id=wm></table></div>
 <div><h4 id=diffh>Difference</h4><table id=diff></table></div>
</div>
<p>Auto-plays (ticks noop in the background). <span class=key>click</span>=click(x,y) &nbsp;
   <span class=key>arrows</span>=move &nbsp; <span class=key>space</span>=play/pause &nbsp;
   <span class=key>Z</span>=undo &nbsp; <span class=key>R</span>=reset</p>
<script>
async function j(u){return await (await fetch(u)).json()}
function fill(tbl,grid){tbl.innerHTML='';
 for(let y=0;y<grid.length;y++){let tr=tbl.insertRow();
  for(let x=0;x<grid.length;x++){let td=tr.insertCell();
   let c=grid[y][x]; td.style.background=c==='transparent'?'#000':c;
   td.title=`(${x},${y}) ${c}`; td.onclick=()=>act('click',x,y);}}}
function fillDiff(tbl,a,b){tbl.innerHTML='';   // disagreements only, in their own grid
 for(let y=0;y<a.length;y++){let tr=tbl.insertRow();
  for(let x=0;x<a.length;x++){let td=tr.insertCell();
   let same=a[y][x]===b[y][x];
   td.style.background= same ? '#1a1a1a' : '#f33';
   td.title=same?`(${x},${y}) match`:`(${x},${y}) engine=${a[y][x]} wm=${b[y][x]}`;
   td.onclick=()=>act('click',x,y);}}}
function render(s){
 document.getElementById('hud').textContent=`${s.game} [${s.current}]  frame=${s.frame}  disagree=${s.disagree} cells`;
 document.getElementById('diffh').textContent=`Difference (${s.disagree} cells)`;
 let best=s.best||{}; let bestRuns=new Set(Object.values(best));
 let sel=document.getElementById('model');
 if(sel.options.length!==s.models.length){sel.innerHTML='';
  s.models.forEach(m=>{let o=document.createElement('option');o.value=m;
   o.text=(bestRuns.has(m)?'★ ':'')+m;sel.add(o)});}   // star = best for its game
 sel.value=s.current;
 let bsel=document.getElementById('best');
 if(bsel.options.length<=1){
  Object.keys(best).sort().forEach(g=>{let o=document.createElement('option');
   o.value=best[g];o.text=`${g}  (${best[g]})`;bsel.add(o)});}
 bsel.value=bestRuns.has(s.current)?s.current:"";  // reflect current if it's a best
 fill(document.getElementById('eng'),s.engine);
 fill(document.getElementById('wm'),s.wm);
 fillDiff(document.getElementById('diff'),s.engine,s.wm);
}
// User inputs are QUEUED and never dropped; auto-play ticks only enqueue a noop
// when idle, so they never starve user clicks/arrows.
let inflight=false, playing=true, timer=null, queue=[];
function smp(){ return document.getElementById('samp').checked ? '&sample=1' : ''; }
async function drain(){
 if(inflight || queue.length===0) return;
 inflight=true;
 const url=queue.shift();
 try{ render(await j(url)); }
 finally{ inflight=false; if(queue.length) drain(); }
}
function act(a,x,y){
 let u=`/api/action?a=${a}`; if(a==='click')u+=`&x=${x}&y=${y}`; u+=smp();
 queue.push(u); drain();
}
function tick(){ if(playing && !inflight && queue.length===0){ queue.push(`/api/action?a=noop`+smp()); drain(); } }
function setSpeed(){ if(timer) clearInterval(timer);
 timer=setInterval(tick, parseInt(document.getElementById('speed').value)); }
function togglePlay(){ playing=!playing;
 document.getElementById('play').innerHTML = playing ? '&#9208; Pause' : '&#9654; Play'; }
async function loadModel(){
 let was=playing; playing=false; queue=[];
 render(await j(`/api/load_model?name=${document.getElementById('model').value}`));
 playing=was;
}
async function loadBest(){
 let v=document.getElementById('best').value; if(!v) return;
 let was=playing; playing=false; queue=[];
 render(await j(`/api/load_model?name=${v}`));
 playing=was;
}
async function init(){render(await j('/api/state')); setSpeed();}
document.addEventListener('keydown',e=>{
 const m={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down'};
 if(m[e.key]){e.preventDefault();act(m[e.key])}
 else if(e.key===' '){e.preventDefault();togglePlay()}   // space toggles play/pause
 else if(e.key.toLowerCase()==='z'){e.preventDefault();playing=false;  // pause so it doesn't immediately re-step
   document.getElementById('play').innerHTML='&#9654; Play';act('undo')}
 else if(e.key.toLowerCase()==='r'){act('reset')}});
init();
</script></body></html>"""


@app.route("/")
def index():
    return HTML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="nca_wm/autumn/runs")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    S["device"] = args.device
    S["runs"] = discover_runs(args.runs_dir)
    S["best"] = load_best(args.runs_dir, S["runs"])
    if not S["runs"]:
        raise SystemExit(f"no trained models under {args.runs_dir}")
    load_model(next(iter(S["runs"])))
    print(f"Serving {len(S['runs'])} model(s) {list(S['runs'])} at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
