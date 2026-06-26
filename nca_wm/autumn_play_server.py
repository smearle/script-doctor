#!/usr/bin/env python3
"""Minimal web player for AutumnBench programs.

Drives the pybind `mara-autumn-cpp` interpreter (installed in this venv) and
serves an interactive colored grid. Arrow keys = move, space = noop step,
click a cell = click(x,y), R = reset. Dropdown switches program.

Run:
    .venv/bin/python3 nca_wm/autumn_play_server.py --host 0.0.0.0 --port 8765
Then open http://<host>:8765  (port-forward with `ssh -L 8765:localhost:8765 ...`
if the box isn't directly reachable).
"""
import argparse
import json
import os

from flask import Flask, jsonify, request

from MARA.autumn_cpp.interpreter_module import Interpreter
from MARA.autumn_cpp.autumnstdlib import autumnstdlib

TESTS_DIR = "/home/jupyter-smearle/mara/MARA/domains/autumnbench/Autumn.wasm/tests"

app = Flask(__name__)
STATE = {"itp": None, "name": None, "seed": 0, "frame": 0}


def list_programs():
    return sorted(
        f[:-5] for f in os.listdir(TESTS_DIR) if f.endswith(".sexp")
    )


def load(name, seed=0):
    # seed 0 is degenerate in the interpreter (randomPositions -> (0,0), e.g. ants
    # food always spawns top-left); pick a non-zero seed when none is given.
    if not seed:
        seed = int.from_bytes(os.urandom(3), "big") | 1
    prog = open(os.path.join(TESTS_DIR, f"{name}.sexp")).read()
    itp = Interpreter()
    itp.set_verbose(False)
    itp.run_script(prog, autumnstdlib, "", seed)
    STATE.update(itp=itp, name=name, seed=seed, frame=0)


def snapshot():
    itp = STATE["itp"]
    d = json.loads(itp.render_all())
    gs = d.get("GRID_SIZE", 16)
    try:
        bg = itp.get_background() or "black"
    except Exception:
        bg = "black"
    # cell -> (color, list-of-object-type-names) ; topmost color wins for display
    grid = [[None] * gs for _ in range(gs)]
    types = [k for k in d if k != "GRID_SIZE"]
    for k in types:
        for c in d[k]:
            p = c["position"]
            x, y = int(p["x"]), int(p["y"])
            if 0 <= x < gs and 0 <= y < gs:
                grid[y][x] = c.get("color", bg)
    return {
        "name": STATE["name"], "grid_size": gs, "bg": bg, "grid": grid,
        "types": types, "frame": STATE["frame"], "seed": STATE["seed"],
    }


@app.route("/api/programs")
def programs():
    return jsonify(list_programs())


@app.route("/api/state")
def state():
    if STATE["itp"] is None:
        load(list_programs()[0])
    return jsonify(snapshot())


@app.route("/api/load")
def api_load():
    name = request.args.get("name", list_programs()[0])
    seed = int(request.args.get("seed", 0))
    load(name, seed)
    return jsonify(snapshot())


@app.route("/api/action")
def action():
    itp = STATE["itp"]
    a = request.args.get("a", "step")
    if a in ("left", "right", "up", "down"):
        getattr(itp, a)(); itp.step()   # engine protocol: input then step
    elif a == "click":
        itp.click(int(request.args["x"]), int(request.args["y"])); itp.step()
    elif a == "reset":
        load(STATE["name"], STATE["seed"])
        return jsonify(snapshot())
    else:  # step / noop
        itp.step()
    STATE["frame"] += 1
    return jsonify(snapshot())


HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>AutumnBench player</title>
<style>
 body{font-family:monospace;background:#111;color:#ddd;margin:16px}
 #grid{border-collapse:collapse;margin-top:10px}
 #grid td{width:22px;height:22px;border:1px solid #333;padding:0}
 select,button{font-family:monospace;font-size:14px;margin-right:6px}
 #hud{margin-top:8px;color:#9c9}
 .key{color:#9cf}
</style></head><body>
<h3>AutumnBench player</h3>
<div>
 <select id=prog></select>
 seed <input id=seed type=number value=0 style="width:60px">
 <button onclick="load()">load</button>
 <button onclick="act('reset')">reset (R)</button>
 <button onclick="act('step')">step / noop (space)</button>
</div>
<div id=hud></div>
<table id=grid></table>
<p><span class=key>arrows</span>=move &nbsp; <span class=key>space</span>=noop step &nbsp;
   <span class=key>click cell</span>=click(x,y) &nbsp; <span class=key>R</span>=reset</p>
<script>
async function j(u){return await (await fetch(u)).json()}
function render(s){
 document.getElementById('hud').textContent =
   `${s.name}  ${s.grid_size}x${s.grid_size}  frame=${s.frame}  seed=${s.seed}  types=[${s.types}]`;
 let t=document.getElementById('grid'); t.innerHTML='';
 for(let y=0;y<s.grid_size;y++){let tr=t.insertRow();
  for(let x=0;x<s.grid_size;x++){let td=tr.insertCell();
   let c=s.grid[y][x]||s.bg; td.style.background=c==='transparent'?s.bg:c;
   td.title=`(${x},${y}) ${c}`;
   td.onclick=()=>act('click',x,y);}}
}
async function refresh(){render(await j('/api/state'))}
async function act(a,x,y){let u=`/api/action?a=${a}`;if(a==='click')u+=`&x=${x}&y=${y}`;render(await j(u))}
async function load(){let n=document.getElementById('prog').value,sd=document.getElementById('seed').value;
 render(await j(`/api/load?name=${n}&seed=${sd}`))}
async function init(){let ps=await j('/api/programs');let sel=document.getElementById('prog');
 ps.forEach(p=>{let o=document.createElement('option');o.value=o.text=p;sel.add(o)});
 await refresh();}
document.addEventListener('keydown',e=>{
 const m={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down'};
 if(m[e.key]){e.preventDefault();act(m[e.key])}
 else if(e.key===' '){e.preventDefault();act('step')}
 else if(e.key==='r'||e.key==='R'){act('reset')}});
init();
</script></body></html>"""


@app.route("/")
def index():
    return HTML


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--program", default=None)
    args = ap.parse_args()
    load(args.program or list_programs()[0])
    print(f"Serving {len(list_programs())} programs at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
