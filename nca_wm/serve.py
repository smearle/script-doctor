"""Interactive Flask web app for the trained NCA world model.

Extracted from ``nca_wm/train.py`` so the training entrypoint stays focused on
training. Imports a few helpers from ``train.py`` lazily to avoid a circular
import (this module is itself only imported when ``--serve`` is requested).
"""
from __future__ import annotations

import base64
import io
import json

import jax
import jax.numpy as jnp
import numpy as np

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv


def serve_world_model(
    game_infos: list[dict],
    ps_parser,
    bundles: list[dict],
    initial_game_id: int = 0,
    level_i: int = 0,
    port: int = 8000,
    host: str = "0.0.0.0",
):
    """Serve trained world model(s) as an interactive web app.

    Multi-game *and* multi-model aware: ``game_infos`` lists every game the
    user can pick, and ``bundles`` is a parallel list (one per game) holding
    that game's model. Each bundle is a dict with keys:

      ``apply_fn``    — jitted ``model.apply`` for this game's checkpoint,
      ``params``      — that checkpoint's (unwrapped) params,
      ``conditional`` — whether apply takes game tokens/mask,
      ``max_pad``     — (C, H, W) the model was trained to pad inputs to,
      ``max_tok_len`` — token length for conditional models.

    A single checkpoint covering many games supplies the same bundle object
    for each of its games; serving several independent single-game runs in
    one viewer supplies a distinct bundle per game. On game switch the active
    bundle's apply/params/padding take over, so heterogeneous models (e.g. a
    20-channel mario model and a 9-channel coins model) coexist in one viewer.
    """
    from flask import Flask, jsonify
    import PIL.Image

    from nca_wm.train import (
        N_ACTIONS,
        _multihot_to_objects,
        _pad_state_for_model,
        _unpad_pred,
    )

    assert len(bundles) == len(game_infos), (
        f"bundles ({len(bundles)}) must be parallel to game_infos "
        f"({len(game_infos)})")

    app = Flask(__name__)

    state: dict = {}

    def _pad_tokens(tids, max_tok_len):
        pad = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        L = min(len(tids), max_tok_len)
        pad[:L] = tids[:L]
        mask[:L] = True
        return pad, mask

    def _build_env(game_id: int, level_i: int):
        """Create a fresh (env, backend, real_obs) tuple at level start."""
        info = game_infos[game_id]
        backend = CppPuzzleScriptBackend()
        backend.compile_game(ps_parser, info["name"])
        # compile_game does not initialize a level; without this load_level call
        # process_input runs on uninitialized backend state and segfaults.
        backend.cpp_engine.load_level(level_i)
        env = CppPuzzleScriptEnv(info["json_str"], level_i=level_i, max_episode_steps=10000)
        real_obs, _ = env.reset()
        return env, backend, real_obs

    def _replay(env, backend, actions):
        """Replay an action list on a fresh env+backend, returning final real_obs."""
        real_obs = None
        for a in actions:
            real_obs, _, _, _, _ = env.step(a)
            backend.process_input(a)
            again = 0
            while backend.againing and again < 1000:
                backend.process_input(-1)
                again += 1
        return real_obs

    def _realtime_interval(info: dict) -> float | None:
        """Seconds-per-tick for a real-time game (``realtime_interval`` prelude),
        or None for turn-based games. The browser uses this to auto-fire the
        no-op tick action (id 5) between key presses."""
        try:
            meta = json.loads(info["json_str"]).get("metadata", {})
        except Exception:
            return None
        if "realtime_interval" not in meta:
            return None
        try:
            return float(meta["realtime_interval"])
        except (TypeError, ValueError):
            return None

    def _switch(game_id: int, level_i: int):
        info = game_infos[game_id]
        b = bundles[game_id]
        max_C, max_H, max_W = b["max_pad"]
        env, backend, real_obs = _build_env(game_id, level_i)
        n_objs, grid_h, grid_w = env.observation_shape
        pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        state.clear()
        state.update(
            game_id=game_id,
            level_i=level_i,
            env=env,
            backend=backend,
            n_objs=n_objs,
            grid_h=grid_h,
            grid_w=grid_w,
            real_obs=real_obs,
            pred_state=pred_state,
            step=0,
            diverged=False,
            last_action=None,
            env_actions=[],   # actions actually fed to the real env, in order
            history=[],       # snapshots taken pre-step; pop on undo
            realtime_interval=_realtime_interval(info),
        )
        if b["conditional"]:
            tids = info.get("token_ids", [])
            pad, mask = _pad_tokens(tids, b["max_tok_len"])
            state["tokens"] = jnp.array(pad[None])
            state["mask"] = jnp.array(mask[None])

    def _snapshot():
        """Push a snapshot of state we want to be able to restore via undo.

        Stored before each step; undo pops one snapshot and restores. The env
        and backend aren't snapshotted (C++ internal state isn't serializable);
        instead we record env_actions and rebuild on undo by replay.
        """
        state["history"].append(dict(
            pred_state=state["pred_state"],
            step=state["step"],
            last_action=state["last_action"],
            diverged=state["diverged"],
            env_actions=list(state["env_actions"]),
        ))

    def _undo():
        """Pop one snapshot and restore. No-op if history is empty."""
        if not state["history"]:
            return
        snap = state["history"].pop()
        state["pred_state"] = snap["pred_state"]
        state["step"] = snap["step"]
        state["last_action"] = snap["last_action"]
        state["diverged"] = snap["diverged"]
        # If env_actions changed (real step was undone), rebuild env+backend
        # and replay the trimmed action list. Pure dream-step undo is cheaper:
        # env_actions list is unchanged so we skip the rebuild.
        if snap["env_actions"] != state["env_actions"]:
            env, backend, real_obs = _build_env(state["game_id"], state["level_i"])
            replayed = _replay(env, backend, snap["env_actions"])
            state["env"] = env
            state["backend"] = backend
            state["env_actions"] = list(snap["env_actions"])
            if replayed is not None:
                state["real_obs"] = replayed
            else:
                state["real_obs"] = real_obs

    def _render_obs(obs_native):
        objects = _multihot_to_objects(obs_native)
        frame = state["backend"].render_frame_from_objects(
            objects, state["grid_w"], state["grid_h"]
        )
        buf = io.BytesIO()
        PIL.Image.fromarray(frame).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _crop_pred():
        pred_bin = (state["pred_state"] > 0.5).astype(jnp.uint8)
        return _unpad_pred(pred_bin, state["n_objs"], state["grid_h"], state["grid_w"])

    def _feedback_pred(logits):
        """Threshold model logits and crop+re-pad to zero every off-grid padding
        cell before feeding back as the next AR state. Without this, content the
        model paints in the padding (left unconstrained because the training
        loss always masks padding cells) persists across steps — the player can "leave" through a wall
        into the padding and be carried back later. Mirrors the train/eval AR path."""
        b = bundles[state["game_id"]]
        max_C, max_H, max_W = b["max_pad"]
        pred_bin = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.uint8)
        pred_native = _unpad_pred(pred_bin, state["n_objs"],
                                  state["grid_h"], state["grid_w"])
        return _pad_state_for_model(pred_native, max_C, max_H, max_W)

    def _apply_step(action):
        b = bundles[state["game_id"]]
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        if b["conditional"]:
            logits, win_logit, _ = b["apply_fn"](
                b["params"], state["pred_state"], a_oh, state["tokens"], state["mask"]
            )
        else:
            logits, win_logit, _ = b["apply_fn"](b["params"], state["pred_state"], a_oh)
        return logits, win_logit

    def _state_payload():
        real_b64 = _render_obs(state["real_obs"])
        pred_native = _crop_pred()
        pred_b64 = _render_obs(pred_native)
        l1 = float(np.abs(pred_native.astype(np.int32) - state["real_obs"].astype(np.int32)).sum())
        info = game_infos[state["game_id"]]
        return dict(
            real=real_b64, pred=pred_b64,
            step=state["step"], l1=l1, diverged=state["diverged"],
            game=info["name"], game_id=state["game_id"], level_i=state["level_i"],
            n_levels=info.get("n_levels", 1),
            realtime_interval=state.get("realtime_interval"),
        )

    _switch(initial_game_id, level_i)

    @app.route("/")
    def index():
        return HTML_PAGE

    @app.route("/api/games")
    def list_games():
        return jsonify([
            {"name": g["name"], "n_levels": g.get("n_levels", 1)}
            for g in game_infos
        ])

    @app.route("/api/select/<int:game_id>/<int:level_i>")
    def select(game_id: int, level_i: int):
        if game_id < 0 or game_id >= len(game_infos):
            return jsonify(error="invalid game"), 400
        nl = game_infos[game_id].get("n_levels", 1)
        if level_i < 0 or level_i >= nl:
            return jsonify(error="invalid level"), 400
        _switch(game_id, level_i)
        return jsonify(_state_payload())

    @app.route("/api/state")
    def get_state():
        return jsonify(_state_payload())

    @app.route("/api/step/<int:action>")
    def step(action):
        if action < 0 or action >= N_ACTIONS:
            return jsonify(error="invalid action"), 400
        _snapshot()
        logits, win_logit = _apply_step(action)
        state["pred_state"] = _feedback_pred(logits)
        state["pred_won"] = float(jax.nn.sigmoid(win_logit)[0])
        state["last_action"] = action
        state["real_obs"], _, done, _, info = state["env"].step(action)
        state["env_actions"].append(action)
        state["backend"].process_input(action)
        again_steps = 0
        while state["backend"].againing and again_steps < 1000:
            state["backend"].process_input(-1)
            again_steps += 1
        state["step"] += 1
        payload = _state_payload()
        payload.update(won=bool(info.get("won", False)), done=bool(done),
                       pred_won=state["pred_won"])
        return jsonify(payload)

    @app.route("/api/step_dream/<int:action>")
    def step_dream(action):
        """Step only the world model (no real env) — pure dreaming."""
        if action < 0 or action >= N_ACTIONS:
            return jsonify(error="invalid action"), 400
        _snapshot()
        logits, win_logit = _apply_step(action)
        state["pred_state"] = _feedback_pred(logits)
        state["pred_won"] = float(jax.nn.sigmoid(win_logit)[0])
        state["step"] += 1
        state["diverged"] = True
        state["last_action"] = action
        pred_b64 = _render_obs(_crop_pred())
        return jsonify(pred=pred_b64, step=state["step"], pred_won=state["pred_won"])

    @app.route("/api/undo")
    def undo():
        _undo()
        payload = _state_payload()
        payload["can_undo"] = bool(state["history"])
        return jsonify(payload)

    @app.route("/api/reset")
    def reset():
        _switch(state["game_id"], state["level_i"])
        return jsonify(_state_payload())

    print(f"\nServing NCA world model player at http://{host}:{port}")
    print(f"  initial: {game_infos[initial_game_id]['name']} L{level_i}")
    print(f"  available games: {len(game_infos)}\n")
    app.run(host=host, port=port, debug=False)


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<title>NCA World Model Player</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee; font-family: monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  h1 { margin-bottom: 4px; font-size: 1.4em; color: #e94560; }
  .subtitle { color: #888; margin-bottom: 16px; font-size: 0.9em; }
  .selectors {
    display: flex; gap: 12px; align-items: center;
    margin-bottom: 16px; padding: 10px 16px;
    background: #16213e; border-radius: 8px;
  }
  .selectors label { color: #888; font-size: 0.85em; }
  .selectors select {
    background: #0f3460; color: #eee; border: 1px solid #333;
    padding: 4px 8px; border-radius: 4px; font-family: monospace;
    font-size: 0.9em; min-width: 140px;
  }
  .game-row {
    display: flex; gap: 24px; align-items: flex-start;
    flex-wrap: wrap; justify-content: center;
  }
  .panel { text-align: center; }
  .panel h2 { font-size: 1em; margin-bottom: 8px; }
  .panel h2.real { color: #4ecca3; }
  .panel h2.pred { color: #e94560; }
  .panel img.game-img {
    image-rendering: pixelated;
    border: 2px solid #333;
    min-width: 256px; min-height: 200px;
    background: #111;
  }
  .info {
    margin-top: 16px; padding: 12px 20px;
    background: #16213e; border-radius: 8px;
    display: flex; gap: 24px; font-size: 0.95em;
  }
  .info .val { color: #4ecca3; font-weight: bold; }
  .info .warn { color: #e94560; }
  .controls {
    margin-top: 12px; color: #666; font-size: 0.85em;
    line-height: 1.6;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    color: #fff; font-size: 0.8em; margin-left: 8px;
    vertical-align: middle;
  }
  .badge.dream { background: #e94560; }
  .badge.hidden { display: none; }
</style>
</head>
<body>
  <h1>NCA World Model Player
    <span id="dreamBadge" class="badge dream hidden">DREAM</span>
  </h1>
  <p class="subtitle">Real game engine vs. learned NCA world model</p>
  <div class="selectors">
    <label for="gameSelect">game:</label>
    <select id="gameSelect"></select>
    <label for="levelSelect">level:</label>
    <select id="levelSelect"></select>
  </div>
  <div class="game-row">
    <div class="panel">
      <h2 class="real">Real Engine</h2>
      <img id="realImg" class="game-img" src="" />
    </div>
    <div class="panel">
      <h2 class="pred">NCA Prediction</h2>
      <img id="predImg" class="game-img" src="" />
    </div>
  </div>
  <div class="info">
    <div>Step: <span class="val" id="stepVal">0</span></div>
    <div>L1 divergence: <span class="val" id="l1Val">0</span></div>
  </div>
  <div class="info" id="speedRow" style="display:none; align-items:center;">
    <label for="speedSlider">Speed: <span class="val" id="speedVal">1.0</span>x</label>
    <input type="range" id="speedSlider" min="0" max="3" step="0.25" value="1"
           style="width:220px; vertical-align:middle;">
    <span id="pausedTag" class="warn" style="display:none;">PAUSED</span>
  </div>
  <div class="controls">
    Arrows / WASD = move &nbsp;|&nbsp; X = action &nbsp;|&nbsp;
    Z = undo &nbsp;|&nbsp; R = restart &nbsp;|&nbsp; V = dream mode
  </div>

<script>
const KEY_MAP = {
  ArrowUp: 0, ArrowLeft: 1, ArrowDown: 2, ArrowRight: 3,
  w: 0, a: 1, s: 2, d: 3, x: 4,
};
let dreaming = false;
let busy = false;
let games = [];          // [{name, n_levels}, ...]
let currentGame = 0;
let currentLevel = 0;
let realtimeTimer = null;   // driver interval handle (real-time games only)
let baseInterval = null;    // active game's native realtime_interval (s), or null
let speed = 1.0;            // speed multiplier from the slider (0 = paused)
const ACTION_QUEUE_MAX = 4; // cap so a held key can't build an unbounded backlog
let pendingActions = [];    // queued player key actions (never dropped on collision)

// Single point that advances the world by one action. The server keeps one
// shared state object and isn't concurrency-safe, so `busy` serializes every
// request (steps, reset, undo) — but unlike before we never *drop* a keypress
// on a collision; it waits in the queue / retries on the next driver tick.
async function sendStep(action) {
  busy = true;
  try {
    const endpoint = dreaming ? '/api/step_dream/' : '/api/step/';
    update(await (await fetch(endpoint + action)).json());
  } finally {
    busy = false;
  }
}

// Run an arbitrary one-shot command (reset/undo) once the engine is idle, so a
// real-time tick in flight never makes us silently ignore the key.
async function whenIdle(fn) {
  while (busy) await new Promise(r => setTimeout(r, 5));
  busy = true;                       // atomic: no await between check and set
  try { return await fn(); } finally { busy = false; }
}

// Drain one step: a queued player action takes priority; in real-time games an
// empty queue advances the world with the no-op tick (5). If a step is already
// in flight we just return and let the next drive() call pick it up.
async function drive() {
  if (busy) return;
  let action = pendingActions.shift();
  if (action === undefined) {
    if (!(baseInterval && speed > 0)) return;   // turn-based & idle: nothing to do
    action = 5;                                  // real-time: passive tick
  }
  await sendStep(action);
}

// Player key -> queue (never dropped). Turn-based games have no driver loop, so
// kick processing immediately; real-time games are driven by the interval.
function enqueue(action) {
  pendingActions.push(action);
  if (pendingActions.length > ACTION_QUEUE_MAX) pendingActions.shift();  // drop oldest
  if (!realtimeTimer) drive();
}

// (Re)build the real-time driver loop at the current speed. The effective tick
// period is base/speed (higher speed = faster). speed 0 pauses the world.
function restartDriver() {
  if (realtimeTimer) { clearInterval(realtimeTimer); realtimeTimer = null; }
  if (baseInterval && baseInterval > 0 && speed > 0) {
    const periodMs = Math.max(20, (baseInterval / speed) * 1000);
    realtimeTimer = setInterval(drive, periodMs);
  }
}

// Called from update() with the active game's realtime_interval (or null for
// turn-based games). Shows/hides the speed slider and (re)starts the driver.
function setRealtime(interval) {
  const next = (interval && interval > 0) ? interval : null;
  if (next === baseInterval) return;
  baseInterval = next;
  const row = document.getElementById('speedRow');
  if (row) row.style.display = baseInterval ? 'flex' : 'none';
  restartDriver();
}

async function fetchState() {
  const r = await fetch('/api/state');
  update(await r.json());
}

function update(d) {
  if (d.real) document.getElementById('realImg').src = 'data:image/png;base64,' + d.real;
  if (d.pred) document.getElementById('predImg').src = 'data:image/png;base64,' + d.pred;
  if (d.step !== undefined) document.getElementById('stepVal').textContent = d.step;
  if (d.l1 !== undefined) {
    const el = document.getElementById('l1Val');
    el.textContent = d.l1.toFixed(0);
    el.className = d.l1 > 20 ? 'val warn' : 'val';
  }
  if (d.game_id !== undefined) currentGame = d.game_id;
  if (d.level_i !== undefined) currentLevel = d.level_i;
  if ('realtime_interval' in d) setRealtime(d.realtime_interval);
}

function populateLevelSelect(gameId) {
  const lvl = document.getElementById('levelSelect');
  const n = (games[gameId] || {n_levels: 1}).n_levels;
  lvl.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = 'L' + i;
    lvl.appendChild(opt);
  }
  lvl.value = Math.min(currentLevel, n - 1);
}

async function selectGameLevel(gameId, levelI) {
  busy = true;
  dreaming = false;
  document.getElementById('dreamBadge').classList.add('hidden');
  update(await (await fetch(`/api/select/${gameId}/${levelI}`)).json());
  busy = false;
}

async function init() {
  games = await (await fetch('/api/games')).json();
  const sel = document.getElementById('gameSelect');
  games.forEach((g, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = g.name;
    sel.appendChild(opt);
  });
  // Read current state to know which game is active.
  const cur = await (await fetch('/api/state')).json();
  update(cur);
  sel.value = currentGame;
  populateLevelSelect(currentGame);
  document.getElementById('levelSelect').value = currentLevel;

  const speedSlider = document.getElementById('speedSlider');
  speedSlider.addEventListener('input', () => {
    speed = parseFloat(speedSlider.value);
    document.getElementById('speedVal').textContent = speed.toFixed(2);
    document.getElementById('pausedTag').style.display = speed > 0 ? 'none' : 'inline';
    restartDriver();   // apply the new tick rate (or pause) immediately
  });

  sel.addEventListener('change', async () => {
    currentGame = parseInt(sel.value);
    populateLevelSelect(currentGame);
    currentLevel = 0;
    await selectGameLevel(currentGame, 0);
  });
  document.getElementById('levelSelect').addEventListener('change', async (e) => {
    currentLevel = parseInt(e.target.value);
    await selectGameLevel(currentGame, currentLevel);
  });
}

document.addEventListener('keydown', async (e) => {
  const key = e.key;
  // Don't intercept while a select is focused.
  if (document.activeElement && document.activeElement.tagName === 'SELECT') return;

  if (key === 'r' || key === 'R') {
    pendingActions = [];
    dreaming = false;
    document.getElementById('dreamBadge').classList.add('hidden');
    whenIdle(async () => update(await (await fetch('/api/reset')).json()));
    return;
  }
  if (key === 'z' || key === 'Z') {
    whenIdle(async () => update(await (await fetch('/api/undo')).json()));
    return;
  }
  if (key === 'v' || key === 'V') {
    dreaming = !dreaming;
    document.getElementById('dreamBadge').classList.toggle('hidden', !dreaming);
    return;
  }

  const action = KEY_MAP[key];
  if (action === undefined) return;
  e.preventDefault();
  enqueue(action);   // queued, never dropped — processed by the driver / drive()
});

init();
</script>
</body>
</html>
"""
