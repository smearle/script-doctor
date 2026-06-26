"""Serve a trained NCA world model from a saved run dir.

Usage:
    python nca_wm/serve_wm.py --load nca_wm/logs/scaling_4_long_v1
    python nca_wm/serve_wm.py                      # picker UI

Reads config.json, params.pkl, and game_infos.pkl from the run dir, rebuilds the
model at the trained recipe, and launches the interactive web app from
nca_wm.serve.serve_world_model.

If ``--load`` is omitted, scans ``nca_wm/logs/`` for valid run dirs and serves
a small picker page; selecting a run re-execs the script with ``--load`` set.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzlescript_jax.utils import init_ps_lark_parser
from nca_wm.serve import serve_world_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOGS_DIR = os.path.join(REPO_ROOT, "nca_wm", "logs")


def _unwrap_wm(params):
    """Joint checkpoints are {'wm': ..., 'dec': ...}; WM-only is the param dict."""
    if isinstance(params, dict) and {"wm", "dec"} <= set(params.keys()):
        return params["wm"]
    return params


def _build_wm(cfg, game_infos):
    """Reconstruct just the world-model module (we don't need the decoder for serving)."""
    arch = cfg.get("architecture", "rule_attn")
    conditional = cfg.get("conditional", True)
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos), default=1), 1)
    pool_kwargs = dict(
        axis_pool=cfg.get("axis_pool", False),
        axis_cummax=cfg.get("axis_cummax", False),
        global_pool=cfg.get("global_pool", False),
    )

    # Unconditional runs were trained with NCAWorldModel — the `architecture`
    # field in their config still says "rule_attn" (config-bookkeeping quirk
    # in train.py: arch flag is recorded even when the conditional encoder
    # is bypassed). Dispatch on `conditional`, not `architecture`.
    if not conditional:
        from nca_wm.train import NCAWorldModel
        return NCAWorldModel(
            n_hid=cfg["n_hid"],
            n_steps=cfg["n_nca_steps"],
            n_out=max_C,
            n_repeats=cfg.get("n_nca_repeats", 1),
            use_layernorm=cfg.get("use_layernorm", False),
            input_skip=cfg.get("input_skip", True),
            **pool_kwargs,
        ), max_tok_len

    vocab_size = int(cfg["vocab_size"])
    if arch == "rule_attn":
        from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
        return RuleAttnNCAWorldModel(
            n_hid=cfg["n_hid"],
            n_steps=cfg["n_nca_steps"],
            n_out=max_C,
            vocab_size=vocab_size + 1,
            enc_d_model=cfg["d_model"],
            enc_n_self_layers=cfg["n_enc_layers"],
            n_slots=cfg["n_slots"],
            n_app_slots=cfg.get("n_app_slots", 0),
            d_slot=cfg["d_slot"],
            n_attn_heads=cfg["n_heads"],
            max_seq_len=max_tok_len + 1,
            n_repeats=cfg.get("n_nca_repeats", 1),
            use_layernorm=cfg.get("use_layernorm", False),
            input_skip=cfg.get("input_skip", False),
            adaptive_halt=cfg.get("adaptive_halt", False),
            use_vq=cfg.get("vq_codebook", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
            **pool_kwargs,
        ), max_tok_len
    if arch == "film":
        from nca_wm.train import ConditionalNCAWorldModel
        return ConditionalNCAWorldModel(
            n_hid=cfg["n_hid"],
            n_steps=cfg["n_nca_steps"],
            n_out=max_C,
            vocab_size=vocab_size + 1,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_enc_layers=cfg["n_enc_layers"],
            d_z=cfg["d_z"],
            max_seq_len=max_tok_len + 1,
            sprite_decoder=False,
            use_layernorm=cfg.get("use_layernorm", False),
            **pool_kwargs,
        ), max_tok_len
    raise NotImplementedError(
        f"serve_wm only supports rule_attn / film conditional architectures; "
        f"got {arch!r}. (Single-game unconditional checkpoints have a different "
        f"shape — serve them via train.py --serve for now.)"
    )


def _scan_runs(logs_dir):
    """Return list of run dirs under logs_dir that contain a usable checkpoint.

    A run is usable iff config.json, game_infos.pkl, and at least one of
    params.pkl / params_best.pkl are present.
    """
    if not os.path.isdir(logs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(logs_dir)):
        d = os.path.join(logs_dir, name)
        if not os.path.isdir(d):
            continue
        has_cfg = os.path.exists(os.path.join(d, "config.json"))
        has_gi = os.path.exists(os.path.join(d, "game_infos.pkl"))
        has_p = (os.path.exists(os.path.join(d, "params.pkl"))
                 or os.path.exists(os.path.join(d, "params_best.pkl")))
        if has_cfg and has_gi and has_p:
            try:
                cfg = json.load(open(os.path.join(d, "config.json")))
            except Exception:
                cfg = {}
            try:
                with open(os.path.join(d, "game_infos.pkl"), "rb") as f:
                    n_games = len(pickle.load(f))
            except Exception:
                n_games = -1
            runs.append({
                "name": name,
                "path": d,
                "n_games": n_games,
                "arch": cfg.get("architecture", "?"),
                "n_hid": cfg.get("n_hid", "?"),
                "conditional": cfg.get("conditional", True),
                "mtime": os.path.getmtime(d),
            })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _pick_run(logs_dir, port, host):
    """Show the picker UI and return the run dir the user selects.

    Runs the picker on (host, port). On selection, the picker shuts its
    server down and returns the chosen path; main() then proceeds to
    serve_world_model in the same process — no execv, no inherited FDs.
    Returns None if the server is shut down without a selection.
    """
    import threading
    import time

    from flask import Flask, jsonify, request
    from werkzeug.serving import make_server

    runs = _scan_runs(logs_dir)
    print(f"\nPicker UI at http://{host}:{port} — found {len(runs)} run(s) in {logs_dir}")
    if not runs:
        print("  (no valid run dirs found — directory must contain "
              "config.json, params*.pkl, game_infos.pkl)")

    app = Flask(__name__)
    selected: dict = {"run": None}
    holder: dict = {"server": None}

    @app.route("/")
    def index():
        opts = "\n".join(
            f'<option value="{r["path"]}">{r["name"]} '
            f'&nbsp;|&nbsp; {r["n_games"]} games &nbsp;|&nbsp; {r["arch"]} '
            f'&nbsp;|&nbsp; n_hid={r["n_hid"]} '
            f'&nbsp;|&nbsp; {"cond" if r["conditional"] else "uncond"}</option>'
            for r in runs
        )
        return PICKER_HTML.replace("__OPTIONS__", opts)

    @app.route("/api/runs")
    def api_runs():
        return jsonify(runs)

    @app.route("/load", methods=["POST"])
    def load():
        run = request.form.get("run") or (request.json or {}).get("run")
        if not run or not os.path.isdir(run):
            return jsonify(error=f"invalid run: {run!r}"), 400
        selected["run"] = run

        def _shutdown():
            time.sleep(0.3)  # let the HTTP response flush first
            srv = holder["server"]
            if srv is not None:
                srv.shutdown()

        threading.Thread(target=_shutdown, daemon=True).start()
        return jsonify(ok=True)

    server = make_server(host, port, app, threaded=True)
    holder["server"] = server
    try:
        server.serve_forever()
    finally:
        server.server_close()  # release the listening socket
    return selected["run"]


def main():
    p = argparse.ArgumentParser(description="Serve a trained NCA world model.")
    p.add_argument("--load", action="append", default=None, metavar="DIR",
                   help="Run directory containing config.json, params.pkl, "
                        "and game_infos.pkl. Repeatable: pass --load twice to "
                        "serve several checkpoints in one viewer (the game "
                        "picker switches both the game and its model). If "
                        "omitted, a picker UI is shown.")
    p.add_argument("--logs_dir", default=DEFAULT_LOGS_DIR,
                   help="Directory scanned for runs when --load is omitted.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--game", default=None,
                   help="Initial game name (default: first game in checkpoint).")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--extra_games_dir", action="append", default=None,
                   help="Dir(s) prepended to the game-search path so games "
                        "living outside the default dirs (e.g. custom_games/"
                        "autumn ports) resolve by name. Repeatable.")
    args = p.parse_args()

    # Register extra game dirs before any compile_game() call (the server
    # recompiles each game by name to build its renderer backend).
    if args.extra_games_dir:
        from puzzlescript_jax.preprocessing import add_extra_games_dir
        for d in args.extra_games_dir:
            add_extra_games_dir(d)
            print(f"[serve_wm] registered games dir: {d}")

    load_dirs = args.load
    if not load_dirs:
        run_dir = _pick_run(args.logs_dir, args.port, args.host)
        if run_dir is None:
            print("Picker exited without a selection.")
            return
        load_dirs = [run_dir]
        print(f"\nPicker selected: {run_dir}\n"
              f"Loading model — JIT compile can take 30–90s.")

    import jax

    # Load each run, build its model, and append one bundle per game it owns.
    # Several single-game runs thus merge into one viewer; the game picker
    # routes apply/params/padding to the active game's bundle.
    game_infos: list[dict] = []
    bundles: list[dict] = []
    for d in load_dirs:
        with open(os.path.join(d, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(d, "params.pkl"), "rb") as f:
            params = pickle.load(f)
        with open(os.path.join(d, "game_infos.pkl"), "rb") as f:
            run_infos = pickle.load(f)

        wm_params = _unwrap_wm(params)
        model, max_tok_len = _build_wm(cfg, run_infos)
        apply_fn = jax.jit(model.apply)
        conditional = cfg.get("conditional", True)
        # Models pad inputs to the max (C,H,W) seen across the run's games.
        run_max_pad = (
            max(g["n_objs"] for g in run_infos),
            max(g["H"] for g in run_infos),
            max(g["W"] for g in run_infos),
        )
        bundle = dict(apply_fn=apply_fn, params=wm_params,
                      conditional=conditional, max_pad=run_max_pad,
                      max_tok_len=max_tok_len)
        for info in run_infos:
            game_infos.append(info)
            bundles.append(bundle)

        print(f"Loaded {os.path.basename(d.rstrip('/'))}: "
              f"{len(run_infos)} game(s) {[g['name'] for g in run_infos]}, "
              f"arch={cfg.get('architecture','rule_attn')}, "
              f"{'cond' if conditional else 'uncond'}, max_pad={run_max_pad}")

    initial_game_id = 0
    if args.game is not None:
        for i, info in enumerate(game_infos):
            if info["name"] == args.game:
                initial_game_id = i
                break
        else:
            names = [g["name"] for g in game_infos]
            raise ValueError(f"Game {args.game!r} not loaded; available: {names}")

    print(f"\nServing {len(game_infos)} game(s) from {len(load_dirs)} run(s): "
          f"{[g['name'] for g in game_infos]}")

    serve_world_model(
        game_infos, init_ps_lark_parser(), bundles,
        initial_game_id=initial_game_id,
        level_i=args.level,
        port=args.port,
        host=args.host,
    )


PICKER_HTML = r"""<!DOCTYPE html>
<html>
<head>
<title>NCA World Model — pick a run</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 18px; }
  select { width: 100%; padding: 8px; font-family: ui-monospace, monospace; font-size: 13px; }
  button { padding: 8px 18px; margin-top: 12px; font-size: 14px; cursor: pointer; }
  #status { margin-top: 14px; color: #666; min-height: 1.4em; }
  #spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid #ccc;
             border-top-color: #333; border-radius: 50%; animation: s 0.8s linear infinite;
             vertical-align: -2px; margin-right: 6px; }
  @keyframes s { to { transform: rotate(360deg); } }
  .hidden { display: none; }
</style>
</head>
<body>
  <h1>Select a run to serve</h1>
  <form id="f">
    <select id="run" name="run" size="20">
      __OPTIONS__
    </select>
    <br>
    <button type="submit">Load</button>
  </form>
  <div id="status"></div>
<script>
const f = document.getElementById('f');
const status = document.getElementById('status');

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const run = document.getElementById('run').value;
  if (!run) { status.textContent = 'Pick a run first.'; return; }
  status.innerHTML = '<span id="spinner"></span> Re-launching server with selected run. The model needs to load and JIT-compile (can take 30–90s)…';
  // Fire-and-forget: the server will execv() and the response may never arrive.
  fetch('/load', {method: 'POST',
                  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                  body: 'run=' + encodeURIComponent(run)}).catch(() => {});
  // Poll /api/state — the *real* server exposes it; the picker does not.
  // First poll after a short delay so the picker has a chance to die.
  setTimeout(poll, 1500);
});

async function poll() {
  try {
    const r = await fetch('/api/state', {cache: 'no-store'});
    if (r.ok) { window.location.reload(); return; }
  } catch (_) {}
  setTimeout(poll, 1000);
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
