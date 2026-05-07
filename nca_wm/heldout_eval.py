"""Zero-shot held-out game evaluation for a trained NCA world model.

Loads a trained checkpoint, evaluates its single-step + autoregressive
prediction quality on a list of games NOT in the training set, and compares
to an identity-predictor baseline.

Held-out games must satisfy:
  - n_objs <= model.n_out (the readout has fixed channel width)
  - tokenized rule sequence <= model.max_seq_len - 1 (CLS uses one slot)

Spatial dims are unconstrained (convs and rule-attn are size-flexible).

Usage:
    python nca_wm/heldout_eval.py \\
        --load nca_wm/logs/scaling_14_joint_v1 \\
        --heldout_games blank,sumo,the_undertaking,wrappingrecipe,rigidfail1,constellationz \\
        --n_random_episodes 5 --max_steps 30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.train import (
    ConditionalNCAWorldModel,
    NCAWorldModel,
    N_ACTIONS,
    _pad_offsets,
    _pad_state_for_model,
    _wm_p,
    _load_npz_dict,
    _save_npz_dict,
    _solution_from_sol_dir,
)
# Repo root, used to resolve js_sols / cpp_sols dirs.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cache for search-action sequences on heldout games. Re-used across all
# checkpoints since search is model-independent.
_HELDOUT_SEARCH_CACHE = os.path.join(_REPO_ROOT, "nca_wm", "data_cache",
                                     "heldout_search")
from nca_wm.tokenize_game import (
    VOCAB_SIZE_BASE, VOCAB_SIZE_EXT, VOCAB_SIZE_EXT_V2,
    tokenize_game, get_game_tree_from_js,
)
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser


# Default held-out set: the 6 games in `scaling_large` but not `scaling_14`.
DEFAULT_HELDOUT = [
    "blank", "sumo", "the_undertaking", "wrappingrecipe",
    "rigidfail1", "constellationz",
]


# ----------------------------------------------------------------------
# Checkpoint loading & model rebuild
# ----------------------------------------------------------------------

def _load_run(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    best_path = os.path.join(save_dir, "params_best.pkl")
    if os.path.exists(best_path):
        with open(best_path, "rb") as f:
            params = pickle.load(f)
        print(f"  loaded params_best.pkl")
    else:
        print(f"  loaded params.pkl (no params_best.pkl found)")
    return cfg, params, game_infos


def _build_model(cfg: dict, game_infos: list[dict]):
    """Reconstruct the model object using the train-time config + game_infos."""
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max((len(g.get("token_ids", [])) for g in game_infos), default=1)
    max_tok_len = max(max_tok_len, 1)
    if "vocab_size" in cfg:
        vocab_size = int(cfg["vocab_size"])
    elif cfg.get("kernel_sep", False):
        vocab_size = VOCAB_SIZE_EXT_V2
    elif cfg.get("encode_sprites", False):
        vocab_size = VOCAB_SIZE_EXT
    else:
        vocab_size = VOCAB_SIZE_BASE
    pool_kwargs = {
        "axis_pool": cfg.get("axis_pool", False),
        "axis_cummax": cfg.get("axis_cummax", False),
        "global_pool": cfg.get("global_pool", False),
    }
    if not cfg.get("conditional", False):
        return NCAWorldModel(
            n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"],
            n_out=max_C,
            use_layernorm=cfg.get("use_layernorm", False),
            input_skip=cfg.get("input_skip", True),
            n_repeats=cfg.get("n_nca_repeats", 1),
            **pool_kwargs,
        )
    arch = cfg.get("architecture", "film")
    if arch == "rule_attn":
        from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
        return RuleAttnNCAWorldModel(
            n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
            vocab_size=vocab_size + 1,
            enc_d_model=cfg["d_model"],
            enc_n_self_layers=cfg["n_enc_layers"],
            n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
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
        )
    return ConditionalNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
        vocab_size=vocab_size + 1,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_enc_layers=cfg["n_enc_layers"], d_z=cfg["d_z"],
        max_seq_len=max_tok_len + 1,
        sprite_decoder=cfg.get("sprite_loss_weight", 0.0) > 0.0,
        use_layernorm=cfg.get("use_layernorm", False),
        **pool_kwargs,
    )


# ----------------------------------------------------------------------
# Build held-out game_info from a name
# ----------------------------------------------------------------------

def _build_heldout_game_info(
    name: str, ps_parser, *,
    encode_sprites: bool, kernel_sep: bool,
) -> dict | None:
    backend = CppPuzzleScriptBackend()
    try:
        json_str = backend.compile_and_serialize(ps_parser, name)
    except Exception as e:
        print(f"  SKIP {name}: compile failed ({e})")
        return None
    try:
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    except Exception as e:
        print(f"  SKIP {name}: env init failed ({e})")
        return None
    n_objs = env0.observation_shape[0]
    n_levels = env0.num_levels
    if n_levels < 1:
        print(f"  SKIP {name}: no playable levels")
        return None
    game_max_H, game_max_W = 0, 0
    for li in range(n_levels):
        env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
        lC, lH, lW = env_li.observation_shape
        n_objs = max(n_objs, lC)
        game_max_H = max(game_max_H, lH)
        game_max_W = max(game_max_W, lW)
    try:
        tree, canonical_ids = get_game_tree_from_js(ps_parser, name)
        token_ids = tokenize_game(
            tree, canonical_ids,
            encode_sprites=encode_sprites,
        )
    except Exception as e:
        print(f"  SKIP {name}: tokenize failed ({e})")
        return None
    return {
        "name": name, "json_str": json_str,
        "n_objs": n_objs, "H": game_max_H, "W": game_max_W,
        "n_levels": n_levels,
        "token_ids": list(token_ids),
        "sprite_tensor": None,
    }


# ----------------------------------------------------------------------
# Rollout with model + identity baseline computed in lockstep
# ----------------------------------------------------------------------

def _rollout_with_identity(
    apply_fn, params, info: dict, level_i: int,
    *, max_C: int, model_token_capacity: int,
    actions: list[int] | None = None, max_steps: int = 30,
    teacher_forced: bool = False, conditional: bool = True,
    seed: int = 0,
) -> dict:
    """Run a single rollout. Returns model + identity per-step metrics.

    Model prediction is fed back autoregressively (or replaced by real obs
    if teacher_forced=True). Identity baseline is the trivial predictor that
    says the next state equals the current real state — so its per-step
    "wrong cells" equals the number of cells that actually changed in the
    real env at that step.
    """
    n_objs = info["n_objs"]
    json_str = info["json_str"]
    info_H = info["H"]
    info_W = info["W"]

    n_steps = len(actions) if actions else max_steps
    env = CppPuzzleScriptEnv(json_str, level_i=level_i,
                             max_episode_steps=n_steps)
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape
    # Pad to whichever is larger: the saved info dims (training shape) or
    # the actual level dims. Needed when training shape differs from authored
    # level shape (e.g. synthetic-trained model evaluated on bigger
    # authored levels). We round up to next-pow2 (min 8) to match training's
    # bucket convention (`_next_pow2` in train.py), so models see eval inputs at
    # the same bucketed shape they learned to operate on. Without this, a model
    # trained at bucket (8,8)
    # eval'd on a 7x7 native shape produces garbage because its mask-derived
    # features were learned for the (8,8) canvas.
    def _next_pow2(x: int, min_val: int = 8) -> int:
        v = max(min_val, int(x))
        p = 1
        while p < v:
            p <<= 1
        return p
    H_eval = max(_next_pow2(info_H), _next_pow2(H))
    W_eval = max(_next_pow2(info_W), _next_pow2(W))
    total_tiles = n_objs * H * W
    total_cells = H * W

    # Token padding: the model prepends CLS internally, so raw game tokens may
    # occupy at most encoder_max_seq_len - 1 positions.
    if conditional:
        tids = info.get("token_ids", [])
        if len(tids) > model_token_capacity:
            print(f"    WARNING: {info['name']} tokens ({len(tids)}) exceed "
                  f"model raw-token capacity ({model_token_capacity}); truncating")
            tids = tids[:model_token_capacity]
        padded_tok = np.zeros(model_token_capacity, dtype=np.int32)
        padded_mask = np.zeros(model_token_capacity, dtype=np.bool_)
        padded_tok[:len(tids)] = tids
        padded_mask[:len(tids)] = True
        gt = jnp.array(padded_tok[None])
        gm = jnp.array(padded_mask[None])

    pred_state = _pad_state_for_model(real_obs, max_C, H_eval, W_eval)

    rng = np.random.RandomState(seed)
    model_wrong_cells = []
    identity_wrong_cells = []
    model_wrong_tiles = []
    model_first_div = -1

    prev_real = real_obs.copy()
    for t in range(n_steps):
        action = actions[t] if actions else int(rng.randint(N_ACTIONS))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        if conditional:
            logits, _wl, _sl = apply_fn(_wm_p(params), pred_state, a_oh, gt, gm)
        else:
            logits, _wl, _sl = apply_fn(_wm_p(params), pred_state, a_oh)
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        real_next, _, done, truncated, _ = env.step(action)

        # Slice prediction down to real env's spatial extent — top-left
        # aligned to match training-time bucket padding.
        pred_binary = np.array(
            pred_next[0, :n_objs, :H, :W] > 0.5,
            dtype=np.uint8,
        )
        # Model error.
        m_mismatch = (pred_binary != real_next)
        m_wrong_bits = int(m_mismatch.sum())
        m_wrong_cells = int(m_mismatch.any(axis=0).sum())
        # Identity error: predict next == current. Mismatch == cells that changed.
        id_mismatch = (prev_real != real_next)
        id_wrong_cells = int(id_mismatch.any(axis=0).sum())

        model_wrong_cells.append(m_wrong_cells)
        model_wrong_tiles.append(m_wrong_bits)
        identity_wrong_cells.append(id_wrong_cells)
        if model_first_div == -1 and m_wrong_cells > 0:
            model_first_div = t

        if teacher_forced:
            pred_state = _pad_state_for_model(real_next, max_C, H_eval, W_eval)
        else:
            # Match training-time padding: zero outside the level's actual
            # (n_objs, H, W) extent. Otherwise binarized predictions in unused
            # channels / off-level cells leak in as OOD nonzero input the next
            # step, which the model never trained to tolerate.
            clean = jnp.zeros_like(pred_next)
            clean = clean.at[:, :n_objs, :H, :W].set(pred_next[:, :n_objs, :H, :W])
            pred_state = clean

        prev_real = real_next
        if done or truncated:
            break

    return {
        "model_wrong_cells": np.array(model_wrong_cells),
        "model_wrong_tiles": np.array(model_wrong_tiles),
        "identity_wrong_cells": np.array(identity_wrong_cells),
        "model_first_div": model_first_div,
        "total_tiles": total_tiles,
        "total_cells": total_cells,
    }


def _get_heldout_search_actions(
    name: str, level_i: int, algo: str, json_str: str,
    *, search_n_steps: int = 100_000, search_timeout_ms: int = 60_000,
) -> list[int] | None:
    """Return the BFS- or A*-optimal action sequence for a heldout (game,
    level, algo). Tries cached → js_sols → cpp_sols → live C++ search,
    then caches whatever it finds. Returns None if the engine cannot
    solve within the budget.
    """
    os.makedirs(_HELDOUT_SEARCH_CACHE, exist_ok=True)

    def _safe_path(s: str) -> str:
        # Filenames cannot contain '/', and we keep the rest readable.
        return s.replace("/", "_")

    cache_path = os.path.join(
        _HELDOUT_SEARCH_CACHE,
        f"{_safe_path(name)}_L{level_i}_{algo}_{search_n_steps}_{search_timeout_ms}.npz",
    )
    cached = _load_npz_dict(cache_path)
    cache_valid = False
    if cached is not None and len(cached.get("actions", [])) > 0:
        source_algo = str(cached.get("source_algo", ""))
        cache_valid = source_algo == algo
    if cache_valid:
        return cached["actions"].tolist()

    # cpp_sols use the C++-backend action convention; drop-in compatible.
    sol_actions = _solution_from_sol_dir(
        os.path.join(_REPO_ROOT, "data", "cpp_sols"),
        name, level_i, translate_js_to_jax=False,
        algos=(algo,),
    )
    source_kind = "cpp_sols" if sol_actions is not None else ""
    # js_sols use the JS-engine action convention; remap to JAX/CPP.
    if sol_actions is None:
        sol_actions = _solution_from_sol_dir(
            os.path.join(_REPO_ROOT, "data", "js_sols"),
            name, level_i, translate_js_to_jax=True,
            algos=(algo,),
        )
        source_kind = "js_sols" if sol_actions is not None else ""

    if sol_actions is None:
        # Live C++ search.
        try:
            backend = CppPuzzleScriptBackend()
            backend.load_from_json(json_str)
            backend.load_level("", level_i)
            result = backend.run_search(
                algo, game_text="", level_i=level_i,
                n_steps=search_n_steps, timeout_ms=search_timeout_ms,
            )
            if result.actions:
                sol_actions = list(result.actions)
                source_kind = "live_search"
        except Exception as e:
            print(f"      [search] {algo} {name} L{level_i} live failed: {e}")
            return None

    if sol_actions is not None:
        _save_npz_dict(cache_path, {
            "actions": np.asarray(sol_actions, dtype=np.int32),
            "source_algo": np.asarray(algo),
            "source_kind": np.asarray(source_kind),
        })
    return sol_actions


def _aggregate_episodes(eps: list[dict]) -> dict:
    """Mean per-step error rates across episodes, padded to longest."""
    max_len = max(len(e["model_wrong_cells"]) for e in eps)
    n_eps = len(eps)

    def _pad(arr_name):
        out = np.full((n_eps, max_len), np.nan)
        for i, e in enumerate(eps):
            a = e[arr_name]
            out[i, :len(a)] = a
        return out

    model_cells = _pad("model_wrong_cells")
    id_cells = _pad("identity_wrong_cells")
    total_cells = eps[0]["total_cells"]

    model_cer = np.nanmean(model_cells, axis=0) / total_cells
    id_cer = np.nanmean(id_cells, axis=0) / total_cells
    first_divs = np.array([
        max_len if e["model_first_div"] < 0 else e["model_first_div"]
        for e in eps
    ])
    return {
        "model_cell_err_per_step": model_cer,        # (T,)
        "identity_cell_err_per_step": id_cer,        # (T,)
        "model_cell_err_step1": float(model_cer[0]),
        "identity_cell_err_step1": float(id_cer[0]),
        "model_cell_err_mean": float(np.nanmean(model_cer)),
        "identity_cell_err_mean": float(np.nanmean(id_cer)),
        "mean_first_div": float(first_divs.mean()),
        "total_cells": int(total_cells),
        "n_episodes": n_eps,
        "n_steps": int(max_len),
    }


# ----------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------

def evaluate_heldout(
    save_dir: str,
    heldout_games: list[str],
    *,
    n_random_episodes: int = 5,
    max_steps: int = 30,
    include_train_sample: int = 0,
    out_subdir: str = "heldout_eval",
    seed: int = 0,
    max_levels_per_game: int | None = None,
    allow_training_games: bool = False,
):
    print(f"\n=== Loading checkpoint from {save_dir} ===")
    cfg, params, train_game_infos = _load_run(save_dir)
    model = _build_model(cfg, train_game_infos)

    # Architecture-derived bounds.
    max_C = max(g["n_objs"] for g in train_game_infos)
    train_max_seq_len = max(len(g.get("token_ids", [])) for g in train_game_infos)
    train_max_seq_len = max(train_max_seq_len, 1)
    # Model's pos-embed table is sized to train_max_seq_len + 1 (for CLS).
    model_max_seq_len = train_max_seq_len + 1
    model_token_capacity = model_max_seq_len - 1

    print(f"  model: {type(model).__name__}, n_hid={cfg['n_hid']}, "
          f"n_out={max_C}, max_seq_len={model_max_seq_len}, "
          f"raw_token_capacity={model_token_capacity}, "
          f"arch={cfg.get('architecture', 'film')}")
    print(f"  trained on {len(train_game_infos)} games: "
          f"{[g['name'] for g in train_game_infos]}")
    train_game_names = {g["name"] for g in train_game_infos}

    print(f"\n=== Building held-out game infos ===")
    ps_parser = init_ps_lark_parser()
    heldout_infos = []
    skipped = []
    encode_sprites = cfg.get("encode_sprites", False)
    kernel_sep = cfg.get("kernel_sep", False)
    for name in heldout_games:
        if name in train_game_names and not allow_training_games:
            print(f"  SKIP {name}: in training set, not held-out")
            skipped.append((name, "in_training_set"))
            continue
        if name in train_game_names and allow_training_games:
            print(f"  ALLOW {name}: in training set, rebuilding authored levels for eval")
        info = _build_heldout_game_info(
            name, ps_parser,
            encode_sprites=encode_sprites, kernel_sep=kernel_sep,
        )
        if info is None:
            skipped.append((name, "build_failed"))
            continue
        if info["n_objs"] > max_C:
            print(f"  SKIP {name}: n_objs={info['n_objs']} > model.n_out={max_C}")
            skipped.append((name, f"n_objs_too_large_{info['n_objs']}"))
            continue
        if len(info["token_ids"]) > model_token_capacity:
            print(f"  WARNING: {name} tokens ({len(info['token_ids'])}) > "
                  f"model raw-token capacity ({model_token_capacity}) — will truncate")
        print(f"  OK   {name}: n_objs={info['n_objs']} ({max_C}), "
              f"shape=({info['H']}, {info['W']}), "
              f"tokens={len(info['token_ids'])}, levels={info['n_levels']}")
        heldout_infos.append(info)

    # Optionally pull a random sample of training games for in-distribution comparison.
    train_sample_infos = []
    if include_train_sample > 0:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_game_infos),
                         size=min(include_train_sample, len(train_game_infos)),
                         replace=False)
        train_sample_infos = [train_game_infos[i] for i in idx]
        print(f"\n=== In-distribution control: "
              f"{[g['name'] for g in train_sample_infos]} ===")

    apply_fn = jax.jit(model.apply)
    conditional = cfg.get("conditional", False)

    # ------------------------------------------------------------------
    # Run rollouts.
    # ------------------------------------------------------------------
    results = {"heldout": {}, "train_sample": {}, "skipped": skipped,
               "config": cfg}
    rng_seed = seed

    def _eval_one_game(info, bucket: str):
        name = info["name"]
        per_level = {}
        n_levels_to_eval = info["n_levels"]
        if max_levels_per_game is not None:
            n_levels_to_eval = min(n_levels_to_eval, max_levels_per_game)
        for li in range(n_levels_to_eval):
            print(f"\n  [{bucket}] {name} L{li} "
                  f"(shape={info['H']}x{info['W']}, n_objs={info['n_objs']}) "
                  f"-- {n_random_episodes} random eps x {max_steps} steps")
            t0 = time.time()
            eps = []
            for ep_i in range(n_random_episodes):
                r = _rollout_with_identity(
                    apply_fn, params, info, level_i=li,
                    max_C=max_C, model_token_capacity=model_token_capacity,
                    actions=None, max_steps=max_steps,
                    teacher_forced=False,
                    conditional=conditional,
                    seed=rng_seed + 1000 * li + ep_i,
                )
                eps.append(r)
            agg = _aggregate_episodes(eps)
            tf_eps = []
            for ep_i in range(n_random_episodes):
                r = _rollout_with_identity(
                    apply_fn, params, info, level_i=li,
                    max_C=max_C, model_token_capacity=model_token_capacity,
                    actions=None, max_steps=max_steps,
                    teacher_forced=True,
                    conditional=conditional,
                    seed=rng_seed + 1000 * li + ep_i,
                )
                tf_eps.append(r)
            agg_tf = _aggregate_episodes(tf_eps)
            per_level[li] = {"random": agg, "random_tf": agg_tf}
            print(f"    AR step1 model={agg['model_cell_err_step1']:.3f} "
                  f"identity={agg['identity_cell_err_step1']:.3f}  "
                  f"mean model={agg['model_cell_err_mean']:.3f} "
                  f"identity={agg['identity_cell_err_mean']:.3f}  "
                  f"first_div={agg['mean_first_div']:.1f}  "
                  f"({time.time()-t0:.1f}s)")
            print(f"    TF step1 model={agg_tf['model_cell_err_step1']:.4f} "
                  f"identity={agg_tf['identity_cell_err_step1']:.3f}")

            # --- Search rollouts (BFS-optimal and A*-optimal, AR mode) ---
            # Search is deterministic so a single rollout per algo suffices.
            for algo in ("bfs", "astar"):
                t_search = time.time()
                actions = _get_heldout_search_actions(
                    info["name"], li, algo, info["json_str"],
                )
                if actions is None or len(actions) == 0:
                    print(f"    {algo:5s} no solution within budget; skipped")
                    continue
                r = _rollout_with_identity(
                    apply_fn, params, info, level_i=li,
                    max_C=max_C, model_token_capacity=model_token_capacity,
                    actions=actions, max_steps=len(actions),
                    teacher_forced=False,
                    conditional=conditional,
                    seed=rng_seed + 1000 * li,
                )
                agg_s = _aggregate_episodes([r])
                per_level[li][algo] = agg_s
                print(f"    {algo:5s} step1 model={agg_s['model_cell_err_step1']:.3f} "
                      f"identity={agg_s['identity_cell_err_step1']:.3f}  "
                      f"mean model={agg_s['model_cell_err_mean']:.3f} "
                      f"identity={agg_s['identity_cell_err_mean']:.3f}  "
                      f"({time.time()-t_search:.1f}s, n_actions={len(actions)})")
        return per_level

    print(f"\n=== Evaluating {len(heldout_infos)} held-out games ===")
    for info in heldout_infos:
        results["heldout"][info["name"]] = _eval_one_game(info, "heldout")

    if train_sample_infos:
        print(f"\n=== Evaluating {len(train_sample_infos)} in-distribution control games ===")
        for info in train_sample_infos:
            results["train_sample"][info["name"]] = _eval_one_game(info, "train_sample")

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    out_dir = os.path.join(save_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "results.json")

    def _to_jsonable(x):
        if isinstance(x, dict):
            return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    with open(summary_path, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    print(f"\nSaved per-game JSON to {summary_path}")

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return results

    _plot_summary(results, out_dir)
    return results


def _plot_summary(results: dict, out_dir: str):
    import matplotlib.pyplot as plt

    def _flatten(bucket: str):
        rows = []
        for name, per_level in results.get(bucket, {}).items():
            # Average across levels (weighted by n_steps for fairness)
            mod_means = []
            id_means = []
            mod_step1s = []
            id_step1s = []
            mod_tf_step1s = []
            for li, level in per_level.items():
                ar = level["random"]
                tf = level["random_tf"]
                mod_means.append(ar["model_cell_err_mean"])
                id_means.append(ar["identity_cell_err_mean"])
                mod_step1s.append(ar["model_cell_err_step1"])
                id_step1s.append(ar["identity_cell_err_step1"])
                mod_tf_step1s.append(tf["model_cell_err_step1"])
            rows.append({
                "name": name,
                "model_mean": np.mean(mod_means),
                "identity_mean": np.mean(id_means),
                "model_step1_ar": np.mean(mod_step1s),
                "identity_step1_ar": np.mean(id_step1s),
                "model_step1_tf": np.mean(mod_tf_step1s),
            })
        return rows

    held = _flatten("heldout")
    train = _flatten("train_sample")

    if not held and not train:
        return

    # ---- Figure 1: bar chart, model vs identity per game ----
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.7 * (len(held) + len(train) + 1)), 5))
    rows = [(r, "heldout") for r in held] + [(r, "train") for r in train]
    names = [r["name"] for r, _ in rows]
    model_step1 = [r["model_step1_tf"] for r, _ in rows]
    model_step1_ar = [r["model_step1_ar"] for r, _ in rows]
    identity_step1 = [r["identity_step1_ar"] for r, _ in rows]

    x = np.arange(len(names))
    w = 0.27
    bars_id = ax.bar(x - w, identity_step1, w, color="0.7", label="identity (step 1)")
    bars_m_tf = ax.bar(x, model_step1, w, color="#3a7", label="model TF (step 1)")
    bars_m_ar = ax.bar(x + w, model_step1_ar, w, color="#26b", label="model AR (step 1)")

    for i, (_, kind) in enumerate(rows):
        for bar in [bars_id[i], bars_m_tf[i], bars_m_ar[i]]:
            if kind == "heldout":
                bar.set_edgecolor("crimson")
                bar.set_linewidth(2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("cell error rate (step 1)")
    ax.set_title("Held-out (red border) vs in-training (no border): "
                 "per-game step-1 cell error rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "step1_bar.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"  saved {p1}")

    # ---- Figure 2: per-step rollout curves ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for bucket, ax in zip(["heldout", "train_sample"], axes):
        r_dict = results.get(bucket, {})
        if not r_dict:
            ax.set_title(f"{bucket}: (none)")
            continue
        for name, per_level in r_dict.items():
            curves_model = []
            curves_id = []
            for li, level in per_level.items():
                curves_model.append(level["random"]["model_cell_err_per_step"])
                curves_id.append(level["random"]["identity_cell_err_per_step"])
            # Pad and average across levels per game
            T = max(len(c) for c in curves_model)
            cm = np.full((len(curves_model), T), np.nan)
            ci = np.full((len(curves_id), T), np.nan)
            for i, (a, b) in enumerate(zip(curves_model, curves_id)):
                cm[i, :len(a)] = a
                ci[i, :len(b)] = b
            ax.plot(np.nanmean(cm, axis=0), label=f"{name} model", lw=1.6)
            ax.plot(np.nanmean(ci, axis=0), "--", color=ax.lines[-1].get_color(),
                    alpha=0.5, label=f"{name} identity")
        ax.set_title(f"{bucket}: per-step cell error (autoregressive)")
        ax.set_xlabel("step")
        if ax is axes[0]:
            ax.set_ylabel("cell error rate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "rollout_curves.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"  saved {p2}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True,
                    help="Path to a trained-checkpoint save_dir.")
    ap.add_argument("--heldout_games", default=";".join(DEFAULT_HELDOUT),
                    help=f"Semicolon-separated game names (some PuzzleScript filenames "
                         f"contain commas, so we deliberately do not split on ','). "
                         f"Default: {DEFAULT_HELDOUT}")
    ap.add_argument("--n_random_episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--include_train_sample", type=int, default=3,
                    help="How many training games to also re-eval (control). 0 = skip.")
    ap.add_argument("--out_subdir", default="heldout_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_levels_per_game", type=int, default=None,
                    help="Cap the number of levels evaluated per held-out game (for fast turnaround).")
    ap.add_argument("--allow_training_games", action="store_true",
                    help="Do not skip names that were in the training set; rebuild/evaluate their authored levels.")
    args = ap.parse_args()

    heldout = [g.strip() for g in args.heldout_games.split(";") if g.strip()]
    evaluate_heldout(
        args.load, heldout,
        n_random_episodes=args.n_random_episodes,
        max_steps=args.max_steps,
        include_train_sample=args.include_train_sample,
        out_subdir=args.out_subdir,
        seed=args.seed,
        max_levels_per_game=args.max_levels_per_game,
        allow_training_games=args.allow_training_games,
    )


if __name__ == "__main__":
    main()
