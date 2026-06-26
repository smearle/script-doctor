"""GIF / image rendering for the NCA world model.

Training-time and post-training visualization: rollout GIFs (real vs.
predicted vs. error), per-step activation grids, sprite-kernel overlays,
interactive playback, and side-by-side comparisons. Imports the model classes,
the shared inference helpers, and the leaf state/data helpers, but never
train.py, so it carries no import cycle.
"""
import os
import sys
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import wandb

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from nca_wm.models import NCAWorldModel, ConditionalNCAWorldModel
from nca_wm.state_ops import _multihot_to_objects
from nca_wm.data_collection import _enabled_actions
from nca_wm.inference import make_apply_fn, _pad_state_for_model, _unpad_pred

N_ACTIONS = 6  # 0-3 move, 4 action, 5 no-op real-time tick (realtime games only)


def _labeled_channel_grid(logits_chw, obj_names, pad=2, scale=4):
    """Render per-object output channels as a labeled grid.

    Always shows ALL channels in order, labeled with object names.

    Args:
        logits_chw: (C, H, W) logits or probabilities.
        obj_names: list of C object name strings.
        scale: upscale each cell by this factor for readability.
    Returns:
        (grid_H, grid_W, 3) uint8 image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    C, H, W = logits_chw.shape
    logits_f = np.clip(np.array(logits_chw, dtype=np.float32), -50, 50)
    probs = 1.0 / (1.0 + np.exp(-logits_f))

    # Always show all channels in order
    ncols = min(10, C)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.5),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for idx in range(C):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.imshow(probs[idx], vmin=0, vmax=1, cmap="magma",
                  interpolation="nearest", aspect="equal")
        name = obj_names[idx] if idx < len(obj_names) else f"ch{idx}"
        # Truncate long names
        if len(name) > 14:
            name = name[:12] + ".."
        ax.set_title(name, fontsize=5, pad=2)

    fig.tight_layout(pad=0.3)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


def _render_training_gif(
    apply_fn, params, game_info,
    *,
    max_C: int, max_H: int, max_W: int,
    save_path: str,
    backend_render,
    n_steps: int = 15,
    seed: int = 0,
    conditional: bool,
    game_tokens=None, game_mask=None,
    banner_text: str = "",
    level_i: int = 0,
    actions: list[int] | None = None,
):
    """Render a (real | NCA prediction | diff) side-by-side rollout GIF.

    Designed to be fast enough to run intermittently during training — a
    single short rollout (default 15 steps) rendered at native resolution.
    Works for any architecture (rule_attn included) since it only calls the
    model's own ``apply_fn`` — no viz/intermediates rebuild.

    Left panel = real env rollout (declared sprites). Middle = the model's
    autoregressive prediction, thresholded to multihot and rendered with the
    same sprites (learned sprite kernels if the model has a decoder). Right =
    the real frame with mispredicted cells tinted red, so AR drift is visible.

    ``actions`` (if given) replaces the seeded random action sequence — pass
    the exact action row used by eval to render the rollout behind a metric.
    ``level_i`` selects the level (default 0).
    """
    import imageio.v2 as imageio
    from puzzlescript_jax.font import (
        draw_text as _ps_draw_text, GLYPH_H_COMPACT,
    )

    rng = np.random.RandomState(seed)
    json_str = game_info["json_str"]
    n_objs = game_info["n_objs"]
    n_loop = len(actions) if actions is not None else n_steps

    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=n_loop)
    real_obs, _ = env.reset()
    _, H, W = real_obs.shape

    # Load this level into the provided renderer (sprite data was pre-loaded
    # by the caller via compile_game).
    backend_render.load_level(game_text="", level_i=level_i)

    # Fetch learned sprite kernels once for the whole rollout. Only active
    # when the underlying model has sprite_decoder=True (otherwise the
    # sprite_logits slot is zeros and we fall back to declared sprites).
    learned_sprites_u8 = None
    if conditional and game_tokens is not None:
        gt0 = jnp.array(game_tokens[None])
        gm0 = jnp.array(game_mask[None])
        dummy_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
        dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)
        _, _, sprite_logits = apply_fn(params, dummy_state, dummy_action, gt0, gm0)
        sprite_logits_np = np.asarray(sprite_logits[0])  # (n_out, 5, 5, 4)
        # Decoder-off models (encode_sprites=False, e.g. rule_attn here) emit
        # identically-zero sprite logits — sigmoid(0)=0.5 is a uniform grey, so
        # gate on the RAW logits, not the sigmoid. Otherwise the prediction
        # panel renders as grey mush instead of the game's declared sprites.
        if float(np.abs(sprite_logits_np).max()) > 1e-4:
            sp = 1.0 / (1.0 + np.exp(-sprite_logits_np[:n_objs]))
            learned_sprites_u8 = (sp * 255.0).clip(0, 255).astype(np.uint8)

    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}

    def _render_multihot_declared(mh: np.ndarray) -> np.ndarray:
        """(n_objs, H, W) binary → (5H, 5W, 3) uint8, using the game's
        declared sprites via the C++ backend renderer."""
        return backend_render.render_frame_from_objects(
            _multihot_to_objects(mh), W, H
        )[..., :3]

    def _render_multihot_learned(mh: np.ndarray) -> np.ndarray:
        """(n_objs, H, W) binary → image using learned sprite kernels."""
        return _render_obs_with_sprite_kernels(
            mh.astype(np.float32), learned_sprites_u8
        )

    def _tint_diff(real_img: np.ndarray, mismatch_hw: np.ndarray) -> np.ndarray:
        """Real render with mispredicted cells tinted red — makes AR drift
        visible at a glance (a cell is 'wrong' if ANY object channel differs)."""
        out = real_img.astype(np.float32)
        mask = np.kron(mismatch_hw.astype(np.float32),
                       np.ones((5, 5), dtype=np.float32))[..., None]
        red = np.array([255.0, 40.0, 40.0], dtype=np.float32)
        out = out * (1.0 - 0.6 * mask) + red * (0.6 * mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Prediction render: learned sprite kernels if the model has a sprite
    # decoder, else the game's declared sprites. (The old soft sigmoid-composite
    # panel was dropped — for decoder-free models like rule_attn it just blended
    # all object sprites at ~0.5 alpha into grey mush.)
    _render_pred_hard = (_render_multihot_learned if learned_sprites_u8 is not None
                         else _render_multihot_declared)

    def render_panels(real_obs_raw, pred_hard, step_i, action):
        """real | NCA prediction | diff (red = mispredicted cell), all using
        the same sprites so the two rollouts are directly comparable."""
        real_img = _render_multihot_declared(real_obs_raw)
        pred_img = _render_pred_hard(pred_hard)
        mismatch = (pred_hard.astype(np.uint8)
                    != real_obs_raw.astype(np.uint8)).any(axis=0)
        diff_img = _tint_diff(real_img, mismatch)

        gap = 4
        banner_h = GLYPH_H_COMPACT + 4
        hR, wR = real_img.shape[:2]
        canvas = np.zeros((hR + banner_h, wR * 3 + gap * 2, 3), dtype=np.uint8)
        canvas[banner_h:banner_h + hR, :wR] = real_img
        canvas[banner_h:banner_h + hR, wR + gap: 2 * wR + gap] = pred_img
        canvas[banner_h:banner_h + hR, 2 * (wR + gap):] = diff_img
        # Short action codes that render readable in the PuzzleScript font.
        # Lowercase 'v' is drawn as a left-leaning slash here, so use 'V'.
        short_act = {0: "^", 1: "<", 2: "V", 3: ">", 4: "x"}
        a_str = short_act.get(action, "-") if step_i > 0 else "r"
        line = f"t{step_i:02d} {a_str} d{int(mismatch.sum())} {banner_text}"
        _ps_draw_text(canvas, line, x=2, y=2, color=(255, 255, 255),
                      compact=True)
        return canvas

    # Pad real_obs to model shape for the prediction stream
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)

    # At t=0 the "prediction" is just a copy of real (no model call yet).
    frames = [render_panels(real_obs, real_obs, 0, -1)]
    if conditional:
        gt = jnp.array(game_tokens[None])
        gm = jnp.array(game_mask[None])

    acts = _enabled_actions(json_str)
    for t in range(n_loop):
        action = int(actions[t]) if actions is not None else int(rng.choice(acts))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])
        if conditional:
            logits, _, _sprite_logits = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, _, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, _ = env.step(action)
        # Slice pred back to real (top-left-aligned) extent — matches the
        # top-left padding done by _pad_state_for_model.
        pred_crop = np.array(
            pred_state[0, :n_objs, :H, :W] > 0.5, dtype=np.uint8,
        )
        frames.append(render_panels(real_obs, pred_crop, t + 1, action))
        if done or truncated:
            break

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.15, loop=0)
    return save_path


def _render_rollout_frames(
    model, params, apply_fn, json_str, backend_render,
    level_i, n_objs, grid_h, grid_w, max_H, max_W,
    n_steps, actions=None, label="", obj_names=None,
    game_tokens=None, game_mask=None,
):
    """Run a rollout and return labeled frames.

    Each frame is composed vertically:
      - Banner with label
      - Game renders: real | NCA prediction
      - Hidden activation grid (final NCA step)
      - Per-object output channel predictions
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    conditional = game_tokens is not None
    if conditional:
        gt = jnp.array(game_tokens[None])
        gm = jnp.array(game_mask[None])

    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=n_steps)
    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, max_H, max_W)
    frames = []

    # Model variant with intermediates. Reconstruct the SAME architecture as
    # `model` (copying its config fields) so the trained params load and the
    # forward is identical — only return_intermediates is flipped on. The
    # earlier code always built a ConditionalNCAWorldModel for the conditional
    # branch, which crashed for rule_attn (no `d_model` field).
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    pool_kwargs = dict(
        axis_pool=model.axis_pool, axis_cummax=model.axis_cummax,
        global_pool=model.global_pool,
    )
    if isinstance(model, RuleAttnNCAWorldModel):
        model_viz = RuleAttnNCAWorldModel(
            n_hid=model.n_hid, n_steps=model.n_steps, n_out=model.n_out,
            vocab_size=model.vocab_size, max_seq_len=model.max_seq_len,
            enc_d_model=model.enc_d_model,
            enc_n_self_layers=model.enc_n_self_layers,
            n_slots=model.n_slots, n_app_slots=model.n_app_slots,
            d_slot=model.d_slot, n_attn_heads=model.n_attn_heads,
            use_vq=model.use_vq, vq_codebook_size=model.vq_codebook_size,
            vq_commitment_weight=model.vq_commitment_weight,
            use_layernorm=model.use_layernorm, input_skip=model.input_skip,
            n_repeats=model.n_repeats, adaptive_halt=model.adaptive_halt,
            history=model.history, return_intermediates=True, **pool_kwargs,
        )
    elif conditional:
        model_viz = ConditionalNCAWorldModel(
            n_hid=model.n_hid, n_steps=model.n_steps, n_out=model.n_out,
            vocab_size=model.vocab_size, d_model=model.d_model,
            n_heads=model.n_heads, n_enc_layers=model.n_enc_layers,
            d_z=model.d_z, max_seq_len=model.max_seq_len,
            return_intermediates=True, history=model.history, **pool_kwargs,
        )
    else:
        model_viz = NCAWorldModel(
            n_hid=model.n_hid, n_steps=model.n_steps, n_out=model.n_out,
            input_skip=model.input_skip, n_repeats=model.n_repeats,
            use_layernorm=model.use_layernorm,
            return_intermediates=True, history=model.history, **pool_kwargs,
        )
    viz_fn = jax.jit(model_viz.apply)

    # Object names for labeling channels (pad with generic names for extra channels)
    if obj_names is None:
        obj_names = [f"ch{i}" for i in range(model.n_out)]
    while len(obj_names) < model.n_out:
        obj_names.append(f"pad{len(obj_names)}")

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = PIL.ImageFont.load_default()

    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
    # Per-frame durations: NCA intermediate steps are fast, game steps hold longer
    durations = []

    def _compose_frame(game_img, hid_img, banner_text):
        """Stack banner + game render + activation grid vertically."""
        sections = [game_img, hid_img]
        max_w = max(s.shape[1] for s in sections)
        text_bbox = font.getbbox(banner_text)
        text_w = text_bbox[2] - text_bbox[0] + 8
        bh = 18
        bw = max(max_w, text_w)
        banner = PIL.Image.new("RGB", (bw, bh), (0, 0, 0))
        draw = PIL.ImageDraw.Draw(banner)
        draw.text((4, 2), banner_text, fill=(255, 255, 255), font=font)
        final_w = max(bw, max_w)
        parts = [np.array(banner)]
        for s in sections:
            if s.shape[1] < final_w:
                p = np.zeros((s.shape[0], final_w - s.shape[1], 3), dtype=np.uint8)
                s = np.concatenate([s, p], axis=1)
            parts.append(s)
        return np.concatenate(parts, axis=0)

    max_steps = len(actions) if actions else n_steps
    acts = _enabled_actions(json_str)
    last_obj_img = None
    for t in range(max_steps):
        action = actions[t] if actions else int(np.random.choice(acts))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # --- Game renders ---
        real_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend_render.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        game_row = np.concatenate([real_frame, pred_frame], axis=1)

        # --- NCA internals ---
        # viz_fn is raw model_viz.apply (no safety-net wrapper), so supply
        # zero (masked) history explicitly when the model expects it. The
        # activation panel visualizes hidden state at a single step; zero
        # history is an acceptable context-free view here.
        _vkw = {}
        if int(getattr(model, "history", 0)) > 0:
            k = int(model.history)
            _vkw = dict(
                hist_states=jnp.zeros((pred_state.shape[0], k)
                                      + tuple(pred_state.shape[1:]),
                                      pred_state.dtype),
                hist_actions=jnp.zeros((pred_state.shape[0], k), jnp.int32),
            )
        if conditional:
            logits, win_logit, _sprite_logits, intermediates = viz_fn(params, pred_state, a_oh, gt, gm, **_vkw)
        else:
            logits, win_logit, _sprite_logits, intermediates = viz_fn(params, pred_state, a_oh, **_vkw)

        pred_won_prob = float(jax.nn.sigmoid(win_logit)[0])

        # Build per-NCA-step output channel images (only game object channels,
        # cropped from the top-left, matching _pad_state_for_model's
        # top-left-aligned input padding).
        n_nca_steps = len(intermediates["readouts"])
        obj_imgs = []
        for readout in intermediates["readouts"]:
            r_np = np.array(readout[0, :n_objs, :grid_h, :grid_w])
            obj_img = _labeled_channel_grid(r_np, obj_names[:n_objs])
            obj_imgs.append(obj_img)
        last_obj_img = obj_imgs[-1]

        # Scale up game renders to match the height of one object grid
        target_game_h = obj_imgs[0].shape[0]
        if game_row.shape[0] < target_game_h:
            scale_factor = target_game_h / game_row.shape[0]
            new_w = int(game_row.shape[1] * scale_factor)
            game_pil = PIL.Image.fromarray(game_row).resize(
                (new_w, target_game_h), PIL.Image.NEAREST
            )
            game_row = np.array(game_pil)

        # Emit one frame per NCA intermediate step (animated quickly)
        for nca_i, oimg in enumerate(obj_imgs):
            text = (f"{label} t={t} {action_names.get(action, '?')}  (real | NCA)  "
                    f"NCA {nca_i}/{n_nca_steps}  pred_win={pred_won_prob:.2f}")
            frame = _compose_frame(game_row, oimg, text)
            frames.append(frame)
            durations.append(0.05)
        durations[-1] = 0.3

        # Step NCA + env
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        real_obs, _, done, truncated, info = env.step(action)
        real_won = bool(info.get("won", False))

        if done or truncated:
            # Render the terminal (post-action) state with win annotations
            real_frame_f = backend_render.render_frame_from_objects(
                _multihot_to_objects(real_obs), grid_w, grid_h
            )
            pred_obs_f = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
            pred_frame_f = backend_render.render_frame_from_objects(
                _multihot_to_objects(pred_obs_f), grid_w, grid_h
            )
            game_row_f = np.concatenate([real_frame_f, pred_frame_f], axis=1)
            if game_row_f.shape[0] < target_game_h:
                new_w = int(game_row_f.shape[1] * (target_game_h / game_row_f.shape[0]))
                game_pil = PIL.Image.fromarray(game_row_f).resize(
                    (new_w, target_game_h), PIL.Image.NEAREST
                )
                game_row_f = np.array(game_pil)
            blank_act = np.zeros_like(last_obj_img)
            reason = "win" if real_won else ("done" if done else "truncated")
            text = (f"{label} terminal ({reason})  real_win={int(real_won)}  "
                    f"pred_win={pred_won_prob:.2f}")
            frame = _compose_frame(game_row_f, blank_act, text)
            frames.append(frame)
            durations.append(1.5)
            break

    return frames, durations


def render_multigame_gifs(
    model: NCAWorldModel,
    params,
    game_infos: list[dict],
    ps_parser,
    n_steps_per_game: int = 30,
    save_dir: str = ".",
    step_label: int | None = None,
    search_algos: list[str] = ("bfs", "astar"),
    search_n_steps: int = 100_000,
    search_timeout_ms: int = -1,
):
    """Render a single combined GIF: for each game and level, random rollout then search rollout.

    The GIF filename includes the training step count for easy comparison across checkpoints.
    """
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)
    from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
    from nca_wm.baselines import CNNWorldModel, UNetWorldModel, ViTWorldModel
    conditional = isinstance(
        model,
        (ConditionalNCAWorldModel, RuleAttnNCAWorldModel,
         CNNWorldModel, UNetWorldModel, ViTWorldModel),
    )
    os.makedirs(save_dir, exist_ok=True)

    # Prepare padded token arrays for conditional rendering
    if conditional:
        max_tok_len = max(len(g.get("token_ids", [])) for g in game_infos)
        max_tok_len = max(max_tok_len, 1)

    def _get_token_kwargs(info):
        if not conditional:
            return {}
        tids = info.get("token_ids", [])
        padded = np.zeros(max_tok_len, dtype=np.int32)
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        padded[:len(tids)] = tids
        mask[:len(tids)] = True
        return {"game_tokens": padded, "game_mask": mask}

    apply_fn = make_apply_fn(model)
    tag = f"_step{step_label}" if step_label is not None else ""

    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = info["n_levels"]
        cond_kwargs = _get_token_kwargs(info)

        backend_render = CppPuzzleScriptBackend()
        backend_render.compile_game(ps_parser, name)

        # Get object names for this game
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        obj_names = getattr(env0, "_canonical_ids", None)

        game_frames = []
        game_durations = []

        for level_i in range(n_levels):
            env_li = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10)
            _, grid_h, grid_w = env_li.observation_shape

            # Random rollout
            label = f"{name} L{level_i} random"
            print(f"  {label}")
            frames, durs = _render_rollout_frames(
                model, params, apply_fn, json_str, backend_render,
                level_i, n_objs, grid_h, grid_w, max_H, max_W,
                n_steps_per_game, label=label, obj_names=obj_names,
                **cond_kwargs,
            )
            game_frames.extend(frames)
            game_durations.extend(durs)

            # Search rollout(s)
            backend_search = CppPuzzleScriptBackend()
            backend_search.load_from_json(json_str)
            for algo in search_algos:
                try:
                    backend_search.load_level("", level_i)
                    result = backend_search.run_search(
                        algo, game_text="", level_i=level_i,
                        n_steps=search_n_steps, timeout_ms=search_timeout_ms,
                    )
                    if not result.actions:
                        continue
                    label = f"{name} L{level_i} {algo} ({'win' if result.solved else 'no win'})"
                    print(f"  {label} ({len(result.actions)} steps)")
                    frames, durs = _render_rollout_frames(
                        model, params, apply_fn, json_str, backend_render,
                        level_i, n_objs, grid_h, grid_w, max_H, max_W,
                        n_steps=len(result.actions), actions=result.actions,
                        label=label, obj_names=obj_names,
                        **cond_kwargs,
                    )
                    game_frames.extend(frames)
                    game_durations.extend(durs)
                except Exception as e:
                    print(f"    {algo} L{level_i} failed: {e}")

        if game_frames:
            # Pad frames for this game to uniform size
            max_fh = max(f.shape[0] for f in game_frames)
            max_fw = max(f.shape[1] for f in game_frames)
            padded_frames = []
            for f in game_frames:
                pf = np.zeros((max_fh, max_fw, 3), dtype=np.uint8)
                pf[:f.shape[0], :f.shape[1]] = f
                padded_frames.append(pf)

            safe_name = name.replace(" ", "_")
            gif_path = os.path.join(save_dir, f"rollout_{safe_name}{tag}.gif")
            imageio.mimsave(gif_path, padded_frames, duration=game_durations, loop=0)
            print(f"Saved {name} rollout GIF ({len(padded_frames)} frames) to {gif_path}")
            if wandb.run is not None:
                wandb.log({f"eval/rollout_{safe_name}": wandb.Video(gif_path, fps=5, format="gif")})


def _render_obs_with_sprite_kernels(
    obs: np.ndarray,
    sprite_kernels: np.ndarray,
    *,
    soft_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-numpy lookup-table renderer: (n_objs, H, W) + (n_objs, 5, 5, 4) → (5H, 5W, 3).

    - ``obs``: multihot float, used as the per-channel alpha multiplier.
      Typically 0/1 (discrete) but any [0,1] value works for soft rendering.
    - ``sprite_kernels``: (n_objs, 5, 5, 4) uint8 or float. Alpha = channel 4.
    - ``soft_probs`` (optional): (n_objs, H, W) in [0,1]; if provided, overrides
      obs for alpha weighting (lets us render the continuous sigmoid of model
      logits, not the thresholded state).

    Channels are painted in index order — later channels composited over earlier.
    No cross-channel weights: pure per-channel lookup, exactly the user's
    lookup-table formulation.
    """
    n_objs, H, W = obs.shape
    sh = sprite_kernels.shape[1]
    sw = sprite_kernels.shape[2]
    # Normalize kernels to float [0,1]
    if sprite_kernels.dtype == np.uint8:
        kernels_f = sprite_kernels.astype(np.float32) / 255.0
    else:
        kernels_f = np.asarray(sprite_kernels, dtype=np.float32)
    out = np.zeros((H * sh, W * sw, 3), dtype=np.float32)
    weights = obs if soft_probs is None else soft_probs
    for c in range(n_objs):
        rgb = kernels_f[c, ..., :3]           # (5, 5, 3)
        alpha = kernels_f[c, ..., 3:4]        # (5, 5, 1)
        for y in range(H):
            for x in range(W):
                w_c = float(weights[c, y, x])
                if w_c <= 0:
                    continue
                eff_alpha = alpha * w_c        # (5, 5, 1)
                patch = out[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw]
                out[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw] = (
                    patch * (1.0 - eff_alpha) + rgb * eff_alpha
                )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def play_world_model(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    save_dir: str = "nca_wm/logs/play",
):
    """Interactive play: step through the NCA world model with keyboard input.

    Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit.
    Each step renders the predicted state as an image and saves a GIF at the end.
    """
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=10000)
    apply_fn = make_apply_fn(model)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
    frames = []
    os.makedirs(save_dir, exist_ok=True)

    key_to_action = {"w": 0, "a": 1, "s": 2, "d": 3, "x": 4}
    action_names = {0: "up", 1: "left", 2: "down", 3: "right", 4: "action"}
    step_i = 0

    def _render_and_save(pred_state, real_obs, step_i):
        """Render predicted vs real side by side."""
        pred_obs = np.array(pred_state[0] > 0.5, dtype=np.uint8)
        try:
            # Real: step the backend env to match, render from engine
            real_frame = backend.render_frame()
            # Predicted: convert multihot to objects array and render
            pred_objects = _multihot_to_objects(pred_obs)
            pred_frame = backend.render_frame_from_objects(pred_objects, grid_w, grid_h)
            # Side by side: real | predicted
            combined = np.concatenate([real_frame, pred_frame], axis=1)
            frames.append(combined)
            frame_path = os.path.join(save_dir, f"step_{step_i:04d}.png")
            imageio.imwrite(frame_path, combined)
            return frame_path
        except Exception as e:
            print(f"  (render error: {e})")
            return None

    # Keep backend engine in sync for rendering
    backend.load_level("", level_i)

    print("\n--- NCA World Model: Interactive Play ---")
    print("Controls: w=up, a=left, s=down, d=right, x=action, r=restart, q=quit")
    print("Left side = real engine, Right side = NCA prediction\n")

    frame_path = _render_and_save(pred_state, real_obs, step_i)
    if frame_path:
        print(f"  Step {step_i}: initial state -> {frame_path}")

    while True:
        try:
            key = input(f"Step {step_i}> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "q":
            break
        elif key == "r":
            real_obs, _ = env.reset()
            pred_state = jnp.array(real_obs[None], dtype=jnp.float32)
            backend.load_level("", level_i)
            step_i = 0
            frame_path = _render_and_save(pred_state, real_obs, step_i)
            print(f"  Restarted -> {frame_path}")
            continue
        elif key not in key_to_action:
            print(f"  Unknown key '{key}'. Use w/a/s/d/x/r/q.")
            continue

        action = key_to_action[key]
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        # NCA world model step
        logits, _win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        # Real env step (both gym env and backend engine for rendering)
        real_obs, _, done, _, info = env.step(action)
        backend.process_input(action)
        while backend.againing:
            backend.process_input(-1)
        step_i += 1

        # Divergence
        real_f = jnp.array(real_obs[None], dtype=jnp.float32)
        l1 = float(jnp.abs(pred_state - real_f).sum())

        frame_path = _render_and_save(pred_state, real_obs, step_i)
        status = f"  Step {step_i}: {action_names[action]}  L1={l1:.0f}"
        if info.get("won"):
            status += "  WIN!"
        if frame_path:
            status += f"  -> {frame_path}"
        print(status)

        if done:
            print("  Level complete!")

    # Save session as GIF
    if frames:
        gif_path = os.path.join(save_dir, "play_session.gif")
        imageio.mimsave(gif_path, frames, duration=0.3, loop=0)
        print(f"\nSaved play session GIF to {gif_path}")


def render_rollout_comparison(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    level_i: int = 0,
    n_steps: int = 30,
    actions: list[int] | None = None,
    save_path: str = "nca_wm_rollout.gif",
    pad_H: int | None = None,
    pad_W: int | None = None,
):
    """Render a side-by-side GIF: real env (top) vs world model prediction (bottom).

    If `actions` is provided, replays that sequence. Otherwise uses random actions.
    pad_H, pad_W: if set, pad observations spatially to these dims for the model.
    """
    import PIL.Image
    import PIL.ImageDraw
    import PIL.ImageFont

    max_steps = len(actions) if actions else n_steps
    env = CppPuzzleScriptEnv(json_str, level_i=level_i, max_episode_steps=max_steps)
    apply_fn = make_apply_fn(model)
    n_objs, grid_h, grid_w = env.observation_shape

    real_obs, _ = env.reset()
    pred_state = _pad_state_for_model(real_obs, model.n_out, pad_H, pad_W)
    frames = []

    try:
        font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except OSError:
        font = PIL.ImageFont.load_default()

    def _with_banner(img: np.ndarray, text: str) -> np.ndarray:
        w = img.shape[1]
        bh = 18
        banner = PIL.Image.new("RGB", (w, bh), (0, 0, 0))
        PIL.ImageDraw.Draw(banner).text((4, 2), text, fill=(255, 255, 255), font=font)
        return np.concatenate([np.array(banner), img], axis=0)

    pred_won_prob = 0.0
    acts = _enabled_actions(json_str)
    for t in range(max_steps):
        action = actions[t] if actions else int(np.random.choice(acts))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        real_frame = backend.render_frame_from_objects(
            _multihot_to_objects(real_obs), grid_w, grid_h
        )
        pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
        pred_frame = backend.render_frame_from_objects(
            _multihot_to_objects(pred_obs), grid_w, grid_h
        )
        combined = np.concatenate([real_frame, pred_frame], axis=0)
        text = f"t={t} action={action}  pred_win={pred_won_prob:.2f}  (real / NCA)"
        frames.append(_with_banner(combined, text))

        logits, win_logit, _sprite_logits = apply_fn(params, pred_state, a_oh)
        pred_state = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        pred_won_prob = float(jax.nn.sigmoid(win_logit)[0])

        real_obs, _, done, truncated, info = env.step(action)
        real_won = bool(info.get("won", False))
        if done or truncated:
            real_frame = backend.render_frame_from_objects(
                _multihot_to_objects(real_obs), grid_w, grid_h
            )
            pred_obs = _unpad_pred(pred_state, n_objs, grid_h, grid_w)
            pred_frame = backend.render_frame_from_objects(
                _multihot_to_objects(pred_obs), grid_w, grid_h
            )
            combined = np.concatenate([real_frame, pred_frame], axis=0)
            reason = "win" if real_won else ("done" if done else "truncated")
            text = (f"terminal ({reason})  real_win={int(real_won)}  "
                    f"pred_win={pred_won_prob:.2f}")
            frames.append(_with_banner(combined, text))
            break

    if frames:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, frames, duration=0.2, loop=0)
        print(f"Saved rollout GIF ({len(frames)} frames) to {save_path}")


def render_post_training_gifs(
    model: NCAWorldModel,
    params,
    json_str: str,
    backend: CppPuzzleScriptBackend,
    search_data: dict | None,
    level_i: int = 0,
    save_dir: str = ".",
    n_random_steps: int = 50,
):
    """Render comparison GIFs after training: one random rollout + one per search algo."""
    # Random rollout
    print("Rendering random rollout comparison GIF...")
    render_rollout_comparison(
        model, params, json_str, backend, level_i=level_i,
        n_steps=n_random_steps,
        save_path=os.path.join(save_dir, "random_rollout.gif"),
    )

    # Search trajectories
    if search_data is not None and len(search_data["states"]) > 0:
        print("Rendering search trajectory comparison GIF...")
        search_actions = search_data["actions"].tolist()
        render_rollout_comparison(
            model, params, json_str, backend, level_i=level_i,
            actions=search_actions,
            save_path=os.path.join(save_dir, "search_rollout.gif"),
        )
