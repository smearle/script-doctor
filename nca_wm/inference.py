"""Model-forward / inference helpers for the NCA world model.

Thin, training-loop-independent wrappers around ``model.apply`` shared by the
training loop, evaluation, and rendering: parameter-tree unwrapping, jitted
apply with automatic zero-history padding, the forward-only eval-metrics fn,
and inference-time state padding/unpadding. Depends only on the jax stack, so
eval / rendering can import it without pulling in ``train.py``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _wm_p(params):
    """Extract WM params from a possibly-joint params tree.

    When joint token-decoder training is enabled, ``params`` is shaped
    ``{"wm": <wm_params>, "dec": <decoder_params>}``. All eval / render code
    only needs the WM half — this helper unwraps both shapes uniformly.
    """
    if isinstance(params, dict) and set(params.keys()) >= {"wm", "dec"}:
        return params["wm"]
    return params


def make_apply_fn(model):
    """Jitted model.apply that always returns a (logits, win, sprite) 3-tuple.

    The model can return additional trailing aux tensors (slots, vq_aux,
    halt_aux) under various flags, but the eval / rollout / rendering
    code paths only need the first three. This wrapper hides the
    branching so those callers stay simple.

    When ``history > 0`` and a caller does not supply ``hist_states``, this
    injects a zero (fully masked) history of the right width so the model's
    embed layer receives the expected channel count. Zero history is the
    honest default for one-shot / context-free calls (sprite kernels,
    single-frame viz); rollout paths that have a real trajectory pass their
    own ``hist_states``/``hist_actions`` and bypass this.
    """
    _jit = jax.jit(model.apply)
    history = int(getattr(model, "history", 0))
    def apply_fn(*args, **kwargs):
        if history > 0 and kwargs.get("hist_states") is None:
            state = args[1]  # (B, C, H, W) — state is always the 2nd arg
            kwargs["hist_states"] = jnp.zeros(
                (state.shape[0], history) + tuple(state.shape[1:]), state.dtype)
            kwargs["hist_actions"] = jnp.zeros(
                (state.shape[0], history), jnp.int32)
        out = _jit(*args, **kwargs)
        return out[0], out[1], out[2]
    return apply_fn


def make_eval_forward(model, conditional: bool):
    """Forward-only (no grad) eval for per-game diagnostics during training."""
    def _metrics(logits, states, next_states, spatial_mask=None):
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        correct = (preds == next_states).astype(jnp.float32)
        mask = spatial_mask.astype(jnp.float32)
        mask_sum = jnp.maximum(mask.sum(), 1.0)
        acc = (correct * mask).sum() / mask_sum
        changed = (states != next_states) & (spatial_mask > 0)
        bce = optax.sigmoid_binary_cross_entropy(logits, next_states)
        bce = (bce * mask).sum() / mask_sum
        n_changed = changed.sum()
        changed_correct = ((preds == next_states) & changed).sum()
        change_acc = jnp.where(n_changed > 0, changed_correct / n_changed, 1.0)
        return bce, acc, change_acc

    if conditional:
        @jax.jit
        def _eval_forward(wm_params, states, action_onehots, next_states,
                          game_tokens, game_masks, spatial_mask=None,
                          hist_states=None, hist_actions=None):
            logits, _, _sprite_logits = model.apply(
                wm_params, states, action_onehots, game_tokens, game_masks,
                hist_states=hist_states, hist_actions=hist_actions)
            return _metrics(logits, states, next_states, spatial_mask)

        def eval_forward(params, states, action_onehots, next_states,
                         game_tokens, game_masks, spatial_mask=None,
                         hist_states=None, hist_actions=None):
            return _eval_forward(_wm_p(params), states, action_onehots,
                                 next_states, game_tokens, game_masks, spatial_mask,
                                 hist_states, hist_actions)
    else:
        @jax.jit
        def _eval_forward(wm_params, states, action_onehots, next_states,
                          spatial_mask=None, hist_states=None, hist_actions=None):
            logits, _, _sprite_logits = model.apply(
                wm_params, states, action_onehots,
                hist_states=hist_states, hist_actions=hist_actions)
            return _metrics(logits, states, next_states, spatial_mask)

        def eval_forward(params, states, action_onehots, next_states, spatial_mask=None,
                         hist_states=None, hist_actions=None):
            return _eval_forward(_wm_p(params), states, action_onehots, next_states,
                                 spatial_mask, hist_states, hist_actions)
    return eval_forward


def _pad_state_for_model(obs: np.ndarray, target_C: int,
                         target_H: int | None = None,
                         target_W: int | None = None) -> jnp.ndarray:
    """Pad a (C, H, W) observation to (1, target_C, eff_H, eff_W) for the model.

    Level is top-left-aligned to match training-time bucket padding (where
    ``s_buf[i, :C_g, :H_g, :W_g] = ...``). Centering would put real cells in
    different (row,col) positions than the model saw during training, which is
    catastrophic because the model is conditioned on the input-derived padding
    mask.

    eff_H/W = max(target, obs) so OOD-larger eval levels (e.g. authored width
    19 vs training max 16) are run at their own size — the NCA is convolutional
    and handles arbitrary (H,W).
    """
    C, H, W = obs.shape
    tH = max(target_H or H, H)
    tW = max(target_W or W, W)
    if C == target_C and H == tH and W == tW:
        return jnp.array(obs[None], dtype=jnp.float32)
    padded = np.zeros((1, target_C, tH, tW), dtype=np.float32)
    padded[0, :C, :H, :W] = obs
    return jnp.array(padded)


def _unpad_pred(pred_state: jnp.ndarray, n_objs: int,
                H: int | None = None, W: int | None = None) -> np.ndarray:
    """Extract (n_objs, H, W) uint8 from padded (1, model_n_out, pad_H, pad_W) prediction,
    cropping from the top-left to match training-time top-left-aligned padding."""
    cropped = pred_state[0, :n_objs]
    pad_H = int(cropped.shape[1])
    pad_W = int(cropped.shape[2])
    if H is not None and H < pad_H:
        cropped = cropped[:, :H, :]
    if W is not None and W < pad_W:
        cropped = cropped[:, :, :W]
    return np.array(cropped > 0.5, dtype=np.uint8)
