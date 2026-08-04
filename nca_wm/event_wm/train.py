"""Train the event world model on recurrent_data trajectory batches.

Same data pipeline as ``train_recurrent`` (RecurrentNCA) and
``mario_transformer_baseline`` (AttnBeliefModel) — the third column for the
MARIO2 architecture comparison, and the harness for the hidden-state
(>1-step-dependency) games.

CLI::

    python -m nca_wm.event_wm.train --games mario mario_breakable \
        --k 8 --run-name mario2_event_wm [--wandb-project sd-event-wm]

Loss: legality-masked token cross-entropy, masked to valid steps. Nothing
else. Comparable headline metric: ``nll_bits_step`` — bits per (valid) env
step of the next-frame likelihood.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from .data import (
    GameData,
    build_trajectory_batch,
    load_dataset_from_caches,
    to_event_batch,
    audit_batch,
)
from .model import (
    EWMConfig,
    EventWorldModel,
    dec_positions,
    legality_mask,
    masked_logits,
)
from .tokenizer import Geom

LN2 = float(np.log(2.0))


def make_loss_fn(net: EventWorldModel, cfg: EWMConfig):
    def loss_fn(params, batch):
        logits = net.apply(
            params, batch["obs"], batch["actions"], batch["dec_in"],
            batch["dec_step"], batch["dec_pos"], batch["step_valid"])
        mask = legality_mask(cfg, batch["obs"], batch["dec_in"],
                             batch["dec_step"])
        logp = jax.nn.log_softmax(masked_logits(logits, mask), axis=-1)
        tgt = batch["dec_tgt"]
        pad_id = cfg.vocab - 1
        valid = (tgt != pad_id)
        tgt_safe = jnp.where(valid, tgt, 0)
        nll = -jnp.take_along_axis(logp, tgt_safe[..., None], axis=-1)[..., 0]
        nll = nll * valid
        n_valid = valid.sum()
        loss = nll.sum() / jnp.maximum(n_valid, 1)
        n_steps = batch["step_valid"].sum()
        pred = jnp.argmax(masked_logits(logits, mask), axis=-1)
        correct = (pred == tgt) & valid
        eventful = valid & (tgt != cfg.vocab - 3)          # != EOF
        # target legality audit (canonical targets must always be legal)
        tgt_legal = jnp.take_along_axis(
            mask, tgt_safe[..., None], axis=-1)[..., 0]
        metrics = {
            "nll_bits_tok": loss / LN2,
            "nll_bits_step": nll.sum() / jnp.maximum(n_steps, 1) / LN2,
            "acc": correct.sum() / jnp.maximum(n_valid, 1),
            "acc_eventful": (correct & eventful).sum()
            / jnp.maximum(eventful.sum(), 1),
            "target_legal": (tgt_legal * valid).sum()
            / jnp.maximum(n_valid, 1),
            "tokens_per_step": n_valid / jnp.maximum(n_steps, 1),
        }
        return loss, metrics

    return loss_fn


def make_train_step(net: EventWorldModel, cfg: EWMConfig):
    loss_fn = make_loss_fn(net, cfg)

    @jax.jit
    def train_step(state: TrainState, batch):
        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params, batch)
        metrics["grad_norm"] = optax.global_norm(grads)
        return state.apply_gradients(grads=grads), metrics

    return train_step, jax.jit(loss_fn)


def make_optimizer(cfg: EWMConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, cfg.warmup_steps,
        max(cfg.total_steps, cfg.warmup_steps + 1), cfg.lr * 0.05)
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(schedule, weight_decay=cfg.weight_decay),
    )


def init_params(net: EventWorldModel, cfg: EWMConfig, rng) -> Any:
    b, l = 2, 8
    return net.init(
        rng,
        jnp.zeros((b, cfg.t_len, cfg.c_chan, cfg.h, cfg.w), jnp.float32),
        jnp.zeros((b, cfg.t_len), jnp.int32),
        jnp.zeros((b, l), jnp.int32),
        jnp.zeros((b, l), jnp.int32),
        jnp.zeros((b, l), jnp.int32),
        jnp.ones((b, cfg.t_len), bool),
    )


def save_checkpoint(path: str, cfg: EWMConfig, params: Any, step: int,
                    opt_state: Any = None) -> None:
    os.makedirs(path, exist_ok=True)
    meta = cfg.asdict()
    meta["step"] = step
    with open(os.path.join(path, "ewm.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(path, "params.msgpack"), "wb") as f:
        f.write(serialization.to_bytes(params))
    if opt_state is not None:
        with open(os.path.join(path, "opt.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(opt_state))


def load_checkpoint(path: str):
    cfg = EWMConfig.load(os.path.join(path, "ewm.json"))
    with open(os.path.join(path, "ewm.json")) as f:
        step = int(json.load(f).get("step", 0))
    net = EventWorldModel(cfg)
    template = init_params(net, cfg, jax.random.PRNGKey(0))
    with open(os.path.join(path, "params.msgpack"), "rb") as f:
        params = serialization.from_bytes(template, f.read())
    return cfg, net, params, step


class BatchSource:
    """Samples per-game trajectory batches and event-tokenizes them."""

    def __init__(self, games: list[GameData], geom: Geom, cfg: EWMConfig,
                 seed: int, split: str = "train"):
        self.games = games
        self.geom = geom
        self.cfg = cfg
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.overflow = 0

    def next(self):
        cfg = self.cfg
        gd = self.games[self.rng.integers(len(self.games))]
        rows = gd.train_rows if self.split == "train" else gd.val_rows
        target = self.rng.choice(rows, size=cfg.batch_size,
                                 replace=len(rows) < cfg.batch_size)
        s, a, t, v = build_trajectory_batch(
            gd, target, cfg.t_len - 1, self.rng, cfg.c_chan, cfg.h, cfg.w)
        batch, ovf = to_event_batch(self.geom, s, a, t, v, cfg.max_len)
        self.overflow += ovf
        batch["dec_pos"] = dec_positions(batch["dec_step"])
        return batch


def _prefetch(src: BatchSource, q: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        batch = src.next()
        try:
            q.put(batch, timeout=1.0)
        except queue.Full:
            continue


def train(cfg: EWMConfig, game_names: list[str],
          max_transitions: int = 200_000, val_frac: float = 0.05,
          resume_from: str = "", audit_batches: int = 2) -> str:
    run_dir = cfg.run_dir or os.path.join("nca_wm", "logs", cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[ewm] run dir: {run_dir} | devices: {jax.devices()}")

    dataset, game_infos = load_dataset_from_caches(
        game_names, max_transitions, val_frac,
        ancestor_closed=True, max_grid_dim=32, seed=cfg.seed)
    games = [GameData(g, dataset, info) for g, info in enumerate(game_infos)]
    max_c = max(gd.n_objs for gd in games)
    max_h = max(gd.H for gd in games)
    max_w = max(gd.W for gd in games)
    cfg.c_chan, cfg.h, cfg.w = max_c, max_h, max_w
    geom = Geom(max_c, max_h, max_w)
    cfg.save(os.path.join(run_dir, "config.json"))
    print(f"[ewm] games={[gd.name for gd in games]} geom=({max_c},{max_h},"
          f"{max_w}) vocab={geom.vocab} L={cfg.t_len}")

    net = EventWorldModel(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    start_step = 0
    if resume_from:
        _, _, params, start_step = load_checkpoint(resume_from)
        print(f"[ewm] resumed from {resume_from} (step {start_step})")
    else:
        params = init_params(net, cfg, rng)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[ewm] parameters: {n_params / 1e6:.2f}M")

    tx = make_optimizer(cfg)
    state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    train_step, eval_fn = make_train_step(net, cfg)

    # one BatchSource per prefetch thread (numpy Generators are not
    # thread-safe); metrics read the first one.
    train_srcs = [BatchSource(games, geom, cfg, cfg.seed + 1 + i, "train")
                  for i in range(2)]
    train_src = train_srcs[0]
    val_src = BatchSource(games, geom, cfg, cfg.seed + 101, "val")

    # P0 bijection audit on fresh batches before any training.
    for _ in range(audit_batches):
        gd = games[train_src.rng.integers(len(games))]
        target = train_src.rng.choice(gd.train_rows, size=cfg.batch_size)
        s, a, t, v = build_trajectory_batch(
            gd, target, cfg.t_len - 1, train_src.rng, max_c, max_h, max_w)
        b, _ = to_event_batch(geom, s, a, t, v, cfg.max_len)
        audit_batch(geom, b, t)
    print(f"[ewm] bijection audit passed ({audit_batches} batches)")

    wb = None
    if cfg.wandb_project:
        try:
            import wandb

            wb = wandb.init(project=cfg.wandb_project,
                            entity=cfg.wandb_entity or None,
                            name=cfg.run_name, dir=run_dir,
                            config=cfg.asdict() | {
                                "n_params": n_params,
                                "model": "event_wm",
                                "games": game_names},
                            resume="allow")
        except Exception as exc:  # noqa: BLE001
            print(f"[ewm] wandb disabled ({exc})")

    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    threads = [threading.Thread(target=_prefetch, args=(src, q, stop),
                                daemon=True) for src in train_srcs]
    for th in threads:
        th.start()

    t_start = time.time()
    best_val = float("inf")
    try:
        for step in range(start_step, cfg.total_steps):
            if cfg.max_hours and \
                    time.time() - t_start > cfg.max_hours * 3600:
                print(f"[ewm] wall-clock cap at step {step}")
                break
            batch = q.get()
            state, m = train_step(state, {k: jnp.asarray(v)
                                          for k, v in batch.items()})
            if step % 50 == 0:
                m = jax.device_get(m)
                log = {f"train/{k}": float(v) for k, v in m.items()}
                log["step"] = step
                if float(m["target_legal"]) < 0.9999:
                    print(f"[ewm] WARNING: illegal target tokens "
                          f"(legal={float(m['target_legal']):.6f})")
                if step % cfg.eval_every == 0:
                    vb = val_src.next()
                    _, vm = eval_fn(state.params,
                                    {k: jnp.asarray(v)
                                     for k, v in vb.items()})
                    vm = jax.device_get(vm)
                    log.update({f"val/{k}": float(v) for k, v in vm.items()})
                    if float(vm["nll_bits_step"]) < best_val:
                        best_val = float(vm["nll_bits_step"])
                        save_checkpoint(os.path.join(run_dir, "ckpt_best"),
                                        cfg, state.params, step)
                    elapsed = time.time() - t_start
                    print(f"[ewm] step {step}/{cfg.total_steps} "
                          f"train {float(m['nll_bits_step']):.4f} "
                          f"val {float(vm['nll_bits_step']):.4f} bits/step | "
                          f"acc_ev {float(vm['acc_eventful']):.4f} | "
                          f"{step - start_step and (step - start_step) / elapsed:.2f} it/s",
                          flush=True)
                if wb:
                    try:
                        wb.log(log, step=step)
                    except Exception:  # noqa: BLE001
                        wb = None
            if cfg.ckpt_every and step and step % cfg.ckpt_every == 0:
                save_checkpoint(os.path.join(run_dir, f"ckpt_{step:07d}"),
                                cfg, state.params, step, state.opt_state)
    finally:
        stop.set()
        save_checkpoint(os.path.join(run_dir, "ckpt_final"), cfg,
                        state.params, int(state.step), state.opt_state)
        print(f"[ewm] final checkpoint: {run_dir}/ckpt_final "
              f"(overflow-dropped steps: {train_src.overflow})")
        if wb:
            try:
                wb.finish()
            except Exception:  # noqa: BLE001
                pass
    return run_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--k", type=int, default=8,
                   help="history depth; trajectory length L = k+1")
    p.add_argument("--max-transitions", type=int, default=200_000)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--resume-from", default="")
    p.add_argument("--smoke", action="store_true")
    for f in EWMConfig.__dataclass_fields__.values():
        if f.name in ("c_chan", "h", "w", "t_len", "num_actions"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.name == "conv_channels":
            p.add_argument(flag, type=str, default=None)
        elif isinstance(f.default, bool):
            p.add_argument(flag, type=lambda s: s.lower() == "true",
                           default=None)
        else:
            p.add_argument(flag, type=type(f.default), default=None)
    a = p.parse_args(argv)
    overrides = {
        f: getattr(a, f) for f in EWMConfig.__dataclass_fields__
        if hasattr(a, f) and getattr(a, f) is not None
    }
    if "conv_channels" in overrides:
        overrides["conv_channels"] = tuple(
            int(x) for x in overrides["conv_channels"].split(","))
    if a.smoke:
        overrides = dict(
            d_enc=64, enc_layers=2, enc_heads=2, d_dec=64, dec_layers=2,
            dec_heads=2, batch_size=4, total_steps=30, warmup_steps=5,
            eval_every=10, run_name="ewm_smoke",
        ) | overrides
    cfg = EWMConfig(t_len=a.k + 1, **overrides)
    train(cfg, a.games, a.max_transitions, a.val_frac, a.resume_from)


if __name__ == "__main__":
    main()
