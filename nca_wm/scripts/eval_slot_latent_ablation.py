#!/usr/bin/env python
"""Evaluate how much a slot-token decoder relies on saved latent slots."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nca_wm.token_decoder import SlotTokenDecoder, decoder_loss, shift_right


ABLATIONS = (
    "original",
    "zero_slots",
    "mean_slots",
    "shuffled_across_games",
    "shuffled_within_game",
)


def _checkpoint_path(path: str):
    if os.path.isdir(path):
        return os.path.join(path, "slot_ae.pkl")
    return path


def _load_json(path: str):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _make_ablated_slots(slots: np.ndarray, mode: str, rng: np.random.Generator):
    if mode == "original":
        return slots
    if mode == "zero_slots":
        return np.zeros_like(slots)
    if mode == "mean_slots":
        mean_slot = slots.reshape(-1, slots.shape[-1]).mean(axis=0)
        return np.broadcast_to(mean_slot[None, None, :], slots.shape).copy()
    if mode == "shuffled_across_games":
        order = rng.permutation(slots.shape[0])
        return slots[order].copy()
    if mode == "shuffled_within_game":
        out = slots.copy()
        for i in range(out.shape[0]):
            out[i] = out[i, rng.permutation(out.shape[1])]
        return out
    raise ValueError(f"Unknown ablation mode {mode}")


def _per_game_accuracy(preds: np.ndarray, tokens: np.ndarray, mask: np.ndarray,
                       game_names: list[str]):
    rows = []
    for i, name in enumerate(game_names):
        n = int(mask[i].sum())
        if n == 0:
            continue
        correct = int(((preds[i] == tokens[i]) & mask[i]).sum())
        rows.append({
            "name": name,
            "acc": float(correct / n),
            "n_tokens": n,
        })
    return rows


def _evaluate_one(path: str, seed: int):
    ckpt_path = _checkpoint_path(path)
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    run_dir = os.path.dirname(ckpt_path)
    summary = _load_json(os.path.join(run_dir, "summary.json"))
    ckpt_args = ckpt.get("args", {})
    tokens = np.asarray(ckpt["tokens"], dtype=np.int32)
    mask = np.asarray(ckpt["mask"], dtype=np.bool_)
    slots = np.asarray(ckpt["slots_all"], dtype=np.float32)
    game_names = ckpt.get("game_names", [f"game_{i}" for i in range(tokens.shape[0])])

    decoder = SlotTokenDecoder(
        vocab_size=int(ckpt_args["vocab_size"]),
        max_seq_len=int(tokens.shape[1]),
        d_model=int(ckpt_args.get("dec_d_model", 128)),
        n_layers=int(ckpt_args.get("dec_n_layers", 4)),
        n_heads=int(ckpt_args.get("dec_n_heads", 4)),
        d_slot=int(ckpt_args.get("d_slot", slots.shape[-1])),
    )
    dec_params = ckpt["dec_params"]
    tokens_j = jnp.asarray(tokens)
    mask_j = jnp.asarray(mask)
    inputs_j = shift_right(tokens_j, bos_id=0)

    @jax.jit
    def eval_slots(slots_j):
        logits = decoder.apply(dec_params, inputs_j, slots_j, deterministic=True)
        loss, acc = decoder_loss(logits, tokens_j, mask_j)
        preds = jnp.argmax(logits, axis=-1)
        return loss, acc, preds

    rng = np.random.default_rng(seed)
    results = []
    original_acc = None
    original_per_game = None
    for mode in ABLATIONS:
        ablated = _make_ablated_slots(slots, mode, rng)
        loss, acc, preds = eval_slots(jnp.asarray(ablated))
        preds_np = np.asarray(preds)
        per_game = _per_game_accuracy(preds_np, tokens, mask, game_names)
        row = {
            "mode": mode,
            "loss": float(loss),
            "token_acc": float(acc),
            "mean_per_game_acc": float(np.mean([r["acc"] for r in per_game])),
            "min_per_game_acc": float(min(r["acc"] for r in per_game)),
            "worst_game": min(per_game, key=lambda r: r["acc"])["name"],
            "per_game": per_game,
        }
        if mode == "original":
            original_acc = row["token_acc"]
            original_per_game = {r["name"]: r["acc"] for r in per_game}
            row["token_acc_drop"] = 0.0
            row["mean_per_game_acc_drop"] = 0.0
        else:
            row["token_acc_drop"] = float(original_acc - row["token_acc"])
            drops = [
                original_per_game[r["name"]] - r["acc"]
                for r in per_game
                if r["name"] in original_per_game
            ]
            row["mean_per_game_acc_drop"] = float(np.mean(drops))
        results.append(row)

    model_name = summary.get("latent_model", ckpt_args.get("latent_model", os.path.basename(run_dir)))
    out = {
        "checkpoint": ckpt_path,
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "model": model_name,
        "seed": seed,
        "results": results,
    }
    return out


def _write_one_markdown(path: str, result: dict):
    lines = [
        f"# Latent Ablation: {result['run_name']} ({result['model']})",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        "",
        "| ablation | token acc | token drop | mean game acc | mean game drop | min game acc | worst game |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["results"]:
        lines.append(
            f"| `{row['mode']}` | {row['token_acc']:.6f} | "
            f"{row['token_acc_drop']:.6f} | {row['mean_per_game_acc']:.6f} | "
            f"{row['mean_per_game_acc_drop']:.6f} | {row['min_per_game_acc']:.6f} | "
            f"`{row['worst_game']}` |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary_markdown(path: str, all_results: list[dict]):
    lines = [
        "# Latent Ablation Summary",
        "",
        "| run | model | ablation | token acc | token drop | mean game acc | mean game drop | min game acc |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in all_results:
        for row in result["results"]:
            lines.append(
                f"| `{result['run_name']}` | `{result['model']}` | `{row['mode']}` | "
                f"{row['token_acc']:.6f} | {row['token_acc_drop']:.6f} | "
                f"{row['mean_per_game_acc']:.6f} | "
                f"{row['mean_per_game_acc_drop']:.6f} | "
                f"{row['min_per_game_acc']:.6f} |"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+",
                   help="Run dirs or slot_ae.pkl paths.")
    p.add_argument("--out_dir", default=None,
                   help="Aggregate output dir. Defaults to common parent.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    all_results = []
    for path in args.checkpoints:
        result = _evaluate_one(path, args.seed)
        all_results.append(result)
        run_dir = result["run_dir"]
        json_path = os.path.join(run_dir, "latent_ablation.json")
        md_path = os.path.join(run_dir, "latent_ablation.md")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        _write_one_markdown(md_path, result)
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")

    ckpt_dirs = [os.path.dirname(_checkpoint_path(p)) for p in args.checkpoints]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.commonpath(ckpt_dirs)
        if len(ckpt_dirs) > 1:
            out_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) else out_dir
    os.makedirs(out_dir, exist_ok=True)
    summary_json = os.path.join(out_dir, "latent_ablation_summary.json")
    summary_md = os.path.join(out_dir, "latent_ablation_summary.md")
    with open(summary_json, "w") as f:
        json.dump(all_results, f, indent=2)
    _write_summary_markdown(summary_md, all_results)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
