#!/usr/bin/env python
"""Diagnose hard-code collapse in a saved slot VQ-VAE checkpoint."""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle

import numpy as np


def _apply_slot_pre_norm_np(slots: np.ndarray, mode: str, eps: float = 1e-6):
    if mode == "none":
        return slots
    if mode == "layernorm":
        mean = slots.mean(axis=-1, keepdims=True)
        var = np.square(slots - mean).mean(axis=-1, keepdims=True)
        return (slots - mean) / np.sqrt(var + eps)
    if mode == "l2":
        norm = np.linalg.norm(slots, axis=-1, keepdims=True)
        return slots / np.maximum(norm, eps) * np.sqrt(float(slots.shape[-1]))
    raise ValueError(f"Unknown slot_pre_norm={mode}")


def _soft_entropy(dists: np.ndarray, temp: float):
    temp = max(float(temp), 1e-6)
    logits = -dists / temp
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)
    avg_probs = probs.mean(axis=0)
    avg_entropy = -(
        avg_probs * np.log(np.clip(avg_probs, 1e-12, 1.0))
    ).sum()
    return entropy, float(np.exp(avg_entropy))


def _load_checkpoint(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("latent_params") is None:
        raise ValueError(f"{path} does not contain latent_params")
    params = ckpt["latent_params"].get("params", ckpt["latent_params"])
    if "codebook" not in params:
        raise ValueError(f"{path} does not contain a VQ codebook")
    return ckpt, np.asarray(params["codebook"])


def _write_markdown(path: str, result: dict):
    lines = [
        "# VQ Diagnostic",
        "",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- slot_pre_norm: `{result['slot_pre_norm']}`",
        f"- n_games: {result['n_games']}",
        f"- n_slots_per_game: {result['n_slots_per_game']}",
        f"- codebook_size: {result['codebook_size']}",
        f"- hard_util: {result['hard_util']}",
        f"- recomputed_hard_util: {result['recomputed_hard_util']}",
        f"- uses_saved_vq_indices: {result['uses_saved_vq_indices']}",
        f"- soft_perplexity_from_slots: {result['soft_perplexity_from_slots']:.3f}",
        f"- mean_nearest_distance: {result['mean_nearest_distance']:.6f}",
        f"- mean_distance_margin: {result['mean_distance_margin']:.6f}",
        "",
        "## Active Codes",
        "",
        "| code | count |",
        "|---:|---:|",
    ]
    for code, count in sorted(
        result["code_counts"].items(), key=lambda kv: int(kv[0])
    ):
        lines.append(f"| {code} | {count} |")
    lines.extend([
        "",
        "## Per-Game Summary",
        "",
        "| game | active codes | mean nearest dist | mean margin | mean slot norm | mean soft entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for game in result["games"]:
        active = ",".join(str(c) for c in game["active_codes"])
        lines.append(
            f"| `{game['name']}` | {active} | "
            f"{game['mean_nearest_distance']:.6f} | "
            f"{game['mean_distance_margin']:.6f} | "
            f"{game['mean_slot_norm']:.6f} | "
            f"{game['mean_soft_entropy']:.6f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to slot_ae.pkl")
    p.add_argument("--out_dir", default=None,
                   help="Defaults to the checkpoint directory.")
    p.add_argument("--slot_pre_norm", choices=["auto", "none", "layernorm", "l2"],
                   default="auto")
    p.add_argument("--entropy_temp", type=float, default=None,
                   help="Defaults to checkpoint args.vq_entropy_temp or 1.0.")
    args = p.parse_args()

    ckpt, codebook = _load_checkpoint(args.checkpoint)
    ckpt_args = ckpt.get("args", {})
    slot_pre_norm = (
        ckpt_args.get("slot_pre_norm", "none")
        if args.slot_pre_norm == "auto"
        else args.slot_pre_norm
    )
    entropy_temp = (
        ckpt_args.get("vq_entropy_temp", 1.0)
        if args.entropy_temp is None
        else args.entropy_temp
    )
    raw_slots = np.asarray(ckpt["raw_slots_all"], dtype=np.float32)
    slots = _apply_slot_pre_norm_np(raw_slots, slot_pre_norm).astype(np.float32)
    flat = slots.reshape(-1, slots.shape[-1])
    dists = (
        np.square(flat).sum(axis=1, keepdims=True)
        - 2.0 * flat @ codebook.T
        + np.square(codebook).sum(axis=1, keepdims=True).T
    )
    recomputed_idx = np.argmin(dists, axis=1)
    saved_indices = ckpt.get("vq_indices")
    if saved_indices is not None:
        nearest_idx = np.asarray(saved_indices, dtype=np.int64).reshape(-1)
    else:
        nearest_idx = recomputed_idx
    nearest_dist = dists[np.arange(len(flat)), nearest_idx]
    masked = dists.copy()
    masked[np.arange(len(flat)), nearest_idx] = np.inf
    second_idx = np.argmin(masked, axis=1)
    second_dist = masked[np.arange(len(flat)), second_idx]
    margin = second_dist - nearest_dist
    soft_entropy, soft_perplexity = _soft_entropy(dists, entropy_temp)

    n_games, n_slots = slots.shape[:2]
    indices = nearest_idx.reshape(n_games, n_slots)
    nearest_dist = nearest_dist.reshape(n_games, n_slots)
    second_dist = second_dist.reshape(n_games, n_slots)
    second_idx = second_idx.reshape(n_games, n_slots)
    margin = margin.reshape(n_games, n_slots)
    soft_entropy = soft_entropy.reshape(n_games, n_slots)
    slot_norm = np.linalg.norm(slots, axis=-1)
    counts = np.bincount(indices.reshape(-1), minlength=len(codebook))
    code_counts = {
        str(int(i)): int(c)
        for i, c in enumerate(counts)
        if c > 0
    }

    game_names = ckpt.get("game_names", [f"game_{i}" for i in range(n_games)])
    games = []
    for i, name in enumerate(game_names):
        slot_rows = []
        for j in range(n_slots):
            slot_rows.append({
                "slot": int(j),
                "code": int(indices[i, j]),
                "second_code": int(second_idx[i, j]),
                "nearest_distance": float(nearest_dist[i, j]),
                "second_distance": float(second_dist[i, j]),
                "distance_margin": float(margin[i, j]),
                "slot_norm": float(slot_norm[i, j]),
                "soft_entropy": float(soft_entropy[i, j]),
            })
        games.append({
            "name": name,
            "active_codes": [int(x) for x in sorted(np.unique(indices[i]))],
            "mean_nearest_distance": float(nearest_dist[i].mean()),
            "mean_distance_margin": float(margin[i].mean()),
            "mean_slot_norm": float(slot_norm[i].mean()),
            "mean_soft_entropy": float(soft_entropy[i].mean()),
            "slots": slot_rows,
        })

    result = {
        "checkpoint": args.checkpoint,
        "slot_pre_norm": slot_pre_norm,
        "entropy_temp": float(entropy_temp),
        "n_games": int(n_games),
        "n_slots_per_game": int(n_slots),
        "codebook_size": int(len(codebook)),
        "hard_util": int(np.count_nonzero(counts)),
        "recomputed_hard_util": int(np.unique(recomputed_idx).size),
        "uses_saved_vq_indices": bool(saved_indices is not None),
        "code_counts": code_counts,
        "soft_perplexity_from_slots": soft_perplexity,
        "mean_nearest_distance": float(nearest_dist.mean()),
        "mean_distance_margin": float(margin.mean()),
        "games": games,
    }

    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "vq_diagnostic.json")
    md_path = os.path.join(out_dir, "vq_diagnostic.md")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    _write_markdown(md_path, result)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"hard_util={result['hard_util']} "
        f"soft_perplexity={result['soft_perplexity_from_slots']:.1f} "
        f"mean_margin={result['mean_distance_margin']:.6f}"
    )


if __name__ == "__main__":
    main()
