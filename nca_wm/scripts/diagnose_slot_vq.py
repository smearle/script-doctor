#!/usr/bin/env python
"""Diagnose hard-code collapse in a saved slot VQ-VAE checkpoint."""
from __future__ import annotations

import argparse
import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def _joint_counts(indices: np.ndarray):
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim == 2:
        flat = idx.reshape(-1, 1)
    else:
        flat = np.moveaxis(idx, 0, -1).reshape(-1, idx.shape[0])
    counts = {}
    for row in flat:
        key = ",".join(str(int(x)) for x in row)
        counts[key] = counts.get(key, 0) + 1
    return {k: int(v) for k, v in sorted(counts.items())}


def _joint_util(indices: np.ndarray):
    return len(_joint_counts(indices))


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
        f"- vq_quantizer: `{result['vq_quantizer']}`",
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
        result["code_counts"].items(), key=lambda kv: tuple(int(x) for x in kv[0].split(","))
    ):
        lines.append(f"| {code} | {count} |")
    if result.get("stage_code_counts"):
        lines.extend(["", "## Per-Stage Active Codes", ""])
        for i, counts in enumerate(result["stage_code_counts"]):
            lines.extend([
                f"### Stage {i}",
                "",
                "| code | count |",
                "|---:|---:|",
            ])
            for code, count in sorted(counts.items(), key=lambda kv: int(kv[0])):
                lines.append(f"| {code} | {count} |")
            lines.append("")
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
    saved_indices = ckpt.get("vq_indices")
    n_games, n_slots = slots.shape[:2]
    slot_norm = np.linalg.norm(slots, axis=-1)

    saved_quantizer = ckpt_args.get("vq_quantizer")
    if codebook.ndim == 2:
        quantizer = "single"
        codebooks = codebook[None, ...]
    elif codebook.ndim == 3:
        quantizer = saved_quantizer or "residual"
        codebooks = codebook
        if quantizer not in ("residual", "product"):
            quantizer = "residual"
    else:
        raise ValueError(f"Unexpected codebook shape {codebook.shape}")

    def run_quantizer(use_saved: bool):
        residual = slots.copy()
        product_chunks = (
            np.split(slots, codebooks.shape[0], axis=-1)
            if quantizer == "product" else None
        )
        stage_indices = []
        stage_second = []
        stage_nearest_dist = []
        stage_second_dist = []
        stage_margin = []
        stage_entropy = []
        stage_perp = []
        saved = None if saved_indices is None else np.asarray(saved_indices, dtype=np.int64)
        if quantizer == "single" and saved is not None and saved.ndim == 2:
            saved = saved[None, ...]
        for q, cb in enumerate(codebooks):
            stage_input = product_chunks[q] if quantizer == "product" else residual
            flat = stage_input.reshape(-1, stage_input.shape[-1])
            dists = (
                np.square(flat).sum(axis=1, keepdims=True)
                - 2.0 * flat @ cb.T
                + np.square(cb).sum(axis=1, keepdims=True).T
            )
            recomputed = np.argmin(dists, axis=1)
            if use_saved and saved is not None:
                nearest = saved[q].reshape(-1)
            else:
                nearest = recomputed
            nearest_d = dists[np.arange(len(flat)), nearest]
            masked = dists.copy()
            masked[np.arange(len(flat)), nearest] = np.inf
            second = np.argmin(masked, axis=1)
            second_d = masked[np.arange(len(flat)), second]
            entropy, perp = _soft_entropy(dists, entropy_temp)
            idx_2d = nearest.reshape(n_games, n_slots)
            stage_indices.append(idx_2d)
            stage_second.append(second.reshape(n_games, n_slots))
            stage_nearest_dist.append(nearest_d.reshape(n_games, n_slots))
            stage_second_dist.append(second_d.reshape(n_games, n_slots))
            stage_margin.append((second_d - nearest_d).reshape(n_games, n_slots))
            stage_entropy.append(entropy.reshape(n_games, n_slots))
            stage_perp.append(perp)
            if quantizer == "residual":
                residual = residual - cb[nearest].reshape(residual.shape)
        return {
            "indices": np.stack(stage_indices, axis=0),
            "second_idx": np.stack(stage_second, axis=0),
            "nearest_dist": np.stack(stage_nearest_dist, axis=0),
            "second_dist": np.stack(stage_second_dist, axis=0),
            "margin": np.stack(stage_margin, axis=0),
            "soft_entropy": np.stack(stage_entropy, axis=0),
            "stage_soft_perplexities": [float(x) for x in stage_perp],
            "soft_perplexity": float(np.mean(stage_perp)),
        }

    saved_diag = run_quantizer(use_saved=True)
    recomputed_diag = run_quantizer(use_saved=False)
    indices_q = saved_diag["indices"]
    if quantizer == "single":
        indices = indices_q[0]
        codebook_size = int(codebooks.shape[1])
        counts = np.bincount(indices.reshape(-1), minlength=codebook_size)
        code_counts = {
            str(int(i)): int(c)
            for i, c in enumerate(counts)
            if c > 0
        }
        hard_util = int(np.count_nonzero(counts))
        recomputed_hard_util = int(np.unique(recomputed_diag["indices"][0]).size)
    else:
        indices = np.moveaxis(indices_q, 0, -1)
        codebook_size = int(codebooks.shape[1])
        code_counts = _joint_counts(indices_q)
        hard_util = _joint_util(indices_q)
        recomputed_hard_util = _joint_util(recomputed_diag["indices"])
    nearest_dist = saved_diag["nearest_dist"].mean(axis=0)
    second_dist = saved_diag["second_dist"].mean(axis=0)
    second_idx = (
        saved_diag["second_idx"][0]
        if quantizer == "single"
        else np.moveaxis(saved_diag["second_idx"], 0, -1)
    )
    margin = saved_diag["margin"].mean(axis=0)
    soft_entropy = saved_diag["soft_entropy"].mean(axis=0)
    soft_perplexity = saved_diag["soft_perplexity"]
    stage_code_counts = []
    for q in range(codebooks.shape[0]):
        counts_q = np.bincount(indices_q[q].reshape(-1), minlength=codebook_size)
        stage_code_counts.append({
            str(int(i)): int(c)
            for i, c in enumerate(counts_q)
            if c > 0
        })

    game_names = ckpt.get("game_names", [f"game_{i}" for i in range(n_games)])
    games = []
    for i, name in enumerate(game_names):
        slot_rows = []
        for j in range(n_slots):
            slot_rows.append({
                "slot": int(j),
                "code": (
                    int(indices[i, j]) if quantizer == "single"
                    else [int(x) for x in indices[i, j]]
                ),
                "second_code": (
                    int(second_idx[i, j]) if quantizer == "single"
                    else [int(x) for x in second_idx[i, j]]
                ),
                "nearest_distance": float(nearest_dist[i, j]),
                "second_distance": float(second_dist[i, j]),
                "distance_margin": float(margin[i, j]),
                "slot_norm": float(slot_norm[i, j]),
                "soft_entropy": float(soft_entropy[i, j]),
            })
        games.append({
            "name": name,
            "active_codes": (
                [int(x) for x in sorted(np.unique(indices[i]))]
                if quantizer == "single"
                else sorted({",".join(str(int(x)) for x in row)
                             for row in indices[i].reshape(-1, indices.shape[-1])})
            ),
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
        "vq_quantizer": quantizer,
        "n_codebooks": int(codebooks.shape[0]),
        "codebook_size": codebook_size,
        "hard_util": int(hard_util),
        "recomputed_hard_util": int(recomputed_hard_util),
        "uses_saved_vq_indices": bool(saved_indices is not None),
        "code_counts": code_counts,
        "stage_code_counts": stage_code_counts,
        "stage_hard_utils": [int(len(c)) for c in stage_code_counts],
        "stage_mean_margins": [float(x) for x in saved_diag["margin"].mean(axis=(1, 2))],
        "stage_soft_perplexities": saved_diag["stage_soft_perplexities"],
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
