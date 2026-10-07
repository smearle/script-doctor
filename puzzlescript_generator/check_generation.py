"""Audit the sampling path of a trained generator.

1. Cache consistency: logits from prefill + one-token cached decoding must match a full
   teacher-forced forward pass over the same tokens (bf16 noise only).
2. Self-consistency of samples: the model's own NLL of its T=1.0 samples (scored by a
   full forward pass) should be close to the sampling entropy it reported, i.e. samples
   come from the distribution the model defines.

    python check_generation.py --data DIR --run DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-docs", type=int, default=8)
    args = ap.parse_args()
    prep = json.loads((args.data / "prep_report.json").read_text())
    ck = torch.load(args.run / "best.pt", map_location="cuda", weights_only=False)
    model = GPT(GPTConfig(**ck["config"])).cuda().eval()
    model.load_state_dict(ck["model"])
    arr = np.fromfile(args.data / "test.bin", dtype=np.uint16)
    off = np.load(args.data / "test_offsets.npy")
    out = {"cache_vs_full": [], "sample_selfconsistency": []}
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for d in range(args.n_docs):
            toks = torch.tensor(arr[off[d]:min(off[d + 1], off[d] + 700)].astype(np.int64), device="cuda")[None]
            T = toks.shape[1]
            full = model(toks).float()
            p0 = min(100, T - 1)
            caches = model.new_caches(1, T, torch.bfloat16, "cuda")
            inc = [model(toks[:, :p0], caches, 0)[:, -1].float()]
            for t in range(p0, T - 1):
                inc.append(model(toks[:, t:t + 1], caches, t)[:, -1].float())
            inc = torch.stack(inc, 1)  # predictions for positions p0..T-1
            ref = full[:, p0 - 1:T - 1]
            lp_inc = F.log_softmax(inc, -1)
            lp_ref = F.log_softmax(ref, -1)
            tgt = toks[:, p0:T]
            out["cache_vs_full"].append({
                "doc": d, "T": T,
                "max_abs_logprob_diff_on_targets": float((lp_inc.gather(-1, tgt[..., None]) -
                                                          lp_ref.gather(-1, tgt[..., None])).abs().max()),
                "nll_full": float(-lp_ref.gather(-1, tgt[..., None]).mean()),
                "nll_cached": float(-lp_inc.gather(-1, tgt[..., None]).mean()),
                "argmax_agreement": float((inc.argmax(-1) == ref.argmax(-1)).float().mean())})
        prompt = torch.full((8, 1), prep["bos"], dtype=torch.long, device="cuda")
    outs, done = model.generate(prompt, 2048, prep["eos"], temperature=1.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for o in outs:
            if len(o) < 2:
                continue
            seq = torch.tensor([[prep["bos"]] + o], device="cuda")
            lp = F.log_softmax(model(seq).float(), -1)
            nll = -lp[0, :-1].gather(-1, seq[0, 1:, None]).squeeze(-1)
            ent = -(lp[0, :-1].exp() * lp[0, :-1]).sum(-1)
            out["sample_selfconsistency"].append({"n_tokens": len(o), "nll_per_token": float(nll.mean()),
                                                  "entropy_per_token": float(ent.mean())})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
