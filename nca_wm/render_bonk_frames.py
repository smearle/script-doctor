"""Render the pre-bonk-jump frames so we can spot-check whether the break states
in mario_breakable are really distinct from the (Player+Step)-identical STAY
states in mario -- or whether they only differ in some incidental channel.

For each (Player+Step)-matched pair (breakable BREAKS vs mario STAYS): a composite
view of each + every channel that DIFFERS between them. Plus a channel-diff
summary and a montage of all break pre-states.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nca_wm.state_ops import _unpack_states

STEP, PLAYER, UP = 6, 7, 0
OUT = "/tmp/bonk_frames"
os.makedirs(OUT, exist_ok=True)


def load(name, n=20000, seed=0):
    f = glob.glob(f"rollout_data/{name}/level_0/bfs_transitions_v5_200000_-1_capall.npz")[0]
    d = np.load(f, allow_pickle=True)
    W = int(d["W"])
    Sp, Nsp, A = d["states"], d["next_states"], np.asarray(d["actions"], np.int64)
    idx = np.sort(np.random.RandomState(seed).choice(len(Sp), min(n, len(Sp)), replace=False))
    return (_unpack_states(Sp[idx], W).astype(np.float32),
            _unpack_states(Nsp[idx], W).astype(np.float32), A[idx])


def composite(S):
    C, H, W = S.shape
    img = np.ones((H, W, 3)) * 0.97
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for c in range(C):
        if c in (PLAYER, STEP):
            continue
        m = S[c] > 0.5
        if m.any():
            img[m] = cmap[c % 20][:3] * 0.55 + 0.25
    img[S[STEP] > 0.5] = [0.15, 0.75, 0.15]
    img[S[PLAYER] > 0.5] = [0.9, 0.1, 0.1]
    return img


def main():
    Sb, Nsb, Ab = load("mario_breakable")
    Sm, Nsm, Am = load("mario")
    C = Sb.shape[1]
    broke = (Sb[:, STEP] > 0.5) & ~(Nsb[:, STEP] > 0.5)
    brk_rows = np.where((Ab == UP) & broke.reshape(len(Sb), -1).any(1))[0]

    PS = [PLAYER, STEP]
    mkeys = {}
    for ti in np.where(Am == UP)[0]:
        mkeys.setdefault((Sm[ti, PS] > 0.5).tobytes(), []).append(ti)

    pairs = []
    for ti in brk_rows:
        k = (Sb[ti, PS] > 0.5).tobytes()
        if k not in mkeys:
            continue
        bh, bw = np.argwhere(broke[ti])[0]
        for mti in mkeys[k]:
            if Nsm[mti, STEP, bh, bw] > 0.5:
                pairs.append((ti, mti, (int(bh), int(bw))))
                break
    print(f"{len(brk_rows)} break-jumps; {len(pairs)} (Player+Step)-matched-and-STAY pairs")

    diffcount = np.zeros(C, int)
    for ti, mti, _ in pairs:
        d = ((Sb[ti] > 0.5) != (Sm[mti] > 0.5)).reshape(C, -1).any(1)
        diffcount += d
    print("channels that differ within matched pairs (ch: #pairs / "
          f"{len(pairs)}):")
    for c in range(C):
        if diffcount[c]:
            print(f"   ch{c:2d}: {int(diffcount[c])}")

    for i, (ti, mti, cell) in enumerate(pairs):
        diffch = [c for c in range(C) if ((Sb[ti, c] > 0.5) != (Sm[mti, c] > 0.5)).any()]
        ncols = 1 + len(diffch)
        fig, axs = plt.subplots(2, ncols, figsize=(2.1 * ncols, 4.4), squeeze=False)
        for r, (S, lab) in enumerate([(Sb[ti], "mario_breakable  (BREAKS)"),
                                       (Sm[mti], "mario  (STAYS)")]):
            axs[r, 0].imshow(composite(S)); axs[r, 0].axis("off")
            axs[r, 0].set_title(lab, fontsize=8)
            axs[r, 0].plot(cell[1], cell[0], "yx", ms=9, mew=2)
            for j, c in enumerate(diffch):
                axs[r, j+1].imshow(S[c], cmap="gray", vmin=0, vmax=1)
                axs[r, j+1].set_title(f"ch{c}", fontsize=7); axs[r, j+1].axis("off")
        fig.suptitle(f"pair {i}: IDENTICAL Player+Step;  differing channels = {diffch}",
                     fontsize=9)
        fig.tight_layout(); fig.savefig(f"{OUT}/pair_{i:02d}.png", dpi=95); plt.close(fig)

    n = len(brk_rows); cols = 8; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for k, ti in enumerate(brk_rows):
        ax = axs.flat[k]; ax.imshow(composite(Sb[ti]))
        ax.set_title(f"#{ti}", fontsize=6)
    fig.suptitle("All mario_breakable break-jump pre-states (red=Mario, green=Step)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/all_break_states.png", dpi=95); plt.close(fig)
    print(f"wrote {len(pairs)} pair PNGs + montage -> {OUT}")


if __name__ == "__main__":
    main()
