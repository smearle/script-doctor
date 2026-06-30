"""Does a FULL pre-bonk state (all channels) that BREAKS in mario_breakable also
occur in mario (where it STAYS)? Run over ALL transitions (no subsampling).

Packed state bytes are an exact identity key (packing is deterministic), so we
match full states without unpacking -- only mario_breakable needs unpacking to
detect which UP transitions actually break.
"""
import glob

import numpy as np

from nca_wm.state_ops import _unpack_states

STEP, UP = 6, 0
CHUNK = 100_000


def cache(name):
    f = glob.glob(f"rollout_data/{name}/level_0/bfs_transitions_v5_200000_-1_capall.npz")[0]
    return np.load(f, allow_pickle=True)


def main():
    # mario_breakable: distinct packed pre-states of UP transitions whose Step breaks
    db = cache("mario_breakable")
    Spb, Nspb, Ab = db["states"], db["next_states"], np.asarray(db["actions"], np.int64)
    Wb = int(db["W"])
    BRK = set()
    n_up_b = 0
    for i in range(0, len(Spb), CHUNK):
        sl = slice(i, i + CHUNK)
        up = Ab[sl] == UP
        S = _unpack_states(Spb[sl], Wb)
        Ns = _unpack_states(Nspb[sl], Wb)
        broke = ((S[:, STEP] > 0) & ~(Ns[:, STEP] > 0)).reshape(len(S), -1).any(1)
        for j in np.where(up & broke)[0]:
            BRK.add(Spb[i + j].tobytes())
        n_up_b += int(up.sum())
    print(f"mario_breakable: {len(Spb):,} transitions, {n_up_b:,} UP, "
          f"{len(BRK):,} distinct UP-break pre-states")

    # mario: distinct packed pre-states of ALL UP transitions (no unpack needed)
    dm = cache("mario")
    Spm, Am = dm["states"], np.asarray(dm["actions"], np.int64)
    MUP = set()
    for i in range(0, len(Spm), CHUNK):
        up = Am[i:i + CHUNK] == UP
        for j in np.where(up)[0]:
            MUP.add(Spm[i + j].tobytes())
    print(f"mario:          {len(Spm):,} transitions, {len(MUP):,} distinct UP pre-states")

    amb = BRK & MUP
    print(f"\n==> AMBIGUOUS full-states (identical all-channel state that BREAKS in "
          f"mario_breakable AND occurs in mario under UP, where Step never breaks): {len(amb):,}")
    print(f"    = {len(amb)/max(len(BRK),1)*100:.1f}% of breakable break-states are genuinely ambiguous")


if __name__ == "__main__":
    main()
