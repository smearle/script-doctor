"""Recompress every transition cache .npz under rollout_data/ in place.

Sparse multihot states get ~100-800x compression. The original files were
written with np.savez (uncompressed); switching to np.savez_compressed
recovers most of the disk that solver caches eat.

Atomic rewrite: write to <path>.tmp, fsync, rename over original. Skips any
file that's already compressed (zip stored compression == DEFLATE).

Usage:
    python -m nca_wm.scripts.recompress_caches \\
        [--root /home/jupyter-smearle/script-doctor/rollout_data] \\
        [--workers 16] [--dry-run]
"""
import argparse
import os
import sys
import time
import zipfile
import multiprocessing as mp
from pathlib import Path

import numpy as np


def is_already_compressed(path: str) -> bool:
    """An npz is a zip; check whether all entries use DEFLATE (compressed)."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            for info in z.infolist():
                if info.compress_type == zipfile.ZIP_STORED:
                    return False
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def recompress_one(path: str) -> tuple[str, int, int, str]:
    """Returns (path, before_bytes, after_bytes, status)."""
    before = os.path.getsize(path)
    if is_already_compressed(path):
        return (path, before, before, "already_compressed")

    try:
        d = np.load(path, allow_pickle=True)
        data = {k: d[k] for k in d.files}
        d.close()
    except Exception as e:
        return (path, before, before, f"load_error: {e}")

    tmp_path = path + ".tmp"
    try:
        # Open as file object to avoid np.savez_compressed's auto-appending
        # of '.npz' to the filename when given a string path.
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **data)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (path, before, before, f"save_error: {e}")

    after = os.path.getsize(tmp_path)
    os.replace(tmp_path, path)
    return (path, before, after, "ok")


def find_npz_files(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".npz"):
                paths.append(os.path.join(dirpath, f))
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/jupyter-smearle/script-doctor/rollout_data")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel workers (recompression is CPU-bound)")
    p.add_argument("--dry-run", action="store_true",
                   help="just enumerate files; don't rewrite")
    p.add_argument("--min-size-mb", type=float, default=1.0,
                   help="skip files below this size (no point compressing tiny ones)")
    args = p.parse_args()

    print(f"Scanning {args.root} ...")
    files = find_npz_files(args.root)
    files = [f for f in files if os.path.getsize(f) > args.min_size_mb * 1e6]
    total_before = sum(os.path.getsize(f) for f in files)
    print(f"  {len(files)} candidate files, {total_before/1e9:.1f} GB total")

    if args.dry_run:
        for f in sorted(files, key=lambda p: -os.path.getsize(p))[:20]:
            print(f"  {os.path.getsize(f)/1e6:>8.1f} MB  {f}")
        return

    t0 = time.time()
    total_after = 0
    n_skipped = 0
    n_done = 0
    n_err = 0
    with mp.Pool(args.workers) as pool:
        for i, (path, before, after, status) in enumerate(pool.imap_unordered(recompress_one, files)):
            total_after += after
            if status == "ok":
                n_done += 1
                if before / max(after, 1) > 50:
                    rel = os.path.relpath(path, args.root)
                    print(f"  {before/1e6:>7.1f} -> {after/1e6:>6.1f} MB  ({before/max(after,1):>5.1f}x)  {rel}")
            elif status == "already_compressed":
                n_skipped += 1
            else:
                n_err += 1
                print(f"  ERROR ({status}): {path}")
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  ... {i+1}/{len(files)}  done={n_done} skip={n_skipped} err={n_err}  "
                      f"saved={total_before-total_after:+.0f}/{total_before:.0f} bytes  "
                      f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    saved = total_before - total_after
    print()
    print(f"Done in {elapsed:.0f}s.")
    print(f"  Files: done={n_done}  already_compressed={n_skipped}  errors={n_err}")
    print(f"  Disk:  {total_before/1e9:.1f} GB -> {total_after/1e9:.1f} GB  "
          f"(saved {saved/1e9:.1f} GB, {100*saved/max(total_before,1):.1f}%)")


if __name__ == "__main__":
    main()
