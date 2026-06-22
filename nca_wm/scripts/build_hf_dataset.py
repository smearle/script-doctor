#!/usr/bin/env python3
"""Stage the deduped PuzzleScript corpus for a public Hugging Face dataset.

Reads dedup_master.json + the master dir, emits a staging folder with:
  data/puzzlescript_games.jsonl  one row per KEPT deduped game (+ content)
  dedup_master.json              full dedupe manifest (all clusters)
  quarantine_ps_plus_manifest.txt  PS+ files removed before dedupe
  README.md                      dataset card (written separately)

Per-row `provenance` is anonymized to "curated_collection" (content hash found
in any provenance/*.jsonl) vs "gist_scrape" — no individual handles are emitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MASTER = Path("/home/jupyter-smearle/puzzlescript-gists")
STAGE = Path("/tmp/hf_puzzlescript")


def curated_hashes() -> set[str]:
    out: set[str] = set()
    pdir = MASTER / "provenance"
    for jf in pdir.glob("*.jsonl"):
        for ln in jf.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                ch = json.loads(ln).get("content_hash")
                if ch:
                    out.add(ch)
            except Exception:
                pass
    return out


def main() -> None:
    (STAGE / "data").mkdir(parents=True, exist_ok=True)
    report = json.loads((MASTER / "dedup_master.json").read_text())
    curated = curated_hashes()

    n = 0
    with (STAGE / "data" / "puzzlescript_games.jsonl").open("w") as out:
        for c in report["candidates"]:
            fn = c["file"]
            p = MASTER / fn
            if not p.is_file():
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            ch = hashlib.sha1(p.read_bytes()).hexdigest()
            gk = c.get("group_key", "")
            mech = lev = None
            if gk.startswith("T:"):
                _, mech, lev = gk.split(":", 2)
            row = {
                "id": fn[:-4],
                "content": content,
                "provenance": "curated_collection" if ch in curated else "gist_scrape",
                "parse_status": c.get("parse_status"),
                "n_objects": c.get("n_objs"),
                "n_levels": c.get("n_levels"),
                "mechanics_hash": mech,
                "levels_hash": lev,
                "n_duplicates": c.get("n_dups", 0),
                "duplicate_ids": [m[:-4] for m in c.get("members", [])],
            }
            out.write(json.dumps(row) + "\n")
            n += 1

    # Copy the full dedupe manifest and the PS+ quarantine manifest.
    (STAGE / "dedup_master.json").write_text((MASTER / "dedup_master.json").read_text())
    q = MASTER / "_quarantine_ps_plus" / "_MANIFEST.txt"
    if q.is_file():
        (STAGE / "quarantine_ps_plus_manifest.txt").write_text(q.read_text())

    n_curated = sum(1 for _ in [])  # placeholder
    print(f"rows written: {n}")
    print(f"curated content hashes: {len(curated)}")
    sz = (STAGE / 'data' / 'puzzlescript_games.jsonl').stat().st_size / 1e6
    print(f"jsonl size: {sz:.1f} MB")


if __name__ == "__main__":
    main()
