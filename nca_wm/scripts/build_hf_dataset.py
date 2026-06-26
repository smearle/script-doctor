#!/usr/bin/env python3
"""Stage the FULL PuzzleScript gist corpus for the public Hugging Face dataset.

We publish every (vanilla) game, not just one-per-dedupe-cluster, and tag each row
with its dedupe cluster so deduplication is a one-line filter the user can apply
themselves (`is_dedup_representative == True`). The dedupe is fully reproducible
from the shipped manifest (`dedup_master.json`) + script (`dedup_master.py`).

Reads:
  <master>/dedup_master.json        groups every file into a dedupe cluster
  <master>/provenance/*.jsonl       content hashes per named source collection (one file each)
  <ps_plus_list>                    PuzzleScript-Plus / non-vanilla files to exclude
Emits into <stage>:
  data/puzzlescript_games.jsonl     one row per kept vanilla game (+ content + tags)
  dedup_master.json                 the full dedupe manifest
  dedup_master.py, detect_non_vanilla.py  the scripts that produced the tags
  quarantine_ps_plus_manifest.txt   non-vanilla files excluded

Each row credits the named collection(s) it was reconciled from
(`source_collections`, e.g. "PuzzleScript Gallery", "Pedro's PuzzleScript
Archive", "itch.io"), or "GitHub gist" when it was only seen via a raw gist
scrape. No GitHub handles are emitted (the gist id is the row id).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = Path("/home/jupyter-smearle/puzzlescript-gists")


# Display names for the named source collections, keyed by provenance file stem.
# The `increpare` corpus is just the GitHub gist set, so it is not a distinct
# named archive: those rows fall through to the "GitHub gist" default.
COLLECTION_NAMES = {
    "gallery": "PuzzleScript Gallery",
    "itch": "itch.io",
    "pedro": "Pedro's PuzzleScript Archive",
}


def source_by_hash(master: Path) -> dict[str, set[str]]:
    """Map each content hash to the named collection(s) it was seen in.

    Each provenance/<collection>.jsonl records the hashes reconciled from one
    named, human-curated corpus; we keep the collection name so the published
    rows credit the archive a game came from rather than flattening every
    source into one anonymous category. Provenance files with no entry in
    COLLECTION_NAMES (e.g. the gist-equivalent `increpare` set) are skipped so
    their rows default to "GitHub gist".
    """
    out: dict[str, set[str]] = {}
    for jf in (master / "provenance").glob("*.jsonl"):
        coll = COLLECTION_NAMES.get(jf.stem)
        if coll is None:
            continue
        for ln in jf.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                ch = json.loads(ln).get("content_hash")
            except Exception:
                continue
            if ch:
                out.setdefault(ch, set()).add(coll)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    ap.add_argument("--stage", type=Path, default=Path("/tmp/hf_puzzlescript"))
    ap.add_argument("--ps-plus-list", type=Path, default=Path("/tmp/ps_plus_remove.txt"))
    args = ap.parse_args()
    master, stage = args.master, args.stage
    (stage / "data").mkdir(parents=True, exist_ok=True)

    report = json.loads((master / "dedup_master.json").read_text())
    sources = source_by_hash(master)
    ps_plus = set()
    if args.ps_plus_list.is_file():
        ps_plus = {l.strip() for l in args.ps_plus_list.read_text().splitlines() if l.strip()}

    # file -> dedupe cluster record (every file is a representative or a member).
    rec_by_file: dict[str, dict] = {}
    for c in report["candidates"]:
        rep = c["file"]
        grp = c.get("members") or [rep]
        gk = c.get("group_key", "")
        mech = lev = None
        if gk.startswith("T:"):
            _, mech, lev = gk.split(":", 2)
        meta = {"group": gk, "rep": rep, "n_in_group": len(grp),
                "parse_status": c.get("parse_status"), "n_objs": c.get("n_objs"),
                "n_levels": c.get("n_levels"), "mech": mech, "lev": lev}
        for f in grp:
            rec_by_file[f] = meta

    n = n_rep = n_excluded = 0
    with (stage / "data" / "puzzlescript_games.jsonl").open("w") as out:
        for p in sorted(master.glob("*.txt")):
            fn = p.name
            if fn in ps_plus:
                n_excluded += 1
                continue
            m = rec_by_file.get(fn)
            if m is None:                       # not fingerprinted yet; skip
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            ch = hashlib.sha1(p.read_bytes()).hexdigest()
            is_rep = (fn == m["rep"])
            n_rep += is_rep
            out.write(json.dumps({
                "id": fn[:-4],
                "content": content,
                "source_collections": sorted(sources.get(ch, {"GitHub gist"})),
                "parse_status": m["parse_status"],
                "n_objects": m["n_objs"],
                "n_levels": m["n_levels"],
                "mechanics_hash": m["mech"],
                "levels_hash": m["lev"],
                "dedup_group": m["group"],
                "is_dedup_representative": is_rep,
                "n_in_dedup_group": m["n_in_group"],
            }) + "\n")
            n += 1

    # Ship the manifests + the scripts that produced the dedupe / vanilla tags.
    shutil.copy(master / "dedup_master.json", stage / "dedup_master.json")
    here = Path(__file__).parent
    for s in ("dedup_master.py", "detect_non_vanilla.py"):
        if (here / s).is_file():
            shutil.copy(here / s, stage / s)
    if args.ps_plus_list.is_file():
        shutil.copy(args.ps_plus_list, stage / "quarantine_ps_plus_manifest.txt")

    (stage / "README.md").write_text(_README.format(n=f"{n:,}", n_rep=f"{n_rep:,}"))

    sz = (stage / "data" / "puzzlescript_games.jsonl").stat().st_size / 1e6
    print(f"rows: {n} (dedup representatives: {n_rep}) | excluded PS+: {n_excluded}")
    print(f"hashes with a named collection: {len(sources)} | jsonl: {sz:.1f} MB | stage: {stage}")


_README = """---
pretty_name: PuzzleScript Human-Authored Games (Full Gist Corpus)
license: other
language:
  - en
tags:
  - puzzlescript
  - games
  - code
  - world-models
  - grid-puzzles
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/puzzlescript_games.jsonl
---

# PuzzleScript Human-Authored Games (Full Gist Corpus)

**{n}** human-authored [PuzzleScript](https://www.puzzlescript.net/) games — the
complete source text of each — collected from public GitHub gists.

This is the **full** corpus: every distinct gist is kept, and each row is tagged
with its deduplication cluster so you can reduce to a unique set with a one-line
filter. The deduplication is reproducible from the shipped `dedup_master.json` +
`dedup_master.py`; non-vanilla PuzzleScript-Plus files are excluded (listed in
`quarantine_ps_plus_manifest.txt`, reproducible via `detect_non_vanilla.py`).

Each game defines rewrite rules (mechanics) and one or more levels (initial
states) — useful for **world models** of grid-puzzle dynamics, code generation,
program synthesis, and game design.

## Fields

| field | description |
|---|---|
| `id` | the raw GitHub gist id |
| `content` | full PuzzleScript source |
| `source_collections` | named archive(s) this game was reconciled from (e.g. `PuzzleScript Gallery`, `Pedro's PuzzleScript Archive`, `itch.io`), or `GitHub gist` if seen only via a raw gist scrape |
| `parse_status` | `ok`, `parse_error`, `preprocess_error`, `timeout` |
| `n_objects`, `n_levels` | counts (parsed games) |
| `mechanics_hash`, `levels_hash` | canonical, name/art-invariant fingerprints |
| `dedup_group` | deduplication cluster key |
| `is_dedup_representative` | `True` for one game per cluster |
| `n_in_dedup_group` | cluster size |

## Deduplicating

```python
from datasets import load_dataset
ds = load_dataset("smearle/puzzlescript-gists", split="train")
unique = ds.filter(lambda r: r["is_dedup_representative"])   # {n_rep} distinct games
```

A game is a duplicate only when **both** its mechanics and levels fingerprints
match an already-kept game, so genuine small variants are retained.

## Provenance & license

Collected from public GitHub gists (the canonical PuzzleScript share target) and
several public PuzzleScript archives, credited per-row in `source_collections`:

- the [PuzzleScript Gallery](https://www.puzzlescript.net/Gallery/index.html),
- [Pedro's PuzzleScript Archive](https://pedrosworks.com/) (the PuzzleScript Game Database),
- and itch.io.

No GitHub handles are emitted; the gist `id` is retained as the identifier.
These are third-party, human-authored works redistributed for research; treat
each game as belonging to its original author. License: `other`.
"""


if __name__ == "__main__":
    main()
