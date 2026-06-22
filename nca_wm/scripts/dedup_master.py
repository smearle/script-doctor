#!/usr/bin/env python3
"""Incremental, layered dedupe for the puzzlescript-gists master corpus.

Unlike `nca_wm/scripts/dedup_games.py` (which dedups the curated repo
corpora using pre-cached Lark trees + `games_metadata.json`), this script
is tailored for the flat, gist-hash-named master directory at
`../puzzlescript-gists` where neither cached trees nor metadata exist. It
parses every game fresh and caches the result so re-running on newly added
games is cheap.

GOAL
----
Keep any game that has a chance of providing *novel training material* for a
world model — i.e. anything that differs in **mechanics** or **levels**. A
game is dropped as a duplicate ONLY when both its mechanics fingerprint AND
its levels fingerprint match an already-kept game.

LAYERED PIPELINE (cheap -> expensive; each stage only refines survivors)
-----------------------------------------------------------------------
0. exact      : sha1 of raw bytes.
1. normalized : whitespace-collapsed + casefolded sha1.
2. lite       : run the engine's own `preprocess_ps` (handles comment
                stripping incl. nesting / unmatched ')' / leading /*..*/,
                SOUNDS removal, message stripping, whitespace + header
                normalization), then hash the mechanics sections and the
                LEVELS section SEPARATELY. Name/art-SENSITIVE; no parser.
3. tokens     : Lark parse -> StripPuzzleScript -> GenPSTree ->
                `tokenize_game(encode_sprites=False)`. Name/color/sprite
                -INVARIANT canonical fingerprint, split into a mechanics
                token-hash and a levels token-hash (via `include_levels`).
                Functional prelude directives that the tree/tokenizer drop
                (realtime_interval, again_interval, norepeat_action,
                throttle_movement) are folded into the mechanics hash from
                the preprocessed text so they still count as a difference.

GROUPING KEY (the dedupe identity of a game), in priority order:
    parsed ok           -> ("T", mech_hash, lev_hash)   # canonical tokens
    preprocessed only   -> ("L", mech_lite, lev_lite)   # lite sections
    unparseable+unpreproc -> ("R", content_hash)        # exact content
Games are never merged across namespaces, so an unparseable game is never
silently collapsed into a parsed one.

INCREMENTAL CACHE
-----------------
`--cache` is a JSONL keyed by content_hash (sha1 of raw bytes). Each line
holds the full fingerprint set for one unique content. On rerun we sha1 each
file (cheap), skip anything already in the cache, parse only the new content,
and append. Same content across runs -> same key -> zero recompute. Because
the key is content (not filename), renamed/re-downloaded gists are free too.

USAGE
-----
    .venv/bin/python3 -m nca_wm.scripts.dedup_master \
        --master-dir ../puzzlescript-gists \
        --workers 24
    # rerun later after adding files -- only new content is parsed:
    .venv/bin/python3 -m nca_wm.scripts.dedup_master --master-dir ../puzzlescript-gists
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Prelude directives that change dynamics / the WM's inputs but are NOT
# captured by the parsed tree or `tokenize_game`. We fold these into the
# mechanics fingerprint straight from the preprocessed text. (noaction,
# require_player_movement, run_rules_on_level_start ARE in the token stream
# already; included here too for a single consistent signature.)
FUNCTIONAL_PRELUDE_KEYWORDS = (
    "realtime_interval", "again_interval", "norepeat_action",
    "throttle_movement", "noaction", "require_player_movement",
    "run_rules_on_level_start",
)

# Canonical section headers emitted by preprocess_ps, in order.
_SECTION_HEADERS = ("OBJECTS", "LEGEND", "SOUNDS", "COLLISIONLAYERS",
                    "RULES", "WINCONDITIONS", "LEVELS")


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


def _norm_hash(text: str) -> str:
    return _sha(re.sub(r"\s+", " ", text).strip().lower())


def _functional_prelude_sig(preprocessed: str) -> str:
    """Stable signature of functional prelude directives.

    Scans the prelude region (everything before the first section header)
    of the *preprocessed* text and records `keyword value` for each
    functional directive, sorted for order-independence.
    """
    # Prelude = text before the first canonical section header line.
    cut = len(preprocessed)
    for h in _SECTION_HEADERS:
        m = re.search(rf"^{h}\n", preprocessed, flags=re.MULTILINE)
        if m:
            cut = min(cut, m.start())
    prelude = preprocessed[:cut]
    found = []
    for ln in prelude.splitlines():
        s = ln.strip()
        if not s:
            continue
        kw = s.split()[0].lower()
        if kw in FUNCTIONAL_PRELUDE_KEYWORDS:
            found.append(re.sub(r"\s+", " ", s.lower()))
    return "|".join(sorted(found))


def _split_preprocessed_sections(preprocessed: str) -> dict[str, str]:
    """Split preprocess_ps output into {header: body}. Prelude under '' key."""
    pat = r"^(" + "|".join(_SECTION_HEADERS) + r")\n"
    parts = re.split(pat, preprocessed, flags=re.MULTILINE)
    out = {"": parts[0]}
    for i in range(1, len(parts), 2):
        out[parts[i]] = parts[i + 1] if i + 1 < len(parts) else ""
    return out


def _lite_hashes(preprocessed: str) -> tuple[str, str]:
    """(mechanics_lite, levels_lite) from preprocessed text. Name-sensitive."""
    secs = _split_preprocessed_sections(preprocessed)
    mech_parts = [_functional_prelude_sig(preprocessed)]
    for h in ("OBJECTS", "LEGEND", "COLLISIONLAYERS", "RULES", "WINCONDITIONS"):
        body = secs.get(h, "")
        mech_parts.append(re.sub(r"[ \t]+", " ", body).strip())
    lev = re.sub(r"[ \t]+$", "", secs.get("LEVELS", ""), flags=re.MULTILINE).strip()
    return _sha("\x1e".join(mech_parts)), _sha(lev)


class _Timeout:
    """SIGALRM timeout (worker processes run tasks on their main thread)."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def __enter__(self):
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._raise)
            signal.alarm(self.seconds)

    def __exit__(self, *a):
        if self.seconds and hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("parse timeout")


# ---- worker globals (per-process, lazily built) -------------------------
_W: dict = {}


def _worker_init():
    """Import the parser stack once per worker process."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    from puzzlescript_jax.utils import init_ps_lark_parser
    from puzzlescript_jax.preprocessing import preprocess_ps, StripPuzzleScript
    from puzzlescript_jax.gen_tree import GenPSTree
    from nca_wm.tokenize_game import tokenize_game
    _W["parser"] = init_ps_lark_parser()
    _W["preprocess_ps"] = preprocess_ps
    _W["Strip"] = StripPuzzleScript
    _W["Gen"] = GenPSTree
    _W["tokenize_game"] = tokenize_game


def _fingerprint(content_hash: str, raw: str, parse_timeout: int) -> dict:
    """Compute the full layered fingerprint for one unique content string."""
    if not _W:
        _worker_init()
    rec: dict = {
        "content_hash": content_hash,
        "norm_hash": _norm_hash(raw),
        "parse_status": "ok",
        "mech_lite": None, "lev_lite": None,
        "mech_hash": None, "lev_hash": None,
        "n_levels": None, "n_objs": None,
    }
    # Stage 2: engine preprocessing + lite section hashes.
    try:
        pre = _W["preprocess_ps"](raw)
        rec["mech_lite"], rec["lev_lite"] = _lite_hashes(pre)
        func_sig = _functional_prelude_sig(pre)
    except Exception as e:
        rec["parse_status"] = "preprocess_error:" + type(e).__name__
        return rec
    # Stage 3: parse -> canonical token fingerprint.
    try:
        with _Timeout(parse_timeout):
            lark_tree = _W["parser"].parse(pre)
        min_tree = _W["Strip"]().transform(lark_tree)
        tree = _W["Gen"]().transform(min_tree)
    except TimeoutError:
        rec["parse_status"] = "timeout"
        return rec
    except Exception as e:
        rec["parse_status"] = "parse_error:" + type(e).__name__
        return rec
    try:
        objs = tree.objects
        canonical_ids = list(objs.keys()) if isinstance(objs, dict) \
            else [o.name for o in objs]
        tok = _W["tokenize_game"]
        mech_tokens = tok(tree, canonical_ids, encode_sprites=False,
                          include_levels=False)
        full_tokens = tok(tree, canonical_ids, encode_sprites=False,
                          include_levels=True)
        # mech_tokens == [..mech body.., EOS]; full == [..mech body.., ..levels.., EOS]
        mech_body = mech_tokens[:-1]
        lev_body = full_tokens[len(mech_tokens) - 1:-1]
        rec["mech_hash"] = _sha(",".join(map(str, mech_body)) + "#" + func_sig)
        rec["lev_hash"] = _sha(",".join(map(str, lev_body)))
        rec["n_objs"] = len(canonical_ids)
        rec["n_levels"] = len(getattr(tree, "levels", []) or [])
    except Exception as e:
        rec["parse_status"] = "tokenize_error:" + type(e).__name__
    return rec


def _group_key(rec: dict) -> str:
    if rec["parse_status"] == "ok" and rec["mech_hash"]:
        return f"T:{rec['mech_hash']}:{rec['lev_hash']}"
    if rec["mech_lite"]:
        return f"L:{rec['mech_lite']}:{rec['lev_lite']}"
    return f"R:{rec['content_hash']}"


# ---- driver -------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.is_file():
        with path.open() as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    cache[r["content_hash"]] = r
                except Exception:
                    continue
    return cache


def _load_provenance(master: Path) -> dict[str, str]:
    """content_hash -> human filename, best-effort, for nicer canonical names."""
    out: dict[str, str] = {}
    pdir = master / "provenance"
    if not pdir.is_dir():
        return out
    for jf in pdir.glob("*.jsonl"):
        try:
            for ln in jf.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                ch = r.get("content_hash")
                fn = r.get("filename")
                if ch and fn and ch not in out:
                    out[ch] = fn
        except Exception:
            continue
    return out


_HEXNAME = re.compile(r"^[0-9a-f]{20,}$")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-dir", default=str(REPO_ROOT.parent / "puzzlescript-gists"))
    ap.add_argument("--cache", default=None,
                    help="JSONL fingerprint cache (default <master>/dedup_cache.jsonl)")
    ap.add_argument("--out", default=None,
                    help="Output report (default <master>/dedup_master.json)")
    ap.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 4) - 2))
    ap.add_argument("--parse-timeout", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="Debug: only N files.")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="Append new cache records to disk every N completions.")
    args = ap.parse_args()

    master = Path(args.master_dir).resolve()
    cache_path = Path(args.cache) if args.cache else master / "dedup_cache.jsonl"
    out_path = Path(args.out) if args.out else master / "dedup_master.json"

    files = sorted(f for f in os.listdir(master) if f.endswith(".txt"))
    if args.limit:
        files = files[:args.limit]
    print(f"master: {master}\nfiles: {len(files)}  cache: {cache_path}")

    # Map files -> content_hash (cheap), find which content is uncached.
    cache = _load_cache(cache_path)
    print(f"cache: {len(cache)} fingerprints loaded")
    file_to_hash: dict[str, str] = {}
    raw_by_hash: dict[str, str] = {}
    need: dict[str, str] = {}  # content_hash -> raw (only uncached)
    for fn in files:
        try:
            b = (master / fn).read_bytes()
        except Exception:
            continue
        ch = hashlib.sha1(b).hexdigest()
        file_to_hash[fn] = ch
        if ch not in cache and ch not in need:
            need[ch] = b.decode("utf-8", "replace")
    print(f"unique content: {len(set(file_to_hash.values()))}  "
          f"new to fingerprint: {len(need)}")

    # Fingerprint new content in parallel, appending to cache as we go.
    t0 = time.time()
    n_done = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if need:
        buf: list[dict] = []

        def _flush():
            if not buf:
                return
            with cache_path.open("a") as f:
                for r in buf:
                    f.write(json.dumps(r) + "\n")
            buf.clear()

        items = list(need.items())
        if args.workers <= 1:
            _worker_init()
            for ch, raw in items:
                r = _fingerprint(ch, raw, args.parse_timeout)
                cache[ch] = r
                buf.append(r)
                n_done += 1
                if n_done % args.flush_every == 0:
                    _flush()
                    print(f"  [{n_done}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_worker_init) as pool:
                futs = {pool.submit(_fingerprint, ch, raw, args.parse_timeout): ch
                        for ch, raw in items}
                for fut in as_completed(futs):
                    ch = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {"content_hash": ch, "norm_hash": None,
                             "parse_status": "worker_crash:" + type(e).__name__,
                             "mech_lite": None, "lev_lite": None,
                             "mech_hash": None, "lev_hash": None,
                             "n_levels": None, "n_objs": None}
                    cache[ch] = r
                    buf.append(r)
                    n_done += 1
                    if n_done % args.flush_every == 0:
                        _flush()
                        print(f"  [{n_done}/{len(items)}] {time.time()-t0:.0f}s", flush=True)
        _flush()
    print(f"fingerprinting done: {n_done} new in {time.time()-t0:.0f}s")

    # Build dedupe groups over ALL current files.
    prov = _load_provenance(master)
    groups: dict[str, list[str]] = {}          # group_key -> [filenames]
    status_counts: dict[str, int] = {}
    for fn in files:
        ch = file_to_hash.get(fn)
        rec = cache.get(ch)
        if rec is None:
            continue
        st = rec.get("parse_status", "?").split(":")[0]
        status_counts[st] = status_counts.get(st, 0) + 1
        groups.setdefault(_group_key(rec), []).append(fn)

    def _canon(members: list[str]) -> str:
        # Prefer a human-readable provenance name, then non-hex filename,
        # then shortest stem, then lex.
        def k(fn):
            ch = file_to_hash[fn]
            prov_name = prov.get(ch)
            stem = fn[:-4]
            return (
                0 if prov_name else 1,
                0 if not _HEXNAME.match(stem) else 1,
                len(stem),
                stem,
            )
        return min(members, key=k)

    candidates = []
    for gk, members in groups.items():
        rep = _canon(members)
        rec = cache[file_to_hash[rep]]
        prov_name = prov.get(file_to_hash[rep])
        candidates.append({
            "file": rep,
            "label": (prov_name[:-4] if prov_name and prov_name.endswith(".txt")
                      else prov_name) or rep[:-4],
            "content_hash": rec["content_hash"],
            "group_key": gk,
            "parse_status": rec["parse_status"],
            "n_objs": rec.get("n_objs"),
            "n_levels": rec.get("n_levels"),
            "n_dups": len(members) - 1,
            "members": sorted(members) if len(members) > 1 else [],
        })
    candidates.sort(key=lambda c: c["label"].lower())

    # Novelty diagnostics among PARSED games: how many share mechanics but
    # differ in levels (level-only novelty) vs share levels but differ in
    # mechanics (mechanics-only novelty).
    parsed = [cache[file_to_hash[c["file"]]] for c in candidates
              if c["parse_status"] == "ok" and cache[file_to_hash[c["file"]]]["mech_hash"]]
    mech_groups: dict[str, int] = {}
    lev_groups: dict[str, int] = {}
    for r in parsed:
        mech_groups[r["mech_hash"]] = mech_groups.get(r["mech_hash"], 0) + 1
        lev_groups[r["lev_hash"]] = lev_groups.get(r["lev_hash"], 0) + 1

    out = {
        "master_dir": str(master),
        "n_files": len(files),
        "n_unique_content": len(set(file_to_hash.values())),
        "n_after_dedup": len(candidates),
        "removed_as_dups": len(files) - len(candidates),
        "parse_status_counts": status_counts,
        "n_parsed_candidates": len(parsed),
        "n_distinct_mechanics": len(mech_groups),
        "n_distinct_levelsets": len(lev_groups),
        "n_mechanics_shared_by_2plus": sum(1 for v in mech_groups.values() if v > 1),
        "candidates": candidates,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"  {len(files)} files -> {len(candidates)} kept "
          f"(removed {len(files) - len(candidates)} dups)")
    print(f"  parse status: {status_counts}")
    print(f"  parsed candidates: {len(parsed)}  "
          f"distinct mechanics: {len(mech_groups)}  "
          f"distinct level-sets: {len(lev_groups)}")


if __name__ == "__main__":
    main()
