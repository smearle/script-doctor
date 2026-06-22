#!/usr/bin/env python3
"""Flag non-vanilla (PuzzleScript Plus / malformed) games via a whitelist lexer.

Vanilla PuzzleScript's section headers and prelude directives are CLOSED sets
(defined in puzzlescript_jax/syntax.lark). Anything outside them is either
PuzzleScript Plus or a typo/malformed file. This detector needs no parser: it
splits off the prelude region (text before the first vanilla section header)
and scans line-initial tokens, plus looks for non-vanilla section headers.

Classification (high → low confidence that it's PS+ specifically):
  ps_plus_section : has a TAGS or MAPPINGS section header (PS+-only sections;
                    impossible in vanilla -> 0 false positives).
  nonvanilla_prelude : a prelude directive outside the 22 vanilla keywords.
                    Captures PS+ directives (case_sensitive, tween_length, ...)
                    but also typos -> see the per-keyword histogram to judge.
  vanilla : neither signal.

Usage:
    .venv/bin/python3 -m nca_wm.scripts.detect_non_vanilla \
        --master-dir ../puzzlescript-gists [--out non_vanilla.json] [--emit-removal-list FILE]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

# Sourced verbatim from puzzlescript_jax/syntax.lark (keep in sync).
VANILLA_SECTIONS = {"OBJECTS", "LEGEND", "SOUNDS", "COLLISIONLAYERS",
                    "RULES", "WINCONDITIONS", "LEVELS"}
VANILLA_PRELUDE = {
    "title", "author", "homepage", "color_palette", "again_interval",
    "background_color", "debug", "flickscreen", "key_repeat_interval",
    "noaction", "norepeat_action", "noundo", "norestart", "realtime_interval",
    "require_player_movement", "run_rules_on_level_start", "scanline",
    "text_color", "throttle_movement", "verbose_logging", "youtube",
    "zoomscreen",
}
PS_PLUS_SECTIONS = {"TAGS", "MAPPINGS"}

# Positive whitelist of PuzzleScript Plus prelude directives (PS+'s own added
# keywords). Using a whitelist — rather than "not in VANILLA_PRELUDE" — avoids
# flagging typos and prose (this/the/if/hello) that leak in when a file has no
# recognizable early section header.
PS_PLUS_PRELUDE = {
    "case_sensitive", "sprite_size", "tween_length", "tween_easing",
    "tween_snap", "smoothscreen", "nosmoothscreen",
    "level_select", "level_select_lock", "level_select_unlocked_ahead",
    "level_select_solve_symbol", "continue_is_level_select",
    "text_controls", "message_text_align", "status_line", "local_radius",
    "skip_title_screen", "runtime_metadata_twiddling", "nothing_win",
    "mouse_left", "mouse_right", "mouse_up", "mouse_down", "mouse_drag",
    "mouse_static", "mouse_clickdrag", "author_color", "background_image",
}

_DELIM = re.compile(r"^=+$")
_WORD = re.compile(r"^[a-z_]+$")


def classify(text: str) -> dict:
    """Return {category, ps_plus_sections[], nonvanilla_prelude_kw[]}."""
    lines = text.split("\n")
    # Prelude = everything before the first standalone vanilla section header.
    prelude_end = len(lines)
    for i, ln in enumerate(lines):
        if ln.strip().upper() in VANILLA_SECTIONS:
            prelude_end = i
            break

    # Prelude directives outside vanilla, split into known-PS+ (whitelist) vs
    # ambiguous (typo / prose / other). Only the PS+ set drives classification.
    psplus_kw = []
    ambiguous_kw = []
    for ln in lines[:prelude_end]:
        s = ln.strip()
        if not s or _DELIM.match(s) or s.startswith("("):
            continue
        tok = s.split()[0].lower()
        if tok in VANILLA_PRELUDE:
            continue
        if tok in PS_PLUS_PRELUDE:
            psplus_kw.append(tok)
        elif _WORD.match(tok):
            ambiguous_kw.append(tok)

    # Section headers: TAGS/MAPPINGS are clean PS+ (zero false positives).
    # Other delimiter-flanked barewords not in the vanilla set are recorded
    # separately as "other non-vanilla sections" (typos / other dialects) and
    # do NOT drive the PS+ classification.
    ps_sections = []
    other_sections = []
    up = [ln.strip().upper() for ln in lines]
    for i, u in enumerate(up):
        if u in PS_PLUS_SECTIONS:
            ps_sections.append(u)
            continue
        if u and re.match(r"^[A-Z_]+$", u) and u not in VANILLA_SECTIONS:
            prev_d = i > 0 and _DELIM.match(lines[i - 1].strip())
            next_d = i + 1 < len(lines) and _DELIM.match(lines[i + 1].strip())
            if prev_d and next_d:
                other_sections.append(u)

    # Classify (PS+ = clean structural OR whitelisted prelude directive).
    if ps_sections or psplus_kw:
        cat = "ps_plus"
    elif other_sections or ambiguous_kw:
        cat = "other_nonvanilla"
    else:
        cat = "vanilla"
    return {"category": cat,
            "ps_plus_sections": sorted(set(ps_sections)),
            "psplus_prelude_kw": sorted(set(psplus_kw)),
            "other_sections": sorted(set(other_sections)),
            "ambiguous_prelude_kw": sorted(set(ambiguous_kw))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-dir", required=True)
    ap.add_argument("--cache", default=None,
                    help="dedup_cache.jsonl for parse_status cross-tab "
                         "(default <master>/dedup_cache.jsonl)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-removal-list", default=None,
                    help="Write newline-separated filenames flagged ps_plus_section "
                         "(highest-confidence PS+) to this path.")
    args = ap.parse_args()

    master = Path(args.master_dir).resolve()
    files = sorted(f for f in os.listdir(master) if f.endswith(".txt"))

    # Optional parse_status by content hash.
    status_by_hash = {}
    cache_path = Path(args.cache) if args.cache else master / "dedup_cache.jsonl"
    if cache_path.is_file():
        for ln in cache_path.read_text().splitlines():
            if ln.strip():
                try:
                    r = json.loads(ln)
                    status_by_hash[r["content_hash"]] = r.get("parse_status", "?")
                except Exception:
                    pass

    cats = collections.Counter()
    psplus_kw_hist = collections.Counter()
    sec_hist = collections.Counter()
    other_sec_hist = collections.Counter()
    crosstab = collections.Counter()  # (category, parse_status_kind)
    removal = []
    records = {}
    for fn in files:
        b = (master / fn).read_bytes()
        text = b.decode("utf-8", "replace")
        c = classify(text)
        cats[c["category"]] += 1
        for k in c["psplus_prelude_kw"]:
            psplus_kw_hist[k] += 1
        for s in c["ps_plus_sections"]:
            sec_hist[s] += 1
        for s in c["other_sections"]:
            other_sec_hist[s] += 1
        ch = hashlib.sha1(b).hexdigest()
        st = status_by_hash.get(ch, "uncached").split(":")[0]
        crosstab[(c["category"], st)] += 1
        records[fn] = c
        if c["category"] == "ps_plus":
            removal.append(fn)

    print(f"files: {len(files)}")
    print("categories:", dict(cats))
    print("\nPS+ section headers seen:", dict(sec_hist))
    print("top PS+ prelude directives:")
    for k, n in psplus_kw_hist.most_common(25):
        print(f"  {k:28} {n}")
    print("\nother non-vanilla section headers (typos/other dialects):",
          dict(other_sec_hist.most_common(15)))
    print("\ncategory x parse_status:")
    for (cat, st), n in sorted(crosstab.items()):
        print(f"  {cat:20} {st:18} {n}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"counts": dict(cats), "ps_plus_sections": dict(sec_hist),
             "psplus_prelude_kw": dict(psplus_kw_hist),
             "other_sections": dict(other_sec_hist),
             "crosstab": {f"{a}|{b}": n for (a, b), n in crosstab.items()},
             "records": records}, indent=2))
        print(f"\nwrote {args.out}")
    if args.emit_removal_list:
        Path(args.emit_removal_list).write_text("\n".join(removal) + "\n")
        print(f"wrote {len(removal)} filenames to {args.emit_removal_list}")


if __name__ == "__main__":
    main()
