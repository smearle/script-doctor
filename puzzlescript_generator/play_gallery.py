"""Playable HTML gallery of games generated for fixed levels (level_eval.py output).

    python play_gallery.py --eval DIR --ps-repo DIR --ps-commit SHA --out FILE.html [--wrap]

--eval is a level_eval.py output directory (samples.jsonl, candidates.jsonl). Per level with
at least one generated game that is playable and changes the level under random play
(ps_check.js --dynamics), the page lists the level's human original, then one game per
behaviour class (equal ps_probe.js fingerprints, as in level_eval.summarize), puzzles first.
Playable games that never change the level are counted, not listed.

Games run in the reference engine at the checker's commit: PuzzleScript's own export page
(src/standalone.html, MIT licence) with its scripts inlined, as upstream compile.js builds
standalone_inlined.txt. Changes to that page:
- progress is kept in memory, since every game would otherwise share one storage key;
- a game starts on its first level, not the title screen;
- the title and footer are hidden;
- script errors and wins are posted to the gallery.
Each game is played in a fresh iframe.

Without --wrap the output is a page body for the Artifact tool, which adds the document
skeleton; --wrap writes a whole document for opening from disk.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADERS = ("OBJECTS", "LEGEND", "SOUNDS", "COLLISIONLAYERS", "RULES", "WINCONDITIONS", "LEVELS")
SCRIPT_RE = re.compile(r'<script src="(js/[A-Za-z0-9_/-]+\.js)"></script>')
MEMORY_STORAGE = """'use strict';
// Gallery build (play_gallery.py): progress is kept in memory for this game only.
const GALLERY_STORE = new Map();
function storage_has(key) { return GALLERY_STORE.has(key); }
function storage_get(key) { return GALLERY_STORE.has(key) ? GALLERY_STORE.get(key) : null; }
function storage_get_int(key, defaultValue) {
    const v = parseInt(storage_get(key), 10);
    return isNaN(v) ? defaultValue : v;
}
function storage_set(key, value) { GALLERY_STORE.set(key, String(value)); }
function storage_remove(key) { GALLERY_STORE.delete(key); }
"""
POST_HOOKS = """<script>
// Gallery build (play_gallery.py): report script errors to the gallery page.
window.addEventListener('error', function (e) {
  try { parent.postMessage({ psGalleryError: String(e.message || 'script error') }, '*'); } catch (_) {}
});
</script>"""
WIN_HOOK = """<script>
// Gallery build (play_gallery.py): tell the gallery when the level is won.
(function () {
  const next = nextLevel;
  nextLevel = function () {
    try { parent.postMessage({ psGalleryWon: true }, '*'); } catch (_) {}
    return next.apply(this, arguments);
  };
})();
</script>"""
HIDE_CHROME = ("<style>/* Gallery build: the gallery shows the title and links. */"
               ".title,.footer{display:none!important}.gameContainer{top:0!important;bottom:0!important}"
               "body{margin:0;overflow:hidden}</style>")
START_OLD = '<script>const sourceCode="__GAMEDAT__";compile(["restart"],sourceCode);</script>'
START_NEW = '<script>const sourceCode="__GAMEDAT__";compile(["loadFirstNonMessageLevel"],sourceCode);</script>'


def git_show(repo: str, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], check=True,
                          capture_output=True, text=True).stdout


def player_template(repo: str, commit: str) -> str:
    """PuzzleScript's export page at `commit`, scripts inlined; __GAMETITLE__ and "__GAMEDAT__" left open."""
    page = git_show(repo, commit, "src/standalone.html")

    def inline(m):
        code = MEMORY_STORAGE if m.group(1) == "js/storagewrapper.js" else git_show(repo, commit, "src/" + m.group(1))
        if "</script" in code.lower():
            raise ValueError(f"{m.group(1)} contains a script end tag")
        return "<script>\n" + code + "\n</script>"

    page, n = SCRIPT_RE.subn(inline, page)
    for old, new in (("<!--___SCRIPTINSERT___-->", POST_HOOKS), (START_OLD, WIN_HOOK + "\n" + START_NEW),
                     ("</head>", HIDE_CHROME + "</head>")):
        if page.count(old) != 1:
            raise ValueError(f"export page changed: {old[:40]!r} found {page.count(old)} times")
        page = page.replace(old, new)
    page = (page.replace("___BGCOLOR___", "#000000").replace("___TEXTCOLOR___", "#cccccc")
            .replace("__HOMEPAGE_STRIPPED_PROTOCOL__", "puzzlescript.net")
            .replace("__HOMEPAGE__", "https://www.puzzlescript.net"))
    licence = git_show(repo, commit, "LICENSE").strip()
    head, rest = page.split("\n", 1)
    note = (f"<!-- PuzzleScript {commit}, src/standalone.html with its scripts inlined for a gallery "
            f"(play_gallery.py).\n{licence}\n-->")
    if n < 10 or not head.lower().startswith("<!doctype"):
        raise ValueError("unexpected export page")
    return head + "\n" + note + "\n" + rest


def sections(src: str) -> dict:
    out, cur = {"prelude": []}, "prelude"
    for line in src.split("\n"):
        s = line.strip()
        if s in HEADERS:
            cur = s
            out[cur] = []
        elif s:
            out.setdefault(cur, []).append(s)
    return out


def level_info(src: str) -> dict:
    """Objects, legend, layers, flags and the level's cell colours from a standard one-level text."""
    sec = sections(src)
    objs, lines = {}, sec["OBJECTS"]
    for name, colour in zip(lines[0::2], lines[1::2]):
        objs[name.lower()] = colour
    legend = {}
    for line in sec["LEGEND"]:
        name, rhs = (s.strip() for s in line.split("=", 1))
        words = rhs.lower().split()
        op = "and" if "and" in words else "or"
        legend[name.lower()] = {"op": op, "members": [w for w in words if w not in ("and", "or")]}

    def expand(name, seen=()):
        if name in objs:
            return [name]
        ent = legend.get(name)
        if not ent or name in seen:
            return []
        return [o for m in ent["members"] for o in expand(m, seen + (name,))]

    layers = {}
    for i, line in enumerate(sec["COLLISIONLAYERS"]):
        for name in re.split(r"[,\s]+", line.lower()):
            for o in expand(name):
                layers.setdefault(o, i)
    grid = sec["LEVELS"]
    glyph_objs = {".": ["background"]}
    for ch in {c for row in grid for c in row} - {"."}:
        ent = legend[ch]
        # A cell lists objects with "and"; a background or player defined as a property shows its first member.
        glyph_objs[ch] = [expand(m)[0] for m in ent["members"]]
    pal, cells, in_level = [], [], set()
    for row in grid:
        out = []
        for ch in row:
            top = max(glyph_objs[ch], key=lambda o: layers.get(o, -1))
            in_level.update(glyph_objs[ch])
            colour = objs[top]
            if colour not in pal:
                pal.append(colour)
            out.append(pal.index(colour))
        cells.append(out)
    flags = [line.split()[0].lower() for line in sec["prelude"]]
    return {"width": len(grid[0]), "height": len(grid), "pal": pal, "cells": cells, "objects": objs,
            "legend": {k: v for k, v in legend.items() if len(k) > 1}, "layers": layers, "flags": flags,
            "realtime": "realtime_interval" in flags,
            "inLevel": sorted(in_level, key=lambda o: (layers.get(o, 99), o))}


def game_record(r: dict, kind: str) -> dict:
    sec = sections(r["standard"])
    c, p = r["check"], r["probe"]
    roll, bfs = c["rollout"], c["bfs"]
    dynamic = roll["changed"] > 0
    return {"id": r["id"], "kind": kind, "temp": r["temp"],
            "num": int(r["id"].rsplit("-", 1)[1]) if kind == "gen" else None,
            "src": r["standard"], "rules": sec.get("RULES", []), "wins": sec.get("WINCONDITIONS", []),
            "rollout": {k: roll[k] for k in ("changed", "distinct", "won")},
            "bfs": {k: bfs[k] for k in ("solved", "sol_len", "iters", "timeout")},
            "states": p.get("distinct_states"), "stochastic": bool(p.get("stochastic")),
            "puzzle": dynamic and not roll["won"] and bool(bfs["solved"]) and (bfs["sol_len"] or 0) >= 5,
            "class_size": 1}


def build(args) -> tuple[str, dict]:
    ev = Path(args.eval)
    rows = [json.loads(l) for l in open(ev / "samples.jsonl")]
    cands = {json.loads(l)["id"]: json.loads(l) for l in open(ev / "candidates.jsonl")}
    human = {r["level"]: r for r in rows if r["id"].startswith("human-")}
    gen = defaultdict(list)
    for r in rows:
        if not r["id"].startswith("human-"):
            gen[r["level"]].append(r)
    totals = {"samples": sum(len(v) for v in gen.values()), "playable": 0, "dynamic": 0, "listed": 0, "puzzles": 0}
    levels = []
    for lid, samples in gen.items():
        play = [r for r in samples if r.get("check", {}).get("ok")]
        dyn = [r for r in play if r["check"]["rollout"]["changed"] > 0]
        classes, recs = defaultdict(list), []
        for r in dyn:
            if r.get("probe", {}).get("ok"):
                recs.append(game_record(r, "gen"))
                classes[tuple(r["probe"]["probes"])].append(recs[-1])
        reps = []
        for members in classes.values():
            members.sort(key=lambda g: (not g["puzzle"], not g["bfs"]["solved"], -(g["states"] or 0), g["temp"], g["num"]))
            rep = dict(members[0], class_size=len(members))
            reps.append(rep)
        reps.sort(key=lambda g: (not g["puzzle"], -(g["bfs"]["sol_len"] or 0) if g["puzzle"] else 0,
                                 g["rollout"]["won"], -(g["states"] or 0), g["temp"], g["num"]))
        n_puzzles = sum(g["puzzle"] for g in recs)
        totals["playable"] += len(play)
        totals["dynamic"] += len(dyn)
        totals["listed"] += len(reps)
        totals["puzzles"] += n_puzzles
        if not reps:
            continue
        h = human[lid]
        info = level_info(h["standard"])
        short = lid.split("-")[0][:8] + "-" + lid.rsplit("-", 1)[1]
        levels.append({"id": lid, "short": short, **info,
                       "orig": {"objects": h["n_objects"], "rules": h["n_rules"], "inLevel": cands[lid]["n_objects"]},
                       "stats": {"samples": len(samples), "playable": len(play), "dynamic": len(dyn),
                                 "puzzles": n_puzzles},
                       "games": [game_record(h, "human")] + reps})
    levels.sort(key=lambda lv: (-(len(lv["games"]) - 1), -lv["stats"]["puzzles"], lv["id"]))
    n_levels = len(gen)
    colophon = (
        "<p><strong>How these games were made.</strong> The model is a decoder-only transformer with about 30M "
        "parameters (8 layers, width 512), trained on 63,126 level-first documents from 15,424 deduplicated "
        "PuzzleScript mechanics. Canonical games have no sprites, names or sounds: each object is a coloured "
        "square, and colours repeat after 24 objects. The prompts are " + str(n_levels) + " held-out levels with "
        "their objects, legend, collision layers and flags; the model wrote 64 rule sets per level at each of the "
        "temperatures 0.6 and 0.8.</p>"
        "<p><strong>Checks.</strong> Every sample was compiled by the reference engine (PuzzleScript "
        f"<code>{html.escape(args.ps_commit[:8])}</code>), which then played a 200-move seeded random rollout, a "
        "breadth-first search of up to 20,000 steps and 16 shared 48-move random probes. A puzzle changes the "
        "level, is not won by the random rollout, and has a search solution of at least 5 moves. "
        f"{n_levels - len(levels)} of the {n_levels} levels have no generated game that changes the level and are "
        "not shown. These games run in that same engine, with progress kept only while the game is open.</p>"
        f"<p><strong>Files.</strong> Samples and scores: <code>{html.escape(args.cite)}</code>. Built by "
        "<code>experiments/puzzlescript_generator_20261006/play_gallery.py</code>.</p>"
        "<details><summary>PuzzleScript licence</summary><pre>"
        + html.escape(git_show(args.ps_repo, args.ps_commit, "LICENSE").strip()) + "</pre></details>")
    return {"totals": totals, "levels": levels, "colophon": colophon}


def script_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ps-repo", required=True)
    ap.add_argument("--ps-commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cite", default="results/puzzlescript_generator_20261006/level-first-01/workspace/diag-rules/",
                    help="where the page says the samples live")
    ap.add_argument("--wrap", action="store_true", help="write a whole HTML document")
    args = ap.parse_args()
    data = build(args)
    page = (HERE / "play_gallery.template.html").read_text()
    page = page.replace("__PLAYER_JSON__", script_json(player_template(args.ps_repo, args.ps_commit)), 1)
    page = page.replace("__GALLERY_JSON__", script_json(data), 1)
    if args.wrap:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + page.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page)
    t = data["totals"]
    print(f"{args.out}: {len(page) / 1e6:.2f} MB, {len(data['levels'])} levels, {t['listed']} generated games listed "
          f"({t['samples']} samples, {t['playable']} playable, {t['dynamic']} dynamic, {t['puzzles']} puzzles)")


if __name__ == "__main__":
    main()
