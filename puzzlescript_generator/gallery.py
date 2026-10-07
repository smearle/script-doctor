"""Contact sheets of generated PuzzleScript games, rendered by the reference engine.

For each arm of sample_eval.py the sheet shows the FIRST n playable texts in sample
order (samples are i.i.d., so this is an unbiased draw from the playable ones; the
header gives the arm's playable rate). For levels_t1.0 each tile pairs the human game's
original first level (left) with the generated first level (right) under the same rules.

    python gallery.py --eval DIR --data DIR --engine-dir DIR --out DIR [--n 16]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TILE = 200
LABEL = 30
TITLE_RE = re.compile(r"(?im)^\s*title\s+(.+?)\s*$")


def render(items, engine_dir, node):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "text": it["text"]}) + "\n")
    out = subprocess.run([node, str(HERE / "ps_render.js"), str(engine_dir), f.name],
                         capture_output=True, text=True, timeout=900, check=True).stdout
    Path(f.name).unlink()
    imgs = {}
    for line in out.splitlines():
        r = json.loads(line)
        if "rgb_b64" in r:
            a = np.frombuffer(base64.b64decode(r["rgb_b64"]), dtype=np.uint8)
            imgs[r["id"]] = Image.fromarray(a.reshape(r["height"], r["width"], 3))
    return imgs


def fit(img, box):
    if img is None:
        return Image.new("RGB", (box, box), (60, 0, 0))
    s = max(1, min(box // max(1, img.width), box // max(1, img.height)))
    img = img.resize((img.width * s, img.height * s), Image.NEAREST)
    if img.width > box or img.height > box:
        img.thumbnail((box, box), Image.NEAREST)
    canvas = Image.new("RGB", (box, box), (24, 24, 24))
    canvas.paste(img, ((box - img.width) // 2, (box - img.height) // 2))
    return canvas


def title(text):
    m = TITLE_RE.search(text)
    return (m.group(1) if m else "(untitled)")[:28]


def sheet(tiles, header, cols, tile_w):
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * tile_w, 40 + rows * (TILE + LABEL)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((8, 12), header, fill=(0, 0, 0))
    for k, (img, label) in enumerate(tiles):
        x, y = (k % cols) * tile_w, 40 + (k // cols) * (TILE + LABEL)
        im.paste(img, (x, y))
        d.text((x + 4, y + TILE + 6), label, fill=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--node", default="node")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.eval / "eval_report.json").read_text())
    human = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(args.data / "test_texts.jsonl")}

    for arm in ("uncond_t1.0", "uncond_t0.8", "levels_t1.0", "human_test"):
        recs = [json.loads(l) for l in open(args.eval / f"{arm}.jsonl")]
        ok = [r for r in recs if r["check"].get("ok")][:args.n]
        header = (f"{arm}: first {len(ok)} playable of {len(recs)} samples "
                  f"(playable rate {report[arm]['playable']:.1%})")
        if arm == "levels_t1.0":
            src = [{"id": "src-" + r["id"], "text": human[r["source_game"]]} for r in ok]
            imgs = render(ok + src, args.engine_dir, args.node)
            tiles = []
            for r in ok:
                pair = Image.new("RGB", (2 * TILE + 8, TILE), (255, 255, 255))
                pair.paste(fit(imgs.get("src-" + r["id"]), TILE), (0, 0))
                pair.paste(fit(imgs.get(r["id"]), TILE), (TILE + 8, 0))
                tiles.append((pair, f"{title(r['text'])}: original | generated"))
            im = sheet(tiles, header, 4, 2 * TILE + 16)
        else:
            imgs = render(ok, args.engine_dir, args.node)
            bfs = lambda r: r["check"].get("bfs", {})
            tiles = [(fit(imgs.get(r["id"]), TILE),
                      f"{title(r['text'])} | BFS {'solved ' + str(bfs(r).get('sol_len')) if bfs(r).get('solved') else 'unsolved'}")
                     for r in ok]
            im = sheet(tiles, header, 4, TILE + 8)
        im.save(args.out / f"gallery_{arm}.png")
        print("wrote", args.out / f"gallery_{arm}.png", len(ok))


if __name__ == "__main__":
    main()
