"""Resolve the master's fallback (non-gist-named) variant files to the gist they
are a revision of, by matching their content against each candidate gist's
revision history. A fallback game is title-named because no gist id was in its
source filename; but it is a *revision* of a gist we hold, so its exact content
should appear in some commit of that gist.

For each fallback file: gather candidate gist ids by title (from the master,
pedro staging manifest, and gallery games_dat.js), then walk each candidate
gist's commit history (/gists/<id>/<sha>) until a revision's content hash matches.
On a match, record gist_id (+ revision sha) and rename to <gist_id>_<short>.txt.
"""
import json, re, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ps_dataset as B
import requests

MASTER = B.DEFAULT_MASTER
GID = re.compile(r'^([0-9a-f]{20}|[0-9a-f]{32}|[0-9]+)$')


def title_candidates():
    t2g = defaultdict(set)
    for l in (MASTER / 'manifest.jsonl').read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l); g = r.get('gist_id')
        if g and GID.match(g) and r.get('title'):
            t2g[r['title'].strip().lower()].add(g)
    pm = B.DEFAULT_STAGING / 'pedro' / '_manifest.jsonl'
    if pm.is_file():
        for l in pm.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get('title') and r.get('gist_id'):
                    t2g[r['title'].strip().lower()].add(r['gist_id'])
    gd = B.REPO_ROOT / 'PuzzleScript' / 'src' / 'games_dat.js'
    if gd.is_file():
        txt = gd.read_text()
        try:
            arr = json.loads(re.sub(r',\s*([\]}])', r'\1', txt[txt.index('['):txt.rindex(']') + 1]))
            for e in arr:
                m = re.search(r'p=([0-9a-fA-F]+)', str(e.get('url', '')))
                if m and e.get('title'):
                    t2g[e['title'].strip().lower()].add(m.group(1).lower())
        except ValueError:
            pass
    return t2g


def main():
    token = B.get_token()
    H = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
         "X-GitHub-Api-Version": "2022-11-28"}
    t2g = title_candidates()

    fallbacks = [json.loads(l) for l in (MASTER / 'manifest.jsonl').read_text().splitlines()
                 if l.strip() and '"master_file"' in l and '"gist_id": null' in l]
    print(f"{len(fallbacks)} fallback files to resolve")

    def gh(url):
        for _ in range(4):
            try:
                r = requests.get(url, headers=H, timeout=60)
            except requests.RequestException:
                time.sleep(2); continue
            if r.status_code in (403, 429):
                time.sleep(int(r.headers.get("Retry-After", 5)) + 1); continue
            return r
        return None

    resolved = {}   # master_file -> (gist_id, sha)
    for r in fallbacks:
        mf = MASTER / r['master_file']
        if not mf.exists():
            continue
        target = B.content_hash(mf.read_text(errors='replace'))
        cands = sorted(t2g.get((r.get('title') or '').strip().lower(), set()))[:10]
        hit = None
        for gid in cands:
            g = gh(f"https://api.github.com/gists/{gid}")
            if not g or not g.ok:
                continue
            for ver in (g.json().get('history') or [])[:40]:
                sha = ver.get('version')
                if not sha:
                    continue
                rv = gh(f"https://api.github.com/gists/{gid}/{sha}")
                if not rv or not rv.ok:
                    continue
                for f in (rv.json().get('files') or {}).values():
                    c = f.get('content')
                    if c and B.content_hash(c) == target:
                        hit = (gid, sha); break
                if hit:
                    break
            if hit:
                break
        if hit:
            resolved[r['master_file']] = hit
            print(f"  RESOLVED {r['master_file']}  -> gist {hit[0]} rev {hit[1][:8]}")
        else:
            print(f"  unresolved ({len(cands)} candidates) {r['master_file']}")

    print(f"\nresolved {len(resolved)}/{len(fallbacks)} fallbacks to a gist revision")

    # Apply: rename files to <gist_id>_<short>.txt and patch the manifest rows.
    lines = (MASTER / 'manifest.jsonl').read_text().splitlines()
    out = []
    for l in lines:
        if not l.strip():
            continue
        r = json.loads(l)
        mf = r.get('master_file')
        if mf in resolved:
            gid, sha = resolved[mf]
            short = B.content_hash((MASTER / mf).read_text(errors='replace'))[:10]
            newname = f"{gid}_{short}.txt"
            (MASTER / mf).rename(MASTER / newname)
            r['gist_id'] = gid
            r['gist_revision'] = sha
            r['master_file'] = newname
        out.append(json.dumps(r, ensure_ascii=False))
    (MASTER / 'manifest.jsonl').write_text("\n".join(out) + "\n")
    print("manifest + filenames updated")


if __name__ == '__main__':
    main()
