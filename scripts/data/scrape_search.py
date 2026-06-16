#!/usr/bin/env python3
"""
Harvest PuzzleScript share links of the form:
  https://www.puzzlescript.net/play.html?p=<ID>

at scale using a search engine API (Bing Web Search API), then fetch the
corresponding PuzzleScript source via GitHub Gist API and save to:

  data/scraped_games_search/

Filenames:
  TITLE_by_AUTHOR.txt
and if duplicates, append _N before .txt (N starts at 2)

Requirements:
  pip install requests

Environment:
  BING_API_KEY        (required) Azure Bing Search v7 key
  BING_ENDPOINT       (optional) default: https://api.bing.microsoft.com/v7.0/search
  GITHUB_TOKEN        (optional but recommended) increases GitHub API rate limit

Notes:
- PuzzleScript play.html?p=... corresponds to a GitHub gist id. :contentReference[oaicite:0]{index=0}
- Respect search engine ToS and be gentle with rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

import requests


PLAY_URL_RE = re.compile(r"https?://(?:www\.)?puzzlescript\.net/play\.html\?p=([A-Za-z0-9]+)")
TITLE_RE = re.compile(r"(?im)^\s*title\s+(.+?)\s*$")
AUTHOR_RE = re.compile(r"(?im)^\s*author\s+(.+?)\s*$")


@dataclass(frozen=True)
class FoundGame:
    gist_id: str
    play_url: str


def sanitize_filename_component(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # Replace filesystem-hostile characters
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = s.replace("\u200e", "").replace("\u200f", "")  # strip some bidi marks if present
    s = s.strip(" .")
    if not s:
        s = "UNKNOWN"
    return s[:max_len]


def extract_gist_ids_from_text(text: str) -> Set[str]:
    return set(m.group(1) for m in PLAY_URL_RE.finditer(text or ""))


def parse_gist_id_from_url(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        if u.netloc.endswith("puzzlescript.net") and u.path.endswith("/play.html"):
            qs = parse_qs(u.query)
            p = qs.get("p", [None])[0]
            if p and re.fullmatch(r"[A-Za-z0-9]+", p):
                return p
    except Exception:
        return None
    return None


def bing_search(
    *,
    query: str,
    api_key: str,
    endpoint: str,
    count: int,
    offset: int,
    market: str = "en-US",
    safe_search: str = "Off",
    timeout_s: int = 30,
) -> Dict:
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "mkt": market,
        "count": count,
        "offset": offset,
        "safeSearch": safe_search,
        "textDecorations": False,
        "textFormat": "Raw",
    }
    r = requests.get(endpoint, headers=headers, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def harvest_play_links_via_bing(
    *,
    max_results: int,
    per_page: int,
    sleep_s: float,
    market: str,
    verbose: bool,
) -> List[FoundGame]:
    api_key = os.environ.get("BING_API_KEY")
    if not api_key:
        raise SystemExit("Missing env var BING_API_KEY")
    endpoint = os.environ.get("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

    # Query engineered for recall; you can add variants or multiple queries if desired.
    query = 'site:puzzlescript.net/play.html?p='

    found: Dict[str, FoundGame] = {}
    offset = 0

    while len(found) < max_results:
        batch_n = min(per_page, max_results - len(found))
        if verbose:
            print(f"[bing] offset={offset} count={batch_n} found_so_far={len(found)}")

        data = bing_search(
            query=query,
            api_key=api_key,
            endpoint=endpoint,
            count=batch_n,
            offset=offset,
            market=market,
        )

        items = (data.get("webPages") or {}).get("value") or []
        if not items:
            if verbose:
                print("[bing] no more results")
            break

        for it in items:
            url = it.get("url") or ""
            snippet = it.get("snippet") or ""
            name = it.get("name") or ""

            gid = parse_gist_id_from_url(url)
            if gid:
                found.setdefault(gid, FoundGame(gist_id=gid, play_url=url))

            # Also scan snippet/title for embedded links.
            for gid2 in extract_gist_ids_from_text(" ".join([url, name, snippet])):
                play_url = f"https://www.puzzlescript.net/play.html?p={gid2}"
                found.setdefault(gid2, FoundGame(gist_id=gid2, play_url=play_url))

        offset += len(items)
        time.sleep(max(0.0, sleep_s))

    return list(found.values())[:max_results]


def github_gist_get(gist_id: str, session: requests.Session) -> Dict:
    url = f"https://api.github.com/gists/{gist_id}"
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        raise FileNotFoundError(f"gist not found: {gist_id}")
    r.raise_for_status()
    return r.json()


def choose_gist_file(gist_json: Dict) -> Tuple[str, Dict]:
    files = gist_json.get("files") or {}
    if not files:
        raise ValueError("gist has no files")
    # Prefer typical PuzzleScript naming
    for preferred in ("script.txt", "readme.txt", "game.txt"):
        if preferred in files:
            return preferred, files[preferred]
    # else first file deterministically
    first_name = sorted(files.keys())[0]
    return first_name, files[first_name]


def fetch_file_content(file_obj: Dict, session: requests.Session) -> str:
    content = file_obj.get("content")
    truncated = bool(file_obj.get("truncated"))
    raw_url = file_obj.get("raw_url")

    if content is not None and not truncated:
        return content

    if not raw_url:
        raise ValueError("file content truncated but no raw_url provided")

    r = session.get(raw_url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_title_author(source: str) -> Tuple[str, str]:
    title = "UNKNOWN_TITLE"
    author = "UNKNOWN_AUTHOR"
    m = TITLE_RE.search(source)
    if m:
        title = m.group(1).strip()
    m = AUTHOR_RE.search(source)
    if m:
        author = m.group(1).strip()
    return title, author


def unique_path_for_game(out_dir: Path, title: str, author: str) -> Path:
    base = f"{sanitize_filename_component(title)}_by_{sanitize_filename_component(author)}"
    p = out_dir / f"{base}.txt"
    if not p.exists():
        return p
    n = 2
    while True:
        p2 = out_dir / f"{base}_{n}.txt"
        if not p2.exists():
            return p2
        n += 1


def build_github_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-results", type=int, default=5000)
    ap.add_argument("--per-page", type=int, default=50, help="Bing results per request (<=50)")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep between Bing requests")
    ap.add_argument("--market", type=str, default="en-US")
    ap.add_argument("--out", type=str, default="data/scraped_games_search")
    ap.add_argument("--log", type=str, default="data/scraped_games_search/_scrape_log.jsonl")
    ap.add_argument("--skip-existing", action="store_true", help="skip IDs already in log")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    already: Set[str] = set()
    if args.skip_existing and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "gist_id" in rec:
                        already.add(str(rec["gist_id"]))
                except Exception:
                    continue
        if args.verbose:
            print(f"[log] loaded {len(already)} existing gist_ids")

    games = harvest_play_links_via_bing(
        max_results=args.max_results,
        per_page=min(args.per_page, 50),
        sleep_s=args.sleep,
        market=args.market,
        verbose=args.verbose,
    )

    gh = build_github_session()

    saved = 0
    skipped = 0
    errors = 0

    with log_path.open("a", encoding="utf-8") as logf:
        for g in games:
            if g.gist_id in already:
                skipped += 1
                continue

            try:
                gist = github_gist_get(g.gist_id, gh)
                fname, fobj = choose_gist_file(gist)
                source = fetch_file_content(fobj, gh)
                title, author = parse_title_author(source)

                path = unique_path_for_game(out_dir, title, author)
                path.write_text(source, encoding="utf-8")

                rec = {
                    "gist_id": g.gist_id,
                    "play_url": g.play_url,
                    "gist_html_url": gist.get("html_url"),
                    "gist_file": fname,
                    "title": title,
                    "author": author,
                    "saved_path": str(path),
                    "saved_at_unix": int(time.time()),
                }
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()

                saved += 1
                if args.verbose:
                    print(f"[saved] {g.gist_id} -> {path.name}")

            except Exception as e:
                errors += 1
                rec = {
                    "gist_id": g.gist_id,
                    "play_url": g.play_url,
                    "error": repr(e),
                    "saved_at_unix": int(time.time()),
                }
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
                if args.verbose:
                    print(f"[error] {g.gist_id}: {e}")

    print(json.dumps({"saved": saved, "skipped": skipped, "errors": errors, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
