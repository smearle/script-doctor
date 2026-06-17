#!/usr/bin/env python3
"""
Scrape itch.io "made with PuzzleScript" listings.

For each game:
- fetch game page
- try to find a gist link and download PS source from gist
- else find the embedded HTML5 iframe, download its standalone HTML
  and try to extract PuzzleScript source from it
- save extracted source to: data/scraped_games_itchio/TITLE_by_AUTHOR[_N].txt

Notes:
- This will successfully extract classic PuzzleScript exports that embed the
  source in a <script type="text/plain" ...> block.
- PuzzleScript Next / custom exports may not embed plaintext source; in that case
  the script will save a .html alongside a .failed marker for manual handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("data/scraped_games_itchio")
ITCH_MANIFEST = OUT_DIR / "_itch_manifest.jsonl"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# ----------------------------
# Helpers
# ----------------------------

def sleep_polite(delay: float):
    if delay > 0:
        time.sleep(delay)

def safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\/\\:*?\"<>|]+", "_", s)  # windows-unsafe
    s = re.sub(r"[\n\r\t]+", " ", s)
    s = s.strip(" ._-")
    if not s:
        s = "untitled"
    return s[:max_len]

def ensure_unique_path(base: Path, ext: str, used: Dict[str, int]) -> Path:
    key = base.name
    n = used.get(key, 0)
    used[key] = n + 1
    if n == 0:
        return base.with_suffix(ext)
    return base.with_name(f"{base.name}_{n}").with_suffix(ext)

def get(url: str, timeout: int = 30) -> requests.Response:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def try_get(url: str, timeout: int = 30) -> Optional[requests.Response]:
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None

# ----------------------------
# Itch parsing
# ----------------------------

@dataclass
class ItchGame:
    url: str
    title: str = ""
    author: str = ""

def parse_listing_games(listing_html: str, listing_url: str) -> List[ItchGame]:
    """
    Listing pages include many <a class="title game_link" href="...">Title</a>
    and an author link nearby. We just grab urls; per-game metadata is refined later.
    """
    soup = BeautifulSoup(listing_html, "html.parser")
    games: List[ItchGame] = []

    # Most stable: links with class game_link
    for a in soup.select("a.game_link"):
        href = a.get("href")
        if not href:
            continue
        # Filter out non-game links
        if "itch.io" not in href and not href.startswith("/"):
            continue
        url = urljoin(listing_url, href)
        # Avoid duplicates in same page
        if any(g.url == url for g in games):
            continue
        games.append(ItchGame(url=url))
    return games

def parse_game_title_author(game_html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(game_html, "html.parser")

    # Title: <h1 class="game_title">... or meta property
    title = ""
    h1 = soup.select_one("h1.game_title")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()

    # Author: often "Author" field or an <a href="/profile/...">
    author = ""
    # itch pages often have .game_info_panel .user_link
    a = soup.select_one(".game_info_panel .user_link, a.user_link")
    if a:
        author = a.get_text(" ", strip=True)
    if not author:
        # fallback: in game_header
        a2 = soup.select_one(".game_header .user_link")
        if a2:
            author = a2.get_text(" ", strip=True)

    title = title or "untitled"
    author = author or "unknown_author"
    return title, author

def find_gist_urls(text: str) -> List[str]:
    # Grab any gist.github.com links
    urls = re.findall(r"https?://gist\.github\.com/[A-Za-z0-9_.-]+/[0-9a-fA-F]+", text)
    # Some people link raw gist URLs directly
    urls += re.findall(r"https?://gist\.githubusercontent\.com/[^\s\"'<>]+", text)
    # De-dupe preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def find_play_ids(text: str) -> List[str]:
    # puzzlescript.net/play.html?p=...
    ids = re.findall(r"https?://(?:www\.)?puzzlescript\.net/play\.html\?p=([A-Za-z0-9]+)", text)
    seen = set()
    out = []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out

def find_iframe_src(game_html: str, game_url: str) -> Optional[str]:
    soup = BeautifulSoup(game_html, "html.parser")

    # Most itch HTML5 embeds use an iframe in the page body.
    # Heuristics: prefer hwcdn.net/html/... and/or any iframe src containing "hwcdn"
    iframes = soup.find_all("iframe")
    if not iframes:
        return None

    candidates = []
    for f in iframes:
        src = f.get("src")
        if not src:
            continue
        full = urljoin(game_url, src)
        candidates.append(full)

    # Prefer CDN html bundles
    def score(u: str) -> int:
        s = 0
        if "hwcdn.net" in u:
            s += 50
        if "/html/" in u:
            s += 20
        if u.endswith(".html") or "index.html" in u:
            s += 10
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0] if candidates else None

# ----------------------------
# Gist extraction
# ----------------------------

def extract_ps_from_gist(gist_url: str) -> Optional[str]:
    """
    If gist_url is a web page:
      - parse for "Raw" links and download likely PS-like files
    If gist_url is already a raw gist URL:
      - download directly
    """
    if "gist.githubusercontent.com" in gist_url:
        r = try_get(gist_url)
        if not r:
            return None
        text = r.text
        return normalize_ps_source(text) if looks_like_ps_source(text) else None

    r = try_get(gist_url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    raw_links = []
    for a in soup.select('a[href*="gist.githubusercontent.com"][href*="/raw/"]'):
        href = a.get("href")
        if href:
            raw_links.append(urljoin(gist_url, href))

    # Some gists use relative raw links
    if not raw_links:
        for a in soup.select('a[data-testid="raw-button"]'):
            href = a.get("href")
            if href:
                raw_links.append(urljoin("https://gist.github.com", href))

    # Try each raw link, pick first that looks like PS
    for raw in raw_links:
        rr = try_get(raw)
        if not rr:
            continue
        text = rr.text
        if looks_like_ps_source(text):
            return normalize_ps_source(text)

    # Fallback: sometimes the gist content is rendered in-page; try extracting code blocks
    for pre in soup.select("table.highlight td.blob-code"):
        # too messy to reconstruct line-by-line reliably; skip
        break

    return None

# ----------------------------
# Gist id resolution + verification
# ----------------------------

_GH_TOKEN_CACHE: List[Optional[str]] = []
_TITLE_RE = re.compile(r"(?im)^\s*title\s+(.+?)\s*$")


def get_gh_token() -> Optional[str]:
    if _GH_TOKEN_CACHE:
        return _GH_TOKEN_CACHE[0]
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        hosts = Path.home() / ".config" / "gh" / "hosts.yml"
        if hosts.is_file():
            for line in hosts.read_text().splitlines():
                line = line.strip()
                if line.startswith("oauth_token:"):
                    tok = line.split(":", 1)[1].strip()
                    break
    _GH_TOKEN_CACHE.append(tok)
    return tok


def gist_id_from_url(url: str) -> Optional[str]:
    """Pull the raw gist id from a gist.github.com or gist.githubusercontent.com URL."""
    m = re.search(r"gist\.github\.com/[A-Za-z0-9_.-]+/([0-9a-fA-F]{20,32}|[0-9]+)", url)
    if m:
        return m.group(1).lower()
    m = re.search(r"gist\.githubusercontent\.com/[A-Za-z0-9_.-]+/([0-9a-fA-F]{20,32}|[0-9]+)/", url)
    return m.group(1).lower() if m else None


def fetch_gist_source_api(gist_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (ps_source, gist_description) for a gist id via the GitHub API, or (None, None)."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = get_gh_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        r = SESSION.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=60)
    except requests.RequestException:
        return None, None
    if not r.ok:
        return None, None
    gist = r.json()
    desc = gist.get("description")
    files = gist.get("files", {})
    if "script.txt" in files and files["script.txt"].get("content"):
        return normalize_ps_source(files["script.txt"]["content"]), desc
    for f in files.values():
        c = f.get("content")
        if c and looks_like_ps_source(c):
            return normalize_ps_source(c), desc
    return None, desc


def _norm_cmp(s: str) -> str:
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return hashlib.sha1("\n".join(lines).encode("utf-8", "replace")).hexdigest()


def _ps_title(src: str) -> Optional[str]:
    m = _TITLE_RE.search(src)
    return m.group(1).strip().lower() if m else None


def verify_gist(embedded_src: Optional[str], gist_src: Optional[str], itch_title: str) -> str:
    """Classify how confidently a candidate gist corresponds to this itch game.

    'exact'   - gist source matches the source actually embedded on the itch page
    'title'   - no embedded source to compare, but the gist's TITLE matches the itch title
    'mismatch'- both present but contents differ (likely a stale/miscopied link)
    'unverified' - gist is the only source available; nothing to cross-check against
    """
    if gist_src is None:
        return "no_gist_source"
    if embedded_src is not None:
        if _norm_cmp(embedded_src) == _norm_cmp(gist_src):
            return "exact"
        gt = _ps_title(gist_src)
        if gt and gt == (_ps_title(embedded_src) or itch_title.strip().lower()):
            return "mismatch_title_ok"  # same game title, different content (version drift)
        return "mismatch"
    gt = _ps_title(gist_src)
    if gt and gt == itch_title.strip().lower():
        return "title"
    return "unverified"


# ----------------------------
# Standalone HTML -> source extraction
# ----------------------------

PS_SECTION_MARKERS = ["OBJECTS", "LEGEND", "RULES", "WINCONDITIONS", "LEVELS"]

def looks_like_ps_source(s: str) -> bool:
    ss = s.upper()
    hits = sum(1 for m in PS_SECTION_MARKERS if m in ss)
    return hits >= 3

def normalize_ps_source(s: str) -> str:
    # Trim leading/trailing whitespace while preserving internal newlines
    return s.strip("\ufeff").strip()

def extract_ps_from_standalone_html(html: str) -> Optional[str]:
    """
    Classic PuzzleScript exports embed the full plaintext source in a <script type="text/plain">.
    We search for script tags and choose one whose text looks like PS source.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Common in classic exports
    for script in soup.find_all("script"):
        t = script.get("type", "")
        if t and "text/plain" not in t:
            continue
        text = script.string
        if not text:
            # Sometimes bs4 doesn't treat it as .string; use get_text
            text = script.get_text()
        if not text:
            continue
        if looks_like_ps_source(text):
            return normalize_ps_source(text)

    # Fallback: raw substring search (some exports embed source in JS strings)
    # This is intentionally conservative to avoid saving minified JS.
    # We look for a contiguous block containing the section markers.
    upper = html.upper()
    idxs = [upper.find(m) for m in PS_SECTION_MARKERS]
    idxs = [i for i in idxs if i != -1]
    if len(idxs) >= 3:
        start = min(idxs)
        # Guess an end near the end of LEVELS section or before </script>
        end = upper.find("</SCRIPT>", start)
        if end == -1:
            end = min(len(html), start + 200000)  # cap
        chunk = html[start:end]
        # Strip HTML entities if any (rare here)
        chunk = re.sub(r"&lt;", "<", chunk)
        chunk = re.sub(r"&gt;", ">", chunk)
        if looks_like_ps_source(chunk):
            return normalize_ps_source(chunk)

    return None

# ----------------------------
# Main scrape logic
# ----------------------------

def log_itch(record: dict):
    with ITCH_MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_embedded_source(game_html: str, game_url: str) -> Optional[str]:
    """Extract the PuzzleScript source actually embedded in the itch HTML5 iframe."""
    iframe = find_iframe_src(game_html, game_url)
    if not iframe:
        return None
    rr = try_get(iframe)
    # Sometimes the iframe is a container that itself embeds another iframe.
    if rr and ("<iframe" in rr.text.lower()) and ("hwcdn.net" not in iframe):
        nested = find_iframe_src(rr.text, iframe)
        if nested:
            rrr = try_get(nested)
            if rrr:
                rr = rrr
    return extract_ps_from_standalone_html(rr.text) if rr else None


def scrape_one_game(game_url: str, out_dir: Path, used: Dict[str, int], delay: float, save_html_on_fail: bool) -> bool:
    r = try_get(game_url)
    if not r:
        return False

    title, author = parse_game_title_author(r.text)
    base_name = safe_filename(f"{title}_by_{author}")
    base_path = out_dir / base_name

    # Candidate gist ids advertised on the itch page: explicit gist links and
    # puzzlescript.net/play.html?p=<gist_id> links (the play id IS the gist id).
    candidate_ids = []
    for gu in find_gist_urls(r.text):
        gid = gist_id_from_url(gu)
        if gid and gid not in candidate_ids:
            candidate_ids.append(gid)
    for pid in find_play_ids(r.text):
        pid = pid.lower()
        if pid not in candidate_ids:
            candidate_ids.append(pid)

    # Ground truth: the source actually embedded/playable on the itch page.
    embedded_src = get_embedded_source(r.text, game_url)

    # Verify each candidate gist against the embedded source (or the title).
    gist_evidence = []
    best = None  # (rank, gist_id, gist_src, verdict)
    rank = {"exact": 4, "title": 3, "unverified": 2, "mismatch_title_ok": 1, "mismatch": 0, "no_gist_source": -1}
    for gid in candidate_ids:
        gist_src, _ = fetch_gist_source_api(gid)
        verdict = verify_gist(embedded_src, gist_src, title)
        gist_evidence.append({"gist_id": gid, "verdict": verdict})
        cand = (rank.get(verdict, -1), gid, gist_src, verdict)
        if best is None or cand[0] > best[0]:
            best = cand

    # Choose what to save: prefer the embedded itch source (ground truth); else a
    # gist source only if it is the sole source available.
    if embedded_src is not None:
        saved_src, saved_from = embedded_src, "embedded"
    elif best is not None and best[2] is not None:
        saved_src, saved_from = best[2], "gist"
    else:
        saved_src, saved_from = None, None

    # Only attach a gist id we actually trust (matches the embedded source, or the
    # only source with a title match). Mismatches are recorded but NOT trusted.
    trusted_gist_id = None
    if best is not None and best[3] in ("exact", "title", "unverified"):
        trusted_gist_id = best[1]

    if saved_src is None:
        log_itch({"itch_url": game_url, "title": title, "author": author,
                  "saved_file": None, "saved_from": None,
                  "gist_candidates": gist_evidence, "trusted_gist_id": None})
        if save_html_on_fail:
            fail_path = ensure_unique_path(base_path, ".failed", used)
            fail_path.write_text(f"No extractable source.\nGame: {game_url}\n", encoding="utf-8")
        sleep_polite(delay)
        return False

    out_path = ensure_unique_path(base_path, ".txt", used)
    out_path.write_text(saved_src, encoding="utf-8")
    log_itch({"itch_url": game_url, "title": title, "author": author,
              "saved_file": out_path.name, "saved_from": saved_from,
              "gist_candidates": gist_evidence, "trusted_gist_id": trusted_gist_id,
              "content_sha1": _norm_cmp(saved_src)})
    sleep_polite(delay)
    return True

def scrape_listing(base_listing_url: str, pages: int, delay: float, save_html_on_fail: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used: Dict[str, int] = {}

    seen_game_urls = set()
    total = 0
    ok = 0

    for page in range(1, pages + 1):
        url = base_listing_url if page == 1 else f"{base_listing_url}?page={page}"
        resp = try_get(url)
        if not resp:
            print(f"[listing] failed: {url}", file=sys.stderr)
            continue

        games = parse_listing_games(resp.text, url)
        # Filter to likely actual game pages on subdomains/user pages
        # (still keep duplicates out)
        filtered = []
        for g in games:
            if g.url in seen_game_urls:
                continue
            # discard links back to itch listing domains
            parsed = urlparse(g.url)
            if parsed.netloc.endswith("itch.io") and parsed.netloc.count(".") >= 2:
                filtered.append(g)
                seen_game_urls.add(g.url)

        print(f"[listing] page {page}: {len(filtered)} new game URLs")

        for g in filtered:
            total += 1
            try:
                success = scrape_one_game(g.url, OUT_DIR, used, delay, save_html_on_fail)
                if success:
                    ok += 1
                    print(f"[ok] {g.url}")
                else:
                    print(f"[no] {g.url}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[err] {g.url}: {e}", file=sys.stderr)

        sleep_polite(delay)

    print(f"Done. Extracted {ok}/{total} games into {OUT_DIR}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--listing",
        default="https://itch.io/games/made-with-puzzlescript",
        help="Base listing URL (default: made-with-puzzlescript). You can swap in e.g. /games/html5/made-with-puzzlescript",
    )
    ap.add_argument("--pages", type=int, default=5, help="How many listing pages to scrape (default: 5)")
    ap.add_argument("--delay", type=float, default=0.7, help="Delay between requests in seconds (default: 0.7)")
    ap.add_argument("--save-html-on-fail", action="store_true", help="Save iframe HTML when source extraction fails")
    args = ap.parse_args()

    scrape_listing(args.listing, args.pages, args.delay, args.save_html_on_fail)

if __name__ == "__main__":
    main()
