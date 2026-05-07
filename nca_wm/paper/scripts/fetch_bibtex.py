#!/usr/bin/env python3
"""Fetch BibTeX entries for a list of paper queries.

Tries Google Scholar first, then DBLP, then arXiv as fallbacks.
Scholar is usually captcha-blocked from server IPs, so DBLP/arXiv normally
do the work.

Usage:
    python fetch_bibtex.py "Hafner Dreamer mastering diverse 2023" ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Optional

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _get(url: str, timeout: int = 30, accept: str = "*/*") -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_scholar_bibtex(query: str) -> Optional[str]:
    q = urllib.parse.quote_plus(query)
    url = f"https://scholar.google.com/scholar?q={q}&hl=en"
    try:
        html = _get(url)
    except Exception as e:
        print(f"[scholar] {query!r}: {e}", file=sys.stderr)
        return None
    if "not a robot" in html or "captcha" in html.lower():
        print(f"[scholar] captcha for {query!r}", file=sys.stderr)
        return None
    m = re.search(r'data-cid="([0-9a-f]{8,})"', html)
    if not m:
        return None
    cid = m.group(1)
    try:
        cite_html = _get(
            f"https://scholar.google.com/scholar?q=info:{cid}:scholar.google.com/"
            f"&output=cite&hl=en"
        )
    except Exception as e:
        print(f"[scholar] cite fetch {query!r}: {e}", file=sys.stderr)
        return None
    bm = re.search(r'href="([^"]*scholar\.bib[^"]*)"', cite_html)
    if not bm:
        return None
    try:
        return _get(bm.group(1).replace("&amp;", "&")).strip()
    except Exception as e:
        print(f"[scholar] bib fetch {query!r}: {e}", file=sys.stderr)
        return None


def fetch_dblp_bibtex(query: str) -> Optional[str]:
    q = urllib.parse.quote_plus(query)
    try:
        payload = json.loads(
            _get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=1")
        )
    except Exception as e:
        print(f"[dblp] search {query!r}: {e}", file=sys.stderr)
        return None
    hits = (
        payload.get("result", {})
        .get("hits", {})
        .get("hit", [])
    )
    if not hits:
        print(f"[dblp] no hits for {query!r}", file=sys.stderr)
        return None
    info = hits[0].get("info", {})
    key = info.get("key")
    if not key:
        return None
    try:
        bib = _get(
            f"https://dblp.org/rec/{key}.bib?param=1",
            accept="application/x-bibtex",
        )
    except Exception as e:
        print(f"[dblp] bib {query!r}: {e}", file=sys.stderr)
        return None
    return bib.strip()


def fetch_arxiv_bibtex(query: str, arxiv_id: Optional[str] = None) -> Optional[str]:
    """Build a minimal bibtex entry from the arXiv API."""
    if arxiv_id:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    else:
        q = urllib.parse.quote_plus(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=1"
    try:
        xml = _get(url)
    except Exception as e:
        print(f"[arxiv] search {query!r}: {e}", file=sys.stderr)
        return None
    title = re.search(r"<entry>.*?<title>([^<]+)</title>", xml, re.S)
    year = re.search(r"<entry>.*?<published>(\d{4})", xml, re.S)
    aid = re.search(r"<entry>.*?<id>http[s]?://arxiv\.org/abs/([^<]+)</id>", xml, re.S)
    authors = re.findall(r"<author>\s*<name>([^<]+)</name>", xml)
    if not (title and year and aid):
        return None
    aid_s = aid.group(1).strip()
    first = authors[0].split()[-1].lower() if authors else "anon"
    key = f"{first}{year.group(1)}arxiv{aid_s.replace('.', '').replace('/', '')[:6]}"
    auth_str = " and ".join(authors)
    return (
        f"@article{{{key},\n"
        f"  title={{{title.group(1).strip()}}},\n"
        f"  author={{{auth_str}}},\n"
        f"  journal={{arXiv preprint arXiv:{aid_s}}},\n"
        f"  year={{{year.group(1)}}}\n"
        f"}}"
    )


def fetch_one(query: str, arxiv_id: Optional[str] = None) -> Optional[str]:
    for fn, name in (
        (fetch_scholar_bibtex, "scholar"),
        (fetch_dblp_bibtex, "dblp"),
        (lambda q: fetch_arxiv_bibtex(q, arxiv_id), "arxiv"),
    ):
        bib = fn(query)
        if bib:
            print(f"% source={name} query={query!r}")
            return bib
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("queries", nargs="+")
    p.add_argument(
        "--arxiv",
        action="append",
        default=[],
        help="optional arxiv_id paired with the query of the same index",
    )
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()

    arxiv_ids = list(args.arxiv) + [None] * (len(args.queries) - len(args.arxiv))

    for q, aid in zip(args.queries, arxiv_ids):
        bib = fetch_one(q, aid)
        if bib:
            print(bib)
            print()
        else:
            print(f"% (no result for {q!r})\n")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
