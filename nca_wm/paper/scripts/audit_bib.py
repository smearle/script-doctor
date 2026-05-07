#!/usr/bin/env python3
"""Audit nca_wm/paper/references.bib against canonical metadata sources.

For each entry, query DBLP for the title, fetch the top hit's authoritative
record, and compare title / authors / year / venue. Optionally fall through
to the arXiv API if the entry has an arXiv id and DBLP yields nothing.

Reports each entry as PASS / WARN / MISMATCH; a final summary lists every
mismatch with a one-line suggested fix. Read-only — does not modify the
.bib file.

Usage:
    python audit_bib.py [--bib references.bib] [--keys key1 key2 ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

UA = "Mozilla/5.0 (X11; Linux x86_64) Chrome/124.0"


# ---------------------------------------------------------------------------
# bib parsing
# ---------------------------------------------------------------------------

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}\s*\n", re.S)
FIELD_RE = re.compile(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)")


def _strip_braces(v: str) -> str:
    v = v.strip().rstrip(",").strip()
    if v.startswith("{") and v.endswith("}"):
        v = v[1:-1]
    if v.startswith('"') and v.endswith('"'):
        v = v[1:-1]
    return v


def parse_bib(text: str) -> list[dict]:
    out = []
    for m in ENTRY_RE.finditer(text):
        kind, key, body = m.group(1), m.group(2), m.group(3)
        fields = {f[0].lower(): _strip_braces(f[1]) for f in FIELD_RE.findall(body)}
        fields["__type__"] = kind
        fields["__key__"] = key
        out.append(fields)
    return out


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\\{}~\"]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^\w\s.-]", "", s)
    return s


def _surnames(authors: str) -> list[str]:
    parts = re.split(r"\s+and\s+", authors, flags=re.I)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "," in p:
            out.append(p.split(",", 1)[0].strip())
        else:
            tokens = [t for t in p.split() if t]
            out.append(tokens[-1] if tokens else "")
    return [_norm(s) for s in out if s]


# ---------------------------------------------------------------------------
# remote sources
# ---------------------------------------------------------------------------


def _get(url: str, timeout: int = 25, accept: str = "*/*") -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


@dataclass
class CanonRecord:
    source: str
    title: str
    authors: list[str]  # surnames only
    year: Optional[int]
    venue: str
    raw: dict = field(default_factory=dict)


_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "via", "with",
    "we", "our", "this", "their", "these", "than", "into",
}


def _query_terms(text: str, n: int = 5) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w.lower() not in _STOP and len(w) > 1]
    return " ".join(words[:n])


def dblp_lookup(title: str, authors_hint: list[str] | None = None) -> Optional[CanonRecord]:
    q = _query_terms(title, n=5)
    if authors_hint:
        q = q + " " + authors_hint[0]
    if not q.strip():
        return None
    url = "https://dblp.org/search/publ/api?q=" + urllib.parse.quote_plus(q) + "&format=json&h=5"
    try:
        d = json.loads(_get(url))
    except Exception as e:
        sys.stderr.write(f"[dblp] {q!r}: {e}\n")
        # one retry with even shorter query
        q2 = _query_terms(title, n=3)
        url2 = "https://dblp.org/search/publ/api?q=" + urllib.parse.quote_plus(q2) + "&format=json&h=5"
        try:
            d = json.loads(_get(url2))
        except Exception as e2:
            sys.stderr.write(f"[dblp retry] {q2!r}: {e2}\n")
            return None
    hits = d.get("result", {}).get("hits", {}).get("hit", [])
    if not hits:
        return None
    # Pick the conference/journal hit over CoRR if both exist for the same title.
    target_title = _norm(title)
    scored: list[tuple[int, dict]] = []
    for h in hits:
        info = h.get("info", {})
        ht = _norm(info.get("title", ""))
        if not ht:
            continue
        score = 0
        if ht == target_title:
            score += 100
        elif target_title in ht or ht in target_title:
            score += 50
        if not info.get("key", "").startswith("journals/corr/"):
            score += 5
        scored.append((score, info))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    info = scored[0][1]
    raw_authors = info.get("authors", {}).get("author", [])
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    surnames = []
    for a in raw_authors:
        name = a.get("text") if isinstance(a, dict) else a
        if not name:
            continue
        # strip dblp's disambiguation digits like "Sam Earle 0001"
        name = re.sub(r"\s+\d{4}$", "", name)
        surnames.append(_norm(name.split()[-1]))
    year = info.get("year")
    try:
        year = int(year) if year else None
    except ValueError:
        year = None
    return CanonRecord(
        source="dblp",
        title=info.get("title", "").rstrip("."),
        authors=surnames,
        year=year,
        venue=info.get("venue", ""),
        raw=info,
    )


def arxiv_lookup(arxiv_id: str) -> Optional[CanonRecord]:
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        xml = _get(url)
    except Exception as e:
        sys.stderr.write(f"[arxiv] {arxiv_id}: {e}\n")
        return None
    title = re.search(r"<entry>.*?<title>([^<]+)</title>", xml, re.S)
    year = re.search(r"<entry>.*?<published>(\d{4})", xml, re.S)
    authors = re.findall(r"<author>\s*<name>([^<]+)</name>", xml)
    if not (title and year):
        return None
    surnames = [_norm(a.split()[-1]) for a in authors if a.strip()]
    return CanonRecord(
        source="arxiv",
        title=re.sub(r"\s+", " ", title.group(1).strip()),
        authors=surnames,
        year=int(year.group(1)),
        venue="arXiv",
    )


# ---------------------------------------------------------------------------
# audit logic
# ---------------------------------------------------------------------------


@dataclass
class Diff:
    field: str
    have: str
    canon: str
    severity: str  # info | warn | mismatch


def audit_entry(entry: dict, canon: CanonRecord) -> list[Diff]:
    diffs: list[Diff] = []

    have_title = _norm(entry.get("title", ""))
    canon_title = _norm(canon.title)
    if have_title and canon_title and have_title != canon_title:
        # PuzzleScript subtitles, casing of acronyms, etc — only flag big drift
        if abs(len(have_title) - len(canon_title)) > 3 or have_title not in canon_title and canon_title not in have_title:
            diffs.append(Diff("title", entry.get("title", ""), canon.title, "warn"))

    have_year = entry.get("year", "").strip()
    if canon.year and have_year and str(canon.year) != have_year:
        # Conf-vs-arXiv year drift is common; warn but allow ±1
        sev = "warn" if abs(int(have_year) - canon.year) <= 1 else "mismatch"
        diffs.append(Diff("year", have_year, str(canon.year), sev))

    have_surn = _surnames(entry.get("author", ""))
    canon_surn = canon.authors
    if have_surn and canon_surn:
        if len(have_surn) != len(canon_surn):
            diffs.append(
                Diff(
                    "author count",
                    f"{len(have_surn)} ({','.join(have_surn[:3])}...)",
                    f"{len(canon_surn)} ({','.join(canon_surn[:3])}...)",
                    "warn",
                )
            )
        else:
            for i, (a, b) in enumerate(zip(have_surn, canon_surn)):
                if a != b and not (a in b or b in a):
                    diffs.append(Diff(f"author[{i}]", a, b, "mismatch"))

    have_venue = (entry.get("booktitle") or entry.get("journal") or "").strip()
    if canon.venue and have_venue:
        nv = _norm(canon.venue)
        nh = _norm(have_venue)
        if nv and nh and nv not in nh and nh not in nv:
            # Map common abbreviations
            aliases = {
                "iclr": "international conference on learning representations",
                "icml": "international conference on machine learning",
                "neurips": "advances in neural information processing systems",
                "nips": "advances in neural information processing systems",
                "corl": "conference on robot learning",
                "cog": "ieee conference on games",
                "aaai": "aaai conference",
            }
            if not any(
                (k in nv and aliases[k] in nh) or (k in nh and aliases[k] in nv)
                for k in aliases
            ):
                diffs.append(Diff("venue", have_venue, canon.venue, "warn"))

    return diffs


def find_arxiv_id(entry: dict) -> Optional[str]:
    blob = " ".join(str(v) for v in entry.values())
    m = re.search(r"arXiv[:\s]*(\d{4}\.\d{4,5})", blob, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4}\.\d{4,5})", entry.get("journal", ""))
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bib", default="nca_wm/paper/references.bib")
    p.add_argument("--keys", nargs="*", default=None)
    p.add_argument("--sleep", type=float, default=2.0)
    args = p.parse_args()

    text = Path(args.bib).read_text()
    entries = parse_bib(text)
    if args.keys:
        wanted = set(args.keys)
        entries = [e for e in entries if e["__key__"] in wanted]

    print(f"Auditing {len(entries)} entries from {args.bib}")
    print(f"(Scholar is captcha-blocked from this host; using DBLP + arXiv as canonical source)\n")

    summary: list[tuple[str, list[Diff]]] = []

    for ent in entries:
        key = ent["__key__"]
        kind = ent["__type__"]
        if kind == "misc":
            print(f"  SKIP  {key}  (@misc — software/blog/dataset; no DBLP record)")
            summary.append((key, []))
            continue
        title = ent.get("title", "")
        if not title:
            print(f"  WARN  {key}  (no title)")
            continue

        canon = dblp_lookup(title, _surnames(ent.get("author", "")))
        if not canon:
            aid = find_arxiv_id(ent)
            if aid:
                canon = arxiv_lookup(aid)
        if not canon:
            print(f"  ?     {key}  (no canonical record found)")
            summary.append((key, []))
            time.sleep(args.sleep)
            continue

        diffs = audit_entry(ent, canon)
        if not diffs:
            print(f"  PASS  {key}  ({canon.source}: {canon.title[:60]})")
        else:
            sev = "MISMATCH" if any(d.severity == "mismatch" for d in diffs) else "WARN"
            print(f"  {sev:8s} {key}  ({canon.source}: {canon.title[:60]})")
            for d in diffs:
                print(f"           {d.severity:8s} {d.field}: have={d.have!r}  canon={d.canon!r}")
        summary.append((key, diffs))
        time.sleep(args.sleep)

    print("\n=== SUMMARY ===")
    mismatches = [(k, ds) for k, ds in summary if any(d.severity == "mismatch" for d in ds)]
    warns = [(k, ds) for k, ds in summary if ds and not any(d.severity == "mismatch" for d in ds)]
    print(f"  mismatches: {len(mismatches)}")
    for k, ds in mismatches:
        for d in ds:
            if d.severity == "mismatch":
                print(f"    {k}.{d.field}: {d.have!r} -> {d.canon!r}")
    print(f"  warnings:   {len(warns)}")
    for k, ds in warns:
        for d in ds:
            print(f"    {k}.{d.field} ({d.severity}): {d.have!r} -> {d.canon!r}")


if __name__ == "__main__":
    main()
