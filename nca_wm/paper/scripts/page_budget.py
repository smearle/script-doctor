"""Report how many pages a compiled paper PDF spends before the bibliography.

Usage:
    python scripts/page_budget.py [PDF]
    # default PDF: ../main.pdf relative to this script

Strategy: shell out to `pdftotext -layout -f N -l N` per page and look for a
line whose stripped content equals "References" (the bibliography heading
emitted by \\bibliography{...}). The first such page is treated as the start
of the bibliography. Also detects the appendix start (line equal to "A " +
heading text or matching `^[A-Z]\\s+\\S` on a fresh page after References)
via the hyperref .out file when available.

Page-budget convention reported:
    main_body_pages   = bib_start - 1   (everything pre-bibliography:
                                         abstract, sections, figures, tables)
    bibliography_pages = appendix_start - bib_start  (or total - bib_start + 1
                                         if no appendix detected)
    appendix_pages    = total - appendix_start + 1   (if appendix detected)

NeurIPS-style limits: main_body_pages is the number to compare to the
camera-ready / submission cap (e.g. 9 for NeurIPS main paper).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def total_pages(pdf: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report page count for {pdf}")


def page_text(pdf: Path, page: int) -> str:
    return subprocess.check_output(
        ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
        text=True,
    )


def find_bibliography_page(pdf: Path, n_pages: int):
    """Return (page, shared_with_body) for first page whose text contains a
    standalone 'References' heading. `shared_with_body` is True when main-body
    text precedes the heading on that page (matters for page-limit rules:
    NeurIPS counts a shared page against the main-body limit). None if not
    found."""
    # Allow optional leading line number (NeurIPS submission template wraps
    # each line in `lineno`, so pdftotext yields e.g. "  346   References").
    pat = re.compile(r"^\s*(?:\d+\s+)?References\s*$", re.MULTILINE)
    for p in range(1, n_pages + 1):
        text = page_text(pdf, p)
        m = pat.search(text)
        if not m:
            continue
        prefix = text[: m.start()]
        # Strip line-number-only lines and blank lines; what's left is body text.
        meaningful = re.sub(
            r"^\s*\d*\s*$", "", prefix, flags=re.MULTILINE
        ).strip()
        return p, bool(meaningful)
    return None


def find_appendix_page_from_out(pdf: Path) -> int | None:
    """Parse the matching .out (hyperref bookmarks) file for the first
    appendix entry (section.A) and resolve its page from the .aux file."""
    aux = pdf.with_suffix(".aux")
    if not aux.exists():
        return None
    # \newlabel{...}{{A}{P}{...}{section.A}{}}  -> grab P for first appendix sec
    pat = re.compile(
        r"\\newlabel\{[^}]+\}\{\{[A-Z](?:\.\d+)?\}\{(\d+)\}\{[^}]*\}\{section\.A\}"
    )
    for line in aux.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1))
    # fallback: search for section.A page in toc lines
    pat2 = re.compile(
        r"\\contentsline \{section\}\{\\numberline \{A\}[^}]*\}\{(\d+)\}\{section\.A\}"
    )
    for line in aux.read_text(errors="ignore").splitlines():
        m = pat2.search(line)
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_pdf = Path(__file__).resolve().parent.parent / "main.pdf"
    ap.add_argument("pdf", nargs="?", type=Path, default=default_pdf)
    args = ap.parse_args()

    pdf: Path = args.pdf
    if not pdf.exists():
        print(f"error: {pdf} not found", file=sys.stderr)
        return 1

    n = total_pages(pdf)
    bib_info = find_bibliography_page(pdf, n)
    app = find_appendix_page_from_out(pdf)

    print(f"PDF                 : {pdf}")
    print(f"Total pages         : {n}")
    if bib_info is None:
        print("Bibliography page   : (not found — no 'References' heading)")
        print(f"Pre-bibliography    : {n} (entire doc)")
        return 0

    bib, shared = bib_info
    body_count = bib if shared else bib - 1
    print(f"Bibliography starts : page {bib} ({'shared with main body' if shared else 'fresh page'})")
    print(f"Main-body pages     : {body_count}   <-- compare to page limit")
    if shared:
        print(f"  (page {bib} has main-body text before References, so it counts)")
    else:
        print(f"  (pages 1-{bib - 1} are main body only)")

    if app is not None and app >= bib:
        bib_pages = app - bib
        appendix_pages = n - app + 1
        print(f"Bibliography pages  : {bib_pages} (pages {bib}-{app - 1})")
        print(f"Appendix starts     : page {app}")
        print(f"Appendix pages      : {appendix_pages} (pages {app}-{n})")
    else:
        bib_pages = n - bib + 1
        print(f"Bibliography pages  : {bib_pages} (pages {bib}-{n})")
        print("Appendix            : (not detected)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
