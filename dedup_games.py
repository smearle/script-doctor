"""Precompute unique game lists for each dataset tier.

Each tier is deduped against all preceding tiers AND within itself (so if
gallery_games/ contains two differently-named files with identical content,
only the first alphabetically is kept).

Deduplication is based on stripped (whitespace-normalized, lowercased) game text.

The dedup chain mirrors the logical tier ordering:

    custom_games/  (seed — always trusted, never deduped)
      ↓
    gallery_games/           → unique_games["gallery"]
      ↓
    data/scraped_games/      → unique_games["pedro"]
      ↓
    data/scraped_games_increpare/ → unique_games["increpare"]

Outputs:
    data/unique_games_per_dataset.json  -  {"gallery": [...], "pedro": [...], "increpare": [...]}
"""
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_GAMES_DIR = os.path.join(BASE_DIR, 'custom_games')
GALLERY_GAMES_DIR = os.path.join(BASE_DIR, 'gallery_games')
GAMES_DIR = os.path.join(BASE_DIR, 'data', 'scraped_games')
INCREPARE_GAMES_DIR = os.path.join(BASE_DIR, 'data', 'scraped_games_increpare')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'unique_games_per_dataset.json')


def strip_text(text: str) -> str:
    """Normalize game text for comparison: lowercase, collapse whitespace."""
    return ' '.join(text.lower().split())


def _read_stripped(path: str) -> str | None:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return strip_text(f.read())
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def dedup_directory(directory: str, known_texts: set[str]) -> list[str]:
    """Return sorted unique game names from directory, excluding known texts and intra-dir dupes.

    Adds newly seen texts to known_texts in place.
    """
    if not os.path.isdir(directory):
        return []

    unique = []
    n_total = 0
    n_dupes = 0

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.txt'):
            continue
        n_total += 1
        text = _read_stripped(os.path.join(directory, fname))
        if text is None:
            continue
        if text in known_texts:
            n_dupes += 1
        else:
            unique.append(fname[:-4])
            known_texts.add(text)

    print(f"  {directory}: {n_total} files, {n_dupes} dupes removed, {len(unique)} unique")
    return sorted(unique)


def main():
    known_texts: set[str] = set()

    # Seed with custom_games (always trusted, never filtered)
    print("Seeding with custom_games/...")
    if os.path.isdir(CUSTOM_GAMES_DIR):
        for fname in os.listdir(CUSTOM_GAMES_DIR):
            if not fname.endswith('.txt'):
                continue
            text = _read_stripped(os.path.join(CUSTOM_GAMES_DIR, fname))
            if text is not None:
                known_texts.add(text)
    print(f"  {len(known_texts)} seed texts from custom_games/")

    result = {}

    print("\nDeduplicating gallery_games/...")
    result["gallery"] = dedup_directory(GALLERY_GAMES_DIR, known_texts)

    print("\nDeduplicating data/scraped_games/...")
    result["pedro"] = dedup_directory(GAMES_DIR, known_texts)

    print("\nDeduplicating data/scraped_games_increpare/...")
    result["increpare"] = dedup_directory(INCREPARE_GAMES_DIR, known_texts)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSummary:")
    for tier, games in result.items():
        print(f"  {tier:>10}: {len(games)} unique games")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
