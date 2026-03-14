"""Precompute which games in scraped_games_increpare are unique relative to
custom_games, gallery_games, and scraped_games.

Deduplication is based on comparing stripped (whitespace-normalized) game text.

Outputs:
    data/unique_increpare_games.json  - sorted list of unique game names (no .txt)
"""
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_GAMES_DIR = os.path.join(BASE_DIR, 'custom_games')
GALLERY_GAMES_DIR = os.path.join(BASE_DIR, 'gallery_games')
GAMES_DIR = os.path.join(BASE_DIR, 'data', 'scraped_games')
INCREPARE_GAMES_DIR = os.path.join(BASE_DIR, 'data', 'scraped_games_increpare')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'unique_increpare_games.json')


def strip_text(text: str) -> str:
    """Normalize game text for comparison: lowercase, collapse whitespace."""
    return ' '.join(text.lower().split())


def collect_stripped_texts(directory: str) -> set[str]:
    """Read all .txt files in a directory and return a set of their stripped contents."""
    texts = set()
    if not os.path.isdir(directory):
        return texts
    for fname in os.listdir(directory):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                texts.add(strip_text(f.read()))
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
    return texts


def main():
    # Build the set of known game texts from the base directories
    print("Collecting texts from custom_games, gallery_games, scraped_games...")
    known_texts = set()
    for d in [CUSTOM_GAMES_DIR, GALLERY_GAMES_DIR, GAMES_DIR]:
        texts = collect_stripped_texts(d)
        print(f"  {d}: {len(texts)} files")
        known_texts |= texts

    print(f"Total known unique texts: {len(known_texts)}")

    # Now scan increpare games
    print(f"Scanning {INCREPARE_GAMES_DIR}...")
    unique_games = []
    n_dupes = 0
    n_total = 0
    n_errors = 0

    for fname in sorted(os.listdir(INCREPARE_GAMES_DIR)):
        if not fname.endswith('.txt'):
            continue
        n_total += 1
        path = os.path.join(INCREPARE_GAMES_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = strip_text(f.read())
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            n_errors += 1
            continue

        if text in known_texts:
            n_dupes += 1
        else:
            game_name = fname[:-4]  # strip .txt
            unique_games.append(game_name)
            known_texts.add(text)  # avoid intra-increpare dupes too

    unique_games.sort()

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(unique_games, f, indent=2)

    print(f"\nResults:")
    print(f"  Total increpare games:  {n_total}")
    print(f"  Duplicates (removed):   {n_dupes}")
    print(f"  Read errors:            {n_errors}")
    print(f"  Unique games:           {len(unique_games)}")
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
