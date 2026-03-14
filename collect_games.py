"""Collect PuzzleScript game source files into dataset directories.

Usage:
    python collect_games.py gallery          # Download gallery games from PuzzleScript.net repo
    python collect_games.py pedro [--update] # Scrape gists from pedros.works (the "pedro" dataset)
    python collect_games.py dedup            # Run deduplication for increpare dataset

The increpare dataset (data/scraped_games_increpare) is assumed to already exist on disk.
"""
from argparse import ArgumentParser
import glob
import json
import os
import re
import shutil
import time

import requests
from dotenv import load_dotenv
from pathvalidate import sanitize_filename


GALLERY_GAMES_DIR = 'gallery_games'
SCRAPED_GAMES_DIR = os.path.join('data', 'scraped_games')
GAMES_DAT_PATH = os.path.join('PuzzleScript', 'src', 'games_dat.js')


def return_keyval(d, key):
    if key in d:
        return d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            return return_keyval(v, key)
    return None


def _load_games_dat():
    """Parse games_dat.js into a list of dicts with url/title/author."""
    with open(GAMES_DAT_PATH, encoding='utf-8') as f:
        text = f.read()
    json_text = text[text.index('['):text.rindex(']') + 1]
    # Fix trailing commas (invalid JSON but valid JS)
    json_text = re.sub(r',\s*\]', ']', json_text)
    json_text = re.sub(r',\s*\}', '}', json_text)
    return json.loads(json_text)


def _fetch_gist_script(gist_id, headers, max_retries=5):
    """Fetch a PuzzleScript game source from a GitHub gist. Returns script text or None."""
    git_url = f"https://api.github.com/gists/{gist_id}"
    for attempt in range(max_retries):
        try:
            response = requests.get(git_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code in (403, 429):
                # Rate limited — back off using Retry-After or exponential
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait = int(retry_after) + 1
                else:
                    wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160s
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"  Error fetching gist {gist_id}: {e}")
            return None

        gist = response.json()
        if 'script.txt' in gist['files']:
            return gist['files']['script.txt']['content']
        # Fall back to first file with content
        return return_keyval(gist['files'], 'content')

    print(f"  Failed after {max_retries} retries for gist {gist_id}")
    return None


def _title_to_filename(title):
    """Convert a game title to a safe filename (without extension)."""
    name = title.replace(' ', '_')
    return sanitize_filename(name, replacement_text='_')


def collect_gallery():
    """Collect gallery games: copy demo files, then fetch remaining gists."""
    os.makedirs(GALLERY_GAMES_DIR, exist_ok=True)
    load_dotenv()

    # Step 1: Copy demo .txt files from local PuzzleScript repo
    ps_src_demo = os.path.join('PuzzleScript', 'src', 'demo')
    if os.path.isdir(ps_src_demo):
        demos = glob.glob(os.path.join(ps_src_demo, '*.txt'))
        for eg in demos:
            dest = os.path.join(GALLERY_GAMES_DIR, os.path.basename(eg))
            shutil.copy(eg, dest)
        print(f"Copied {len(demos)} demo files from {ps_src_demo}")
    else:
        print(f"Warning: {ps_src_demo} not found, skipping demo files")

    # Step 2: Fetch gists for gallery games not already on disk
    if not os.path.isfile(GAMES_DAT_PATH):
        print(f"Warning: {GAMES_DAT_PATH} not found, skipping gist fetch")
        return

    gallery_entries = _load_games_dat()
    existing = {f[:-4] for f in os.listdir(GALLERY_GAMES_DIR) if f.endswith('.txt')}
    print(f"Gallery entries in games_dat.js: {len(gallery_entries)}")
    print(f"Already on disk: {len(existing)}")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers["Authorization"] = f"Bearer {token}"

    fetched = 0
    skipped = 0
    errors = 0

    for entry in gallery_entries:
        title = entry['title']
        filename = _title_to_filename(title)

        if filename in existing:
            skipped += 1
            continue

        url = entry.get('url', '')
        if 'p=' not in url:
            print(f"  Skipping {title}: no gist ID in URL")
            errors += 1
            continue

        gist_id = url.split('p=')[1].strip('"')
        print(f"  Fetching: {title} (gist {gist_id})")

        script = _fetch_gist_script(gist_id, headers)
        if script is None:
            errors += 1
            continue

        dest = os.path.join(GALLERY_GAMES_DIR, f"{filename}.txt")
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(script)
        existing.add(filename)
        fetched += 1

    total = len(glob.glob(os.path.join(GALLERY_GAMES_DIR, '*.txt')))
    print(f"\nGallery collection done:")
    print(f"  Fetched from gists: {fetched}")
    print(f"  Already on disk:    {skipped}")
    print(f"  Errors:             {errors}")
    print(f"  Total gallery games: {total}")


def collect_pedro(update=False):
    """Scrape PuzzleScript gists referenced by pedros.works."""
    os.makedirs(SCRAPED_GAMES_DIR, exist_ok=True)
    load_dotenv()

    ps_urls_path = os.path.join("data", "ps_urls.txt")
    if update or not os.path.isfile(ps_urls_path):
        js_url = "https://pedros.works/puzzlescript/hyper/PGDGame.js"
        response = requests.get(js_url)
        response.raise_for_status()

        game_links = re.findall(r'https?://\S+', response.text)
        print(response.text)
        [print(g) for g in game_links]
        print(f"Total links: {len(game_links)}")

        ps_links = [g for g in game_links if "puzzlescript.net/play" in g]
        ps_links = set(ps_links)
        print(f"Total PuzzleScript links: {len(ps_links)}")

        with open(ps_urls_path, "w") as f:
            f.write("\n".join(ps_links))
    else:
        with open(ps_urls_path, "r") as f:
            ps_links = f.read().splitlines()

    visited_ps_links_path = "data/visited_ps_links.txt"
    if os.path.isfile(visited_ps_links_path) and not update:
        with open(visited_ps_links_path, "r") as f:
            visited_ps_links = set(f.read().splitlines())
    else:
        with open(visited_ps_links_path, "w") as f:
            f.write("")
            visited_ps_links = []

    for link in ps_links:
        if link in visited_ps_links:
            print(f"Skipping {link}")
            continue
        if 'play' not in link:
            breakpoint()
        else:
            hack_link = link.replace('play.html?p=', 'editor.html?hack=')
            gist_id = hack_link.split('hack=')[1].strip('"')
            print(f'hack link: {hack_link}')
            git_url = f"https://api.github.com/gists/{gist_id}"

            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                "X-GitHub-Api-Version": "2022-11-28"
            }

            def add_to_visited(filename):
                with open(visited_ps_links_path, "a", encoding='utf-8') as f:
                    f.write(f"{link}, {filename}\n")

            try:
                response = requests.get(git_url, headers=headers)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"Error: {e}")
                add_to_visited(filename="error.txt")
                continue

            gist = response.json()
            if 'script.txt' not in gist['files']:
                script = return_keyval(gist['files'], 'content')
            else:
                script = gist['files']['script.txt']['content']

            title_match = re.search(r'(?i)title (.+)', script)
            if not title_match:
                breakpoint()
            title = title_match.groups()[0]
            title = title.replace(' ', '_')
            filename = sanitize_filename(title, replacement_text='_')
            script_path = os.path.join(SCRAPED_GAMES_DIR, filename)

            dupe_filenames = {}

            if os.path.isfile(script_path):
                if filename not in dupe_filenames:
                    n_prev_dupes = 2
                    dupe_filenames[filename] = n_prev_dupes
                else:
                    n_prev_dupes += 1
                    dupe_filenames[filename] += n_prev_dupes
                filename = f'{filename}_{n_prev_dupes}'
            else:
                filename = str(filename)
            filename += '.txt'
            script_path = os.path.join(SCRAPED_GAMES_DIR, filename)

            with open(script_path, "w", encoding='utf-8') as f:
                f.write(script)

            add_to_visited(filename=filename)

    script_files = os.listdir(SCRAPED_GAMES_DIR)
    print(f"Total pedro scripts: {len(script_files)}")


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('gallery', help='Collect gallery games')

    pedro_parser = subparsers.add_parser('pedro', help='Scrape pedro dataset from gists')
    pedro_parser.add_argument("--update", action="store_true",
                              help="Re-fetch the list of PuzzleScript URLs")

    subparsers.add_parser('dedup', help='Run dedup_games.py to compute unique increpare games')

    args = parser.parse_args()

    if args.command == 'gallery':
        collect_gallery()
    elif args.command == 'pedro':
        collect_pedro(update=args.update)
    elif args.command == 'dedup':
        import dedup_games
        dedup_games.main()
