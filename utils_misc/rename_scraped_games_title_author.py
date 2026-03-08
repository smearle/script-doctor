import json
import os
import re
from typing import Dict, List, Tuple


SCRAPED_GAMES_DIR = os.path.join("data", "scraped_games")
COLLISIONS_JSON_PATH = os.path.join("data", "scraped_games_rename_collisions.json")

SECTION_HEADERS = {
    "OBJECTS",
    "LEGEND",
    "SOUNDS",
    "COLLISIONLAYERS",
    "RULES",
    "WINCONDITIONS",
    "LEVELS",
}


def _sanitize_component(value: str, fallback: str) -> str:
    value = value.strip()
    if not value:
        value = fallback
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]", "", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._-")
    return value or fallback


def _extract_title_author(content: str, fallback_title: str) -> Tuple[str, str]:
    title = None
    author = None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper() in SECTION_HEADERS:
            break
        if stripped.startswith("====="):
            continue
        if title is None:
            match = re.match(r"^title\s+(.+)$", stripped, flags=re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                continue
        if author is None:
            match = re.match(r"^author\s+(.+)$", stripped, flags=re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                continue
    title = title or fallback_title
    author = author or "Unknown"
    return title, author


def _compute_targets(files: List[str]) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, str]]]]:
    targets: Dict[str, str] = {}
    collisions: Dict[str, List[Dict[str, str]]] = {}
    base_counts: Dict[str, int] = {}

    existing_filenames = set(os.listdir(SCRAPED_GAMES_DIR))

    for filename in files:
        src_path = os.path.join(SCRAPED_GAMES_DIR, filename)
        if not os.path.isfile(src_path):
            continue
        if not filename.lower().endswith(".txt"):
            continue
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        fallback_title = os.path.splitext(filename)[0]
        title, author = _extract_title_author(content, fallback_title=fallback_title)
        safe_title = _sanitize_component(title, fallback="untitled")
        safe_author = _sanitize_component(author, fallback="unknown")

        base = f"{safe_title}_by_{safe_author}"
        count = base_counts.get(base, 0) + 1
        base_counts[base] = count
        suffix = "" if count == 1 else f"_{count}"
        target_name = f"{base}{suffix}.txt"

        while target_name in existing_filenames and target_name != filename:
            count += 1
            base_counts[base] = count
            suffix = f"_{count}"
            target_name = f"{base}{suffix}.txt"

        targets[src_path] = os.path.join(SCRAPED_GAMES_DIR, target_name)
        existing_filenames.add(target_name)

        if count > 1 or target_name != f"{base}.txt":
            collisions.setdefault(base, []).append(
                {
                    "source": src_path,
                    "target": os.path.join(SCRAPED_GAMES_DIR, target_name),
                }
            )

    return targets, collisions


def _rename_with_temp(targets: Dict[str, str]) -> None:
    temp_paths = {}
    for idx, (src, _) in enumerate(targets.items()):
        temp_name = f".__rename_tmp__{idx}__.txt"
        temp_path = os.path.join(SCRAPED_GAMES_DIR, temp_name)
        os.rename(src, temp_path)
        temp_paths[src] = temp_path

    for src, dest in targets.items():
        temp_path = temp_paths[src]
        if temp_path == dest:
            continue
        os.rename(temp_path, dest)


def main() -> None:
    files = sorted(os.listdir(SCRAPED_GAMES_DIR))
    targets, collisions = _compute_targets(files)

    _rename_with_temp(targets)

    with open(COLLISIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(collisions, f, indent=2, ensure_ascii=False)

    print(f"Renamed {len(targets)} files in {SCRAPED_GAMES_DIR}.")
    print(f"Collision report saved to {COLLISIONS_JSON_PATH}.")


if __name__ == "__main__":
    main()
