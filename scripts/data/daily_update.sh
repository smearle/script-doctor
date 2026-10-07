#!/usr/bin/env bash
# Daily PuzzleScript gist dataset sweeper.
#
# Discovers newly-published gists, (weekly) enumerates the full repertoire of every
# known + newly-discovered author, folds them into the local master, then re-derives
# the deduped/vanilla view and refreshes the public HF dataset — but only if the
# corpus actually grew. Runs on this box because it needs local state the public
# dataset doesn't carry: the owner manifest (to seed author enumeration), the
# incremental dedup cache, and the GitHub token.
#
# Install (daily 04:00, no overlapping runs):
#   crontab -l | { cat; echo "0 4 * * * flock -n /tmp/ps_daily.lock /home/jupyter-smearle/script-doctor/scripts/data/daily_update.sh"; } | crontab -
set -uo pipefail

ROOT=/home/jupyter-smearle/script-doctor
MASTER=/home/jupyter-smearle/puzzlescript-gists
P="$ROOT/.venv/bin/python3"
cd "$ROOT" || exit 1
export GITHUB_TOKEN=$(grep -m1 oauth_token /home/jupyter-smearle/.config/gh/hosts.yml | awk '{print $2}')

mkdir -p logs
exec >>"logs/daily_update_$(date +%F).log" 2>&1
echo "===== daily_update $(date -u +%FT%TZ) ====="

before=$(find "$MASTER" -maxdepth 1 -name '*.txt' | wc -l)

# 1. Discovery: recent-gist marker search (early-stops once it hits seen pages).
( cd puzzlescript-analysis && "$P" -u trawl_gists_html.py --max-pages 100 ) || echo "trawl failed"

# 2. Weekly (Sundays), or on every run with FULL=1: the slow-moving discovery sources
#    (PuzzleScript Google Group, Internet Archive play links, newest itch.io
#    PuzzleScript games), then the author-enum closure — the heavy step. Catches new
#    authors' full repertoire + existing authors' newly-published games. Consolidate
#    first so owners found by those sources seed the enumeration. (The pedrosworks.com
#    game list stopped being served in Oct 2026; its last copy, from 2026-06-17, is
#    already in the staging dir.)
if [ "$(date +%u)" = "7" ] || [ "${FULL:-0}" = "1" ]; then
  "$P" scripts/data/build_ps_dataset.py forum || echo "forum failed"
  "$P" scripts/data/build_ps_dataset.py wayback || echo "wayback failed"
  "$P" scripts/data/scrape_itch.py --listing https://itch.io/games/newest/made-with-puzzlescript \
    --pages 5 --skip-known || echo "itch failed"
  "$P" scripts/data/build_ps_dataset.py consolidate || echo "consolidate failed"
  "$P" scripts/data/build_ps_dataset.py authors || echo "authors failed"
fi

# 3. Fold everything into the master (order matters: consolidate rewrites the
#    gist-keyed manifest, then reconcile appends variants, then backfill links).
"$P" scripts/data/build_ps_dataset.py consolidate || { echo "consolidate failed"; exit 1; }
"$P" scripts/data/build_ps_dataset.py reconcile --add || echo "reconcile failed"
"$P" scripts/data/backfill_fallback_gists.py || echo "backfill failed"

after=$(find "$MASTER" -maxdepth 1 -name '*.txt' | wc -l)
echo "master: $before -> $after"

# 4. Refresh the public HF dataset only if the corpus grew.
if [ "$after" -gt "$before" ]; then
  "$P" -m nca_wm.scripts.detect_non_vanilla --master-dir "$MASTER" --emit-removal-list /tmp/ps_plus_remove.txt
  "$P" -m nca_wm.scripts.dedup_master --master-dir "$MASTER" --workers 16
  rm -rf /tmp/hf_puzzlescript
  "$P" nca_wm/scripts/build_hf_dataset.py
  if "$P" - <<PY
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="/tmp/hf_puzzlescript", repo_id="smearle/puzzlescript-gists",
                      repo_type="dataset", commit_message="Daily refresh: ${after} games")
print("HF refreshed")
PY
  then
    echo "HF refreshed ($after games)"
    rm -rf /tmp/hf_puzzlescript  # staging copy is rebuilt every run; 209's disk is tight
  else
    echo "HF push FAILED; staging kept at /tmp/hf_puzzlescript"
  fi
else
  echo "no new games; HF push skipped"
fi
echo "===== done $(date -u +%FT%TZ) ====="
