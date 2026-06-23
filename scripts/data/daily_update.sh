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

# 2. Author-enum closure — the heavy step, so weekly (Sundays). Catches new
#    authors' full repertoire + existing authors' newly-published games.
if [ "$(date +%u)" = "7" ]; then
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
  "$P" - <<PY
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="/tmp/hf_puzzlescript", repo_id="smearle/puzzlescript-gists",
                      repo_type="dataset", commit_message="Daily refresh: ${after} games")
print("HF refreshed")
PY
  echo "HF refreshed ($after games)"
else
  echo "no new games; HF push skipped"
fi
echo "===== done $(date -u +%FT%TZ) ====="
