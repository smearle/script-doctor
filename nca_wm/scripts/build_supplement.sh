#!/usr/bin/env bash
# Build the anonymized supplemental zip for NeurIPS submission.
#
# Run from the repo root:
#   bash nca_wm/scripts/build_supplement.sh
#
# Output: ../nca_wm_supplement.zip (one level above the repo, to keep it out
# of the next bundle).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

STAGE_DIR="$(mktemp -d -t nca_wm_supplement_XXXXXX)"
OUT_ROOT="$STAGE_DIR/nca_wm_supplement"
OUT_ZIP="${REPO_ROOT}/../nca_wm_supplement.zip"

echo "[1/5] staging code into $OUT_ROOT"
mkdir -p "$OUT_ROOT"

# Whitelist of paths to copy verbatim from the repo. Everything else is
# excluded by default; nothing outside this list ends up in the zip.
PATHS=(
  "nca_wm"
  "puzzlescript_jax"
  "puzzlescript_cpp"
  "backends"
  "conf"
  "gallery_games"
  "custom_games"
  "data/scraped_games"
  "evolve_level_cpp.py"
  "evolve_games_agentic.py"
)

# rsync filters: prune logs / build artifacts / caches / VCS metadata.
RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.gitignore"
  "--exclude=.gitattributes"
  "--exclude=.github"
  "--exclude=__pycache__"
  "--exclude=*.pyc"
  "--exclude=.DS_Store"
  "--exclude=wandb"
  "--exclude=outputs"
  "--exclude=multirun"
  "--exclude=nca_wm/logs"
  "--exclude=nca_wm/logs_*"
  "--exclude=nca_wm/sweep_launch_logs"
  "--exclude=nca_wm/plots"
  "--exclude=nca_wm/figures/*.gif"
  # Internal narrative docs (likely contain identifying paths or
  # team-internal context; not needed for reproducing the paper).
  "--exclude=nca_wm/SYNTH_HANDOFF.md"
  "--exclude=nca_wm/RUNNING_REPORT.md"
  "--exclude=nca_wm/SCALING_REPORT.md"
  "--exclude=nca_wm/SCALING_RESULTS.md"
  "--exclude=nca_wm/ARCHITECTURE_REPORT.md"
  "--exclude=nca_wm/GAME_CURRICULUM_REPORT.md"
  "--exclude=nca_wm/RULE_GP_DESIGN.md"
  "--exclude=nca_wm/token_ablation_summary.md"
  # All .log / .out files anywhere are stdout dumps or LaTeX build
  # artifacts (handled below via find too, but exclude here so they
  # never reach staging).
  "--exclude=*.log"
  "--exclude=*.out"
  # Paper LaTeX build artifacts (keep .tex, .bib, .pdf, .sty, figures)
  "--exclude=nca_wm/paper/*.aux"
  "--exclude=nca_wm/paper/*.fls"
  "--exclude=nca_wm/paper/*.fdb_latexmk"
  "--exclude=nca_wm/paper/*.bbl"
  "--exclude=nca_wm/paper/*.blg"
  "--exclude=nca_wm/paper/*.synctex.gz"
  "--exclude=nca_wm/paper/*.toc"
)

for p in "${PATHS[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "  warning: $p missing, skipping" >&2
    continue
  fi
  rsync -a --quiet "${RSYNC_EXCLUDES[@]}" --relative "./$p" "$OUT_ROOT/"
done

# Drop the build script itself from the supplement (no need to ship it).
rm -f "$OUT_ROOT/nca_wm/scripts/build_supplement.sh"

echo "[2/5] scrubbing identifying strings"

# Replace absolute repo path with a placeholder users set themselves, plus
# anonymize author tokens. Restrict to text files.
SCRUB_TARGETS=()
while IFS= read -r -d '' f; do
  SCRUB_TARGETS+=("$f")
done < <(find "$OUT_ROOT" -type f \( \
    -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.tex" -o \
    -name "*.bib" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" -o \
    -name "*.json" -o -name "*.txt" -o -name "Makefile" -o \
    -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \
  \) -print0)

# Path & identity scrubbing. Use ERE so we can use a single rule that
# catches both the canonical `/home/jupyter-smearle/...` and the typo'd
# `/home/jupyter-earle/...` form that appears in some launchers.
#
#   /home/jupyter-<anything>/script-doctor -> ${REPO}
#   /home/jupyter-<anything>               -> ${HOME}
#   smearle93@gmail.com                    -> anonymous@example.com
#   Sam Earle                              -> Anonymous Author
#   smearle93 / smearle                    -> anonymous
sed -i -E \
  -e 's|/home/jupyter-[A-Za-z0-9_-]+/script-doctor|${REPO}|g' \
  -e 's|/home/jupyter-[A-Za-z0-9_-]+|${HOME}|g' \
  -e 's|smearle93@gmail\.com|anonymous@example.com|g' \
  -e 's|Sam Earle|Anonymous Author|g' \
  -e 's|smearle93|anonymous|g' \
  -e 's|smearle|anonymous|g' \
  "${SCRUB_TARGETS[@]}"

# Verify nothing slipped through. We restrict to identifying tokens (not
# @gmail in general — many bundled human-authored games legitimately
# contain @gmail addresses in their author messages, which we keep).
LEAKS="$(grep -rIl -E 'smearle|Sam Earle|/home/jupyter-' "$OUT_ROOT" || true)"
if [[ -n "$LEAKS" ]]; then
  echo "  ERROR: identifying strings still present in:" >&2
  echo "$LEAKS" >&2
  exit 1
fi

echo "[3/5] writing top-level README and LICENSE-PLACEHOLDER"

# Top-level README: the SUPPLEMENT_README.md from nca_wm/ becomes README.md
if [[ -f "$OUT_ROOT/nca_wm/SUPPLEMENT_README.md" ]]; then
  mv "$OUT_ROOT/nca_wm/SUPPLEMENT_README.md" "$OUT_ROOT/README.md"
fi

cat > "$OUT_ROOT/LICENSE-PLACEHOLDER.txt" <<'EOF'
The released code, training launchers, and result aggregators in this
supplement will be distributed under a permissive open-source license
(e.g. MIT or Apache 2.0) upon de-anonymization. The PuzzleScript engine
(puzzlescript_cpp/, puzzlescript_jax/) and the human-authored game
corpus (gallery_games/, custom_games/, data/scraped_games/) are
derivatives of upstream projects with their own licenses; see the cited
PuzzleJax dataset paper for provenance. License files will accompany
the camera-ready release.
EOF

echo "[4/5] zipping"
rm -f "$OUT_ZIP"
( cd "$STAGE_DIR" && zip -qr "$OUT_ZIP" "nca_wm_supplement" )

echo "[5/5] cleanup"
rm -rf "$STAGE_DIR"

ls -lh "$OUT_ZIP"
echo "done. zip at: $OUT_ZIP"
