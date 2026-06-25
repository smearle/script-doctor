#!/bin/bash
# Submit an Autumn WM training sweep on torch: one SLURM job per (GAME, POOL, NHID, SEED) cell.
# Run ON torch (ssh torch) after rsyncing the {game}_train_seq.npz data + code.
#
# Usage:  GAMES="nim lights logic_gates" POOLS="mean meanmax" NHIDS="96" SEEDS="0 1" \
#           UPDATES=8000 bash nca_wm/scripts/torch_autumn_submit.sh
set -u
GAMES=${GAMES:?set GAMES (space list)}
POOLS=${POOLS:-meanmax}
NHIDS=${NHIDS:-96}
SEEDS=${SEEDS:-0}
UPDATES=${UPDATES:-8000}

n=0
for g in $GAMES; do
  if [ ! -f "nca_wm/autumn/data/${g}_train_seq.npz" ]; then
    echo "SKIP $g (no data)"; continue
  fi
  for pool in $POOLS; do
    for nhid in $NHIDS; do
      for seed in $SEEDS; do
        tag=${pool}_h${nhid}_s${seed}
        sbatch --export=ALL,GAME=$g,POOL=$pool,NHID=$nhid,SEED=$seed,UPDATES=$UPDATES,TAG=$tag \
          --job-name=aut_${g}_${tag} nca_wm/scripts/torch_autumn_train.sbatch
        n=$((n+1))
      done
    done
  done
done
echo "submitted $n jobs"
