#!/bin/bash
#SBATCH --job-name=gp_evolve_llm235b
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon,h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=game_synth/gp_evolve_llm_%j.out
#SBATCH --error=game_synth/gp_evolve_llm_%j.out

# GP evolution of PuzzleScript games with WM-loss fitness, using the live
# Qwen3-VL-235B vLLM daemon (../picbreeder-vlm) as the LLM mutation operator.
set -uo pipefail
cd /scratch/se2161/script-doctor
export HF_HOME=/scratch/se2161/hf_cache
export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH

BASE_URL=$(cat /scratch/se2161/picbreeder-vlm/vllm_daemon.endpoint)
echo "[gp_evolve] daemon endpoint: $BASE_URL"
MODEL_ID=$(.venv/bin/python - "$BASE_URL" <<'PY'
import sys, urllib.request, json
print(json.load(urllib.request.urlopen(sys.argv[1] + "/models", timeout=30))["data"][0]["id"])
PY
) || { echo "DAEMON UNREACHABLE"; exit 1; }
echo "[gp_evolve] served model id: $MODEL_ID"

.venv/bin/python -u -m game_synth.gp_evolve \
  --operator mixed --llm-frac 0.5 \
  --llm-backend endpoint \
  --llm-model "$MODEL_ID" \
  --llm-base-url "$BASE_URL" \
  --gens 20 --pop 80 --offspring 12 --updates-per-gen 200 \
  --out game_synth/gp_evolve_llm235b
echo "[gp_evolve] DONE"
