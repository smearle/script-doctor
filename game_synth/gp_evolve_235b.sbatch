#!/bin/bash
#SBATCH --job-name=gp_evolve_235b
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=5:00:00
#SBATCH --requeue
#SBATCH --output=game_synth/gp_evolve_235b_%j.out
#SBATCH --error=game_synth/gp_evolve_235b_%j.out

# Self-contained: serve Qwen3-VL-235B (TP=4, mirrors picbreeder daemon args) for
# the duration of this job, then run GP evolution with it as the LLM mutation
# operator. WM runs on CPU to avoid contending with vLLM's GPUs.
set -uo pipefail
cd /scratch/se2161/script-doctor
export HF_HOME=/scratch/se2161/hf_cache
export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
PORT=8765
BASE="http://127.0.0.1:${PORT}/v1"

echo "[job] launching vLLM Qwen3-VL-235B (TP=4) on :${PORT}"
# vLLM lives in picbreeder-vlm's venv (script-doctor's venv has no vllm); model
# weights are cached under the shared HF_HOME so load is not a fresh download.
VLLM_PY=/scratch/se2161/picbreeder-vlm/.venv/bin/python
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --port "${PORT}" --host 127.0.0.1 --trust-remote-code \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --tensor-parallel-size 4 --max-num-seqs 16 \
  > game_synth/vllm_235b_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT

echo "[job] waiting for vLLM to load (up to ~45 min)…"
for i in $(seq 1 270); do
  if curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "[job] vLLM ready after $((i*10))s"; READY=1; break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "[job] vLLM died; see vllm log"; tail -20 game_synth/vllm_235b_${SLURM_JOB_ID}.log; exit 1; fi
  sleep 10
done
[ "${READY:-0}" = 1 ] || { echo "[job] vLLM did not become ready"; exit 1; }
MODEL_ID=$(curl -s "http://127.0.0.1:${PORT}/v1/models" | .venv/bin/python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
echo "[job] served model: $MODEL_ID"

.venv/bin/python -u -m game_synth.gp_evolve \
  --operator mixed --llm-frac 0.5 \
  --llm-backend endpoint --llm-model "$MODEL_ID" --llm-base-url "$BASE" \
  --device cpu \
  --fitness loss --no-random --max-rules 8 \
  --gens 15 --pop 60 --offspring 10 --updates-per-gen 150 \
  --out game_synth/gp_evolve_235b_run
echo "[job] DONE"
