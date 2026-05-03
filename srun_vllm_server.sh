#!/usr/bin/env bash
set -euo pipefail

# Override these at launch time as needed, e.g.
#   VLLM_MODEL=google/gemma-4-26B-A4B-it N_GENS=5 ./srun_vllm_server.sh
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
CURRICULUM_MODEL="${CURRICULUM_MODEL:-$VLLM_MODEL}"
PORT="${PORT:-8000}"
SEED_GAMES="${SEED_GAMES:-scaling_4}"
SET_SIZE="${SET_SIZE:-4}"
POP_SIZE="${POP_SIZE:-4}"
N_GENS="${N_GENS:-3}"
N_CHILDREN="${N_CHILDREN:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_GRES="${GPU_GRES:-gpu:1}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-8:00:00}"

srun --gres="$GPU_GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME_LIMIT" --pty bash -lc "
    source .venv/bin/activate
    python -m vllm.entrypoints.openai.api_server \\
        --model '$VLLM_MODEL' \\
        --port '$PORT' \\
        --max-model-len '$MAX_MODEL_LEN' &
    server_pid=\$!
    trap 'kill \$server_pid 2>/dev/null || true' EXIT
    sleep 30
    python -m nca_wm.game_curriculum \\
        --model '$CURRICULUM_MODEL' \\
        --vllm_base_url 'http://localhost:$PORT/v1' \\
        --seed_games '$SEED_GAMES' \\
        --set_size '$SET_SIZE' \\
        --pop_size '$POP_SIZE' \\
        --n_generations '$N_GENS' \\
        --n_children_per_gen '$N_CHILDREN'
"
