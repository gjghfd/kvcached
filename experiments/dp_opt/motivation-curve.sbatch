#!/usr/bin/env bash
#SBATCH --job-name=qwen-dp-curve
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=01:30:00
#SBATCH --time-min=01:00:00
set -euo pipefail
export DP_OPT_BENCHMARK_MODE=motivation DP_OPT_GRAPH_CURVE=1 DP_OPT_EXECUTION_MODE=graph
export DP_OPT_MAX_MODEL_LEN=4096 DP_OPT_KV_CACHE_BYTES=47244640256
export DP_OPT_MAX_NUM_SEQS=256 DP_OPT_MAX_BATCHED_TOKENS=4096
export DP_OPT_RUN_LABEL=sparse
export DP_OPT_CAPTURE_SIZES=1,8,16,32,64,128,256
bash "${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}/experiments/dp_opt/run-dual-qwen.sh"
