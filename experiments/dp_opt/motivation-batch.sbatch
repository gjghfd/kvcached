#!/usr/bin/env bash
#SBATCH --job-name=qwen-dp-batch
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=02:30:00
#SBATCH --time-min=02:00:00
export DP_OPT_BENCHMARK_MODE=motivation DP_OPT_BATCH_SUITE=1
export DP_OPT_MAX_MODEL_LEN=33792 DP_OPT_KV_CACHE_BYTES=47244640256
export DP_OPT_MAX_NUM_SEQS=256 DP_OPT_MAX_BATCHED_TOKENS=4096
exec bash "${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}/experiments/dp_opt/run-dual-qwen.sh"
