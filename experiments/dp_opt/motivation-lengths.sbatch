#!/usr/bin/env bash
#SBATCH --job-name=qwen-dp-lengths
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=01:30:00
#SBATCH --time-min=01:00:00
export DP_OPT_BENCHMARK_MODE=motivation DP_OPT_LENGTH_SUITE=1
export DP_OPT_MAX_MODEL_LEN=10240 DP_OPT_KV_CACHE_BYTES=25769803776
exec bash "${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}/experiments/dp_opt/run-dual-qwen.sh"
