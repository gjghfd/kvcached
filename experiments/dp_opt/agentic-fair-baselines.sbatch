#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-fair-baselines
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_TRACE_POLICIES="${DP_OPT_TRACE_POLICIES:-no_move no_move_fair own_model kv_only ilp}"
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S="${DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S:-30}"
bash "$repo/experiments/dp_opt/agentic-revalidated.sbatch"
