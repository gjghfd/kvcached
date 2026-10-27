#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-revalidated
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_REVALIDATE_PREPARED=1
bash "$repo/experiments/dp_opt/agentic-measured.sbatch"
