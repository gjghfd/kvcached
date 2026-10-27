#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-admission-cost
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=02:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_ADMISSION_COST_PILOT=1
export DP_OPT_KV_TRANSPORT=dp_echo
bash "$repo/experiments/dp_opt/agentic-transfer.sbatch"
