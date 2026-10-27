#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-echo
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=02:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_TRANSFER_ECHO_PILOT=1
bash "$repo/experiments/dp_opt/agentic-transfer.sbatch"
