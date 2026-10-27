#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-transfer
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=02:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_TRANSFER_PILOT=1 DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_KV_TRANSPORT="${DP_OPT_KV_TRANSPORT:-host_file}"
bash "$repo/experiments/dp_opt/agentic-trace.sbatch"
