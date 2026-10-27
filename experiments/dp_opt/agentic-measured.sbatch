#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-measured
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
set -euo pipefail
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_MEASURED_CONTROL=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MEASURED_CONTROL_ENV="${DP_OPT_MEASURED_CONTROL_ENV:?Supply frozen measured control env}"
export DP_OPT_MEASURED_CONTROL_ENV_SHA256="${DP_OPT_MEASURED_CONTROL_ENV_SHA256:?Freeze control env SHA256}"
bash "$repo/experiments/dp_opt/agentic-formal.sbatch"
