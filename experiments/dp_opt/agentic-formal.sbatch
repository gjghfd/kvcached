#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-formal
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
set -euo pipefail
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_FROZEN_TRACE_REPLAY=1
export DP_OPT_FROZEN_TRACE_PATH="$root/state/live-trace-241086/frozen/trace.json"
export DP_OPT_FROZEN_TRACE_SHA256=6d43e314ddf3b7a64c883dc19bcdefb1210bb655b8c8d48c5af56580d4f1c006
export DP_OPT_FROZEN_RATE_DECISION_SHA256=7f1d0527dae71a27bf5e34bab74726e849760adac8427d01ebf41aac3d7df7cc
export DP_OPT_FROZEN_BUDGET_MIB=11328
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8
bash "$repo/experiments/dp_opt/agentic-trace.sbatch"
