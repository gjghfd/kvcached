#!/usr/bin/env bash
#SBATCH --job-name=qwen-kvonly-stress
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=16:00:00
set -euo pipefail
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
evidence="$root/state/live-trace-$SLURM_JOB_ID"
mkdir -p "$evidence"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$evidence/gpu-memory-before-load.csv"
if ! awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { invalid=1 }
    END { exit (NR != 2 || invalid) }' "$evidence/gpu-memory-before-load.csv"; then
    printf 'Allocated pair lacks required free A800 memory.\n' >&2
    printf '1\n' > "$evidence/exit-status"
    exit 1
fi
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
export DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0 DP_OPT_FROZEN_TRACE_REPLAY=0
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_TRANSFER_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0
export DP_OPT_STRESS_SWEEP=1
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776
export DP_OPT_TRACE_BUDGET_MIB=20480 DP_OPT_REQUEST_TIMEOUT_S=900 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1 DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1
export DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8
export DP_OPT_TRACE_POLICIES="no_move no_move_fair own_model kv_only ilp"
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_TRACE_PATH="$root/state/agentic-100k-input/trace.json"
export DP_OPT_PROMPT_BANK="$root/state/agentic-100k-input/prompt-bank.json"
bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
