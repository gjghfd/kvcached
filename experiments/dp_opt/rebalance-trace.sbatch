#!/usr/bin/env bash
#SBATCH --job-name=qwen-kv-rebalance
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=20:00:00
set -euo pipefail
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export DP_OPT_TRACE_PATH="${DP_OPT_REBALANCE_TRACE_PATH:?Supply the independently frozen paired trace}"
test "$(sha256sum "$DP_OPT_TRACE_PATH" | awk '{print $1}')" = "${DP_OPT_REBALANCE_TRACE_SHA256:?Freeze trace SHA256}"
export DP_OPT_MOVE_FIXED_S="${DP_OPT_MOVE_FIXED_S:?Freeze measured fixed control cost}"
export DP_OPT_MOVE_BANDWIDTH_GIB_S="${DP_OPT_MOVE_BANDWIDTH_GIB_S:?Freeze measured copy bandwidth}"
export DP_OPT_PROMPT_BANK="${DP_OPT_PROMPT_BANK:-$root/state/agentic-100k-input/prompt-bank.json}"
export DP_OPT_TRACE_BUDGET_MIB="${DP_OPT_REBALANCE_BUDGET_MIB:?Use the same frozen budget for every policy}"
export DP_OPT_STRESS_SWEEP=0 DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_TRANSFER_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8
export DP_OPT_REQUEST_TIMEOUT_S=900 DP_OPT_DEADLINE_S=120 DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_TRACE_POLICIES="${DP_OPT_TRACE_POLICIES:-no_move no_move_fair own_model kv_only ilp kv_only_rebalance}"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
evidence="$root/state/live-trace-$SLURM_JOB_ID"
mkdir -p "$evidence"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$evidence/gpu-memory-before-load.csv"
if ! awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { invalid=1 }
    END { exit (NR != 2 || invalid) }' "$evidence/gpu-memory-before-load.csv"; then
    printf 'Allocated pair lacks required free A800 memory.\n' >&2
    printf '1\n' > "$evidence/exit-status"
    exit 1
fi
bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
