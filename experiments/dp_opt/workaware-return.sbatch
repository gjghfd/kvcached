#!/usr/bin/env bash
#SBATCH --job-name=qwen-workaware-return
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=20:00:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$repo/experiments/dp_opt:$DEPLOY_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
evidence="$root/state/live-trace-$SLURM_JOB_ID"
mkdir -p "$evidence"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$evidence/exit-status"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=1600 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0
export DP_OPT_REBALANCE_MAX_MOVES=4 DP_OPT_REBALANCE_CANDIDATE_LIMIT=8
export DP_OPT_REBALANCE_MIN_GAIN_S=1 DP_OPT_REBALANCE_COOLDOWN_S=10 DP_OPT_REBALANCE_RETRY_S=2
export DP_OPT_PROMPT_BANK="$root/state/agentic-100k-input/prompt-bank.json"
export DP_OPT_TRACE_PATH="${DP_OPT_WORKAWARE_TRACE_PATH:?Supply frozen low-Code-rate trace}"
test "$(sha256sum "$DP_OPT_TRACE_PATH" | awk '{print $1}')" = "${DP_OPT_WORKAWARE_TRACE_SHA256:?Freeze trace SHA}"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
unset DP_OPT_WORKAWARE_FIT DP_OPT_WORKAWARE_FIT_SHA256 DP_OPT_WORKAWARE_SOURCE_ROOT
export DP_OPT_WORKAWARE_FIXTURE_ROOT="$evidence/frozen"
mkdir -p "$DP_OPT_WORKAWARE_FIXTURE_ROOT"
for population in 2 9; do
    python "$repo/experiments/dp_opt/workaware_fixture.py" --inputs "$root/state/agentic-100k-input" \
        --population "$population" --output "$DP_OPT_WORKAWARE_FIXTURE_ROOT/recovery-$population.json"
done
sha256sum "$DP_OPT_WORKAWARE_FIXTURE_ROOT"/recovery-*.json > "$evidence/fixture.sha256"
phase() {
    name="$1"
    export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence/$name"
    mkdir -p "$DP_OPT_PHASE_EVIDENCE_ROOT"
    printf '{"phase":"%s"}\n' "$name" > "$evidence/phase.json"
    ln -sfn "$DP_OPT_PHASE_EVIDENCE_ROOT/progress.json" "$evidence/progress.json"
    nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
        END { exit (NR != 2 || bad) }' "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
}
export DP_OPT_TRACE_POLICIES=no_move DP_OPT_WORKAWARE_CADENCE_PILOT=1
phase cadence
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_workaware_pilot.py" \
    "$evidence/cadence" --output "$evidence/workaware-fit.json" > "$evidence/cadence-verification.json"
export DP_OPT_WORKAWARE_CADENCE_PILOT=0
export DP_OPT_WORKAWARE_FIT="$evidence/workaware-fit.json"
export DP_OPT_WORKAWARE_FIT_SHA256="$(sha256sum "$DP_OPT_WORKAWARE_FIT" | awk '{print $1}')"
export DP_OPT_WORKAWARE_SOURCE_ROOT="$evidence/cadence"
export DP_OPT_WORKAWARE_RATE_PRIORS="0.0005 ${DP_OPT_WORKAWARE_CONV_RATE:?Freeze external Conv rate}"
for population in 2 9; do
    export DP_OPT_TRACE_PATH="${DP_OPT_WORKAWARE_FIXTURE_ROOT:?Supply frozen fixtures}/recovery-$population.json"
    export DP_OPT_TRACE_POLICIES="kv_only_headroom kv_only_workaware"
    phase "recovery-$population"
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
        "$evidence/recovery-$population" --prompt-bank "$DP_OPT_PROMPT_BANK" \
        --output "$evidence/recovery-$population-verification.json"
done
export DP_OPT_TRACE_PATH="$DP_OPT_WORKAWARE_TRACE_PATH"
export DP_OPT_TRACE_POLICIES="no_move no_move_fair kv_only kv_only_headroom kv_only_workaware"
phase paired-trace
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
    "$evidence/paired-trace" --prompt-bank "$DP_OPT_PROMPT_BANK" --output "$evidence/trace-verification.json"
printf '{"status":"passed","new_algorithm_benefit_established":false,"next":"independent_archive_and_tail_analysis"}\n' > "$evidence/summary.json"
