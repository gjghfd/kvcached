#!/usr/bin/env bash
#SBATCH --job-name=qwen-robust-native
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=06:00:00
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
parent_job="${DP_OPT_ROBUST_PARENT_JOB:?Specify independently verified parent job}"
[[ "$parent_job" =~ ^[1-9][0-9]*$ ]]
parent="$root/state/live-trace-$parent_job"
test "$(cat "$parent/exit-status")" = 0
test "$(sha256sum "$parent/long-context-fit.json" | awk '{print $1}')" = "${DP_OPT_ROBUST_FIT_SHA256:?Freeze parent fit SHA}"
cp -a "${DP_OPT_ROBUST_UPLOAD:?Specify protected upload}/frozen" "$evidence/frozen"
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=1800 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0 DP_OPT_WORKAWARE_CADENCE_PILOT=0
export DP_OPT_WORKAWARE_PROFILES=1 DP_OPT_WORKAWARE_PHASE_PROFILES=1 DP_OPT_RANDOM_BASELINE=1
export DP_OPT_WORKAWARE_OPTIMIZER=robust
export DP_OPT_REBALANCE_MAX_MOVES=4 DP_OPT_REBALANCE_CANDIDATE_LIMIT=8
export DP_OPT_REBALANCE_MIN_GAIN_S=1 DP_OPT_REBALANCE_COOLDOWN_S=10 DP_OPT_REBALANCE_RETRY_S=2
export DP_OPT_WORKAWARE_FIT="$parent/long-context-fit.json" DP_OPT_WORKAWARE_FIT_SHA256="$DP_OPT_ROBUST_FIT_SHA256"
export DP_OPT_WORKAWARE_SOURCE_ROOT="$parent/long-context-calibration"
export DP_OPT_WORKAWARE_RATE_PRIORS='0.00025 0.005652527937486215'
export DP_OPT_TRACE_POLICIES='random_no_return kv_only_workaware'
export DP_OPT_PROMPT_BANK="$root/state/agentic-100k-input/prompt-bank.json"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
unset DP_OPT_WORKAWARE_CALIBRATION_PLAN
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/long_conv_return_probe.py" \
    --validate-upload "$evidence/frozen" --inputs "$root/state/agentic-100k-input" > "$evidence/input-verification.json"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python -m pytest -q \
    "$repo/tests/test_dp_opt_robust_live_integration.py" "$repo/tests/test_dp_opt_robust_adapter.py" \
    "$repo/tests/test_dp_opt_zero_arrival_trace.py" > "$evidence/launch-regression.log" 2>&1
for seed in 20261030 20261031; do
    export DP_OPT_TRACE_SEED="$seed"
    export DP_OPT_TRACE_PATH="$evidence/frozen/probe-$seed.json"
    export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence/native-$seed"
    mkdir -p "$DP_OPT_PHASE_EVIDENCE_ROOT"
    printf '{"phase":"native-%s"}\n' "$seed" > "$evidence/phase.json"
    ln -sfn "$DP_OPT_PHASE_EVIDENCE_ROOT/progress.json" "$evidence/progress.json"
    nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
        END { exit (NR != 2 || bad) }' "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
        "$DP_OPT_PHASE_EVIDENCE_ROOT" --prompt-bank "$DP_OPT_PROMPT_BANK" --output "$evidence/native-$seed-verification.json"
done
printf '{"status":"robust_native_pairs_completed_requires_independent_validation","research_goal_complete":false}\n' \
    > "$evidence/summary.json"
