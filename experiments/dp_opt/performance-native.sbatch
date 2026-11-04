#!/usr/bin/env bash
#SBATCH --job-name=qwen-public-perf-native
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:30:00
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
    exit_code=$?
    trap - EXIT
    printf '%s\n' "$exit_code" > "$evidence/exit-status"
    exit "$exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
upload="${DP_OPT_PERFORMANCE_UPLOAD:?Supply immutable public traces and verified default-head costs}"
cp -a "$upload/frozen" "$evidence/frozen"
cp -a "$upload/reused-context-calibration" "$evidence/public-context-calibration"
cp -p "$upload/reused-context-fit.json" "$evidence/reused-context-fit.json"
export DP_OPT_PERFORMANCE_NUMERICS=observe
unset VLLM_BATCH_INVARIANT
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=7200 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0
export DP_OPT_WORKAWARE_CADENCE_PILOT=0 DP_OPT_WORKAWARE_PROFILES=1 DP_OPT_WORKAWARE_PHASE_PROFILES=1
export DP_OPT_RANDOM_BASELINE=1 DP_OPT_WORKAWARE_OPTIMIZER=robust
export DP_OPT_REBALANCE_MAX_MOVES=4 DP_OPT_REBALANCE_CANDIDATE_LIMIT=8
export DP_OPT_REBALANCE_MIN_GAIN_S=1 DP_OPT_REBALANCE_COOLDOWN_S=10 DP_OPT_REBALANCE_RETRY_S=2
export DP_OPT_PROMPT_BANK="$evidence/frozen/inputs/prompt-bank.json"
export DP_OPT_WORKAWARE_SOURCE_ROOT="$evidence/public-context-calibration"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_workaware_profile.py" \
    "$DP_OPT_WORKAWARE_SOURCE_ROOT" --output "$evidence/public-context-fit.json" \
    > "$evidence/cost-reuse-verification.json"
export DP_OPT_WORKAWARE_FIT="$evidence/public-context-fit.json"
export DP_OPT_WORKAWARE_FIT_SHA256="$(sha256sum "$DP_OPT_WORKAWARE_FIT" | awk '{print $1}')"
export DP_OPT_WORKAWARE_RATE_PRIORS='0.001 0.012'
export DP_OPT_TRACE_POLICIES='no_move random_no_return kv_only_headroom kv_only_workaware'
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python -m pytest -q \
    "$repo/tests/test_dp_opt_performance_numerics.py" > "$evidence/performance-startup-tests.log"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/performance_public.py" \
    --make-trace-launch "$evidence/performance-trace-launch.sh" > "$evidence/performance-launch-receipt.json"
bash -n "$evidence/performance-trace-launch.sh"
for seed in 2026103101 2026103102; do
    for population in 2 3 4; do
        name="native-$seed-N$population"
        export DP_OPT_TRACE_PATH="$evidence/frozen/$name.json"
        export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence/$name"
        mkdir -p "$DP_OPT_PHASE_EVIDENCE_ROOT"
        printf '{"phase":"%s","greedy_gate":"observation_only","is_debug_run":false}\n' "$name" > "$evidence/phase.json"
        ln -sfn "$DP_OPT_PHASE_EVIDENCE_ROOT/progress.json" "$evidence/progress.json"
        nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
            > "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
        bash "$evidence/performance-trace-launch.sh"
        env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/performance_public.py" \
            --verify-phase "$DP_OPT_PHASE_EVIDENCE_ROOT" --prompt-bank "$DP_OPT_PROMPT_BANK" \
            --output "$evidence/$name-verification.json"
    done
done
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/performance_public.py" \
    --module search --root "$evidence" --action freeze-calibration > "$evidence/calibration-priors.json"
printf '{"status":"native_performance_completed_rate_traces_frozen","quality_equivalence_unresolved":true,"research_goal_complete":false}\n' \
    > "$evidence/summary.json"
