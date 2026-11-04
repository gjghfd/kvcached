#!/usr/bin/env bash
#SBATCH --job-name=qwen-public-response-search
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=18:00:00
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
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$evidence/exit-status"
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
upload="${DP_OPT_PUBLIC_UPLOAD:?Supply frozen independently reconstructed public inputs}"
cp -a "$upload/frozen" "$evidence/frozen"
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=7200 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0
export DP_OPT_WORKAWARE_PROFILES=1 DP_OPT_WORKAWARE_PHASE_PROFILES=1 DP_OPT_RANDOM_BASELINE=1
export DP_OPT_WORKAWARE_OPTIMIZER=robust
export DP_OPT_REBALANCE_MAX_MOVES=4 DP_OPT_REBALANCE_CANDIDATE_LIMIT=8
export DP_OPT_REBALANCE_MIN_GAIN_S=1 DP_OPT_REBALANCE_COOLDOWN_S=10 DP_OPT_REBALANCE_RETRY_S=2
export DP_OPT_PROMPT_BANK="$evidence/frozen/inputs/prompt-bank.json"
export DP_OPT_TRACE_PATH="$evidence/frozen/trace.json"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
unset DP_OPT_WORKAWARE_FIT DP_OPT_WORKAWARE_FIT_SHA256 DP_OPT_WORKAWARE_SOURCE_ROOT
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/public_workload_search.py" \
    --root "$evidence" --action validate > "$evidence/input-verification.json"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python -m pytest -q \
    "$repo/tests/test_dp_opt_public_workload_search.py" "$repo/tests/test_dp_opt_public_response_inputs.py" \
    "$repo/tests/test_dp_opt_robust_live_integration.py" "$repo/tests/test_dp_opt_workaware_profile_verify.py" \
    "$repo/tests/test_dp_opt_trace.py" \
    > "$evidence/launch-regression.log" 2>&1
phase() {
    name="$1"
    export DP_OPT_WORKAWARE_OPTIMIZER="$(env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python -c \
        'import os; from public_workload_search import phase_optimizer; print(phase_optimizer(os.environ["DP_OPT_TRACE_POLICIES"].split()))')"
    export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence/$name"
    mkdir -p "$DP_OPT_PHASE_EVIDENCE_ROOT"
    printf '{"phase":"%s","public_original_response_lengths":true}\n' "$name" > "$evidence/phase.json"
    ln -sfn "$DP_OPT_PHASE_EVIDENCE_ROOT/progress.json" "$evidence/progress.json"
    nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
        END { exit (NR != 2 || bad) }' "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
}
export DP_OPT_WORKAWARE_CALIBRATION_PLAN="$evidence/frozen/calibration-plan.json"
export DP_OPT_WORKAWARE_CADENCE_PILOT=1 DP_OPT_TRACE_POLICIES=no_move
phase public-context-calibration
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_workaware_profile.py" \
    "$evidence/public-context-calibration" --output "$evidence/public-context-fit.json" \
    > "$evidence/profile-verification.json"
export DP_OPT_WORKAWARE_CADENCE_PILOT=0
unset DP_OPT_WORKAWARE_CALIBRATION_PLAN
export DP_OPT_WORKAWARE_FIT="$evidence/public-context-fit.json"
export DP_OPT_WORKAWARE_FIT_SHA256="$(sha256sum "$DP_OPT_WORKAWARE_FIT" | awk '{print $1}')"
export DP_OPT_WORKAWARE_SOURCE_ROOT="$evidence/public-context-calibration"
export DP_OPT_WORKAWARE_RATE_PRIORS='0.001 0.012'
export DP_OPT_TRACE_POLICIES='no_move random_no_return kv_only_headroom kv_only_workaware'
for seed in 2026103101 2026103102; do
    for population in 2 3 4; do
        name="native-$seed-N$population"
        export DP_OPT_TRACE_PATH="$evidence/frozen/$name.json"
        phase "$name"
        env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
            "$evidence/$name" --prompt-bank "$DP_OPT_PROMPT_BANK" --output "$evidence/$name-verification.json"
    done
done
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/public_workload_search.py" \
    --root "$evidence" --action freeze-calibration > "$evidence/measured-rate-priors.json"
rates_text="$(env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python -c \
    'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["conv_rates_rps"])))' \
    "$evidence/calibration-frozen/calibration-config.json")"
read -r -a rates <<< "$rates_text"
test "${#rates[@]}" -eq 3
export DP_OPT_TRACE_POLICIES=no_move
for ordinal in 0 1 2; do
    export DP_OPT_WORKAWARE_RATE_PRIORS="0.001 ${rates[$ordinal]}"
    export DP_OPT_TRACE_PATH="$evidence/calibration-frozen/rate-$ordinal.json"
    phase "rate-$ordinal"
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
        "$evidence/rate-$ordinal" --prompt-bank "$DP_OPT_PROMPT_BANK" --output "$evidence/rate-$ordinal-verification.json"
done
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/public_workload_search.py" \
    --root "$evidence" --action analyze > "$evidence/search-analysis.json"
printf '{"status":"public_mechanism_and_rate_calibration_completed","research_goal_complete":false}\n' \
    > "$evidence/summary.json"
