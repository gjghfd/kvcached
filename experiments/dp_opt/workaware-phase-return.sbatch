#!/usr/bin/env bash
#SBATCH --job-name=qwen-early-return
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=32:00:00
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
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=1800 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_MAX_MOVES=8 DP_OPT_CANDIDATE_LIMIT=8 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S=30
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0 DP_OPT_WAIT_AWARE=0
export DP_OPT_WORKAWARE_PROFILES=1 DP_OPT_WORKAWARE_PHASE_PROFILES=1 DP_OPT_RANDOM_BASELINE=1
export DP_OPT_REBALANCE_MAX_MOVES=4 DP_OPT_REBALANCE_CANDIDATE_LIMIT=8
export DP_OPT_REBALANCE_MIN_GAIN_S=1 DP_OPT_REBALANCE_COOLDOWN_S=10 DP_OPT_REBALANCE_RETRY_S=2
export DP_OPT_PROMPT_BANK="$root/state/agentic-100k-input/prompt-bank.json"
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset DP_OPT_MEASURED_CONTROL_CONFIG DP_OPT_MEASURED_CONTROL_CONFIG_SHA256 DP_OPT_MEASURED_CONTROL_ENV
upload="${DP_OPT_WORKAWARE_PHASE_UPLOAD:?Supply protected frozen upload}"
cp "$upload/frozen/phase-config.json" "$evidence/phase-config.json"
cp "$upload/frozen/trace.json" "$evidence/original-trace.json"
cp "$upload/frozen/lower-trace.json" "$evidence/lower-trace.json"
cp "$upload/frozen/calibration-plan.json" "$evidence/calibration-plan.json"
sha256sum "$evidence"/{phase-config,original-trace,lower-trace,calibration-plan}.json > "$evidence/frozen-input.sha256"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/workaware_phase_campaign.py" \
    --validate-upload "$evidence" --inputs "$root/state/agentic-100k-input"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    -u DP_OPT_WORKAWARE_PHASE_PROFILES -u DP_OPT_RANDOM_BASELINE python -m pytest -q \
    "$repo/tests/test_dp_opt_release_observation.py" "$repo/tests/test_dp_opt_workaware_phase.py" \
    "$repo/tests/test_dp_opt_workaware_profile_verify.py" "$repo/tests/test_dp_opt_exact_constraint_reuse.py" \
    "$repo/tests/test_dp_opt_anchored_cadence.py" "$repo/tests/test_dp_opt_phase_reuse.py" \
    > "$evidence/launch-regression.log" 2>&1
phase() {
    name="$1"
    export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence/$name"
    mkdir -p "$DP_OPT_PHASE_EVIDENCE_ROOT"
    printf '{"phase":"%s"}\n' "$name" > "$evidence/phase.json"
    ln -sfn "$DP_OPT_PHASE_EVIDENCE_ROOT/progress.json" "$evidence/progress.json"
    nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
        > "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
        END { exit (NR != 2 || bad) }' "$DP_OPT_PHASE_EVIDENCE_ROOT/gpu-memory-before-load.csv"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
}
export DP_OPT_WORKAWARE_CADENCE_PILOT=1 DP_OPT_TRACE_POLICIES=no_move
export DP_OPT_WORKAWARE_CALIBRATION_PLAN="$evidence/calibration-plan.json"
export DP_OPT_TRACE_PATH="$evidence/original-trace.json"
reuse="${DP_OPT_WORKAWARE_REUSE_SOURCE:-}"
if [[ -n "$reuse" ]]; then
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/workaware_phase_campaign.py" \
        --validate-reuse-source "$reuse" --phase-config "$evidence/phase-config.json" > "$evidence/reused-raw-verification.json"
    supplement="$reuse"
else
    phase calibration-n567
    supplement="$evidence/calibration-n567"
fi
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/workaware_profile_bundle.py" \
    --legacy "${DP_OPT_WORKAWARE_LEGACY_SOURCE:?Supply immutable legacy calibration}" \
    --supplement "$supplement" --output "$evidence/composite-profile" --anchor-supplement \
    > "$evidence/composite-verification.json"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_workaware_profile.py" \
    "$evidence/composite-profile/supplement" --output "$evidence/supplement-fit.json" > "$evidence/supplement-verification.json"
if [[ -n "$reuse" ]]; then
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/workaware_phase_campaign.py" \
        --validate-reused-profile "$evidence/composite-profile" --phase-config "$evidence/phase-config.json" \
        > "$evidence/reused-profile-verification.json"
    cp -a "$evidence/composite-profile/supplement" "$evidence/calibration-n567"
fi
export DP_OPT_WORKAWARE_CADENCE_PILOT=0
unset DP_OPT_WORKAWARE_CALIBRATION_PLAN
export DP_OPT_WORKAWARE_FIT="$evidence/composite-profile/fit.json"
export DP_OPT_WORKAWARE_FIT_SHA256="$(sha256sum "$DP_OPT_WORKAWARE_FIT" | awk '{print $1}')"
export DP_OPT_WORKAWARE_SOURCE_ROOT="$evidence/composite-profile"
export DP_OPT_WORKAWARE_RATE_PRIORS="0.0005 ${DP_OPT_WORKAWARE_CONV_RATE:?Freeze Conv rate}"
export DP_OPT_TRACE_POLICIES='random_no_return kv_only_workaware'
phase cache-fixed-original
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
    "$evidence/cache-fixed-original" --prompt-bank "$DP_OPT_PROMPT_BANK" \
    --output "$evidence/cache-fixed-verification.json"
export DP_OPT_TRACE_PATH="$evidence/lower-trace.json"
export DP_OPT_WORKAWARE_RATE_PRIORS="0.00025 $DP_OPT_WORKAWARE_CONV_RATE"
export DP_OPT_TRACE_POLICIES='kv_only_workaware random_no_return no_move_fair no_move kv_only kv_only_headroom'
phase lower-code-rate
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/verify_trace.py" \
    "$evidence/lower-code-rate" --prompt-bank "$DP_OPT_PROMPT_BANK" \
    --output "$evidence/lower-rate-verification.json"
