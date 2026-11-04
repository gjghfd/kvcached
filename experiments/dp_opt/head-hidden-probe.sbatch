#!/usr/bin/env bash
#SBATCH --job-name=qwen-head-hidden-probe
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=03:00:00
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
cp -a "${DP_OPT_HEAD_HIDDEN_UPLOAD:?Supply unchanged original inputs}/frozen" "$evidence/frozen"
export DP_OPT_PHASE_EVIDENCE_ROOT="$evidence"
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=7200 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_TRACE_REPEATS=6 DP_OPT_TRACE_POLICIES=no_move DP_OPT_WORKAWARE_OPTIMIZER=point
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_WORKAWARE_CADENCE_PILOT=0 DP_OPT_WORKAWARE_PROFILES=0 DP_OPT_WORKAWARE_PHASE_PROFILES=0
export DP_OPT_RANDOM_BASELINE=0 DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0
unset DP_OPT_WORKAWARE_FIT DP_OPT_WORKAWARE_FIT_SHA256 DP_OPT_WORKAWARE_SOURCE_ROOT
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset VLLM_BATCH_INVARIANT
export DP_OPT_PROMPT_BANK="$evidence/frozen/inputs/prompt-bank.json"
export DP_OPT_TRACE_PATH="$evidence/frozen/trace.json"
printf '{"phase":"head-hidden-serial-reference-snapshots","is_benefit":false}\n' > "$evidence/phase.json"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python -m pytest -q \
    "$repo/tests/test_dp_opt_head_hidden_probe.py" "$repo/tests/test_dp_opt_fp32_head_probe.py" \
    "$repo/tests/test_dp_opt_static_batch_probe.py" > "$evidence/head-hidden-startup-tests.log"
test "$(grep -c 'python.*live_trace[.]py' "$repo/experiments/dp_opt/trace-benefit.sbatch")" = 1
sed 's@/live_trace[.]py"@/head_hidden_probe.py"@' "$repo/experiments/dp_opt/trace-benefit.sbatch" \
    > "$evidence/head-hidden-launch.sh"
sha256sum "$repo/experiments/dp_opt/head_hidden_probe.py" "$repo/experiments/dp_opt/head_hidden_worker.py" \
    "$repo/experiments/dp_opt/head-hidden-probe.sbatch" "$evidence/head-hidden-launch.sh" > "$evidence/head-hidden-source.sha256"
bash "$evidence/head-hidden-launch.sh"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python \
    "$repo/experiments/dp_opt/head_hidden_probe.py" --verify "$evidence" \
    --output "$evidence/independent-head-hidden-verification.json"
