#!/usr/bin/env bash
#SBATCH --job-name=qwen-fp32-head-probe
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
base="$root/state/live-trace-$SLURM_JOB_ID"
mkdir -p "$base"
finish() {
    exit_code=$?
    trap - EXIT
    printf '%s\n' "$exit_code" > "$base/exit-status"
    exit "$exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp -a "${DP_OPT_FP32_HEAD_UPLOAD:?Supply the unchanged static diagnostic inputs}/frozen" "$base/frozen"
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776 KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_BUDGET_MIB=10240 DP_OPT_REQUEST_TIMEOUT_S=7200 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1 DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
export DP_OPT_REVALIDATE_PREPARED=1 DP_OPT_PREPLAN=1 DP_OPT_PAUSE_SYNC_INTERVAL=32
export DP_OPT_KV_TRANSPORT=dp_echo DP_OPT_TRANSFER_CHUNK_MIB=32 DP_OPT_WORKAWARE_OPTIMIZER=point
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRANSFER_PILOT=0
export DP_OPT_LAYOUT_COST_PILOT=0 DP_OPT_ADMISSION_COST_PILOT=0 DP_OPT_STRESS_SWEEP=0
export DP_OPT_WORKAWARE_CADENCE_PILOT=0 DP_OPT_WORKAWARE_PROFILES=0 DP_OPT_WORKAWARE_PHASE_PROFILES=0
export DP_OPT_RANDOM_BASELINE=0 DP_OPT_FROZEN_TRACE_REPLAY=0 DP_OPT_MEASURED_CONTROL=0
unset DP_OPT_WORKAWARE_FIT DP_OPT_WORKAWARE_FIT_SHA256 DP_OPT_WORKAWARE_SOURCE_ROOT
unset DP_OPT_LAYOUT_COST_LOOKUP DP_OPT_LAYOUT_COST_LOOKUP_SHA256 DP_OPT_LAYOUT_COST_SOURCE_ROOT
unset VLLM_BATCH_INVARIANT
printf '{"phase":"fp32-head-compute-preflight","is_benefit":false}\n' > "$base/phase.json"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python -m pytest -q \
    "$repo/tests/test_dp_opt_fp32_head_probe.py" "$repo/tests/test_dp_opt_static_batch_probe.py" \
    "$repo/tests/test_dp_opt_batch_numerics_probe.py" > "$base/fp32-head-startup-tests.log"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python - <<'PY'
import torch
operands = torch.ones((2, 16), dtype=torch.bfloat16, device="cuda")
logits = torch.mm(operands, operands.t(), out_dtype=torch.float32)
assert logits.dtype == torch.float32 and torch.equal(logits, torch.full_like(logits, 16))
print("CUDA BF16 operands / FP32 GEMM output passed", torch.__version__)
PY
site="$ENV_PREFIX/lib/python3.10/site-packages"
mkdir "$base/installed-numerics-source"
cp "$site/vllm/model_executor/layers/logits_processor.py" "$base/installed-numerics-source/"
cp "$site/vllm/config/model.py" "$base/installed-numerics-source/model-config.py"
cp "$site/torch/version.py" "$base/installed-numerics-source/torch-version.py"
sha256sum "$base/installed-numerics-source/"* > "$base/installed-numerics-source.sha256"
test "$(grep -c 'python.*live_trace[.]py' "$repo/experiments/dp_opt/trace-benefit.sbatch")" = 1
sed 's@/live_trace[.]py"@/fp32_head_probe.py"@' "$repo/experiments/dp_opt/trace-benefit.sbatch" \
    > "$base/fp32-head-launch.sh"
sha256sum "$repo/experiments/dp_opt/fp32_head_probe.py" \
    "$repo/experiments/dp_opt/fp32_head_worker.py" "$repo/experiments/dp_opt/fp32-head-probe.sbatch" \
    "$base/fp32-head-launch.sh" > "$base/fp32-head-source.sha256"
for stage in static migration; do
    evidence="$base/$stage"
    mkdir "$evidence"
    cp -a "$base/frozen" "$evidence/frozen"
    export DP_OPT_HEAD_PROBE_STAGE="$stage" DP_OPT_PHASE_EVIDENCE_ROOT="$evidence"
    export DP_OPT_PROMPT_BANK="$evidence/frozen/inputs/prompt-bank.json" DP_OPT_TRACE_PATH="$evidence/frozen/trace.json"
    if [[ "$stage" == static ]]; then
        export DP_OPT_TRACE_REPEATS=6 DP_OPT_TRACE_POLICIES=no_move
    else
        export DP_OPT_TRACE_REPEATS=1 DP_OPT_TRACE_POLICIES='no_move random_no_return kv_only_headroom'
    fi
    printf '{"phase":"fp32-head-%s","is_benefit":false}\n' "$stage" > "$base/phase.json"
    bash "$base/fp32-head-launch.sh"
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python \
        "$repo/experiments/dp_opt/fp32_head_probe.py" --verify "$evidence" --stage "$stage" \
        --output "$evidence/independent-fp32-head-verification.json"
done
printf '{"phase":"fp32-head-static-and-migration-strict-verified-not-benefit","is_benefit":false}\n' > "$base/phase.json"
