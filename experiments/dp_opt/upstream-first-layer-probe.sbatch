#!/usr/bin/env bash
#SBATCH --job-name=qwen-upstream-first-layer-probe
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=03:00:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
upload="${DP_OPT_UPSTREAM_UPLOAD:?Supply immutable upstream upload}"
evidence="$root/state/live-trace-${SLURM_JOB_ID:?Require actual allocation}"
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
test ! -e "$evidence/runtime"
cp -a "$upload/runtime" "$evidence/runtime"
find "$evidence/runtime" -type f -exec chmod u+w {} +
printf '%s\n' '{"scope":"private_execution_copy_only","uploaded_runtime_remains_readonly":true,"reason":"copied_tamper_test_fixtures_must_be_writable","runtime_bytes_checked_by_manifest":true}' \
    > "$evidence/runtime-file-mode-policy.json"
cd "$evidence/runtime"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$evidence/runtime/experiments/dp_opt:$DEPLOY_ROOT/scripts"
cd "$evidence"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
    > "$evidence/gpu-memory-before-load.csv"
awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
    END { exit (NR != 2 || bad) }' "$evidence/gpu-memory-before-load.csv"
export DP_OPT_UPSTREAM_COMPUTE_CONFIRMED=1
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    -u VLLM_BATCH_INVARIANT python -m pytest -q -p no:cacheprovider \
    "$evidence/runtime/tests/test_dp_opt_upstream_first_layer_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_upstream_first_layer_campaign.py" \
    "$evidence/runtime/tests/test_dp_opt_head_hidden_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_fp32_head_probe.py" \
    "$evidence/runtime/tests/test_dp_opt_static_batch_probe.py" \
    > "$evidence/upstream-startup-tests.log" 2>&1
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python -c 'import upstream_first_layer_worker, upstream_first_layer_probe; import head_hidden_worker, fp32_head_worker; from upstream_first_layer_campaign import InstalledProvenanceUpstreamWorker; print("actual compute worker imports passed", InstalledProvenanceUpstreamWorker.__module__)' \
    > "$evidence/upstream-compute-imports.log" 2>&1
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python "$evidence/runtime/experiments/dp_opt/upstream_first_layer_campaign.py" \
    --compute-upload "$upload" --compute-evidence "$evidence"
