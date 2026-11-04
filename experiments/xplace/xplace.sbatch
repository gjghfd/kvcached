#!/usr/bin/env bash
#SBATCH --job-name=xplace
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=05:00:00
# Submit with: sbatch --time=... --output=<tag>/logs/slurm-%j.out --export=ALL,XPLACE_TAG=...,XPLACE_PHASES=... xplace.sbatch
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
tag="${XPLACE_TAG:?Set XPLACE_TAG to the uploaded campaign tag}"
base="$root/xplace/$tag"
upload="$base/upload"
work="$base/job-$SLURM_JOB_ID"
out="$work/out"
mkdir -p "$out"
finish() {
    code=$?
    trap - EXIT
    rm -rf "${DP_OPT_TRANSFER_ROOT:-/nonexistent-xplace}"
    printf '%s\n' "$code" > "$work/exit-status"
    date -Is > "$work/finished-at"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is > "$work/started-at"
(cd "$upload" && sha256sum -c --quiet MANIFEST.sha256) > "$work/manifest-check.log" 2>&1
cp -a "$upload/runtime" "$upload/tests" "$upload/inputs" "$work/"
chmod -R u+w "$work/runtime" "$work/tests" "$work/inputs"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv,noheader,nounits > "$work/gpus.csv"
nvidia-smi topo -m > "$work/topology.txt" 2>&1 || true
python3 - "$work/gpus.csv" <<'EOF'
import sys
rows = [line.split(", ") for line in open(sys.argv[1]).read().strip().splitlines()]
assert len(rows) == 4, f"expected 4 GPUs, got {len(rows)}"
assert all("A800" in row[2] for row in rows), rows
assert all(float(row[4]) >= 70000 for row in rows), f"GPU memory already in use: {rows}"
EOF
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$work/runtime:$DEPLOY_ROOT/scripts"
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 DP_OPT_LIVE_PLACEMENT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENGINE_READY_TIMEOUT_S=1800
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KVCACHED_MIN_RESERVED_PAGES=0 KVCACHED_MAX_RESERVED_PAGES=0 KVCACHED_PAGE_PREALLOC_ENABLED=false
export KVCACHED_GPU_UTILIZATION=0.98 DP_OPT_PAUSE_SYNC_INTERVAL=32 DP_OPT_TRANSFER_CHUNK_MIB=32
export DP_OPT_TRANSFER_ROOT="$(mktemp -d /tmp/xplace-$SLURM_JOB_ID.XXXXXX)"
unset VLLM_BATCH_INVARIANT
cd "$work"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT python - <<'EOF' > "$work/import-check.log" 2>&1
import importlib
for name in ("engine", "cost_model", "planner", "fit", "fairness", "controller", "peer_transfer",
             "trace_runner", "calibrate", "live_scheduler", "worker", "main"):
    importlib.import_module(name)
print("imports ok")
EOF
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python -m pytest -q -p no:cacheprovider "$work/tests" > "$work/startup-tests.log" 2>&1
checkpoint="$(python -c 'from qwen_common import checkpoint_path; print(checkpoint_path())')"
phases="${XPLACE_PHASES:?Set XPLACE_PHASES}"
policies="${XPLACE_POLICIES:-no_move+ca_full}"
args=(--checkpoint "$checkpoint" --output "$out" --prompt-bank "$work/inputs/prompt-bank.json"
      --phases "${phases//+/,}" --policies "${policies//+/,}"
      --budget-margin-gib "${XPLACE_BUDGET_MARGIN_GIB:-3}" --seed "${XPLACE_SEED:-20261103}"
      --drain-timeout-s "${XPLACE_DRAIN_TIMEOUT_S:-5400}")
[[ -n "${XPLACE_TRACE:-}" ]] && args+=(--trace "$work/inputs/$XPLACE_TRACE")
[[ -n "${XPLACE_STEP_PARAMS:-}" ]] && args+=(--step-params "$work/inputs/$XPLACE_STEP_PARAMS")
[[ -n "${XPLACE_PLANNER_PARAMS:-}" ]] && args+=(--planner-params "$work/inputs/$XPLACE_PLANNER_PARAMS")
[[ -n "${XPLACE_BUDGET_CAP_GIB:-}" ]] && args+=(--budget-cap-gib "$XPLACE_BUDGET_CAP_GIB")
[[ -n "${XPLACE_PROBE_MAX_MS:-}" ]] && args+=(--probe-max-ms "$XPLACE_PROBE_MAX_MS")
python "$work/runtime/main.py" "${args[@]}" > "$work/main.log" 2>&1
