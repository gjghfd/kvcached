#!/usr/bin/env bash
#SBATCH --job-name=qwen-public-perf-followup
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:30:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
upload="${DP_OPT_PERFORMANCE_FOLLOWUP_UPLOAD:?Supply the immutable one-stage followup package}"
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
test ! -e "$evidence/runtime"
cp -a "$upload/runtime" "$evidence/runtime"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$evidence/runtime/experiments/dp_opt:$DEPLOY_ROOT/scripts"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
    > "$evidence/gpu-memory-before-load.csv"
awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { bad=1 }
    END { exit (NR != 2 || bad) }' "$evidence/gpu-memory-before-load.csv"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    python "$evidence/runtime/experiments/dp_opt/performance_followup.py" \
    --compute-upload "$upload" --compute-evidence "$evidence"
