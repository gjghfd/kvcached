#!/usr/bin/env bash
#SBATCH --job-name=qwen-efficient-public-search
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
unset VLLM_BATCH_INVARIANT
launch="$(env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python "$repo/experiments/dp_opt/efficient_public_launch.py" \
    --make-launch search --output "$evidence/launch-sources")"
bash "$launch"
