#!/usr/bin/env bash
#SBATCH --job-name=kvcached-tests
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset ENABLE_KVCACHED KVCACHED_AUTOPATCH
cd "$repo"
python -m pytest -q tests/test_dp_namespace.py tests/test_dp_opt_benchmark.py \
    tests/test_dp_opt_independent_benchmark.py \
    tests/test_dp_opt_motivation.py \
    tests/test_vllm_tp_world_size.py tests/test_vllm_async_unmap_ordering.py \
    tests/test_vllm_engine_core_shutdown.py
