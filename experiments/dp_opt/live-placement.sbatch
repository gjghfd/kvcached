#!/usr/bin/env bash
#SBATCH --job-name=qwen-live-placement
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=00:45:00
#SBATCH --time-min=00:20:00
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/experiments/dp_opt:$DEPLOY_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 DP_OPT_LIVE_PLACEMENT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KVCACHED_MIN_RESERVED_PAGES=0 KVCACHED_MAX_RESERVED_PAGES=0 KVCACHED_PAGE_PREALLOC_ENABLED=false
mode="${DP_OPT_LIVE_MODE:-allocator_suite}"
max_model_len="${DP_OPT_MAX_MODEL_LEN:-32768}"
cache_bytes="${DP_OPT_LIVE_CACHE_BYTES:-4294967296}"
if [[ "$mode" == benefit || "$mode" == benefit_suite ]]; then
    evidence="$root/state/live-benefit-$SLURM_JOB_ID"
else
    evidence="$root/state/live-placement-$SLURM_JOB_ID"
fi
base_evidence="$evidence"
mkdir -p "$evidence"
test -s "$HOME/kvcached/paracloud/state/build-ready"
export DP_OPT_TRANSFER_ROOT="$(mktemp -d /tmp/dp-placement-$SLURM_JOB_ID.XXXXXX)"
cleanup() {
    status=$?
    trap - EXIT
    rm -f "$DP_OPT_TRANSFER_ROOT"/*.pt
    rmdir "$DP_OPT_TRANSFER_ROOT"
    printf '%s\n' "$status" > "$base_evidence/exit-status"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_PAUSE_SYNC_INTERVAL python -m pytest -q \
    "$repo/tests/test_dp_opt_ilp_reference.py" "$repo/tests/test_dp_opt_live_placement.py" \
    "$repo/tests/test_dp_opt_live_benefit.py"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_PAUSE_SYNC_INTERVAL python -m pytest -q \
    "$repo/tests/test_vllm_tp_world_size.py" "$repo/tests/test_vllm_async_unmap_ordering.py" \
    "$repo/tests/test_vllm_engine_core_shutdown.py"
checkpoint="$(python -c 'from qwen_common import checkpoint_path; print(checkpoint_path())')"
python - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError('Exactly two allocated GPUs are required')
if min(torch.cuda.get_device_properties(index).total_memory for index in range(2)) < 75 * 1024**3:
    raise RuntimeError('Two resident BF16 EP2 tenants require at least 75 GiB per GPU')
PY
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv > "$evidence/hardware.txt"
nvidia-smi topo -m > "$evidence/topology.txt"
sha256sum "$repo/experiments/dp_opt/live_"*.py "$repo/experiments/dp_opt/ilp_reference.py" \
    "$repo/kvcached/integration/vllm/patches.py" "$repo/tests/test_dp_opt_live_placement.py" \
    "$repo/tests/test_dp_opt_live_benefit.py" "$repo/experiments/dp_opt/verify_benefit.py" \
    "$repo/experiments/dp_opt/calibrated_cost.py" "$repo/experiments/dp_opt/live-placement.sbatch" \
    "$repo/experiments/dp_opt/worker.py" \
    "$root/state/sparse-fit-231467.json" > "$evidence/source.sha256"
date -Is > "$evidence/started-at"
run_mode() {
arrival_tokens=2048
if [[ "${DP_OPT_LIVE_SCENARIO:-packing}" == pressure ]]; then
    arrival_tokens=2816
fi
python "$repo/experiments/dp_opt/live_benchmark.py" --checkpoint "$checkpoint" \
    --fit "$root/state/sparse-fit-231467.json" --output "$evidence" \
    --mode "$1" --population "${DP_OPT_LIVE_POPULATION:-8}" \
    --output-tokens "${DP_OPT_LIVE_OUTPUT_TOKENS:-256}" --policy "${DP_OPT_LIVE_POLICY:-ilp}" \
    --input-tokens "${DP_OPT_LIVE_INPUT_TOKENS:-64}" \
    --arrival-input-tokens "${DP_OPT_LIVE_ARRIVAL_INPUT_TOKENS:-$arrival_tokens}" \
    --headroom-pages "${DP_OPT_LIVE_HEADROOM_PAGES:-2}" \
    --scenario "${DP_OPT_LIVE_SCENARIO:-packing}" \
    --max-model-len "$max_model_len" --cache-bytes "$cache_bytes" \
    --port "$((30000 + SLURM_JOB_ID % 1000 * 20))" \
    --allow-uncalibrated
}
if [[ "$mode" == benefit_suite ]]; then
    python - "$base_evidence/suite-config.json" "$max_model_len" "$cache_bytes" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({"scenario": "tradeoff", "dp_pause_sync_interval": 1,
                                      "max_model_len": int(sys.argv[2]), "cache_bytes": int(sys.argv[3]),
                                      "cases": [{"case": "early", "repeats": 6, "seed": 20261021, "minimum_decode_tokens": 8},
                                                {"case": "late", "repeats": 3, "seed": 20261022, "minimum_decode_tokens": 192}]}, indent=2) + "\n")
PY
    test "${DP_OPT_PAUSE_SYNC_INTERVAL:-32}" = 1
    for specification in early:6:20261021:8 late:3:20261022:192; do
        IFS=: read -r case_name repeats seed minimum_decode <<< "$specification"
        export DP_OPT_BENEFIT_CASE="$case_name"
        evidence="$base_evidence/$case_name"
        mkdir -p "$evidence"
        cp "$base_evidence/source.sha256" "$base_evidence/hardware.txt" "$base_evidence/topology.txt" "$base_evidence/started-at" "$evidence/"
        printf '{"active_case":"%s","status":"running"}\n' "$case_name" > "$base_evidence/progress.json"
        if python "$repo/experiments/dp_opt/live_benefit.py" --checkpoint "$checkpoint" \
            --fit "$root/state/sparse-fit-231467.json" --output "$evidence" --scenario tradeoff --control-warmup \
            --repeats "$repeats" --seed "$seed" --minimum-decode-tokens "$minimum_decode" \
            --max-model-len "$max_model_len" --cache-bytes "$cache_bytes" \
            --port "$((30000 + SLURM_JOB_ID % 1000 * 20))"; then
            printf '0\n' > "$evidence/exit-status"
        else
            status=$?
            printf '%s\n' "$status" > "$evidence/exit-status"
            exit "$status"
        fi
        python "$repo/experiments/dp_opt/verify_benefit.py" "$evidence" --output "$base_evidence/verification-$case_name.json"
        printf '{"completed_case":"%s","status":"passed"}\n' "$case_name" > "$base_evidence/progress.json"
    done
    printf '{"status":"passed","completed_cases":["early","late"]}\n' > "$base_evidence/summary.json"
elif [[ "$mode" == benefit ]]; then
    python "$repo/experiments/dp_opt/live_benefit.py" --checkpoint "$checkpoint" \
        --fit "$root/state/sparse-fit-231467.json" --output "$evidence" \
        --repeats "${DP_OPT_BENEFIT_REPEATS:-3}" \
        --output-tokens "${DP_OPT_BENEFIT_OUTPUT_TOKENS:-256}" \
        --seed "${DP_OPT_BENEFIT_SEED:-42}" \
        --minimum-decode-tokens "${DP_OPT_BENEFIT_MINIMUM_DECODE_TOKENS:-8}" \
        --scenario "${DP_OPT_BENEFIT_SCENARIO:-capacity}" --control-warmup \
        --max-model-len "$max_model_len" --cache-bytes "$cache_bytes" \
        --port "$((30000 + SLURM_JOB_ID % 1000 * 20))"
elif [[ "$mode" == allocator_suite ]]; then
    for case in packing:no_move pressure:no_move pressure:own_model pressure:ilp; do
        export DP_OPT_LIVE_SCENARIO="${case%:*}"
        export DP_OPT_LIVE_POLICY="${case#*:}"
        evidence="$base_evidence/${DP_OPT_LIVE_SCENARIO}-${DP_OPT_LIVE_POLICY}"
        mkdir -p "$evidence"
        run_mode allocator
    done
    evidence="$base_evidence"
elif [[ "$mode" == suite ]]; then
    for mode in migration admission; do
        evidence="$base_evidence/$mode"
        mkdir -p "$evidence"
        run_mode "$mode"
    done
    evidence="$base_evidence"
else
    run_mode "$mode"
fi
