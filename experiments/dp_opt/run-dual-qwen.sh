#!/usr/bin/env bash
set -euo pipefail
source "$HOME/aegaeon/paracloud/scripts/common.sh"
require_compute
load_toolchains
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
deploy="${KVCACHED_DEPLOY_ROOT:-$HOME/kvcached/paracloud}"
test -s "$deploy/state/build-ready"
export PATH="$root/env/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib:$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo/experiments/dp_opt:$DEPLOY_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1
export KVCACHED_MIN_RESERVED_PAGES=0 KVCACHED_MAX_RESERVED_PAGES=0
export KVCACHED_PAGE_PREALLOC_ENABLED=false
mode="${DP_OPT_BENCHMARK_MODE:-combined}"
max_model_len="${DP_OPT_MAX_MODEL_LEN:-32768}"
kv_cache_bytes="${DP_OPT_KV_CACHE_BYTES:-4294967296}"
max_num_seqs="${DP_OPT_MAX_NUM_SEQS:-32}"
max_num_batched_tokens="${DP_OPT_MAX_BATCHED_TOKENS:-512}"
execution_mode="${DP_OPT_EXECUTION_MODE:-eager}"
execution_args=(--enforce-eager)
if [[ "$execution_mode" == graph ]]; then
    capture_sizes="${DP_OPT_CAPTURE_SIZES:-1,8,16,32,64,128,256}"
    [[ "$capture_sizes" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid capture sizes' >&2; exit 1; }
    execution_args=(-cc "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$capture_sizes]}")
elif [[ "$execution_mode" != eager ]]; then
    printf 'Unknown execution mode: %s\n' "$execution_mode" >&2
    exit 1
fi
service_count=2
worker_class=worker.VerificationWorker
if [[ "$mode" == motivation ]]; then
    service_count=1
    worker_class=motivation_worker.MotivationWorker
    export VLLM_SERVER_DEV_MODE=1
fi
run_label="${DP_OPT_RUN_LABEL:-}"
[[ "$run_label" =~ ^[a-zA-Z0-9_-]*$ ]] || { echo 'Invalid run label' >&2; exit 1; }
job_tag="$SLURM_JOB_ID${run_label:+-$run_label}"
evidence="$deploy/state/dual-qwen-$job_tag"
mkdir -p "$evidence" "$deploy/logs"
export DP_OPT_EVIDENCE_ROOT="$evidence"
ports="${DP_OPT_PORT_BASE:-$((20000 + (SLURM_JOB_ID % 1000) * 20))}"
temporary_root="$(mktemp -d "/tmp/kq-$SLURM_JOB_ID.XXXXXX")"
endpoint="$deploy/state/endpoint-$job_tag"
pids=()
monitor_pid=''
cleanup() {
    status=$?
    trap - EXIT INT TERM
    rm -f "$endpoint"
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    sleep 5
    for pid in "${pids[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$temporary_root"
    printf '%s\n' "$status" > "$evidence/exit-status"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
{
    date -Is
    hostname
    scontrol show job "$SLURM_JOB_ID"
    nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
    nvidia-smi topo -m
    printf 'execution_mode=%s\ncapture_sizes=%s\nrun_label=%s\n' "$execution_mode" "${capture_sizes:-none}" "$run_label"
    git -C "$repo" rev-parse HEAD
    git -C "$repo" branch --show-current
    git -C "$repo" remote -v
    git -C "$repo" diff
    sha256sum "$repo"/experiments/dp_opt/*.py "$repo"/experiments/dp_opt/*.sh \
        "$repo"/experiments/dp_opt/*.sbatch "$repo/kvcached/dp_namespace.py" \
        "$repo/kvcached/integration/vllm/patches.py" "$repo"/tests/test_dp_opt*.py
} > "$evidence/hardware-source.txt" 2>&1
python - <<'PY'
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError('Exactly two allocated GPUs are required')
if min(torch.cuda.get_device_properties(index).total_memory for index in range(2)) < 75 * 1024**3:
    raise RuntimeError('Dual BF16 EP=2 services require at least 75 GiB per GPU')
PY
if [[ "$mode" == independent || "$mode" == motivation ]]; then
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH python -m pytest -q \
        "$repo/tests/test_dp_namespace.py" "$repo/tests/test_dp_opt_benchmark.py" \
        "$repo/tests/test_dp_opt_independent_benchmark.py" "$repo/tests/test_dp_opt_motivation.py" \
        "$repo/tests/test_vllm_tp_world_size.py" \
        "$repo/tests/test_vllm_async_unmap_ordering.py" "$repo/tests/test_vllm_engine_core_shutdown.py"
fi
checkpoint="$(python -c 'from qwen_common import checkpoint_path; print(checkpoint_path())')"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu --format=csv --loop=1 \
    > "$evidence/gpu-timeline.csv" &
monitor_pid=$!
for service_index in $(seq 1 "$service_count"); do
    service="instance-$service_index"
    port="$((ports + (service_index - 1) * 10))"
    mkdir -p "$temporary_root/$service"
    DP_OPT_SERVICE="$service" KVCACHED_IPC_NAME="kq${SLURM_JOB_ID}_s$service_index" \
        TMPDIR="$temporary_root/$service" VLLM_PORT="$((port + 2))" \
        setsid python -m vllm.entrypoints.cli.main serve "$checkpoint" \
        --host 127.0.0.1 --port "$port" --api-server-count 1 \
        --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507-$service" \
        --tensor-parallel-size 1 --data-parallel-size 2 --data-parallel-size-local 2 \
        --data-parallel-backend mp --data-parallel-address 127.0.0.1 \
        --data-parallel-rpc-port "$((port + 1))" \
        --enable-expert-parallel --enable-ep-weight-filter \
        --all2all-backend allgather_reducescatter --moe-backend triton \
        --worker-cls "$worker_class" --dtype bfloat16 \
        --no-enable-dbo --no-enable-eplb \
        "${execution_args[@]}" --max-model-len "$max_model_len" --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_num_batched_tokens" \
        --gpu-memory-utilization 0.43 --kv-cache-memory-bytes "$kv_cache_bytes" \
        --no-enable-prefix-caching --no-async-scheduling \
        > "$deploy/logs/dual-qwen-$job_tag-$service.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    healthy=0
    for attempt in $(seq 1 180); do
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -100 "$deploy/logs/dual-qwen-$job_tag-$service.log" >&2
            exit 1
        fi
        if curl --fail --silent --max-time 5 "http://127.0.0.1:$port/health" >/dev/null; then
            healthy=1
            break
        fi
        sleep 10
    done
    if [[ "$healthy" != 1 ]]; then echo "$service readiness timeout" >&2; exit 1; fi
done
printf '%s %s %s\n' "$(hostname)" "$ports" "$((ports + 10))" > "$endpoint"
if [[ "$mode" == motivation ]]; then
    if [[ "${DP_OPT_GRAPH_CURVE:-0}" == 1 ]]; then
        python "$repo/experiments/dp_opt/motivation.py" \
            --url "http://127.0.0.1:$ports" --checkpoint "$checkpoint" --evidence-root "$evidence" \
            --requests 128 --input-tokens 1024 --output-tokens 512 \
            --case-name "graph-curve-$run_label" --expected-cudagraph-mode FULL --capture-sizes "$capture_sizes"
    elif [[ "${DP_OPT_GRAPH_SUITE:-0}" == 1 ]]; then
        python "$repo/experiments/dp_opt/graph_sweep.py" \
            --url "http://127.0.0.1:$ports" --checkpoint "$checkpoint" --evidence-root "$evidence"
    elif [[ "${DP_OPT_BATCH_SUITE:-0}" == 1 ]]; then
        resume_args=()
        if [[ -n "${DP_OPT_RESUME_ROOT:-}" ]]; then
            resume_args=(--resume-root "$DP_OPT_RESUME_ROOT")
        fi
        python "$repo/experiments/dp_opt/batch_sweep.py" \
            --url "http://127.0.0.1:$ports" --checkpoint "$checkpoint" --evidence-root "$evidence" \
            "${resume_args[@]}"
    elif [[ "${DP_OPT_LENGTH_SUITE:-0}" == 1 ]]; then
        for lengths in '128 256 short' '128 1024 output_long' '8192 1024 input_output_long'; do
            read -r input_tokens output_tokens case_name <<< "$lengths"
            python "$repo/experiments/dp_opt/motivation.py" \
                --url "http://127.0.0.1:$ports" --checkpoint "$checkpoint" --evidence-root "$evidence" \
                --input-tokens "$input_tokens" --output-tokens "$output_tokens" \
                --case-name "$case_name" --extremes-only --profile-phases
        done
    else
        python "$repo/experiments/dp_opt/motivation.py" \
            --url "http://127.0.0.1:$ports" --checkpoint "$checkpoint" --evidence-root "$evidence" \
            --input-tokens "${DP_OPT_INPUT_TOKENS:-128}" --output-tokens "${DP_OPT_OUTPUT_TOKENS:-256}"
    fi
    printf '%s\n' "$SLURM_JOB_ID" > "$deploy/state/motivation-verified"
elif [[ "$mode" == independent ]]; then
    python "$repo/experiments/dp_opt/independent_benchmarks.py" \
        --ports "$ports" "$((ports + 10))" --checkpoint "$checkpoint" --evidence-root "$evidence" \
        --rounds "${DP_OPT_BENCHMARK_ROUNDS:-3}"
    printf '%s\n' "$SLURM_JOB_ID" > "$deploy/state/independent-benchmark-verified"
else
    python "$repo/experiments/dp_opt/benchmark.py" --ports "$ports" "$((ports + 10))" \
        --checkpoint "$checkpoint" --evidence-root "$evidence"
    printf '%s\n' "$SLURM_JOB_ID" > "$deploy/state/dual-qwen-verified"
fi
