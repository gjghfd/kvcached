#!/usr/bin/env bash
#SBATCH --job-name=qwen-trace-benefit
#SBATCH --partition=gpu_a800,gpu_h100,gpu_h200
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
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
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENGINE_READY_TIMEOUT_S=1800
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KVCACHED_MIN_RESERVED_PAGES=0 KVCACHED_MAX_RESERVED_PAGES=0 KVCACHED_PAGE_PREALLOC_ENABLED=false
export DP_OPT_PAUSE_SYNC_INTERVAL=32
evidence="${DP_OPT_PHASE_EVIDENCE_ROOT:-$root/state/live-trace-$SLURM_JOB_ID}"
if [[ "${DP_OPT_CAPACITY_PILOT:-0}" == 1 ]]; then
    evidence="$evidence/capacity"
elif [[ "${DP_OPT_RATE_PILOT:-0}" == 1 ]]; then
    evidence="$evidence/pilot"
elif [[ "${DP_OPT_RATE_CALIBRATION:-0}" == 1 ]]; then
    evidence="$evidence/calibration"
elif [[ "${DP_OPT_TRANSFER_PILOT:-0}" == 1 ]]; then
    evidence="$evidence/transfer"
fi
mkdir -p "$evidence"
export DP_OPT_TRANSFER_ROOT="$(mktemp -d /tmp/dp-trace-$SLURM_JOB_ID.XXXXXX)"
cleanup() {
    result=$?
    trap - EXIT
    rm -f "$DP_OPT_TRANSFER_ROOT"/*.pt
    rmdir "$DP_OPT_TRANSFER_ROOT"
    printf '%s\n' "$result" > "$evidence/exit-status"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_PAUSE_SYNC_INTERVAL -u DP_OPT_KV_TRANSPORT python -m pytest -q \
    "$repo/tests/test_dp_opt_live_placement.py" "$repo/tests/test_dp_opt_live_benefit.py" \
    "$repo/tests/test_dp_opt_trace.py" "$repo/tests/test_dp_opt_ilp_reference.py" \
    "$repo/tests/test_dp_opt_migration_precheck.py" "$repo/tests/test_dp_opt_trace_planning.py" \
    "$repo/tests/test_dp_opt_candidate_revalidation.py" "$repo/tests/test_dp_opt_queue_fairness.py" \
    "$repo/tests/test_dp_opt_rebalance.py" "$repo/tests/test_dp_opt_amortized_rebalance.py" \
    "$repo/tests/test_dp_opt_headroom_rebalance.py"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_KV_TRANSPORT python -m pytest -q \
    "$repo/tests/test_dp_opt_admission_cost.py" "$repo/tests/test_dp_opt_wait_gate.py"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_KV_TRANSPORT python -m pytest -q \
    "$repo/tests/test_dp_opt_layout_cost.py" "$repo/tests/test_dp_opt_measured_control.py"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_KV_TRANSPORT python -m pytest -q \
    "$repo/tests/test_dp_opt_peer_transfer.py"
env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT \
    -u DP_OPT_PAUSE_SYNC_INTERVAL -u VLLM_USE_V2_MODEL_RUNNER -u DP_OPT_KV_TRANSPORT python -m pytest -q \
    "$repo/tests/test_vllm_tp_world_size.py" "$repo/tests/test_vllm_async_unmap_ordering.py" \
    "$repo/tests/test_vllm_engine_core_shutdown.py"
checkpoint="$(python -c 'from qwen_common import checkpoint_path; print(checkpoint_path())')"
if [[ "${DP_OPT_MEASURED_CONTROL:-0}" == 1 && "${DP_OPT_CAPACITY_PILOT:-0}" != 1 ]]; then
    test "$(sha256sum "$DP_OPT_MEASURED_CONTROL_ENV" | cut -d' ' -f1)" = "$DP_OPT_MEASURED_CONTROL_ENV_SHA256"
    source "$DP_OPT_MEASURED_CONTROL_ENV"
    test "$(sha256sum "$DP_OPT_MEASURED_CONTROL_CONFIG" | cut -d' ' -f1)" = "$DP_OPT_MEASURED_CONTROL_CONFIG_SHA256"
    cp "$DP_OPT_MEASURED_CONTROL_ENV" "$evidence/control-config.env"
    cp "$DP_OPT_MEASURED_CONTROL_CONFIG" "$evidence/control-config.json"
fi
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv > "$evidence/hardware.txt"
nvidia-smi topo -m > "$evidence/topology.txt"
date -Is > "$evidence/started-at"
sha256sum "$repo/experiments/dp_opt/live_"*.py "$repo/experiments/dp_opt/poisson_trace.py" \
    "$repo/experiments/dp_opt/trace-benefit.sbatch" "$repo/experiments/dp_opt/ilp_reference.py" \
    "$repo/experiments/dp_opt/worker.py" "$repo/kvcached/integration/vllm/patches.py" \
    "$root/state/sparse-fit-231467.json" "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}" > "$evidence/source.sha256"
if [[ -n "${DP_OPT_PROMPT_BANK:-}" ]]; then
    sha256sum "$DP_OPT_PROMPT_BANK" >> "$evidence/source.sha256"
fi
if [[ "${DP_OPT_FROZEN_TRACE_REPLAY:-0}" == 1 ]]; then
    sha256sum "$(dirname "$DP_OPT_FROZEN_TRACE_PATH")/rate-decision.json" >> "$evidence/source.sha256"
fi
if [[ "${DP_OPT_RATE_CALIBRATION:-0}" == 1 || "${DP_OPT_CAPACITY_PILOT:-0}" == 1 ]]; then
    sha256sum "$root/state/agentic-100k-input/conv-marks.json" \
        "$root/state/agentic-100k-input/code-marks.json" >> "$evidence/source.sha256"
fi
sha256sum "$repo/experiments/dp_opt/agentic-trace.sbatch" \
    "$repo/experiments/dp_opt/agentic-formal.sbatch" \
    "$repo/experiments/dp_opt/agentic-wait.sbatch" \
    "$repo/experiments/dp_opt/agentic_trace.py" \
    "$repo/experiments/dp_opt/agentic_capacity.py" \
    "$repo/experiments/dp_opt/freeze_agentic_calibration.py" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/peer_transfer.py" "$repo/experiments/dp_opt/transfer_pilot.py" \
    "$repo/experiments/dp_opt/agentic-transfer.sbatch" "$repo/experiments/dp_opt/agentic-echo.sbatch" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/verify_rebalance.py" "$repo/experiments/dp_opt/rebalance-trace.sbatch" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/amortized_rebalance.py" "$repo/experiments/dp_opt/verify_trace.py" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/headroom_rebalance.py" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/agentic-boundary.sbatch" >> "$evidence/source.sha256"
extra_args=()
if [[ "${DP_OPT_WORKAWARE_CADENCE_PILOT:-0}" == 1 || -n "${DP_OPT_WORKAWARE_FIT:-}" ]]; then
    while IFS= read -r filename; do
        sha256sum "$repo/experiments/dp_opt/$filename" >> "$evidence/source.sha256"
    done < <(python -c 'from workaware_evidence import runtime_files; print("\n".join(runtime_files()))')
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_KV_TRANSPORT \
        python -m pytest -q "$repo/tests/test_dp_opt_workaware_cost.py" \
        "$repo/tests/test_dp_opt_workaware_rebalance.py" "$repo/tests/test_dp_opt_workaware_ilp.py" \
        "$repo/tests/test_dp_opt_workaware_pilot.py" "$repo/tests/test_dp_opt_workaware_live.py"
fi
if [[ "${DP_OPT_WORKAWARE_CADENCE_PILOT:-0}" == 1 ]]; then
    extra_args+=(--workaware-cadence-pilot)
fi
if [[ -n "${DP_OPT_WORKAWARE_FIT:-}" ]]; then
    read -r -a workaware_rates <<< "${DP_OPT_WORKAWARE_RATE_PRIORS:?Freeze two external rate priors}"
    extra_args+=(--workaware-fit "$DP_OPT_WORKAWARE_FIT"
        --workaware-fit-sha256 "${DP_OPT_WORKAWARE_FIT_SHA256:?Freeze measured fit SHA}"
        --workaware-source-root "${DP_OPT_WORKAWARE_SOURCE_ROOT:?Supply independently verifiable raw calibration}"
        --workaware-rate-priors "${workaware_rates[@]}")
fi
if [[ "${DP_OPT_STRESS_SWEEP:-0}" == 1 ]]; then
    read -r -a stress_outputs <<< "${DP_OPT_STRESS_OUTPUTS:-8192}"
    read -r -a stress_budgets <<< "${DP_OPT_STRESS_BUDGETS:-10240 11328}"
    read -r -a stress_multipliers <<< "${DP_OPT_STRESS_RATE_MULTIPLIERS:-1}"
    extra_args+=(--stress-sweep --stress-conv-marks "$root/state/agentic-100k-input/conv-marks.json"
        --stress-code-marks "$root/state/agentic-100k-input/code-marks.json"
        --calibration-conv-marks "$root/state/agentic-100k-input/conv-marks.json"
        --calibration-code-marks "$root/state/agentic-100k-input/code-marks.json"
        --stress-outputs "${stress_outputs[@]}" --stress-budgets "${stress_budgets[@]}"
        --stress-rate-multipliers "${stress_multipliers[@]}" --stress-code-rate "${DP_OPT_STRESS_CODE_RATE:-0.001}"
        --stress-window-s "${DP_OPT_STRESS_WINDOW_S:-1800}" --stress-warmup-s "${DP_OPT_STRESS_WARMUP_S:-600}"
        --seed "${DP_OPT_STRESS_SEED:-20261024}")
    if [[ -n "${DP_OPT_STRESS_FIXED_CONV_RATE:-}" ]]; then
        extra_args+=(--stress-fixed-conv-rate "$DP_OPT_STRESS_FIXED_CONV_RATE")
    fi
    sha256sum "$repo/experiments/dp_opt/stress-sweep.sbatch" "$repo/experiments/dp_opt/stress_trace.py" \
        "$repo/experiments/dp_opt/stress_sweep.py" "$repo/experiments/dp_opt/analyze_stress_sweep.py" \
        "$repo/experiments/dp_opt/verify_stress_sweep.py" \
        "$root/state/agentic-100k-input/conv-marks.json" "$root/state/agentic-100k-input/code-marks.json" >> "$evidence/source.sha256"
    env -u ENABLE_KVCACHED -u KVCACHED_AUTOPATCH -u DP_OPT_LIVE_PLACEMENT -u DP_OPT_KV_TRANSPORT \
        python -m pytest -q "$repo/tests/test_dp_opt_stress_trace.py" "$repo/tests/test_dp_opt_stress_sweep.py"
fi
if [[ "${DP_OPT_TRANSFER_PILOT:-0}" == 1 && "${DP_OPT_CAPACITY_PILOT:-0}" != 1 ]]; then
    extra_args+=(--transfer-pilot --transfer-conv-marks "$root/state/agentic-100k-input/conv-marks.json")
    if [[ "${DP_OPT_LAYOUT_COST_PILOT:-0}" == 1 ]]; then
        extra_args=(--layout-cost-pilot --admission-conv-marks "$root/state/agentic-100k-input/conv-marks.json")
    elif [[ "${DP_OPT_ADMISSION_COST_PILOT:-0}" == 1 ]]; then
        extra_args=(--admission-cost-pilot --admission-conv-marks "$root/state/agentic-100k-input/conv-marks.json")
    elif [[ "${DP_OPT_TRANSFER_BOUNDARY_PILOT:-0}" == 1 ]]; then
        extra_args+=(--transfer-transports dp_echo --transfer-pause-modes native_keep default32_boundary)
    elif [[ "${DP_OPT_TRANSFER_ECHO_PILOT:-0}" == 1 ]]; then
        extra_args+=(--transfer-transports dp_allgatherv dp_echo)
    fi
fi
sha256sum "$repo/experiments/dp_opt/admission_cost_pilot.py" \
    "$repo/experiments/dp_opt/agentic-admission-cost.sbatch" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/layout_cost_pilot.py" \
    "$repo/experiments/dp_opt/agentic-layout-cost.sbatch" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/freeze_measured_control.py" \
    "$repo/experiments/dp_opt/agentic-measured.sbatch" >> "$evidence/source.sha256"
sha256sum "$repo/experiments/dp_opt/queue_fairness.py" \
    "$repo/experiments/dp_opt/verify_queue_fairness.py" >> "$evidence/source.sha256"
if [[ -n "${DP_OPT_TRACE_POLICIES:-}" ]]; then
    read -r -a trace_policies <<< "$DP_OPT_TRACE_POLICIES"
    extra_args+=(--policies "${trace_policies[@]}")
fi
if [[ -n "${DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S:-}" ]]; then
    extra_args+=(--fairness-queue-threshold-s "$DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S")
    sha256sum "$repo/experiments/dp_opt/agentic-fair-baselines.sbatch" >> "$evidence/source.sha256"
fi
if [[ "${DP_OPT_REVALIDATE_PREPARED:-0}" == 1 ]]; then
    sha256sum "$repo/experiments/dp_opt/agentic-revalidated.sbatch" \
        "$repo/experiments/dp_opt/verify_candidate_revalidation.py" >> "$evidence/source.sha256"
fi
if [[ "${DP_OPT_WAIT_AWARE:-0}" == 1 ]]; then
    extra_args+=(--wait-aware)
fi
if [[ "${DP_OPT_PREPLAN:-0}" == 1 ]]; then
    extra_args+=(--preplan)
fi
if [[ "${DP_OPT_REVALIDATE_PREPARED:-0}" == 1 && "${DP_OPT_CAPACITY_PILOT:-0}" != 1 ]]; then
    extra_args+=(--revalidate-prepared)
fi
if [[ -n "${DP_OPT_LAYOUT_COST_LOOKUP:-}" ]]; then
    extra_args+=(--layout-cost-lookup "$DP_OPT_LAYOUT_COST_LOOKUP"
        --layout-cost-lookup-sha256 "${DP_OPT_LAYOUT_COST_LOOKUP_SHA256:?Freeze layout lookup SHA256}"
        --layout-cost-source-root "${DP_OPT_LAYOUT_COST_SOURCE_ROOT:?Supply independently verifiable raw sources}")
    sha256sum "$DP_OPT_LAYOUT_COST_LOOKUP" >> "$evidence/source.sha256"
fi
if [[ -n "${DP_OPT_MEASURED_CONTROL_CONFIG:-}" ]]; then
    extra_args+=(--measured-control-config "$DP_OPT_MEASURED_CONTROL_CONFIG"
        --measured-control-config-sha256 "$DP_OPT_MEASURED_CONTROL_CONFIG_SHA256")
    sha256sum "$DP_OPT_MEASURED_CONTROL_CONFIG" >> "$evidence/source.sha256"
fi
if [[ -n "${DP_OPT_WAIT_INCOMING_HANDSHAKE_S:-}" ]]; then
    extra_args+=(--wait-incoming-handshake-s "$DP_OPT_WAIT_INCOMING_HANDSHAKE_S")
fi
if [[ -n "${DP_OPT_WAIT_OLD_TRANSACTION_FIXED_S:-}" ]]; then
    extra_args+=(--wait-old-transaction-fixed-s "$DP_OPT_WAIT_OLD_TRANSACTION_FIXED_S")
fi
if [[ -n "${DP_OPT_WAIT_COPY_SECONDS_PER_GIB:-}" ]]; then
    extra_args+=(--wait-copy-seconds-per-gib "$DP_OPT_WAIT_COPY_SECONDS_PER_GIB")
fi
if [[ -n "${DP_OPT_WAIT_SOLVE_FLOOR_S:-}" ]]; then
    extra_args+=(--wait-solve-floor-s "$DP_OPT_WAIT_SOLVE_FLOOR_S")
fi
if [[ -n "${DP_OPT_MOVE_FIXED_S:-}" ]]; then
    extra_args+=(--move-fixed-s "$DP_OPT_MOVE_FIXED_S")
fi
if [[ -n "${DP_OPT_MOVE_BANDWIDTH_GIB_S:-}" ]]; then
    extra_args+=(--move-bandwidth-gib-s "$DP_OPT_MOVE_BANDWIDTH_GIB_S")
fi
if [[ -n "${DP_OPT_TRACE_SEED:-}" ]]; then
    extra_args+=(--seed "$DP_OPT_TRACE_SEED")
fi
if [[ -n "${DP_OPT_PROMPT_BANK:-}" ]]; then
    extra_args+=(--prompt-bank "$DP_OPT_PROMPT_BANK")
fi
if [[ "${DP_OPT_RATE_PILOT:-0}" == 1 ]]; then
    extra_args+=(--pilot)
fi
if [[ "${DP_OPT_RATE_CALIBRATION:-0}" == 1 || "${DP_OPT_CAPACITY_PILOT:-0}" == 1 ]]; then
    extra_args+=(--calibration-conv-marks "$root/state/agentic-100k-input/conv-marks.json")
    extra_args+=(--calibration-code-marks "$root/state/agentic-100k-input/code-marks.json")
fi
if [[ "${DP_OPT_CAPACITY_PILOT:-0}" == 1 ]]; then
    extra_args+=(--capacity-pilot)
fi
if [[ -n "${DP_OPT_TARGET_CONV_MIB:-}" ]]; then
    extra_args+=(--pilot-target-conv-mib "$DP_OPT_TARGET_CONV_MIB")
fi
if [[ -n "${DP_OPT_CALIBRATION_CONV_RATES:-}" ]]; then
    read -r -a calibration_rates <<< "$DP_OPT_CALIBRATION_CONV_RATES"
    extra_args+=(--calibration-conv-rates "${calibration_rates[@]}")
fi
if [[ -n "${DP_OPT_CALIBRATION_CODE_RATE:-}" ]]; then
    extra_args+=(--calibration-code-rate "$DP_OPT_CALIBRATION_CODE_RATE")
fi
if [[ -n "${DP_OPT_CALIBRATION_DURATION_S:-}" ]]; then
    extra_args+=(--calibration-duration-s "$DP_OPT_CALIBRATION_DURATION_S")
fi
if [[ -n "${DP_OPT_MAX_MOVES:-}" ]]; then
    extra_args+=(--max-moves "$DP_OPT_MAX_MOVES")
fi
if [[ -n "${DP_OPT_CANDIDATE_LIMIT:-}" ]]; then
    extra_args+=(--candidate-limit "$DP_OPT_CANDIDATE_LIMIT")
fi
extra_args+=(--rebalance-max-moves "${DP_OPT_REBALANCE_MAX_MOVES:-4}"
    --rebalance-overhead-fraction "${DP_OPT_REBALANCE_OVERHEAD_FRACTION:-0.01}"
    --rebalance-headroom-quantile "${DP_OPT_REBALANCE_HEADROOM_QUANTILE:-0.95}"
    --rebalance-headroom-history-limit "${DP_OPT_REBALANCE_HEADROOM_HISTORY_LIMIT:-64}"
    --rebalance-candidate-limit "${DP_OPT_REBALANCE_CANDIDATE_LIMIT:-8}"
    --rebalance-min-gain-s "${DP_OPT_REBALANCE_MIN_GAIN_S:-1}"
    --rebalance-retry-s "${DP_OPT_REBALANCE_RETRY_S:-2}"
    --rebalance-cooldown-s "${DP_OPT_REBALANCE_COOLDOWN_S:-10}")
if [[ -n "${DP_OPT_REBALANCE_PAUSE_FIXED_S:-}" ]]; then
    extra_args+=(--rebalance-pause-fixed-s "$DP_OPT_REBALANCE_PAUSE_FIXED_S")
fi
python "$repo/experiments/dp_opt/live_trace.py" --checkpoint "$checkpoint" \
    --fit "$root/state/sparse-fit-231467.json" --trace "$DP_OPT_TRACE_PATH" --output "$evidence" \
    --max-model-len "${DP_OPT_MAX_MODEL_LEN:-32768}" \
    --cache-bytes "${DP_OPT_KV_CACHE_BYTES:-4294967296}" \
    --kv-budget-mib "${DP_OPT_TRACE_BUDGET_MIB:-2112}" \
    --request-timeout-s "${DP_OPT_REQUEST_TIMEOUT_S:-900}" \
    --deadline-s "${DP_OPT_DEADLINE_S:-120}" \
    --repeats "${DP_OPT_TRACE_REPEATS:-1}" --port "$((30000 + SLURM_JOB_ID % 1000 * 20))" \
    "${extra_args[@]}"
