#!/usr/bin/env bash
#SBATCH --job-name=qwen-agentic-trace
#SBATCH --partition=gpu_a800
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
set -euo pipefail
root="${KVCACHED_DATA_ROOT:-$HOME/run/kvcached-paracloud}"
repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"
evidence="$root/state/live-trace-$SLURM_JOB_ID"
mkdir -p "$evidence"
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$evidence/exit-status"
    exit "$result"
}
trap finish EXIT
printf 'SLURM_JOB_ID=%s\nSLURM_JOB_GPUS=%s\nCUDA_VISIBLE_DEVICES=%s\nCUDA_DEVICE_ORDER=%s\n' \
    "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}" "${CUDA_DEVICE_ORDER:-}" \
    > "$evidence/gpu-binding-before-load.txt"
scontrol show job -dd "$SLURM_JOB_ID" >> "$evidence/gpu-binding-before-load.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free --format=csv,noheader,nounits \
    > "$evidence/gpu-memory-before-load.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv \
    > "$evidence/gpu-processes-before-load.csv" 2>&1 || true
if ! awk -F, 'NF != 4 || $2 !~ /A800/ || ($4 + 0) < 70000 { invalid=1 }
    END { exit (NR != 2 || invalid) }' "$evidence/gpu-memory-before-load.csv"; then
    printf '{"stage":"a800_resource_preflight_rejected","required_free_mib_per_gpu":70000}\n' > "$evidence/phase.json"
    printf 'Allocated A800 pair lacks initial free memory; preserve bindings and existing GPU processes.\n' >&2
    exit 1
fi
export DP_OPT_MAX_MODEL_LEN=131072 DP_OPT_KV_CACHE_BYTES=25769803776
export DP_OPT_TRACE_BUDGET_MIB=20480 DP_OPT_REQUEST_TIMEOUT_S=900 DP_OPT_DEADLINE_S=120
export DP_OPT_RELEASE_UNUSED_CACHE=1
export KVCACHED_GPU_UTILIZATION=0.98
export DP_OPT_TRACE_PATH="$root/state/agentic-100k-input/trace.json"
export DP_OPT_PROMPT_BANK="$root/state/agentic-100k-input/prompt-bank.json"
export DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_CAPACITY_PILOT=1
printf '{"stage":"a800_capacity_pilot_running"}\n' > "$evidence/phase.json"
bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
source "$evidence/capacity/capacity-config.env"
if [[ "${DP_OPT_TRANSFER_PILOT:-0}" == 1 ]]; then
    test "$DP_OPT_TRACE_BUDGET_MIB" -ge 11328
    export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_CALIBRATION=0 DP_OPT_TRACE_BUDGET_MIB=11328
    printf '{"stage":"transfer_cost_pilot_running"}\n' > "$evidence/phase.json"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
    printf '{"stage":"transfer_cost_pilot_complete"}\n' > "$evidence/phase.json"
    exit 0
fi
if [[ "${DP_OPT_FROZEN_TRACE_REPLAY:-0}" == 1 ]]; then
    test "$(sha256sum "$DP_OPT_FROZEN_TRACE_PATH" | cut -d' ' -f1)" = "$DP_OPT_FROZEN_TRACE_SHA256"
    test "$(sha256sum "$(dirname "$DP_OPT_FROZEN_TRACE_PATH")/rate-decision.json" | cut -d' ' -f1)" = "$DP_OPT_FROZEN_RATE_DECISION_SHA256"
    test "$DP_OPT_TRACE_BUDGET_MIB" -ge "$DP_OPT_FROZEN_BUDGET_MIB"
    mkdir "$evidence/frozen"
    cp "$DP_OPT_FROZEN_TRACE_PATH" "$evidence/frozen/trace.json"
    cp "$(dirname "$DP_OPT_FROZEN_TRACE_PATH")/rate-decision.json" "$evidence/frozen/rate-decision.json"
    sha256sum "$evidence/frozen/trace.json" "$evidence/frozen/rate-decision.json" > "$evidence/frozen/source.sha256"
    printf '{"source_job":241086,"trace_sha256":"%s","budget_mib_per_gpu":%s,"max_moves":%s,"candidate_limit":%s,"scope":"same_frozen_trace_new_control_path_no_rate_reselection"}\n' \
        "$DP_OPT_FROZEN_TRACE_SHA256" "$DP_OPT_FROZEN_BUDGET_MIB" "$DP_OPT_MAX_MOVES" "$DP_OPT_CANDIDATE_LIMIT" \
        > "$evidence/frozen-replay.json"
    export DP_OPT_TRACE_PATH="$evidence/frozen/trace.json" DP_OPT_TRACE_BUDGET_MIB="$DP_OPT_FROZEN_BUDGET_MIB"
    export DP_OPT_TARGET_CONV_MIB=4347 DP_OPT_TRACE_REPEATS=1
    export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_CALIBRATION=0
    export DP_OPT_RANK_LOCAL_ADMISSION=1 DP_OPT_CONCURRENT_ADMISSION=1
    printf '{"stage":"formal_trace_running"}\n' > "$evidence/phase.json"
    bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
    printf '{"stage":"formal_trace_complete"}\n' > "$evidence/phase.json"
    exit 0
fi
export DP_OPT_CALIBRATION_CONV_RATES="0.42 0.51 0.61 0.70"
export DP_OPT_CALIBRATION_CODE_RATE=0.004
export DP_OPT_CALIBRATION_DURATION_S=1200
printf '{"conv_rates_rps":[0.42,0.51,0.61,0.70],"code_rate_rps":0.004,"source":"240982_fixed_30_120s_arrival_subset_all_completed_before_Code","reference_conv_rate_rps":0.80,"reference_mean_reserved_mib_seconds":7181.851477095108,"reference_requests":82,"prior_target_conv_reserved_mib":4347,"multipliers":[0.70,0.85,1,1.15],"central_formula":"target/mean(block_rounded_full_KV_MiB*exact_resident_seconds)","rate_precision_rps":0.01,"Code_rate_unchanged":true,"seed_unchanged":20261021,"prior_calibration_duration_s":300,"calibration_duration_s":1200,"prior_nonmonotone_calibration_job":241036,"scope":"clean_lower_interference_rate_estimate_then_full_mixed_openloop_validation_not_stability_or_benefit_guarantee"}\n' \
    > "$evidence/calibration-bracket.json"
export DP_OPT_CAPACITY_PILOT=0 DP_OPT_RATE_CALIBRATION=1
export DP_OPT_RANK_LOCAL_ADMISSION=1
export DP_OPT_CONCURRENT_ADMISSION=1
printf '{"stage":"open_loop_calibration_running"}\n' > "$evidence/phase.json"
bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
printf '{"stage":"calibration_complete_freezing_trace"}\n' > "$evidence/phase.json"
"$root/env/bin/python" "$repo/experiments/dp_opt/freeze_agentic_calibration.py" \
    --calibration "$evidence/calibration" --inputs "$root/state/agentic-100k-input" \
    --output "$evidence/frozen"
export DP_OPT_TRACE_PATH="$evidence/frozen/trace.json" DP_OPT_TRACE_REPEATS=1
printf 'export DP_OPT_TRACE_PATH="%s"\nexport DP_OPT_TRACE_REPEATS=1\n' "$DP_OPT_TRACE_PATH" \
    > "$evidence/launch-config.env"
export DP_OPT_RATE_PILOT=0 DP_OPT_RATE_CALIBRATION=0
printf '{"stage":"formal_trace_running"}\n' > "$evidence/phase.json"
bash "$repo/experiments/dp_opt/trace-benefit.sbatch"
printf '{"stage":"formal_trace_complete"}\n' > "$evidence/phase.json"
