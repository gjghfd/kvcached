import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from verify_agentic_calibration import verify


def interval_overlap(start, end, intervals):
    return any(start < right and end > left for left, right in intervals)


def interval_peak(intervals):
    events = sorted([(start, 1) for start, end in intervals if start < end]
                    + [(end, -1) for start, end in intervals if start < end])
    active, peak = 0, 0
    for timestamp, change in events:
        active += change
        peak = max(peak, active)
    return peak


def summarize_control(requests, decisions):
    admitted = [decision for decision in decisions
                if decision["admitted"] or decision.get("submitted", False)]
    by_request = {decision["request_id"]: decision for decision in admitted}
    if len(by_request) != len(admitted) or set(by_request) != {request["request_id"] for request in requests}:
        raise ValueError("Control analysis requires one submission per completed request")
    code_prefill = [(request.get("reserved_at", request["submitted_at"]), request["first_token_at"])
                    for request in requests if request["model"] == 0]
    code_resident = [(request.get("reserved_at", request["submitted_at"]), request["completed_at"])
                     for request in requests if request["model"] == 0]
    records = []
    for request in requests:
        decision = by_request[request["request_id"]]
        duration = decision["end_to_end_control_s"]
        end = request["arrived_at"] + decision["elapsed_from_arrival_s"]
        start = end - duration
        if duration < 0 or decision["elapsed_from_arrival_s"] < duration - 1e-6:
            raise ValueError("Control interval precedes request arrival or has negative duration")
        before_submit = min(duration, max(0, request["submitted_at"] - start))
        reserved = request.get("reserved_at", request["submitted_at"])
        records.append({"request_id": request["request_id"], "model": request["model"],
                        "queue_s": reserved - request["arrived_at"],
                        "queue_before_submission_s": request["submitted_at"] - request["arrived_at"],
                        "submission_to_reservation_s": reserved - request["submitted_at"],
                        "control_s": duration, "control_before_submission_s": before_submit,
                        "control_after_submission_s": duration - before_submit,
                        "control_start_at_derived": start, "control_end_at_derived": end,
                        "control_overlaps_code_prefill": interval_overlap(start, end, code_prefill),
                        "arrival_during_code_prefill": any(left <= request["arrived_at"] < right
                                                           for left, right in code_prefill),
                        "reservation_pending_at_submit": decision.get("reservation_pending", False),
                        "reservation_time_source": request.get("reservation_time_source", "legacy_submitted_at")})

    def group(selected):
        fields = ("queue_s", "queue_before_submission_s", "submission_to_reservation_s", "control_s",
                  "control_before_submission_s", "control_after_submission_s")
        return {"requests": len(selected),
                "mean": {field: statistics.mean(record[field] for record in selected) if selected else None
                         for field in fields},
                "total": {field: sum(record[field] for record in selected) for field in fields}}

    conv = [record for record in records if record["model"] == 1]
    return {"by_model": {str(model): group([record for record in records if record["model"] == model])
                         for model in (0, 1)},
            "conv_by_control_overlap": {str(overlaps).lower(): group([
                record for record in conv if record["control_overlaps_code_prefill"] == overlaps])
                                         for overlaps in (False, True)},
            "peak_code_prefill_intervals": interval_peak(code_prefill),
            "peak_code_resident_intervals": interval_peak(code_resident),
            "reservation_pending_submissions": sum(record["reservation_pending_at_submit"] for record in records),
            "reservation_time_sources": dict(collections.Counter(record["reservation_time_source"] for record in records)),
            "records": records,
            "scope": "derived_client_control_intervals_include_RPC_kernel_boundary_and_wait_not_pure_CPU_or_GPU_kernel_time"}


def summarize_monitoring(samples, transitions, duration):
    accepted = [sample["elapsed_s"] for sample in samples if 0 <= sample["elapsed_s"] < duration]
    rejected = [sample["elapsed_s"] for sample in transitions if 0 <= sample["elapsed_s"] < duration]
    boundaries = [0, *accepted, duration]
    if any(right < left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Monitoring samples must be ordered inside the arrival window")
    return {"accepted_samples_in_arrival_window": len(accepted),
            "transition_samples_preserved_in_arrival_window": len(rejected),
            "maximum_accepted_sample_gap_s_including_window_boundaries": max(
                right - left for left, right in zip(boundaries, boundaries[1:])),
            "segments": [{"start_s": start, "end_s": end,
                          "accepted_samples": sum(start <= timestamp < end for timestamp in accepted),
                          "transition_samples_preserved": sum(start <= timestamp < end for timestamp in rejected)}
                         for start, end in ((0, duration / 3), (duration / 3, duration / 2),
                                           (duration / 2, duration * 5 / 6), (duration * 5 / 6, duration))],
            "scope": "missing_samples_remain_unknown_not_zero_KV_or_queue_no_time_weighted_stability_certificate"}


def summarize_residency_window(requests, origin, start_s, end_s, block_size, block_bytes):
    if (not all(math.isfinite(value) for value in (origin, start_s, end_s, block_size, block_bytes))
            or not 0 <= start_s < end_s or min(block_size, block_bytes) <= 0):
        raise ValueError("Residency reconstruction requires a finite positive window and block geometry")
    results = []
    for model in (0, 1):
        selected = [request for request in requests if request["model"] == model]
        reserved_seconds, resident_seconds, events = 0, 0, []
        for request in selected:
            reserved = request.get("reserved_at", request["submitted_at"])
            completed = request["completed_at"]
            if not math.isfinite(reserved) or not math.isfinite(completed) or reserved > completed:
                raise ValueError("Invalid request lifetime for residency reconstruction")
            left = max(reserved - origin, start_s)
            right = min(completed - origin, end_s)
            if left >= right:
                continue
            memory = math.ceil((request["prompt_tokens"] + len(request["token_ids"])) / block_size) * block_bytes / 1024**2
            reserved_seconds += memory * (right - left)
            resident_seconds += right - left
            events.extend(((left, memory), (right, -memory)))
        active, peak = 0, 0
        for timestamp, change in sorted(events):
            active += change
            peak = max(peak, active)
        results.append({"model": model,
                        "time_average_logical_reserved_mib": reserved_seconds / (end_s - start_s),
                        "time_average_resident_count": resident_seconds / (end_s - start_s),
                        "peak_logical_reserved_mib": peak,
                        "reservation_time_sources": dict(collections.Counter(
                            request.get("reservation_time_source", "legacy_submitted_at") for request in selected))})
    return {"start_s": start_s, "end_s": end_s, "models": results,
            "scope": "full_logical_reservation_to_client_completion_not_allocator_release_or_physical_mapping_no_missing_sample_imputation"}


def analyze(root, inputs, allow_partial=False):
    verified = verify(root, inputs, allow_partial=allow_partial)
    trials = []
    for index, trial in enumerate(verified["trials"]):
        directory = root / f"calibration-{index}"
        requests = json.loads((directory / "requests.json").read_text())
        decisions = json.loads((directory / "decisions.json").read_text())
        population = json.loads((directory / "population.json").read_text())
        workload = json.loads((directory / "workload.json").read_text())
        geometry = decisions[0]["observations"][0][0]
        duration = trial["duration_s"]
        residency = [summarize_residency_window(requests, workload["origin"], start, duration,
                                               geometry["block_size"], geometry["block_bytes"])
                     for start in (0, duration / 3)]
        reconstructed_conv = residency[1]["models"][1]["time_average_logical_reserved_mib"]
        transitions_path = directory / "monitor-transitions.jsonl"
        transitions = [json.loads(line) for line in transitions_path.read_text().splitlines()] if transitions_path.exists() else []
        trials.append({"conv_rate_rps": trial["conv_rate_rps"], "metrics": trial["metrics"],
                       "mean_sampled_conv_reserved_mib": trial["mean_sampled_conv_reserved_mib"],
                       "global_enough_local_blocked_code_samples": trial["global_enough_local_blocked_code_samples"],
                       "control": summarize_control(requests, decisions),
                       "monitoring": summarize_monitoring(population, transitions, duration),
                       "request_lifetime_residency_windows": residency,
                       "conv_sample_mean_minus_reconstructed_time_mean_mib":
                       trial["mean_sampled_conv_reserved_mib"] - reconstructed_conv})
    return {"status": "verified_calibration_control_analyzed", "calibration_complete": verified["calibration_complete"],
            "guardrails": verified["guardrails"], "trials": trials,
            "scope": "finite_calibration_diagnostics_not_four_policy_benefit_or_causal_kernel_profile"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root, args.inputs, allow_partial=args.allow_partial)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
