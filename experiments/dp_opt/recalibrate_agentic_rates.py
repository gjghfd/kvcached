import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from verify_agentic_calibration import verify


def code_rate_from_residency(requests, maximum_rate=.02, target_resident_count=.25):
    if (not requests or any(not math.isfinite(value) or value <= 0
                            for value in (maximum_rate, target_resident_count))):
        raise ValueError("Require Code requests and positive finite rate/occupancy limits")
    services = []
    for request in requests:
        if request["model"] != 0 or request.get("reservation_time_source") != "scheduler_monotonic":
            raise ValueError("Code rate requires exact scheduler reservation timestamps")
        resident = request["completed_at"] - request["reserved_at"]
        if not math.isfinite(resident) or resident <= 0:
            raise ValueError("Code resident time must be finite and positive")
        services.append(resident)
    measured = statistics.mean(services)
    raw_rate = min(maximum_rate, target_resident_count / measured)
    rate = math.floor(raw_rate * 1000) / 1000
    if rate <= 0:
        raise ValueError("Measured Code service needs a rate below the supported millirps precision")
    return {"code_rate_rps": rate, "code_rate_raw_rps": raw_rate,
            "code_mean_resident_s": measured, "code_resident_samples": len(services),
            "target_code_resident_count": target_resident_count, "maximum_code_rate_rps": maximum_rate,
            "predicted_code_resident_count": rate * measured,
            "formula": "floor_0.001(min(maximum_rate,target_resident_count/mean_exact_Code_resident_seconds))",
            "scope": "finite_A800_service_recalibration_prespecified_occupancy_target_not_equilibrium_or_benefit_guarantee"}


def recalibrate(root, inputs):
    verified = verify(root, inputs)
    config = json.loads((root / "config.json").read_text())
    requests = [request for index in range(len(verified["trials"]))
                for request in json.loads((root / f"calibration-{index}/requests.json").read_text())
                if request["model"] == 0]
    setting = code_rate_from_residency(requests)
    sources = [root / "config.json", root / "calibration-summary.json",
               inputs / "code-marks.json", inputs / "conv-marks.json", inputs / "prompt-bank.json"]
    sources.extend(root / f"calibration-{index}/requests.json" for index in range(len(verified["trials"])))
    return {**setting, "conv_rates_rps": config["calibration_conv_rates"],
            "prior_code_rate_rps": config["calibration_code_rate"],
            "prior_predicted_code_resident_count": config["calibration_code_rate"] * setting["code_mean_resident_s"],
            "seed_unchanged": config["seed"], "calibration_duration_s_unchanged": config["calibration_duration_s"],
            "prior_guardrails": verified["guardrails"],
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "changes": "Code_rate_only_using_A800_residency_keep_Conv_bracket_seed_data_graph_budget_rule_and_guardrails"}


def conv_bracket_from_pressure(candidates, target_mib):
    if (not math.isfinite(target_mib) or target_mib <= 0 or len(candidates) < 2
            or any(not math.isfinite(candidate[field]) or candidate[field] <= 0
                   for candidate in candidates for field in ("conv_rate_rps", "mean_conv_reserved_mib"))):
        raise ValueError("Pressure refinement requires finite positive observations and target")
    ordered = sorted(candidates, key=lambda candidate: candidate["conv_rate_rps"])
    if len({candidate["conv_rate_rps"] for candidate in ordered}) != len(ordered):
        raise ValueError("Pressure refinement requires distinct observed rates")
    stable = [candidate for candidate in ordered if candidate["eligible"]]
    brackets = [(left, right) for left, right in zip(stable, stable[1:])
                if left["mean_conv_reserved_mib"] < target_mib < right["mean_conv_reserved_mib"]]
    if not brackets:
        raise ValueError("No queue-stable observed pressure bracket around target")
    lower, upper = min(brackets, key=lambda pair: pair[1]["conv_rate_rps"] - pair[0]["conv_rate_rps"])
    width = upper["conv_rate_rps"] - lower["conv_rate_rps"]
    central = lower["conv_rate_rps"] + width * (target_mib - lower["mean_conv_reserved_mib"]) / (
        upper["mean_conv_reserved_mib"] - lower["mean_conv_reserved_mib"])
    step = width / 8
    rates = [round(central + offset * step, 2) for offset in (-1, 0, 1, 2)]
    if (len(set(rates)) != 4 or any(not lower["conv_rate_rps"] < rate < upper["conv_rate_rps"]
                                   for rate in rates)):
        raise ValueError("Refined rate precision or observed bracket is insufficient")
    return {"conv_rates_rps": rates, "central_conv_rate_raw_rps": central,
            "step_raw_rps": step, "offsets": [-1, 0, 1, 2], "target_conv_reserved_mib": target_mib,
            "observed_stable_bracket": [lower, upper], "observed_candidates": ordered,
            "formula": "r*=r_lo+(target-K_lo)*(r_hi-r_lo)/(K_hi-K_lo); step=(r_hi-r_lo)/8; round_0.01(r*+[-1,0,1,2]*step)",
            "scope": "local_pressure_interpolation_before_policy_outcomes_not_linear_service_law_or_stability_guarantee"}


def refine_conv(root, inputs):
    verified = verify(root, inputs)
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "calibration-summary.json").read_text())
    candidates, sources = [], [root / "config.json", root / "calibration-summary.json",
                               inputs / "code-marks.json", inputs / "conv-marks.json", inputs / "prompt-bank.json"]
    for index, trial in enumerate(verified["trials"]):
        directory = root / f"calibration-{index}"
        samples = json.loads((directory / "population.json").read_text())
        duration = trial["duration_s"]
        early = [sample for sample in samples if duration / 3 <= sample["elapsed_s"] < duration / 2]
        late = [sample for sample in samples if duration * 5 / 6 <= sample["elapsed_s"] < duration]
        queue = lambda selection: statistics.mean(max(0, sample["unfinished"] - sample["active"]
                                                       + sample.get("capacity_pending", 0)) for sample in selection)
        growth = queue(late) - queue(early)
        conv_queue = trial["tenants"]["1"]["mean_queue_s"]
        drain = trial["metrics"]["drain_s"]
        candidates.append({"conv_rate_rps": trial["conv_rate_rps"],
                           "mean_conv_reserved_mib": trial["mean_sampled_conv_reserved_mib"],
                           "mean_conv_queue_s": conv_queue, "drain_s": drain,
                           "external_queue_growth_requests": growth,
                           "eligible": conv_queue <= 5 and drain <= 90 and growth <= 10})
        sources.extend(directory / filename for filename in ("requests.json", "population.json"))
    setting = conv_bracket_from_pressure(candidates, summary["target_conv_reserved_mib"])
    return {**setting, "code_rate_rps_unchanged": config["calibration_code_rate"],
            "seed_unchanged": config["seed"], "calibration_duration_s_unchanged": config["calibration_duration_s"],
            "prior_guardrails": verified["guardrails"],
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "changes": "Conv_bracket_only_from_completed_no_move_pressure_keep_Code_seed_data_runtime_budget_rule_and_guardrails"}


def long_window_bracket(root, inputs):
    verified = verify(root, inputs)
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "calibration-summary.json").read_text())
    candidates, sources = [], [root / "config.json", root / "calibration-summary.json",
                               inputs / "code-marks.json", inputs / "conv-marks.json", inputs / "prompt-bank.json"]
    for index, trial in enumerate(verified["trials"]):
        directory = root / f"calibration-{index}"
        requests = json.loads((directory / "requests.json").read_text())
        workload = json.loads((directory / "workload.json").read_text())
        decisions = json.loads((directory / "decisions.json").read_text())
        geometry = decisions[0]["observations"][0][0]
        origin, duration = workload["origin"], trial["duration_s"]
        start, end = duration / 10, duration * .4
        first_code = min(request["arrived_at"] - origin for request in requests if request["model"] == 0)
        selected = [request for request in requests if request["model"] == 1
                    and start <= request["arrived_at"] - origin < end]
        if (not selected or end >= first_code or any(request["completed_at"] - origin >= first_code
                or request.get("reservation_time_source") != "scheduler_monotonic" for request in selected)):
            continue
        costs = [math.ceil((request["prompt_tokens"] + len(request["token_ids"])) / geometry["block_size"])
                 * geometry["block_bytes"] / 1024**2 * (request["completed_at"] - request["reserved_at"])
                 for request in selected]
        mean_queue = statistics.mean(request["reserved_at"] - request["arrived_at"] for request in selected)
        candidates.append({"conv_rate_rps": trial["conv_rate_rps"], "selected_requests": len(selected),
                           "arrival_interval_s": [start, end], "first_code_arrival_s": first_code,
                           "latest_selected_completion_s": max(request["completed_at"] - origin for request in selected),
                           "mean_conv_queue_s": mean_queue, "mean_conv_reserved_mib_seconds": statistics.mean(costs),
                           "eligible": mean_queue <= 5})
        sources.extend(directory / filename for filename in ("requests.json", "workload.json", "decisions.json"))
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("No complete exact-timestamp fixed-arrival window before Code for rate estimate")
    reference = max(eligible, key=lambda candidate: candidate["conv_rate_rps"])
    central = summary["target_conv_reserved_mib"] / reference["mean_conv_reserved_mib_seconds"]
    rates = [round(central * factor, 2) for factor in (.7, .85, 1, 1.15)]
    if len(set(rates)) != 4 or min(rates) <= 0:
        raise ValueError("Long-window rate precision is insufficient")
    return {"conv_rates_rps": rates, "central_conv_rate_raw_rps": central,
            "multipliers": [.7, .85, 1, 1.15], "reference": reference, "all_observed_clean_candidates": candidates,
            "target_conv_reserved_mib": summary["target_conv_reserved_mib"],
            "code_rate_rps_unchanged": config["calibration_code_rate"], "seed_unchanged": config["seed"],
            "prior_calibration_duration_s": config["calibration_duration_s"], "calibration_duration_s": 1200,
            "prior_guardrails": verified["guardrails"],
            "formula": "r*=target/mean(block_rounded_full_KV_MiB*exact_resident_seconds); round_0.01(r*[.7,.85,1,1.15])",
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "scope": "fixed_arrival_subset_all_completions_before_first_Code_no_length_filter_clean_lower_interference_estimate_not_mixed_stability_or_benefit_guarantee",
            "changes": "extend_each_calibration_to_formal_1200s_window_and_measure_grid_from_clean_KV_seconds_keep_Code_seed_runtime_budget_rule_and_guardrail_thresholds"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refine-conv-bracket", action="store_true")
    mode.add_argument("--long-window-bracket", action="store_true")
    args = parser.parse_args()
    result = (long_window_bracket(args.root, args.inputs) if args.long_window_bracket else
              refine_conv(args.root, args.inputs) if args.refine_conv_bracket else recalibrate(args.root, args.inputs))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
