import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def background_pressure(population, decisions, start_s, end_s, budget_mib):
    if not population or not math.isfinite(budget_mib) or budget_mib <= 0 or end_s <= start_s:
        raise ValueError("Require observations, positive budget and measurement window")
    idle = [[set() for rank in range(2)] for model in range(2)]
    code_pool_mib = []
    for decision in decisions:
        for field in ("observations", "after_admission"):
            observations = decision.get(field)
            if not observations:
                continue
            for model in range(2):
                for rank, observation in enumerate(observations[model]):
                    if not observation["requests"]:
                        idle[model][rank].add(observation["mapped_bytes"] / 1024**2)
                    elif model == 0:
                        if len(observation["requests"]) != 1:
                            raise ValueError("Require isolated single-Code pool demand")
                        code_pool_mib.append(observation["mapped_bytes"] / 1024**2)
    if any(len(values) != 1 for model in idle for values in model) or not code_pool_mib:
        raise ValueError("Require measured constant idle pools and single-Code allocations")
    idle_mib = [[next(iter(values)) for values in model] for model in idle]
    duration = 0
    physical_mib_seconds = 0
    logical_mib_seconds = 0
    accepted = 0
    for current, following in zip(population, population[1:]):
        if following["elapsed_s"] <= current["elapsed_s"]:
            raise ValueError("Population timestamps must strictly increase")
        overlap = min(end_s, following["elapsed_s"]) - max(start_s, current["elapsed_s"])
        if (overlap <= 0 or sum(current["counts"][0])
                or current.get("admission_stable_across_observation") is False):
            continue
        conv_mapped = [current["mapped_gib"][rank] * 1024 - idle_mib[0][rank] for rank in range(2)]
        if any(value < idle_mib[1][rank] - 1e-6 for rank, value in enumerate(conv_mapped)):
            raise ValueError("Mapped background is smaller than measured idle pools")
        duration += overlap
        accepted += 1
        physical_mib_seconds += overlap * sum(conv_mapped) / 2
        logical_mib_seconds += overlap * sum(current["reserved_mib"][1]) / 2
    if duration <= 0:
        raise ValueError("No Code-absent background intervals in measurement window")
    physical = physical_mib_seconds / duration
    code = max(code_pool_mib)
    return {"evaluation_start_s": start_s, "evaluation_end_s": end_s,
            "background_covered_s": duration, "background_intervals": accepted,
            "idle_pool_mib": idle_mib, "budget_mib_per_gpu": budget_mib,
            "Conv_physical_mean_mib_per_gpu": physical,
            "Conv_logical_mean_mib_per_gpu": logical_mib_seconds / duration,
            "Code_single_physical_pool_mib": code,
            "local_physical_ratio": (physical + code) / budget_mib,
            "aggregate_physical_ratio": (2 * physical + code + sum(idle_mib[0]) / 2) / (2 * budget_mib),
            "scope": "time_weighted_left_hold_Code_absent_mean_DP_rank_background_not_stationarity_or_packability_certificate"}


def rate_for_ratio(pressure, original_rate, target_ratio, maximum_rate=None):
    if any(not math.isfinite(value) or value <= 0 for value in (original_rate, target_ratio)):
        raise ValueError("Require positive finite rate and ratio")
    idle = sum(pressure["idle_pool_mib"][1]) / 2
    measured = pressure["Conv_physical_mean_mib_per_gpu"]
    target = target_ratio * pressure["budget_mib_per_gpu"] - pressure["Code_single_physical_pool_mib"]
    if target <= idle or measured <= idle:
        raise ValueError("Require measurable dynamic background and target above fixed idle pages")
    rate = original_rate * (target - idle) / (measured - idle)
    ceiling = original_rate if maximum_rate is None else maximum_rate
    if not math.isfinite(ceiling) or ceiling <= 0 or rate >= ceiling:
        raise ValueError("Candidate must remain below the original Conv rate ceiling")
    return {"target_local_physical_ratio": target_ratio, "target_Conv_physical_mib_per_gpu": target,
            "Conv_rate_rps_original": original_rate, "Conv_rate_rps_candidate": rate,
            "Conv_rate_fraction": rate / original_rate, "Conv_rate_rps_ceiling": ceiling,
            "scope": "fixed_idle_plus_rate_proportional_dynamic_residency_first_order_candidate_requires_GPU_measurement"}


def rate_for_bracket(first_pressure, first_rate, second_pressure, second_rate, target_ratio, maximum_rate):
    if any(not math.isfinite(value) or value <= 0
           for value in (first_rate, second_rate, target_ratio, maximum_rate)):
        raise ValueError("Require positive finite rates and ratio")
    for key in ("idle_pool_mib", "budget_mib_per_gpu", "Code_single_physical_pool_mib",
                "evaluation_start_s", "evaluation_end_s"):
        if first_pressure[key] != second_pressure[key]:
            raise ValueError("Bracket profiles changed memory geometry or measurement window")
    lower, upper = sorted(((first_rate, first_pressure), (second_rate, second_pressure)), key=lambda sample: sample[0])
    lower_rate, lower_pressure = lower
    upper_rate, upper_pressure = upper
    lower_mib = lower_pressure["Conv_physical_mean_mib_per_gpu"]
    upper_mib = upper_pressure["Conv_physical_mean_mib_per_gpu"]
    target = target_ratio * lower_pressure["budget_mib_per_gpu"] - lower_pressure["Code_single_physical_pool_mib"]
    idle = sum(lower_pressure["idle_pool_mib"][1]) / 2
    if (not all(math.isfinite(value) for value in (lower_mib, upper_mib, target))
            or not idle < lower_mib < target < upper_mib
            or not lower_rate < upper_rate < maximum_rate):
        raise ValueError("Require increasing measured pressure bracketing target below original rate ceiling")
    rate = lower_rate + (target - lower_mib) * (upper_rate - lower_rate) / (upper_mib - lower_mib)
    return {"target_local_physical_ratio": target_ratio, "target_Conv_physical_mib_per_gpu": target,
            "Conv_rate_rps_original": first_rate, "Conv_rate_rps_candidate": rate,
            "Conv_rate_fraction": rate / first_rate, "Conv_rate_rps_ceiling": maximum_rate,
            "bracket": [{"Conv_rate_rps": measured_rate,
                         "Conv_physical_mean_mib_per_gpu": pressure["Conv_physical_mean_mib_per_gpu"]}
                        for measured_rate, pressure in (lower, upper)],
            "scope": "safeguarded_measured_pressure_secant_interpolation_not_benefit_selection_requires_GPU_measurement"}


def lower_conv_rate(trace, rate, parent_sha256):
    original = trace["model_rates_rps"][1]
    if not math.isfinite(rate) or not 0 < rate < original:
        raise ValueError("Only finite positive lower Conv rates are supported")
    lowered = copy.deepcopy(trace)
    scale = original / rate
    lowered["requests"] = []
    for mark in trace["requests"]:
        selected = copy.deepcopy(mark)
        if selected["model"] == 1:
            selected["arrival_s"] *= scale
        if selected["arrival_s"] < trace["duration_s"]:
            lowered["requests"].append(selected)
    lowered["requests"].sort(key=lambda mark: (mark["arrival_s"], mark["model"]))
    if {mark["model"] for mark in lowered["requests"]} != {0, 1}:
        raise ValueError("Lowered trace must still contain both tenants")
    lowered["model_rates_rps"][1] = rate
    lowered["total_rate_rps"] = sum(lowered["model_rates_rps"])
    lowered["rate_change"] = {"parent_trace_sha256": parent_sha256, "model": 1,
                              "original_rate_rps": original, "new_rate_rps": rate,
                              "arrival_time_scale": scale,
                              "method": "Poisson_time_dilation_same_seed_Conv_mark_prefix_Code_unchanged"}
    return lowered


def read_pressure(root):
    read = lambda name: json.loads((root / name).read_text())
    config, trace = read("config.json"), read("trace.json")
    start = trace["evaluation_start_s"]
    end = start + trace["evaluation_duration_s"]
    return {policy: background_pressure(read(f"r0-{policy}/population.json"),
                                         read(f"r0-{policy}/decisions.json"), start, end,
                                         config["kv_budget_mib"])
            for policy in config["policies"]}


def validate_profile(config, trace, profile_config, profile_trace):
    ignored = {"trace", "output", "port", "frozen_trace_sha256"}
    if ({key: value for key, value in profile_config.items() if key not in ignored}
            != {key: value for key, value in config.items() if key not in ignored}):
        raise ValueError("Calibration profile changed model, controlled memory settings or runtime controls")
    for key in ("duration_s", "evaluation_start_s", "evaluation_duration_s", "prompt_bank_sha256",
                "synthetic_Conv_output_tokens"):
        if profile_trace[key] != trace[key]:
            raise ValueError("Calibration profile changed workload settings")
    if ([mark for mark in profile_trace["requests"] if mark["model"] == 0]
            != [mark for mark in trace["requests"] if mark["model"] == 0]):
        raise ValueError("Calibration must preserve the complete Code stream")


def prepare(source, output, target_ratio, profile=None, bracket_profile=None):
    if output.exists():
        raise ValueError("Preserve existing frozen source; refuse overwrite")
    config = json.loads((source / "config.json").read_text())
    trace_bytes = (source / "trace.json").read_bytes()
    trace = json.loads(trace_bytes)
    profile = source if profile is None else profile
    profile_config = json.loads((profile / "config.json").read_text())
    profile_trace = json.loads((profile / "trace.json").read_text())
    validate_profile(config, trace, profile_config, profile_trace)
    start = trace["evaluation_start_s"]
    pressure = background_pressure(json.loads((profile / "r0-no_move/population.json").read_text()),
                                   json.loads((profile / "r0-no_move/decisions.json").read_text()),
                                   start, start + trace["evaluation_duration_s"], config["kv_budget_mib"])
    bracket_metadata = {}
    if bracket_profile is None:
        setting = rate_for_ratio(pressure, profile_trace["model_rates_rps"][1], target_ratio,
                                 maximum_rate=trace["model_rates_rps"][1])
    else:
        bracket_config = json.loads((bracket_profile / "config.json").read_text())
        bracket_trace = json.loads((bracket_profile / "trace.json").read_text())
        validate_profile(config, trace, bracket_config, bracket_trace)
        bracket_pressure = background_pressure(
            json.loads((bracket_profile / "r0-no_move/population.json").read_text()),
            json.loads((bracket_profile / "r0-no_move/decisions.json").read_text()),
            start, start + trace["evaluation_duration_s"], config["kv_budget_mib"])
        setting = rate_for_bracket(pressure, profile_trace["model_rates_rps"][1], bracket_pressure,
                                  bracket_trace["model_rates_rps"][1], target_ratio, trace["model_rates_rps"][1])
        bracket_metadata = {"bracket_profile": str(bracket_profile.resolve()),
                            "bracket_trace_sha256": hashlib.sha256((bracket_profile / "trace.json").read_bytes()).hexdigest(),
                            "bracket_pressure": bracket_pressure}
    lowered = lower_conv_rate(trace, setting["Conv_rate_rps_candidate"], hashlib.sha256(trace_bytes).hexdigest())
    output.mkdir(parents=True)
    lowered_bytes = (json.dumps(lowered, indent=2) + "\n").encode()
    (output / "trace.json").write_bytes(lowered_bytes)
    config["frozen_trace_sha256"] = hashlib.sha256(lowered_bytes).hexdigest()
    (output / "source-config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"parent": str(source.resolve()), "calibration_profile": str(profile.resolve()),
                "calibration_trace_sha256": hashlib.sha256((profile / "trace.json").read_bytes()).hexdigest(),
                "source_Conv_rate_rps": trace["model_rates_rps"][1],
                "baseline_pressure": pressure, "setting": setting,
                "trace_sha256": config["frozen_trace_sha256"],
                "counts": [sum(mark["model"] == model for mark in lowered["requests"]) for model in range(2)],
                "main_window_counts": [sum(mark["model"] == model and start <= mark["arrival_s"] <
                                            start + trace["evaluation_duration_s"]
                                            for mark in lowered["requests"]) for model in range(2)],
                "all_new_offered_requests_must_drain": True, "changed_only_Conv_arrival_rate": True,
                **bracket_metadata}
    (output / "pressure-setting.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-ratio", type=float, default=1.1)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--bracket-profile", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.target_ratio, args.profile, args.bracket_profile), indent=2))


if __name__ == "__main__":
    main()
