import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from agentic_trace import generate_mixed_trace
from kv_pressure import background_pressure
from stress_trace import long_output_marks
from verify_trace import request_metrics


CONV_RATES = (.00565, .0064, .0072)
CODE_RATE = .001
SEED = 2026103001
DURATION_S = 3600
WARMUP_S = 600
BUDGET_MIB = 10240
INPUT_FILES = ("code-marks.json", "conv-marks.json", "prompt-bank.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_trace(code, conv, prompt_sha, rate):
    trace = generate_mixed_trace(code, long_output_marks(conv, 24576), CODE_RATE, rate,
        DURATION_S, SEED, max_model_len=131072, allow_empty_tenants=True)
    trace.update({"prompt_bank_sha256": prompt_sha, "synthetic_Conv_output_tokens": 24576,
        "sampling": "verified_SWE_smith_history_Azure_input_explicit_synthetic_Conv_output",
        "evaluation_start_s": WARMUP_S, "evaluation_duration_s": DURATION_S - WARMUP_S,
        "warmup": "natural_Poisson_no_bootstrap_all_requests_recorded_and_drained",
        "scope": "rate_service_KV_queue_calibration_not_final_benefit_evaluation"})
    return trace


def prepare(inputs, destination):
    if destination.exists():
        raise ValueError("Preserve existing frozen open-loop calibration inputs")
    code = json.loads((inputs / "code-marks.json").read_text())
    conv = json.loads((inputs / "conv-marks.json").read_text())
    if any(mark["prompt_tokens"] < 100000 or mark["output_tokens"] != 64 for mark in code):
        raise ValueError("Open-loop Code requires unchanged real >=100K input and O64")
    prompt_sha = digest(inputs / "prompt-bank.json")
    traces = {f"rate-{ordinal}.json": make_trace(code, conv, prompt_sha, rate)
              for ordinal, rate in enumerate(CONV_RATES)}
    destination.mkdir(parents=True)
    for name, trace in traces.items():
        (destination / name).write_text(json.dumps(trace, indent=2) + "\n")
    (destination / "trace.json").write_bytes((destination / "rate-0.json").read_bytes())
    config = {"schema_version": 1, "policies": ["no_move"], "code_rate_rps": CODE_RATE,
        "conv_rates_rps": list(CONV_RATES), "seed": SEED, "duration_s": DURATION_S,
        "warmup_s": WARMUP_S, "budget_mib": BUDGET_MIB, "synthetic_Conv_output_tokens": 24576,
        "allow_empty_tenants": True, "input_sha256": {name: digest(inputs / name) for name in INPUT_FILES},
        "trace_sha256": {name: digest(destination / name) for name in traces},
        "controls": "BF16_128K_sparse_pause32_dp_echo_two_independent_DP2_EP2_attentionTP1_tenants",
        "selection": {"target_local_physical_ratio": 1.2, "ratio_interval": [1.1, 1.3],
            "minimum_Code_absent_coverage_fraction": .7, "maximum_pending_growth": .5,
            "tie_break": "nearest_ratio_then_lowest_Conv_rate", "select_using_policy_benefit": False},
        "final_Poisson_evaluation": False, "research_goal_complete": False}
    (destination / "open-loop-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def validate_upload(root, inputs):
    config = json.loads((root / "open-loop-config.json").read_text())
    if (config.get("schema_version") != 1 or config.get("conv_rates_rps") != list(CONV_RATES)
            or config.get("code_rate_rps") != CODE_RATE or config.get("seed") != SEED
            or config.get("duration_s") != DURATION_S or config.get("warmup_s") != WARMUP_S
            or config.get("budget_mib") != BUDGET_MIB or config.get("policies") != ["no_move"]
            or config.get("synthetic_Conv_output_tokens") != 24576
            or config.get("allow_empty_tenants") is not True
            or config.get("selection") != {"target_local_physical_ratio": 1.2, "ratio_interval": [1.1, 1.3],
                "minimum_Code_absent_coverage_fraction": .7, "maximum_pending_growth": .5,
                "tie_break": "nearest_ratio_then_lowest_Conv_rate", "select_using_policy_benefit": False}):
        raise ValueError("Open-loop calibration differs from its predeclared controls")
    if set(config["input_sha256"]) != set(INPUT_FILES):
        raise ValueError("Open-loop calibration must freeze all original input banks")
    for name, expected in config["input_sha256"].items():
        if digest(inputs / name) != expected:
            raise ValueError("Original input bank changed after open-loop freeze")
    expected_names = {f"rate-{ordinal}.json" for ordinal in range(len(CONV_RATES))}
    if set(config["trace_sha256"]) != expected_names:
        raise ValueError("Open-loop calibration must retain every predeclared rate")
    code = json.loads((inputs / "code-marks.json").read_text())
    conv = json.loads((inputs / "conv-marks.json").read_text())
    for ordinal, rate in enumerate(CONV_RATES):
        name = f"rate-{ordinal}.json"
        if (digest(root / name) != config["trace_sha256"][name]
                or json.loads((root / name).read_text()) != make_trace(code, conv,
                    config["input_sha256"]["prompt-bank.json"], rate)):
            raise ValueError("Open-loop draw, original marks or timing cannot be independently rebuilt")
    if (root / "trace.json").read_bytes() != (root / "rate-0.json").read_bytes():
        raise ValueError("Default upload pointer differs from first frozen trace")
    return config


def population_windows(population):
    midpoint = (WARMUP_S + DURATION_S) / 2
    windows = ((WARMUP_S, midpoint), (midpoint, DURATION_S))
    totals = [{"covered_s": 0., "pending_seconds": 0., "active_request_seconds": 0.} for _ in windows]
    for current, following in zip(population, population[1:]):
        if (not all(math.isfinite(value) for value in (current["elapsed_s"], following["elapsed_s"]))
                or following["elapsed_s"] <= current["elapsed_s"]):
            raise ValueError("Open-loop population timestamps must be finite and strictly increasing")
        for ordinal, (start, end) in enumerate(windows):
            overlap = min(end, following["elapsed_s"]) - max(start, current["elapsed_s"])
            if overlap <= 0 or current.get("admission_stable_across_observation") is False:
                continue
            if current["capacity_pending"] < 0 or any(count < 0 for counts in current["counts"] for count in counts):
                raise ValueError("Open-loop queue/population counts cannot be negative")
            totals[ordinal]["covered_s"] += overlap
            totals[ordinal]["pending_seconds"] += overlap * current["capacity_pending"]
            totals[ordinal]["active_request_seconds"] += overlap * sum(sum(counts) for counts in current["counts"])
    for total in totals:
        total["mean_pending"] = total["pending_seconds"] / total["covered_s"] if total["covered_s"] else None
        total["mean_active"] = total["active_request_seconds"] / total["covered_s"] if total["covered_s"] else None
    return totals


def analyze_phase(phase):
    case = phase / "r0-no_move"
    read = lambda path: json.loads(path.read_text())
    trace, workload = read(phase / "trace.json"), read(case / "workload.json")
    records = read(case / "requests.json")
    if (len(records) != len(trace["requests"])
            or {record["request_id"] for record in records} != {mark["request_id"] for mark in trace["requests"]}):
        raise ValueError("Open-loop calibration must drain all offered requests without duplicates")
    population, decisions = read(case / "population.json"), read(case / "decisions.json")
    windows = population_windows(population)
    try:
        pressure = background_pressure(population, decisions, WARMUP_S, DURATION_S, BUDGET_MIB)
        pressure_error = None
    except ValueError as error:
        pressure, pressure_error = None, str(error)
    start = workload["origin"] + WARMUP_S
    metrics, service = {}, {}
    for model, name in ((0, "Code"), (1, "Conv")):
        selected = [record for record in records if record["model"] == model and start <= record["arrived_at"]
                    < workload["origin"] + DURATION_S]
        metrics[name] = request_metrics(selected, start, DURATION_S - WARMUP_S, 120)
        service[name] = {"count": len(selected), "mean_reserved_to_completed_s": statistics.mean(
            record["completed_at"] - record["reserved_at"] for record in selected) if selected else None}
    covered = all(window["covered_s"] >= .7 * (DURATION_S - WARMUP_S) / 2 for window in windows)
    growth = windows[1]["mean_pending"] - windows[0]["mean_pending"] if covered else None
    eligible = bool(covered and pressure is not None
        and pressure["background_covered_s"] >= .7 * (DURATION_S - WARMUP_S)
        and 1.1 <= pressure["local_physical_ratio"] <= 1.3 and growth <= .5)
    return {"rate_rps": trace["model_rates_rps"], "offered_counts": [sum(mark["model"] == model
        for mark in trace["requests"]) for model in (0, 1)], "measurement_metrics": metrics,
        "measured_admitted_service": service, "queue_windows": windows, "pending_growth": growth,
        "physical_pressure": pressure, "physical_pressure_unavailable_reason": pressure_error,
        "eligible_for_separate_evaluation_rate_candidate": eligible,
        "stationarity_established": False, "research_goal_complete": False}


def analyze(root):
    result = {"cases": {f"rate-{ordinal}": analyze_phase(root / f"rate-{ordinal}")
                         for ordinal in range(len(CONV_RATES))}}
    eligible = [(abs(case["physical_pressure"]["local_physical_ratio"] - 1.2), case["rate_rps"][1], name)
                for name, case in result["cases"].items() if case["eligible_for_separate_evaluation_rate_candidate"]]
    result.update({"candidate_case": min(eligible)[2] if eligible else None,
        "selection_scope": "predeclared_single_policy_measured_pressure_and_pending_screen_not_benefit_tuning",
        "final_trace_frozen": False, "research_goal_complete": False})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--validate-upload", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.validate_upload):
        parser.error("Choose exactly one prepare or validate-upload action")
    result = prepare(args.inputs, args.prepare) if args.prepare else validate_upload(args.validate_upload, args.inputs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
