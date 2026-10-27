import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def calculate_rates(pilot, conv_marks, target_conv_mib=23552):
    if pilot["status"] != "passed" or target_conv_mib <= 0 or not conv_marks:
        raise ValueError("Require a successful pilot and positive population target")
    experiments = pilot["experiments"]
    populations = [entry["population"] for entry in experiments]
    services = [entry["conv_mean_service_s"] for entry in experiments]
    if len(set(populations)) < 2 or any(value <= 0 for value in services):
        raise ValueError("Require distinct positive pilot populations and services")
    outputs = {entry["conv_mark"]["output_tokens"] for entry in experiments}
    if len(outputs) != 1:
        raise ValueError("Population fit requires the same Conv mark")
    pilot_output = outputs.pop()
    mean_population, mean_service = statistics.mean(populations), statistics.mean(services)
    slope = sum((population - mean_population) * (service - mean_service)
                for population, service in zip(populations, services)) / sum(
                    (population - mean_population) ** 2 for population in populations)
    intercept = mean_service - slope * mean_population
    if slope < 0 or intercept <= 0:
        raise ValueError("Population fit has invalid service parameters")
    count = sum(mark.get("count", 1) for mark in conv_marks)
    if count <= 0 or any(min(mark["prompt_tokens"], mark["output_tokens"], mark.get("count", 1)) <= 0
                         for mark in conv_marks):
        raise ValueError("Require positive paired Conv lengths and weights")
    mean_output = sum(mark["output_tokens"] * mark.get("count", 1) for mark in conv_marks) / count
    memory_outputs = sum(1.5 * math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / 16)
                         * mark["output_tokens"] * mark.get("count", 1) for mark in conv_marks) / count
    resident_memory = memory_outputs / mean_output
    target_population = target_conv_mib / resident_memory
    target_service = (intercept + slope * target_population) * mean_output / pilot_output
    conv_rate = target_population / target_service
    code_services = [entry["code_service_s"] for entry in experiments]
    if any(value <= 0 for value in code_services):
        raise ValueError("Require positive Code service times")
    code_rate = min(.02, .25 / statistics.mean(code_services))
    return {"scope": "finite_population_pilot_extrapolation_not_steady_state_or_benefit_guarantee",
            "service_model": "S_conv(N,O)=(intercept+slope*N)*O/pilot_output",
            "intercept_s": intercept, "slope_s_per_request": slope,
            "pilot_output_tokens": pilot_output, "mean_conv_output_tokens": mean_output,
            "mean_conv_memory_output_mib_tokens": memory_outputs,
            "service_biased_resident_memory_mib": resident_memory,
            "target_conv_mib": target_conv_mib, "target_population": target_population,
            "target_service_s": target_service, "conv_rate_raw_rps": conv_rate,
            "code_mean_service_s": statistics.mean(code_services), "code_rate_raw_rps": code_rate,
            "conv_rate_rps": math.floor(conv_rate * 100) / 100,
            "code_rate_rps": math.floor(code_rate * 1000) / 1000,
            "pilot_population_range": [min(populations), max(populations)],
            "population_extrapolated": not min(populations) <= target_population <= max(populations),
            "assumptions": ["Service scales with output length; input-dependent cost is not separately calibrated",
                            "Synchronous finite batches with one Code are not open-loop Poisson equilibrium",
                            "Preserve paired P/O correlation in E[b*O], not E[b]*E[O]",
                            "No outcome-based seed, SLO or request filtering"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--conv-marks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate_rates(json.loads(args.pilot.read_text()), json.loads(args.conv_marks.read_text()))
    result["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (args.pilot, args.conv_marks)}
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
