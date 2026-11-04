import argparse
import hashlib
import json
import random
from pathlib import Path

from workaware_calibration_plan import screen_plan


POLICIES = ("no_move", "random_no_return", "kv_only_headroom", "kv_only_workaware")


def calibration_plan():
    return {"schema_version": 1, "seed": 20261030, "models": [0, 1], "repeats": [0, 1],
            "output_tokens": 3072, "window_tokens": [1536, 2560], "budget_mib": 10240,
            "populations": [{"population": population, "prompt_tokens": [8192, 16384, 24576, 31744]}
                            for population in (2, 3)]}


def weighted_mark(marks, generator):
    if not marks or any(type(mark.get("count", 1)) is not int or mark.get("count", 1) <= 0 for mark in marks):
        raise ValueError("Require a nonempty bank with positive integer weights")
    position = generator.randrange(sum(mark.get("count", 1) for mark in marks))
    for mark in marks:
        position -= mark.get("count", 1)
        if position < 0:
            return {name: value for name, value in mark.items() if name != "count"}
    raise ValueError("Invalid weighted bank")


def make_probe(code_marks, conv_marks, prompt_bank_sha256, seed, output_tokens=24576):
    if (type(seed) is not int or type(output_tokens) is not int or output_tokens <= 8192
            or not isinstance(prompt_bank_sha256, str) or len(prompt_bank_sha256) != 64
            or any(character not in "0123456789abcdef" for character in prompt_bank_sha256)):
        raise ValueError("Freeze the seed, longer synthetic output and prompt bank SHA")
    generator = random.Random(seed)
    requests = []
    for index in range(2):
        mark = weighted_mark(conv_marks, generator)
        if (type(mark.get("prompt_tokens")) is not int or not 2048 <= mark["prompt_tokens"] <= 7999
                or type(mark.get("output_tokens")) is not int or mark["output_tokens"] <= 0
                or mark["prompt_tokens"] + output_tokens > 131072):
            raise ValueError("Preserve the declared AzureConv input bank and supported sequence length")
        requests.append({**mark, "request_id": f"long-conv-{seed}-{index}", "model": 1,
                         "arrival_s": float(index), "original_output_tokens": mark["output_tokens"],
                         "original_source_fragment": mark.get("source_fragment", "Azure-Conv-paired-slice"),
                         "output_tokens": output_tokens, "synthetic_output_tokens": output_tokens,
                         "source_fragment": "Azure-Conv-input-synthetic-long-output"})
    code = weighted_mark(code_marks, random.Random(seed + 100))
    if (type(code.get("prompt_tokens")) is not int or code["prompt_tokens"] < 100000
            or code.get("output_tokens") != 64 or code["prompt_tokens"] + code["output_tokens"] > 131072
            or not code.get("prompt_key")):
        raise ValueError("Code retains a real >=100K prompt bank entry and output64")
    requests.append({**code, "request_id": f"long-conv-{seed}-code", "model": 0, "arrival_s": 120.})
    return {"schema_version": 2, "max_model_len": 131072, "duration_s": 1200., "requests": requests,
            "prompt_bank_sha256": prompt_bank_sha256, "seed": seed,
            "sampling": "full_weighted_AzureConv_bank_two_requests_and_real_SWE_smith_mark",
            "arrival": "closed_N2_mechanism_probe_not_Poisson_final_trace",
            "model_names": ["SWE-smith-Code", "Azure-Conversation-synthetic-long-output"],
            "evaluation_start_s": 0., "evaluation_duration_s": 1200.,
            "synthetic_Conv_output_tokens": output_tokens,
            "quiet_interval_scope": "no_new_background_births_during_closed_population_probe"}


def prepare(inputs, destination):
    if destination.exists():
        raise ValueError("Preserve existing frozen probe inputs")
    read = lambda name: json.loads((inputs / name).read_text())
    prompt_hash = hashlib.sha256((inputs / "prompt-bank.json").read_bytes()).hexdigest()
    plan = calibration_plan()
    screen = screen_plan(plan)
    traces = {f"probe-{seed}.json": make_probe(read("code-marks.json"), read("conv-marks.json"), prompt_hash, seed)
              for seed in (20261030, 20261031, 20261032)}
    destination.mkdir(parents=True)
    payloads = {"calibration-plan.json": plan, **traces}
    for name, payload in payloads.items():
        (destination / name).write_text(json.dumps(payload, indent=2) + "\n")
    config = {"schema_version": 1, "output_tokens": 24576, "population": 2, "code_arrival_s": 120.,
              "duration_s": 1200., "seeds": [20261030, 20261031, 20261032], "policies": list(POLICIES),
              "input_sha256": {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
                               for name in ("code-marks.json", "conv-marks.json", "prompt-bank.json")},
              "frozen_sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                                for name in payloads}, "calibration_screen": screen,
              "rate_prior_rps": [0.00025, 0.005652527937486215],
              "rate_scope": "cold_arrival_cost_prior_not_measured_RPS_of_closed_fixture",
              "profile_scope": "new_independent_N2_N3_fit_no_legacy_profile_or_coefficient_overwrite",
              "paused_migration": True, "overlap": False, "research_goal_complete": False}
    (destination / "probe-config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def validate_upload(root, inputs):
    config = json.loads((root / "probe-config.json").read_text())
    if (config.get("schema_version") != 1 or config.get("output_tokens") != 24576
            or config.get("population") != 2 or config.get("policies") != list(POLICIES)
            or config.get("seeds") != [20261030, 20261031, 20261032]
            or config.get("code_arrival_s") != 120. or config.get("duration_s") != 1200.
            or config.get("rate_prior_rps") != [0.00025, 0.005652527937486215]
            or config.get("paused_migration") is not True or config.get("overlap") is not False):
        raise ValueError("Long Conv probe controls changed after freezing")
    expected = {"calibration-plan.json", *(f"probe-{seed}.json" for seed in config["seeds"])}
    if set(config["frozen_sha256"]) != expected:
        raise ValueError("Probe must freeze the exact calibration and all three independent input draws")
    for name, digest in config["frozen_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen probe input changed")
    for name, digest in config["input_sha256"].items():
        if name not in ("code-marks.json", "conv-marks.json", "prompt-bank.json"):
            raise ValueError("Unknown dataset input path")
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Original input bank changed")
    if set(config["input_sha256"]) != {"code-marks.json", "conv-marks.json", "prompt-bank.json"}:
        raise ValueError("Probe input bank hashes are incomplete")
    plan = json.loads((root / "calibration-plan.json").read_text())
    if plan != calibration_plan() or screen_plan(plan) != config["calibration_screen"]:
        raise ValueError("Long-context calibration plan changed")
    read = lambda name: json.loads((inputs / name).read_text())
    for seed in config["seeds"]:
        expected_trace = make_probe(read("code-marks.json"), read("conv-marks.json"),
                                    config["input_sha256"]["prompt-bank.json"], seed)
        if json.loads((root / f"probe-{seed}.json").read_text()) != expected_trace:
            raise ValueError("Probe does not reproduce from the full weighted bank")
    return {"status": "frozen_probe_independently_rebuilt_not_GPU_benefit", "calibration": screen_plan(plan)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-upload", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.validate_upload is None):
        parser.error("Choose preparation or independent upload validation")
    result = validate_upload(args.validate_upload, args.inputs) if args.validate_upload else prepare(args.inputs, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
