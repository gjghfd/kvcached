import argparse
import json
import math
import random
import statistics
from pathlib import Path

from agentic_trace import generate_mixed_trace
from amortized_campaign import paired_window
from long_conv_open_loop import analyze_phase
from public_response_inputs import digest, validate
from verify_trace import verify
from workaware_calibration_plan import screen_plan


SEEDS = (2026103101, 2026103102)
POPULATIONS = (2, 3, 4)
POLICIES = ("no_move", "random_no_return", "kv_only_headroom", "kv_only_workaware")
CALIBRATION_SEED = 2026103103
CODE_RATE = .001
PRESSURE_TARGETS = (1.1, 1.2, 1.3)
BUDGET_MIB = 10240


def phase_optimizer(policies):
    return "robust" if "kv_only_workaware" in policies else "point"


def public_tenant_name(inputs):
    manifest = inputs / "input-manifest.json"
    label = json.loads(manifest.read_text())["dataset_label"] if manifest.exists() else "openthoughts3"
    return {"openthoughts3": "OpenThoughts3-original-reasoning",
            "stratos": "Bespoke-Stratos-original-reasoning"}.get(label, label + "-original-response")


def write_new(path, payload):
    with path.open("x") as target:
        json.dump(payload, target, indent=2)
        target.write("\n")


def original_trace(trace, inputs):
    bank = json.loads((inputs / "prompt-bank.json").read_text())
    conv = {mark["prompt_key"]: mark for mark in json.loads((inputs / "conv-marks.json").read_text())}
    code = {mark["prompt_key"]: mark for mark in json.loads((inputs / "code-marks.json").read_text())}
    if trace["prompt_bank_sha256"] != digest(inputs / "prompt-bank.json"):
        raise ValueError("Trace must freeze its reconstructed public prompt bank")
    identifiers = set()
    for mark in trace["requests"]:
        if mark["model"] not in (0, 1):
            raise ValueError("Public experiment has exactly two independent tenants")
        expected = (code if mark["model"] == 0 else conv).get(mark.get("prompt_key"))
        if (expected is None or mark["request_id"] in identifiers
                or any(mark.get(field) != value for field, value in expected.items() if field != "count")
                or mark["prompt_key"] not in bank["prompts"]
                or mark.get("synthetic_output_tokens") is not None):
            raise ValueError("Trace must preserve original public marks and prohibit synthetic fallback/extension")
        identifiers.add(mark["request_id"])
    return trace


def make_native(inputs, seed, population):
    if seed not in SEEDS or population not in POPULATIONS:
        raise ValueError("Native search is restricted to the predeclared seed/population grid")
    conv = json.loads((inputs / "conv-marks.json").read_text())
    code = json.loads((inputs / "code-marks.json").read_text())
    generator = random.Random(seed)
    requests = [{**{field: value for field, value in generator.choice(conv).items() if field != "count"},
                 "model": 1, "request_id": f"public-{seed}-conv-{ordinal}", "arrival_s": float(ordinal)}
                for ordinal in range(population)]
    requests.append({**{field: value for field, value in random.Random(seed + 100).choice(code).items()
                        if field != "count"}, "model": 0, "request_id": f"public-{seed}-code", "arrival_s": 90.})
    trace = {"schema_version": 2, "seed": seed, "duration_s": 900., "max_model_len": 131072,
             "prompt_bank_sha256": digest(inputs / "prompt-bank.json"), "requests": requests,
             "model_names": ["SWE-smith-Code", public_tenant_name(inputs)],
             "arrival": "closed_population_mechanism_not_Poisson", "evaluation_start_s": 0.,
             "evaluation_duration_s": 900., "population": population,
             "sampling": "full_short_input_public_hash_bank_original_response_lengths",
             "scope": "return_vs_natural_recovery_not_final_TTFT_tail_evaluation"}
    return original_trace(trace, inputs)


def make_poisson(inputs, seed, conv_rate, duration=3600, code_rate=CODE_RATE):
    code = json.loads((inputs / "code-marks.json").read_text())
    conv = json.loads((inputs / "conv-marks.json").read_text())
    trace = generate_mixed_trace(code, conv, code_rate, conv_rate, duration, seed,
                                 max_model_len=131072, allow_empty_tenants=True)
    trace.update({"prompt_bank_sha256": digest(inputs / "prompt-bank.json"),
                  "model_names": ["SWE-smith-Code", public_tenant_name(inputs)],
                  "sampling": "full_short_input_public_hash_bank_original_response_lengths",
                  "evaluation_start_s": 600., "evaluation_duration_s": duration - 600,
                  "warmup": "natural_Poisson_no_bootstrap_all_zero_draws_retained_full_drain"})
    return original_trace(trace, inputs)


def prepare(root):
    inputs = root / "frozen/inputs"
    manifest = validate(inputs)
    plan = {"schema_version": 1, "seed": 20261031, "models": [0, 1], "repeats": [0, 1],
            "output_tokens": 3072, "window_tokens": [1536, 2560], "budget_mib": BUDGET_MIB,
            "populations": [{"population": population, "prompt_tokens": [4096, 8192, 16384]}
                            for population in POPULATIONS]}
    screening = screen_plan(plan)
    frozen = root / "frozen"
    names = []
    for seed in SEEDS:
        for population in POPULATIONS:
            name = f"native-{seed}-N{population}.json"
            write_new(frozen / name, make_native(inputs, seed, population))
            names.append(name)
    write_new(frozen / "calibration-plan.json", plan)
    write_new(frozen / "trace.json", make_native(inputs, SEEDS[0], POPULATIONS[0]))
    config = {"schema_version": 1, "inputs_manifest_sha256": digest(inputs / "input-manifest.json"),
              "dataset": manifest["dataset_label"], "seeds": list(SEEDS), "populations": list(POPULATIONS),
              "policies": list(POLICIES), "calibration_seed": CALIBRATION_SEED,
              "code_rate_rps": CODE_RATE, "pressure_targets": list(PRESSURE_TARGETS),
              "trace_sha256": {name: digest(frozen / name) for name in names},
              "calibration_plan_sha256": digest(frozen / "calibration-plan.json"),
              "cost_profile_screen": screening, "publication_authorized": False,
              "research_goal_complete": False}
    write_new(frozen / "search-config.json", config)
    validate_frozen(frozen)
    return config


def validate_frozen(frozen):
    inputs = frozen / "inputs"
    validate(inputs)
    config = json.loads((frozen / "search-config.json").read_text())
    if (config["seeds"] != list(SEEDS) or config["populations"] != list(POPULATIONS)
            or config["policies"] != list(POLICIES)
            or config["calibration_seed"] != CALIBRATION_SEED
            or config["code_rate_rps"] != CODE_RATE
            or config["pressure_targets"] != list(PRESSURE_TARGETS)
            or config["inputs_manifest_sha256"] != digest(inputs / "input-manifest.json")
            or config["calibration_plan_sha256"] != digest(frozen / "calibration-plan.json")):
        raise ValueError("Frozen public search changed declared inputs, contexts or comparison grid")
    expected_names = {f"native-{seed}-N{population}.json" for seed in SEEDS for population in POPULATIONS}
    if set(config["trace_sha256"]) != expected_names:
        raise ValueError("Every declared seed/population must be preserved")
    for seed in SEEDS:
        for population in POPULATIONS:
            name = f"native-{seed}-N{population}.json"
            if (digest(frozen / name) != config["trace_sha256"][name]
                    or json.loads((frozen / name).read_text()) != make_native(inputs, seed, population)):
                raise ValueError("Public trace draw changed; do not reroll outcomes")
    if json.loads((frozen / "trace.json").read_text()) != make_native(inputs, SEEDS[0], POPULATIONS[0]):
        raise ValueError("Default public trace pointer changed")
    return config


def measured_rate_priors(marks, native_records, code_pool_mib):
    selected = [record for record in native_records if record["model"] == 1]
    if not selected or not 0 < code_pool_mib < BUDGET_MIB:
        raise ValueError("Require measured native Conv service and fitting actual Code pool")
    step = statistics.mean((record["completed_at"] - record["first_token_at"]) / (len(record["token_ids"]) - 1)
                           for record in selected)
    moment = statistics.mean(math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / 1024)
                             * 96 * mark["output_tokens"] for mark in marks)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Native Conv TPOT must be positive and finite")
    rates = [2 * (target * BUDGET_MIB - code_pool_mib) / (moment * step) for target in PRESSURE_TARGETS]
    return {"conv_rates_rps": rates, "measured_native_Conv_step_s": step,
            "measured_Code_physical_pool_mib": code_pool_mib,
            "bank_mean_independent_page_times_output_mib_tokens": moment,
            "scope": "native_service_page_upper_prior_requires_open_loop_actual_pressure_verification"}


def native_rate_priors(root):
    records, code_pools = [], []
    inputs = root / "frozen/inputs"
    for seed in SEEDS:
        for population in (3, 4):
            phase = root / f"native-{seed}-N{population}"
            verify(phase, inputs / "prompt-bank.json")
            records.extend(json.loads((phase / "r0-no_move/requests.json").read_text()))
            for decision in json.loads((phase / "r0-no_move/decisions.json").read_text()):
                for field in ("observations", "after_admission"):
                    observed = decision.get(field)
                    if observed:
                        code_pools.extend(rank["mapped_bytes"] / 1024**2 for rank in observed[0]
                                          if len(rank["requests"]) == 1)
    if not code_pools:
        raise ValueError("Native pilot must witness an actual single-Code physical pool")
    marks = json.loads((inputs / "conv-marks.json").read_text())
    return measured_rate_priors(marks, records, max(code_pools))


def freeze_calibration(root):
    validate_frozen(root / "frozen")
    destination = root / "calibration-frozen"
    if destination.exists():
        raise ValueError("Preserve the initial measured-rate calibration; never overwrite or reroll")
    inputs = root / "frozen/inputs"
    priors = native_rate_priors(root)
    destination.mkdir()
    for ordinal, rate in enumerate(priors["conv_rates_rps"]):
        write_new(destination / f"rate-{ordinal}.json", make_poisson(inputs, CALIBRATION_SEED, rate))
    config = {**priors, "code_rate_rps": CODE_RATE, "seed": CALIBRATION_SEED, "duration_s": 3600,
              "warmup_s": 600, "pressure_targets": list(PRESSURE_TARGETS),
              "trace_sha256": {f"rate-{ordinal}.json": digest(destination / f"rate-{ordinal}.json")
                               for ordinal in range(len(PRESSURE_TARGETS))}}
    write_new(destination / "calibration-config.json", config)
    return config


def analyze(root):
    inputs = root / "frozen/inputs"
    validate_frozen(root / "frozen")
    result = {"native": {}, "calibration": {}, "research_goal_complete": False,
              "scope": "public_source_mechanism_and_rate_screen_not_final_heldout_benefit"}
    for seed in SEEDS:
        for population in POPULATIONS:
            name = f"native-{seed}-N{population}"
            phase = root / name
            result["native"][name] = {"verification": verify(phase, inputs / "prompt-bank.json"),
                "comparisons": {baseline: paired_window(phase, baseline, "kv_only_workaware", 0, 900)
                                for baseline in POLICIES if baseline != "kv_only_workaware"}}
    calibration = root / "calibration-frozen"
    config = json.loads((calibration / "calibration-config.json").read_text())
    priors = native_rate_priors(root)
    if (any(config.get(field) != value for field, value in priors.items())
            or config["code_rate_rps"] != CODE_RATE or config["seed"] != CALIBRATION_SEED
            or config["duration_s"] != 3600 or config["warmup_s"] != 600
            or config["pressure_targets"] != list(PRESSURE_TARGETS)):
        raise ValueError("Calibration rate priors must independently reconstruct from native measured service")
    for ordinal, rate in enumerate(config["conv_rates_rps"]):
        name = f"rate-{ordinal}"
        trace = make_poisson(inputs, CALIBRATION_SEED, rate)
        if (json.loads((root / name / "trace.json").read_text()) != trace
                or digest(root / name / "trace.json") != config["trace_sha256"][name + ".json"]):
            raise ValueError("Open-loop trace differs from its independent native-service freeze")
        result["calibration"][name] = {"verification": verify(root / name, inputs / "prompt-bank.json"),
                                       **analyze_phase(root / name)}
    eligible = [(abs(case["physical_pressure"]["local_physical_ratio"] - 1.2), case["rate_rps"][1], name)
                for name, case in result["calibration"].items()
                if case["eligible_for_separate_evaluation_rate_candidate"]]
    result["candidate_case"] = min(eligible)[2] if eligible else None
    result["selection_rule"] = "closest_actual_pressure_to_1p2_with_coverage_queue_guardrails_not_benefit"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--action", choices=("prepare", "validate", "freeze-calibration", "analyze"), required=True)
    args = parser.parse_args()
    actions = {"prepare": prepare, "validate": lambda root: validate_frozen(root / "frozen"),
               "freeze-calibration": freeze_calibration, "analyze": analyze}
    print(json.dumps(actions[args.action](args.root), indent=2))


if __name__ == "__main__":
    main()
