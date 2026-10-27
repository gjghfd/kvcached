import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_workaware_cost import paired_contrasts
from verify_workaware_pilot import verify
from workaware_pilot import SOURCE_FILES, calibration_cases, stable_sample
from workaware_profile_cost import fit_profiles


def verify_profile(root):
    root = Path(root)
    fit = verify(root)
    trials = json.loads((root / "workaware-trials.json").read_text())
    rebuilt = [{"case": trial["case"], "sample": stable_sample(trial["requests"])} for trial in trials]
    if any(trial["sample"] != expected["sample"] for trial, expected in zip(trials, rebuilt)):
        raise ValueError("Population profile samples disagree with full raw calibration")
    contrasts = paired_contrasts(rebuilt, fit)
    if contrasts["unmatched_groups"]:
        raise ValueError("Population profiles require every balanced paired calibration group")
    profiles = fit_profiles(contrasts["contrasts"])
    expected = {"{}:{}".format(case["model"], case["population"]) for case in calibration_cases()}
    if set(profiles["profiles"]) != expected:
        raise ValueError("Population profiles omit calibrated tenants or populations")
    for profile in profiles["profiles"].values():
        samples = [trial["sample"] for trial in rebuilt if trial["case"]["model"] == profile["model"]
                   and trial["case"]["population"] == profile["population"]]
        profile["max_context_tokens"] = max(context / count for sample in samples
                                             for context, count in zip(sample["context_tokens"], sample["counts"]) if count)
        profile["max_attention_tokens"] = max(max(sample["context_tokens"]) for sample in samples)
        profile["domain_scope"] = "measured_population_context_upper_envelope_not_all_joint_combinations"
    profiles.update({"status": "raw_verified_workaware_population_profiles", "source_verification": "raw_verified",
                     "source_files_sha256": fit["source_files_sha256"],
                     "algorithm_scope": "paired_population_costs_and_changed_segment_uncertainty"})
    return {**fit, "population_profiles": profiles}


def same_payload(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(same_payload(before[name], after[name]) for name in before)
    if isinstance(before, list) and isinstance(after, list):
        return len(before) == len(after) and all(same_payload(left, right) for left, right in zip(before, after))
    if type(before) is float and type(after) is float:
        return math.isfinite(before) and math.isfinite(after) and math.isclose(before, after, rel_tol=1e-8, abs_tol=1e-12)
    return type(before) is type(after) and before == after


def load_verified_profile(path, expected_sha256, source_root):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("Population profile requires a frozen SHA256")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Population profile differs from frozen SHA256")
    fit = json.loads(payload)
    if fit.get("population_profiles", {}).get("status") != "raw_verified_workaware_population_profiles":
        raise ValueError("Diagnostic population profiles are not usable by serving")
    for name in SOURCE_FILES:
        if hashlib.sha256((Path(source_root) / name).read_bytes()).hexdigest() != fit["source_files_sha256"].get(name):
            raise ValueError("Population profile raw calibration source changed")
    if not same_payload(fit, verify_profile(source_root)):
        raise ValueError("Population profile does not reproduce from full raw calibration")
    return fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit = verify_profile(args.root)
    args.output.write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit["population_profiles"], indent=2))


if __name__ == "__main__":
    main()
