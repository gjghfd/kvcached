import argparse
import json
from pathlib import Path

from public_response_inputs import digest
from workaware_calibration_plan import planned_cases, screen_plan


POPULATIONS = (5, 6, 7, 8, 9)
PROMPT_LIMITS = (16384, 14336, 11264, 8192, 8192)
REQUIRED_PROFILES = tuple(f"{model}:{population}" for model in (0, 1) for population in range(2, 10))


def calibration_plan():
    return {"schema_version": 1, "seed": 2026103106, "models": [0, 1], "repeats": [0, 1],
            "output_tokens": 3072, "window_tokens": [1536, 2560], "budget_mib": 10240,
            "populations": [{"population": population, "prompt_tokens": [1024, longest]}
                            for population, longest in zip(POPULATIONS, PROMPT_LIMITS)]}


def calibration_controls(plan):
    if plan != calibration_plan():
        raise ValueError("Population calibration cannot be retuned after public workload outcomes")
    cases = planned_cases(plan)
    return {"schema_version": 1, "cases": len(cases),
            "requests": sum(case["population"] for case in cases),
            "screen": screen_plan(plan), "required_profiles": list(REQUIRED_PROFILES),
            "scope": "two_resident_tenants_one_active_per_cost_case_not_joint_active",
            "outside_population_context_envelope": "no_forecast_no_extrapolation",
            "response_lengths": "synthetic_cost_only_public_benefit_responses_unchanged"}


def verify_calibration(root):
    from verify_workaware_profile import same_payload, verify_profile

    root = Path(root)
    config = json.loads((root / "frozen/evaluation-config.json").read_text())
    plan_path = root / "frozen/population-calibration-plan.json"
    plan = json.loads(plan_path.read_text())
    controls = calibration_controls(plan)
    if config.get("population_calibration") != {**controls, "plan_sha256": digest(plan_path)}:
        raise ValueError("Population calibration differs from its predeclared controls")
    raw = root / "public-population-calibration"
    if json.loads((raw / "config.json").read_text()).get("workaware_case_plan") != plan:
        raise ValueError("GPU population cases differ from the frozen feasible plan")
    raw_fit = verify_profile(raw)
    parent_path = root / "parent-public-context-fit.json"
    if digest(parent_path) != config["fit_sha256"]:
        raise ValueError("Original parent calibration SHA changed")
    parent_fit = json.loads(parent_path.read_text())
    bundle = root / "public-population-profile"
    if not same_payload(verify_profile(bundle / "legacy"), parent_fit):
        raise ValueError("Population supplement cannot refit the parent cadence anchor")
    anchor_manifest = bundle / "supplement/workaware-cadence-anchor.json"
    if (not anchor_manifest.is_file() or json.loads(anchor_manifest.read_text()) !=
            {"schema_version": 1, "source": "cadence-anchor"}):
        raise ValueError("Population supplement requires the exact cadence anchor manifest")
    if not same_payload(verify_profile(bundle / "supplement/cadence-anchor"), parent_fit):
        raise ValueError("Population supplement uses a different independently verified parent anchor")
    fit = verify_profile(bundle)
    if (not same_payload(fit, json.loads((bundle / "fit.json").read_text()))
            or set(fit["population_profiles"]["profiles"]) != set(REQUIRED_PROFILES)
            or raw_fit["cases"] != controls["cases"] or raw_fit["requests"] != controls["requests"]):
        raise ValueError("Population profile omits exact calibrated cases or coverage")
    supplement = verify_profile(bundle / "supplement")
    if not same_payload(supplement.get("coefficients"), parent_fit["coefficients"]):
        raise ValueError("Population supplement changes fixed parent cadence coefficients")
    for name in raw_fit["source_files_sha256"]:
        if digest(raw / name) != digest(bundle / "supplement" / name):
            raise ValueError("Anchored supplement differs from its original GPU raw source")
    if not same_payload(raw_fit["population_profiles"]["profiles"], supplement["population_profiles"]["profiles"]):
        raise ValueError("Cadence anchoring changes independently measured population contrasts")
    return {**controls, "status": "GPU_raw_population_cost_SHA_verified_not_benefit",
            "parent_fit_sha256": config["fit_sha256"], "fit_sha256": digest(bundle / "fit.json"),
            "plan_sha256": digest(plan_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_calibration(args.root), indent=2))


if __name__ == "__main__":
    main()
