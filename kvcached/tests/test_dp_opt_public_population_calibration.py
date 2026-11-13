import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def population(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_population_calibration")


def test_public_population_plan_is_feasible_and_covers_every_rank(population):
    plan = population.calibration_plan()
    controls = population.calibration_controls(plan)
    assert controls["cases"] == 320
    assert controls["requests"] == 2320
    assert controls["screen"]["maximum_mapped_mib"] == 9984
    assert controls["scope"] == "two_resident_tenants_one_active_per_cost_case_not_joint_active"
    assert controls["outside_population_context_envelope"] == "no_forecast_no_extrapolation"
    cases = population.planned_cases(plan)
    for model in (0, 1):
        for repeat in (0, 1):
            for entry in plan["populations"]:
                for length in entry["prompt_tokens"]:
                    selected = [case["rank_zero"] for case in cases if case["model"] == model
                                and case["repeat"] == repeat and case["population"] == entry["population"]
                                and case["prompt_tokens"] == length]
                    assert sorted(selected) == list(range(entry["population"] + 1))


@pytest.mark.parametrize("field,value", [("seed", 0), ("budget_mib", 11264), ("models", [1]),
                                        ("output_tokens", 8192), ("repeats", [0])])
def test_population_plan_does_not_accept_outcome_retuning(population, field, value):
    plan = population.calibration_plan()
    plan[field] = value
    with pytest.raises(ValueError, match="retuned"):
        population.calibration_controls(plan)


def test_planned_population_contrasts_are_identifiable_with_context_and_layout_holdouts(population):
    contrast = importlib.import_module("analyze_workaware_cost")
    profile = importlib.import_module("workaware_profile_cost")
    calibrated = importlib.import_module("calibrated_cost")
    trials = []
    for case in population.planned_cases(population.calibration_plan()):
        counts = [case["rank_zero"], case["population"] - case["rank_zero"]]
        contexts = [count * (case["prompt_tokens"] + 2048) for count in counts]
        graph = calibrated.graph_padding(counts, calibrated.SPARSE_CAPTURE_SIZES)
        trials.append({"case": case, "sample": {"counts": counts, "context_tokens": contexts,
                       "cadence_s": .01 + graph * .0001 + max(contexts) * 1e-8}})
    fit = {"capture_sizes": calibrated.SPARSE_CAPTURE_SIZES,
           "coefficients": {"graph_batch_s": .0001, "attention_token_s": 1e-8}}
    rows = contrast.paired_contrasts(trials, fit)
    assert not rows["unmatched_groups"]
    profiles = profile.fit_profiles(rows["contrasts"], extended_validation=True)["profiles"]
    assert set(profiles) == {f"{model}:{count}" for model in (0, 1) for count in range(5, 10)}
    assert all(result["contrast_residual_bound_s"] < 1e-12 for result in profiles.values())


def verification_fixture(population, tmp_path, monkeypatch):
    verifier = importlib.import_module("verify_workaware_profile")
    plan = population.calibration_plan()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    plan_path = frozen / "population-calibration-plan.json"
    plan_path.write_text(json.dumps(plan))
    controls = population.calibration_controls(plan)
    parent_fit = {"population_profiles": {"profiles": {f"{model}:{count}": {"model": model, "population": count}
                  for model in (0, 1) for count in range(2, 5)}}}
    parent_path = tmp_path / "parent-public-context-fit.json"
    parent_path.write_text(json.dumps(parent_fit))
    config = {"fit_sha256": population.digest(parent_path),
              "population_calibration": {**controls, "plan_sha256": population.digest(plan_path)}}
    (frozen / "evaluation-config.json").write_text(json.dumps(config))
    raw = tmp_path / "public-population-calibration"
    raw.mkdir()
    raw_config = json.dumps({"workaware_case_plan": plan})
    (raw / "config.json").write_text(raw_config)
    supplement_profiles = {f"{model}:{count}": {"model": model, "population": count}
                           for model in (0, 1) for count in range(5, 10)}
    raw_fit = {"cases": controls["cases"], "requests": controls["requests"],
               "source_files_sha256": {"config.json": population.digest(raw / "config.json")},
               "population_profiles": {"profiles": supplement_profiles}}
    bundle = tmp_path / "public-population-profile"
    for name in ("legacy", "supplement"):
        (bundle / name).mkdir(parents=True)
    (bundle / "supplement/config.json").write_text(raw_config)
    combined = {**parent_fit, "population_profiles": {"profiles": {
                **parent_fit["population_profiles"]["profiles"], **supplement_profiles}}}
    (bundle / "fit.json").write_text(json.dumps(combined))
    fits = {raw: raw_fit, bundle: combined, bundle / "legacy": parent_fit,
            bundle / "supplement": copy.deepcopy(raw_fit)}
    monkeypatch.setattr(verifier, "verify_profile", lambda path: copy.deepcopy(fits[Path(path)]))
    return fits


def test_population_verification_preserves_parent_and_measured_contrasts(population, tmp_path, monkeypatch):
    verification_fixture(population, tmp_path, monkeypatch)
    result = population.verify_calibration(tmp_path)
    assert result["status"] == "GPU_raw_population_cost_SHA_verified_not_benefit"
    assert result["fit_sha256"] == population.digest(tmp_path / "public-population-profile/fit.json")


@pytest.mark.parametrize("changed", ["plan", "raw_plan", "parent", "fit", "raw_copy", "contrasts"])
def test_population_verification_rejects_changed_provenance(population, tmp_path, monkeypatch, changed):
    fits = verification_fixture(population, tmp_path, monkeypatch)
    if changed == "plan":
        path = tmp_path / "frozen/population-calibration-plan.json"
        plan = population.calibration_plan()
        plan["seed"] += 1
        path.write_text(json.dumps(plan))
    elif changed == "raw_plan":
        (tmp_path / "public-population-calibration/config.json").write_text("{}")
    elif changed == "parent":
        (tmp_path / "parent-public-context-fit.json").write_text("{}")
    elif changed == "fit":
        (tmp_path / "public-population-profile/fit.json").write_text("{}")
    elif changed == "raw_copy":
        (tmp_path / "public-population-profile/supplement/config.json").write_text("{}")
    else:
        fits[tmp_path / "public-population-profile/supplement"]["population_profiles"]["profiles"]["1:5"]["population"] = 6
    with pytest.raises(ValueError):
        population.verify_calibration(tmp_path)
