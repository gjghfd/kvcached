import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("kv_pressure")


def observation(mapped_mib, occupied=False):
    return {"mapped_bytes": mapped_mib * 1024**2, "requests": [{}] if occupied else []}


def decisions():
    return [{"observations": [[observation(96), observation(96)], [observation(96), observation(96)]],
             "after_admission": [[observation(9600, True), observation(96)],
                                 [observation(96), observation(96)]]}]


def sample(elapsed, mapped, code=False, stable=True):
    return {"elapsed_s": elapsed, "mapped_gib": [mapped / 1024] * 2,
            "counts": [[int(code), 0], [1, 1]], "reserved_mib": [[0, 0], [1000, 1000]],
            "admission_stable_across_observation": stable}


def test_time_weighting_and_idle_Code_not_double_counted(module):
    population = [sample(0, 2192), sample(2, 4192), sample(10, 4192)]
    result = module.background_pressure(population, decisions(), 0, 10, 10240)
    assert result["Conv_physical_mean_mib_per_gpu"] == pytest.approx(3696)
    assert result["Conv_logical_mean_mib_per_gpu"] == 1000
    assert result["local_physical_ratio"] == pytest.approx((3696 + 9600) / 10240)
    assert result["aggregate_physical_ratio"] == pytest.approx((2 * 3696 + 9600 + 96) / 20480)


def test_Code_residence_and_unstable_samples_excluded(module):
    population = [sample(0, 2192), sample(2, 4192, code=True),
                  sample(4, 4192, stable=False), sample(6, 2192), sample(10, 2192)]
    result = module.background_pressure(population, decisions(), 0, 10, 10240)
    assert result["background_covered_s"] == 6
    assert result["Conv_physical_mean_mib_per_gpu"] == 2096


def test_lower_rate_accounts_for_fixed_idle_Conv_pages(module):
    pressure = {"idle_pool_mib": [[96, 96], [96, 96]], "Conv_physical_mean_mib_per_gpu": 2845,
                "budget_mib_per_gpu": 10240, "Code_single_physical_pool_mib": 9600}
    result = module.rate_for_ratio(pressure, .0307909199195414, 1.1)
    assert result["target_Conv_physical_mib_per_gpu"] == pytest.approx(1664)
    assert result["Conv_rate_rps_candidate"] == pytest.approx(.0307909199195414 * 1568 / 2749)


@pytest.mark.parametrize("ratio", [0, float("nan"), float("inf"), .94, 1.3])
def test_invalid_pressure_targets_rejected(module, ratio):
    pressure = {"idle_pool_mib": [[96, 96], [96, 96]], "Conv_physical_mean_mib_per_gpu": 2845,
                "budget_mib_per_gpu": 10240, "Code_single_physical_pool_mib": 9600}
    with pytest.raises(ValueError):
        module.rate_for_ratio(pressure, .03, ratio)


def test_time_dilation_preserves_Code_lengths_and_Conv_mark_prefix(module):
    original = {"model_rates_rps": [.001, .03], "total_rate_rps": .031, "duration_s": 100,
                "requests": [{"model": 1, "arrival_s": 10, "request_id": "conv0", "prompt_tokens": 4000},
                             {"model": 0, "arrival_s": 35, "request_id": "code0", "prompt_tokens": 100000},
                             {"model": 1, "arrival_s": 60, "request_id": "conv1", "prompt_tokens": 5000}]}
    untouched = copy.deepcopy(original)
    result = module.lower_conv_rate(original, .015, "a" * 64)
    assert original == untouched
    assert result["requests"] == [{**original["requests"][0], "arrival_s": 20}, original["requests"][1]]
    assert result["model_rates_rps"] == [.001, .015]
    assert result["rate_change"]["arrival_time_scale"] == 2


@pytest.mark.parametrize("rate", [0, -.1, .03, .04, float("nan"), float("inf")])
def test_only_lower_finite_rates_supported(module, rate):
    with pytest.raises(ValueError):
        module.lower_conv_rate({"model_rates_rps": [.001, .03]}, rate, "a" * 64)


def test_unordered_population_rejected(module):
    with pytest.raises(ValueError, match="strictly increase"):
        module.background_pressure([sample(2, 2192), sample(1, 2192)], decisions(), 0, 10, 10240)


def test_feedback_can_increase_candidate_below_original_rate(module):
    pressure = {"idle_pool_mib": [[96, 96], [96, 96]],
                "Conv_physical_mean_mib_per_gpu": 1368.6668626136982,
                "budget_mib_per_gpu": 10240, "Code_single_physical_pool_mib": 9600}
    result = module.rate_for_ratio(pressure, .01756044902350957, 1.1,
                                  maximum_rate=.0307909199195414)
    assert result["Conv_rate_rps_candidate"] == pytest.approx(.021635500127908054)
    assert .01756044902350957 < result["Conv_rate_rps_candidate"] < result["Conv_rate_rps_ceiling"]


def test_feedback_rejects_crossing_original_rate_ceiling(module):
    pressure = {"idle_pool_mib": [[96, 96], [96, 96]], "Conv_physical_mean_mib_per_gpu": 1368,
                "budget_mib_per_gpu": 10240, "Code_single_physical_pool_mib": 9600}
    with pytest.raises(ValueError, match="ceiling"):
        module.rate_for_ratio(pressure, .01756, 1.3, maximum_rate=.03079)


def pressure_fixture(mapped_mib):
    return {"idle_pool_mib": [[96, 96], [96, 96]], "budget_mib_per_gpu": 10240,
            "Code_single_physical_pool_mib": 9600, "evaluation_start_s": 600,
            "evaluation_end_s": 2400, "Conv_physical_mean_mib_per_gpu": mapped_mib}


def test_bracket_interpolates_measured_pressure_not_gain(module):
    lower = pressure_fixture(1368.6668626136982)
    upper = pressure_fixture(1968.5292805757963)
    result = module.rate_for_bracket(lower, .01756044902350957, upper, .021635500127908054,
                                     1.1, .0307909199195414)
    reverse = module.rate_for_bracket(upper, .021635500127908054, lower, .01756044902350957,
                                      1.1, .0307909199195414)
    assert result["Conv_rate_rps_candidate"] == pytest.approx(.01956673845194241)
    assert reverse["Conv_rate_rps_candidate"] == result["Conv_rate_rps_candidate"]
    assert result["target_Conv_physical_mib_per_gpu"] == pytest.approx(1664)
    assert "requires_GPU_measurement" in result["scope"]


@pytest.mark.parametrize("target", [1, 1.2, float("nan"), float("inf")])
def test_bracket_refuses_extrapolation_or_nonfinite_target(module, target):
    with pytest.raises(ValueError):
        module.rate_for_bracket(pressure_fixture(1368), .01756, pressure_fixture(1968), .02163,
                                target, .03079)


@pytest.mark.parametrize("field,value", [("budget_mib_per_gpu", 11264),
                                        ("Code_single_physical_pool_mib", 9504),
                                        ("idle_pool_mib", [[96, 96], [96, 192]]),
                                        ("evaluation_end_s", 2500)])
def test_bracket_requires_identical_geometry_and_window(module, field, value):
    upper = pressure_fixture(1968)
    upper[field] = value
    with pytest.raises(ValueError, match="geometry or measurement window"):
        module.rate_for_bracket(pressure_fixture(1368), .01756, upper, .02163, 1.1, .03079)


@pytest.mark.parametrize("lower_mib,upper_mib,lower_rate,upper_rate,ceiling", [
    (1968, 1368, .01756, .02163, .03079),
    (1368, 1968, .01756, .01756, .03079),
    (1368, 1968, .01756, .02163, .02163),
    (float("nan"), 1968, .01756, .02163, .03079),
    (1368, 1968, 0, .02163, .03079),
])
def test_bracket_rejects_invalid_measurement_or_ceiling(module, lower_mib, upper_mib, lower_rate, upper_rate, ceiling):
    with pytest.raises(ValueError):
        module.rate_for_bracket(pressure_fixture(lower_mib), lower_rate,
                                pressure_fixture(upper_mib), upper_rate, 1.1, ceiling)


def test_profile_validation_allows_paths_but_not_scheduler_changes(module):
    config = {"checkpoint": "qwen", "kv_budget_mib": 10240, "rebalance_max_moves": 4,
              "output": "old", "trace": "old", "port": 1}
    trace = {"duration_s": 2400, "evaluation_start_s": 600, "evaluation_duration_s": 1800,
             "prompt_bank_sha256": "bank", "synthetic_Conv_output_tokens": 8192,
             "requests": [{"model": 0, "arrival_s": 1000}]}
    profile = {**config, "output": "new", "trace": "new", "port": 2}
    module.validate_profile(config, trace, profile, trace)
    with pytest.raises(ValueError, match="runtime controls"):
        module.validate_profile(config, trace, {**profile, "rebalance_max_moves": 8}, trace)
    with pytest.raises(ValueError, match="complete Code stream"):
        module.validate_profile(config, trace, profile, {**trace, "requests": [{"model": 0, "arrival_s": 1001}]})
