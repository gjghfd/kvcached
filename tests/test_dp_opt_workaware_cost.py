import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_cost")


def fit():
    return {"status": "raw_verified_workaware_cadence_fit", "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
            "scope": "one_active_tenant_other_resident_idle", "source_verification": "raw_verified",
            "coefficients": {"base_s": .01, "graph_batch_s": .001, "attention_token_s": 1e-7},
            "residual_bound_s": 1e-5, "max_population": 9, "max_context_tokens": 32768}


def request(identifier, remaining=1000, context=4096):
    return {"id": identifier, "movable": True, "max_tokens": remaining + 100,
            "output_tokens": 100, "computed_tokens": context}


def observed(requests):
    geometry = {"uniform_page_geometry": True, "block_size": 16, "page_block_capacity": 64,
                "page_bytes": 96 * 1024**2}
    return [[{**geometry, "requests": []}, {**geometry, "requests": []}],
            [{**geometry, "requests": requests}, {**geometry, "requests": []}]]


def test_short_and_long_remaining_work_have_separate_completion_segments(module):
    segments = module.remaining_segments(observed([request("short", 10), request("long", 100)]), fit(), 1000)
    assert [(segment["start_step"], segment["end_step"]) for segment in segments] == [(0, 10), (10, 100)]
    assert segments[0]["population"] == 2 and segments[1]["population"] == 1
    assert set(segments[1]["request_sources"]) == {"long"}


def test_partial_return_eliminates_real_default_capture_jump(module):
    population = observed([request(f"request-{index}") for index in range(9)])
    segments = module.remaining_segments(population, fit())
    original = module.evaluate_segments(segments)
    partial = module.evaluate_segments(segments, ["request-0"])
    balanced = module.evaluate_segments(segments, [f"request-{index}" for index in range(4)])
    assert original["graph_cost_s"] > partial["graph_cost_s"] == balanced["graph_cost_s"] == 0
    assert original["attention_cost_s"] > partial["attention_cost_s"] > balanced["attention_cost_s"]


def test_two_requests_use_capture_one_when_balanced_not_dense_capture_two(module):
    segments = module.remaining_segments(observed([request("first"), request("second")]), fit())
    assert module.evaluate_segments(segments)["graph_cost_s"] == pytest.approx(14)
    assert module.evaluate_segments(segments, ["first"])["graph_cost_s"] == 0


def test_attention_weights_use_context_and_remaining_exposure(module):
    segments = module.remaining_segments(observed([request("large", context=12000), request("small", context=2000)]), fit())
    returned = module.evaluate_segments(segments, ["large"])
    assert returned["attention_cost_s"] < module.evaluate_segments(segments)["attention_cost_s"]
    assert returned["attention_cost_s"] == module.evaluate_segments(segments, ["small"])["attention_cost_s"]


def test_forecast_domain_limit_does_not_modify_or_truncate_request(module):
    population = observed([request("long", 1000, 32760)])
    untouched = copy.deepcopy(population)
    segments = module.remaining_segments(population, fit())
    assert segments[-1]["end_step"] == 8
    assert population == untouched


@pytest.mark.parametrize("alteration", ["mixed", "unmovable", "duplicate", "outside", "unfinished_fit"])
def test_unsupported_or_unverified_conditions_are_explicit(module, alteration):
    population, calibration = observed([request("first")]), fit()
    if alteration == "mixed":
        population[0][0]["requests"] = [request("other")]
    elif alteration == "unmovable":
        population[1][0]["requests"][0]["movable"] = False
    elif alteration == "duplicate":
        population[1][1]["requests"] = [request("first")]
    elif alteration == "outside":
        population[1][0]["requests"][0]["computed_tokens"] = 40000
    else:
        calibration["source_verification"] = "pending"
    with pytest.raises(ValueError):
        module.remaining_segments(population, calibration)


def test_low_arrival_rate_reduces_soft_headroom_cost_without_relaxing_physical_caps(module):
    snapshot = {"capacity_gib": [10, 10]}
    profiles = [{"rate_rps": .01, "tokens": 100000}, {"rate_rps": 0, "tokens": 0}]
    high = module.headroom_modes(snapshot, observed([]), profiles, 100, [20, 0])
    profiles[0]["rate_rps"] = .0005
    low = module.headroom_modes(snapshot, observed([]), profiles, 100, [20, 0])
    assert len(high) == len(low) == 3
    assert high[0]["risk_cost_s"] > low[0]["risk_cost_s"] > 0
    assert all(scenario["snapshot"]["capacity_gib"] == [10, 10] for scenario in low)
    assert low[1]["snapshot"]["final_capacity_gib"][0] < 1


def test_empty_history_does_not_invent_future_long_request(module):
    scenarios = module.headroom_modes({"capacity_gib": [10, 10]}, observed([]),
                                     [{"rate_rps": .01, "tokens": 0}, {"rate_rps": 0, "tokens": 0}], 100, [20, 20])
    assert len(scenarios) == 1 and scenarios[0]["guards"] == [] and scenarios[0]["risk_cost_s"] == 0


def test_unknown_migration_is_not_silently_ignored(module):
    with pytest.raises(ValueError, match="absent"):
        module.evaluate_segments(module.remaining_segments(observed([request("first")]), fit()), ["unknown"])


def test_nonnegative_fit_identifies_graph_and_attention_but_is_not_raw_verified(module):
    samples = []
    for counts in ([1, 1], [2, 0], [2, 1], [3, 0], [4, 4], [9, 0]):
        for context in (2000, 8000):
            padded = next(size for size in (1, 8, 16, 32, 64, 128, 256) if size >= max(counts))
            peak = max(counts) * context
            samples.append({"counts": counts, "context_tokens": [count * context for count in counts],
                            "cadence_s": .01 + .001 * padded + 1e-7 * peak})
    result = module.fit_cadence(samples)
    assert result["coefficients"] == pytest.approx(fit()["coefficients"])
    assert result["residual_bound_s"] < 1e-9
    assert result["status"] == "unverified_fit_requires_raw_calibration_verification"


def test_unidentifiable_fit_is_rejected(module):
    with pytest.raises(ValueError, match="identifiable"):
        module.fit_cadence([{"counts": [1, 1], "context_tokens": [4096, 4096], "cadence_s": .02}] * 6)
