import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_profile_cost")


def contrast_rows():
    rows = []
    for population, graph_delta in ((2, 7), (3, 0), (8, 0), (9, 8)):
        for repeat in (0, 1):
            for context in (4000, 8000, 16000):
                for delta in sorted({0, graph_delta}):
                    if population == 2 and delta == 0:
                        continue
                    rows.append({"model": 1, "population": population, "repeat": repeat,
                                 "graph_delta": delta, "peak_context_delta_tokens": context,
                                 "measured_cadence_delta_s": (.006 if delta else 0) + 1e-7 * context})
    return rows


def segment(sources, weights, uncertainty=100):
    return {"request_sources": sources, "attention_weights": weights, "uncertainty_request_s": uncertainty}


def test_profiles_fit_non_linear_sparse_transition_by_population(module):
    profiles = module.fit_profiles(contrast_rows())["profiles"]
    assert profiles["1:2"]["graph_penalties_s"]["8"] == pytest.approx(.006)
    assert profiles["1:9"]["graph_penalties_s"]["16"] == pytest.approx(.006)
    assert profiles["1:3"]["graph_penalties_s"] == {"8": 0}
    for profile in profiles.values():
        assert profile["attention_token_s"] == pytest.approx(1e-7)
        assert profile["contrast_residual_bound_s"] == pytest.approx(0, abs=1e-14)


def test_uncertainty_uses_heldout_repeat_not_training_residual(module):
    rows = contrast_rows()
    for row in rows:
        if row["repeat"] == 1:
            row["measured_cadence_delta_s"] += .001
    assert module.fit_profiles(rows)["profiles"]["1:2"]["contrast_residual_bound_s"] == pytest.approx(.001)


def test_profiles_do_not_pool_different_tenants(module):
    rows = contrast_rows()
    rows += [{**row, "model": 0, "measured_cadence_delta_s": row["measured_cadence_delta_s"] * 2} for row in rows]
    profiles = module.fit_profiles(rows)["profiles"]
    assert profiles["0:2"]["attention_token_s"] == pytest.approx(2e-7)
    assert profiles["1:2"]["attention_token_s"] == pytest.approx(1e-7)


def test_identifiability_and_holdout_are_required(module):
    with pytest.raises(ValueError, match="held-out"):
        module.fit_profiles([row for row in contrast_rows() if row["repeat"] == 0])
    with pytest.raises(ValueError, match="identifiable"):
        module.fit_profiles([row for row in contrast_rows() if row["population"] == 2 and row["peak_context_delta_tokens"] == 4000])


def test_placement_invariant_single_request_horizon_does_not_block_return(module):
    segments = [segment({"short": 0, "long": 0}, {"short": 1000, "long": 2000}, 3),
                segment({"long": 0}, {"long": 10000}, 1000)]
    assert module.changed_segment_uncertainty(segments, ["short"]) == 3
    assert module.changed_segment_uncertainty(segments, ["long"]) == 3
    assert module.changed_segment_uncertainty(segments, []) == 0


def test_rank_symmetric_swap_is_not_charged_uncertainty(module):
    segments = [segment({"first": 0, "second": 1}, {"first": 5000, "second": 7000})]
    assert module.changed_segment_uncertainty(segments, ["first", "second"]) == 0
    assert module.changed_segment_uncertainty(segments, ["second"]) == 100


def test_unknown_moves_are_rejected(module):
    with pytest.raises(ValueError, match="absent"):
        module.changed_segment_uncertainty([segment({"first": 0}, {"first": 1000})], ["unknown"])


def forecasting_fixture():
    fit = {"status": "raw_verified_workaware_cadence_fit", "source_verification": "raw_verified",
           "scope": "one_active_tenant_other_resident_idle", "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
           "coefficients": {"base_s": .01, "graph_batch_s": .001, "attention_token_s": 1e-7},
           "residual_bound_s": .01, "max_population": 9, "max_context_tokens": 32768}
    requests = [{"id": f"request-{remaining}", "max_tokens": remaining + 1, "output_tokens": 1,
                 "computed_tokens": 5000, "movable": True} for remaining in (1, 2, 3)]
    observed = [[{"requests": []}, {"requests": []}], [{"requests": requests}, {"requests": []}]]
    return observed, fit


def test_population_profiles_follow_completion_segments(module):
    observed, fit = forecasting_fixture()
    profiles = module.fit_profiles(contrast_rows())
    segments = module.profile_segments(observed, fit, profiles)
    assert [entry["population"] for entry in segments] == [3, 2, 1]
    assert segments[1]["graph_costs_s"] == pytest.approx([.012, 0, .012])
    assert segments[1]["attention_difference_cost_s"] == pytest.approx(1e-7)
    assert segments[-1]["uncertainty_request_s"] == 0


def test_unmeasured_future_population_truncates_not_extrapolates(module):
    observed, fit = forecasting_fixture()
    profiles = module.fit_profiles([row for row in contrast_rows() if row["population"] == 3])
    segments = module.profile_segments(observed, fit, profiles)
    assert [entry["population"] for entry in segments] == [3]


def test_missing_measured_graph_is_not_silently_zero(module):
    observed, fit = forecasting_fixture()
    profiles = module.fit_profiles(contrast_rows())
    del profiles["profiles"]["1:2"]["graph_penalties_s"]["8"]
    with pytest.raises(ValueError, match="uncalibrated"):
        module.profile_segments(observed, fit, profiles)


@pytest.mark.parametrize("field", ("max_context_tokens", "max_attention_tokens"))
def test_population_specific_domain_is_not_borrowed_from_larger_populations(module, field):
    observed, fit = forecasting_fixture()
    profiles = module.fit_profiles(contrast_rows())
    profiles["profiles"]["1:3"][field] = 4000
    with pytest.raises(ValueError, match="No measured"):
        module.profile_segments(observed, fit, profiles)


def test_candidate_attention_peak_must_stay_in_population_domain(module):
    from workaware_cost import evaluate_segments

    observed, fit = forecasting_fixture()
    requests = observed[1][0]["requests"][:2]
    observed[1] = [{"requests": requests[:1]}, {"requests": requests[1:]}]
    profiles = module.fit_profiles(contrast_rows())
    profiles["profiles"]["1:2"]["max_attention_tokens"] = 7000
    segments = module.profile_segments(observed, fit, profiles)
    assert evaluate_segments(segments)["performance_cost_s"] >= 0
    with pytest.raises(ValueError, match="outside measured"):
        evaluate_segments(segments, [requests[0]["id"]])
