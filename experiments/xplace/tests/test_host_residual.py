import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from host_residual import HostResidualModel, fit_host_residual


GROUP = ["instance-1", 0, 0, "worker_sample"]


def episodes(durations, outcomes=None):
    outcomes = outcomes or ["returned"] * len(durations)
    return [dict(id=f"episode-{index}", case=f"case-{index}", group=GROUP, duration_s=duration, outcome=outcome)
            for index, (duration, outcome) in enumerate(zip(durations, outcomes))]


def model(rows, **changes):
    return HostResidualModel(fit_host_residual(rows, **dict(source_id="synthetic-test", minimum_at_risk=1,
                                                           minimum_cases=1, **changes)))


def predict(estimator, age=0, lag=0, horizon=10, group=GROUP):
    return estimator.predict(group, observed_age_s=age, observation_lag_s=lag, restriction_s=horizon)


def test_complete_empirical_mean_and_conditional_tail_are_learned_not_constants():
    estimator = model(episodes([1, 2, 4]))
    assert predict(estimator)["full_mean_s"] == pytest.approx(7 / 3)
    assert predict(estimator, age=1)["full_mean_s"] == pytest.approx(2)
    assert predict(estimator, age=2)["full_mean_s"] == pytest.approx(2)


def test_receipt_lag_does_not_condition_on_unobserved_survival_at_decision():
    estimator = model(episodes([1, 3]))
    result = predict(estimator, age=0.5, lag=1)
    assert result["already_returned_probability"] == 0.5
    assert result["full_mean_s"] == 0.75
    assert predict(estimator, age=1.5)["full_mean_s"] == 1.5


def test_lag_past_all_observed_returns_is_zero_time_not_gpu_idle_claim():
    result = predict(model(episodes([1, 3])), age=0.5, lag=4)
    assert result["full_mean_s"] == 0
    assert result["already_returned_probability"] == 1
    assert result["target"] == "host_return_not_device_completion_or_output_milestone"
    assert not result["prediction_gate_passed"]


def test_restricted_mean_and_full_mean_are_not_conflated():
    result = predict(model(episodes([1, 4])), horizon=2)
    assert result["restricted_mean_s"] == 1.5
    assert result["full_mean_s"] is None
    assert result["unresolved_at_horizon_probability"] == 0.5


def test_censored_tail_is_not_extrapolated_or_dropped():
    estimator = model(episodes([1, 2], ["returned", "right_censored"]))
    identified = predict(estimator, horizon=2)
    assert identified["restricted_mean_s"] == 1.5
    assert identified["full_mean_s"] is None
    missing = predict(estimator, horizon=3)
    assert missing["status"] == "unavailable" and missing["reason"] == "unidentified_censored_tail"
    assert missing["restricted_mean_s"] is None
    assert missing["restricted_mean_lower_bound_s"] == 1.5


def test_tied_return_and_censor_both_enter_same_risk_set():
    estimator = model(episodes([1, 1, 3], ["returned", "right_censored", "returned"]))
    result = predict(estimator, horizon=3)
    assert result["full_mean_s"] == pytest.approx(1 + 2 * 2 / 3)


def test_failed_call_is_a_recorded_terminal_cause_not_success():
    estimator = model(episodes([1, 3], ["failed", "returned"]))
    result = predict(estimator, horizon=4)
    assert result["full_mean_s"] == 2
    assert result["failed_return_probability_by_horizon"] == 0.5
    assert predict(estimator, age=1)["failed_return_probability_by_horizon"] == 0


@pytest.mark.parametrize("age", [3, 4, 10])
def test_no_survivors_gives_unavailable_not_zero_fallback(age):
    result = predict(model(episodes([1, 3])), age=age)
    assert result["status"] == "unavailable" and result["reason"] == "insufficient_conditional_support"


def test_tail_case_support_matters_not_only_total_number_of_cases():
    rows = episodes([1, 2, 10, 11])
    rows[-1]["case"] = rows[-2]["case"]
    estimator = HostResidualModel(fit_host_residual(rows, source_id="test", minimum_at_risk=2, minimum_cases=2))
    assert predict(estimator, age=0)["status"] == "estimated_exploratory"
    result = predict(estimator, age=2)
    assert result["status"] == "unavailable" and result["cases_at_risk"] == 1


def test_unseen_model_lane_or_host_kind_never_borrows_a_different_group():
    estimator = model(episodes([1, 3]))
    for index, replacement in [(0, "instance-2"), (1, 1), (2, 1), (3, "worker_execute")]:
        group = list(GROUP)
        group[index] = replacement
        assert predict(estimator, group=group)["reason"] == "unseen_group"


def test_model_is_detached_from_mutable_training_and_parameter_inputs():
    rows = episodes([1, 3])
    artifact = fit_host_residual(rows, source_id="test", minimum_at_risk=1, minimum_cases=1)
    estimator = HostResidualModel(artifact)
    rows[0]["duration_s"] = 100
    artifact["groups"].clear()
    assert predict(estimator)["full_mean_s"] == 2
    with pytest.raises(ValueError, match="changed"):
        HostResidualModel(artifact)


@pytest.mark.parametrize("field,value", [("duration_s", -1), ("duration_s", True), ("duration_s", float("nan")),
    ("duration_s", float("inf")), ("outcome", "missing"), ("case", ""), ("id", ""), ("group", [])])
def test_bad_episode_fails_fit_instead_of_silently_changing_training_population(field, value):
    rows = episodes([1, 3])
    rows[0][field] = value
    with pytest.raises(ValueError):
        model(rows)


def test_duplicate_episodes_cannot_inflate_support():
    rows = episodes([1, 3])
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        model(rows)


@pytest.mark.parametrize("name,value", [("observed_age_s", -1), ("observation_lag_s", -1),
    ("restriction_s", 0), ("observed_age_s", True), ("observation_lag_s", float("inf"))])
def test_invalid_time_query_rejected(name, value):
    query = dict(observed_age_s=0, observation_lag_s=0, restriction_s=1)
    query[name] = value
    with pytest.raises(ValueError):
        model(episodes([1, 3])).predict(GROUP, **query)


@pytest.mark.parametrize("scale", [0.1, 1.0, 3.7])
def test_residual_model_scales_with_time_without_refitting_selection(scale):
    result = predict(model(episodes([scale, 3 * scale])), age=0.5 * scale, lag=scale, horizon=10 * scale)
    assert result["full_mean_s"] == pytest.approx(0.75 * scale)
    assert result["already_returned_probability"] == pytest.approx(0.5)


def test_all_censoring_can_identify_only_an_observed_restricted_interval():
    estimator = model(episodes([3, 4], ["right_censored", "right_censored"]))
    assert predict(estimator, horizon=2)["restricted_mean_s"] == 2
    assert predict(estimator, horizon=5)["reason"] == "unidentified_censored_tail"


def test_empty_training_has_no_artificial_prior():
    assert predict(model([]))["reason"] == "unseen_group"
