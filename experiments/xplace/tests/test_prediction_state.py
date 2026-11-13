import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, default_dp2_params
from planner import DEFAULT_PARAMS, plan_admission, plan_return
from prediction_state import joint_prediction_state
from test_scoring_state import observed_state


def live_inputs(counts=(2, 2)):
    state, _ = observed_state(counts)
    clock = "job:test/single-host-monotonic"
    delivery = {"kind": "live_scheduler_snapshot", "clock_id": clock,
                "available_at": [[10.2, 10.2], [10.2, 10.2]]}
    execution = {"kind": "live_execution_snapshot", "clock_id": clock,
                 "workers": [{"model": model, "dp_rank": rank, "tp_rank": lane,
                              "observed_at": 10.0, "available_at": 10.2, "wave_active": False,
                              "host_call": "none", "host_call_started_at": None}
                             for model in (0, 1) for rank in (0, 1) for lane in (0, 1)]}
    kwargs = {"as_of": 10.5, "max_age_s": 1.0, "max_skew_s": 0.5, "clock_id": clock,
              "tensor_parallel_size": 2, "logical_delivery": delivery, "execution_delivery": execution}
    return state, kwargs


def test_complete_live_state_never_becomes_a_horizon_or_gpu_idle_claim():
    state, kwargs = live_inputs((0, 0))
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "available" and not result["errors"]
    assert result["logical_candidate"][1]["phase"] == "idle"
    assert result["device_activity"] == "unknown"
    assert result["execution_observed"]["device_activity"] == "unknown"
    assert result["forecast_status"] == "unavailable_not_estimated"
    assert not {"cost", "other_active", "horizon", "horizon_steps"}.intersection(result)


@pytest.mark.parametrize("field", ["logical_delivery", "execution_delivery"])
def test_missing_delivery_is_explicitly_unavailable_not_solo(field):
    state, kwargs = live_inputs((0, 0))
    kwargs.pop(field)
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable"
    assert {"component": "execution_observed" if field == "execution_delivery" else field,
            "reason": "missing_delivery"} in result["errors"]


@pytest.mark.parametrize("field", ["logical_delivery", "execution_delivery"])
@pytest.mark.parametrize("changed,value,reason", [
    ("kind", "post_stop_export", "not_live_delivery"),
    ("clock_id", "another-node", "clock_domain_mismatch"),
])
def test_archive_timestamps_and_other_clocks_are_not_online_evidence(field, changed, value, reason):
    state, kwargs = live_inputs()
    kwargs[field][changed] = value
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable"
    assert any(error["reason"] == reason for error in result["errors"])


@pytest.mark.parametrize("observed,available,reason", [
    (10.0, 10.6, "not_available_at_decision"),
    (10.0, 9.9, "availability_before_observation"),
    (8.0, 10.0, "stale_observation"),
    (10.0, None, "missing_or_invalid_timestamp"),
    (float("nan"), 10.0, "missing_or_invalid_timestamp"),
    (10.0, True, "missing_or_invalid_timestamp"),
])
def test_worker_availability_not_event_time_controls_causal_prefix(observed, available, reason):
    state, kwargs = live_inputs()
    kwargs["execution_delivery"]["workers"][0].update(observed_at=observed, available_at=available)
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable" and result["execution_observed"]["reason"] == reason


@pytest.mark.parametrize("available,reason", [
    (10.6, "not_available_at_decision"), (9.9, "availability_before_observation"),
    (float("inf"), "missing_or_invalid_timestamp"), (False, "missing_or_invalid_timestamp"),
])
def test_scheduler_snapshot_requires_receipt_time(available, reason):
    state, kwargs = live_inputs()
    kwargs["logical_delivery"]["available_at"][0][0] = available
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable" and result["logical_delivery"]["reason"] == reason


@pytest.mark.parametrize("change,reason", [
    ({"model": True}, "invalid_worker_identity"),
    ({"tp_rank": 2}, "invalid_worker_identity"),
    ({"wave_active": None}, "missing_wave_state"),
    ({"host_call": "idle"}, "missing_host_call_state"),
    ({"host_call": "sample", "host_call_started_at": 10.1}, "invalid_host_call_start"),
    ({"host_call": "dummy", "host_call_started_at": None}, "invalid_host_call_start"),
    ({"host_call_started_at": 9.0}, "invalid_host_call_start"),
])
def test_incomplete_or_ambiguous_worker_metadata_is_rejected(change, reason):
    state, kwargs = live_inputs()
    kwargs["execution_delivery"]["workers"][0].update(change)
    result = joint_prediction_state(state, **kwargs)
    assert result["execution_observed"]["reason"] == reason


@pytest.mark.parametrize("variant,reason", [
    ("missing", "incomplete_worker_lanes"), ("duplicate", "duplicate_worker"),
    ("wave", "inconsistent_rank_wave"), ("malformed", "malformed_worker"),
])
def test_all_lanes_and_consistent_rank_wave_are_required(variant, reason):
    state, kwargs = live_inputs()
    workers = kwargs["execution_delivery"]["workers"]
    if variant == "missing":
        workers.pop()
    elif variant == "duplicate":
        workers.append(copy.deepcopy(workers[0]))
    elif variant == "wave":
        workers[0]["wave_active"] = True
    else:
        workers[0] = None
    assert joint_prediction_state(state, **kwargs)["execution_observed"]["reason"] == reason


def test_joint_skew_boundary_and_inclusive_availability_boundary():
    state, kwargs = live_inputs()
    kwargs["execution_delivery"]["workers"][0].update(observed_at=9.5, available_at=10.5)
    assert joint_prediction_state(state, **kwargs)["status"] == "available"
    kwargs["max_skew_s"] = 0.499
    result = joint_prediction_state(state, **kwargs)
    assert result["errors"] == [{"component": "joint_clock", "reason": "snapshot_skew_exceeded"}]


def test_candidate_layout_does_not_relocate_observed_wave_or_calls():
    state, kwargs = live_inputs()
    workers = kwargs["execution_delivery"]["workers"]
    for worker in workers:
        if worker["model"] == 1 and worker["dp_rank"] == 0:
            worker.update(wave_active=True, host_call="dummy", host_call_started_at=9.0)
    before = copy.deepcopy((state, kwargs))
    move = {"id": "conv-0-0", "model": 1, "source": 0, "destination": 1}
    arrival = {"model": 0, "prompt_tokens": 101167, "max_tokens": 999999, "true_remaining": 999999}
    result = joint_prediction_state(state, **kwargs, moves=iter([move]), arrival=arrival, target=1)
    assert result["status"] == "available"
    assert result["logical_observed"][1]["counts"]["decode"] == [2, 2]
    assert result["logical_candidate"][1]["counts"]["decode"] == [1, 3]
    assert result["logical_candidate"][0]["counts"]["prefill"][1] == 1
    assert result["execution_observed"]["workers"] == workers
    assert result["directional_indices"][0]["peer_counts"]["decode"] == [1, 3]
    assert result["directional_indices"][1]["own_counts"]["decode"] == [1, 3]
    assert result["directional_indices"][1]["peer_counts"]["prefill"][1] == 1
    assert "true_remaining" not in str(result) and "max_tokens" not in str(result)
    assert (state, kwargs) == before
    result["execution_observed"]["workers"][0]["host_call"] = "sample"
    assert (state, kwargs) == before


def test_tenant_swap_swaps_directional_features_not_an_implicit_conv_index():
    state, kwargs = live_inputs((3, 1))
    original = joint_prediction_state(state, **kwargs)
    swapped = copy.deepcopy(state)
    swapped["scoring_observations"].reverse()
    kwargs["logical_delivery"]["available_at"].reverse()
    for worker in kwargs["execution_delivery"]["workers"]:
        worker["model"] = 1 - worker["model"]
    mirrored = joint_prediction_state(swapped, **kwargs)
    for model in (0, 1):
        for key in ("own_counts", "peer_counts", "own_accounted_contexts", "peer_accounted_contexts"):
            assert mirrored["directional_indices"][model][key] == original["directional_indices"][1 - model][key]


@pytest.mark.parametrize("changed,value", [
    ("as_of", True), ("max_age_s", -1), ("max_skew_s", float("nan")),
    ("clock_id", ""), ("tensor_parallel_size", False), ("tensor_parallel_size", 0),
])
def test_invalid_query_parameters_are_not_defaulted(changed, value):
    state, kwargs = live_inputs()
    kwargs[changed] = value
    with pytest.raises(ValueError, match="joint-state"):
        joint_prediction_state(state, **kwargs)


def test_contract_does_not_change_planner_scoring_or_capacity():
    state, kwargs = live_inputs((10, 2))
    before = copy.deepcopy(state)
    model = StepModel(default_dp2_params())
    arrival = {"model": 0, "tokens": 101167, "prompt_tokens": 101167}
    admission = plan_admission(state, arrival, "ca", model, DEFAULT_PARAMS)
    periodic = plan_return(state, model, DEFAULT_PARAMS)
    assert joint_prediction_state(state, **kwargs)["forecast_status"] == "unavailable_not_estimated"
    assert state == before
    assert plan_admission(state, arrival, "ca", model, DEFAULT_PARAMS) == admission
    assert plan_return(state, model, DEFAULT_PARAMS) == periodic


@pytest.mark.parametrize("available", [None, [], [[10.2], [10.2]], [[10.2, 10.2], None]])
def test_partial_scheduler_delivery_does_not_accept_complete_ranks_only(available):
    state, kwargs = live_inputs()
    kwargs["logical_delivery"]["available_at"] = available
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable"
    assert result["logical_delivery"]["reason"] == "incomplete_rank_delivery"


@pytest.mark.parametrize("variant,reason", [
    ("absent", "missing_rank_observations"), ("stale", "stale_timestamp"),
    ("future", "future_timestamp"), ("malformed", "malformed_observations"),
])
def test_invalid_logical_snapshot_never_reaches_candidate_score(variant, reason):
    state, kwargs = live_inputs()
    if variant == "absent":
        state.pop("scoring_observations")
    elif variant == "malformed":
        state["scoring_observations"][0][0]["requests"] = None
    else:
        state["scoring_observations"][0][0]["observed_at"] = 8.0 if variant == "stale" else 11.0
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "unavailable"
    assert any(error["reason"] == reason for error in result["errors"])
    assert "directional_indices" not in result and "cost" not in result


def test_unseen_completion_fields_are_not_copied_or_used():
    state, kwargs = live_inputs()
    original = joint_prediction_state(state, **kwargs)
    for worker in kwargs["execution_delivery"]["workers"]:
        worker.update(future_return_at=20.0, gpu_completed_at=20.0, actual_remaining_tokens=100)
    assert joint_prediction_state(state, **kwargs) == original


def test_false_wave_does_not_erase_a_current_dummy_call_or_prove_idle():
    state, kwargs = live_inputs((0, 0))
    kwargs["execution_delivery"]["workers"][0].update(host_call="dummy", host_call_started_at=9.0)
    result = joint_prediction_state(state, **kwargs)
    assert result["status"] == "available"
    assert result["execution_observed"]["workers"][0]["host_call"] == "dummy"
    assert result["device_activity"] == "unknown"


def test_invalid_candidate_is_not_silently_accepted_as_partial_query():
    state, kwargs = live_inputs()
    original = copy.deepcopy(state)
    move = {"id": "unknown", "model": 1, "source": 0, "destination": 1}
    with pytest.raises(ValueError, match="candidate peer move"):
        joint_prediction_state(state, **kwargs, moves=[move])
    assert state == original


def test_missing_worker_collection_and_invalid_rank_count_are_explicit():
    state, kwargs = live_inputs()
    kwargs["execution_delivery"]["workers"] = None
    assert joint_prediction_state(state, **kwargs)["execution_observed"]["reason"] == "missing_workers"
    state["num_ranks"] = False
    with pytest.raises(ValueError, match="rank count"):
        joint_prediction_state(state, **kwargs)
