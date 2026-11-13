import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, default_dp2_params, integrated_cost
from planner import DEFAULT_PARAMS, build_state, plan_admission, plan_return
from scoring_state import candidate_peer_state, request_phase
from test_planner import make_observed


def observed_state(counts=(2, 2)):
    observed = make_observed(list(counts), num_ranks=2)
    for tenant in observed:
        for rank in tenant:
            rank.update(observed_at=10.0, phase_schema=1, scheduler_paused=False)
            for request in rank["requests"]:
                request.update(phase="decode", in_flight_tokens=0)
    return build_state(observed, [60.0] * 2, request_growth={}), observed


@pytest.mark.parametrize("computed,prompt,output,running,expected", [
    (40, 100, 0, False, "waiting"), (40, 100, 0, True, "prefill"),
    (100, 100, 0, True, "prefill"), (100, 100, 1, True, "decode"),
    (110, 100, 10, False, "waiting"), (50, 100, 10, True, "prefill"),
])
def test_phase_uses_visible_progress_not_move_eligibility(computed, prompt, output, running, expected):
    assert request_phase(computed, prompt, output, running) == expected


def test_state_copies_only_causal_fields_and_never_replayed_output_length():
    state, observed = observed_state()
    request = observed[1][0]["requests"][0]
    request.update(max_tokens=90000, remaining=12345)
    state = build_state(observed, [60.0] * 2)
    captured = state["scoring_observations"][1][0]["requests"][0]
    assert captured["prompt_tokens"] == request["prompt_tokens"]
    assert captured["output_tokens"] == request["output_tokens"]
    assert not {"max_tokens", "remaining", "reservation_tokens"} & captured.keys()
    request["phase"] = "waiting"
    observed[1][0]["observed_at"] = 200
    assert captured["phase"] == "decode"
    assert state["scoring_observations"][1][0]["observed_at"] == 10


def test_candidate_moves_and_arrival_preserve_source_and_do_not_invent_decode():
    state, _ = observed_state()
    original = copy.deepcopy(state)
    moves = [{"id": "conv-0-0", "model": 1, "source": 0, "destination": 1}]
    descriptor = candidate_peer_state(state, 0, 10.5, 1.0, moves,
                                      {"model": 1, "prompt_tokens": 101167, "max_tokens": 777777}, 0)
    assert descriptor["status"] == "available" and descriptor["phase"] == "mixed"
    assert descriptor["counts"] == {"waiting": [0, 0], "prefill": [1, 0], "decode": [1, 3]}
    assert descriptor["decode_bucket"] == 8
    arrival = descriptor["requests"][-1]
    assert arrival["hypothetical"] and arrival["computed_tokens"] == arrival["output_tokens"] == 0
    assert arrival["prompt_tokens"] == 101167 and "max_tokens" not in arrival
    assert state == original


def test_peer_state_is_tenant_symmetric_and_empty_snapshot_is_explicit():
    state, observed = observed_state()
    original = candidate_peer_state(state, 0, 10.5, 1.0)
    swapped = build_state(list(reversed(observed)), [60.0] * 2)
    mirrored = candidate_peer_state(swapped, 1, 10.5, 1.0)
    assert {key: value for key, value in original.items() if key != "model"} == {
        key: value for key, value in mirrored.items() if key != "model"}
    solo = candidate_peer_state(state, 1, 10.5, 1.0)
    assert solo["status"] == "available" and solo["phase"] == "idle" and solo["decode_bucket"] == 0


@pytest.mark.parametrize("field,value,reason", [
    ("observed_at", None, "missing_or_invalid_timestamp"),
    ("observed_at", float("nan"), "missing_or_invalid_timestamp"),
    ("observed_at", 11, "future_timestamp"), ("observed_at", 8, "stale_timestamp"),
    ("phase_schema", None, "missing_phase_schema_or_pause_state"),
    ("scheduler_paused", None, "missing_phase_schema_or_pause_state"),
    ("rank", 1, "missing_or_invalid_rank"),
])
def test_unknown_or_stale_peer_is_not_solo_or_a_prediction(field, value, reason):
    state, _ = observed_state()
    state["scoring_observations"][1][0][field] = value
    result = candidate_peer_state(state, 0, 10.5, 1.0)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert "phase" not in result and "counts" not in result


def test_missing_legacy_phase_and_inconsistent_phase_are_rejected():
    legacy = build_state(make_observed([1, 0], num_ranks=2), [60.0] * 2)
    assert candidate_peer_state(legacy, 0, 10.5, 1.0)["status"] == "unavailable"
    state, _ = observed_state()
    request = state["scoring_observations"][1][0]["requests"][0]
    request["phase"] = None
    assert candidate_peer_state(state, 0, 10.5, 1.0)["reason"] == "missing_or_invalid_phase"
    request.update(phase="decode", computed_tokens=0)
    assert candidate_peer_state(state, 0, 10.5, 1.0)["reason"] == "inconsistent_phase"


def test_waiting_inflight_and_paused_do_not_change_phase_or_claim_gpu_completion():
    state, _ = observed_state()
    rank = state["scoring_observations"][1][0]
    rank["scheduler_paused"] = True
    rank["requests"][0]["phase"] = "waiting"
    rank["requests"][1]["in_flight_tokens"] = 1
    result = candidate_peer_state(state, 0, 10.5, 1.0)
    assert result["counts"]["waiting"] == [1, 0] and result["counts"]["decode"] == [1, 2]
    assert result["scheduler_paused"] == [True, False]
    assert result["interpretation"] == "instantaneous_post_action_not_horizon_prediction"
    assert result["accounted_contexts"][0] == rank["requests"][1]["computed_tokens"]


@pytest.mark.parametrize("changes", [{"destination": 2}, {"source": 1}, {"id": "unknown"}])
def test_invalid_move_never_silently_changes_candidate(changes):
    state, _ = observed_state()
    move = {"id": "conv-0-0", "model": 1, "source": 0, "destination": 1, **changes}
    with pytest.raises(ValueError, match="candidate peer move"):
        candidate_peer_state(state, 0, 10.5, 1.0, [move])


def test_integrated_cost_forwards_explicit_bucket_without_reinterpreting_it():
    params = default_dp2_params()
    params["interference"]["0"]["decode"] = {"1": 0.01, "8": 0.05, "16": 0.3}
    model = StepModel(params)
    requests = [(0, 4000, 100)]
    solo = integrated_cost(model, 0, requests, 2, 10)
    bucket8 = integrated_cost(model, 0, requests, 2, 10, other_active="decode", other_bucket=8)
    default1 = integrated_cost(model, 0, requests, 2, 10, other_active="decode")
    assert bucket8 - solo == pytest.approx(0.5)
    assert default1 - solo == pytest.approx(0.1)


@pytest.mark.parametrize("as_of,max_age", [(float("nan"), 1), (10, -1), (10, float("inf")),
                                          (True, 1), (10, True)])
def test_invalid_clock_or_bound_is_not_accepted(as_of, max_age):
    state, _ = observed_state()
    with pytest.raises(ValueError, match="freshness bound"):
        candidate_peer_state(state, 0, as_of, max_age)


def test_freshness_boundary_and_phase_transition_use_only_selected_snapshot():
    state, observed = observed_state((1, 0))
    assert candidate_peer_state(state, 0, 11, 1)["status"] == "available"
    assert candidate_peer_state(state, 0, 11.001, 1)["reason"] == "stale_timestamp"
    observed[1][0]["requests"][0].update(phase="prefill", computed_tokens=40, output_tokens=0)
    changed = build_state(observed, [60.0] * 2)
    assert candidate_peer_state(changed, 0, 10.5, 1)["phase"] == "prefill"
    assert candidate_peer_state(state, 0, 10.5, 1)["phase"] == "decode"


def test_missing_token_counts_or_duplicate_ids_reject_descriptor():
    state, _ = observed_state()
    rank = state["scoring_observations"][1][0]
    rank["requests"][0]["in_flight_tokens"] = None
    assert candidate_peer_state(state, 0, 10.5, 1)["reason"] == "missing_or_invalid_token_count"
    rank["requests"][0]["in_flight_tokens"] = 0
    rank["requests"][1]["id"] = rank["requests"][0]["id"]
    assert candidate_peer_state(state, 0, 10.5, 1)["reason"] == "duplicate_request_id"


def test_metadata_does_not_silently_enable_unvalidated_horizon_scoring():
    state, _ = observed_state((10, 2))
    legacy = copy.deepcopy(state)
    legacy.pop("scoring_observations")
    arrival = {"model": 0, "tokens": 101167, "prompt_tokens": 101167}
    for operation in (lambda current: plan_return(current, StepModel(default_dp2_params()), DEFAULT_PARAMS),
                      lambda current: plan_admission(current, arrival, "ca", StepModel(default_dp2_params()),
                                                     DEFAULT_PARAMS)):
        assert operation(state) == operation(legacy)
