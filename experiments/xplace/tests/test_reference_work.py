import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from reference_work import audit_reference_outcomes, prepare_reference_work, project_reference_layout, token_ledger
from test_causal_observation import transport_input


def reference_input():
    packet, query = transport_input()
    for rank in packet["scheduler"]["ranks"]:
        for request in rank["snapshot"]["requests"]:
            request.update(prompt_tokens=100, computed_tokens=109, output_tokens=10,
                           in_flight_tokens=0, phase="decode")
    return packet, dict(query, additional_output_tokens=4)


def reference_state(**changes):
    packet, query = reference_input()
    query.update(changes)
    reference = prepare_reference_work(packet, **query)
    assert reference["status"] == "available", reference
    return reference


def arrival_input():
    return {"id": "new-request", "model": 0, "prompt_tokens": 1000, "arrived_at": 9.0,
            "available_at": 9.1, "clock_id": "test-single-host-clock"}


def outcome(reference, request, received_at, *, output=None, event="progress"):
    return {"model": request["model"], "id": request["id"], "clock_id": reference["clock_id"],
            "reference_id": reference["reference_id"], "received_at": received_at,
            "observed_at": received_at, "event": event,
            "output_tokens": request["ledger"]["target_output_tokens"] if output is None else output}


@pytest.mark.parametrize("computed,inflight,output,budget,retired,target", [
    (0, 0, 0, 1, 0, 100),
    (60, 20, 0, 4, 40, 103),
    (100, 40, 0, 1, 60, 100),
    (100, 0, 0, 1, 100, 100),
    (100, 0, 1, 4, 100, 104),
    (109, 0, 10, 4, 109, 113),
    (110, 1, 10, 4, 109, 113),
    (20, 0, 10, 4, 20, 113),
    (108, 5, 10, 4, 103, 113),
])
def test_ledger_conserves_disjoint_input_intervals(computed, inflight, output, budget, retired, target):
    result = token_ledger(dict(prompt_tokens=100, computed_tokens=computed, in_flight_tokens=inflight,
                               output_tokens=output), budget)
    assert result["retired_prefix_tokens"] == retired
    assert result["target_input_prefix_tokens"] == target
    assert result["target_output_tokens"] == output + budget
    assert result["input_tokens_until_target"] == inflight + result["unissued_tokens"] == target - retired
    cursor = retired
    for segment in result["segments"]:
        assert segment["start"] == cursor
        assert segment["tokens"] == segment["stop"] - cursor > 0
        cursor = segment["stop"]
    assert cursor == target
    assert sum(segment["tokens"] for segment in result["segments"] if segment["delivery"] == "in_flight") == inflight
    assert sum(segment["tokens"] for segment in result["segments"] if segment["kind"] == "prompt") == max(0, 100 - retired)
    assert result["completion_requires_output_receipt"]


def test_inflight_final_prefill_produces_first_output_without_extra_decode_charge():
    result = token_ledger(dict(prompt_tokens=100, computed_tokens=100, in_flight_tokens=40, output_tokens=0), 4)
    assert result["segments"] == [
        {"delivery": "in_flight", "kind": "prompt", "start": 60, "stop": 100, "tokens": 40},
        {"delivery": "unissued", "kind": "future_decode_input", "start": 100, "stop": 103, "tokens": 3},
    ]


def test_recomputation_replays_known_prefix_without_faking_ten_new_outputs():
    result = token_ledger(dict(prompt_tokens=100, computed_tokens=20, in_flight_tokens=0, output_tokens=10), 4)
    assert [(segment["kind"], segment["tokens"]) for segment in result["segments"]] == [
        ("prompt", 80), ("known_output_input", 10), ("future_decode_input", 3)]
    assert result["target_output_tokens"] == 14


@pytest.mark.parametrize("field,value", [("computed_tokens", True), ("computed_tokens", 111),
                                        ("in_flight_tokens", 110), ("in_flight_tokens", -1),
                                        ("prompt_tokens", 0), ("output_tokens", 1.5)])
def test_unsupported_ledger_is_rejected(field, value):
    request = dict(prompt_tokens=100, computed_tokens=109, in_flight_tokens=0, output_tokens=10)
    request[field] = value
    with pytest.raises(ValueError):
        token_ledger(request, 4)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_reference_budget_requires_explicit_positive_integer(budget):
    packet, query = reference_input()
    query["additional_output_tokens"] = budget
    with pytest.raises(ValueError):
        prepare_reference_work(packet, **query)


def test_same_reference_for_baseline_all_moves_and_arrival_targets():
    reference = reference_state(arrival=arrival_input())
    before = copy.deepcopy(reference)
    plans = [project_reference_layout(reference), project_reference_layout(reference, arrival_target=0),
             project_reference_layout(reference, arrival_target=1),
             project_reference_layout(reference, moves=[dict(model=1, id="conv-0-0", source=0, destination=1)],
                                      arrival_target=0)]
    assert {plan["reference_id"] for plan in plans} == {reference["reference_id"]}
    assert {plan["observation_sha256"] for plan in plans} == {reference["observation_sha256"]}
    assert reference == before
    assert plans[0]["directional_indices"][0]["own"]["external_waiting"] == 1
    assert plans[1]["directional_indices"][0]["own"]["counts"]["prefill"] == [1, 0]
    assert plans[3]["directional_indices"][1]["own"]["counts"]["decode"] == [1, 3]
    assert reference["requests"][0]["ledger"]["input_tokens_until_target"] == 1003
    assert reference["requests"][0]["rank"] is None
    for plan in plans:
        assert plan["forecast_status"] == "unavailable_not_estimated"
        assert not plan["capacity_checked"] and not plan["migration_drain_and_cost_checked"]


def test_no_future_fields_or_candidate_layout_affect_reference():
    packet, query = reference_input()
    expected = prepare_reference_work(packet, **query, arrival=arrival_input())
    for rank in packet["scheduler"]["ranks"]:
        for request in rank["snapshot"]["requests"]:
            request.update(remaining=9999, max_tokens=8888, finished_at=1000, future_phase="idle")
    arrival = dict(arrival_input(), remaining=7777, max_tokens=6666)
    actual = prepare_reference_work(packet, **query, arrival=arrival)
    assert actual == expected
    assert "remaining" not in repr(actual) and "max_tokens" not in repr(actual)
    assert actual["observations"]["workers"][0]["host_call"] == "sample"
    assert not actual["observations"]["rank_waves"][0]["wave_active"]
    assert actual["device_activity"] == "unknown"


def test_directional_indices_are_symmetric_under_tenant_swap():
    packet, query = reference_input()
    original = project_reference_layout(prepare_reference_work(packet, **query))
    for section, field in (("scheduler", "ranks"), ("execution", "rank_waves"), ("execution", "workers")):
        for record in packet[section][field]:
            record["model"] = 1 - record["model"]
    swapped = project_reference_layout(prepare_reference_work(packet, **query))
    for model in (0, 1):
        for side in ("own", "peer"):
            expected = dict(original["directional_indices"][model][side])
            expected["model"] = 1 - expected["model"]
            assert swapped["directional_indices"][1 - model][side] == expected


@pytest.mark.parametrize("mutation", ["stale", "wrong_clock", "missing_worker", "late_receipt", "unsupported_counter"])
def test_reference_cannot_bypass_observation_or_accounting_rejection(mutation):
    packet, query = reference_input()
    if mutation == "stale":
        query["as_of"] = 12.0
    elif mutation == "wrong_clock":
        query["clock_id"] = "other-clock"
    elif mutation == "missing_worker":
        packet["execution"]["workers"].pop()
    elif mutation == "late_receipt":
        packet["scheduler"]["ranks"][0]["available_at"] = 20.0
    else:
        packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0]["in_flight_tokens"] = 200
    reference = prepare_reference_work(packet, **query)
    assert reference["status"] == "unavailable" and reference["errors"]
    assert "reference_id" not in reference
    with pytest.raises(ValueError):
        project_reference_layout(reference)


@pytest.mark.parametrize("mutation", ["work", "source", "budget", "time"])
def test_mutable_reference_tampering_is_detected(mutation):
    reference = reference_state()
    if mutation == "work":
        reference["requests"][0]["ledger"]["target_output_tokens"] += 1
    elif mutation == "source":
        reference["observations"]["workers"][0]["host_call"] = "none"
    elif mutation == "budget":
        reference["additional_output_tokens"] = 3
    else:
        reference["as_of"] = 11.0
    with pytest.raises(ValueError):
        project_reference_layout(reference)
    with pytest.raises(ValueError):
        audit_reference_outcomes(reference, [])


@pytest.mark.parametrize("field,value", [("source", 1), ("destination", 0), ("destination", True),
                                        ("destination", 2), ("model", True), ("id", "absent")])
def test_invalid_move_rejected(field, value):
    move = dict(model=1, id="conv-0-0", source=0, destination=1)
    move[field] = value
    with pytest.raises(ValueError):
        project_reference_layout(reference_state(), moves=[move])


def test_duplicate_move_and_inflight_decode_migration_are_rejected():
    move = dict(model=1, id="conv-0-0", source=0, destination=1)
    with pytest.raises(ValueError):
        project_reference_layout(reference_state(), moves=[move, move])
    packet, query = reference_input()
    packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0].update(computed_tokens=110, in_flight_tokens=1)
    with pytest.raises(ValueError):
        project_reference_layout(prepare_reference_work(packet, **query), moves=[move])


@pytest.mark.parametrize("field,value", [("id", ""), ("model", True), ("prompt_tokens", 0),
                                        ("arrived_at", 11), ("available_at", 11),
                                        ("available_at", 8), ("clock_id", "other-clock")])
def test_arrival_requires_same_clock_and_actual_receipt(field, value):
    arrival = arrival_input()
    arrival[field] = value
    with pytest.raises(ValueError):
        reference_state(arrival=arrival)


def test_duplicate_arrival_rejected_and_empty_cohort_unavailable():
    with pytest.raises(ValueError):
        reference_state(arrival=dict(arrival_input(), model=1, id="conv-0-0"))
    packet, query = reference_input()
    for rank in packet["scheduler"]["ranks"]:
        rank["snapshot"]["requests"] = []
    reference = prepare_reference_work(packet, **query)
    assert reference["status"] == "unavailable"
    assert reference["errors"] == [{"reason": "empty_reference_cohort"}]


def test_outcome_receipts_measure_variable_milestones_not_fixed_wall_time():
    reference = reference_state()
    observations = [outcome(reference, request, 11.0 + index) for index, request in enumerate(reference["requests"])]
    observations.extend(outcome(reference, request, 14.0) for request in reference["requests"][:-1])
    result = audit_reference_outcomes(reference, observations)
    assert result["all_milestones_reached"] and result["prediction_gate_eligible"]
    assert not result["prediction_gate_passed"]
    assert result["observed_request_seconds"] == 8.0
    assert len(result["rows"]) == 4 and result["joint_end_received_at"] == 14.0
    assert reference["forecast_status"] == "unavailable_not_estimated"


@pytest.mark.parametrize("event", ["eos", "max_tokens", "timeout", "aborted", "preempted", "error"])
def test_early_completion_missing_and_failures_are_preserved_not_zero_cost(event):
    reference = reference_state()
    result = audit_reference_outcomes(reference, [outcome(reference, reference["requests"][0], 12.0,
                                                        output=11, event=event)])
    assert len(result["rows"]) == 4
    assert result["rows"][0]["status"] == "censored_" + event
    assert all(row["status"] == "missing" for row in result["rows"][1:])
    assert result["observed_request_seconds"] is None
    assert not result["prediction_gate_eligible"] and not result["prediction_gate_passed"]
    assert result["closed_cohort_violations"][0]["event"] == event


def test_early_milestone_does_not_remove_fast_peer_from_coupled_horizon():
    reference = reference_state()
    requests = reference["requests"]
    samples = [outcome(reference, request, 11.0 + index) for index, request in enumerate(requests)]
    samples.append(outcome(reference, requests[0], 12.0, output=15, event="eos"))
    result = audit_reference_outcomes(reference, samples)
    assert result["all_milestones_reached"] and result["observed_request_seconds"] == 8.0
    assert not result["prediction_gate_eligible"]
    assert result["closed_cohort_violations"] == [dict(model=1, id="conv-0-0", event="eos",
                                                     observed_at=12.0, received_at=12.0)]
    assert result["rows"][0]["last_output_tokens"] == 15


def test_eos_at_last_milestone_is_not_an_earlier_peer_phase_change():
    reference = reference_state()
    samples = [outcome(reference, request, 12.0, event="eos") for request in reference["requests"]]
    result = audit_reference_outcomes(reference, samples)
    assert result["prediction_gate_eligible"] and not result["closed_cohort_violations"]


def test_fast_peer_without_tail_observations_is_not_validated_by_silence():
    reference = reference_state()
    samples = [outcome(reference, request, 11.0 + index) for index, request in enumerate(reference["requests"])]
    result = audit_reference_outcomes(reference, samples)
    assert result["all_milestones_reached"] and not result["prediction_gate_eligible"]
    assert len(result["joint_tail_uncovered"]) == 3
    assert not result["closed_cohort_violations"]


def test_late_delivery_does_not_turn_old_terminal_sample_into_post_horizon_eos():
    reference = reference_state()
    samples = [outcome(reference, request, 11.0 + index) for index, request in enumerate(reference["requests"])]
    terminal = outcome(reference, reference["requests"][0], 15.0, event="eos")
    terminal["observed_at"] = 12.0
    result = audit_reference_outcomes(reference, [*samples, terminal])
    assert result["closed_cohort_violations"][0]["observed_at"] == 12.0
    assert not result["prediction_gate_eligible"]


@pytest.mark.parametrize("field,value", [("reference_id", "other"), ("clock_id", "other"), ("model", True),
                                        ("id", "absent"), ("received_at", 10.0), ("received_at", float("nan")),
                                        ("observed_at", 9.0), ("observed_at", 13.0), ("observed_at", None),
                                        ("output_tokens", True), ("output_tokens", 9), ("event", "reached")])
def test_bad_outcome_data_rejected(field, value):
    reference = reference_state()
    sample = outcome(reference, reference["requests"][0], 12.0)
    sample[field] = value
    with pytest.raises(ValueError):
        audit_reference_outcomes(reference, [sample])


def test_out_of_order_or_post_terminal_events_rejected():
    reference = reference_state()
    request = reference["requests"][0]
    first = outcome(reference, request, 12.0)
    for second in (outcome(reference, request, 11.0), outcome(reference, request, 13.0, output=10)):
        with pytest.raises(ValueError):
            audit_reference_outcomes(reference, [first, second])
    with pytest.raises(ValueError):
        audit_reference_outcomes(reference, [dict(first, event="eos"), outcome(reference, request, 13.0)])
