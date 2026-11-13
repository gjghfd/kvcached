import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from accounting_capability import prepare_bound_reference_work
from coupled_duration import FEATURES, PHASES, ResponseSurface, receipt_time, simulate_coupled_duration
from reference_work import fingerprint, project_reference_layout
from scoring_state import request_phase
from test_accounting_capability import bound_input, native_sources
from test_reference_work import arrival_input


def make_reference(specifications, budget=1, arrival=None):
    packet, query = bound_input()
    template = copy.deepcopy(packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0])
    for record in packet["scheduler"]["ranks"]:
        requests = []
        for spec in specifications:
            if (spec["model"], spec.get("rank", 0)) != (record["model"], record["dp_rank"]):
                continue
            prompt, output, computed = spec.get("prompt", 4), spec.get("output", 1), spec.get("computed", 4)
            request = dict(template, id=spec["id"], prompt_tokens=prompt, output_tokens=output,
                           computed_tokens=computed, in_flight_tokens=spec.get("inflight", 0),
                           phase=request_phase(computed, prompt, output, True))
            requests.append(request)
        record["snapshot"]["requests"] = requests
    reference = prepare_bound_reference_work(packet, **dict(query, additional_output_tokens=budget, arrival=arrival))
    assert reference["status"] == "available", reference
    return reference


def initial_state(reference):
    return dict(reference_id=reference["reference_id"], observation_sha256=reference["observation_sha256"],
                as_of=reference["as_of"], source_id="synthetic-explicit-asof-condition-not-observation",
                model="single_batch_barrier_v1", requests=[dict(model=row["model"], id=row["id"],
                    prefix_tokens=row["ledger"]["retired_prefix_tokens"],
                    output_tokens=row["ledger"]["observed_output_tokens"]) for row in reference["requests"]],
                batches=[], pending_receipts=[])


def parameters(durations=None):
    durations = durations or {}
    rows = {}
    for model in (0, 1):
        for own in PHASES:
            for peer in PHASES:
                value = durations.get((model, own, peer), durations.get((model, own), durations.get(own, 1.0)))
                rows[f"{model}:{own}:{peer}"] = dict(intercept_s=value,
                    coefficients={name: 0.0 for name in FEATURES}, support={name: [0, 1000000] for name in FEATURES})
    return dict(kind="coupled_service_response_v1", source_id="synthetic-arithmetic-only", rows=rows)


def run(reference, *, initial=None, params=None, action=None, **changes):
    action = action or {}
    layout = project_reference_layout(reference, moves=[{name: move[name] for name in
        ("model", "id", "source", "destination")} for move in action.get("moves", [])],
        arrival_target=action.get("arrival_target"))
    query = dict(action=action, receipt=dict(period_s=0.25, offset_s=0.0, delay_s=0.0),
                 limits=[dict(max_tokens_per_rank=128, max_requests_per_rank=16, chunk_tokens=4) for _ in (0, 1)],
                 max_time_s=100.0, max_events=1000)
    query.update(changes)
    return simulate_coupled_duration(reference, layout, initial if initial is not None else initial_state(reference),
                                    ResponseSurface(params or parameters()), **query)


def complete(result):
    assert result["status"] == "complete_conditional_rollout", result
    assert result["objective_s"] is not None and not result["prediction_gate_passed"]
    assert result["deployment_status"] == "local_conditional_kernel_not_validated"
    return {(row["model"], row["id"]): row for row in result["outcomes"]}


def rejected(result, reason):
    assert result["status"] == "unavailable", result
    assert result["objective_s"] is None and not result["prediction_gate_passed"]
    assert any(reason in error for error in result["errors"]), result
    assert all("milestone_s" in row for row in result["outcomes"])


def test_prompt_first_output_and_decode_use_shared_numeric_clock(native_sources):
    reference = make_reference([dict(model=0, id="new", output=0, computed=0)], budget=2)
    result = run(reference, params=parameters(dict(prefill=2, decode=2)))
    rows = complete(result)
    assert rows[0, "new"]["milestone_s"] == result["objective_s"] == 6.0
    returned = [event for event in result["events"] if event["event"] == "input_return"]
    assert [(event["prefix_tokens"], event["output_tokens"]) for event in returned] == [(4, 1), (5, 2)]


def test_recompute_does_not_turn_known_tokens_into_new_outputs(native_sources):
    reference = make_reference([dict(model=0, id="replay", output=3, computed=0)], budget=2)
    result = run(reference)
    rows = complete(result)
    returned = [event for event in result["events"] if event["event"] == "input_return"]
    assert [(event["prefix_tokens"], event["output_tokens"]) for event in returned] == [(4, 3), (7, 4), (8, 5)]
    assert rows[0, "replay"]["milestone_s"] == 6.0


def test_peer_phase_change_reprices_remaining_fraction_not_entire_batch(native_sources):
    reference = make_reference([dict(model=0, id="prefill", computed=0, output=0), dict(model=1, id="decode")])
    params = parameters({(0, "prefill"): 2, (0, "decode"): 8, (1, "decode", "prefill"): 8,
                         (1, "decode", "return"): 2, (1, "decode", "decode"): 2})
    result = run(reference, params=params)
    rows = complete(result)
    assert rows[0, "prefill"]["milestone_s"] == 3.0
    assert rows[1, "decode"]["milestone_s"] == 4.5
    assert result["objective_s"] == 7.5
    assert any(event["event"] == "stage_complete" and event["model"] == 1
               and event["stage"] == "compute" and event["time_s"] == 3.5 for event in result["events"])


def test_fast_peer_keeps_computing_after_its_own_milestone(native_sources):
    reference = make_reference([dict(model=0, id="fast"), dict(model=1, id="slow")])
    result = run(reference, params=parameters({(0, "decode"): 0.5, "return": 0.25, (1, "decode"): 4.0}))
    rows = complete(result)
    assert rows[0, "fast"]["milestone_s"] == 0.75
    assert rows[0, "fast"]["output_tokens"] > 2
    assert rows[1, "slow"]["milestone_s"] == 4.25
    assert all(event["descriptors"][0]["phase"] != "empty" for event in result["events"]
               if event["event"] == "service_interval")


def test_simultaneous_events_are_processed_once_for_both_tenants(native_sources):
    reference = make_reference([dict(model=model, id="same") for model in (0, 1)], budget=3)
    result = run(reference)
    rows = complete(result)
    assert rows[0, "same"]["milestone_s"] == rows[1, "same"]["milestone_s"] == 6.0
    assert result["objective_s"] == 12.0


@pytest.mark.parametrize("stage,fraction,expected", [("compute", 0.25, 1.5), ("return", 0.5, 0.5)])
def test_inflight_residual_not_recharged_or_derived_from_host_flags(native_sources, stage, fraction, expected):
    reference = make_reference([dict(model=0, id="flight", computed=5, inflight=1)])
    initial = initial_state(reference)
    initial["batches"] = [dict(model=0, stage=stage, remaining_fraction=fraction,
                               allocations=[dict(id="flight", tokens=1)])]
    result = run(reference, initial=initial, params=parameters(dict(decode=2)))
    assert complete(result)[0, "flight"]["milestone_s"] == expected
    assert next(event for event in result["events"] if event["event"] == "input_return")["prefix_tokens"] == 5


def test_zero_remaining_input_still_waits_for_pending_sample(native_sources):
    reference = make_reference([dict(model=0, id="pending", computed=4, output=0)])
    rejected(run(reference), "missing_pending_sample_return")
    initial = initial_state(reference)
    initial["batches"] = [dict(model=0, stage="return", remaining_fraction=0.5,
                               allocations=[dict(id="pending", tokens=0)])]
    assert complete(run(reference, initial=initial))[0, "pending"]["milestone_s"] == 0.5


def test_sample_age_predicted_progress_keeps_original_target_and_pending_receipt(native_sources):
    reference = make_reference([dict(model=0, id="aged")], budget=2)
    initial = initial_state(reference)
    initial["requests"][0].update(prefix_tokens=6, output_tokens=3)
    rejected(run(reference, initial=initial), "missing_sample_age_output_receipts")
    initial["pending_receipts"] = [dict(model=0, id="aged", output_tokens=3, after_s=0.5)]
    result = run(reference, initial=initial)
    assert complete(result)[0, "aged"]["milestone_s"] == 0.5
    assert result["reference_id"] == reference["reference_id"]


def test_rank_barrier_uses_max_rank_features_and_explicit_batch_limits(native_sources):
    reference = make_reference([dict(model=0, id="first", rank=0), dict(model=0, id="second", rank=0),
                                dict(model=0, id="third", rank=1)])
    params = parameters()
    params["rows"]["0:decode:empty"]["coefficients"]["batch"] = 2
    result = run(reference, params=params)
    rows = complete(result)
    assert {row["milestone_s"] for row in rows.values()} == {6.0}
    first = next(event for event in result["events"] if event["event"] == "service_interval")
    assert first["descriptors"][0]["rank_counts"] == [2, 1]


def test_migration_drains_residual_before_transfer_and_changes_rank_afterwards(native_sources):
    reference = make_reference([dict(model=0, id="move")])
    initial = initial_state(reference)
    initial["batches"] = [dict(model=0, stage="residual", allocations=[], remaining_fraction=0.5)]
    action = dict(moves=[dict(model=0, id="move", source=0, destination=1, bytes=4096)])
    result = run(reference, initial=initial, action=action)
    row = complete(result)[0, "move"]
    assert row["rank"] == 1 and row["milestone_s"] == 4.5
    stages = [event["stage"] for event in result["events"] if event["event"] == "stage_complete"]
    assert stages == ["residual", "drain", "transfer", "compute", "return"]


def test_pending_batch_finishes_before_migration_without_replaying_its_tokens(native_sources):
    reference = make_reference([dict(model=0, id="move")], budget=2)
    initial = initial_state(reference)
    initial["batches"] = [dict(model=0, stage="compute", remaining_fraction=0.5,
                               allocations=[dict(id="move", tokens=1)])]
    action = dict(moves=[dict(model=0, id="move", source=0, destination=1, bytes=128)])
    result = run(reference, initial=initial, action=action)
    assert complete(result)[0, "move"]["milestone_s"] == 5.5
    assert [event["prefix_tokens"] for event in result["events"] if event["event"] == "input_return"] == [5, 6]


def test_queued_arrival_remains_in_both_objectives_with_different_admission_forecasts(native_sources):
    arrival = dict(arrival_input(), prompt_tokens=4)
    reference = make_reference([dict(model=1, id="resident")], arrival=arrival)
    initial = initial_state(reference)
    baseline = dict(admissions=[dict(model=0, id=arrival["id"], rank=0, after_s=3)])
    candidate = dict(arrival_target=0, admissions=[dict(model=0, id=arrival["id"], rank=0, after_s=0)])
    base, proposed = run(reference, initial=initial, action=baseline), run(reference, initial=initial, action=candidate)
    assert complete(base)[0, arrival["id"]]["milestone_s"] == 5.0
    assert complete(proposed)[0, arrival["id"]]["milestone_s"] == 2.0
    assert base["reference_id"] == proposed["reference_id"]
    assert base["initialization_sha256"] == proposed["initialization_sha256"]
    assert base["response_sha256"] == proposed["response_sha256"]
    assert base["protocol_sha256"] == proposed["protocol_sha256"]


def test_no_admission_forecast_is_not_zero_cost_or_successful_subset(native_sources):
    reference = make_reference([dict(model=1, id="resident")], arrival=dict(arrival_input(), prompt_tokens=4))
    result = run(reference, max_time_s=5)
    assert result["status"] == "time_limit_censored" and result["objective_s"] is None
    assert len(result["outcomes"]) == 2
    assert any(row["milestone_s"] is not None for row in result["outcomes"])
    assert any(row["milestone_s"] is None for row in result["outcomes"])


def test_empty_unadmitted_cohort_stalls_explicitly(native_sources):
    reference = make_reference([], arrival=arrival_input())
    result = run(reference)
    assert result["status"] == "stalled_unserved_cohort" and result["objective_s"] is None


@pytest.mark.parametrize("field,value", [("max_time_s", 0.5), ("max_events", 1)])
def test_bounded_rollout_preserves_incomplete_outcomes(native_sources, field, value):
    result = run(make_reference([dict(model=0, id="bounded")]), **{field: value})
    assert result["status"].endswith("limit_censored") and result["objective_s"] is None
    assert result["outcomes"][0]["milestone_s"] is None


def test_target_on_last_allowed_event_and_time_boundary_is_not_false_censor(native_sources):
    reference = make_reference([dict(model=0, id="boundary")])
    assert complete(run(reference, max_time_s=2, max_events=2))[0, "boundary"]["milestone_s"] == 2


@pytest.mark.parametrize("field,value", [("source_id", None), ("as_of", 0), ("reference_id", "other"),
                                        ("observation_sha256", "other"), ("model", "multi_batch")])
def test_missing_residual_or_wrong_source_never_becomes_zero_delay(native_sources, field, value):
    reference = make_reference([dict(model=0, id="missing")])
    initial = initial_state(reference)
    initial[field] = value
    rejected(run(reference, initial=initial), "missing_or_unbound_asof_residual_estimate")


@pytest.mark.parametrize("field,value", [("prefix_tokens", -1), ("prefix_tokens", True),
                                        ("output_tokens", 0), ("output_tokens", 1.5)])
def test_invalid_estimated_progress_fails_with_every_cohort_row(native_sources, field, value):
    reference = make_reference([dict(model=model, id="request") for model in (0, 1)])
    initial = initial_state(reference)
    initial["requests"][0][field] = value
    result = run(reference, initial=initial)
    assert result["status"] == "unavailable" and len(result["outcomes"]) == 2


def test_missing_inflight_and_multi_batch_are_not_inferred_from_wave(native_sources):
    reference = make_reference([dict(model=0, id="flight", computed=5, inflight=1)])
    rejected(run(reference), "missing_inflight_residual")
    initial = initial_state(reference)
    batch = dict(model=0, stage="compute", remaining_fraction=1, allocations=[dict(id="flight", tokens=1)])
    initial["batches"] = [batch, copy.deepcopy(batch)]
    rejected(run(reference, initial=initial), "unsupported_multiple_inflight_batches")


@pytest.mark.parametrize("mutation,reason", [("missing_pair", "uncovered_phase_pair"),
    ("support", "outside_response_support"), ("coefficient", "invalid_tokens"),
    ("nan", "invalid_intercept"), ("zero", "invalid_service_duration")])
def test_response_never_falls_back_to_failed_constant(native_sources, mutation, reason):
    reference = make_reference([dict(model=0, id="coverage")])
    params = parameters()
    row = params["rows"]["0:decode:empty"]
    if mutation == "missing_pair":
        del params["rows"]["0:decode:empty"]
    elif mutation == "support":
        row["support"]["context"] = [0, 1]
    elif mutation == "coefficient":
        del row["coefficients"]["tokens"]
    elif mutation == "nan":
        row["intercept_s"] = None
    else:
        row["intercept_s"] = 0
    rejected(run(reference, params=params), reason)


@pytest.mark.parametrize("completed,period,offset,delay,expected", [
    (2, 1, 0, 0, 2), (2.125, 0.5, 0, 0.25, 2.75), (0, 1, 0.5, 0, 0.5), (1, 1, 0, 2, 3)])
def test_controller_receipt_cadence_not_device_completion(completed, period, offset, delay, expected):
    assert receipt_time(completed, dict(period_s=period, offset_s=offset, delay_s=delay)) == expected


def test_receipt_delay_changes_objective_not_gpu_progress(native_sources):
    reference = make_reference([dict(model=0, id="receipt")])
    result = run(reference, receipt=dict(period_s=1, offset_s=0, delay_s=2))
    assert complete(result)[0, "receipt"]["milestone_s"] == 4
    assert next(event for event in result["events"] if event["event"] == "input_return")["time_s"] == 2


def test_unbound_reference_cannot_be_used_as_duration_entrypoint(native_sources):
    reference = make_reference([dict(model=0, id="unbound")])
    reference["deployment_status"] = "unvalidated_no_runtime_capability_binding"
    reference["reference_id"] = fingerprint({key: value for key, value in reference.items() if key != "reference_id"})
    rejected(run(reference), "unbound_accounting_reference")


def test_model_and_rank_permutation_preserve_symmetric_objective(native_sources):
    specs = [dict(model=0, id="prefill", computed=0, output=0, rank=0), dict(model=1, id="decode", rank=1)]
    original = run(make_reference(specs), params=parameters(dict(prefill=3, decode=1)))
    swapped = run(make_reference([dict(spec, model=1 - spec["model"], rank=1 - spec["rank"]) for spec in specs]),
                  params=parameters(dict(prefill=3, decode=1)))
    complete(original)
    complete(swapped)
    assert original["objective_s"] == swapped["objective_s"]
    assert {row["id"]: row["milestone_s"] for row in original["outcomes"]} == {
        row["id"]: row["milestone_s"] for row in swapped["outcomes"]}


def test_inputs_and_capacity_reference_are_unchanged(native_sources):
    reference = make_reference([dict(model=0, id="immutable")])
    initial, params = initial_state(reference), parameters()
    saved = copy.deepcopy((reference, initial, params))
    complete(run(reference, initial=initial, params=params))
    assert (reference, initial, params) == saved


def test_mutated_response_parameters_cannot_reuse_an_old_hash(native_sources):
    reference = make_reference([dict(model=0, id="hash")])
    response = ResponseSurface(parameters())
    response.parameters["rows"]["0:decode:empty"]["intercept_s"] = 99
    result = simulate_coupled_duration(reference, project_reference_layout(reference), initial_state(reference),
        response, action={}, receipt=dict(period_s=1, offset_s=0, delay_s=0),
        limits=[dict(max_tokens_per_rank=128, max_requests_per_rank=16, chunk_tokens=4) for _ in (0, 1)],
        max_time_s=100, max_events=100)
    rejected(result, "response_parameters_changed")


def test_unmodeled_waiting_resident_is_not_silently_activated(native_sources):
    packet, query = bound_input()
    packet["scheduler"]["ranks"][2]["snapshot"]["requests"][0]["phase"] = "waiting"
    reference = prepare_bound_reference_work(packet, **query)
    assert reference["status"] == "available"
    rejected(run(reference), "unsupported_resident_waiting_admission")


def test_directional_peer_batch_coefficient_is_not_own_batch(native_sources):
    reference = make_reference([dict(model=0, id="one"), dict(model=1, id="two"), dict(model=1, id="three")])
    params = parameters()
    for peer in PHASES:
        params["rows"][f"0:decode:{peer}"]["coefficients"]["peer_batch"] = 2
    params["rows"]["1:decode:decode"]["intercept_s"] = 20
    result = run(reference, params=params)
    complete(result)
    first = next(event for event in result["events"] if event["event"] == "service_interval")
    assert first["full_service_s"] == [5.0, 20.0]
    assert first["descriptors"][0]["batch"] == 1 and first["descriptors"][1]["batch"] == 2


@pytest.mark.parametrize("scale", [0.1, 0.3, 1.0, 3.7])
def test_clock_scale_covariance_and_long_fractional_intervals(native_sources, scale):
    reference = make_reference([dict(model=0, id="first"), dict(model=1, id="second")], budget=9)
    params = parameters({(0, "decode"): 0.7 * scale, (1, "decode"): 1.3 * scale, "return": 0.2 * scale})
    result = run(reference, params=params, receipt=dict(period_s=0.25 * scale, offset_s=0, delay_s=0))
    rows = complete(result)
    for row in rows.values():
        assert row["prefix_tokens"] == 4 + row["output_tokens"] - 1
    assert result["objective_s"] / scale == pytest.approx(21.75)
    assert all(event["until_s"] > event["time_s"] for event in result["events"]
               if event["event"] == "service_interval")


def test_unsupported_transfer_preserves_partial_diagnostics_not_free_migration(native_sources):
    reference = make_reference([dict(model=0, id="move")])
    action = dict(moves=[dict(model=0, id="move", source=0, destination=1, bytes=1024)])
    params = parameters()
    del params["rows"]["0:transfer:empty"]
    result = run(reference, params=params, action=action)
    rejected(result, "uncovered_phase_pair:0:transfer:empty")
    assert result["outcomes"][0]["rank"] == 0
    assert result["events"][-1]["stage"] == "drain"


@pytest.mark.parametrize("completed,expected", [(13.5, 13.5), (math.nextafter(13.5, math.inf), 13.5),
    (13.500001, 13.75), (math.nextafter(13.5, -math.inf), 13.5)])
def test_ulp_boundary_does_not_add_a_period_or_deliver_before_completion(completed, expected):
    received = receipt_time(completed, dict(period_s=0.25, offset_s=0, delay_s=0))
    assert received >= completed
    assert received == pytest.approx(expected)


def test_boundary_stabilization_keeps_positive_receipt_delay():
    completed = math.nextafter(13.5, math.inf)
    assert receipt_time(completed, dict(period_s=0.25, offset_s=0, delay_s=0.5)) == completed + 0.5
