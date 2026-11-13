import copy
import hashlib
import json

from causal_observation import causal_joint_state
from cost_model import capture_bucket
from prediction_state import finite_number


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def token_ledger(request, additional_output_tokens):
    """Partition scheduler-accounted input work, not elapsed GPU time or remaining output length."""
    keys = ("computed_tokens", "in_flight_tokens", "prompt_tokens", "output_tokens")
    if (not isinstance(request, dict) or type(additional_output_tokens) is not int or additional_output_tokens < 1
            or any(type(request.get(key)) is not int or request[key] < 0 for key in keys)):
        raise ValueError("Invalid reference budget or token counters")
    computed, inflight, prompt, output = (request[key] for key in keys)
    if prompt < 1 or inflight > computed or computed > prompt + output:
        raise ValueError("Unsupported non-speculative token ledger")
    retired = computed - inflight
    target = prompt + output + additional_output_tokens - 1
    segments = []
    for delivery, lower, upper in (("in_flight", retired, computed), ("unissued", computed, target)):
        for kind, start, stop in (("prompt", 0, prompt), ("known_output_input", prompt, prompt + output),
                                  ("future_decode_input", prompt + output, target)):
            begin, end = max(lower, start), min(upper, stop)
            if begin < end:
                segments.append({"delivery": delivery, "kind": kind, "start": begin, "stop": end,
                                 "tokens": end - begin})
    return {"prompt_tokens": prompt, "observed_output_tokens": output, "computed_tokens": computed,
            "retired_prefix_tokens": retired, "in_flight_tokens": inflight,
            "additional_output_tokens": additional_output_tokens, "target_output_tokens": output + additional_output_tokens,
            "target_input_prefix_tokens": target, "unissued_tokens": target - computed,
            "input_tokens_until_target": target - retired,
            "prompt_unretired_tokens": max(0, prompt - retired), "segments": segments,
            "completion_requires_output_receipt": True}


def prepare_reference_work(transport, *, additional_output_tokens, as_of, max_age_s, max_skew_s,
                           clock_id, num_ranks, tensor_parallel_size, arrival=None):
    """Build one pre-action cohort and work budget for all layouts; this is not a predictor."""
    if type(additional_output_tokens) is not int or additional_output_tokens < 1:
        raise ValueError("Reference output budget must be a positive integer")
    observed = causal_joint_state(transport, as_of=as_of, max_age_s=max_age_s, max_skew_s=max_skew_s,
                                  clock_id=clock_id, num_ranks=num_ranks,
                                  tensor_parallel_size=tensor_parallel_size)
    result = {"kind": "causal_reference_work_v1", "status": "unavailable", "as_of": as_of,
              "clock_id": clock_id, "num_ranks": num_ranks, "tensor_parallel_size": tensor_parallel_size,
              "additional_output_tokens": additional_output_tokens, "errors": copy.deepcopy(observed["errors"]),
              "forecast_status": "unavailable_not_estimated", "device_activity": "unknown",
              "objective": "sum_controller_received_additional_output_milestone_seconds",
              "continuation": "closed_cohort_no_eos_all_residents_continue_until_last_milestone",
              "accounting_scope": "non_speculative_contiguous_decoder_prefix_only",
              "deployment_status": "unvalidated_no_runtime_capability_binding"}
    if observed["status"] != "available":
        return result
    requests = []
    for model, descriptor in enumerate(observed["logical_observed"]):
        for request in descriptor["requests"]:
            try:
                ledger = token_ledger(request, additional_output_tokens)
            except ValueError as error:
                result["errors"].append({"model": model, "id": request["id"], "reason": str(error)})
                continue
            requests.append({"model": model, "id": request["id"], "rank": request["rank"],
                             "phase": request["phase"], "external_arrival": False, "ledger": ledger})
    if arrival is not None:
        if (not isinstance(arrival, dict) or type(arrival.get("model")) is not int or arrival["model"] not in (0, 1)
                or not isinstance(arrival.get("id"), str) or not arrival["id"]
                or type(arrival.get("prompt_tokens")) is not int or arrival["prompt_tokens"] < 1
                or arrival.get("clock_id") != clock_id
                or not finite_number(arrival.get("arrived_at")) or not finite_number(arrival.get("available_at"))
                or not arrival["arrived_at"] <= arrival["available_at"] <= as_of):
            raise ValueError("Arrival requires causal controller receipt and identity")
        if any((request["model"], request["id"]) == (arrival["model"], arrival["id"]) for request in requests):
            raise ValueError("Arrival is already in the resident cohort")
        ledger = token_ledger({"computed_tokens": 0, "in_flight_tokens": 0, "prompt_tokens": arrival["prompt_tokens"],
                               "output_tokens": 0}, additional_output_tokens)
        requests.append({"model": arrival["model"], "id": arrival["id"], "rank": None, "phase": "waiting",
                         "external_arrival": True, "arrived_at": arrival["arrived_at"],
                         "available_at": arrival["available_at"], "ledger": ledger})
    if not requests:
        result["errors"].append({"reason": "empty_reference_cohort"})
    if result["errors"]:
        return result
    result.update(status="available", observations=observed, observation_sha256=fingerprint(observed),
                  requests=sorted(requests, key=lambda request: (request["model"], request["id"])))
    result["reference_id"] = fingerprint(result)
    return result


def validate_reference(reference):
    if not isinstance(reference, dict) or reference.get("status") != "available":
        raise ValueError("Reference work is unavailable")
    content = {key: value for key, value in reference.items() if key != "reference_id"}
    if (reference.get("kind") != "causal_reference_work_v1" or fingerprint(content) != reference.get("reference_id")
            or fingerprint(reference.get("observations")) != reference.get("observation_sha256")):
        raise ValueError("Reference work or bound observations changed")


def project_reference_layout(reference, *, moves=(), arrival_target=None):
    """Change assignments, never the common work target or observed in-flight host state."""
    validate_reference(reference)
    requests = {(request["model"], request["id"]): request for request in reference["requests"]}
    assignments = {identity: request["rank"] for identity, request in requests.items()}
    moved = set()
    for move in moves:
        if (not isinstance(move, dict) or not isinstance(move.get("id"), str)
                or any(type(move.get(key)) is not int for key in ("model", "source", "destination"))):
            raise ValueError("Invalid move identity")
        identity = move["model"], move["id"]
        request = requests.get(identity)
        if (request is None or identity in moved or move["source"] != request["rank"]
                or not 0 <= move["destination"] < reference["num_ranks"] or move["destination"] == move["source"]
                or request["external_arrival"] or request["phase"] != "decode" or request["ledger"]["in_flight_tokens"]):
            raise ValueError("Move must reference an observed non-inflight decode resident")
        assignments[identity] = move["destination"]
        moved.add(identity)
    arrivals = [identity for identity, request in requests.items() if request["external_arrival"]]
    if arrival_target is not None:
        if type(arrival_target) is not int or not 0 <= arrival_target < reference["num_ranks"] or len(arrivals) != 1:
            raise ValueError("Invalid arrival target")
        assignments[arrivals[0]] = arrival_target
    layouts, summaries = [], []
    for model in (0, 1):
        counts = {phase: [0] * reference["num_ranks"] for phase in ("waiting", "prefill", "decode")}
        contexts = [0] * reference["num_ranks"]
        external_waiting = 0
        for identity, request in sorted(requests.items()):
            if identity[0] != model:
                continue
            rank = assignments[identity]
            phase = ("prefill" if rank is not None else "waiting") if request["external_arrival"] else request["phase"]
            layouts.append({"model": model, "id": identity[1], "rank": rank, "phase": phase})
            if rank is None:
                external_waiting += 1
            else:
                counts[phase][rank] += 1
                if phase != "waiting":
                    contexts[rank] += request["ledger"]["computed_tokens"]
        summaries.append({"model": model, "counts": counts, "accounted_contexts": contexts,
                          "decode_bucket": capture_bucket(max(counts["decode"])), "external_waiting": external_waiting})
    return {"reference_id": reference["reference_id"], "observation_sha256": reference["observation_sha256"],
            "assignments": layouts, "directional_indices": [{"model": model, "own": copy.deepcopy(summaries[model]),
                "peer": copy.deepcopy(summaries[1 - model])} for model in (0, 1)],
            "forecast_status": "unavailable_not_estimated", "device_activity": "unknown",
            "capacity_checked": False, "migration_drain_and_cost_checked": False}


def audit_reference_outcomes(reference, observations):
    """Offline outcome audit only; censored/failed rows never become predictor input or disappear."""
    validate_reference(reference)
    requests = {(request["model"], request["id"]): request for request in reference["requests"]}
    rows = {identity: {"model": identity[0], "id": identity[1], "status": "missing", "first_reached_at": None,
                       "last_received_at": None, "last_observed_at": None,
                       "last_output_tokens": request["ledger"]["observed_output_tokens"],
                       "terminal": None, "samples": 0} for identity, request in requests.items()}
    terminal_kinds = ("eos", "max_tokens", "timeout", "aborted", "preempted", "error")
    for sample in observations:
        if (not isinstance(sample, dict) or type(sample.get("model")) is not int
                or not isinstance(sample.get("id"), str)):
            raise ValueError("Invalid outcome identity")
        identity = sample["model"], sample["id"]
        if (identity not in requests or sample.get("reference_id") != reference["reference_id"]
                or sample.get("clock_id") != reference["clock_id"]):
            raise ValueError("Outcome does not belong to this reference")
        row, request = rows[identity], requests[identity]
        received_at, output, event = sample.get("received_at"), sample.get("output_tokens"), sample.get("event")
        observed_at = sample.get("observed_at")
        previous_time = row["last_received_at"] if row["last_received_at"] is not None else reference["as_of"]
        previous_observation = row["last_observed_at"] if row["last_observed_at"] is not None else reference["as_of"]
        if (not finite_number(received_at) or received_at < previous_time or type(output) is not int
                or not finite_number(observed_at) or not previous_observation <= observed_at <= received_at
                or output < row["last_output_tokens"] or event not in ("progress", *terminal_kinds)
                or row["terminal"] is not None):
            raise ValueError("Invalid, out-of-order, or post-terminal outcome")
        row.update(last_received_at=received_at, last_observed_at=observed_at,
                   last_output_tokens=output, samples=row["samples"] + 1)
        if output >= request["ledger"]["target_output_tokens"] and row["first_reached_at"] is None:
            row["first_reached_at"] = received_at
        if event in terminal_kinds:
            row["terminal"] = {"event": event, "observed_at": observed_at, "received_at": received_at}
        row["status"] = ("reached" if row["first_reached_at"] is not None else
                         "censored_" + event if event in terminal_kinds else "incomplete")
    complete = all(row["first_reached_at"] is not None for row in rows.values())
    joint_end = max(row["first_reached_at"] for row in rows.values()) if complete else None
    violations, uncovered = [], []
    for row in rows.values():
        terminal = row["terminal"]
        if terminal and (joint_end is None or terminal["observed_at"] < joint_end
                         or terminal["event"] in ("preempted", "aborted", "error")):
            violations.append({"model": row["model"], "id": row["id"], **terminal})
        if joint_end is None or row["last_observed_at"] is None or row["last_observed_at"] < joint_end:
            uncovered.append({"model": row["model"], "id": row["id"]})
    value = sum(row["first_reached_at"] - reference["as_of"] for row in rows.values()) if complete else None
    return {"reference_id": reference["reference_id"], "kind": "offline_reference_outcome_audit_v1",
            "rows": [rows[identity] for identity in sorted(rows)], "all_milestones_reached": complete,
            "joint_end_received_at": joint_end, "observed_request_seconds": value,
            "closed_cohort_violations": violations, "joint_tail_uncovered": uncovered,
            "prediction_gate_eligible": complete and not violations and not uncovered,
            "prediction_gate_passed": False, "forecast_status": "unavailable_not_estimated"}
