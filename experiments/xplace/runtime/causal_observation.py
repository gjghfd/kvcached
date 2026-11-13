import copy
import re

from prediction_state import finite_number, timestamp_error
from scoring_state import candidate_peer_state, capture_scoring_observations


def producer_identity(record):
    identifier = record.get("producer_id")
    pid = record.get("producer_pid")
    if (type(pid) is not int or pid <= 0 or not isinstance(identifier, str)
            or re.fullmatch(r"[0-9a-f]{32}", identifier) is None):
        return None
    return pid, identifier


def causal_joint_state(transport, *, as_of, max_age_s, max_skew_s, clock_id, num_ranks,
                       tensor_parallel_size, moves=(), arrival=None, target=None):
    """Join separate received samples; available never means simultaneous, GPU idle, or predicted."""
    if (not finite_number(as_of) or not finite_number(max_age_s) or max_age_s < 0
            or not finite_number(max_skew_s) or max_skew_s < 0
            or not isinstance(clock_id, str) or not clock_id.strip()
            or any(type(value) is not int or value < 1 for value in (num_ranks, tensor_parallel_size))):
        raise ValueError("Invalid causal query clock, age, skew, or topology")
    result = {"schema": 2, "status": "unavailable", "as_of": as_of, "clock_id": clock_id,
              "num_ranks": num_ranks, "tensor_parallel_size": tensor_parallel_size,
              "max_age_s": max_age_s, "max_skew_s": max_skew_s, "errors": [],
              "device_activity": "unknown", "forecast_status": "unavailable_not_estimated",
              "interpretation": "bounded_skew_pre_action_samples_not_atomic_state_or_horizon"}

    def reject(component, reason, identity=None):
        error = {"component": component, "reason": reason}
        if identity is not None:
            error["identity"] = list(identity)
        result["errors"].append(error)

    if not isinstance(transport, dict):
        reject("transport", "missing_transport")
        return result
    collected_at = transport.get("collected_at")
    if not finite_number(collected_at) or collected_at > as_of:
        reject("transport", "collection_not_available_at_decision")
        return result
    for component, kind in (("scheduler", "live_scheduler_transport_v1"),
                            ("execution", "live_execution_transport_v1")):
        envelope = transport.get(component)
        if not isinstance(envelope, dict) or envelope.get("kind") != kind:
            reject(component, "missing_or_wrong_transport_kind")
        elif envelope.get("clock_id") != clock_id:
            reject(component, "clock_domain_mismatch")
        elif envelope.get("status") != "observed":
            reject(component, "incomplete_transport")
    if result["errors"]:
        return result
    scheduler, execution = transport["scheduler"], transport["execution"]
    execution_collected_at = execution.get("collected_at")
    if not finite_number(execution_collected_at) or execution_collected_at > collected_at:
        reject("execution", "invalid_collection_time")
        return result
    expected_ranks = {(model, rank) for model in (0, 1) for rank in range(num_ranks)}
    expected_workers = {(*rank, lane) for rank in expected_ranks for lane in range(tensor_parallel_size)}
    timestamps, receipts = [], []

    def index_records(records, component, expected, keys):
        if not isinstance(records, list) or len(records) != len(expected):
            reject(component, "incomplete_or_extra_records")
            return {}
        indexed = {}
        for record in records:
            if not isinstance(record, dict):
                reject(component, "malformed_record")
                continue
            identity = tuple(record.get(key) for key in keys)
            if any(type(value) is not int for value in identity) or identity not in expected:
                reject(component, "invalid_identity")
            elif identity in indexed:
                reject(component, "duplicate_identity", identity)
            else:
                indexed[identity] = record
        if set(indexed) != expected:
            reject(component, "missing_identities")
        return indexed

    rank_keys = ("model", "dp_rank")
    scheduler_records = index_records(scheduler.get("ranks"), "scheduler", expected_ranks, rank_keys)
    rank_records = index_records(execution.get("rank_waves"), "rank_waves", expected_ranks, rank_keys)
    worker_records = index_records(execution.get("workers"), "workers", expected_workers, (*rank_keys, "tp_rank"))
    if result["errors"]:
        return result

    def validate_sample(record, component, identity, available_at, ceiling):
        if record.get("clock_id") != clock_id:
            reject(component, "clock_domain_mismatch", identity)
            return False
        reason = timestamp_error(record.get("observed_at"), available_at, as_of, max_age_s)
        if reason:
            reject(component, reason, identity)
            return False
        if available_at > ceiling:
            reject(component, "receipt_after_collection", identity)
            return False
        if producer_identity(record) is None:
            reject(component, "missing_or_invalid_producer", identity)
            return False
        timestamps.append(record["observed_at"])
        receipts.append(available_at)
        return True

    rows = [[None] * num_ranks for _ in (0, 1)]
    scheduler_receipts, waves, workers = [], [], []
    producer_pids, producer_ids = set(), set()
    for component, indexed, role in (("rank_waves", rank_records, "engine"),
                                      ("workers", worker_records, "worker")):
        for identity, record in sorted(indexed.items()):
            if (record.get("status") != "observed" or type(record.get("schema")) is not int
                    or record["schema"] != 1 or record.get("role") != role
                    or record.get("device_activity") != "unknown"):
                reject(component, "invalid_sample_status_or_schema", identity)
                continue
            if not validate_sample(record, component, identity, record.get("available_at"), execution_collected_at):
                continue
            pid, identifier = producer_identity(record)
            if pid in producer_pids or identifier in producer_ids:
                reject(component, "reused_producer_identity", identity)
            producer_pids.add(pid)
            producer_ids.add(identifier)
            keys = (*rank_keys, "observed_at", "available_at", "producer_pid", "producer_id")
            normalized = {key: record[key] for key in keys}
            if role == "engine":
                if (type(record.get("wave_active")) is not bool
                        or record.get("wave_source") != "engine.engines_running"):
                    reject(component, "invalid_rank_wave", identity)
                    continue
                normalized.update(wave_active=record["wave_active"], wave_source=record["wave_source"])
                waves.append(normalized)
            else:
                call = record.get("host_call")
                depth, failures = record.get("host_depth"), record.get("failed_calls")
                started_at = record.get("host_call_started_at")
                if (call not in ("none", "execute", "sample", "dummy")
                        or type(depth) is not int or not 0 <= depth <= 4
                        or type(failures) is not int or failures < 0 or record.get("reason") is not None):
                    reject(component, "invalid_host_state", identity)
                    continue
                if ((call == "none" and (depth != 0 or started_at is not None))
                        or (call != "none" and (depth == 0 or not finite_number(started_at)
                                                 or started_at > record["observed_at"]))):
                    reject(component, "invalid_host_call_start_or_depth", identity)
                    continue
                if failures:
                    reject(component, "host_failures_observed", identity)
                normalized.update(tp_rank=record["tp_rank"], host_call=call, host_depth=depth,
                                  host_call_started_at=started_at, failed_calls=failures)
                workers.append(normalized)
    for identity, record in sorted(scheduler_records.items()):
        snapshot = record.get("snapshot")
        if record.get("status") != "observed" or not isinstance(snapshot, dict):
            reject("scheduler", "unavailable_snapshot", identity)
            continue
        if type(snapshot.get("rank")) is not int or snapshot["rank"] != identity[1]:
            reject("scheduler", "snapshot_rank_mismatch", identity)
            continue
        if not validate_sample(snapshot, "scheduler", identity, record.get("available_at"), collected_at):
            continue
        if producer_identity(snapshot) != producer_identity(rank_records[identity]):
            reject("scheduler", "rank_producer_mismatch", identity)
            continue
        rows[identity[0]][identity[1]] = snapshot
        scheduler_receipts.append({"model": identity[0], "dp_rank": identity[1],
                                   "observed_at": snapshot["observed_at"], "available_at": record["available_at"],
                                   "producer_pid": snapshot["producer_pid"], "producer_id": snapshot["producer_id"]})
    result.update(scheduler_receipts=scheduler_receipts, rank_waves=waves, workers=workers)
    if result["errors"]:
        return result
    skew = max(timestamps) - min(timestamps)
    result["observation_window"] = {"earliest": min(timestamps), "latest": max(timestamps), "skew_s": skew}
    result["availability_window"] = {"earliest_receipt": min(receipts), "latest_receipt": max(receipts),
                                     "execution_collected_at": execution_collected_at, "collected_at": collected_at}
    if skew > max_skew_s:
        reject("joint_clock", "snapshot_skew_exceeded")
        return result
    try:
        projection = capture_scoring_observations(rows)
        state = {"num_ranks": num_ranks, "scoring_observations": projection}
        observed = [candidate_peer_state(state, 1 - model, as_of, max_age_s) for model in (0, 1)]
    except (KeyError, TypeError, AttributeError, ValueError):
        reject("logical", "malformed_scheduler_observations")
        return result
    for model, descriptor in enumerate(observed):
        if descriptor["status"] != "available":
            reject("logical", descriptor["reason"], (model,))
    if result["errors"]:
        return result
    moves = tuple(moves)
    for move in moves:
        if (not isinstance(move, dict) or not isinstance(move.get("id"), str) or not move["id"]
                or any(type(move.get(key)) is not int for key in ("model", "source", "destination"))):
            raise ValueError("Invalid candidate move identity")
    if arrival is not None and not isinstance(arrival, dict):
        raise ValueError("Invalid candidate arrival")
    candidate = [candidate_peer_state(state, 1 - model, as_of, max_age_s, moves, arrival, target)
                 for model in (0, 1)]
    result.update(status="available", logical_observed=observed, logical_candidate=candidate,
                  scoring_observations=projection,
                  directional_indices=[{"model": model, "own_counts": copy.deepcopy(candidate[model]["counts"]),
                      "peer_counts": copy.deepcopy(candidate[1 - model]["counts"]),
                      "own_accounted_contexts": list(candidate[model]["accounted_contexts"]),
                      "peer_accounted_contexts": list(candidate[1 - model]["accounted_contexts"])} for model in (0, 1)])
    return result
