import copy
import math

from scoring_state import candidate_peer_state


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp_error(observed_at, available_at, as_of, max_age_s):
    if not finite_number(observed_at) or not finite_number(available_at):
        return "missing_or_invalid_timestamp"
    if available_at < observed_at:
        return "availability_before_observation"
    if available_at > as_of:
        return "not_available_at_decision"
    if as_of - observed_at > max_age_s:
        return "stale_observation"
    return None


def delivery_error(delivery, kind, clock_id):
    if not isinstance(delivery, dict):
        return "missing_delivery"
    if delivery.get("kind") != kind:
        return "not_live_delivery"
    if delivery.get("clock_id") != clock_id:
        return "clock_domain_mismatch"
    return None


def logical_delivery_state(delivery, observed, as_of, max_age_s, clock_id):
    reason = delivery_error(delivery, "live_scheduler_snapshot", clock_id)
    if reason:
        return {"status": "unavailable", "reason": reason}
    available = delivery.get("available_at")
    if (not isinstance(available, list) or len(available) != 2
            or any(not isinstance(row, list) or len(row) != len(observed[model]["observed_at"])
                   for model, row in enumerate(available))):
        return {"status": "unavailable", "reason": "incomplete_rank_delivery"}
    for model, descriptor in enumerate(observed):
        for rank, timestamp in enumerate(descriptor["observed_at"]):
            reason = timestamp_error(timestamp, available[model][rank], as_of, max_age_s)
            if reason:
                return {"status": "unavailable", "reason": reason, "model": model, "dp_rank": rank}
    return {"status": "available", "available_at": copy.deepcopy(available)}


def execution_delivery_state(delivery, num_ranks, tensor_parallel_size, as_of, max_age_s, clock_id):
    reason = delivery_error(delivery, "live_execution_snapshot", clock_id)
    if reason:
        return {"status": "unavailable", "reason": reason}
    records = delivery.get("workers")
    if not isinstance(records, list):
        return {"status": "unavailable", "reason": "missing_workers"}
    expected = {(model, rank, lane) for model in (0, 1) for rank in range(num_ranks)
                for lane in range(tensor_parallel_size)}
    workers, identities, waves = [], set(), {}
    for record in records:
        if not isinstance(record, dict):
            return {"status": "unavailable", "reason": "malformed_worker"}
        identity = tuple(record.get(key) for key in ("model", "dp_rank", "tp_rank"))
        if any(type(value) is not int for value in identity) or identity not in expected:
            return {"status": "unavailable", "reason": "invalid_worker_identity"}
        if identity in identities:
            return {"status": "unavailable", "reason": "duplicate_worker"}
        identities.add(identity)
        reason = timestamp_error(record.get("observed_at"), record.get("available_at"), as_of, max_age_s)
        if reason:
            return {"status": "unavailable", "reason": reason, "identity": list(identity)}
        if type(record.get("wave_active")) is not bool:
            return {"status": "unavailable", "reason": "missing_wave_state"}
        host_call = record.get("host_call")
        if host_call not in ("none", "execute", "sample", "dummy"):
            return {"status": "unavailable", "reason": "missing_host_call_state"}
        started_at = record.get("host_call_started_at")
        if ((host_call == "none" and started_at is not None)
                or (host_call != "none" and (not finite_number(started_at)
                                              or started_at > record["observed_at"]))):
            return {"status": "unavailable", "reason": "invalid_host_call_start"}
        rank_key = identity[:2]
        if rank_key in waves and waves[rank_key] != record["wave_active"]:
            return {"status": "unavailable", "reason": "inconsistent_rank_wave"}
        waves[rank_key] = record["wave_active"]
        workers.append({key: record[key] for key in ("model", "dp_rank", "tp_rank", "observed_at",
                        "available_at", "wave_active", "host_call")})
        workers[-1]["host_call_started_at"] = started_at
    if identities != expected:
        return {"status": "unavailable", "reason": "incomplete_worker_lanes"}
    workers.sort(key=lambda worker: (worker["model"], worker["dp_rank"], worker["tp_rank"]))
    return {"status": "available", "workers": workers, "interpretation": "observed_pre_action_host_state",
            "device_activity": "unknown"}


def joint_prediction_state(state, *, as_of, max_age_s, max_skew_s, clock_id, tensor_parallel_size,
                           logical_delivery=None, execution_delivery=None, moves=(), arrival=None, target=None):
    """Validate instantaneous causal inputs; never infer a horizon, GPU idle, or a cost."""
    if (not finite_number(as_of) or not finite_number(max_age_s) or max_age_s < 0
            or not finite_number(max_skew_s) or max_skew_s < 0
            or not isinstance(clock_id, str) or not clock_id.strip()
            or type(tensor_parallel_size) is not int or tensor_parallel_size < 1):
        raise ValueError("Invalid joint-state clock, freshness, skew, or topology")
    num_ranks = state.get("num_ranks")
    if type(num_ranks) is not int or num_ranks < 1:
        raise ValueError("Invalid joint-state rank count")
    result = {"schema": 1, "as_of": as_of, "max_age_s": max_age_s, "max_skew_s": max_skew_s,
              "clock_id": clock_id, "num_ranks": num_ranks, "tensor_parallel_size": tensor_parallel_size,
              "status": "unavailable", "errors": [], "device_activity": "unknown",
              "forecast_status": "unavailable_not_estimated",
              "interpretation": "instantaneous_joint_causal_state_not_horizon_prediction"}
    try:
        observed = [candidate_peer_state(state, 1 - model, as_of, max_age_s) for model in (0, 1)]
    except (KeyError, TypeError):
        result["errors"].append({"component": "logical", "reason": "malformed_observations"})
        return result
    result["logical_observed"] = observed
    for model, descriptor in enumerate(observed):
        if descriptor["status"] != "available":
            result["errors"].append({"component": "logical", "model": model, "reason": descriptor["reason"]})
    if result["errors"]:
        return result
    moves = tuple(moves)
    candidate = [candidate_peer_state(state, 1 - model, as_of, max_age_s, moves, arrival, target)
                 for model in (0, 1)]
    result["logical_candidate"] = candidate
    result["directional_indices"] = [{"model": model, "own_counts": copy.deepcopy(candidate[model]["counts"]),
        "peer_counts": copy.deepcopy(candidate[1 - model]["counts"]),
        "own_accounted_contexts": list(candidate[model]["accounted_contexts"]),
        "peer_accounted_contexts": list(candidate[1 - model]["accounted_contexts"])} for model in (0, 1)]
    result["logical_delivery"] = logical_delivery_state(logical_delivery, observed, as_of, max_age_s, clock_id)
    result["execution_observed"] = execution_delivery_state(
        execution_delivery, num_ranks, tensor_parallel_size, as_of, max_age_s, clock_id)
    timestamps = [timestamp for descriptor in observed for timestamp in descriptor["observed_at"]]
    for component in ("logical_delivery", "execution_observed"):
        if result[component]["status"] != "available":
            result["errors"].append({"component": component, "reason": result[component]["reason"]})
    if result["execution_observed"]["status"] == "available":
        timestamps.extend(worker["observed_at"] for worker in result["execution_observed"]["workers"])
    result["snapshot_skew_s"] = max(timestamps) - min(timestamps)
    if result["snapshot_skew_s"] > max_skew_s:
        result["errors"].append({"component": "joint_clock", "reason": "snapshot_skew_exceeded"})
    if not result["errors"]:
        result["status"] = "available"
    return result
