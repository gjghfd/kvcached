"""Exact release-only headroom equivalence; no live solver integration."""

import copy
import math
from fractions import Fraction


def _json_key(value, active=None):
    if active is None:
        active = set()
    value_type = type(value)
    if value is None:
        return ("null",)
    if value_type in (bool, int, str):
        return (value_type.__name__, value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("Scenario values must be finite")
        return ("float", value.hex())
    if value_type not in (dict, list):
        raise ValueError("Scenarios require JSON dictionaries, lists and scalar values")
    identity = id(value)
    if identity in active:
        raise ValueError("Scenario structures must not contain cycles")
    active.add(identity)
    try:
        if value_type is list:
            return ("list", tuple(_json_key(item, active) for item in value))
        if any(type(key) is not str for key in value):
            raise ValueError("Scenario dictionary keys must be strings")
        return ("dict", tuple((key, _json_key(value[key], active)) for key in sorted(value)))
    finally:
        active.remove(identity)


def _number(value, name):
    if (type(value) not in (int, float) or value < 0
            or (type(value) is float and not math.isfinite(value))):
        raise ValueError(f"{name} must be a finite nonnegative number, not a boolean")
    return Fraction(value)


def _rank_pair(snapshot, name):
    values = snapshot.get(name)
    if type(values) is not list or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two ranks")
    return [_number(value, name) for value in values]


def _upper_float(value):
    try:
        bound = float(value)
    except OverflowError as error:
        raise ValueError("Usage upper bounds must be finitely representable") from error
    if math.isfinite(bound) and Fraction(bound) < value:
        bound = math.nextafter(bound, math.inf)
    if not math.isfinite(bound):
        raise ValueError("Usage upper bounds must be finitely representable")
    return bound


def _normalize(entry, original_index):
    if type(entry) is not dict or "risk_cost_s" not in entry or "snapshot" not in entry:
        raise ValueError("Each scenario requires a snapshot and risk_cost_s")
    _json_key(entry)
    risk = _number(entry["risk_cost_s"], "risk_cost_s")
    snapshot = entry["snapshot"]
    if type(snapshot) is not dict:
        raise ValueError("Scenario snapshot must be a dictionary")
    if type(snapshot.get("new_request_count")) is not int or snapshot["new_request_count"] != 0:
        raise ValueError("Exact headroom pruning requires explicit release-only new_request_count=0")
    if "new_gib" not in snapshot or _number(snapshot["new_gib"], "new_gib") != 0:
        raise ValueError("Release-only pruning must not omit incoming allocation from the bound")
    capacities = _rank_pair(snapshot, "capacity_gib")
    initial = _rank_pair(snapshot, "initial_gib")
    final = _rank_pair(snapshot, "final_capacity_gib")
    if any(limit > capacity for limit, capacity in zip(final, capacities)):
        raise ValueError("Final capacities must not exceed physical copy capacities")
    if ("placement_risk_cost_s" in snapshot
            and _number(snapshot["placement_risk_cost_s"], "placement_risk_cost_s") != risk):
        raise ValueError("Snapshot placement_risk_cost_s differs from scenario risk_cost_s")
    movable = snapshot.get("movable")
    if type(movable) is not list:
        raise ValueError("Snapshot movable must be a list")
    identifiers = set()
    inbound = [Fraction(0), Fraction(0)]
    inbound_ids = [[], []]
    for request in movable:
        if (type(request) is not dict or type(request.get("id")) is not str or not request["id"]
                or type(request.get("model")) is not int or request["model"] not in (0, 1)
                or type(request.get("source")) is not int or request["source"] not in (0, 1)):
            raise ValueError("Movable requests require unique string IDs and two-rank model/source integers")
        if request["id"] in identifiers:
            raise ValueError("Movable request IDs must be unique")
        identifiers.add(request["id"])
        if "memory_gib" not in request:
            raise ValueError("Every movable request requires memory_gib")
        destination = 1 - request["source"]
        inbound[destination] += _number(request["memory_gib"], "memory_gib")
        inbound_ids[destination].append(request["id"])
    normalized = copy.deepcopy(snapshot)
    rank_proofs = []
    for rank in (0, 1):
        upper = initial[rank] + inbound[rank]
        nonbinding = upper <= final[rank]
        if nonbinding:
            normalized["final_capacity_gib"][rank] = snapshot["capacity_gib"][rank]
        rank_proofs.append({
            "rank": rank, "initial_gib": snapshot["initial_gib"][rank],
            "inbound_movable_ids": inbound_ids[rank], "inbound_memory_upper_gib": _upper_float(inbound[rank]),
            "final_usage_upper_gib": _upper_float(upper),
            "exact_upper": {"numerator": upper.numerator, "denominator": upper.denominator},
            "original_final_capacity_gib": snapshot["final_capacity_gib"][rank],
            "normalized_final_capacity_gib": normalized["final_capacity_gib"][rank],
            "nonbinding": nonbinding, "source_releases_subtracted": False,
        })
    comparable = {name: value for name, value in normalized.items() if name != "placement_risk_cost_s"}
    member = {"original_index": original_index, "risk_cost_s": entry["risk_cost_s"],
              "snapshot_risk_checked": "placement_risk_cost_s" in snapshot, "rank_proofs": rank_proofs}
    return _json_key(comparable), normalized, risk, member


def prune_headroom_scenarios(scenarios):
    """Return detached scenarios and aligned proof groups, in representative input order.

    Each two-rank movable request can relocate once; there is no incoming allocation.
    A final-usage bound includes all inbound movable memory and subtracts no releases.
    Physical copy capacities, requests and uncertainty are never normalized. Equivalent
    feasible snapshots retain minimum constant risk, breaking ties by original index.
    """
    if type(scenarios) is not list:
        raise ValueError("Scenarios must be a list")
    groups = {}
    for original_index, entry in enumerate(scenarios):
        key, normalized, risk, member = _normalize(entry, original_index)
        if key not in groups:
            groups[key] = {"representative_index": original_index, "risk": risk,
                           "snapshot": normalized, "members": []}
        group = groups[key]
        group["members"].append(member)
        if risk < group["risk"]:
            group.update({"representative_index": original_index, "risk": risk, "snapshot": normalized})
    selected, proofs = [], []
    for group in sorted(groups.values(), key=lambda item: item["representative_index"]):
        representative = group["representative_index"]
        entry = copy.deepcopy(scenarios[representative])
        entry["snapshot"] = group["snapshot"]
        selected.append(entry)
        proofs.append({"representative_index": representative,
                       "original_indices": [member["original_index"] for member in group["members"]],
                       "retained_risk_cost_s": entry["risk_cost_s"], "members": group["members"],
                       "all_other_snapshot_parameters_identical": True,
                       "constant_risk_dominance": True})
    return {"scenarios": selected,
            "proof": {"schema_version": 1, "scope": "exact_release_headroom_pruning_not_live_integration",
                      "method": "initial_plus_all_inbound_movable_without_releases_exact_rational_comparison",
                      "input_count": len(scenarios), "output_count": len(selected), "groups": proofs,
                      "ordering": "original_representative_index_with_earliest_equal_risk",
                      "copy_capacities_unchanged": True, "requests_and_uncertainty_unchanged": True}}
