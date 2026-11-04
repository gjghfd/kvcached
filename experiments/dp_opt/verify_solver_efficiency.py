import copy
import json
import math
from fractions import Fraction
from types import FunctionType

import verify_workaware_robust_rebalance


def exact_payload(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def rational(value):
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        raise ValueError("Certificate geometry and risk require finite nonnegative numbers")
    return Fraction(value)


def outward_float(value):
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("Certified upper bound must remain finite")
    return result


def normalize_original(entry, index):
    snapshot = copy.deepcopy(entry["snapshot"])
    risk = rational(entry["risk_cost_s"])
    if (type(snapshot["new_request_count"]) is not int or snapshot["new_request_count"] != 0
            or rational(snapshot["new_gib"]) != 0):
        raise ValueError("Pruning certificates must describe release-only placement")
    pairs = {name: snapshot[name] for name in ("initial_gib", "final_capacity_gib", "capacity_gib")}
    if any(type(pair) is not list or len(pair) != 2 for pair in pairs.values()):
        raise ValueError("Certificates require exactly two-rank geometry")
    for pair in pairs.values():
        for value in pair:
            rational(value)
    if any(rational(limit) > rational(capacity) for limit, capacity in
           zip(snapshot["final_capacity_gib"], snapshot["capacity_gib"])):
        raise ValueError("Certified final limits cannot exceed physical copy capacities")
    if "placement_risk_cost_s" in snapshot and rational(snapshot["placement_risk_cost_s"]) != risk:
        raise ValueError("Original snapshot risk differs from its scenario")
    movable = snapshot["movable"]
    if (type(movable) is not list or any(type(request.get("id")) is not str or not request["id"]
            or type(request.get("source")) is not int or request["source"] not in (0, 1)
            or type(request.get("model")) is not int or request["model"] not in (0, 1) for request in movable)
            or len({request["id"] for request in movable}) != len(movable)):
        raise ValueError("Certificate movable owners must be unique two-rank request identities")
    proofs = []
    for rank in (0, 1):
        inbound = [request for request in movable if request["source"] != rank]
        memory = sum((rational(request["memory_gib"]) for request in inbound), Fraction(0))
        upper = rational(snapshot["initial_gib"][rank]) + memory
        limit = snapshot["final_capacity_gib"][rank]
        nonbinding = upper <= rational(limit)
        if nonbinding:
            snapshot["final_capacity_gib"][rank] = snapshot["capacity_gib"][rank]
        proofs.append({"rank": rank, "initial_gib": snapshot["initial_gib"][rank],
            "inbound_movable_ids": [request["id"] for request in inbound],
            "inbound_memory_upper_gib": outward_float(memory), "final_usage_upper_gib": outward_float(upper),
            "exact_upper": {"numerator": upper.numerator, "denominator": upper.denominator},
            "original_final_capacity_gib": limit,
            "normalized_final_capacity_gib": snapshot["final_capacity_gib"][rank],
            "nonbinding": nonbinding, "source_releases_subtracted": False})
    comparable = {name: value for name, value in snapshot.items() if name != "placement_risk_cost_s"}
    return comparable, {"original_index": index, "risk_cost_s": entry["risk_cost_s"],
                        "snapshot_risk_checked": "placement_risk_cost_s" in snapshot, "rank_proofs": proofs}


def verify_certificates(original, logged):
    if type(original) is not list or type(logged) is not list or not original or not logged:
        raise ValueError("Require all original and retained scenarios")
    normalized = [normalize_original(entry, index) for index, entry in enumerate(original)]
    covered, representatives, scenarios = [], [], []
    for entry in logged:
        proof = entry["headroom_equivalence_proof"]
        indices = proof["original_indices"]
        if (type(indices) is not list or not indices or any(type(index) is not int or index < 0
                or index >= len(original) for index in indices) or indices != sorted(set(indices))
                or entry["original_scenario_count"] != len(original)
                or entry["pruned_scenario_count"] != len(logged)
                or proof["all_other_snapshot_parameters_identical"] is not True
                or proof["constant_risk_dominance"] is not True):
            raise ValueError("Invalid complete equivalence group or scenario accounting")
        representative = min(indices, key=lambda index: (rational(original[index]["risk_cost_s"]), index))
        comparable = normalized[representative][0]
        if any(exact_payload(normalized[index][0]) != exact_payload(comparable) for index in indices):
            raise ValueError("Pruning merged scenarios with different binding constraints or costs")
        expected = {"representative_index": representative, "original_indices": indices,
            "retained_risk_cost_s": original[representative]["risk_cost_s"],
            "members": [normalized[index][1] for index in indices],
            "all_other_snapshot_parameters_identical": True, "constant_risk_dominance": True}
        if exact_payload(proof) != exact_payload(expected):
            raise ValueError("Independent rational usage/risk certificate reconstruction differs")
        risk = original[representative]["risk_cost_s"]
        snapshot = {**comparable, "placement_risk_cost_s": risk}
        if (exact_payload(entry["snapshot"]) != exact_payload(snapshot)
                or exact_payload(entry["guards"]) != exact_payload(original[representative]["guards"])
                or exact_payload(entry["risk_cost_s"]) != exact_payload(risk)):
            raise ValueError("Retained snapshot changed physical capacities, ownership, uncertainty or risk")
        scenarios.append({**original[representative], "snapshot": snapshot})
        covered.extend(indices)
        representatives.append(representative)
    if sorted(covered) != list(range(len(original))) or representatives != sorted(representatives):
        raise ValueError("Every original scenario must appear exactly once in representative order")
    return scenarios


def verify_plan(check, observed, budgets, fit, settings, phase, blocking, *, controller_planning_upper_s=None):
    backend = verify_workaware_robust_rebalance
    try:
        logged = check["plan"]["scenarios"]
        flags = ["headroom_equivalence_proof" in entry for entry in logged]
        if not any(flags):
            return backend.verify_plan(check, observed, budgets, fit, settings, phase, blocking,
                                       controller_planning_upper_s=controller_planning_upper_s)
        if not all(flags):
            raise ValueError("Cannot mix certified and uncertified pruning scenarios")
        original_count = []

        def certified_modes(*args, **kwargs):
            original = backend.headroom_modes(*args, **kwargs)
            original_count.append(len(original))
            return verify_certificates(original, logged)

        namespace = {**backend.__dict__, "headroom_modes": certified_modes}
        replay = FunctionType(backend._verify_plan.__code__, namespace)
        result = replay(check, observed, budgets, fit, settings, phase, blocking, controller_planning_upper_s)
        if len(original_count) != 1:
            raise ValueError("Independent release replay must reconstruct its original scenario bank once")
        return {**result, "exact_headroom_equivalence_verified": True,
                "original_scenarios": original_count[0], "planner_or_solver_invoked": False}
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed exact solver-efficiency certificate: " + str(error)) from error
