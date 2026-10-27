import copy
import math

from ilp_reference import replay_candidate
from live_controller import admission_planning_snapshot
from workaware_cost import headroom_modes, reference_horizon_s, remaining_segments
from workaware_rebalance import WorkawareInferenceLedger, workaware_gate


def same_replayed_cost(before, after):
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(same_replayed_cost(before[name], after[name]) for name in before)
    if isinstance(before, list):
        return len(before) == len(after) and all(same_replayed_cost(left, right) for left, right in zip(before, after))
    if type(before) is float:
        return (math.isfinite(before) and math.isfinite(after)
                and abs(before - after) <= 4 * max(math.ulp(before), math.ulp(after)))
    return before == after


def verify_plan(check, observed, budgets, fit, settings, phase, blocking):
    calibration = settings["workaware_fit"]
    segments = remaining_segments(observed, calibration, settings["work_interval_steps"])
    snapshot, _, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, True, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], None, new_request_count=0)
    snapshot["performance_cost_tables_s"] = [[0] * len(table) for table in snapshot["performance_cost_tables_s"]]
    snapshot["work_segments"] = segments
    snapshot["migration_activation_cost_s"] = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    cold_costs = []
    profiles = settings["workaware_profile"]["models"]
    for model, profile in enumerate(profiles):
        observation = observed[model][0]
        size = math.ceil(profile["tokens"] / observation["block_size"]) * observation["block_bytes"] / 1024**3
        prior = (population + 1) * (settings["pause_fixed_s"] + size / settings["move_bandwidth_gib_s"])
        costs = profile["cold_cost_samples_s"]
        cold_costs.append(max(prior, sum(costs) / len(costs)) if costs else prior)
    horizon = reference_horizon_s(segments, calibration)
    scenarios = headroom_modes(snapshot, observed, profiles, horizon, cold_costs)
    if (check["unconstrained_snapshot"] != snapshot or check["work_segments"] != segments
            or check["active_population"] != population or check["forecast_horizon_s"] != horizon
            or len(check["plan"]["scenarios"]) != len(scenarios)):
        raise ValueError("Workaware forecast or soft-headroom scenarios differ from current requests/verified costs")
    references, candidates, snapshots = [], [], []
    for logged, scenario in zip(check["plan"]["scenarios"], scenarios):
        expected = {**scenario["snapshot"], "placement_risk_cost_s": scenario["risk_cost_s"]}
        if (not same_replayed_cost(logged["snapshot"], expected)
                or not same_replayed_cost(logged["guards"], scenario["guards"])
                or not same_replayed_cost(logged["risk_cost_s"], scenario["risk_cost_s"])):
            raise ValueError("Workaware protection/risk differs from past arrival profile and physical caps")
        snapshots.append(expected)
        if all(used <= cap + 1e-6 for used, cap in zip(expected["initial_gib"], expected["final_capacity_gib"])):
            references.append(replay_candidate(expected, {"target": None, "moves": [], "status": "unchanged_layout"}))
        candidate = logged["plan"]["winner"]
        if candidate is not None:
            replayed = replay_candidate(expected, candidate)
            if ("objective_s" not in replayed or len(candidate["moves"]) > settings["max_moves"]
                    or not math.isclose(candidate["objective_s"], replayed["objective_s"], abs_tol=1e-8)):
                raise ValueError("Workaware scenario candidate does not replay with work/risk/copy costs")
            candidates.append(replayed)
    reference = min(references, key=lambda value: value["objective_s"])
    if not same_replayed_cost(reference["objective_s"], check["reference"]["objective_s"]) or check["reference"]["moves"]:
        raise ValueError("Workaware unchanged-layout reference includes wrong headroom risk or moves")
    winner = check["winner"]
    if not any(same_replayed_cost(check["snapshot"], candidate) for candidate in [*snapshots, snapshot]):
        raise ValueError("Workaware winner is not an enumerated protected/unprotected placement")
    replayed = replay_candidate(check["snapshot"], winner)
    for name in ("objective_s", "performance_cost_s", "migration_cost_s", "migration_activation_cost_s",
                 "final_counts", "final_gib", "peak_gib", "work_cost"):
        if not same_replayed_cost(winner.get(name), replayed.get(name)):
            raise ValueError("Workaware winner replay mismatch: " + name)
    best = min([reference["objective_s"], *(candidate["objective_s"] for candidate in candidates)])
    if not math.isclose(winner["objective_s"], best, abs_tol=1e-8):
        raise ValueError("Workaware winner does not minimize the enumerated feasible incumbents")
    if "population_profiles" in calibration:
        from workaware_profile_cost import changed_segment_uncertainty

        uncertainty = changed_segment_uncertainty(segments, winner["moves"])
    else:
        uncertainty = sum(segment["uncertainty_request_s"] for segment in segments)
    gate = workaware_gate(reference, winner, settings, uncertainty, blocking, phase)
    if not same_replayed_cost(gate, check["gate"]):
        raise ValueError("Workaware gate omits uncertainty/real blocking or incorrectly reapplies past credit")


def verify_history(decisions, requests, arrivals, config, origin, marks):
    ledger = WorkawareInferenceLedger(config["workaware_rate_priors"], origin,
        config["move_fixed_s"] if config.get("rebalance_pause_fixed_s") is None else config["rebalance_pause_fixed_s"],
        config.get("rebalance_headroom_quantile", .95), config.get("rebalance_headroom_history_limit", 64))
    events = []
    if len({arrival["request_id"] for arrival in arrivals}) != len(arrivals) or len(arrivals) != len(marks):
        raise ValueError("Workaware online arrival journal is incomplete or duplicated")
    for arrival in arrivals:
        mark = marks[arrival["request_id"]]
        if (arrival["model"] != mark["model"] or arrival["total_tokens"] != mark["prompt_tokens"] + mark["output_tokens"]
                or not math.isclose(arrival["arrived_at"], origin + mark["arrival_s"], abs_tol=1e-8)):
            raise ValueError("Workaware arrival size/rate history differs from actual offered requests")
        events.append((arrival["observed_at"], "arrival", arrival))
    for decision in decisions:
        if decision["path"] == "release_rebalance_ilp":
            events.append((decision["workaware_settled_at"], "settle", decision))
        else:
            events.append((decision["workaware_ledger_observed_at"], "admission", decision))
    for request in requests.values():
        events.append((request["workaware_completion_observed_at"], "completion", request))
    events.sort(key=lambda event: event[0])
    cursor = 0
    for decision in sorted((decision for decision in decisions if "balance_evidence" in decision),
                           key=lambda value: value["balance_evidence"]["sampled_at"]):
        sampled = decision["balance_evidence"]["sampled_at"]
        while cursor < len(events) and events[cursor][0] <= sampled:
            observed_at, kind, event = events[cursor]
            if not math.isfinite(observed_at) or observed_at < origin:
                raise ValueError("Workaware history has invalid or pre-origin control/completion evidence")
            if kind == "arrival":
                ledger.arrival(marks[event["request_id"]], event["arrived_at"], observed_at)
            else:
                getattr(ledger, kind)(copy.deepcopy(event), observed_at)
            cursor += 1
        expected = ledger.evidence(decision["observations"], sampled)
        if expected != decision["balance_evidence"]:
            raise ValueError("Workaware rate, cold-cost, credit or epoch history reads future or unobserved events")
