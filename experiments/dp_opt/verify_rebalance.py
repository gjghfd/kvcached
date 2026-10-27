import math

from ilp_reference import replay_candidate
from live_controller import admission_planning_snapshot
from verify_benefit import verify_ownership


def verify_rebalance(decision, budgets, fit, lookup=None, config=None, requests=None):
    if decision.get("new_request_count") != 0 or decision.get("admitted") or decision.get("submitted"):
        raise ValueError("Release rebalance must not create or admit a request")
    settings = decision["settings"]
    workaware = settings.get("policy_mode") == "workaware"
    headroom = settings.get("policy_mode") == "headroom"
    amortized = settings.get("policy_mode") in ("amortized", "headroom", "workaware")
    if config is not None:
        expected = {"max_moves": config.get("rebalance_max_moves", 4),
                    "candidate_limit": config.get("rebalance_candidate_limit", 8),
                    "solver_limit_s": config["solver_limit_s"],
                    "move_bandwidth_gib_s": config["move_bandwidth_gib_s"],
                    "pause_fixed_s": (config["move_fixed_s"] if config.get("rebalance_pause_fixed_s") is None
                                      else config["rebalance_pause_fixed_s"]),
                    "min_gain_s": config.get("rebalance_min_gain_s", 1)}
        if amortized:
            evidence = decision["balance_evidence"]
            if evidence["pause_fixed_s"] < expected["pause_fixed_s"]:
                raise ValueError("Adaptive pause bound fell below its frozen prior")
            expected.update({"pause_fixed_s": evidence["pause_fixed_s"], "policy_mode": "amortized",
                             "overhead_fraction": config.get("rebalance_overhead_fraction", .01),
                             "credit_request_s": evidence["credit_request_s"]})
            if headroom:
                expected.update({"policy_mode": "headroom", "headroom_profile": evidence["headroom_profile"]})
            if workaware:
                expected.update({"policy_mode": "workaware", "headroom_profile": evidence["headroom_profile"],
                    "workaware_profile": evidence["workaware_profile"], "workaware_fit": config["workaware_fit_data"],
                    "work_interval_steps": 128})
        if settings != expected:
            raise ValueError("Release rebalance differs from frozen limits and costs")
    if requests is not None:
        causes = decision.get("release_events", [])
        if not causes or len({cause["request_id"] for cause in causes}) != len(causes):
            raise ValueError("Release rebalance lacks unique completed-request triggers")
        for cause in causes:
            request = requests.get(cause["request_id"])
            if (request is None or request["model"] != cause["model"]
                    or request["completed_at"] != cause["completed_at"]
                    or cause["completed_at"] > decision["started_at"]):
                raise ValueError("Release rebalance ran before its completion trigger")
        if amortized:
            evidence = decision["balance_evidence"]
            sampled = evidence["sampled_at"]
            identifiers = evidence["completed_ids"]
            if (len(identifiers) != len(set(identifiers)) or any(identifier not in requests
                    or requests[identifier]["completed_at"] > sampled for identifier in identifiers)):
                raise ValueError("Migration credit includes duplicate or future completions")
            gross = sum(requests[identifier]["completed_at"] - requests[identifier]["reserved_at"]
                        for identifier in identifiers)
            for ranks in decision["observations"]:
                for rank in ranks:
                    for request in rank["requests"]:
                        if request["id"] not in identifiers:
                            if request["reserved_at"] > sampled:
                                raise ValueError("Migration credit includes future reservation exposure")
                            gross += sampled - request["reserved_at"]
            spent = evidence["spent_blocking_request_s"]
            service = max(0, gross - spent)
            credit = evidence["overhead_fraction"] * service - spent
            for field, calculated in (("gross_service_request_s", gross),
                    ("inference_service_request_s", service), ("unclamped_credit_request_s", credit),
                    ("credit_request_s", max(0, credit))):
                if not math.isclose(evidence[field], calculated, abs_tol=1e-6):
                    raise ValueError("Migration credit does not match past measured inference exposure")
    for field in ("observations", "current_before_pause", "after_boundary", "frozen_observations", "after_rebalance"):
        if decision.get(field) is not None:
            verify_ownership(decision[field], budgets)
    if decision.get("pause_call_s", 0) > 0:
        receipts = decision.get("pause_receipts")
        if receipts is None or len(receipts) != 2 or any(len(ranks) != 2 for ranks in receipts):
            raise ValueError("Release rebalance lacks four DP-boundary receipts")
        for ranks in receipts:
            for rank, receipt in enumerate(ranks):
                step = receipt["boundary_step_counter"]
                times = [receipt[name] for name in ("armed_at", "reached_at", "callback_at")]
                if (receipt["rank"] != rank or type(step) is not int or step <= 0 or step % 32
                        or any(not math.isfinite(value) for value in times) or times != sorted(times)):
                    raise ValueError("Release rebalance did not use default32 consensus boundaries")
    for field, inventory_field in (("before_pause", "observations"), ("after_pause", "frozen_observations")):
        if field not in decision:
            continue
        check, observed = decision[field], decision[inventory_field]
        if fit is None:
            raise ValueError("Release rebalance lacks archived performance calibration")
        snapshot, horizon, population = admission_planning_snapshot(
            observed, 0, 0, 0, budgets, fit, True, "ilp", settings["candidate_limit"],
            0, settings["move_bandwidth_gib_s"], lookup, new_request_count=0)
        phase = "before_pause" if field == "before_pause" else "after_pause"
        pre_solve_blocking = 0
        if field == "after_pause":
            pre_solve_blocking = sum(len(observed[model][rank]["requests"]) * max(0,
                decision["blocking_sampled_at"] - receipt["reached_at"])
                for model, ranks in enumerate(decision["pause_receipts"]) for rank, receipt in enumerate(ranks))
        if workaware:
            from verify_workaware_rebalance import verify_plan

            blocking = pre_solve_blocking + (population * decision["solve_s"] if phase == "after_pause" else 0)
            verify_plan(check, observed, budgets, fit, settings, phase, blocking)
            continue
        if amortized:
            snapshot["migration_activation_cost_s"] = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
            snapshot["migration_budget_s"] = max(0, settings["credit_request_s"] - pre_solve_blocking)
        expected_snapshots = [snapshot]
        if headroom:
            from headroom_rebalance import headroom_scenarios

            guards, scenarios = headroom_scenarios(snapshot, observed, settings["headroom_profile"])
            if (check["unconstrained_snapshot"] != snapshot or check["headroom_guards"] != guards
                    or len(check["plan"]["scenarios"]) != len(scenarios)):
                raise ValueError("Headroom guard does not match past lengths and physical geometry")
            scores = [replay_candidate(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})["objective_s"]]
            for logged, scenario in zip(check["plan"]["scenarios"], scenarios):
                if logged["snapshot"] != scenario["snapshot"] or logged["guards"] != scenario["guards"]:
                    raise ValueError("Headroom ILP changed an enumerated safe-rank constraint")
                candidate = logged["plan"]["winner"]
                if candidate is not None:
                    replayed_candidate = replay_candidate(scenario["snapshot"], candidate)
                    if ("objective_s" not in replayed_candidate or not math.isclose(
                            candidate["objective_s"], replayed_candidate["objective_s"], abs_tol=1e-8)):
                        raise ValueError("Headroom candidate violates copy peak, credit or reserved capacity")
                    scores.append(candidate["objective_s"])
                expected_snapshots.append(scenario["snapshot"])
            if not math.isclose(check["winner"]["objective_s"], min(scores), abs_tol=1e-8):
                raise ValueError("Headroom planning discarded a less costly feasible candidate")
        if check["snapshot"] not in expected_snapshots or check["horizon_steps"] != horizon or check["active_population"] != population:
            raise ValueError("Release rebalance snapshot differs from physical inventory or remaining decode")
        winner = check["winner"]
        replayed = replay_candidate(check["snapshot"], winner)
        reference = replay_candidate(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
        if check["reference"] != reference:
            raise ValueError("Release rebalance reference must be the unchanged population and placement")
        if len(winner["moves"]) > settings["max_moves"] or "objective_s" not in replayed:
            raise ValueError("Release rebalance winner violates move or copy-peak limits")
        if headroom and any(not any(winner["final_gib"][rank] + guard["demand_gib"][rank]
                <= budgets[rank] + 1e-6 for rank in (0, 1)) for guard in guards):
            raise ValueError("Headroom winner destroyed a previously safe future admission")
        for name in ("objective_s", "performance_cost_s", "migration_cost_s", "admission_cost_s", "final_gib", "peak_gib", "final_counts"):
            if winner[name] != replayed[name]:
                raise ValueError("Release rebalance ILP replay mismatch: " + name)
        if amortized and winner["migration_activation_cost_s"] != replayed["migration_activation_cost_s"]:
            raise ValueError("Migration activation cost was not charged once per transaction")
        gate = check["gate"]
        gain = reference["performance_cost_s"] - replayed["objective_s"]
        fixed = population * settings["pause_fixed_s"]
        blocking = gate["measured_blocking_so_far_request_s"]
        expected_blocking = 0
        if field == "after_pause":
            receipts = decision["pause_receipts"]
            expected_blocking = sum(len(observed[model][rank]["requests"]) * max(0,
                decision["blocking_sampled_at"] - receipt["reached_at"])
                for model, ranks in enumerate(receipts) for rank, receipt in enumerate(ranks)) + population * decision["solve_s"]
        if amortized:
            from amortized_rebalance import amortized_gate

            expected_gate = amortized_gate(reference, winner, settings, population, expected_blocking, phase)
            if gate != expected_gate:
                raise ValueError("Amortized gate differs from net gain, credit or sunk-cost accounting")
            continue
        if (not math.isfinite(blocking) or blocking < 0
                or not math.isclose(blocking, expected_blocking, abs_tol=1e-8)
                or not math.isclose(gate["predicted_gain_after_copy_request_s"], gain, abs_tol=1e-8)
                or gate["fixed_control_request_s"] != fixed or gate["min_gain_request_s"] != settings["min_gain_s"]
                or gate["accepted"] != (bool(winner["moves"]) and gain > fixed + blocking + settings["min_gain_s"])):
            raise ValueError("Release rebalance profit gate is inconsistent")
    if not decision["moves"]:
        if decision.get("reason") == "rebalanced":
            raise ValueError("Successful release rebalance lacks migration evidence")
        return
    fresh = decision.get("after_pause")
    if fresh is None or not fresh["gate"]["accepted"] or decision.get("reason") != "rebalanced":
        raise ValueError("Migration lacks a profitable fresh post-boundary ILP solution")
    if [move["request_id"] for move in decision["moves"]] != fresh["winner"]["moves"]:
        raise ValueError("Release rebalance migrations differ from the fresh ordered plan")
    before, after = decision["frozen_observations"], decision.get("after_rebalance")
    if after is None:
        raise ValueError("Release rebalance lacks final physical ownership evidence")
    if workaware:
        used = [sum(after[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
        if any(value > cap + 1e-6 for value, cap in zip(used, fresh["snapshot"]["final_capacity_gib"])):
            raise ValueError("Actual post-copy physical pages violated selected soft-headroom protection")
    if headroom:
        used = [sum(after[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3 for rank in (0, 1)]
        if any(not any(used[rank] + guard["demand_gib"][rank] <= budgets[rank] + 1e-6
                for rank in (0, 1)) for guard in fresh["headroom_guards"]):
            raise ValueError("Actual post-copy physical pages destroyed protected admission headroom")
    inventory = lambda observations: {request["id"]: (model, request["prompt_tokens"], request["max_tokens"],
        request["output_tokens"], request.get("computed_tokens"), len(request["blocks"]))
        for model, ranks in enumerate(observations) for rank in ranks for request in rank["requests"]}
    if inventory(before) != inventory(after):
        raise ValueError("Release rebalance changed request or decode state")
    locations = lambda observations: {request["id"]: (model, rank)
        for model, ranks in enumerate(observations) for rank, observation in enumerate(ranks) for request in observation["requests"]}
    expected_locations = locations(before)
    original_requests = {request["id"]: (observation, request)
        for ranks in before for observation in ranks for request in observation["requests"]}
    for move in decision["moves"]:
        if (expected_locations[move["request_id"]] != (move["model"], move["source"])
                or move["destination"] != 1 - move["source"] or move.get("purpose") != "release_rebalance"):
            raise ValueError("Release rebalance moved the wrong tenant or request")
        observation, request = original_requests[move["request_id"]]
        if (move["kv_bytes"] != len(request["blocks"]) * observation["block_bytes"]
                or len(move["sha256"]) != 64
                or any(used > budget + 1e-6 for used, budget in zip(move["copy_peak_gib"], budgets))):
            raise ValueError("Release rebalance lacks full KV-copy bytes, integrity or safe peak evidence")
        expected_locations[move["request_id"]] = (move["model"], move["destination"])
    if expected_locations != locations(after):
        raise ValueError("Release rebalance final routing differs from committed migrations")
    if [[len(rank["requests"]) for rank in ranks] for ranks in after] != fresh["winner"]["final_counts"]:
        raise ValueError("Release rebalance final population differs from ILP")
    if decision.get("pause_call_s", 0) <= 0 or decision.get("active_blocking_upper_request_s") is None:
        raise ValueError("Release rebalance lacks measured default boundary/control time")
