"""Replay release plans without invoking a planner or an optimization solver.

Point objectives, differential uncertainty and blocking use request-seconds.
Reported planning_s and controller_planning_upper_s use wall seconds. A single
plan cannot prove actual walltime; an optional controller bound checks the log
against caller-supplied external evidence, not a measurement by this verifier.
Already-paid planning and blocking never change the continuation-profit gate.
Solver statuses/bounds are checked for consistency, not proven globally optimal.
"""

import math

from ilp_reference import replay_candidate
from live_controller import admission_planning_snapshot
from verify_benefit import verify_ownership
from verify_workaware_rebalance import same_replayed_cost
from workaware_cost import headroom_modes, reference_horizon_s, remaining_segments
from workaware_profile_cost import changed_segment_uncertainty
from workaware_rebalance import workaware_gate


def _finite(value, positive=False, signed=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (signed or (value > 0 if positive else value >= 0)))


def _equal(logged, expected, name):
    if not same_replayed_cost(logged, expected):
        raise ValueError("Robust release replay mismatch: " + name)


def _within(value, maximum):
    return value <= maximum or abs(value - maximum) <= 4 * max(math.ulp(value), math.ulp(maximum))


def _validate_inputs(observed, budgets, fit, settings, phase, blocking):
    supported = {"max_moves", "candidate_limit", "move_bandwidth_gib_s", "solver_limit_s",
                 "pause_fixed_s", "min_gain_s", "workaware_fit", "workaware_profile",
                 "work_interval_steps", "policy_mode", "credit_request_s", "overhead_fraction",
                 "headroom_profile"}
    calibration = settings["workaware_fit"]
    static = {"performance_cost_tables_s", "joint_performance_costs_s", "imbalance_cost_s"}
    if (set(settings) - supported or static.intersection(fit) or static.intersection(calibration)
            or settings.get("policy_mode", "workaware") != "workaware"):
        raise ValueError("Unsupported static cost configuration")
    if calibration.get("population_profiles", {}).get("status") != "raw_verified_workaware_population_profiles":
        raise ValueError("Require raw-verified population profiles; legacy-only uncertainty is unsupported")
    if (phase not in ("before_pause", "after_pause") or not _finite(blocking)
            or len(budgets) != 2 or any(not _finite(value, True) for value in budgets)
            or type(settings["max_moves"]) is not int or settings["max_moves"] < 0
            or type(settings["candidate_limit"]) is not int or settings["candidate_limit"] < 1
            or type(settings.get("work_interval_steps", 128)) is not int
            or settings.get("work_interval_steps", 128) < 1
            or any(not _finite(settings[name], True) for name in ("solver_limit_s", "move_bandwidth_gib_s"))
            or any(not _finite(settings[name]) for name in ("pause_fixed_s", "min_gain_s"))):
        raise ValueError("Invalid physical budgets, limits, phase or request-second costs")
    if (type(fit["requests"]) is not int or fit["requests"] < 1
            or not _finite(fit["baseline_cadence_s"], True)
            or any(not _finite(fit["capture_shape_posthoc"][name]) for name in ("a", "b"))
            or not _finite(calibration["coefficients"]["base_s"], True)
            or any(not _finite(calibration["coefficients"][name])
                   for name in ("graph_batch_s", "attention_token_s"))
            or not _finite(calibration["residual_bound_s"])):
        raise ValueError("Require positive cadence/base seconds and nonnegative calibrated costs")
    for profile in calibration["population_profiles"]["profiles"].values():
        if (any(not _finite(value) for value in profile["graph_penalties_s"].values())
                or any(not _finite(profile[name]) for name in ("attention_token_s", "contrast_residual_bound_s"))):
            raise ValueError("Invalid graph/residual seconds or attention seconds/token")
    profiles = settings["workaware_profile"]["models"]
    if len(profiles) != 2 or any(not _finite(profile["rate_rps"])
            or type(profile["tokens"]) is not int or profile["tokens"] < 0
            or any(not _finite(value) for value in profile["cold_cost_samples_s"]) for profile in profiles):
        raise ValueError("Invalid past-arrival rate, integer token quantile or cold request-second cost")
    if (len(observed) != 2 or any(len(ranks) != 2 for ranks in observed)
            or sum(any(rank["requests"] for rank in ranks) for ranks in observed) > 1):
        raise ValueError("Require two ranks per tenant and at most one active tenant; mixed work is unverified")
    for ranks in observed:
        for rank in ranks:
            if (rank["uniform_page_geometry"] is not True
                    or any(type(rank[name]) is not int or rank[name] <= 0
                           for name in ("block_size", "block_bytes", "page_bytes", "page_block_capacity"))
                    or rank["page_bytes"] != rank["block_bytes"] * rank["page_block_capacity"]):
                raise ValueError("Invalid measured physical page/block geometry")
            pages = {page["id"] for page in rank["pages"]}
            if (len(pages) != len(rank["pages"])
                    or any(not _finite(page["memory_gib"], True)
                           or not math.isclose(page["memory_gib"], rank["page_bytes"] / 1024**3)
                           for page in rank["pages"])):
                raise ValueError("Invalid physical source-page inventory")
            for request in rank["requests"]:
                if (request["movable"] is not True
                        or any(type(request[name]) is not int or request[name] < 0
                               for name in ("max_tokens", "output_tokens", "computed_tokens"))
                        or any(type(block) is not int or block <= 0
                               or block // rank["page_block_capacity"] not in pages for block in request["blocks"])):
                    raise ValueError("Invalid decode state or unmapped request KV blocks")
    verify_ownership(observed, budgets)


def _replay(snapshot, candidate):
    replay = replay_candidate(snapshot, candidate)
    if "objective_s" not in replay:
        return replay
    uncertainty = changed_segment_uncertainty(snapshot["work_segments"], replay["moves"])
    return {**replay, "uncertainty_request_s": uncertainty,
            "robust_objective_request_s": replay["objective_s"] + uncertainty,
            "scope": "offline_unchanged_error_bound_and_independent_serial_copy_peak_replay"}


def _check_costs(logged, replay, max_moves):
    if "objective_s" not in replay:
        _equal(logged, replay, "infeasible unchanged reference")
        return
    if not isinstance(logged["moves"], list) or len(logged["moves"]) > max_moves:
        raise ValueError("Candidate exceeds the frozen serial move limit")
    for name in ("objective_s", "performance_cost_s", "migration_cost_s", "admission_cost_s",
                 "migration_activation_cost_s", "uncertainty_request_s", "robust_objective_request_s"):
        if not _finite(logged[name]):
            raise ValueError("Invalid finite nonnegative request-second cost: " + name)
    for name in ("target", "moves", "objective_s", "performance_cost_s", "migration_cost_s",
                 "admission_cost_s", "migration_activation_cost_s", "work_cost", "final_counts",
                 "final_gib", "peak_gib", "released_source_pages", "uncertainty_request_s",
                 "robust_objective_request_s"):
        _equal(logged[name], replay[name], name)


def _dominance_bound(snapshot, max_moves):
    if not max_moves or not snapshot["movable"]:
        return 0.
    gross = 0.
    for segment in snapshot["work_segments"]:
        work = [sum(weight for identifier, weight in segment["attention_weights"].items()
                    if segment["request_sources"][identifier] == rank) for rank in (0, 1)]
        current = segment["graph_costs_s"][segment["initial_counts"][0]]
        current += segment["attention_difference_cost_s"] * abs(work[0] - work[1])
        slack = max(1e-10, max(segment["graph_costs_s"]) * 1e-12)
        slack += segment["attention_difference_cost_s"] * max(
            1e-6, sum(segment["attention_weights"].values()) * 1e-12)
        gross += max(slack, current - min(segment["graph_costs_s"]) - segment["uncertainty_request_s"])
    return gross - snapshot["migration_activation_cost_s"] - min(
        request["move_cost_s"] for request in snapshot["movable"])


def _check_solver_claim(candidate, snapshot, max_moves):
    source = candidate["candidate_source"]
    if source not in ("robust_ILP_incumbent", "verified_unchanged_fallback", "unchanged_dominance_certificate"):
        raise ValueError("Unsupported solver candidate-source claim")
    if source != "robust_ILP_incumbent" and candidate["moves"]:
        raise ValueError("An unchanged fallback/certificate cannot claim moves")
    if source == "verified_unchanged_fallback" and candidate["status"] != "feasible_time_limited":
        raise ValueError("A time-limited unchanged fallback is not a solver optimum")
    if source == "unchanged_dominance_certificate":
        bound = _dominance_bound(snapshot, max_moves)
        if candidate["status"] != "optimal" or bound > 0:
            raise ValueError("Unchanged-dominance claim lacks a valid unchanged advantage bound")
        _equal(candidate["upper_bound_move_gain_request_s"], bound, "unchanged dominance bound")
    bound, gap = candidate["robust_dual_bound_request_s"], candidate["robust_mip_gap"]
    if bound is None:
        if gap is not None or candidate["status"] == "optimal":
            raise ValueError("Missing dual bound cannot support a gap or claimed solver optimum")
        return
    objective = candidate["robust_objective_request_s"]
    if (not _finite(bound, signed=True) or not _finite(gap)
            or bound > objective + 1e-6 + 1e-8 * abs(objective)):
        raise ValueError("Invalid solver-reported bound or gap")
    expected_gap = max(0, objective - bound) / max(abs(objective), 1e-10)
    _equal(gap, expected_gap, "solver-reported gap arithmetic")
    if candidate["status"] == "optimal" and not math.isclose(bound, objective, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("Claimed solver optimum has a nonclosed reported bound")


def _verify_plan(check, observed, budgets, fit, settings, phase, blocking, controller_planning_upper_s):
    _validate_inputs(observed, budgets, fit, settings, phase, blocking)
    calibration = settings["workaware_fit"]
    segments = remaining_segments(observed, calibration, settings.get("work_interval_steps", 128))
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
    for name, expected in (("unconstrained_snapshot", snapshot), ("work_segments", segments),
                           ("active_population", population), ("forecast_horizon_s", horizon),
                           ("horizon_steps", max((segment["end_step"] for segment in segments), default=0))):
        _equal(check[name], expected, name)
    _equal(check["scope"], "independent_robust_release_planner_not_live_runtime_or_GPU_benefit", "plan scope")
    _equal(check["plan"]["status"], "workaware_robust_soft_headroom_enumerated", "plan status")
    _equal(check["plan"]["shared_solver_limit_s"], settings["solver_limit_s"], "shared solver limit")
    logged_scenarios = check["plan"]["scenarios"]
    if not isinstance(logged_scenarios, list) or len(logged_scenarios) != len(scenarios):
        raise ValueError("Missing or duplicated soft-headroom scenarios")
    entries, references, statuses, grants = [], [], [], []
    budget_left, exhausted = settings["solver_limit_s"], False
    for index, (logged, scenario) in enumerate(zip(logged_scenarios, scenarios)):
        selected = {**scenario["snapshot"], "placement_risk_cost_s": scenario["risk_cost_s"]}
        for name, expected in (("snapshot", selected), ("guards", scenario["guards"]),
                               ("risk_cost_s", scenario["risk_cost_s"])):
            _equal(logged[name], expected, "scenario " + name)
        reference = _replay(selected, {"target": None, "moves": [], "status": "unchanged_layout"})
        _check_costs(logged["reference"], reference, settings["max_moves"])
        fallback = reference if "objective_s" in reference else None
        if fallback is not None:
            _equal(logged["reference"], reference, "scenario unchanged reference")
            references.append((fallback, selected))
        scenario_plan = logged["plan"]
        candidates = scenario_plan["candidates"]
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Require the single retained solver result per scenario")
        candidate, grant = candidates[0], scenario_plan["solver_budget_s"]
        status = candidate["status"]
        if (not _finite(grant) or not _within(grant, budget_left / (len(scenarios) - index))
                or scenario_plan["status"] != status or (exhausted and grant > 0)):
            raise ValueError("Invalid shared solver grant or scenario status")
        budget_left = max(0., budget_left - grant)
        grants.append(grant)
        statuses.append(status)
        if (grant == 0) != (status == "shared_solver_budget_exhausted"):
            raise ValueError("Exhausted solver budget must have zero grant and no incumbent")
        exhausted = exhausted or grant == 0
        winner = None
        if status in ("optimal", "feasible_time_limited"):
            replay = _replay(selected, candidate)
            if "objective_s" not in replay:
                raise ValueError("Declared feasible incumbent violates independent physical-copy replay")
            _check_costs(candidate, replay, settings["max_moves"])
            _check_solver_claim(candidate, selected, settings["max_moves"])
            winner = {**candidate, **replay, "status": status}
        elif status in ("infeasible", "no_verified_robust_incumbent", "robust_solver_failed",
                        "shared_solver_budget_exhausted"):
            if any(name in candidate for name in ("moves", "objective_s", "robust_objective_request_s")):
                raise ValueError("A failed solver result cannot retain a purported feasible incumbent")
            if status == "infeasible" and fallback is not None:
                raise ValueError("Claimed infeasibility contradicts a verified unchanged layout")
        else:
            raise ValueError("Unsupported solver status; incumbents are not global optimality proofs")
        if fallback is not None and (winner is None
                or fallback["robust_objective_request_s"] <= winner["robust_objective_request_s"]):
            winner = fallback
        _equal(scenario_plan["winner"], winner, "scenario robust winner/fallback")
        entries.append((winner, selected))
    if not references or not _within(math.fsum(grants), settings["solver_limit_s"]):
        raise ValueError("Missing verified no-move reference or exceeded total solver grants")
    reference, selected = min(references, key=lambda entry: entry[0]["robust_objective_request_s"])
    winner = reference
    for candidate, candidate_snapshot in entries:
        if candidate is not None and candidate["robust_objective_request_s"] < winner["robust_objective_request_s"]:
            winner, selected = candidate, candidate_snapshot
    for name, expected in (("reference", reference), ("winner", winner), ("snapshot", selected),
                           ("robust_objective_request_s", winner["robust_objective_request_s"])):
        _equal(check[name], expected, "cross-scenario " + name)
    _equal(check["plan"]["winner"], winner, "top-level robust winner")
    planning = check["planning_s"]
    if not _finite(planning):
        raise ValueError("Reported planning duration must be finite nonnegative wall seconds")
    if controller_planning_upper_s is not None and (not _finite(controller_planning_upper_s)
            or not _within(planning, controller_planning_upper_s)):
        raise ValueError("Reported planning duration exceeds the external controller wall-second bound")
    uncertainty = changed_segment_uncertainty(segments, winner["moves"])
    gate = workaware_gate(reference, winner, settings, uncertainty, blocking, phase)
    active = population * planning if phase == "after_pause" else 0.
    gate.update({"active_planning_blocking_request_s": active,
                 "paid_planning_excluded_from_continuation_profit": phase == "after_pause",
                 "measured_blocking_so_far_request_s": blocking + active})
    _equal(check["gate"], gate, "original continuation gate and paid-blocking observation")
    return {"status": "verified_robust_incumbents_and_fallbacks_not_global_optimality",
            "accepted": gate["accepted"], "scenarios": len(scenarios), "solver_statuses": statuses,
            "total_solver_grants_s": math.fsum(grants), "global_optimality_verified": False,
            "actual_walltime_verified": False, "solver_wall_deadline_verified": False,
            "walltime_scope": ("reported_planning_s_only_not_independent_walltime_proof"
                               if controller_planning_upper_s is None else
                               "reported_planning_s_with_caller_supplied_controller_upper_bound"),
            "controller_planning_upper_s": controller_planning_upper_s}


def verify_plan(check, observed, budgets, fit, settings, phase, blocking, *, controller_planning_upper_s=None):
    """Verify logged economics/physical replay, not solver optimality or actual time.

    A supplied controller upper bound constrains reported walltime only. Even a
    very large paid pause/planning duration cannot reject positive continuation
    profit after marginal copy cost and the unchanged empirical uncertainty.
    """
    try:
        return _verify_plan(check, observed, budgets, fit, settings, phase, blocking, controller_planning_upper_s)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed robust release plan or input inventory: " + str(error)) from error
