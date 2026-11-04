"""Independent release planner; not installed in the live runtime.

Objectives, empirical uncertainty and active blocking use request-seconds.
Bandwidth uses GiB/s; pause uses seconds; attention coefficients use seconds/token.
Paid after-pause planning is recorded but excluded from continuation profit.
Population-specific profiles align the solver's differential uncertainty and gate.
"""

import math
import time

import ilp_reference
from live_controller import admission_planning_snapshot
from verify_benefit import verify_ownership
from workaware_cost import headroom_modes, reference_horizon_s, remaining_segments
from workaware_rebalance import workaware_gate
from workaware_robust_ilp import robust_replay, solve_robust_candidate


def _finite_cost(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _validate_inputs(observed, budgets, fit, calibration, allow_extrapolation, settings, lookup,
                     blocking_cost_s, phase):
    supported = {"max_moves", "candidate_limit", "move_bandwidth_gib_s", "solver_limit_s",
                 "pause_fixed_s", "min_gain_s", "workaware_fit", "workaware_profile",
                 "work_interval_steps", "policy_mode", "credit_request_s", "overhead_fraction",
                 "headroom_profile"}
    static_costs = {"performance_cost_tables_s", "joint_performance_costs_s", "imbalance_cost_s"}
    if (lookup is not None or set(settings) - supported
            or static_costs.intersection(fit) or static_costs.intersection(calibration)
            or settings.get("policy_mode", "workaware") != "workaware"):
        raise ValueError("Unsupported static cost configuration; require release-only workaware costs")
    if calibration.get("population_profiles", {}).get("status") != "raw_verified_workaware_population_profiles":
        raise ValueError("Require population-specific raw-verified uncertainty; legacy global-only guard is unsupported")
    if (type(allow_extrapolation) is not bool or phase not in ("before_pause", "after_pause")
            or not _finite_cost(blocking_cost_s)
            or len(budgets) != 2 or any(not _finite_cost(value, True) for value in budgets)
            or type(settings["max_moves"]) is not int or settings["max_moves"] < 0
            or type(settings["candidate_limit"]) is not int or settings["candidate_limit"] < 1
            or any(not _finite_cost(settings[name], True)
                   for name in ("move_bandwidth_gib_s", "solver_limit_s"))
            or any(not _finite_cost(settings[name]) for name in ("pause_fixed_s", "min_gain_s"))
            or type(settings.get("work_interval_steps", 128)) is not int
            or settings.get("work_interval_steps", 128) < 1):
        raise ValueError("Invalid budgets, limits, request-second costs or planning phase")
    if (type(fit["requests"]) is not int or fit["requests"] < 1
            or not _finite_cost(fit["baseline_cadence_s"], True)
            or any(not _finite_cost(fit["capture_shape_posthoc"][name]) for name in ("a", "b"))):
        raise ValueError("Invalid raw sparse fit population, cadence or dimensionless slopes")
    if (len(observed) != 2 or any(len(ranks) != 2 for ranks in observed)
            or sum(any(rank["requests"] for rank in ranks) for ranks in observed) > 1):
        raise ValueError("Require two ranks per tenant and a single active tenant; mixed work is unverified")
    for ranks in observed:
        for rank in ranks:
            geometry = ("block_size", "block_bytes", "page_bytes", "page_block_capacity")
            if (rank["uniform_page_geometry"] is not True
                    or any(type(rank[name]) is not int or rank[name] <= 0 for name in geometry)
                    or rank["page_bytes"] != rank["block_bytes"] * rank["page_block_capacity"]):
                raise ValueError("Require measured uniform physical page/block geometry")
            pages = {page["id"] for page in rank["pages"]}
            if (len(pages) != len(rank["pages"])
                    or any(not _finite_cost(page["memory_gib"], True)
                           or not math.isclose(page["memory_gib"], rank["page_bytes"] / 1024**3)
                           for page in rank["pages"])
                    or any(type(block) is not int or block <= 0
                           or block // rank["page_block_capacity"] not in pages
                           for request in rank["requests"] for block in request["blocks"])):
                raise ValueError("Request blocks require unique, measured source pages")
            for request in rank["requests"]:
                if (request["movable"] is not True
                        or any(type(request[name]) is not int or request[name] < 0
                               for name in ("max_tokens", "output_tokens", "computed_tokens"))):
                    raise ValueError("Require movable decode requests with integer token counts")
    verify_ownership(observed, budgets)
    coefficients = calibration["coefficients"]
    if (not _finite_cost(coefficients["base_s"], True)
            or any(not _finite_cost(coefficients[name])
                   for name in ("graph_batch_s", "attention_token_s"))
            or not _finite_cost(calibration["residual_bound_s"])):
        raise ValueError("Require positive base seconds and nonnegative graph, attention and residual costs")
    for profile in calibration.get("population_profiles", {}).get("profiles", {}).values():
        if (any(not _finite_cost(value) for value in profile["graph_penalties_s"].values())
                or any(not _finite_cost(profile[name])
                       for name in ("attention_token_s", "contrast_residual_bound_s"))):
            raise ValueError("Require nonnegative finite graph/residual seconds and attention seconds/token")
    profiles = settings["workaware_profile"]["models"]
    if len(profiles) != 2:
        raise ValueError("Require two past-arrival profiles")
    for profile in profiles:
        if (not _finite_cost(profile["rate_rps"])
                or type(profile["tokens"]) is not int or profile["tokens"] < 0
                or any(not _finite_cost(value) for value in profile["cold_cost_samples_s"])):
            raise ValueError("Invalid past-arrival rate, integer KV quantile or request-second cold cost")


def _verified_incumbent(snapshot, candidate, max_moves):
    if (candidate.get("status") not in ("optimal", "feasible_time_limited")
            or candidate.get("target") is not None or not isinstance(candidate.get("moves"), list)
            or len(candidate["moves"]) > max_moves):
        return None
    replay = robust_replay(snapshot, candidate)
    for name in ("objective_s", "robust_objective_request_s", "uncertainty_request_s"):
        if (not _finite_cost(candidate.get(name)) or not _finite_cost(replay.get(name))
                or not math.isclose(candidate[name], replay[name], rel_tol=1e-8, abs_tol=1e-6)):
            return None
    return {**candidate, **replay, "status": candidate["status"]}


def plan_workaware_robust_rebalance(observed, budgets, fit, allow_extrapolation, settings, lookup,
                                   blocking_cost_s, phase):
    started = time.monotonic()
    if not getattr(ilp_reference, "SUPPORTS_WORK_SEGMENTS", False):
        raise RuntimeError("Work-aware ILP integration is unavailable; never omit remaining-work costs")
    calibration = settings["workaware_fit"]
    _validate_inputs(observed, budgets, fit, calibration, allow_extrapolation, settings, lookup,
                     blocking_cost_s, phase)
    segments = remaining_segments(observed, calibration, settings.get("work_interval_steps", 128))
    snapshot, _, population = admission_planning_snapshot(
        observed, 0, 0, 0, budgets, fit, allow_extrapolation, "ilp", settings["candidate_limit"],
        0, settings["move_bandwidth_gib_s"], None, new_request_count=0)
    if snapshot.get("joint_performance_costs_s") is not None:
        raise ValueError("Unsupported static joint cost configuration")
    snapshot["performance_cost_tables_s"] = [[0] * len(table) for table in snapshot["performance_cost_tables_s"]]
    snapshot["work_segments"] = segments
    snapshot["migration_activation_cost_s"] = population * settings["pause_fixed_s"] if phase == "before_pause" else 0
    profiles = settings["workaware_profile"]["models"]
    cold_costs = []
    for model, profile in enumerate(profiles):
        samples = profile["cold_cost_samples_s"]
        observation = observed[model][0]
        size_gib = math.ceil(profile["tokens"] / observation["block_size"]) * observation["block_bytes"] / 1024**3
        prior = (population + 1) * (settings["pause_fixed_s"] + size_gib / settings["move_bandwidth_gib_s"])
        cold_costs.append(max(prior, sum(samples) / len(samples)) if samples else prior)
    horizon = reference_horizon_s(segments, calibration)
    scenarios = headroom_modes(snapshot, observed, profiles, horizon, cold_costs)
    plans, references = [], []
    for scenario in scenarios:
        selected = {**scenario["snapshot"], "placement_risk_cost_s": scenario["risk_cost_s"]}
        reference = robust_replay(selected, {"target": None, "moves": [], "status": "unchanged_layout"})
        if "robust_objective_request_s" in reference:
            references.append({"snapshot": selected, "reference": reference})
        plans.append({**scenario, "snapshot": selected, "reference": reference})
    if not references:
        raise RuntimeError("No independently verified unchanged-layout fallback")
    reference_entry = min(references, key=lambda entry: entry["reference"]["robust_objective_request_s"])
    reference = reference_entry["reference"]
    winner, selected_snapshot = reference, reference_entry["snapshot"]
    budget_left = settings["solver_limit_s"]
    deadline = time.monotonic() + budget_left
    for index, entry in enumerate(plans):
        allocated = min(budget_left / (len(plans) - index), max(0., deadline - time.monotonic()))
        candidate = {"status": "shared_solver_budget_exhausted"}
        if allocated > 0:
            budget_left = max(0., budget_left - allocated)
            try:
                candidate = solve_robust_candidate(entry["snapshot"], settings["max_moves"], allocated)
            except (RuntimeError, TimeoutError) as error:
                candidate = {"status": "robust_solver_failed", "diagnosis": str(error)}
        verified = _verified_incumbent(entry["snapshot"], candidate, settings["max_moves"])
        fallback = entry["reference"] if "robust_objective_request_s" in entry["reference"] else None
        if fallback is not None and (verified is None
                or fallback["robust_objective_request_s"] <= verified["robust_objective_request_s"]):
            verified = fallback
        entry["plan"] = {"status": candidate["status"], "winner": verified, "candidates": [candidate],
                         "solver_budget_s": allocated}
        if verified is not None and verified["robust_objective_request_s"] < winner["robust_objective_request_s"]:
            winner, selected_snapshot = verified, entry["snapshot"]
    if "population_profiles" in calibration:
        from workaware_profile_cost import changed_segment_uncertainty

        uncertainty = changed_segment_uncertainty(segments, winner["moves"])
    else:
        uncertainty = sum(segment["uncertainty_request_s"] for segment in segments)
    gate = workaware_gate(reference, winner, settings, uncertainty, blocking_cost_s, phase)
    planning_s = max(0., time.monotonic() - started)
    active_blocking = population * planning_s if phase == "after_pause" else 0.
    gate.update({"active_planning_blocking_request_s": active_blocking,
                 "paid_planning_excluded_from_continuation_profit": phase == "after_pause",
                 "measured_blocking_so_far_request_s": blocking_cost_s + active_blocking})
    return {"snapshot": selected_snapshot, "unconstrained_snapshot": snapshot,
            "horizon_steps": max((segment["end_step"] for segment in segments), default=0),
            "forecast_horizon_s": horizon, "active_population": population,
            "reference": reference, "winner": winner, "work_segments": segments,
            "robust_objective_request_s": winner["robust_objective_request_s"],
            "plan": {"status": "workaware_robust_soft_headroom_enumerated", "winner": winner,
                     "scenarios": plans, "shared_solver_limit_s": settings["solver_limit_s"]},
            "gate": gate, "planning_s": planning_s,
            "scope": "independent_robust_release_planner_not_live_runtime_or_GPU_benefit"}
