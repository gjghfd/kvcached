"""Typed controller adapter; the running point-policy job does not import it."""

import copy
import math

from verify_workaware_robust_rebalance import verify_plan as verify_raw_plan
from workaware_robust_rebalance import plan_workaware_robust_rebalance


OPTIMIZER = "robust_point_plus_differential_uncertainty"
SCOPE = "robust_release_controller_plan_not_GPU_performance_evidence"


def _finite_time(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def wrap_plan(raw):
    if "controller_adapter" in raw or raw.get("gate", {}).get("phase") not in ("before_pause", "after_pause"):
        raise ValueError("Require an unwrapped robust release plan and original phase")
    result = dict(raw)
    result["scope"] = SCOPE
    result["controller_adapter"] = {
        "schema_version": 1, "optimizer": OPTIMIZER, "controller_planning_s": None,
        "raw_planner_accounting": {"scope": raw["scope"], "planning_s": raw["planning_s"],
                                   "gate": copy.deepcopy(raw["gate"])}}
    result["gate"] = copy.deepcopy(raw["gate"])
    return result


def plan(observed, budgets, fit, allow_extrapolation, settings, lookup, blocking, phase):
    return wrap_plan(plan_workaware_robust_rebalance(
        observed, budgets, fit, allow_extrapolation, settings, lookup, blocking, phase))


def account_paused_controller_planning(check, elapsed_s, blocking_before_s):
    adapter = check.get("controller_adapter", {})
    raw = adapter.get("raw_planner_accounting", {})
    if (check.get("scope") != SCOPE or type(adapter.get("schema_version")) is not int
            or adapter["schema_version"] != 1 or adapter.get("optimizer") != OPTIMIZER
            or check.get("gate", {}).get("phase") != "after_pause"
            or not _finite_time(elapsed_s) or not _finite_time(blocking_before_s)
            or not _finite_time(raw.get("planning_s")) or elapsed_s < raw["planning_s"]
            or not isinstance(raw.get("gate"), dict) or raw["gate"].get("phase") != "after_pause"
            or type(check.get("active_population")) is not int or check["active_population"] < 0):
        raise ValueError("Require after-pause controller walltime at least the original planner time")
    active = check["active_population"] * elapsed_s
    adapter["controller_planning_s"] = elapsed_s
    check["planning_s"] = elapsed_s
    check["gate"] = {**raw["gate"], "active_planning_blocking_request_s": active,
                     "measured_blocking_so_far_request_s": blocking_before_s + active}
    return check


def verify_controller_plan(check, observed, budgets, fit, settings, phase, blocking_before_s,
                           controller_planning_s):
    adapter = check.get("controller_adapter", {})
    if (check.get("scope") != SCOPE or set(adapter) != {
            "schema_version", "optimizer", "controller_planning_s", "raw_planner_accounting"}
            or type(adapter.get("schema_version")) is not int or adapter["schema_version"] != 1
            or adapter.get("optimizer") != OPTIMIZER or not _finite_time(controller_planning_s)
            or not _finite_time(blocking_before_s)):
        raise ValueError("Invalid typed robust controller adapter or measured planning time")
    accounting = adapter["raw_planner_accounting"]
    if not isinstance(accounting, dict) or set(accounting) != {"scope", "planning_s", "gate"}:
        raise ValueError("Missing original robust planner accounting")
    raw = {name: value for name, value in check.items() if name != "controller_adapter"}
    raw.update(accounting)
    verification = verify_raw_plan(raw, observed, budgets, fit, settings, phase, blocking_before_s,
                                   controller_planning_upper_s=controller_planning_s)
    gate = copy.deepcopy(accounting["gate"])
    if phase == "after_pause":
        active = check["active_population"] * controller_planning_s
        gate.update({"active_planning_blocking_request_s": active,
                     "measured_blocking_so_far_request_s": blocking_before_s + active})
        if adapter["controller_planning_s"] != controller_planning_s or check["planning_s"] != controller_planning_s:
            raise ValueError("After-pause accounting omits actual controller/executor waiting")
    elif adapter["controller_planning_s"] is not None or check["planning_s"] != accounting["planning_s"]:
        raise ValueError("Before-pause planning must not be charged as paused inference")
    if check["gate"] != gate:
        raise ValueError("Controller accounting changes continuation economics or omits paid blocking")
    return {**verification, "optimizer": OPTIMIZER,
            "controller_walltime_source": "caller_reported_including_executor_queue_not_independent_clock_proof",
            "paid_controller_planning_excluded_from_continuation_profit": phase == "after_pause"}
