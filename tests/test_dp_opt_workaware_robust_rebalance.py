import copy
import importlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in (
        "workaware_robust_rebalance", "workaware_rebalance", "workaware_robust_ilp", "workaware_cost"))


def legacy_fit():
    return {"requests": 128, "raw_verification": {"status": "raw_verified"},
            "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": .025,
            "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": .3, "b": 0}}


def settings(profiles=True):
    calibration = {"status": "raw_verified_workaware_cadence_fit",
                   "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
                   "scope": "one_active_tenant_other_resident_idle", "source_verification": "raw_verified",
                   "coefficients": {"base_s": .01, "graph_batch_s": 0., "attention_token_s": .004},
                   "residual_bound_s": 1e-6, "max_population": 9, "max_context_tokens": 32768,
                   "max_attention_tokens": 131072}
    if profiles:
        calibration["population_profiles"] = {
            "status": "raw_verified_workaware_population_profiles", "profiles": {
                f"{model}:{population}": {"graph_penalties_s": {"1": 0., "8": 0., "16": 0.},
                                          "attention_token_s": .004, "contrast_residual_bound_s": .001,
                                          "max_context_tokens": 32768, "max_attention_tokens": 131072,
                                          "uncertainty_scope": "synthetic_CPU_fixture_not_GPU_evidence"}
                for model in (0, 1) for population in range(2, 10)}}
    return {"max_moves": 1, "candidate_limit": 8, "move_bandwidth_gib_s": 100.,
            "solver_limit_s": 5., "pause_fixed_s": .001, "min_gain_s": .001,
            "policy_mode": "workaware", "work_interval_steps": 128, "workaware_fit": calibration,
            "workaware_profile": {"models": [
                {"rate_rps": 0., "tokens": 0, "cold_cost_samples_s": []},
                {"rate_rps": 0., "tokens": 0, "cold_cost_samples_s": []}]}}


def observations(request_specs=None, model=1, source=0, page_mode="private"):
    if request_specs is None:
        request_specs = [("first", 30, 256), ("second", 30, 256)]
    geometry = {"uniform_page_geometry": True, "page_block_capacity": 64,
                "page_bytes": 128 * 1024**2, "block_size": 16, "block_bytes": 2 * 1024**2}
    observed = [[{**geometry, "requests": [], "pages": [], "mapped_bytes": 0}
                 for rank in (0, 1)] for tenant in (0, 1)]
    rank = observed[model][source]
    for position, (identifier, remaining, context) in enumerate(request_specs):
        page_id = position + 1 if page_mode != "shared" else position // 2 + 1
        offset = 0 if page_mode != "shared" else 32 * (position % 2)
        blocks = list(range(page_id * 64 + offset, page_id * 64 + offset + 32))
        rank["requests"].append({"id": identifier, "blocks": blocks, "movable": True,
                                 "max_tokens": remaining + 1, "output_tokens": 1,
                                 "computed_tokens": context, "prompt_tokens": context - 1})
    for page_id in sorted({block // 64 for request in rank["requests"] for block in request["blocks"]}):
        owners = sorted(request["id"] for request in rank["requests"]
                        if any(block // 64 == page_id for block in request["blocks"]))
        rank["pages"].append({"id": page_id, "owners": owners, "pinned": page_mode == "pinned",
                              "memory_gib": .125})
    rank["mapped_bytes"] = len(rank["pages"]) * rank["page_bytes"]
    return observed


def plan(modules, observed=None, configured=None, budgets=None, phase="before_pause", blocking=0.):
    return modules[0].plan_workaware_robust_rebalance(
        observations() if observed is None else observed, [1., 1.] if budgets is None else budgets,
        legacy_fit(), True, settings() if configured is None else configured, None, blocking, phase)


def exhaustive_winner(modules, result, max_moves):
    choices = []
    for entry in result["plan"]["scenarios"]:
        identifiers = [request["id"] for request in entry["snapshot"]["movable"]]
        for count in range(min(max_moves, len(identifiers)) + 1):
            for moves in itertools.permutations(identifiers, count):
                replay = modules[2].robust_replay(entry["snapshot"], {"target": None, "moves": list(moves)})
                if "robust_objective_request_s" in replay:
                    choices.append(replay)
    return min(choices, key=lambda candidate: candidate["robust_objective_request_s"])


@pytest.mark.parametrize("model", (0, 1))
@pytest.mark.parametrize("profiles", (False, True))
def test_real_snapshots_preserve_inputs_and_verified_profile_protocol(modules, model, profiles):
    observed, configured = observations(model=model), settings(profiles)
    original = copy.deepcopy((observed, configured))
    if not profiles:
        with pytest.raises(ValueError, match="legacy global-only guard"):
            plan(modules, observed, configured)
        assert (observed, configured) == original
        return
    result = plan(modules, observed, configured)
    assert result["gate"]["accepted"]
    assert len(result["winner"]["moves"]) == 1
    assert (observed, configured) == original
    assert {"snapshot", "unconstrained_snapshot", "horizon_steps", "forecast_horizon_s", "active_population",
            "reference", "winner", "work_segments", "plan", "gate"} <= set(result)
    assert result["active_population"] == 2
    assert result["horizon_steps"] == 30
    assert result["snapshot"]["new_request_count"] == 0
    assert result["snapshot"]["new_gib"] == 0
    assert result["snapshot"]["migration_activation_cost_s"] == .002
    assert result["winner"]["migration_cost_s"] == pytest.approx(2 * .0625 / 100)
    assert result["winner"]["objective_s"] == pytest.approx(
        result["winner"]["performance_cost_s"] + result["winner"]["migration_cost_s"] + .002)
    assert result["robust_objective_request_s"] == pytest.approx(
        result["winner"]["objective_s"] + result["winner"]["uncertainty_request_s"])
    expected_gate = modules[1].workaware_gate(result["reference"], result["winner"], configured,
                                             result["gate"]["uncertainty_request_s"], 0., "before_pause")
    assert all(result["gate"][name] == value for name, value in expected_gate.items())
    assert result["forecast_horizon_s"] == modules[3].reference_horizon_s(
        result["work_segments"], configured["workaware_fit"])
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 1)["robust_objective_request_s"])


def test_original_gain_and_profile_empirical_error_guard_are_not_lowered(modules):
    configured = settings()
    configured["min_gain_s"] = 1000.
    result = plan(modules, configured=configured)
    assert result["winner"]["moves"]
    assert not result["gate"]["accepted"]
    assert result["gate"]["uncertainty_request_s"] == result["winner"]["uncertainty_request_s"]
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 10.
    result = plan(modules, configured=configured)
    assert result["winner"] == result["reference"]
    assert not result["gate"]["accepted"]


def test_legacy_global_guard_cannot_be_silently_optimized_as_changed_segment_error(modules):
    with pytest.raises(ValueError, match="legacy global-only guard"):
        plan(modules, configured=settings(False))


def test_zero_base_intercept_remains_rejected_by_existing_runtime_contract(modules):
    configured = settings()
    configured["workaware_fit"]["coefficients"]["base_s"] = 0.
    with pytest.raises(ValueError, match="positive base seconds"):
        plan(modules, configured=configured)


def test_uncertainty_changes_winner_instead_of_rejecting_point_optimum(modules):
    observed = observations([("short", 1, 128), ("long-a", 3, 256), ("long-b", 3, 256)])
    configured = settings()
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 1.
    old = modules[1].plan_workaware_rebalance(observed, [1., 1.], legacy_fit(), True, configured,
                                              None, 0., "before_pause")
    result = plan(modules, observed, configured)
    assert old["winner"]["moves"][0].startswith("long-")
    assert not old["gate"]["accepted"]
    assert result["winner"]["moves"] == ["short"]
    assert result["winner"]["objective_s"] > old["winner"]["objective_s"]
    assert result["gate"]["accepted"]
    assert result["gate"]["uncertainty_request_s"] == pytest.approx(.006)
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 1)["robust_objective_request_s"])


def test_soft_headroom_can_require_moving_even_when_work_dominance_bound_is_negative(modules):
    observed = observations([("only", 20, 256)])
    idle = observed[0][1]
    idle["pages"] = [{"id": identifier, "owners": [], "pinned": True, "memory_gib": .125}
                     for identifier in (1, 2)]
    idle["mapped_bytes"] = 2 * idle["page_bytes"]
    configured = settings()
    configured["workaware_fit"]["coefficients"]["attention_token_s"] = 0.
    configured["workaware_profile"]["models"][0].update(
        {"tokens": 1025, "rate_rps": 100., "cold_cost_samples_s": [10.]})
    result = plan(modules, observed, configured, [.25, .375])
    assert result["gate"]["accepted"]
    assert result["winner"]["moves"] == ["only"]
    assert result["winner"]["work_cost"]["performance_cost_s"] == (
        result["reference"]["work_cost"]["performance_cost_s"])
    selected = next(entry for entry in result["plan"]["scenarios"] if entry["snapshot"] == result["snapshot"])
    assert selected["guards"][0]["protected_rank"] == 0
    assert selected["reference"]["status"] == "revalidation_headroom_rejected"
    assert modules[2].unchanged_dominance_bound(result["snapshot"], 1) < 0
    assert selected["plan"]["candidates"][0]["candidate_source"] != "unchanged_dominance_certificate"
    assert result["winner"]["final_gib"] == [0., .375]
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 1)["robust_objective_request_s"])


def test_cross_scenario_selection_uses_robust_not_point_objective(modules):
    observed = observations([("short", 1, 128), ("long-a", 3, 256), ("long-b", 3, 200)])
    rank = observed[1][0]
    rank["requests"][1]["blocks"] = list(range(128, 224))
    rank["requests"][2]["blocks"] = list(range(256, 288))
    rank["pages"] = [{"id": identifier, "owners": [owner], "pinned": False, "memory_gib": .125}
                     for identifier, owner in ((1, "short"), (2, "long-a"), (3, "long-a"), (4, "long-b"))]
    rank["mapped_bytes"] = 4 * rank["page_bytes"]
    configured = settings()
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 1.
    configured["workaware_profile"]["models"][0].update(
        {"tokens": 1025, "rate_rps": 100., "cold_cost_samples_s": [100.]})
    old = modules[1].plan_workaware_rebalance(observed, [.5, .375], legacy_fit(), True, configured,
                                              None, 0., "before_pause")
    result = plan(modules, observed, configured, [.5, .375])
    assert old["winner"]["moves"] == ["long-a"]
    assert old["snapshot"]["final_capacity_gib"] == [.25, .375]
    assert result["winner"]["moves"] == ["short"]
    assert result["snapshot"]["final_capacity_gib"] == [.5, .125]
    assert result["gate"]["accepted"]
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 1)["robust_objective_request_s"])


@pytest.mark.parametrize("population", (2, 3, 4))
@pytest.mark.parametrize("page_mode", ("private", "shared", "pinned"))
@pytest.mark.parametrize("source", (0, 1))
def test_serial_copy_owner_and_cross_scenario_objective_match_exhaustive_oracle(
        modules, population, page_mode, source):
    specs = [(f"request-{position}", position + 2, 128 + 16 * position) for position in range(population)]
    observed = observations(specs, source=source, page_mode=page_mode)
    configured = settings()
    configured["max_moves"] = 2
    for profile in configured["workaware_profile"]["models"]:
        profile.update({"tokens": 1025, "rate_rps": .02, "cold_cost_samples_s": [1.]})
    result = plan(modules, observed, configured, [.625, .625])
    assert len(result["plan"]["scenarios"]) == 9
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 2)["robust_objective_request_s"])
    assert all(used <= cap + 1e-6 for used, cap in zip(
        result["winner"]["peak_gib"], result["snapshot"]["capacity_gib"]))
    assert all(used <= cap + 1e-6 for used, cap in zip(
        result["winner"]["final_gib"], result["snapshot"]["final_capacity_gib"]))
    if page_mode == "pinned":
        assert result["winner"]["released_source_pages"] == []


def test_bidirectional_pool_reuse_remains_charged_and_matches_oracle(modules):
    observed = observations([("first", 3, 128), ("second", 3, 256)])
    observed[1][1] = observations([("other", 3, 160)], source=1)[1][1]
    configured = settings()
    configured["max_moves"] = 2
    result = plan(modules, observed, configured, [.375, .375])
    assert result["winner"]["robust_objective_request_s"] == pytest.approx(
        exhaustive_winner(modules, result, 2)["robust_objective_request_s"])
    exchanged = modules[2].robust_replay(result["snapshot"], {"target": None, "moves": ["first", "other"]})
    assert exchanged["released_source_pages"] == []
    assert exchanged["final_gib"] == [.375, .25]


def test_copy_peak_can_reject_point_profitable_move(modules):
    result = plan(modules, budgets=[.25, .0625])
    assert result["winner"] == result["reference"]
    assert not result["gate"]["accepted"]
    assert result["plan"]["scenarios"][0]["plan"]["candidates"][0]["status"] == "optimal"


def test_valid_unchanged_dominance_fastscreen_still_skips_milp(modules, monkeypatch):
    import scipy.optimize

    configured = settings()
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 100.

    def forbidden_solver(*args, **kwargs):
        raise AssertionError("Verified no-move dominance must avoid the MILP")

    monkeypatch.setattr(scipy.optimize, "milp", forbidden_solver)
    result = plan(modules, configured=configured)
    assert result["winner"] == result["reference"]
    candidate = result["plan"]["scenarios"][0]["plan"]["candidates"][0]
    assert candidate["candidate_source"] == "unchanged_dominance_certificate"


def test_time_limited_expensive_incumbent_does_not_displace_reference(modules, monkeypatch):
    import scipy.optimize

    original = scipy.optimize.milp

    def expensive_incumbent(objective, **options):
        lower, upper = np.array(options["bounds"].lb), np.array(options["bounds"].ub)
        lower[:3], upper[:3] = [0, 1, 0], [0, 1, 0]
        options["bounds"] = Bounds(lower, upper)
        result = original(objective, **options)
        assert result.success
        result.status, result.mip_dual_bound = 1, 0.
        return result

    monkeypatch.setattr(scipy.optimize, "milp", expensive_incumbent)
    observed = observations([("short", 1, 128), ("long-a", 3, 256), ("long-b", 3, 256)])
    configured = settings()
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 1.
    result = plan(modules, observed, configured)
    candidate = result["plan"]["scenarios"][0]["plan"]["candidates"][0]
    assert candidate["status"] == "feasible_time_limited"
    assert candidate["candidate_source"] == "verified_unchanged_fallback"
    assert result["winner"] == result["reference"]
    assert not result["gate"]["accepted"]


def test_verified_profitable_time_limited_incumbent_can_be_accepted(modules, monkeypatch):
    original = modules[0].solve_robust_candidate

    def limited_solver(*args, **kwargs):
        result = original(*args, **kwargs)
        result["status"] = "feasible_time_limited"
        return result

    monkeypatch.setattr(modules[0], "solve_robust_candidate", limited_solver)
    result = plan(modules)
    assert result["winner"]["status"] == "feasible_time_limited"
    assert result["gate"]["accepted"]


@pytest.mark.parametrize("fault", ("missing", "exception", "tampered_cost", "unsafe_moves", "expensive"))
def test_solver_failure_and_unverified_incumbents_use_verified_fallback(modules, monkeypatch, fault):
    original = modules[0].solve_robust_candidate

    def faulty_solver(snapshot, max_moves, time_limit_s):
        if fault == "missing":
            return {"status": "no_verified_robust_incumbent"}
        if fault == "exception":
            raise RuntimeError("Synthetic CPU solver failure")
        solved = original(snapshot, max_moves, time_limit_s)
        if fault == "tampered_cost":
            solved["robust_objective_request_s"] -= 1.
        elif fault == "unsafe_moves":
            solved["moves"] = ["unknown"]
        else:
            solved = modules[2].robust_replay(snapshot, {"target": None, "moves": ["first", "second"]})
            solved["status"] = "feasible_time_limited"
        return solved

    monkeypatch.setattr(modules[0], "solve_robust_candidate", faulty_solver)
    configured = settings()
    configured["max_moves"] = 2
    result = plan(modules, configured=configured)
    assert result["winner"] == result["reference"]
    assert not result["gate"]["accepted"]
    assert result["winner"]["robust_objective_request_s"] == result["winner"]["objective_s"]


def test_scenarios_share_total_granted_budget_and_wall_deadline(modules, monkeypatch):
    clock, grants = [0.], []
    monkeypatch.setattr(modules[0], "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def limited_solver(snapshot, max_moves, time_limit_s):
        grants.append(time_limit_s)
        clock[0] += 2.25 if len(grants) == 1 else 1.
        return {"status": "no_verified_robust_incumbent"}

    monkeypatch.setattr(modules[0], "solve_robust_candidate", limited_solver)
    configured = settings()
    configured["solver_limit_s"] = 3.
    configured["workaware_profile"]["models"][0].update({"rate_rps": .01, "tokens": 512})
    result = plan(modules, configured=configured)
    assert grants == pytest.approx([1., .75])
    assert sum(grants) <= configured["solver_limit_s"]
    assert result["plan"]["scenarios"][-1]["plan"]["status"] == "shared_solver_budget_exhausted"
    assert result["winner"] == result["reference"]
    assert not result["gate"]["accepted"]


@pytest.mark.parametrize("phase,planning_s,accepted", (
    ("before_pause", 2., True), ("after_pause", .01, True), ("after_pause", 2., True)))
def test_paid_after_pause_planning_does_not_change_continuation_profit(
        modules, monkeypatch, phase, planning_s, accepted):
    clock = [0.]
    monkeypatch.setattr(modules[0], "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = modules[0].solve_robust_candidate

    def slow_solver(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += planning_s
        return result

    monkeypatch.setattr(modules[0], "solve_robust_candidate", slow_solver)
    observed = observations([("first", 1, 256), ("second", 1, 256)])
    result = plan(modules, observed, phase=phase, blocking=1000000.)
    blocking = 2 * planning_s if phase == "after_pause" else 0.
    assert result["gate"]["accepted"] is accepted
    assert result["gate"]["active_planning_blocking_request_s"] == blocking
    assert result["gate"]["measured_blocking_so_far_request_s"] == 1000000. + blocking
    assert result["gate"]["sunk_cost_excluded_from_continuation_profit"] is (phase == "after_pause")
    assert result["gate"]["paid_planning_excluded_from_continuation_profit"] is (phase == "after_pause")
    assert result["snapshot"]["migration_activation_cost_s"] == (0. if phase == "after_pause" else .002)


def test_activation_is_charged_before_pause_only(modules):
    configured = settings()
    configured["pause_fixed_s"] = 100.
    before = plan(modules, configured=configured)
    after = plan(modules, configured=configured, phase="after_pause", blocking=10000.)
    assert before["winner"] == before["reference"]
    assert not before["gate"]["accepted"]
    assert after["gate"]["accepted"]
    assert after["winner"]["migration_activation_cost_s"] == 0.


@pytest.mark.parametrize("field,value", (
    ("max_moves", True), ("max_moves", -1), ("candidate_limit", 0), ("candidate_limit", True),
    ("work_interval_steps", 0), ("work_interval_steps", True), ("solver_limit_s", 0.),
    ("solver_limit_s", float("inf")), ("solver_limit_s", True), ("move_bandwidth_gib_s", -1.),
    ("pause_fixed_s", -1.), ("pause_fixed_s", True), ("min_gain_s", float("nan")),
    ("joint_performance_costs_s", [[0.]]), ("performance_cost_tables_s", [[0.]]),
    ("imbalance_cost_s", [0., 0.]), ("migration_budget_s", 0.), ("policy_mode", "headroom")))
def test_invalid_or_unsupported_settings_fail_closed(modules, field, value):
    configured = settings()
    configured[field] = value
    with pytest.raises(ValueError):
        plan(modules, configured=configured)


@pytest.mark.parametrize("fault", (
    "mixed", "lookup", "raw_status", "scope", "profile_status", "negative_profile", "boolean_profile",
    "population", "context", "geometry", "page_bytes", "owner", "missing_page", "prefill", "tokens",
    "blocking", "phase", "budgets", "extrapolation", "cold_cost", "rate", "kv_quantile",
    "static_fit", "static_profile", "boolean_fit"))
def test_unverified_domain_geometry_and_cost_units_are_rejected(modules, fault):
    observed, configured = observations(), settings()
    calibration = configured["workaware_fit"]
    fit = legacy_fit()
    lookup, blocking, phase, budgets, extrapolation = None, 0., "before_pause", [1., 1.], True
    if fault == "mixed":
        observed[0][0] = observations([("code", 3, 256)], model=0)[0][0]
    elif fault == "lookup":
        lookup = {"tables": {}}
    elif fault == "raw_status":
        calibration["source_verification"] = "unverified"
    elif fault == "scope":
        calibration["scope"] = "joint"
    elif fault == "profile_status":
        calibration["population_profiles"]["status"] = "diagnostic"
    elif fault in ("negative_profile", "boolean_profile"):
        profile = calibration["population_profiles"]["profiles"]["1:2"]
        profile["attention_token_s"] = -1. if fault == "negative_profile" else True
    elif fault == "population":
        calibration["max_population"] = 1
    elif fault == "context":
        calibration["max_context_tokens"] = 128
    elif fault == "geometry":
        observed[1][0]["uniform_page_geometry"] = False
    elif fault == "page_bytes":
        observed[1][0]["pages"][0]["memory_gib"] = .25
    elif fault == "owner":
        observed[1][0]["pages"][0]["owners"] = ["second"]
    elif fault == "missing_page":
        observed[1][0]["requests"][0]["blocks"] = [9999]
    elif fault == "prefill":
        observed[1][0]["requests"][0]["movable"] = False
    elif fault == "tokens":
        observed[1][0]["requests"][0]["max_tokens"] = True
    elif fault == "blocking":
        blocking = True
    elif fault == "phase":
        phase = "during_copy"
    elif fault == "budgets":
        budgets = [True, 1.]
    elif fault == "extrapolation":
        extrapolation = 1
    elif fault == "cold_cost":
        configured["workaware_profile"]["models"][0]["cold_cost_samples_s"] = [-1.]
    elif fault == "rate":
        configured["workaware_profile"]["models"][0]["rate_rps"] = True
    elif fault == "static_fit":
        fit["performance_cost_tables_s"] = [[0.]]
    elif fault == "static_profile":
        calibration["joint_performance_costs_s"] = [[0.]]
    elif fault == "boolean_fit":
        fit["baseline_cadence_s"] = True
    else:
        configured["workaware_profile"]["models"][0]["tokens"] = True
    with pytest.raises(ValueError):
        modules[0].plan_workaware_robust_rebalance(observed, budgets, fit, extrapolation,
                                                  configured, lookup, blocking, phase)


def test_missing_staged_work_support_cannot_silently_drop_costs(modules, monkeypatch):
    monkeypatch.setattr(modules[0].ilp_reference, "SUPPORTS_WORK_SEGMENTS", False)
    with pytest.raises(RuntimeError, match="never omit"):
        plan(modules)
