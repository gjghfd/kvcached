import ast
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dp_opt_workaware_robust_rebalance import legacy_fit, observations, settings


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in (
        "verify_workaware_robust_rebalance", "workaware_robust_rebalance",
        "verify_workaware_rebalance", "workaware_robust_ilp", "ilp_reference"))


def make_plan(modules, observed=None, configured=None, budgets=None, phase="before_pause", blocking=0.):
    observed = observations() if observed is None else observed
    configured = settings() if configured is None else configured
    budgets = [1., 1.] if budgets is None else budgets
    check = modules[1].plan_workaware_robust_rebalance(
        observed, budgets, legacy_fit(), True, configured, None, blocking, phase)
    return check, observed, configured, budgets


def verify(modules, evidence, phase="before_pause", blocking=0., **options):
    check, observed, configured, budgets = evidence
    return modules[0].verify_plan(check, observed, budgets, legacy_fit(), configured, phase, blocking, **options)


def forbid_planning_and_solving(modules, monkeypatch):
    import scipy.optimize

    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not invoke a planner, robust solver or MILP")

    monkeypatch.setattr(modules[1], "plan_workaware_robust_rebalance", forbidden)
    monkeypatch.setattr(modules[1], "_validate_inputs", forbidden)
    monkeypatch.setattr(modules[1], "solve_robust_candidate", forbidden)
    monkeypatch.setattr(modules[3], "solve_robust_candidate", forbidden)
    monkeypatch.setattr(modules[3], "robust_replay", forbidden)
    monkeypatch.setattr(modules[3], "enumerate_robust_reference", forbidden)
    monkeypatch.setattr(modules[4], "solve_candidate", forbidden)
    monkeypatch.setattr(modules[4], "enumerate_targets", forbidden)
    monkeypatch.setattr(scipy.optimize, "milp", forbidden)


@pytest.mark.parametrize("phase", ("before_pause", "after_pause"))
@pytest.mark.parametrize("model", (0, 1))
@pytest.mark.parametrize("page_mode", ("private", "shared", "pinned"))
def test_real_planner_plans_verify_without_solver_or_input_mutation(modules, monkeypatch, phase, model, page_mode):
    evidence = make_plan(modules, observations(model=model, page_mode=page_mode), phase=phase, blocking=7.)
    original = copy.deepcopy(evidence)
    forbid_planning_and_solving(modules, monkeypatch)
    result = verify(modules, evidence, phase, 7.)
    assert evidence == original
    assert result["accepted"] == evidence[0]["gate"]["accepted"]
    assert result["global_optimality_verified"] is False
    assert result["actual_walltime_verified"] is False
    assert result["solver_wall_deadline_verified"] is False


def test_json_journal_roundtrip_is_independently_replayed(modules, monkeypatch):
    evidence = make_plan(modules, phase="after_pause", blocking=3.)
    check, observed, configured, budgets = json.loads(json.dumps(evidence))
    forbid_planning_and_solving(modules, monkeypatch)
    assert verify(modules, (check, observed, configured, budgets), "after_pause", 3.)["accepted"]


def test_required_soft_headroom_move_has_no_fake_unchanged_reference(modules, monkeypatch):
    observed = observations([("only", 20, 256)])
    idle = observed[0][1]
    idle["pages"] = [{"id": identifier, "owners": [], "pinned": True, "memory_gib": .125}
                     for identifier in (1, 2)]
    idle["mapped_bytes"] = 2 * idle["page_bytes"]
    configured = settings()
    configured["workaware_fit"]["coefficients"]["attention_token_s"] = 0.
    configured["workaware_profile"]["models"][0].update(
        {"tokens": 1025, "rate_rps": 100., "cold_cost_samples_s": [10.]})
    evidence = make_plan(modules, observed, configured, [.25, .375])
    check = evidence[0]
    assert check["winner"]["moves"] == ["only"]
    assert any(entry["reference"]["status"] == "revalidation_headroom_rejected"
               for entry in check["plan"]["scenarios"])
    forbid_planning_and_solving(modules, monkeypatch)
    assert verify(modules, evidence)["accepted"]


def test_bidirectional_destination_pool_reuse_is_physically_replayed(modules, monkeypatch):
    observed = observations([("first", 3, 128), ("second", 3, 256)])
    observed[1][1] = observations([("other", 3, 160)], source=1)[1][1]
    configured = settings()
    configured["max_moves"] = 2
    evidence = make_plan(modules, observed, configured, [.375, .375])
    forbid_planning_and_solving(modules, monkeypatch)
    assert verify(modules, evidence)["accepted"]


def cross_scenario_evidence(modules):
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
    return make_plan(modules, observed, configured, [.5, .375])


def test_error_changes_cross_scenario_winner_and_old_point_verifier_rejects(modules, monkeypatch):
    evidence = cross_scenario_evidence(modules)
    check, observed, configured, budgets = evidence
    assert check["winner"]["moves"] == ["short"]
    assert check["gate"]["accepted"]
    point_winner = min((entry["plan"]["winner"] for entry in check["plan"]["scenarios"]
                        if entry["plan"]["winner"] is not None), key=lambda candidate: candidate["objective_s"])
    assert point_winner["moves"] == ["long-a"]
    assert point_winner["objective_s"] < check["winner"]["objective_s"]
    assert point_winner["robust_objective_request_s"] > check["winner"]["robust_objective_request_s"]
    forbid_planning_and_solving(modules, monkeypatch)
    assert verify(modules, evidence)["accepted"]
    with pytest.raises(ValueError, match="does not minimize"):
        modules[2].verify_plan(check, observed, budgets, legacy_fit(), configured, "before_pause", 0.)


def test_point_cheaper_but_robust_worse_winner_is_rejected(modules):
    evidence = cross_scenario_evidence(modules)
    check = evidence[0]
    entry = min((entry for entry in check["plan"]["scenarios"] if entry["plan"]["winner"] is not None),
                key=lambda scenario: scenario["plan"]["winner"]["objective_s"])
    check["winner"] = copy.deepcopy(entry["plan"]["winner"])
    check["snapshot"] = copy.deepcopy(entry["snapshot"])
    check["plan"]["winner"] = check["winner"]
    check["robust_objective_request_s"] = check["winner"]["robust_objective_request_s"]
    with pytest.raises(ValueError, match="cross-scenario"):
        verify(modules, evidence)


@pytest.mark.parametrize("field", ("objective_s", "performance_cost_s", "migration_cost_s",
                                   "migration_activation_cost_s", "uncertainty_request_s",
                                   "robust_objective_request_s"))
@pytest.mark.parametrize("location", ("winner", "reference", "solver_candidate"))
def test_tampered_point_copy_and_differential_error_costs_are_rejected(modules, field, location):
    evidence = make_plan(modules)
    check = evidence[0]
    selected = (check[location] if location != "solver_candidate"
                else check["plan"]["scenarios"][0]["plan"]["candidates"][0])
    selected[field] += .25
    with pytest.raises(ValueError):
        verify(modules, evidence)


@pytest.mark.parametrize("fault", ("capacity", "initial", "owner", "request_copy", "risk", "guard",
                                   "work_segment", "horizon", "population", "peak", "released_pages",
                                   "final_counts", "reference_moves", "missing_scenario", "duplicate_scenario",
                                   "grant", "total_budget", "negative_grant", "boolean_grant", "zero_grant",
                                   "status", "global_status", "dual_bound", "gap"))
def test_tampered_capacity_ownership_forecast_grants_and_state_claims_are_rejected(modules, fault):
    configured = settings()
    configured["workaware_profile"]["models"][0].update({"tokens": 512, "rate_rps": .01})
    evidence = make_plan(modules, configured=configured)
    check = evidence[0]
    entry = check["plan"]["scenarios"][0]
    candidate = entry["plan"]["candidates"][0]
    if fault == "capacity":
        entry["snapshot"]["final_capacity_gib"][0] -= .1
    elif fault == "initial":
        check["unconstrained_snapshot"]["initial_gib"][0] -= .1
    elif fault == "owner":
        check["snapshot"]["source_pages"][0]["owners"] = []
    elif fault == "request_copy":
        entry["snapshot"]["movable"][0]["move_cost_s"] = 0.
    elif fault == "risk":
        entry["risk_cost_s"] += 1.
    elif fault == "guard":
        entry["guards"][0]["arrival_probability"] = 0.
    elif fault == "work_segment":
        check["work_segments"][0]["uncertainty_request_s"] = 0.
    elif fault == "horizon":
        check["horizon_steps"] += 1
    elif fault == "population":
        check["active_population"] += 1
    elif fault == "peak":
        check["winner"]["peak_gib"][0] += .125
    elif fault == "released_pages":
        check["winner"]["released_source_pages"] = []
    elif fault == "final_counts":
        check["winner"]["final_counts"] = [[0, 0], [2, 0]]
    elif fault == "reference_moves":
        check["reference"]["moves"] = ["first"]
    elif fault == "missing_scenario":
        check["plan"]["scenarios"].pop()
    elif fault == "duplicate_scenario":
        check["plan"]["scenarios"][1] = copy.deepcopy(entry)
    elif fault == "grant":
        entry["plan"]["solver_budget_s"] = configured["solver_limit_s"] + 1.
    elif fault == "total_budget":
        check["plan"]["shared_solver_limit_s"] += 1.
    elif fault == "negative_grant":
        entry["plan"]["solver_budget_s"] = -1.
    elif fault == "boolean_grant":
        entry["plan"]["solver_budget_s"] = True
    elif fault == "zero_grant":
        entry["plan"]["solver_budget_s"] = 0.
    elif fault == "status":
        entry["plan"]["status"] = "feasible_time_limited"
    elif fault == "global_status":
        check["plan"]["status"] = "globally_optimal"
    elif fault == "dual_bound":
        candidate["robust_dual_bound_request_s"] = candidate["robust_objective_request_s"] + 1.
    else:
        candidate["robust_mip_gap"] = 1.
    with pytest.raises(ValueError):
        verify(modules, evidence)


@pytest.mark.parametrize("field", ("accepted_false", "min_gain", "uncertainty", "phase", "paid_flag",
                                   "active_blocking", "measured_blocking", "sunk_flag"))
def test_gate_tampering_is_rejected(modules, field):
    evidence = make_plan(modules, phase="after_pause", blocking=11.)
    gate = evidence[0]["gate"]
    if field == "accepted_false":
        gate["accepted"] = False
    elif field == "min_gain":
        gate["min_gain_request_s"] = 0.
    elif field == "uncertainty":
        gate["uncertainty_request_s"] = 0.
    elif field == "phase":
        gate["phase"] = "before_pause"
    elif field == "paid_flag":
        gate["paid_planning_excluded_from_continuation_profit"] = False
    elif field == "active_blocking":
        gate["active_planning_blocking_request_s"] += 1.
    elif field == "measured_blocking":
        gate["measured_blocking_so_far_request_s"] += 1.
    else:
        gate["sunk_cost_excluded_from_continuation_profit"] = False
    with pytest.raises(ValueError):
        verify(modules, evidence, "after_pause", 11.)


@pytest.mark.parametrize("phase", ("before_pause", "after_pause"))
def test_long_paid_planning_is_observed_not_a_second_continuation_charge(modules, monkeypatch, phase):
    clock = [0.]
    monkeypatch.setattr(modules[1], "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = modules[1].solve_robust_candidate

    def slow_solver(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += 10000.
        return result

    monkeypatch.setattr(modules[1], "solve_robust_candidate", slow_solver)
    evidence = make_plan(modules, phase=phase, blocking=1000000.)
    check = evidence[0]
    assert check["planning_s"] == 10000.
    assert check["gate"]["accepted"]
    assert check["gate"]["active_planning_blocking_request_s"] == (20000. if phase == "after_pause" else 0.)
    forbid_planning_and_solving(modules, monkeypatch)
    result = verify(modules, evidence, phase, 1000000., controller_planning_upper_s=20000.)
    assert result["accepted"]
    assert result["actual_walltime_verified"] is False


def test_future_pause_and_copy_are_charged_but_paid_pause_is_sunk(modules):
    configured = settings()
    configured["pause_fixed_s"] = 1000.
    before = make_plan(modules, configured=configured)
    after = make_plan(modules, configured=configured, phase="after_pause", blocking=2000000.)
    assert not verify(modules, before)["accepted"]
    assert verify(modules, after, "after_pause", 2000000.)["accepted"]
    assert before[0]["snapshot"]["migration_activation_cost_s"] == 2000.
    assert after[0]["snapshot"]["migration_activation_cost_s"] == 0.
    configured["move_bandwidth_gib_s"] = .00001
    expensive_copy = make_plan(modules, configured=configured, phase="after_pause", blocking=2000000.)
    assert not verify(modules, expensive_copy, "after_pause", 2000000.)["accepted"]


def test_minimum_gain_cannot_be_bypassed_even_with_a_cheaper_robust_winner(modules):
    configured = settings()
    configured["min_gain_s"] = 1000.
    evidence = make_plan(modules, configured=configured)
    assert evidence[0]["winner"]["moves"]
    assert not verify(modules, evidence)["accepted"]
    evidence[0]["gate"]["accepted"] = True
    with pytest.raises(ValueError, match="continuation gate"):
        verify(modules, evidence)


@pytest.mark.parametrize("value", (True, -1., float("inf"), float("nan")))
def test_invalid_reported_wall_seconds_are_rejected(modules, value):
    evidence = make_plan(modules)
    evidence[0]["planning_s"] = value
    with pytest.raises(ValueError):
        verify(modules, evidence)


def test_single_plan_cannot_prove_actual_time_and_external_bound_only_checks_observation(modules):
    evidence = make_plan(modules, phase="after_pause", blocking=3.)
    check = evidence[0]
    check["planning_s"] = 1000000.
    check["gate"]["active_planning_blocking_request_s"] = 2000000.
    check["gate"]["measured_blocking_so_far_request_s"] = 2000003.
    report = verify(modules, evidence, "after_pause", 3.)
    assert report["accepted"]
    assert report["walltime_scope"] == "reported_planning_s_only_not_independent_walltime_proof"
    assert report["actual_walltime_verified"] is False
    with pytest.raises(ValueError, match="controller"):
        verify(modules, evidence, "after_pause", 3., controller_planning_upper_s=10.)
    assert verify(modules, evidence, "after_pause", 3., controller_planning_upper_s=2000000.)["accepted"]


@pytest.mark.parametrize("value", (True, -1., float("inf"), float("nan")))
def test_invalid_external_controller_time_bound_is_rejected(modules, value):
    evidence = make_plan(modules)
    with pytest.raises(ValueError, match="controller"):
        verify(modules, evidence, controller_planning_upper_s=value)


@pytest.mark.parametrize("fault", ("missing", "exception", "limited", "limited_unchanged", "limited_no_bound"))
def test_logged_solver_failures_and_time_limited_fallbacks_do_not_claim_global_optimality(
        modules, monkeypatch, fault):
    original = modules[1].solve_robust_candidate

    def boundary_solver(snapshot, max_moves, time_limit_s):
        if fault == "missing":
            return {"status": "no_verified_robust_incumbent"}
        if fault == "exception":
            raise TimeoutError("Synthetic CPU timeout")
        if fault == "limited_unchanged":
            candidate = modules[3].robust_replay(snapshot, {"target": None, "moves": [], "status": "unchanged_layout"})
            candidate["candidate_source"] = "verified_unchanged_fallback"
        else:
            candidate = original(snapshot, max_moves, time_limit_s)
        candidate.update({"status": "feasible_time_limited", "robust_dual_bound_request_s": 0., "robust_mip_gap": 1.})
        if fault == "limited_no_bound":
            candidate["robust_dual_bound_request_s"], candidate["robust_mip_gap"] = None, None
        return candidate

    monkeypatch.setattr(modules[1], "solve_robust_candidate", boundary_solver)
    evidence = make_plan(modules)
    forbid_planning_and_solving(modules, monkeypatch)
    report = verify(modules, evidence)
    assert report["global_optimality_verified"] is False
    assert report["accepted"] is (fault in ("limited", "limited_no_bound"))


def test_expensive_time_limited_incumbent_cannot_replace_verified_reference(modules, monkeypatch):
    def expensive_solver(snapshot, max_moves, time_limit_s):
        candidate = modules[3].robust_replay(
            snapshot, {"target": None, "moves": ["first", "second"], "status": "robust_ILP_candidate"})
        return {**candidate, "status": "feasible_time_limited", "candidate_source": "robust_ILP_incumbent",
                "robust_dual_bound_request_s": 0., "robust_mip_gap": 1.}

    monkeypatch.setattr(modules[1], "solve_robust_candidate", expensive_solver)
    configured = settings()
    configured["max_moves"] = 2
    evidence = make_plan(modules, configured=configured)
    check = evidence[0]
    assert check["winner"] == check["reference"]
    forbid_planning_and_solving(modules, monkeypatch)
    assert not verify(modules, evidence)["accepted"]
    entry = check["plan"]["scenarios"][0]
    expensive = copy.deepcopy(entry["plan"]["candidates"][0])
    expensive["source_candidate_status"] = "feasible_time_limited"
    entry["plan"]["winner"] = expensive
    with pytest.raises(ValueError, match="winner/fallback"):
        verify(modules, evidence)


def test_shared_budget_exhaustion_retains_verified_no_move_reference(modules, monkeypatch):
    clock, calls = [0.], []
    monkeypatch.setattr(modules[1], "time", SimpleNamespace(monotonic=lambda: clock[0]))

    def timed_out(snapshot, max_moves, time_limit_s):
        calls.append(time_limit_s)
        clock[0] += 2.25 if len(calls) == 1 else 1.
        return {"status": "no_verified_robust_incumbent"}

    monkeypatch.setattr(modules[1], "solve_robust_candidate", timed_out)
    configured = settings()
    configured["solver_limit_s"] = 3.
    configured["workaware_profile"]["models"][0].update({"rate_rps": .01, "tokens": 512})
    evidence = make_plan(modules, configured=configured)
    forbid_planning_and_solving(modules, monkeypatch)
    report = verify(modules, evidence)
    assert report["total_solver_grants_s"] == 1.75
    assert report["solver_statuses"][-1] == "shared_solver_budget_exhausted"
    assert not report["accepted"]


def test_unchanged_dominance_certificate_is_checked_without_solver(modules, monkeypatch):
    configured = settings()
    configured["workaware_fit"]["population_profiles"]["profiles"]["1:2"]["contrast_residual_bound_s"] = 100.
    evidence = make_plan(modules, configured=configured)
    candidate = evidence[0]["plan"]["scenarios"][0]["plan"]["candidates"][0]
    assert candidate["candidate_source"] == "unchanged_dominance_certificate"
    forbid_planning_and_solving(modules, monkeypatch)
    assert not verify(modules, evidence)["accepted"]
    candidate["upper_bound_move_gain_request_s"] -= 1.
    with pytest.raises(ValueError, match="dominance"):
        verify(modules, evidence)


@pytest.mark.parametrize("fault", ("legacy_only", "base_zero", "mixed", "owner", "static", "negative_profile",
                                   "boolean_profile", "nonuniform", "negative_blocking"))
def test_unverified_inputs_and_invalid_economic_units_fail_closed(modules, fault):
    evidence = make_plan(modules)
    check, observed, configured, budgets = evidence
    calibration = configured["workaware_fit"]
    blocking = 0.
    if fault == "legacy_only":
        del calibration["population_profiles"]
    elif fault == "base_zero":
        calibration["coefficients"]["base_s"] = 0.
    elif fault == "mixed":
        observed[0][0] = observations([("code", 3, 256)], model=0)[0][0]
    elif fault == "owner":
        observed[1][0]["pages"][0]["owners"] = []
    elif fault == "static":
        configured["joint_performance_costs_s"] = [[0.]]
    elif fault in ("negative_profile", "boolean_profile"):
        calibration["population_profiles"]["profiles"]["1:2"]["attention_token_s"] = (
            -1. if fault == "negative_profile" else True)
    elif fault == "nonuniform":
        observed[1][0]["uniform_page_geometry"] = False
    else:
        blocking = -1.
    with pytest.raises(ValueError):
        modules[0].verify_plan(check, observed, budgets, legacy_fit(), configured, "before_pause", blocking)


def test_verifier_has_no_planner_or_robust_solver_imports_or_calls(modules):
    tree = ast.parse(Path(modules[0].__file__).read_text())
    forbidden = {"workaware_robust_rebalance", "workaware_robust_ilp", "scipy", "numpy"}
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name.split(".")[0] in forbidden for name in imports)
