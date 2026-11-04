import copy
import json

import pytest

import verify_solver_efficiency
import verify_workaware_robust_rebalance
import workaware_robust_adapter
import workaware_robust_ilp
from prune_headroom_scenarios import prune_headroom_scenarios
from solver_efficiency import exact_solver_efficiency
from test_dp_opt_workaware_robust_rebalance import legacy_fit, observations, settings


def scenarios():
    base = {"new_request_count": 0, "new_gib": 0, "capacity_gib": [10, 10],
            "initial_gib": [1, 0], "movable": [{"id": "request", "model": 1, "source": 0,
            "memory_gib": 1}], "work_segments": [{"uncertainty_request_s": 123.}]}
    return [{"snapshot": {**base, "final_capacity_gib": limits}, "risk_cost_s": risk, "guards": guards}
            for limits, risk, guards in (([8, 10], 1., "first"), ([10, 8], 0., "second"))]


def logged_scenarios(original):
    result = prune_headroom_scenarios(original)
    return [{**entry, "snapshot": {**entry["snapshot"], "placement_risk_cost_s": entry["risk_cost_s"]},
             "headroom_equivalence_proof": proof, "original_scenario_count": result["proof"]["input_count"],
             "pruned_scenario_count": result["proof"]["output_count"]}
            for entry, proof in zip(result["scenarios"], result["proof"]["groups"])]


def test_independent_certificate_keeps_minimum_risk_and_does_not_mutate_inputs():
    original = scenarios()
    logged = logged_scenarios(original)
    before = copy.deepcopy((original, logged))
    expected = verify_solver_efficiency.verify_certificates(original, logged)
    assert len(expected) == 1 and expected[0]["risk_cost_s"] == 0
    assert (original, logged) == before


@pytest.mark.parametrize("case", ["omitted", "duplicate", "risk", "bound", "capacity", "uncertainty",
                                 "binding", "new", "guards"])
def test_reject_tampered_or_incomplete_equivalence_proofs(case):
    original = scenarios()
    logged = logged_scenarios(original)
    if case == "omitted":
        logged[0]["headroom_equivalence_proof"]["original_indices"] = [1]
    elif case == "duplicate":
        logged[0]["headroom_equivalence_proof"]["original_indices"] = [0, 1, 1]
    elif case == "risk":
        logged[0]["risk_cost_s"] = 1.
    elif case == "bound":
        logged[0]["headroom_equivalence_proof"]["members"][0]["rank_proofs"][1]["exact_upper"]["numerator"] = 0
    elif case == "capacity":
        logged[0]["snapshot"]["capacity_gib"] = [11, 10]
    elif case == "uncertainty":
        logged[0]["snapshot"]["work_segments"][0]["uncertainty_request_s"] = 0.
    elif case == "binding":
        original[0]["snapshot"]["final_capacity_gib"] = [8, .5]
    elif case == "new":
        original[0]["snapshot"]["new_request_count"] = 1
    else:
        logged[0]["guards"] = "first"
    with pytest.raises(ValueError):
        verify_solver_efficiency.verify_certificates(original, logged)


@pytest.mark.parametrize("phase", ["before_pause", "after_pause"])
def test_full_controller_replay_is_independent_of_planner_solver_and_global_patching(monkeypatch, phase):
    import scipy.optimize

    observed, configured, fit = observations(), settings(), legacy_fit()
    configured["solver_limit_s"] = 1.
    with exact_solver_efficiency():
        check = workaware_robust_adapter.plan(observed, [1., 1.], fit, True, configured, None, 3., phase)
    if phase == "after_pause":
        workaware_robust_adapter.account_paused_controller_planning(check, check["planning_s"] + .001, 3.)
    check = json.loads(json.dumps(check))
    original = copy.deepcopy(check)
    backend_modes = verify_workaware_robust_rebalance.headroom_modes

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent certificate verification must not plan or solve")

    monkeypatch.setattr(scipy.optimize, "milp", forbidden)
    monkeypatch.setattr(workaware_robust_ilp, "robust_replay", forbidden)
    monkeypatch.setattr(workaware_robust_ilp, "solve_robust_candidate", forbidden)
    monkeypatch.setattr(workaware_robust_adapter, "plan", forbidden)
    monkeypatch.setattr("solver_efficiency.prune_headroom_scenarios", forbidden)
    with exact_solver_efficiency():
        result = workaware_robust_adapter.verify_controller_plan(
            check, observed, [1., 1.], fit, configured, phase, 3., check["planning_s"])
    assert result["exact_headroom_equivalence_verified"] is True
    assert result["global_optimality_verified"] is False
    assert result["planner_or_solver_invoked"] is False
    assert original == check
    assert verify_workaware_robust_rebalance.headroom_modes is backend_modes
