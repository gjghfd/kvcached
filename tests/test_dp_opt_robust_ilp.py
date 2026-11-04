import copy
import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return (importlib.import_module("workaware_robust_ilp"), importlib.import_module("workaware_ilp"),
            importlib.import_module("workaware_cost"))


def segment(identifiers, weights, uncertainty, start=0):
    population = len(identifiers)
    return {"model": 1, "population": population, "start_step": start, "end_step": start + 1,
            "steps": 1, "initial_counts": [population, 0], "request_sources": dict.fromkeys(identifiers, 0),
            "attention_weights": dict(zip(identifiers, weights)), "graph_costs_s": [0.] * (population + 1),
            "attention_difference_cost_s": 1., "uncertainty_request_s": uncertainty}


def snapshot():
    identifiers = ["short", "long-a", "long-b"]
    return {"counts": [[0, 0], [3, 0]], "new_request_count": 0, "new_gib": 0,
            "initial_gib": [.375, 0], "capacity_gib": [1, 1], "performance_cost_tables_s": [[0], [0, 0, 0, 0]],
            "movable": [{"id": identifier, "model": 1, "source": 0, "memory_gib": .125, "move_cost_s": 1.}
                        for identifier in identifiers],
            "work_segments": [segment(identifiers, [10, 10, 10], 1),
                              segment(identifiers[1:], [20, 20], 100, 1)]}


def solve_relaxation(modules, state, charges, max_moves=None, robust=True, stages=1):
    candidate, work, _ = modules
    work.validate_work_snapshot(state)
    requests = state["movable"]
    move_dimensions = len(requests) * stages
    work_variables, dimensions = work.allocate_work_variables(state, move_dimensions)
    uncertainty_variables, dimensions = candidate.allocate_uncertainty_variables(state, dimensions)
    objective, upper, integrality = np.zeros(dimensions), np.ones(dimensions), np.zeros(dimensions)
    integrality[:move_dimensions] = 1
    objective[:move_dimensions] = np.repeat(charges, stages)
    rows, lower_bounds, upper_bounds = [], [], []

    def constraint(row, minimum=-np.inf, maximum=np.inf):
        rows.append(row)
        lower_bounds.append(minimum)
        upper_bounds.append(maximum)

    for index in range(len(requests)):
        row = np.zeros(dimensions)
        row[index * stages:(index + 1) * stages] = 1
        constraint(row, maximum=1)
    if max_moves is not None:
        row = np.zeros(dimensions)
        row[:move_dimensions] = 1
        constraint(row, maximum=max_moves)
    work.install_work_constraints(state, work_variables, requests, stages, dimensions,
                                  objective, upper, integrality, constraint)
    if robust:
        candidate.install_uncertainty_constraints(state, work_variables, uncertainty_variables, requests, stages,
                                                  dimensions, objective, integrality, constraint)
    result = milp(objective, integrality=integrality, bounds=Bounds(np.zeros(dimensions), upper),
                  constraints=LinearConstraint(np.array(rows), lower_bounds, upper_bounds), options={"mip_rel_gap": 0})
    assert result.success
    moved = [request["id"] for index, request in enumerate(requests)
             if sum(result.x[index * stages:(index + 1) * stages]) > .5]
    return result.fun, moved


@pytest.mark.parametrize("weights", [[10, 10, 10], [2, 7, 11], [4096, 8192, 16384]])
@pytest.mark.parametrize("uncertainty", [0., .01, 1., 100000.])
@pytest.mark.parametrize("stages", [1, 2])
def test_robust_ILP_matches_exhaustive_assignment_cost(modules, weights, uncertainty, stages):
    candidate, _, cost = modules
    state = snapshot()
    state["work_segments"] = [segment([request["id"] for request in state["movable"]], weights, uncertainty)]
    state["work_segments"][0]["graph_costs_s"] = [5., 0., 0., 5.]
    charges = [.25, .5, 2.]
    value, moved = solve_relaxation(modules, state, charges, stages=stages)
    possible = []
    for assignment in itertools.product((False, True), repeat=3):
        selected = [request["id"] for request, move in zip(state["movable"], assignment) if move]
        score = cost.evaluate_segments(state["work_segments"], selected)["performance_cost_s"]
        score += sum(charge for charge, move in zip(charges, assignment) if move)
        score += candidate.changed_uncertainty(state["work_segments"], selected)
        possible.append(score)
    assert value == pytest.approx(min(possible))
    recomputed = cost.evaluate_segments(state["work_segments"], moved)["performance_cost_s"]
    recomputed += sum(charge for request, charge in zip(state["movable"], charges) if request["id"] in moved)
    recomputed += candidate.changed_uncertainty(state["work_segments"], moved)
    assert value == pytest.approx(recomputed)


def test_robust_selection_finds_safe_alternative_instead_of_rejecting_point_winner(modules):
    candidate, _, _ = modules
    state = snapshot()
    point_value, point_moves = solve_relaxation(modules, state, [1., 1., 1.], max_moves=1, robust=False)
    robust_value, robust_moves = solve_relaxation(modules, state, [1., 1., 1.], max_moves=1)
    assert point_value == pytest.approx(11)
    assert point_moves != ["short"]
    assert 70 - point_value - candidate.changed_uncertainty(state["work_segments"], point_moves) < 0
    assert robust_moves == ["short"]
    assert robust_value == pytest.approx(52)
    assert 70 - robust_value > 1
    reference = candidate.enumerate_robust_reference(state, 1)
    assert reference["status"] == "complete_reference_enumeration"
    assert reference["winner"]["moves"] == robust_moves
    assert reference["winner"]["robust_objective_request_s"] == pytest.approx(robust_value)


def test_rank_swap_has_zero_uncertainty_but_same_graph_different_count_cost_does_not(modules):
    candidate, _, _ = modules
    state = snapshot()
    selected = [request["id"] for request in state["movable"]]
    assert candidate.changed_uncertainty(state["work_segments"], selected) == 0
    current = segment(selected, [2, 1, 1], 7.)
    current["graph_costs_s"] = [3, 0, 0, 3]
    assert candidate.changed_uncertainty([current], ["short"]) == 7


def test_reference_budget_cannot_claim_optimality(modules):
    candidate, _, _ = modules
    result = candidate.enumerate_robust_reference(snapshot(), 3, max_evaluations=1)
    assert result["status"] == "bounded_reference_not_proven"
    assert result["evaluations"] == 1


def test_serial_copy_peak_and_source_owner_replay_remain_required(modules):
    candidate, _, _ = modules
    state = {"counts": [[0, 0], [1, 1]], "new_request_count": 0, "new_gib": 0,
             "initial_gib": [.8, .8], "capacity_gib": [1, 1], "performance_cost_tables_s": [[0], [0, 0, 0]],
             "source_pages": [{"id": "small-page", "rank": 0, "owners": ["small"], "memory_gib": .1},
                              {"id": "large-page", "rank": 1, "owners": ["large"], "memory_gib": .3}],
             "movable": [{"id": "small", "model": 1, "source": 0, "memory_gib": .1, "move_cost_s": 0},
                         {"id": "large", "model": 1, "source": 1, "memory_gib": .3, "move_cost_s": 0}]}
    assert "objective_s" not in candidate.robust_replay(state, {"target": None, "moves": ["large", "small"]})
    assert candidate.robust_replay(state, {"target": None, "moves": ["small", "large"]})["peak_gib"] == pytest.approx([1., .9])
    state["source_pages"][0]["owners"].append("unmoved-owner")
    assert "objective_s" not in candidate.robust_replay(state, {"target": None, "moves": ["small", "large"]})


@pytest.mark.parametrize("maximum,budget", [(-1, 10), (True, 10), (1, 0), (1, True)])
def test_invalid_search_limits_fail_closed(modules, maximum, budget):
    with pytest.raises(ValueError):
        modules[0].enumerate_robust_reference(snapshot(), maximum, budget)


def test_outside_attention_domain_is_not_a_valid_placement(modules):
    candidate, _, _ = modules
    state = copy.deepcopy(snapshot())
    state["work_segments"][0]["attention_difference_limit_tokens"] = 1
    result = candidate.robust_replay(state, {"target": None, "moves": []})
    assert result["status"] == "outside_verified_work_domain"


@pytest.mark.parametrize("page_mode", ["private", "shared", "pinned", "reused"])
@pytest.mark.parametrize("max_moves", [0, 1, 2, 3])
def test_full_robust_ILP_matches_copy_peak_and_owner_reference(modules, page_mode, max_moves):
    candidate, _, _ = modules
    state = snapshot()
    state["capacity_gib"] = [.5, .5]
    state["final_capacity_gib"] = [.25, .5]
    state["migration_activation_cost_s"] = .75
    state["placement_risk_cost_s"] = 2.
    if page_mode != "private":
        state["source_pages"] = [
            {"id": "shared", "rank": 0, "pool": "source-pool", "owners": ["short", "long-a"], "memory_gib": .25},
            {"id": "single", "rank": 0, "pool": "private-pool", "owners": ["long-b"], "memory_gib": .125,
             "pinned": page_mode == "pinned"}]
        if page_mode == "reused":
            for request in state["movable"]:
                request["destination_pool"] = "source-pool"
    oracle = candidate.enumerate_robust_reference(state, max_moves)
    solved = candidate.solve_robust_candidate(state, max_moves=max_moves)
    if oracle["winner"] is None:
        assert solved["status"] == "infeasible"
    else:
        assert solved["status"] == "optimal"
        assert solved["robust_objective_request_s"] == pytest.approx(oracle["winner"]["robust_objective_request_s"])
        assert solved["robust_dual_bound_request_s"] == pytest.approx(solved["robust_objective_request_s"])
        assert solved["final_gib"][0] <= .25


def test_full_robust_ILP_recovers_alternative_without_changing_guard(modules):
    candidate, _, _ = modules
    solved = candidate.solve_robust_candidate(snapshot(), max_moves=1)
    assert solved["status"] == "optimal"
    assert solved["moves"] == ["short"]
    assert solved["uncertainty_request_s"] == 1
    assert solved["robust_objective_request_s"] == pytest.approx(52)


@pytest.mark.parametrize("fault", ["admission", "static", "joint", "budget"])
def test_unsupported_or_invalid_solver_scope_fails_closed(modules, fault):
    state = snapshot()
    if fault == "admission":
        state["new_request_count"] = 1
    elif fault == "static":
        state["performance_cost_tables_s"][1][0] = 1
    elif fault == "joint":
        del state["performance_cost_tables_s"]
        state["joint_performance_costs_s"] = [[0, 0, 0, 0]]
    with pytest.raises(ValueError):
        modules[0].solve_robust_candidate(state, time_limit_s=0 if fault == "budget" else 10)


def test_time_limited_incumbent_cannot_displace_better_verified_unchanged_layout(modules, monkeypatch):
    import scipy.optimize

    original = scipy.optimize.milp

    def expensive_incumbent(objective, **options):
        bounds = options["bounds"]
        lower, upper = np.array(bounds.lb), np.array(bounds.ub)
        lower[:3], upper[:3] = [0, 1, 0], [0, 1, 0]
        options["bounds"] = Bounds(lower, upper)
        result = original(objective, **options)
        assert result.success
        result.status, result.mip_dual_bound = 1, 0.
        return result

    monkeypatch.setattr(scipy.optimize, "milp", expensive_incumbent)
    solved = modules[0].solve_robust_candidate(snapshot(), max_moves=1)
    assert solved["status"] == "feasible_time_limited"
    assert solved["moves"] == []
    assert solved["candidate_source"] == "verified_unchanged_fallback"
    assert solved["robust_objective_request_s"] == 70
    assert solved["robust_mip_gap"] == 1


@pytest.mark.parametrize("uncertainty,activation", [(1000., 0.), (0., 1000.), (100., 1.)])
def test_dominance_bound_skips_solver_only_when_no_move_can_win(modules, monkeypatch, uncertainty, activation):
    import scipy.optimize

    candidate, _, _ = modules
    state = snapshot()
    for current in state["work_segments"]:
        current["uncertainty_request_s"] = uncertainty
    state["migration_activation_cost_s"] = activation
    oracle = candidate.enumerate_robust_reference(state, 3)
    assert oracle["winner"]["moves"] == []

    def forbidden_solver(*args, **kwargs):
        raise AssertionError("Dominated move candidates do not require an ILP call")

    monkeypatch.setattr(scipy.optimize, "milp", forbidden_solver)
    solved = candidate.solve_robust_candidate(state)
    assert solved["status"] == "optimal"
    assert solved["candidate_source"] == "unchanged_dominance_certificate"
    assert solved["robust_objective_request_s"] == pytest.approx(oracle["winner"]["robust_objective_request_s"])


def test_dominance_bound_cannot_skip_a_required_headroom_move(modules):
    candidate, _, _ = modules
    state = snapshot()
    for current in state["work_segments"]:
        current["uncertainty_request_s"] = 1000.
    state["final_capacity_gib"] = [.25, 1.]
    solved = candidate.solve_robust_candidate(state)
    assert solved["status"] == "optimal"
    assert solved["moves"]
    assert solved["candidate_source"] != "unchanged_dominance_certificate"


def test_dominance_bound_includes_feature_equivalence_numerical_slack(modules):
    candidate, _, cost = modules
    state = snapshot()
    state["work_segments"] = [segment([request["id"] for request in state["movable"]], [1e-7, 10, 10], 1e12)]
    state["work_segments"][0]["attention_difference_cost_s"] = 1e9
    assert candidate.changed_uncertainty(state["work_segments"], ["short"]) == 0
    before = cost.evaluate_segments(state["work_segments"], [])["performance_cost_s"]
    after = cost.evaluate_segments(state["work_segments"], ["short"])["performance_cost_s"]
    assert before - after > 1
    assert candidate.unchanged_dominance_bound(state, 1) >= before - after - 1
