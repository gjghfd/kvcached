import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_ilp"), importlib.import_module("workaware_cost")


def snapshot(cost):
    fit = {"status": "raw_verified_workaware_cadence_fit", "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
           "scope": "one_active_tenant_other_resident_idle", "source_verification": "raw_verified",
           "coefficients": {"base_s": .01, "graph_batch_s": .001, "attention_token_s": 1e-7},
           "residual_bound_s": 1e-5, "max_population": 9, "max_context_tokens": 32768}
    requests = [{"id": f"request-{index}", "movable": True, "max_tokens": 100 + remaining,
                 "output_tokens": 100, "computed_tokens": context}
                for index, (remaining, context) in enumerate(((12, 4096), (70, 8192), (100, 16384)))]
    observed = [[{"requests": []}, {"requests": []}], [{"requests": requests}, {"requests": []}]]
    return {"new_request_count": 0, "counts": [[0, 0], [3, 0]], "work_segments": cost.remaining_segments(observed, fit, 100),
            "movable": [{"id": request["id"], "model": 1, "source": 0} for request in requests], "placement_risk_cost_s": 2.}


def test_segment_ILP_matches_independent_all_assignment_enumeration(modules):
    helper, cost = modules
    state = snapshot(cost)
    helper.validate_work_snapshot(state)
    requests = state["movable"]
    variables, dimensions = helper.allocate_work_variables(state, len(requests))
    objective, upper, integrality = np.zeros(dimensions), np.ones(dimensions), np.zeros(dimensions)
    integrality[:len(requests)] = 1
    objective[:len(requests)] = [0.01, .2, .4]
    rows, lower_bounds, upper_bounds = [], [], []

    def constraint(row, minimum=-np.inf, maximum=np.inf):
        rows.append(row)
        lower_bounds.append(minimum)
        upper_bounds.append(maximum)

    helper.install_work_constraints(state, variables, requests, 1, dimensions, objective, upper, integrality, constraint)
    result = milp(objective, integrality=integrality, bounds=Bounds(np.zeros(dimensions), upper),
                  constraints=LinearConstraint(np.array(rows), lower_bounds, upper_bounds))
    assert result.success
    moved = [record["id"] for index, record in enumerate(requests) if result.x[index] > .5]
    expected = min(cost.evaluate_segments(state["work_segments"], [record["id"] for record, move in zip(requests, assignment) if move])["performance_cost_s"]
                   + sum(charge for charge, move in zip([.01, .2, .4], assignment) if move)
                   for assignment in itertools.product((False, True), repeat=len(requests)))
    assert result.fun == pytest.approx(expected)
    assert helper.work_cost(state, moved)["placement_risk_cost_s"] == 2.
    assert cost.evaluate_segments(state["work_segments"], moved)["performance_cost_s"] + sum(
        charge for record, charge in zip(requests, [.01, .2, .4]) if record["id"] in moved) == pytest.approx(expected)


@pytest.mark.parametrize("fault", ["risk", "incoming", "count", "weight", "source", "gap", "identity"])
def test_corrupt_forecasts_cannot_be_silently_optimized(modules, fault):
    helper, cost = modules
    state = snapshot(cost)
    if fault == "risk":
        state["placement_risk_cost_s"] = float("nan")
    elif fault == "incoming":
        state["new_request_count"] = 1
    elif fault == "count":
        state["work_segments"][0]["initial_counts"] = [1, 2]
    elif fault == "weight":
        state["work_segments"][0]["attention_weights"]["request-0"] = -1
    elif fault == "source":
        state["work_segments"][0]["request_sources"]["request-0"] = 1
    elif fault == "gap":
        state["work_segments"][1]["start_step"] += 1
    elif fault == "identity":
        state["movable"][0]["id"] = "absent"
    with pytest.raises(ValueError):
        helper.validate_work_snapshot(state)


def test_legacy_snapshots_do_not_get_new_variables_or_costs(modules):
    helper, _ = modules
    state = {"movable": []}
    helper.validate_work_snapshot(state)
    assert helper.allocate_work_variables(state, 42) == ([], 42)
    assert helper.work_cost(state, ["legacy-move"])["performance_cost_s"] == 0
