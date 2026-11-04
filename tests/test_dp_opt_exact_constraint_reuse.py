import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("ilp_reference"), importlib.import_module("workaware_ilp")


def owner_snapshot():
    return {
        "counts": [[0, 0], [2, 0]], "initial_gib": [1.0, 0.0],
        "capacity_gib": [1.0, 1.0], "new_gib": .3, "imbalance_cost_s": [.0001, .0001],
        "movable": [
            {"id": "first", "model": 1, "source": 0, "memory_gib": .4,
             "move_cost_s": .1, "destination_pool": "1:1"},
            {"id": "second", "model": 1, "source": 0, "memory_gib": .4,
             "move_cost_s": .2, "destination_pool": "1:1"}],
        "source_pages": [
            {"id": "first-0", "rank": 0, "pool": "1:0", "owners": ["first"], "memory_gib": .2},
            {"id": "first-1", "rank": 0, "pool": "1:0", "owners": ["first"], "memory_gib": .2},
            {"id": "second-0", "rank": 0, "pool": "1:0", "owners": ["second"], "memory_gib": .2},
            {"id": "second-1", "rank": 0, "pool": "1:0", "owners": ["second"], "memory_gib": .2},
            {"id": "pinned", "rank": 0, "pool": "1:0", "owners": [], "memory_gib": .2, "pinned": True}]}


def capture_dimensions(monkeypatch):
    optimize = pytest.importorskip("scipy.optimize")
    original = optimize.milp
    recorded = []

    def solve(*args, **kwargs):
        recorded.append(len(args[0] if args else kwargs["c"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize, "milp", solve)
    return recorded


def test_shared_owner_pages_reuse_release_variables_without_dropping_bytes(helpers, monkeypatch):
    reference, _ = helpers
    recorded = capture_dimensions(monkeypatch)
    state = owner_snapshot()
    candidate = reference.solve_candidate(state, 0, max_moves=1, time_limit_s=2)
    assert recorded == [10]
    assert candidate["status"] == "optimal"
    assert candidate["moves"] == ["first"]
    assert candidate["final_gib"] == pytest.approx([.9, .4])
    assert set(candidate["released_source_pages"]) == {"first-0", "first-1"}


@pytest.mark.parametrize("extra", [
    {"pool": "different-pool", "pinned": False},
    {"pool": "1:0", "pinned": True}])
def test_pool_and_pin_scope_do_not_share_release_variables(helpers, monkeypatch, extra):
    reference, _ = helpers
    recorded = capture_dimensions(monkeypatch)
    state = owner_snapshot()
    state["initial_gib"][0] = 1.1
    state["capacity_gib"][0] = 1.1
    state["source_pages"].append({"id": "extra", "rank": 0, "owners": ["first"],
                                  "memory_gib": .1, **extra})
    candidate = reference.solve_candidate(state, 0, max_moves=1, time_limit_s=2)
    assert recorded == [12]
    assert candidate["status"] == "optimal"
    assert ("extra" in candidate["released_source_pages"]) is not extra["pinned"]


def segment(start, end, sources):
    return {"model": 1, "population": len(sources), "initial_counts": [len(sources), 0],
            "request_sources": dict(sources), "start_step": start, "end_step": end,
            "steps": end - start, "graph_costs_s": [1., 0., 1.],
            "attention_weights": {identifier: 100. + start for identifier in sources},
            "attention_difference_cost_s": .01}


def test_identical_membership_reuses_counts_but_not_attention_variables(helpers):
    _, work = helpers
    segments = [segment(0, 10, {"first": 0, "second": 0}),
                segment(10, 20, {"first": 0, "second": 0})]
    variables, dimensions = work.allocate_work_variables({"work_segments": segments}, 2)
    assert variables[0]["counts"] == variables[1]["counts"]
    assert variables[0]["difference"] != variables[1]["difference"]
    assert dimensions == 7
    objective = np.zeros(dimensions)
    upper, integrality = np.ones(dimensions), np.zeros(dimensions)
    rows = []
    requests = [{"id": "first"}, {"id": "second"}]
    work.install_work_constraints({"work_segments": segments}, variables, requests, 1,
                                  dimensions, objective, upper, integrality,
                                  lambda row, **kwargs: rows.append(row))
    assert objective[variables[0]["counts"]].tolist() == [2., 0., 2.]
    assert len(rows) == 6


def test_same_population_different_membership_does_not_reuse_counts(helpers):
    _, work = helpers
    first = segment(0, 10, {"first": 0, "second": 0})
    second = segment(10, 20, {"first": 0, "third": 0})
    variables, dimensions = work.allocate_work_variables({"work_segments": [first, second]}, 2)
    assert set(variables[0]["counts"]).isdisjoint(variables[1]["counts"])
    assert dimensions == 10


def test_weighted_graph_objective_matches_unshared_segment_evaluation(helpers):
    _, work = helpers
    segments = [segment(0, 10, {"first": 0, "second": 0}),
                segment(10, 20, {"first": 0, "second": 0})]
    segments[1]["graph_costs_s"] = [3., .5, 3.]
    original = copy.deepcopy(segments)
    variables, dimensions = work.allocate_work_variables({"work_segments": segments}, 2)
    objective = np.zeros(dimensions)
    upper, integrality = np.ones(dimensions), np.zeros(dimensions)
    work.install_work_constraints({"work_segments": segments}, variables,
                                  [{"id": "first"}, {"id": "second"}], 1,
                                  dimensions, objective, upper, integrality,
                                  lambda row, **kwargs: None)
    assert objective[variables[0]["counts"]].tolist() == [4., .5, 4.]
    assert segments == original
