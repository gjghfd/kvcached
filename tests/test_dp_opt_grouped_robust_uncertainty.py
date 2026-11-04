import importlib
import math

import pytest

from grouped_robust_uncertainty import (allocate_grouped_uncertainty_variables, concentrated_equivalence_key,
                                        install_grouped_uncertainty_constraints)


def snapshot(population=3, source=0, uncertainty=1.):
    identifiers = [f"request-{index}" for index in range(population)]
    counts = [population, 0] if source == 0 else [0, population]
    return {"counts": [[0, 0], counts], "new_request_count": 0, "new_gib": 0,
            "initial_gib": [.125 * counts[0], .125 * counts[1]], "capacity_gib": [1, 1],
            "performance_cost_tables_s": [[0], [0.] * (population + 1)],
            "movable": [{"id": identifier, "model": 1, "source": source,
                         "memory_gib": .125, "move_cost_s": 1.} for identifier in identifiers],
            "work_segments": [{"model": 1, "population": population, "start_step": step,
                "end_step": step + 1, "steps": 1, "initial_counts": counts,
                "request_sources": dict.fromkeys(identifiers, source),
                "attention_weights": {identifier: 10. + position * 3 + step
                                      for position, identifier in enumerate(identifiers)},
                "graph_costs_s": [0.] * (population + 1), "attention_difference_cost_s": .1,
                "uncertainty_request_s": uncertainty + step} for step in range(3)]}


def test_concentrated_segments_share_one_exact_uncertainty_indicator():
    state = snapshot()
    variables, dimensions = allocate_grouped_uncertainty_variables(state, 10)
    assert dimensions == 11
    assert all(positions["changed"] == 10 for positions in variables)
    assert all(concentrated_equivalence_key(segment) is not None for segment in state["work_segments"])


def test_mixed_sources_asymmetric_graph_and_tiny_weights_use_legacy_contract():
    state = snapshot()
    state["work_segments"][0]["request_sources"]["request-0"] = 1
    state["work_segments"][1]["graph_costs_s"][-1] = 1.
    state["work_segments"][2]["attention_weights"]["request-0"] = 1e-9
    assert all(concentrated_equivalence_key(segment) is None for segment in state["work_segments"])
    variables, dimensions = allocate_grouped_uncertainty_variables(state, 10)
    assert dimensions == 19
    assert all(len(positions["equivalent"]) == 2 for positions in variables)


def test_concentration_proof_requires_matching_initial_counts():
    state = snapshot()
    state["work_segments"][0]["initial_counts"] = [2, 1]
    assert concentrated_equivalence_key(state["work_segments"][0]) is None


@pytest.mark.parametrize("population", [1, 2, 3, 4])
@pytest.mark.parametrize("source", [0, 1])
@pytest.mark.parametrize("uncertainty", [0., .5, 1000.])
def test_same_robust_objective_as_exact_serial_copy_oracle(monkeypatch, population, source, uncertainty):
    module = importlib.import_module("workaware_robust_ilp")
    state = snapshot(population, source, uncertainty)
    oracle = module.enumerate_robust_reference(state, max_moves=population)
    monkeypatch.setattr(module, "allocate_uncertainty_variables", allocate_grouped_uncertainty_variables)
    monkeypatch.setattr(module, "install_uncertainty_constraints", install_grouped_uncertainty_constraints)
    result = module.solve_robust_candidate(state, max_moves=population, time_limit_s=2)
    assert result["status"] == "optimal"
    assert math.isclose(result["robust_objective_request_s"],
                        oracle["winner"]["robust_objective_request_s"], rel_tol=1e-8, abs_tol=1e-6)
    replay = module.robust_replay(state, result)
    assert result["uncertainty_request_s"] == replay["uncertainty_request_s"]
