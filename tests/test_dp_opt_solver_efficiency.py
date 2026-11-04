import pytest

import solver_efficiency
import workaware_robust_ilp
import workaware_robust_adapter
import workaware_robust_rebalance


def test_explicit_context_restores_original_functions_even_after_failure():
    before = (workaware_robust_ilp.allocate_uncertainty_variables,
              workaware_robust_ilp.install_uncertainty_constraints, workaware_robust_rebalance.headroom_modes,
              workaware_robust_adapter.verify_raw_plan)
    with pytest.raises(ValueError, match="failure"):
        with solver_efficiency.exact_solver_efficiency():
            assert solver_efficiency.ENABLED
            assert workaware_robust_ilp.allocate_uncertainty_variables is not before[0]
            with pytest.raises(RuntimeError, match="only once"):
                with solver_efficiency.exact_solver_efficiency():
                    pass
            raise ValueError("failure")
    assert not solver_efficiency.ENABLED
    assert before == (workaware_robust_ilp.allocate_uncertainty_variables,
                      workaware_robust_ilp.install_uncertainty_constraints, workaware_robust_rebalance.headroom_modes,
                      workaware_robust_adapter.verify_raw_plan)


def test_pruned_modes_include_independent_equivalence_certificate(monkeypatch):
    base = {"new_request_count": 0, "new_gib": 0, "capacity_gib": [10, 10], "initial_gib": [1, 0],
            "movable": [{"id": "request", "model": 1, "source": 0, "memory_gib": 1}]}
    scenarios = [{"snapshot": {**base, "final_capacity_gib": [8, 10]}, "risk_cost_s": 1},
                 {"snapshot": {**base, "final_capacity_gib": [10, 8]}, "risk_cost_s": 0}]
    monkeypatch.setattr(workaware_robust_rebalance, "headroom_modes", lambda *args: scenarios)
    with solver_efficiency.exact_solver_efficiency():
        modes = workaware_robust_rebalance.headroom_modes()
        assert len(modes) == 1 and modes[0]["risk_cost_s"] == 0
        assert modes[0]["original_scenario_count"] == 2
        assert modes[0]["headroom_equivalence_proof"]["original_indices"] == [0, 1]
