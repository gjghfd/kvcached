from contextlib import contextmanager

import workaware_robust_ilp
import workaware_robust_adapter
import workaware_robust_rebalance
from grouped_robust_uncertainty import allocate_grouped_uncertainty_variables, install_grouped_uncertainty_constraints
from prune_headroom_scenarios import prune_headroom_scenarios
from verify_solver_efficiency import verify_plan


ENABLED = False


@contextmanager
def exact_solver_efficiency():
    global ENABLED

    if ENABLED:
        raise RuntimeError("Install experimental solver efficiency only once before serving")
    original_allocate = workaware_robust_ilp.allocate_uncertainty_variables
    original_install = workaware_robust_ilp.install_uncertainty_constraints
    original_modes = workaware_robust_rebalance.headroom_modes
    original_verify = workaware_robust_adapter.verify_raw_plan

    def modes(*args, **kwargs):
        result = prune_headroom_scenarios(original_modes(*args, **kwargs))
        return [{**entry, "headroom_equivalence_proof": proof,
                 "original_scenario_count": result["proof"]["input_count"],
                 "pruned_scenario_count": result["proof"]["output_count"]}
                for entry, proof in zip(result["scenarios"], result["proof"]["groups"])]

    workaware_robust_ilp.allocate_uncertainty_variables = allocate_grouped_uncertainty_variables
    workaware_robust_ilp.install_uncertainty_constraints = install_grouped_uncertainty_constraints
    workaware_robust_rebalance.headroom_modes = modes
    workaware_robust_adapter.verify_raw_plan = verify_plan
    ENABLED = True
    try:
        yield
    finally:
        workaware_robust_ilp.allocate_uncertainty_variables = original_allocate
        workaware_robust_ilp.install_uncertainty_constraints = original_install
        workaware_robust_rebalance.headroom_modes = original_modes
        workaware_robust_adapter.verify_raw_plan = original_verify
        ENABLED = False
