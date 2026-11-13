import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, default_dp2_params
from planner import DEFAULT_PARAMS, build_state, plan_admission, plan_return, plan_usage
from test_planner import CODE, make_observed
from trace_runner import POLICIES


def test_policy_ablation_switches_are_single_changes():
    baseline = POLICIES["ca_full_v2_g"]
    for name, changed in (("ca_admit_v2_g", "return"), ("ca_periodic_v2_g", "admission"),
                          ("ca_no_imbalance_v2_g", "imbalance_cost")):
        variant = POLICIES[name]
        assert {key for key in set(baseline) | set(variant) if key != "tag"
                and baseline.get(key) != variant.get(key)} == {changed}
        assert variant["general"] and variant["fast"]
        assert not variant.get("isolate") and not variant.get("slice")


def test_cost_ablation_preserves_candidates_capacity_and_horizon():
    state = build_state(make_observed([6, 6], num_ranks=2), [13.0] * 2, request_growth={})
    model = StepModel(default_dp2_params())
    full = plan_admission(state, CODE, "ca", model, DEFAULT_PARAMS)
    ablated = plan_admission(state, CODE, "ca", model, dict(DEFAULT_PARAMS, score_imbalance_cost=False))
    assert full["moves"] and ablated["moves"]
    assert full["horizon_steps"] == ablated["horizon_steps"]
    assert full["candidates_evaluated"] == ablated["candidates_evaluated"]
    keys = ("target", "ordering", "moves", "performance_cost_s", "migration_cost_s")
    assert [{key: row[key] for key in keys} for row in full["alternatives"]] == [
        {key: row[key] for key in keys} for row in ablated["alternatives"]]
    assert all(row["objective_s"] == row["migration_cost_s"] and row["scored_performance_cost_s"] == 0
               for row in ablated["alternatives"])
    assert plan_usage(state, ablated["moves"], CODE, ablated["target"]) is not None


def test_cost_ablation_preserves_periodic_candidates_and_risk():
    state = build_state(make_observed([10, 2], num_ranks=2), [30.0] * 2, request_growth={})
    model = StepModel(default_dp2_params())
    full = plan_return(state, model, DEFAULT_PARAMS)
    ablated = plan_return(state, model, dict(DEFAULT_PARAMS, score_imbalance_cost=False))
    assert full["decisions"]
    assert full["horizon_steps"] == ablated["horizon_steps"]
    keys = ("tenant", "level", "ordering", "moves", "gain_s", "migration_cost_s", "headroom_risk_s", "used_after_gib")
    assert [{key: row[key] for key in keys} for row in full["decisions"]] == [
        {key: row[key] for key in keys} for row in ablated["decisions"]]
    assert not ablated["accepted"] and all(row["scored_gain_s"] == 0 for row in ablated["decisions"])


def test_no_cost_ablation_does_not_claim_unchanged_dp4_candidate_generation():
    state = build_state(make_observed([6] * 4), [20.0] * 4)
    params = dict(DEFAULT_PARAMS, score_imbalance_cost=False)
    with pytest.raises(ValueError, match="two attention groups"):
        plan_admission(state, CODE, "ca", StepModel(default_dp2_params()), params)
