import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workaware_rebalance")


def mark(identifier, model=0):
    return {"request_id": identifier, "model": model, "prompt_tokens": 100000, "output_tokens": 64}


def empty():
    return [[{"requests": []}, {"requests": []}], [{"requests": []}, {"requests": []}]]


def test_rate_uses_exposure_not_only_short_gap_between_two_code_requests(module):
    ledger = module.WorkawareInferenceLedger([.0005, .02], 0, 1.8)
    ledger.arrival(mark("first"), 2400, 2400)
    ledger.arrival(mark("second"), 2600, 2600)
    profile = ledger.evidence(empty(), 2600)["workaware_profile"]
    assert profile["models"][0]["rate_rps"] == pytest.approx(4 / 6600)
    assert profile["models"][0]["rate_rps"] < .001
    assert profile["models"][0]["tokens"] == 100064


def test_empty_history_does_not_take_future_request_size_from_rate_prior(module):
    ledger = module.WorkawareInferenceLedger([.0005, .02], 0, 1.8)
    assert ledger.evidence(empty(), 100)["workaware_profile"]["models"][0]["tokens"] == 0


def test_bounded_quantile_history_keeps_cumulative_rate_count(module):
    ledger = module.WorkawareInferenceLedger([.0005, .02], 0, 1.8, history_limit=2)
    for position in range(3):
        ledger.arrival(mark(str(position)), position + 1, position + 1)
    assert len(ledger.arrivals[0]) == 2 and ledger.arrival_counts[0] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.arrival(mark("0"), 4, 4)


def test_future_arrival_and_pre_origin_exposure_are_rejected(module):
    ledger = module.WorkawareInferenceLedger([.0005, .02], 10, 1.8)
    with pytest.raises(ValueError, match="pre-origin"):
        ledger.arrival(mark("old"), 9, 10)
    ledger.arrival(mark("future"), 20, 20)
    with pytest.raises(ValueError, match="future"):
        ledger.evidence(empty(), 19)


def test_profitable_return_can_execute_before_past_credit_accumulates(module):
    reference = {"objective_s": 100, "moves": []}
    winner = {"objective_s": 20, "moves": ["conv"]}
    result = module.workaware_gate(reference, winner, {"min_gain_s": 1}, 2, 0, "before_pause")
    assert result["accepted"] and result["predicted_net_gain_request_s"] == 80
    assert result["migration_budget_policy"] == "net_gain_with_uncertainty_not_past_credit_hard_cap"


@pytest.mark.parametrize("winner,uncertainty", [({"objective_s": 99.5, "moves": ["conv"]}, 0),
                                             ({"objective_s": 90, "moves": ["conv"]}, 10),
                                             ({"objective_s": 20, "moves": []}, 0)])
def test_return_requires_moves_and_gain_beyond_uncertainty(module, winner, uncertainty):
    result = module.workaware_gate({"objective_s": 100}, winner, {"min_gain_s": 1}, uncertainty, 0, "before_pause")
    assert not result["accepted"]


def test_post_pause_profit_does_not_recharge_sunk_cost_but_history_is_preserved(module):
    result = module.workaware_gate({"objective_s": 100}, {"objective_s": 20, "moves": ["conv"]},
                                  {"min_gain_s": 1}, 2, 50, "after_pause")
    assert result["accepted"] and result["measured_blocking_so_far_request_s"] == 50
    assert result["sunk_cost_excluded_from_continuation_profit"]


def test_unintegrated_solver_cannot_silently_ignore_new_performance_objective(module, monkeypatch):
    monkeypatch.setattr(module.ilp_reference, "SUPPORTS_WORK_SEGMENTS", False, raising=False)
    with pytest.raises(RuntimeError, match="never omit"):
        module.plan_workaware_rebalance([], [], {}, True, {}, None, 0, "before_pause")
