import asyncio
import copy
import importlib
import time
from pathlib import Path

import pytest

from test_dp_opt_rebalance import controller_fake, observations


@pytest.fixture(autouse=True)
def controller_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    solver = importlib.import_module("ilp_reference")
    if not getattr(solver, "SUPPORTS_WORK_SEGMENTS", False):
        pytest.skip("Active frozen runtime has not yet been replaced; test in the isolated integration copy")
    return tuple(importlib.import_module(name) for name in (
        "live_controller", "live_rebalance", "workaware_rebalance", "verify_rebalance", "verify_workaware_rebalance"))


def measured_fit():
    return {"status": "raw_verified_workaware_cadence_fit", "capture_sizes": [1, 8, 16, 32, 64, 128, 256],
            "scope": "one_active_tenant_other_resident_idle", "source_verification": "raw_verified",
            "coefficients": {"base_s": .01, "graph_batch_s": .005, "attention_token_s": 1e-7},
            "residual_bound_s": 1e-6, "max_population": 9, "max_context_tokens": 32768}


def legacy_fit():
    return {"requests": 128, "raw_verification": {"status": "raw_verified"},
            "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": .025,
            "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": .3, "b": 0}}


def ledger_fixture(module, now):
    ledger = module.WorkawareInferenceLedger([.0005, .02], now - 1, .05)
    mark = {"request_id": "released", "model": 0, "prompt_tokens": 100000, "output_tokens": 64}
    ledger.arrival(mark, now - .02, now - .02)
    admission = {"admitted": True, "moves": [{"model": 1}], "request_id": "released", "arriving_model": 0,
                 "observations": [[{"requests": []}] * 2] * 2, "end_to_end_control_s": .01,
                 "path": "blocked_staged_ilp"}
    ledger.admission(admission, now - .015)
    completed = {"request_id": "released", "reserved_at": now - .01, "completed_at": now - .005}
    ledger.completion(completed, now - .004)
    return ledger


def test_real_rebalance_path_uses_new_gate_before_and_after_pause_without_past_credit(modules, tmp_path):
    controller_module, live, ledger_module, verifier, _ = modules
    now = time.monotonic()
    observed = observations(2)
    for request in observed[1][0]["requests"]:
        request.update({"computed_tokens": 4096, "reserved_at": now - .001})
    controller = controller_fake(controller_module, legacy_fit(), tmp_path, observed)
    ledger = ledger_fixture(ledger_module, now)
    result = asyncio.run(live.rebalance_after_release(controller, [10., 10.], balance_ledger=ledger,
                        workaware_fit=measured_fit(), pause_fixed_s=.05, move_bandwidth_gib_s=12))
    assert result["reason"] == "rebalanced" and len(result["moves"]) == 1
    assert result["settings"]["policy_mode"] == "workaware"
    assert result["settings"]["credit_request_s"] < 2 * result["settings"]["pause_fixed_s"]
    assert "migration_budget_s" not in result["after_pause"]["snapshot"]
    assert result["after_pause"]["gate"]["sunk_cost_excluded_from_continuation_profit"]
    assert result["after_pause"]["winner"]["final_counts"] == [[0, 0], [1, 1]]
    verifier.verify_rebalance(result, [10., 10.], legacy_fit())
    corrupt = copy.deepcopy(result)
    corrupt["after_pause"]["gate"]["accepted"] = False
    with pytest.raises(ValueError):
        verifier.verify_rebalance(corrupt, [10., 10.], legacy_fit())


def test_missing_calibration_cannot_silently_fall_back_to_N128(modules, tmp_path):
    controller_module, live, ledger_module, _, _ = modules
    now = time.monotonic()
    observed = observations(2)
    for request in observed[1][0]["requests"]:
        request.update({"computed_tokens": 4096, "reserved_at": now - .001})
    controller = controller_fake(controller_module, legacy_fit(), tmp_path, observed)
    with pytest.raises(ValueError, match="verified raw"):
        asyncio.run(live.rebalance_after_release(controller, [10., 10.], balance_ledger=ledger_fixture(ledger_module, now)))
    controller.pause_boundary.assert_not_called()


def test_outside_measured_domain_defers_without_pausing_inference(modules, tmp_path):
    controller_module, live, ledger_module, _, _ = modules
    now = time.monotonic()
    observed = observations(2)
    for request in observed[1][0]["requests"]:
        request.update({"computed_tokens": 100000, "reserved_at": now - .001})
    controller = controller_fake(controller_module, legacy_fit(), tmp_path, observed)
    result = asyncio.run(live.rebalance_after_release(controller, [10., 10.],
                        balance_ledger=ledger_fixture(ledger_module, now), workaware_fit=measured_fit()))
    assert result["reason"] == "outside_verified_workaware_calibration" and result["deferred"]
    controller.pause_boundary.assert_not_called()


def test_independent_history_rebuilds_counts_exposure_and_cold_cost_and_rejects_future_data(modules):
    _, _, ledger_module, _, verifier = modules
    config = {"workaware_rate_priors": [.0005, .02], "move_fixed_s": .05}
    ledger = ledger_module.WorkawareInferenceLedger(config["workaware_rate_priors"], 0, .05)
    mark = {"request_id": "code", "model": 0, "prompt_tokens": 100000, "output_tokens": 64, "arrival_s": 1.}
    arrival = ledger.arrival(mark, 1., 1.)
    admission = {"path": "blocked_staged_ilp", "admitted": True, "moves": [{"model": 1}], "request_id": "code",
                 "arriving_model": 0, "observations": [[{"requests": []}] * 2] * 2, "end_to_end_control_s": .05}
    ledger.admission(admission, 2.)
    completed = {"request_id": "code", "reserved_at": 2., "completed_at": 4.}
    ledger.completion(completed, 4.1)
    empty = [[{"requests": []}] * 2] * 2
    decision = {"path": "release_rebalance_ilp", "observations": empty,
                "balance_evidence": ledger.evidence(empty, 5.), "active_blocking_upper_request_s": 0,
                "deferred": True}
    ledger.settle(decision, 6.)
    verifier.verify_history([admission, decision], {"code": completed}, [arrival], config, 0, {"code": mark})
    corrupt = copy.deepcopy(decision)
    corrupt["balance_evidence"]["workaware_profile"]["models"][0]["observed_arrivals"] += 1
    with pytest.raises(ValueError, match="future or unobserved"):
        verifier.verify_history([admission, corrupt], {"code": completed}, [arrival], config, 0, {"code": mark})
