import asyncio
import copy
import importlib
import json
import time
from pathlib import Path

import pytest

from test_dp_opt_rebalance import controller_fake, fit, observations, solver_snapshot


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    pytest.importorskip("scipy.optimize")
    return tuple(importlib.import_module(name) for name in (
        "ilp_reference", "amortized_rebalance", "live_rebalance"))


@pytest.mark.parametrize("activation", [0, 3, 20])
def test_fixed_activation_is_charged_once(modules, activation):
    core = modules[0]
    snapshot = solver_snapshot()
    snapshot["migration_activation_cost_s"] = activation
    winner = core.solve_candidate(snapshot, max_moves=4)
    assert len(winner["moves"]) == 4
    assert winner["migration_activation_cost_s"] == activation
    assert winner["objective_s"] == pytest.approx(activation + .04)
    assert core.replay_candidate(snapshot, winner)["objective_s"] == winner["objective_s"]


def test_expensive_migration_keeps_original_layout(modules):
    snapshot = solver_snapshot()
    snapshot["migration_activation_cost_s"] = 100
    winner = modules[0].solve_candidate(snapshot, max_moves=4)
    assert not winner["moves"] and winner["migration_activation_cost_s"] == 0


def test_budget_selects_partial_not_perfect_balance(modules):
    snapshot = solver_snapshot()
    snapshot["performance_cost_tables_s"][1] = [80, 35, 10, 0, 0, 0, 10, 35, 80]
    for request in snapshot["movable"]:
        request["move_cost_s"] = 1
    snapshot.update(migration_activation_cost_s=30, migration_budget_s=32)
    winner = modules[0].solve_candidate(snapshot, max_moves=4)
    assert len(winner["moves"]) == 2 and winner["final_counts"][1] == [6, 2]
    assert winner["objective_s"] == 42
    rejected = modules[0].replay_candidate(snapshot, {"target": None, "moves": [request["id"] for request in snapshot["movable"][:3]]})
    assert rejected["status"] == "revalidation_migration_budget_rejected"


@pytest.mark.parametrize("field", ["migration_activation_cost_s", "migration_budget_s"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_cost_budget_validation(modules, field, value):
    snapshot = solver_snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError):
        modules[0].validate_snapshot(snapshot)


@pytest.mark.parametrize("model", [0, 1])
def test_trigger_does_not_depend_on_tenant_role(modules, model):
    ledger = modules[1].MigrationInferenceLedger(.01, 1)
    ledger.admission({"request_id": "beneficiary", "admitted": True, "moves": [{"model": model}]})
    ledger.completion({"request_id": "ordinary", "reserved_at": 0, "completed_at": 100})
    assert not ledger.eligible()
    ledger.completion({"request_id": "beneficiary", "reserved_at": 1, "completed_at": 2})
    assert ledger.eligible()
    assert ledger.evidence([[{"requests": []}] * 2] * 2, 101)["ready_epochs"] == [
        {"beneficiary": "beneficiary", "models": [model]}]


def test_credit_excludes_measured_stall_and_future_arrivals(modules):
    ledger = modules[1].MigrationInferenceLedger(.1, 1)
    ledger.completion({"request_id": "finished", "reserved_at": 0, "completed_at": 100})
    observed = [[{"requests": [{"id": "active", "reserved_at": 50}]}, {"requests": []}],
                [{"requests": []}, {"requests": []}]]
    assert ledger.evidence(observed, 150)["credit_request_s"] == 20
    ledger.settle({"deferred": True, "active_blocking_upper_request_s": 10})
    evidence = ledger.evidence(observed, 150)
    assert evidence["inference_service_request_s"] == 190 and evidence["credit_request_s"] == 9


def test_continuation_excludes_sunk_cost_but_not_credit(modules):
    settings = {"credit_request_s": 200, "pause_fixed_s": 100, "min_gain_s": 1}
    reference = {"performance_cost_s": 50}
    winner = {"moves": ["request"], "objective_s": 10, "migration_cost_s": 5,
              "migration_activation_cost_s": 0}
    gate = modules[1].amortized_gate(reference, winner, settings, 8, 100, "after_pause")
    assert gate["accepted"] and gate["fixed_control_request_s"] == 0
    assert not modules[1].amortized_gate(reference, winner, settings, 8, 198, "after_pause")["accepted"]


def test_deferral_requires_new_release_not_timer(modules):
    async def scenario():
        attempted = []
        async def attempt(causes):
            attempted.append(causes)
            return {"moves": [], "deferred": True}
        async def record(report):
            return None
        worker = modules[2].ReleaseRebalancer(attempt, record, retry_s=.01,
                                             cooldown_s=.01, retry_on_release_only=True)
        worker.notify({"request_id": "first"})
        await asyncio.sleep(.05)
        assert len(attempted) == 1
        worker.notify({"request_id": "second"})
        await asyncio.sleep(.05)
        assert len(attempted) == 2
        await worker.close()
    asyncio.run(scenario())


def live_amortized_decision(modules, fit, tmp_path, past_service=1000):
    async def scenario():
        observed = observations()
        started = time.monotonic()
        for ranks in observed:
            for rank in ranks:
                for request in rank["requests"]:
                    request["reserved_at"] = started - 10
        ledger = modules[1].MigrationInferenceLedger(.01, .01)
        admission = {"path": "blocked_staged_ilp", "request_id": "beneficiary", "admitted": True,
                     "moves": [{"model": 1}]}
        ledger.admission(admission)
        released = {"request_id": "beneficiary", "model": 0,
                    "reserved_at": started - past_service - 1, "completed_at": started - 1}
        ledger.completion(released)
        controller = controller_fake(importlib.import_module("live_controller"), fit, tmp_path, observed)
        decision = await modules[2].rebalance_after_release(controller, [2, 2], pause_fixed_s=.01,
                                                           min_gain_s=0, move_bandwidth_gib_s=100,
                                                           balance_ledger=ledger)
        decision["release_events"] = [{name: released[name] for name in ("request_id", "model", "completed_at")}]
        return ledger, admission, released, decision, controller
    return asyncio.run(scenario())


def test_live_amortized_migration_passes_independent_replay(modules, fit, tmp_path):
    ledger, admission, released, decision, controller = live_amortized_decision(modules, fit, tmp_path)
    verifier = importlib.import_module("verify_rebalance")
    verifier.verify_rebalance(decision, [2, 2], fit, requests={released["request_id"]: released})
    config = {"move_fixed_s": .01}
    modules[1].verify_amortized_history([admission, decision], config)
    assert decision["reason"] == "rebalanced" and len(decision["moves"]) == 4
    assert decision["after_pause"]["gate"]["sunk_cost_excluded_from_continuation_profit"]
    controller.resume.assert_awaited_once()
    ledger.settle(decision)
    assert not ledger.eligible() and ledger.spent_blocking_request_s > 0


@pytest.mark.parametrize("mutation", ["spent", "epochs", "pause", "charge", "resume"])
def test_amortized_history_rejects_fabricated_credit_and_debt(modules, fit, tmp_path, mutation):
    _, admission, _, decision, _ = live_amortized_decision(modules, fit, tmp_path)
    corrupted = copy.deepcopy(decision)
    if mutation == "spent":
        corrupted["balance_evidence"]["spent_blocking_request_s"] += 1
    elif mutation == "epochs":
        corrupted["balance_evidence"]["ready_epochs"][0]["models"] = [0]
    elif mutation == "pause":
        corrupted["balance_evidence"]["pause_fixed_s"] = 0
    elif mutation == "charge":
        corrupted["active_blocking_upper_request_s"] = 0
    else:
        corrupted["resume_completed_at"] += 1
    with pytest.raises(ValueError, match="history|receipts"):
        modules[1].verify_amortized_history([admission, corrupted], {"move_fixed_s": .01})


def test_recovery_credit_cannot_be_reused_after_paid_abort(modules):
    ledger = modules[1].MigrationInferenceLedger(.01, 1)
    ledger.admission({"request_id": "short", "admitted": True, "moves": [{"model": 1}]})
    ledger.completion({"request_id": "short", "reserved_at": 0, "completed_at": 1000})
    ledger.settle({"moves": [], "deferred": True, "active_blocking_upper_request_s": 4,
                   "observations": [[{"requests": [1, 2]}]],
                   "balance_evidence": {"ready_epochs": [{"beneficiary": "short"}]}})
    evidence = ledger.evidence([], 1000)
    assert evidence["credit_request_s"] == pytest.approx(5.96)
    assert evidence["pause_fixed_s"] == 2 and ledger.eligible()


@pytest.mark.parametrize("mutation", [None, "source", "launch", "missing"])
def test_amortized_runtime_source_archive_linked_to_launch_hash(modules, tmp_path, mutation):
    runner = importlib.import_module("live_trace")
    verifier = importlib.import_module("verify_trace")
    runner.archive_amortized_runtime(tmp_path)
    manifest = json.loads((tmp_path / "runtime-source.json").read_text())
    (tmp_path / "source.sha256").write_text("".join(
        f'{item["sha256"]}  {item["path"]}\n' for item in manifest["sources"].values()))
    if mutation == "source":
        (tmp_path / "runtime-source/amortized_rebalance.py").write_text("changed")
    elif mutation == "launch":
        (tmp_path / "source.sha256").write_text("")
    elif mutation == "missing":
        manifest["sources"].pop("amortized_rebalance.py")
        (tmp_path / "runtime-source.json").write_text(json.dumps(manifest))
    if mutation is None:
        verifier.verify_amortized_runtime(tmp_path)
    else:
        with pytest.raises(ValueError, match="source|SHA"):
            verifier.verify_amortized_runtime(tmp_path)
