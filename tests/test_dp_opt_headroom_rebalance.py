import asyncio
import copy
import importlib
import math
import time
from pathlib import Path

import pytest

from test_dp_opt_rebalance import controller_fake, fit, observations, solver_snapshot


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    pytest.importorskip("scipy.optimize")
    return tuple(importlib.import_module(name) for name in (
        "ilp_reference", "headroom_rebalance", "live_rebalance", "live_controller", "verify_rebalance"))


def profile(model=0, tokens=8):
    histories = [[], []]
    histories[model] = [{"request_id": "arrived", "model": model, "total_tokens": tokens,
                         "arrived_at": 0, "observed_at": 0}]
    return {"quantile": .95, "history_limit": 64, "arrivals": histories}


def geometry():
    return [[{"block_size": 1, "page_block_capacity": 1, "page_bytes": 1024**3,
              "uniform_page_geometry": True} for rank in (0, 1)] for model in (0, 1)]


@pytest.mark.parametrize("value", [[-1, 9], [10, 9], [math.nan, 9], [True, 9], [9]])
def test_final_cap_validation(modules, value):
    snapshot = solver_snapshot()
    snapshot["final_capacity_gib"] = value
    with pytest.raises(ValueError):
        modules[0].validate_snapshot(snapshot)


def test_final_cap_is_not_a_copy_peak_or_initial_cap(modules):
    snapshot = solver_snapshot()
    snapshot["final_capacity_gib"] = [6, 2]
    winner = modules[0].solve_candidate(snapshot, max_moves=4)
    assert winner["final_counts"][1] == [6, 2]
    assert len(winner["moves"]) == 2
    assert "objective_s" in modules[0].replay_candidate(snapshot, winner)
    rejected = modules[0].replay_candidate(snapshot, {"target": None, "moves": ["conv-0", "conv-1", "conv-2"]})
    assert rejected["status"] == "revalidation_headroom_rejected"


@pytest.mark.parametrize("model", [0, 1])
def test_headroom_is_tenant_role_independent(modules, model):
    snapshot = solver_snapshot()
    guards, scenarios = modules[1].headroom_scenarios(snapshot, geometry(), profile(model))
    assert guards == [{"model": model, "total_tokens": 8, "demand_gib": [8, 8], "reference_safe_ranks": [1]}]
    assert len(scenarios) == 2
    plans = [modules[0].enumerate_targets(scenario["snapshot"], max_moves=4)["winner"] for scenario in scenarios]
    feasible = [plan for plan in plans if plan is not None]
    assert min(feasible, key=lambda item: item["objective_s"])["final_counts"][1] == [7, 1]


def test_recovery_can_migrate_an_empty_corridor_to_other_rank(modules):
    snapshot = solver_snapshot(population=4)
    snapshot["capacity_gib"] = [4, 4]
    snapshot["migration_activation_cost_s"] = 0
    guards, scenarios = modules[1].headroom_scenarios(snapshot, geometry(), profile(tokens=4))
    assert guards[0]["reference_safe_ranks"] == [1]
    feasible = [modules[0].enumerate_targets(item["snapshot"], max_moves=4)["winner"] for item in scenarios]
    assert any(item is not None and item["final_gib"][0] == 0 for item in feasible)


def test_no_optimistic_headroom_if_even_original_layout_cannot_fit(modules):
    snapshot = solver_snapshot()
    guards, scenarios = modules[1].headroom_scenarios(snapshot, geometry(), profile(tokens=10))
    assert not guards and len(scenarios) == 1
    assert scenarios[0]["snapshot"]["final_capacity_gib"] == snapshot["capacity_gib"]


def test_quantile_uses_only_bounded_observed_lengths(modules):
    ledger = modules[1].HeadroomInferenceLedger(.01, 1, .95, 2)
    for index, tokens in enumerate((100, 5, 8)):
        ledger.arrival({"model": 0, "request_id": str(index), "prompt_tokens": tokens - 1,
                        "output_tokens": 1}, index, index)
    evidence = ledger.evidence([[{"requests": []}] * 2] * 2, 3)
    assert [item["total_tokens"] for item in evidence["headroom_profile"]["arrivals"][0]] == [5, 8]
    assert modules[1].headroom_scenarios(solver_snapshot(), geometry(), evidence["headroom_profile"])[0][0]["total_tokens"] == 8
    with pytest.raises(ValueError, match="past arrival"):
        ledger.arrival({"model": 1, "request_id": "future", "prompt_tokens": 1, "output_tokens": 1}, 5, 4)


def test_headroom_blocks_bad_restoration_without_pausing(modules, fit, tmp_path):
    async def scenario():
        observed = observations(4)
        now = time.monotonic()
        for request in observed[1][0]["requests"]:
            request["reserved_at"] = now - 1000
        ledger = modules[1].HeadroomInferenceLedger(.1, .1)
        ledger.arrival({"model": 0, "request_id": "beneficiary", "prompt_tokens": 10000, "output_tokens": 64}, now - 200, now - 200)
        ledger.admission({"request_id": "beneficiary", "admitted": True, "moves": [{"model": 1}]})
        ledger.completion({"request_id": "beneficiary", "reserved_at": now - 200, "completed_at": now - 10})
        controller = controller_fake(modules[3], fit, tmp_path, observed)
        report = await modules[2].rebalance_after_release(controller, [1, 1], balance_ledger=ledger)
        assert report["reason"] == "no_profitable_rebalance" and report["deferred"]
        assert report["before_pause"]["headroom_guards"][0]["model"] == 0
        controller.pause_boundary.assert_not_awaited()
        ledger.settle(report)
        assert ledger.eligible()
        modules[4].verify_rebalance(report, [1, 1], fit)
        bad = copy.deepcopy(report)
        bad["before_pause"]["headroom_guards"][0]["total_tokens"] += 1
        with pytest.raises(ValueError, match="Headroom guard"):
            modules[4].verify_rebalance(bad, [1, 1], fit)
    asyncio.run(scenario())


def test_future_arrival_in_planning_evidence_is_rejected(modules):
    marks = {"past": {"request_id": "past", "model": 0, "prompt_tokens": 5, "output_tokens": 1, "arrival_s": 2}}
    arrivals = [{"request_id": "past", "model": 0, "total_tokens": 6, "arrived_at": 102, "observed_at": 103}]
    decisions = [{"path": "release_rebalance_ilp", "moves": [], "deferred": True, "pause_call_s": 0,
                  "balance_evidence": {"completed_ids": [], "ready_epochs": [], "spent_blocking_request_s": 0,
                                       "pause_fixed_s": 1, "sampled_at": 102.5, "headroom_profile": profile(tokens=6)}}]
    decisions[0]["balance_evidence"]["headroom_profile"]["arrivals"] = [arrivals, []]
    with pytest.raises(ValueError, match="looked ahead"):
        modules[1].verify_headroom_history(decisions, {"past": {}}, {"move_fixed_s": 1}, 100, marks, arrivals)


def test_profitable_rebalance_preserves_physical_headroom_after_copy(modules, fit, tmp_path):
    async def scenario():
        observed = observations(4)
        now = time.monotonic()
        for request in observed[1][0]["requests"]:
            request["reserved_at"] = now - 1000
        ledger = modules[1].HeadroomInferenceLedger(.1, .01)
        ledger.arrival({"model": 0, "request_id": "beneficiary", "prompt_tokens": 5000, "output_tokens": 64}, now - 200, now - 200)
        ledger.admission({"request_id": "beneficiary", "admitted": True, "moves": [{"model": 1}]})
        ledger.completion({"request_id": "beneficiary", "reserved_at": now - 200, "completed_at": now - 10})
        controller = controller_fake(modules[3], fit, tmp_path, observed)
        report = await modules[2].rebalance_after_release(controller, [1, 1], pause_fixed_s=.01, balance_ledger=ledger)
        assert report["reason"] == "rebalanced" and len(report["moves"]) == 2
        assert [[len(rank["requests"]) for rank in ranks] for ranks in observed] == [[0, 0], [2, 2]]
        modules[4].verify_rebalance(report, [1, 1], fit)
        changed = copy.deepcopy(report)
        changed["after_pause"]["snapshot"]["final_capacity_gib"][0] += .01
        with pytest.raises(ValueError, match="safe-rank constraint"):
            modules[4].verify_rebalance(changed, [1, 1], fit)
    asyncio.run(scenario())


@pytest.mark.parametrize("mutation", [None, "source", "missing", "old_policy"])
def test_headroom_runtime_source_is_mandatory(modules, tmp_path, mutation):
    import json

    runner = importlib.import_module("live_trace")
    verifier = importlib.import_module("verify_trace")
    runner.archive_amortized_runtime(tmp_path, headroom=True)
    manifest = json.loads((tmp_path / "runtime-source.json").read_text())
    assert len(manifest["sources"]) == 11
    (tmp_path / "source.sha256").write_text("".join(
        f'{source["sha256"]}  {source["path"]}\n' for source in manifest["sources"].values()))
    if mutation == "source":
        (tmp_path / "runtime-source/headroom_rebalance.py").write_text("changed")
    elif mutation == "missing":
        manifest["sources"].pop("headroom_rebalance.py")
        (tmp_path / "runtime-source.json").write_text(json.dumps(manifest))
    elif mutation == "old_policy":
        manifest["policy"] = "kv_only_amortized"
        (tmp_path / "runtime-source.json").write_text(json.dumps(manifest))
    if mutation is None:
        verifier.verify_amortized_runtime(tmp_path, headroom=True)
    else:
        with pytest.raises(ValueError):
            verifier.verify_amortized_runtime(tmp_path, headroom=True)
