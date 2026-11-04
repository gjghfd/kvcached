import asyncio
import copy
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_dp_opt_workaware_robust_rebalance import legacy_fit, observations, settings


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return tuple(importlib.import_module(name) for name in (
        "live_rebalance", "verify_rebalance", "workaware_robust_adapter", "workaware_evidence"))


def ledger(configured):
    evidence = {"credit_request_s": 0., "pause_fixed_s": configured["pause_fixed_s"],
                "overhead_fraction": .01, "headroom_profile": {},
                "workaware_profile": configured["workaware_profile"], "ready_epochs": ["capacity"]}
    return SimpleNamespace(evidence=lambda observed, sampled: copy.deepcopy(evidence))


def controller_fake(observed):
    async def pause(current):
        reached = time.monotonic()
        return [[{"reached_at": reached, "armed_at": reached, "callback_at": reached,
                  "boundary_step_counter": 32, "rank": rank} for rank in (0, 1)] for model in (0, 1)]

    async def migrate(transaction, request, budgets):
        model, source = request["model"], request["source"]
        destination = 1 - source
        old, new = observed[model][source], observed[model][destination]
        record = next(record for record in old["requests"] if record["id"] == request["id"])
        page = next(page for page in old["pages"] if page["owners"] == [record["id"]])
        old["requests"].remove(record)
        old["pages"].remove(page)
        old["mapped_bytes"] -= old["page_bytes"]
        page["id"] = max([page["id"] for page in new["pages"]], default=0) + 1
        record["blocks"] = list(range(page["id"] * new["page_block_capacity"],
                                      page["id"] * new["page_block_capacity"] + len(record["blocks"])))
        new["requests"].append(record)
        new["pages"].append(page)
        new["mapped_bytes"] += new["page_bytes"]
        return {"request_id": record["id"], "model": model, "source": source, "destination": destination,
                "kv_bytes": len(record["blocks"]) * old["block_bytes"], "sha256": "a" * 64,
                "copy_peak_gib": [.5, .5], "elapsed_s": .01}

    return SimpleNamespace(lock=asyncio.Lock(), fit=legacy_fit(), allow_extrapolation=True,
        layout_lookup=None, admission_generation=0, model_admission_generations=[0, 0],
        admission_observations=None, snapshots=AsyncMock(side_effect=lambda **kwargs: copy.deepcopy(observed)),
        begin=AsyncMock(side_effect=lambda transaction: copy.deepcopy(observed)), pause_boundary=AsyncMock(side_effect=pause),
        resume=AsyncMock(), end=AsyncMock(), release_transfer_cache=AsyncMock(return_value=[]),
        migrate=AsyncMock(side_effect=migrate))


def run_return(module, observed=None, configured=None, optimizer="robust"):
    configured = settings() if configured is None else configured

    async def execute():
        controller = controller_fake(observations() if observed is None else observed)
        decision = await module.rebalance_after_release(controller, [1., 1.],
            max_moves=configured["max_moves"], candidate_limit=configured["candidate_limit"],
            move_bandwidth_gib_s=configured["move_bandwidth_gib_s"], solver_limit_s=configured["solver_limit_s"],
            pause_fixed_s=configured["pause_fixed_s"], min_gain_s=configured["min_gain_s"],
            balance_ledger=ledger(configured), workaware_fit=configured["workaware_fit"], workaware_optimizer=optimizer)
        return decision, controller

    return asyncio.run(execute())


def test_actual_controller_await_is_accounted_and_independently_verified(modules):
    decision, controller = run_return(modules[0])
    assert decision["reason"] == "rebalanced"
    assert decision["workaware_optimizer"] == "robust"
    assert len(decision["moves"]) == 1
    controller.resume.assert_awaited_once()
    fresh = decision["after_pause"]
    assert fresh["controller_adapter"]["controller_planning_s"] == decision["solve_s"]
    assert fresh["gate"]["active_planning_blocking_request_s"] == fresh["active_population"] * decision["solve_s"]
    assert fresh["controller_adapter"]["raw_planner_accounting"]["planning_s"] <= decision["solve_s"]
    modules[1].verify_rebalance(decision, [1., 1.], legacy_fit())


def test_point_default_preserves_unwrapped_historical_plans(modules):
    decision, controller = run_return(modules[0], optimizer="point")
    assert decision["reason"] == "rebalanced"
    assert "workaware_optimizer" not in decision["settings"]
    assert "controller_adapter" not in decision["before_pause"]
    modules[1].verify_rebalance(decision, [1., 1.], legacy_fit())
    controller.resume.assert_awaited_once()


@pytest.mark.parametrize("tamper", ["report_marker", "plan_marker", "settings_marker", "clock", "active_cost", "gain"])
def test_live_robust_records_cannot_fall_back_to_point_or_omit_paid_cost(modules, tamper):
    decision, _ = run_return(modules[0])
    if tamper == "report_marker":
        decision.pop("workaware_optimizer")
    elif tamper == "plan_marker":
        decision["before_pause"].pop("controller_adapter")
    elif tamper == "settings_marker":
        decision["settings"].pop("workaware_optimizer")
    elif tamper == "clock":
        decision["solve_s"] += 1.
    elif tamper == "active_cost":
        decision["after_pause"]["gate"]["active_planning_blocking_request_s"] = 0.
    else:
        decision["after_pause"]["gate"]["predicted_net_gain_request_s"] += 1.
    with pytest.raises(ValueError):
        modules[1].verify_rebalance(decision, [1., 1.], legacy_fit())


@pytest.mark.parametrize("scope", ["mixed_active", "legacy_profiles"])
def test_unsupported_robust_calibration_defers_without_pausing(modules, scope):
    configured = settings()
    observed = observations()
    if scope == "mixed_active":
        observed[0] = observations([("code-first", 30, 256), ("code-second", 30, 256)], model=0)[0]
    else:
        configured["workaware_fit"].pop("population_profiles")
    decision, controller = run_return(modules[0], observed, configured)
    assert decision["reason"] == "outside_verified_workaware_calibration"
    assert decision["deferred"]
    controller.pause_boundary.assert_not_awaited()


def test_unknown_optimizer_is_rejected_before_control_activity(modules):
    with pytest.raises(ValueError):
        run_return(modules[0], optimizer="unknown")


@pytest.mark.parametrize("optimizer", ["point", "robust"])
@pytest.mark.parametrize("phase_profiles", [False, True])
@pytest.mark.parametrize("random_baseline", [False, True])
def test_runtime_archive_proves_selected_optimizer_and_all_dependencies(modules, monkeypatch, tmp_path,
                                                                        optimizer, phase_profiles, random_baseline):
    evidence = modules[3]
    monkeypatch.setenv("DP_OPT_WORKAWARE_PHASE_PROFILES", "1" if phase_profiles else "0")
    monkeypatch.setenv("DP_OPT_WORKAWARE_PROFILES", "1")
    monkeypatch.setenv("DP_OPT_RANDOM_BASELINE", "1" if random_baseline else "0")
    evidence.archive_runtime(tmp_path, optimizer=optimizer)
    manifest = json.loads((tmp_path / "workaware-runtime-source.json").read_text())
    (tmp_path / "source.sha256").write_text("\n".join(
        source["sha256"] + "  " + source["path"] for source in manifest["sources"].values()))
    assert evidence.verify_runtime(tmp_path) == manifest
    assert manifest.get("workaware_optimizer", "point") == optimizer
    assert set(evidence.ROBUST_RUNTIME_FILES).issubset(manifest["sources"]) == (optimizer == "robust")
    if optimizer == "robust":
        manifest.pop("workaware_optimizer")
        (tmp_path / "workaware-runtime-source.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="incomplete"):
            evidence.verify_runtime(tmp_path)
