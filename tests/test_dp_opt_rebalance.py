import asyncio
import copy
import importlib
import itertools
import random
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    pytest.importorskip("scipy.optimize")
    return (importlib.import_module("ilp_reference"), importlib.import_module("live_rebalance"),
            importlib.import_module("live_controller"), importlib.import_module("verify_rebalance"))


@pytest.fixture
def fit():
    return {"requests": 128, "raw_verification": {"status": "raw_verified"},
            "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": .025,
            "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": .3, "b": 0}}


def solver_snapshot(population=8):
    return {"counts": [[0, 0], [population, 0]], "new_request_count": 0, "new_gib": 0,
            "initial_gib": [population, 0], "capacity_gib": [population + 1, population + 1],
            "movable": [{"id": f"conv-{index}", "source": 0, "model": 1,
                         "memory_gib": 1, "move_cost_s": .01} for index in range(population)],
            "performance_cost_tables_s": [[0], [10 * abs(2 * rank_zero - population)
                                                for rank_zero in range(population + 1)]]}


def observations(population=8):
    result = [[{"requests": [], "pages": [], "mapped_bytes": 0,
                "uniform_page_geometry": True, "page_block_capacity": 64,
                "page_bytes": 96 * 1024**2, "block_size": 16, "block_bytes": 1536 * 1024,
                "mapped_free_block_slots": 0, "free_blocks": 10000} for rank in (0, 1)] for model in (0, 1)]
    for index in range(population):
        result[1][0]["requests"].append({"id": f"conv-{index}", "blocks": list(range((index + 1) * 64, (index + 1) * 64 + 32)),
                                          "movable": True, "max_tokens": 1100, "output_tokens": 100,
                                          "prompt_tokens": 16})
        result[1][0]["pages"].append({"id": index + 1, "owners": [f"conv-{index}"],
                                       "pinned": False, "memory_gib": 96 / 1024})
    result[1][0]["mapped_bytes"] = population * 96 * 1024**2
    return result


def controller_fake(module, fit, tmp_path, observed):
    controller = module.PlacementController([], fit, tmp_path, allow_extrapolation=True)
    controller.snapshots = AsyncMock(side_effect=lambda **kwargs: copy.deepcopy(observed))
    controller.begin = AsyncMock(side_effect=lambda transaction: copy.deepcopy(observed))

    async def pause(current):
        reached = time.monotonic()
        return [[{"reached_at": reached, "armed_at": reached, "callback_at": reached,
                  "boundary_step_counter": 32, "rank": rank} for rank in (0, 1)] for model in (0, 1)]

    async def migrate(transaction, request, budgets):
        model, source = request["model"], request["source"]
        destination = 1 - source
        old, new = observed[model][source], observed[model][destination]
        record = next(record for record in old["requests"] if record["id"] == request["id"])
        old["requests"].remove(record)
        page = next(page for page in old["pages"] if page["owners"] == [record["id"]])
        old["pages"].remove(page)
        old["mapped_bytes"] -= 96 * 1024**2
        new_page = max([page["id"] for page in new["pages"]], default=0) + 1
        page["id"] = new_page
        record["blocks"] = list(range(new_page * 64, new_page * 64 + 32))
        new["requests"].append(record)
        new["pages"].append(page)
        new["mapped_bytes"] += 96 * 1024**2
        event = {"request_id": request["id"], "model": model, "source": source, "destination": destination,
                 "kv_bytes": 32 * 1536 * 1024, "sha256": "a" * 64, "copy_peak_gib": [.9, .9], "elapsed_s": .01}
        controller.events.append(event)
        return event

    controller.pause_boundary = AsyncMock(side_effect=pause)
    controller.resume = AsyncMock()
    controller.end = AsyncMock()
    controller.release_transfer_cache = AsyncMock(return_value=[])
    controller.migrate = AsyncMock(side_effect=migrate)
    return controller


def test_no_arrival_ilp_balances_without_phantom_request(modules):
    solver, _, _, _ = modules
    snapshot = solver_snapshot()
    report = solver.enumerate_targets(snapshot, max_moves=4)
    winner = report["winner"]
    assert len(report["candidates"]) == 1 and winner["target"] is None
    assert winner["final_counts"] == [[0, 0], [4, 4]] and len(winner["moves"]) == 4
    assert winner["admission_cost_s"] == 0 and winner["final_gib"] == [4, 4]
    assert solver.replay_candidate(snapshot, winner)["final_counts"] == winner["final_counts"]


@pytest.mark.parametrize("patch", [{"new_request_count": True}, {"new_request_count": 2},
                                   {"new_gib": 1}, {"admission_cost_s": [1, 0]}])
def test_no_arrival_rejects_invalid_incoming_state(modules, patch):
    snapshot = solver_snapshot()
    snapshot.update(patch)
    with pytest.raises(ValueError):
        modules[0].enumerate_targets(snapshot)


def test_no_arrival_requires_no_target(modules):
    with pytest.raises(ValueError, match="target=None"):
        modules[0].solve_candidate(solver_snapshot(), 0)


def test_no_arrival_costs_and_copy_peak_limit_migrations(modules):
    snapshot = solver_snapshot()
    snapshot["capacity_gib"][1] = 1.1
    winner = modules[0].enumerate_targets(snapshot, max_moves=4)["winner"]
    assert winner["final_counts"] == [[0, 0], [7, 1]]
    for request in snapshot["movable"]:
        request["move_cost_s"] = 1000
    assert modules[0].enumerate_targets(snapshot)["winner"]["moves"] == []


def test_no_arrival_shared_pinned_pages_remain_charged(modules):
    snapshot = solver_snapshot(2)
    snapshot["source_pages"] = [{"id": "shared", "rank": 0, "memory_gib": 1,
                                 "owners": ["conv-0", "conv-1"], "pinned": False},
                                {"id": "pinned", "rank": 0, "memory_gib": 1,
                                 "owners": [], "pinned": True}]
    winner = modules[0].enumerate_targets(snapshot, max_moves=1)["winner"]
    assert winner["final_counts"] == [[0, 0], [1, 1]]
    assert winner["final_gib"] == [2, 1] and winner["released_source_pages"] == []


def test_no_arrival_ilp_matches_small_exhaustive_layouts(modules):
    randomizer = random.Random(17)
    for _ in range(8):
        snapshot = solver_snapshot(4)
        for request in snapshot["movable"]:
            request["memory_gib"] = randomizer.choice([.5, 1])
            request["move_cost_s"] = randomizer.choice([1, 4, 30])
        snapshot["initial_gib"][0] = sum(request["memory_gib"] for request in snapshot["movable"])
        snapshot["capacity_gib"][1] = randomizer.choice([1, 2, 3])
        costs = []
        for count in range(3):
            for chosen in itertools.permutations(snapshot["movable"], count):
                destination = sum(request["memory_gib"] for request in chosen)
                if destination <= snapshot["capacity_gib"][1]:
                    costs.append(snapshot["performance_cost_tables_s"][1][4 - count]
                                 + sum(request["move_cost_s"] for request in chosen))
        assert modules[0].enumerate_targets(snapshot, max_moves=2)["winner"]["objective_s"] == pytest.approx(min(costs))


def test_empty_model_has_zero_cost_without_dividing_by_zero(modules, fit):
    snapshot = {"counts": [[0, 0], [0, 0]], "new_request_count": 0}
    assert modules[2].performance_tables(snapshot, fit, 1000, False) == [[0], [0]]


def test_live_release_rebalance_preserves_population_and_ownership(modules, fit, tmp_path):
    async def run():
        observed = observations()
        controller = controller_fake(modules[2], fit, tmp_path, observed)
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0, min_gain_s=0)
        assert report["reason"] == "rebalanced" and len(report["moves"]) == 4
        assert report["after_pause"]["snapshot"]["counts"] == [[0, 0], [8, 0]]
        assert report["after_pause"]["winner"]["final_counts"] == [[0, 0], [4, 4]]
        assert controller.admission_generation == 1 and controller.model_admission_generations == [1, 1]
        controller.resume.assert_awaited_once()
        modules[3].verify_rebalance(report, [1, 1], fit)
        return report

    report = asyncio.run(run())
    assert report["end_to_end_control_s_including_preplan"] == pytest.approx(
        report["end_to_end_control_s"] + report["preplan_s"])


@pytest.mark.parametrize("mode", ["empty", "balanced", "expensive", "prefill", "priority"])
def test_rebalance_skips_without_pause(modules, fit, tmp_path, mode):
    async def run():
        observed = observations(0 if mode == "empty" else 8)
        if mode == "prefill":
            observed[1][0]["requests"][0]["movable"] = False
        if mode == "balanced":
            observed = observations(0)
            populated = observations(8)[1][0]
            observed[1] = [copy.deepcopy(populated), copy.deepcopy(populated)]
            for record in observed[1][1]["requests"]:
                record["id"] += "-right"
            for page in observed[1][1]["pages"]:
                page["owners"] = [owner + "-right" for owner in page["owners"]]
        controller = controller_fake(modules[2], fit, tmp_path, observed)
        report = await controller.rebalance_after_release([2, 2], min_gain_s=1e9 if mode == "expensive" else 1,
                                                          admission_busy=lambda: mode == "priority")
        assert not report["moves"]
        controller.pause_boundary.assert_not_awaited()
        assert report["deferred"] == (mode in ("prefill", "priority"))

    asyncio.run(run())


def test_rebalance_stale_before_pause_defers(modules, fit, tmp_path):
    async def run():
        observed = observations()
        controller = controller_fake(modules[2], fit, tmp_path, observed)
        controller.snapshots.side_effect = [copy.deepcopy(observed), observations(7)]
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        assert report["reason"] == "rebalance_stale_before_pause" and report["deferred"]
        controller.pause_boundary.assert_not_awaited()

    asyncio.run(run())


def test_after_boundary_resolve_uses_current_population(modules, fit, tmp_path):
    async def run():
        observed = observations()
        controller = controller_fake(modules[2], fit, tmp_path, observed)

        async def begin(transaction):
            observed[:] = observations(6)
            return copy.deepcopy(observed)

        controller.begin.side_effect = begin
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        assert report["before_pause"]["winner"]["final_counts"] == [[0, 0], [4, 4]]
        assert report["after_pause"]["winner"]["final_counts"] == [[0, 0], [3, 3]]
        assert len(report["moves"]) == 3
        modules[3].verify_rebalance(report, [1, 1], fit)

    asyncio.run(run())


def test_after_pause_unsafe_population_resumes_without_transaction(modules, fit, tmp_path):
    async def run():
        observed = observations()
        unsafe = observations()
        unsafe[1][0]["requests"][0]["movable"] = False
        controller = controller_fake(modules[2], fit, tmp_path, observed)
        controller.snapshots.side_effect = [copy.deepcopy(observed), copy.deepcopy(observed), unsafe]
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        assert report["reason"] == "population_not_all_decode_after_pause"
        controller.begin.assert_not_awaited()
        controller.resume.assert_awaited_once()

    asyncio.run(run())


def test_rebalance_failure_keeps_engines_paused(modules, fit, tmp_path):
    async def run():
        controller = controller_fake(modules[2], fit, tmp_path, observations())
        controller.migrate.side_effect = RuntimeError("uncertain peer completion")
        with pytest.raises(RuntimeError, match="uncertain"):
            await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        controller.resume.assert_not_awaited()
        controller.end.assert_not_awaited()

    asyncio.run(run())


def test_after_boundary_short_remaining_decode_rejects_copy_cost(modules, fit, tmp_path):
    async def run():
        observed = observations()
        controller = controller_fake(modules[2], fit, tmp_path, observed)

        async def begin(transaction):
            for request in observed[1][0]["requests"]:
                request["max_tokens"] = request["output_tokens"] + 1
            return copy.deepcopy(observed)

        controller.begin.side_effect = begin
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        assert report["reason"] == "no_profit_after_fresh_solve" and not report["moves"]
        controller.migrate.assert_not_awaited()
        controller.end.assert_awaited_once()
        controller.resume.assert_awaited_once()
        modules[3].verify_rebalance(report, [1, 1], fit)

    asyncio.run(run())


def test_generation_change_rejects_preplan_even_with_same_ownership(modules, fit, tmp_path):
    async def run():
        observed = observations()
        controller = controller_fake(modules[2], fit, tmp_path, observed)

        async def snapshot(**kwargs):
            if controller.snapshots.await_count == 2:
                controller.admission_generation += 1
            return copy.deepcopy(observed)

        controller.snapshots.side_effect = snapshot
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        assert report["reason"] == "rebalance_stale_before_pause"
        controller.pause_boundary.assert_not_awaited()

    asyncio.run(run())


def test_pending_arrival_after_boundary_has_priority_over_rebalancing(modules, fit, tmp_path):
    async def run():
        controller = controller_fake(modules[2], fit, tmp_path, observations())
        priority = False

        async def begin(transaction):
            nonlocal priority
            priority = True
            return observations()

        controller.begin.side_effect = begin
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0, admission_busy=lambda: priority)
        assert report["reason"] == "admission_has_priority_after_pause" and report["deferred"]
        controller.migrate.assert_not_awaited()
        controller.end.assert_awaited_once()
        controller.resume.assert_awaited_once()

    asyncio.run(run())


def test_rebalance_launcher_is_syntactically_valid_and_preserves_sparse_control():
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    subprocess.run(["bash", "-n", str(root / "rebalance-trace.sbatch"), str(root / "trace-benefit.sbatch")], check=True)
    launcher = (root / "rebalance-trace.sbatch").read_text()
    assert "#SBATCH --partition=gpu_a800" in launcher and "#SBATCH --gpus=2" in launcher
    assert "DP_OPT_STRESS_SWEEP=0" in launcher and "DP_OPT_KV_TRANSPORT=dp_echo" in launcher
    assert "DP_OPT_REBALANCE_TRACE_SHA256:?" in launcher
    assert "DP_OPT_REBALANCE_BUDGET_MIB:?" in launcher
    assert "kv_only ilp kv_only_rebalance" in launcher


def test_completion_worker_coalesces_and_retries_deferred_events(modules):
    async def run():
        calls, records = [], []
        finished = asyncio.Event()

        async def attempt(causes):
            calls.append([cause["request_id"] for cause in causes])
            return {"moves": [], "deferred": len(calls) == 1}

        async def record(report):
            records.append(report)
            if len(records) == 2:
                finished.set()

        worker = modules[1].ReleaseRebalancer(attempt, record, retry_s=.01, cooldown_s=.01)
        worker.notify({"request_id": "code-1"})
        worker.notify({"request_id": "conv-1"})
        await asyncio.wait_for(finished.wait(), 2)
        await worker.close()
        assert calls == [["code-1", "conv-1"], ["code-1", "conv-1"]]
        assert worker.task.done() and not worker.pending

    asyncio.run(run())


def test_completion_worker_propagates_failure(modules):
    async def run():
        worker = modules[1].ReleaseRebalancer(AsyncMock(side_effect=RuntimeError("migration failed")), AsyncMock())
        worker.notify({"request_id": "code-1"})
        with pytest.raises(RuntimeError, match="migration failed"):
            await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["population", "routing", "cost", "trigger"])
def test_independent_rebalance_verifier_rejects_corruption(modules, fit, tmp_path, mutation):
    async def run():
        controller = controller_fake(modules[2], fit, tmp_path, observations())
        report = await controller.rebalance_after_release([1, 1], pause_fixed_s=0)
        if mutation == "population":
            report["after_pause"]["snapshot"]["counts"][0][0] = 1
        elif mutation == "routing":
            report["moves"][0]["model"] = 0
        elif mutation == "cost":
            report["after_pause"]["gate"]["fixed_control_request_s"] = 10
        else:
            report["release_events"] = [{"request_id": "code-1", "model": 0, "completed_at": report["started_at"] + 1}]
        with pytest.raises(ValueError):
            modules[3].verify_rebalance(report, [1, 1], fit, requests={"code-1": report.get("release_events", [{}])[0]}
                                       if mutation == "trigger" else None)

    asyncio.run(run())
