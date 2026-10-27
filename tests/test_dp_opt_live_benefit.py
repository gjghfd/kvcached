import asyncio
import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    for name in ("live_controller", "live_benefit"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("live_controller"), importlib.import_module("live_benefit")


def observation(free=0):
    return {"uniform_page_geometry": True, "page_block_capacity": 64, "page_bytes": 96 * 1024**2,
            "block_size": 16, "block_bytes": 1536 * 1024, "mapped_free_block_slots": free,
            "mapped_bytes": 96 * 1024**2, "requests": [], "pages": []}


def test_pending_reservation_returns_without_capacity_wait(modules):
    async def run():
        record = {"id": "queued-0", "reservation_pending": True, "reserved_at": None}
        controller = SimpleNamespace(snapshots=AsyncMock(return_value=[[{"requests": [record]}, {"requests": []}],
                                                                      [{"requests": []}, {"requests": []}]]))
        task = asyncio.create_task(asyncio.sleep(1))
        try:
            result = await modules[1].wait_reserved(controller, task, "queued", 0, 0)
            assert result["reservation_pending"] and result["reserved_at"] is None
            assert controller.snapshots.await_count == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_reservation_poll_tracks_moved_request_and_exact_scheduler_time(modules):
    async def run():
        pending = {"id": "queued-0", "reservation_pending": True, "prompt_tokens": 16, "max_tokens": 16}
        reserved = {"id": "queued-0", "reservation_pending": False, "reserved_at": 42.5}
        empty = {"requests": []}
        controller = SimpleNamespace(snapshots=AsyncMock(side_effect=[[[empty, {"requests": [pending]}], [empty, empty]],
                                                                       [[empty, {"requests": [reserved]}], [empty, empty]]]),
                                     retry_reservation=AsyncMock())
        task = asyncio.create_task(asyncio.sleep(1))
        try:
            result = await modules[1].wait_reserved(controller, task, "queued", 0, 0,
                                                    wait_for_capacity=True, budgets=[1, 1])
            assert result["reserved_at"] == 42.5 and result["reservation_time_source"] == "scheduler_monotonic"
            controller.retry_reservation.assert_awaited_once_with(0, 1, "queued-0", [1, 1])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_target_only_reservation_confirmation_avoids_other_model_rpc(modules):
    async def run():
        observation = {"requests": [{"id": "request-0", "reservation_pending": False, "reserved_at": 12.5}]}
        controller = SimpleNamespace(call=AsyncMock(return_value=observation), snapshots=AsyncMock())
        task = asyncio.create_task(asyncio.sleep(1))
        try:
            result = await modules[1].wait_reserved(controller, task, "request", 1, 0, target_only=True)
            assert result["reserved_at"] == 12.5 and result["rank_observation"] is observation
            controller.call.assert_awaited_once_with(1, 0, "placement_snapshot")
            controller.snapshots.assert_not_awaited()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_benefit_defaults_use_32k_and_allow_historical_override(modules):
    _, benefit = modules
    required = ["--checkpoint", "checkpoint", "--fit", "fit.json", "--output", "result"]
    defaults = benefit.parse_args(required)
    assert defaults.max_model_len == 32768 and defaults.cache_bytes == 4294967296
    historical = benefit.parse_args([*required, "--scenario", "long_kv", "--output-tokens", "512",
                                    "--max-model-len", "4096", "--cache-bytes", "2147483648"])
    assert historical.max_model_len == 4096 and historical.cache_bytes == 2147483648


@pytest.mark.parametrize("options", (("--max-model-len", "0"), ("--cache-bytes", "0"),
                                     ("--max-model-len", "4095", "--scenario", "long_kv",
                                      "--output-tokens", "512")))
def test_benefit_rejects_unsupported_fixture_lengths(modules, options):
    _, benefit = modules
    with pytest.raises(SystemExit) as error:
        benefit.parse_args(["--checkpoint", "checkpoint", "--fit", "fit.json",
                            "--output", "result", *options])
    assert error.value.code == 2


@pytest.mark.parametrize("free,tokens,pages", ((3, 3072, 3), (63, 3072, 3), (0, 1024, 1),
                                             (20, 320, 0), (19, 320, 1)))
def test_mapped_slot_increment(modules, free, tokens, pages):
    controller, _ = modules
    assert controller.allocation_increment(observation(free), tokens) == pages * 96 / 1024


@pytest.mark.parametrize("field,value", (("uniform_page_geometry", False), ("page_block_capacity", 0),
                                        ("mapped_free_block_slots", -1), ("block_size", 0), ("page_bytes", 0)))
def test_increment_invalid_geometry(modules, field, value):
    controller, _ = modules
    record = observation()
    record[field] = value
    with pytest.raises(ValueError):
        controller.allocation_increment(record, 320)


@pytest.mark.parametrize("tokens", (0, -1, 1.5, True))
def test_increment_invalid_tokens(modules, tokens):
    with pytest.raises(ValueError):
        modules[0].allocation_increment(observation(), tokens)


def test_candidate_selection_exclusive_pages(modules):
    snapshot = {"movable": [{"id": name, "model": 1, "source": 0, "memory_gib": memory}
                            for name, memory in (("shared", 0.01), ("exclusive", 0.09), ("pinned", 0.02))],
                "source_pages": [{"owners": ["exclusive"], "pinned": False, "memory_gib": 0.09},
                                 {"owners": ["shared", "pinned"], "pinned": False, "memory_gib": 0.09},
                                 {"owners": ["pinned"], "pinned": True, "memory_gib": 0.09}]}
    assert [request["id"] for request in modules[0].select_candidates(snapshot, 1)] == ["exclusive"]


def test_tradeoff_fixture_geometry_and_rank_specific_capture(modules):
    _, benefit = modules
    fixture = benefit.workload_fixture("tradeoff")
    assert fixture["counts"] == [[2, 8], [4, 2]] and fixture["total_requests"] == 18
    assert fixture["inputs"][0][0] == [752, 64] and fixture["pages"] == [[2, 3], [2, 1]]

    async def run():
        controller = type("Controller", (), {"call": AsyncMock(return_value=[])})()
        result = await benefit.capture_control(controller, "start")
        assert len(result) == 2 and all(len(ranks) == 2 for ranks in result)
        assert controller.call.await_count == 4
        assert {call.args[:2] for call in controller.call.await_args_list} == {(0, 0), (0, 1), (1, 0), (1, 1)}

    asyncio.run(run())


@pytest.mark.parametrize("policy,arriving_model", (("no_move", 0), ("own_model", 1), ("kv_only", 1), ("ilp", 0)))
def test_direct_slot_admission_includes_resume(modules, monkeypatch, tmp_path, policy, arriving_model):
    module, _ = modules
    records = [[observation(63), observation(63)] for _ in (0, 1)]
    fit = {"raw_verification": {"status": "raw_verified"}, "requests": 128,
           "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": 0.025,
           "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": 0.3, "b": 0.2}}
    monkeypatch.setattr(module, "enumerate_targets", lambda *args, **kwargs: {"winner": None})

    async def run():
        controller = module.PlacementController([], fit, tmp_path, allow_extrapolation=True)
        controller.pause = AsyncMock()
        controller.resume = AsyncMock()
        controller.begin = AsyncMock(return_value=records)
        controller.snapshots = AsyncMock(return_value=records)
        controller.end = AsyncMock()
        controller.check_capacity = AsyncMock()
        admit = AsyncMock()
        result = await controller.schedule_admission(64, 256, [0.1875, 0.1875], admit,
                                                     policy=policy, arriving_model=arriving_model)
        admit.assert_awaited_once()
        controller.resume.assert_awaited_once()
        assert result["admitted"] and not result["moves"]
        assert result["end_to_end_control_s"] >= result["transaction_s_before_resume"]
        assert result["snapshot"]["new_gib"] > result["winner"]["allocation_increment_gib"]
        assert result["after_admission"] == records

    asyncio.run(run())


def test_prefill_race_defers_and_resumes(modules, tmp_path):
    module, _ = modules
    records = [[observation(), observation()] for _ in (0, 1)]
    records[0][0]["requests"] = [{"movable": False}]

    async def run():
        controller = module.PlacementController([], {}, tmp_path)
        controller.pause, controller.resume, controller.begin = AsyncMock(), AsyncMock(), AsyncMock()
        controller.snapshots = AsyncMock(return_value=records)
        result = await controller.schedule_admission(64, 256, [1, 1], AsyncMock())
        assert result["deferred"] and not result["admitted"]
        controller.begin.assert_not_awaited()
        controller.resume.assert_awaited_once()
        assert result["end_to_end_control_s"] >= result["resume_s"]

    asyncio.run(run())


@pytest.mark.parametrize("free_blocks", [1, 100])
@pytest.mark.parametrize("wait_cost", [None, 0])
def test_incoming_b_remaps_cross_tenant_migration(modules, monkeypatch, tmp_path, free_blocks, wait_cost):
    module, _ = modules
    records = [[observation(), observation()] for _ in (0, 1)]
    records[0][0]["requests"] = [{"id": "old-a", "blocks": [1], "movable": True,
                                 "max_tokens": 256, "output_tokens": 8}]
    records[0][0]["pages"] = [{"id": 0, "owners": ["old-a"], "pinned": False, "memory_gib": .09375}]
    fit = {"raw_verification": {"status": "raw_verified"}, "requests": 128,
           "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": .025,
           "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": .3, "b": .2}}

    def solve(snapshot, **options):
        assert snapshot["counts"] == [[0, 0], [1, 0]]
        assert snapshot["movable"][0]["model"] == 1
        return {"winner": {"target": 0, "moves": ["old-a"], "objective_s": 0}}

    monkeypatch.setattr(module, "enumerate_targets", solve)
    after = [[observation(), observation()] for _ in (0, 1)]
    after[1][0]["free_blocks"] = free_blocks

    async def run():
        controller = module.PlacementController([], fit, tmp_path, allow_extrapolation=True)
        controller.pause, controller.resume, controller.end = AsyncMock(), AsyncMock(), AsyncMock()
        controller.begin = AsyncMock(return_value=records)
        controller.snapshots = AsyncMock(side_effect=[records, after, after])
        controller.migrate = AsyncMock(return_value={"model": 0, "request_id": "old-a"})
        controller.release_transfer_cache = AsyncMock(return_value=[])
        controller.check_capacity = AsyncMock()
        admit = AsyncMock()
        result = await controller.schedule_admission(64, 256, [1, 1], admit, arriving_model=1,
                                                     wait_cost_s=wait_cost)
        assert result["admitted"] == (free_blocks >= 20 and wait_cost is None)
        assert result["submitted"] == (free_blocks >= 20 and wait_cost is None)
        if wait_cost is None:
            assert result["actual_allocator_blocked_after_plan"] == (free_blocks < 20)
            assert result["moves"][0]["model"] == 0
            assert controller.migrate.await_args.args[1]["model"] == 0
            controller.release_transfer_cache.assert_awaited_once()
            controller.snapshots.assert_any_await(refresh_physical_capacity=True)
        else:
            assert result["winner"] is None and result["moves"] == []
            controller.migrate.assert_not_awaited()
            controller.release_transfer_cache.assert_not_awaited()
        controller.resume.assert_awaited_once()
        controller.end.assert_awaited_once()
        assert admit.await_count == int(free_blocks >= 20 and wait_cost is None)

    asyncio.run(run())


def test_transfer_cache_release_visits_all_workers_and_refreshes_physical_capacity(modules, tmp_path):
    async def run():
        controller = modules[0].PlacementController([], {}, tmp_path)
        controller.call = AsyncMock(side_effect=lambda model, rank, method, *args: {"model": model, "rank": rank})
        result = await controller.release_transfer_cache()
        assert {(record["model"], record["rank"]) for record in result} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        for call in controller.call.await_args_list:
            assert call.args[2:] == ("collective_rpc", "placement_memory_snapshot", 120, (True,), {})
        controller.call.reset_mock()
        await controller.snapshots(refresh_physical_capacity=True)
        assert controller.call.await_count == 4
        for call in controller.call.await_args_list:
            assert call.args[2:] == ("placement_snapshot", True)

    asyncio.run(run())


def test_summary_counts_queue_tpot_and_deadline(modules):
    _, benefit = modules
    requests = [{"request_id": f"request-{model}", "model": model, "role": "arrival", "arrived_at": 0,
                 "submitted_at": 2, "first_token_at": 3, "completed_at": 5 + model,
                 "token_ids": [1, 2, 3], "cadence": [{"time": 3, "tokens": 1}, {"time": 5, "tokens": 3}]}
                for model in (0, 1)]
    report = benefit.summarize_requests(requests, 0, 5)
    assert report["deadline_met_requests"] == 1
    assert report["output_throughput_tokens_s"] == 1
    assert report["arrival_a_ttft_s"] == 3 and report["mean_queue_s"] == 2
    assert report["mean_tpot_s"] == 1.25 and report["maximum_chunk_gap_s"] == 2
    assert benefit.percentile([1, 2, 3], 0.5) == 2
    with pytest.raises(ValueError):
        benefit.summarize_requests(requests + requests, 0, 5)


def test_raw_verifier_recomputes_paired_metrics(modules, tmp_path):
    _, benefit = modules
    verifier = importlib.import_module("verify_benefit")

    def save(name, value):
        artifact = tmp_path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(value))

    save("config.json", {"repeats": 1, "output_tokens": 256, "deadline_s": 20})
    save("references.json", [{"token_ids": [1] * 256}, {"token_ids": [2] * 256}])
    for model in (0, 1):
        for rank in (0, 1):
            save(f"worker-instance-{model + 1}-dp{rank}.json",
                 {"ipc_name": f"tenant{model}-rank{rank}", "gpu_uuid": f"gpu{rank}",
                  "groups": {name: {"world_size": size} for name, size in (("tp", 1), ("dp", 2), ("ep", 2))},
                  "parameter_bytes": 1, "expert_ids": [list(range(rank * 64, (rank + 1) * 64))] * 48})
    (tmp_path / "exit-status").write_text("0\n")
    reports = []
    for policy in benefit.POLICIES:
        identifiers = [f"r0-{policy}-bootstrap-{model}-{rank}-{index}"
                       for model, counts in enumerate(((3, 3), (4, 3)))
                       for rank, count in enumerate(counts) for index in range(count)]
        identifiers.extend(f"r0-{policy}-arrival-{model}" for model in (0, 1))
        requests = [{"request_id": identifiers[index], "model": int(index == 14),
                     "role": "arrival" if index >= 13 else "bootstrap", "arrived_at": 0,
                     "submitted_at": 1, "first_token_at": 2, "completed_at": 10,
                     "token_ids": [2 if index == 13 else 1] * 256,
                     "cadence": [{"time": 2, "tokens": 1}, {"time": 10, "tokens": 256}]}
                    for index in range(15)]
        report = benefit.summarize_requests(requests, 0, 20)
        report.update({"repeat": 0, "policy": policy, "migration_count": 0, "control_s": 0, "solve_s": 0})
        reports.append(report)
        save(f"r0-{policy}/requests.json", requests)
        save(f"r0-{policy}/workload.json", {"origin": 0, "capacity_gib": [1, 1]})
        before = [[observation(), observation()] for _ in (0, 1)]
        for ranks in before:
            for record in ranks:
                record["mapped_bytes"] = 0
        save(f"r0-{policy}/before.json", before)
        save(f"r0-{policy}/decisions.json", [])
    save("summary.json", {"status": "passed", "trials": reports})
    result = verifier.verify(tmp_path)
    assert result["status"] == "raw_verified" and result["paired_repeats"] == 1
    assert result["ilp_vs_baselines"]["no_move"]["mean"]["throughput_gain_fraction"] == 0
    reports[0]["makespan_s"] += 1
    save("summary.json", {"status": "passed", "trials": reports})
    with pytest.raises(ValueError, match="Raw metric mismatch"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("interval", (1, 32))
def test_runtime_control_checks_all_workers(modules, tmp_path, interval):
    verifier = importlib.import_module("verify_benefit")
    workers = [[{"dp_pause_sync_interval": interval} for _ in (0, 1)] for _ in (0, 1)]
    assert verifier.verify_runtime_control(tmp_path, workers) is None
    control = {"dp_pause_sync_interval": interval, "same_interval_for_all_policies": True,
               "idle_rank_expert_execution_not_disabled": True}
    (tmp_path / "runtime-control.json").write_text(json.dumps(control))
    assert verifier.verify_runtime_control(tmp_path, workers) == control
    workers[1][1]["dp_pause_sync_interval"] = 32 if interval == 1 else 1
    with pytest.raises(ValueError, match="Worker DP pause intervals"):
        verifier.verify_runtime_control(tmp_path, workers)


@pytest.mark.parametrize("field,value", (("dp_pause_sync_interval", 2), ("dp_pause_sync_interval", True),
                                        ("same_interval_for_all_policies", False),
                                        ("idle_rank_expert_execution_not_disabled", False)))
def test_runtime_control_rejects_invalid_configuration(modules, tmp_path, field, value):
    verifier = importlib.import_module("verify_benefit")
    control = {"dp_pause_sync_interval": 1, "same_interval_for_all_policies": True,
               "idle_rank_expert_execution_not_disabled": True}
    control[field] = value
    (tmp_path / "runtime-control.json").write_text(json.dumps(control))
    with pytest.raises(ValueError, match="Invalid or unequal"):
        verifier.verify_runtime_control(tmp_path, [])


def test_decode_boundary_requires_every_bootstrap_request(modules):
    verifier = importlib.import_module("verify_benefit")
    before = [[{"requests": [{"movable": True, "output_tokens": 192}]} for _ in (0, 1)] for _ in (0, 1)]
    verifier.verify_decode_boundary(before, 192)
    before[1][1]["requests"][0]["output_tokens"] = 191
    with pytest.raises(ValueError, match="decode boundary"):
        verifier.verify_decode_boundary(before, 192)
    before[1][1]["requests"][0].update({"output_tokens": 192, "movable": False})
    with pytest.raises(ValueError, match="decode boundary"):
        verifier.verify_decode_boundary(before, 192)


def test_long_decode_fixture_preserves_distinct_capacity_choices(modules):
    controller, benefit = modules
    fixture = benefit.workload_fixture("tradeoff_long")
    assert fixture["counts"] == [[2, 8], [4, 2]] and fixture["total_requests"] == 18
    assert fixture["output_tokens"] == 512 and fixture["arrival_input_tokens"] == 2560
    blocks = [[[math.ceil((length + 512) / 16) for length in rank] for rank in ranks]
              for ranks in fixture["inputs"]]
    pages = [[math.ceil((sum(rank) + 1) / 64) for rank in ranks] for ranks in blocks]
    assert pages == fixture["pages"] == [[2, 5], [3, 2]]
    assert blocks[0][0] == [63, 36] and blocks[1][0] == [37, 37, 37, 64]
    budgets = [(sum(pages[model][rank] for model in (0, 1)) + fixture["headroom"][rank]) * 96
               for rank in (0, 1)]
    assert budgets == [672, 768]
    for rank, free in ((0, 28), (1, 31)):
        increment = controller.allocation_increment(observation(free), 3072) * 1024
        initial = sum(pages[model][rank] for model in (0, 1)) * 96
        assert initial + increment > budgets[rank]
    assert blocks[0][0][1] * 1.5 == 54 and blocks[1][0][-1] * 1.5 == 96


def test_long_kv_global_capacity_but_no_local_admission(modules):
    controller, benefit = modules
    fixture = benefit.workload_fixture("long_kv")
    assert fixture["counts"] == [[3, 3], [3, 3]] and fixture["total_requests"] == 14
    assert fixture["output_tokens"] == 512 and fixture["arrival_input_tokens"] == 3584
    assert fixture["b_arrival_input_tokens"] == 1024
    blocks = [[[math.ceil((length + 512) / 16) for length in rank] for rank in ranks]
              for ranks in fixture["inputs"]]
    pages = [[math.ceil((sum(rank) + 1) / 64) for rank in ranks] for ranks in blocks]
    assert pages == fixture["pages"] == [[11, 11], [8, 8]]
    initial = [sum(pages[model][rank] for model in (0, 1)) * 96 for rank in (0, 1)]
    budgets = [initial[rank] + fixture["headroom"][rank] * 96 for rank in (0, 1)]
    assert budgets == [2112, 2112] and sum(budgets) - sum(initial) == 576
    all_reserved = sum(sum(rank) for ranks in blocks for rank in ranks) * 1.5 + (256 + 96) * 1.5
    assert all_reserved == 3984 < sum(budgets)
    assert controller.allocation_increment(observation(31), 4096) * 1024 == 384
    assert initial[0] + 384 > budgets[0] and initial[1] + 384 > budgets[1]
    assert controller.allocation_increment(observation(31), 224 * 16) * 1024 == 384
    assert controller.allocation_increment(observation(31), 160 * 16) * 1024 == 288
    assert initial[0] - 192 + 384 + 96 == budgets[0] and initial[1] + 288 == budgets[1]


def test_capacity_witness_rejects_insufficient_global_space(modules):
    verifier = importlib.import_module("verify_benefit")
    before = [[observation() for _ in (0, 1)] for _ in (0, 1)]
    for ranks in before:
        for record in ranks:
            record["mapped_bytes"] = 0
    witness = {"before": before, "after_rollback": before, "total_budget_bytes": 0,
               "initial_mapped_bytes": 0, "global_free_bytes": 0, "arrival_a_reserved_bytes": 0,
               "all_request_reserved_bytes": 0}
    with pytest.raises(ValueError, match="Global KV capacity"):
        verifier.verify_capacity_witness(witness, before, {"capacity_gib": [0, 0]})


def long_kv_witness():
    def rank_record(model, rank):
        blocks = 224 if model == 0 else 160
        pages = 11 if model == 0 else 8
        record = observation()
        record["mapped_bytes"] = pages * record["page_bytes"]
        record["requests"] = [{"id": f"tenant{model}-rank{rank}-request{index}",
                               "blocks": list(range(1 + index * blocks, 1 + (index + 1) * blocks))}
                              for index in range(3)]
        record["pages"] = [{"id": page, "owners": sorted(request["id"] for request in record["requests"]
                                                         if any(block // 64 == page for block in request["blocks"]))}
                           for page in range(pages)]
        return record

    before = [[rank_record(model, rank) for rank in (0, 1)] for model in (0, 1)]
    witness = {"before": before, "after_rollback": json.loads(json.dumps(before)),
               "total_budget_bytes": 4224 * 1024**2, "initial_mapped_bytes": 3648 * 1024**2,
               "global_free_bytes": 576 * 1024**2, "arrival_a_reserved_bytes": 384 * 1024**2,
               "all_request_reserved_bytes": 3984 * 1024**2, "probes": []}
    for rank in (0, 1):
        during = json.loads(json.dumps(before[0][rank]))
        blocks = list(range(673, 929))
        during["requests"].append({"id": "probe", "blocks": blocks})
        during["mapped_bytes"] = 15 * during["page_bytes"]
        during["pages"] = [{"id": page, "owners": sorted(request["id"] for request in during["requests"]
                                                         if any(block // 64 == page for block in request["blocks"]))}
                           for page in range(15)]
        peak = [1824 / 1024, 1824 / 1024]
        peak[rank] = 2208 / 1024
        witness["probes"].append({"model": 0, "rank": rank, "allocated": True, "fits_budget": False,
                                  "rollback_verified": True, "additional_mapped_bytes": 384 * 1024**2,
                                  "allocation_peak_gib": peak,
                                  "allocator": {"allocated": True, "blocks": blocks, "during": during}})
    return witness, before, {"capacity_gib": [2112 / 1024, 2112 / 1024]}


def test_capacity_witness_validates_actual_two_rank_probes(modules):
    verifier = importlib.import_module("verify_benefit")
    witness, before, workload = long_kv_witness()
    for ranks in witness["after_rollback"]:
        for rank in ranks:
            rank["pages"].reverse()
    verifier.verify_capacity_witness(witness, before, workload)


@pytest.mark.parametrize("field,value", (("fits_budget", True), ("allocated", False),
                                       ("rollback_verified", False), ("additional_mapped_bytes", 0)))
def test_capacity_witness_rejects_invalid_probe(modules, field, value):
    verifier = importlib.import_module("verify_benefit")
    witness, before, workload = long_kv_witness()
    witness["probes"][0][field] = value
    with pytest.raises(ValueError, match="Actual no-move allocation"):
        verifier.verify_capacity_witness(witness, before, workload)


def test_capacity_witness_rejects_changed_rollback_owner(modules):
    verifier = importlib.import_module("verify_benefit")
    witness, before, workload = long_kv_witness()
    witness["after_rollback"][0][0]["requests"][0]["blocks"].pop()
    with pytest.raises(ValueError, match="changed request/page ownership"):
        verifier.verify_capacity_witness(witness, before, workload)
