import asyncio
import importlib
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(root))

    class Scheduler:
        def add_request(self, request):
            self.requests[request.request_id] = request
            self.waiting.append(request)

        def schedule(self, *args, **kwargs):
            self.scheduled_waiting = list(self.waiting)
            return "scheduled"

    entries = {"vllm.v1.core.sched.scheduler": {"Scheduler": Scheduler},
               "vllm.v1.core.sched.interface": {"PauseState": SimpleNamespace(PAUSED_ALL="paused")},
               "vllm.v1.request": {"RequestStatus": SimpleNamespace(RUNNING="running")},
               "vllm.v1.worker.gpu_input_batch": {"CachedRequestState": SimpleNamespace}}
    for name, members in entries.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("live_scheduler", "live_controller", "live_benchmark"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return SimpleNamespace(scheduler=importlib.import_module("live_scheduler"),
                           controller=importlib.import_module("live_controller"),
                           benchmark=importlib.import_module("live_benchmark"))


@pytest.fixture
def scheduler(modules):
    scheduler = object.__new__(modules.scheduler.PlacementScheduler)
    scheduler.placement_pending = {}
    scheduler.placement_committed = set()
    scheduler.placement_transaction = "epoch"
    scheduler.placement_unreserved = {}
    scheduler.pause_state = "paused"
    scheduler.requests = {}
    scheduler.running = []
    class Queue(list):
        def remove_requests(self, requests):
            for request in requests:
                if request in self:
                    self.remove(request)

        def prepend_request(self, request):
            self.insert(0, request)

    scheduler.waiting = Queue()
    scheduler.skipped_waiting = Queue()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.max_model_len = 4096
    scheduler.kv_cache_manager = SimpleNamespace(allocate_slots=Mock(return_value=object()),
                                                free=Mock(), get_block_ids=Mock(return_value=([7, 8],)))
    scheduler.placement_snapshot = Mock(return_value={"mapped_bytes": 0, "requests": []})
    return scheduler


def request(request_id="request"):
    return SimpleNamespace(request_id=request_id, num_computed_tokens=32, num_prompt_tokens=16,
                           max_tokens=16, prompt_token_ids=[1, 2], output_token_ids=[3],
                           sampling_params=SimpleNamespace(temperature=0), events=[])


def test_transaction_epoch_and_pause_guards(scheduler):
    assert scheduler.placement_begin("epoch") == {"mapped_bytes": 0, "requests": []}
    with pytest.raises(RuntimeError):
        scheduler.placement_begin("different")
    scheduler.pause_state = "running"
    with pytest.raises(RuntimeError):
        scheduler.placement_end("epoch")


def test_snapshot_refresh_invalidates_physical_capacity_only_when_requested(modules):
    scheduler = object.__new__(modules.scheduler.PlacementScheduler)
    manager = SimpleNamespace(in_shrink=False, full_pages=[], avail_pages=[], reserved_blocks=[],
                              _avail_physical_pages_cache=3, num_avail_blocks=0, page_size=2,
                              block_mem_size=1, num_layers=48, num_kv_buffers=1,
                              page_allocator=SimpleNamespace(group_indices_by_page=Mock(return_value={})),
                              _memory_limit_page_bundle_bytes=Mock(return_value=96),
                              get_mapped_memory_size=Mock(return_value=0))
    pool = SimpleNamespace(kv_cache_manager=manager,
                           get_num_free_blocks=Mock(side_effect=lambda: 6 if manager._avail_physical_pages_cache else 12))
    scheduler.kv_cache_manager = SimpleNamespace(block_pool=pool)
    scheduler.requests = {}
    scheduler.parallel_config = SimpleNamespace(data_parallel_rank=0)
    scheduler.cache_config = SimpleNamespace(block_size=16)
    assert scheduler.placement_snapshot()["free_blocks"] == 6
    assert manager._avail_physical_pages_cache == 3
    assert scheduler.placement_snapshot(True)["free_blocks"] == 12
    assert manager._avail_physical_pages_cache is None


def test_prepare_commit_rollback_preserves_state(scheduler):
    original = request()
    state = scheduler.placement_prepare("epoch", {"request_id": "request",
                                                 "request": pickle.dumps(original), "blocks": [1, 2]})
    assert pickle.loads(state["request"]).num_computed_tokens == 32
    assert state["blocks"] == [[7, 8]]
    with pytest.raises(RuntimeError):
        scheduler.placement_end("epoch")
    scheduler.placement_commit("epoch", "request")
    assert scheduler.requests["request"].output_token_ids == [3]
    scheduler.placement_rollback("epoch", "request")
    assert scheduler.running == []
    assert scheduler.requests == {}
    assert scheduler.placement_end("epoch")


def test_snapshot_decode_estimate_uses_observed_progress_and_bounded_history(modules, monkeypatch):
    scheduler = object.__new__(modules.scheduler.PlacementScheduler)
    manager = SimpleNamespace(in_shrink=False, full_pages=[], avail_pages=[], reserved_blocks=[],
                              num_avail_blocks=0, page_size=2, block_mem_size=1, num_layers=48,
                              num_kv_buffers=1, page_allocator=SimpleNamespace(
                                  group_indices_by_page=Mock(return_value={})),
                              _memory_limit_page_bundle_bytes=Mock(return_value=96),
                              get_mapped_memory_size=Mock(return_value=0))
    scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=Mock(return_value=([1],)),
        block_pool=SimpleNamespace(kv_cache_manager=manager, get_num_free_blocks=Mock(return_value=12)))
    active = SimpleNamespace(request_id="decode", num_computed_tokens=16, num_prompt_tokens=16,
                             num_output_tokens=1, num_in_flight_tokens=0, max_tokens=100,
                             sampling_params=SimpleNamespace(temperature=0, prompt_logprobs=None),
                             mm_features=[], lora_request=None, use_structured_output=False,
                             prompt_embeds=None, resumable=False, spec_token_ids=[], _block_hasher=None)
    scheduler.requests = {"decode": active}
    scheduler.running = [active]
    scheduler.placement_unreserved = {}
    scheduler.parallel_config = SimpleNamespace(data_parallel_rank=0)
    scheduler.cache_config = SimpleNamespace(block_size=16)
    scheduler.pause_state = "running"
    clock = {"now": 0.0}
    monkeypatch.setattr(modules.scheduler.time, "monotonic", lambda: clock["now"])
    assert scheduler.placement_snapshot()["requests"][0]["estimated_remaining_decode_s"] is None
    clock["now"], active.num_output_tokens = 1.0, 11
    record = scheduler.placement_snapshot()["requests"][0]
    assert record["decode_rate_tokens_s"] == 10
    assert record["estimated_remaining_decode_s"] == 8.9
    scheduler.pause_state = "paused"
    clock["now"] = 2.0
    scheduler.placement_snapshot()
    assert len(scheduler.placement_decode_progress["decode"]) == 2
    scheduler.pause_state = "running"
    for index in range(3, 13):
        clock["now"], active.num_output_tokens = float(index), index + 11
        scheduler.placement_snapshot()
    assert len(scheduler.placement_decode_progress["decode"]) == 8
    scheduler.requests = {}
    scheduler.placement_snapshot()
    assert scheduler.placement_decode_progress == {}


@pytest.mark.parametrize("failure", (None, RuntimeError("allocation failure"), "wrong_count"))
def test_prepare_failure_frees_destination(scheduler, failure):
    if failure == "wrong_count":
        scheduler.kv_cache_manager.get_block_ids.return_value = ([7],)
    elif isinstance(failure, Exception):
        scheduler.kv_cache_manager.allocate_slots.side_effect = failure
    else:
        scheduler.kv_cache_manager.allocate_slots.return_value = None
    with pytest.raises((MemoryError, RuntimeError, ValueError)):
        scheduler.placement_prepare("epoch", {"request_id": "request", "request": pickle.dumps(request()),
                                             "blocks": [1, 2]})
    scheduler.kv_cache_manager.free.assert_called_once()
    assert not scheduler.placement_pending


def test_prepare_identity_and_duplicate_guards(scheduler):
    with pytest.raises(ValueError):
        scheduler.placement_prepare("epoch", {"request_id": "wrong", "request": pickle.dumps(request()),
                                             "blocks": [1, 2]})
    scheduler.requests["request"] = request()
    with pytest.raises(ValueError):
        scheduler.add_request(request())
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


def test_full_sequence_reservation_and_failure(scheduler):
    scheduler.add_request(request())
    assert scheduler.kv_cache_manager.allocate_slots.call_args.args[1] == 32
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler.add_request(request("failed"))
    assert "failed" in scheduler.requests and "failed" in scheduler.placement_unreserved
    scheduler.kv_cache_manager.free.assert_called_once()


def test_capacity_miss_waits_without_partial_prefill_then_retries_full_reservation(scheduler):
    scheduler.pause_state = "running"
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    blocked = request("blocked")
    scheduler.add_request(blocked)
    assert scheduler.schedule() == "scheduled"
    assert scheduler.scheduled_waiting == []
    assert scheduler.waiting == [blocked]
    scheduler.kv_cache_manager.allocate_slots.return_value = object()
    scheduler.schedule()
    assert not scheduler.placement_unreserved
    assert scheduler.scheduled_waiting == [blocked]
    assert scheduler.kv_cache_manager.allocate_slots.call_args.args[1] == 32


def test_capacity_retry_honors_pause_and_removes_aborted_requests(scheduler):
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler.add_request(request("blocked"))
    scheduler.schedule()
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 1
    scheduler.requests.pop("blocked")
    scheduler.schedule()
    assert not scheduler.placement_unreserved


def test_unexpected_allocation_failure_still_surfaces_and_frees(scheduler):
    scheduler.kv_cache_manager.allocate_slots.side_effect = RuntimeError("driver failure")
    with pytest.raises(RuntimeError, match="driver failure"):
        scheduler.add_request(request())
    assert not scheduler.requests and not scheduler.placement_unreserved
    scheduler.kv_cache_manager.free.assert_called_once()


def test_capacity_wait_preserves_relative_native_queue_order(scheduler):
    scheduler.pause_state = "running"
    scheduler.kv_cache_manager.allocate_slots.side_effect = [object(), None, object(), None]
    first, blocked, last = request("first"), request("blocked"), request("last")
    for entry in (first, blocked, last):
        scheduler.add_request(entry)
    scheduler.schedule()
    assert scheduler.scheduled_waiting == [first, last]
    assert scheduler.waiting == [first, blocked, last]
    assert first.placement_reserved_at > 0 and last.placement_reserved_at > 0
    assert not hasattr(blocked, "placement_reserved_at")


def test_controlled_retry_only_runs_via_explicit_allocator_rpc(scheduler, monkeypatch):
    monkeypatch.setenv("DP_OPT_CONTROLLED_RESERVATION_RETRY", "1")
    scheduler.pause_state = "running"
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler.add_request(request("blocked"))
    scheduler.kv_cache_manager.allocate_slots.return_value = object()
    scheduler.schedule()
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 1
    result = scheduler.placement_retry_reservation("blocked")
    assert not result["reservation_pending"] and result["reserved_at"] > 0
    assert not scheduler.placement_unreserved
    assert scheduler.placement_retry_reservation("aborted") == {"missing": True}


def test_controlled_retry_honors_pause_and_allocator_rejection(scheduler):
    scheduler.kv_cache_manager.allocate_slots.return_value = None
    scheduler.add_request(request("blocked"))
    assert scheduler.placement_retry_reservation("blocked")["reservation_pending"]
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 1
    scheduler.pause_state = "running"
    assert scheduler.placement_retry_reservation("blocked")["reservation_pending"]
    assert scheduler.kv_cache_manager.allocate_slots.call_count == 2


def test_unique_backing_views_and_missing_metadata(modules):
    first = SimpleNamespace(data_ptr=lambda: 1, shape=(8, 32), stride=lambda: (32, 1))
    second = SimpleNamespace(data_ptr=lambda: 2, shape=(8, 32), stride=lambda: (32, 1))
    runner = SimpleNamespace(kv_caches=[SimpleNamespace(_kvcached_block_copy_view=view)
                                      for view in (first, first, second)])
    assert modules.scheduler.unique_views(runner) == [first, second]
    runner.kv_caches.append(object())
    with pytest.raises(RuntimeError):
        modules.scheduler.unique_views(runner)


def test_worker_state_seed_and_identity(modules):
    runner = SimpleNamespace(input_batch=SimpleNamespace(remove_request=Mock()), requests={})
    worker = SimpleNamespace(model_runner=runner)
    state = {"request_id": "request", "request": pickle.dumps(request()), "blocks": [[7, 8]]}
    modules.scheduler.install_runner_state(worker, state)
    assert runner.requests["request"].output_token_ids == [3]
    assert runner.requests["request"].block_ids == ([7, 8],)
    with pytest.raises(ValueError):
        modules.scheduler.install_runner_state(worker, dict(state, request_id="wrong"))
    assert "wrong" not in runner.requests
    modules.scheduler.install_runner_state(worker, {"request_id": "request", "remove": True})
    assert not runner.requests


def test_worker_utilities_require_epoch(modules, scheduler):
    engine = SimpleNamespace(scheduler=scheduler, collective_rpc=Mock(return_value=[True]))
    modules.scheduler.bind_engine(engine)
    with pytest.raises(RuntimeError):
        engine.placement_runner_state("wrong", {})
    engine.collective_rpc.assert_not_called()
    assert engine.placement_runner_state("epoch", {"request_id": "request", "remove": True})


def test_source_removal_drains_real_retired_pages(modules, scheduler):
    manager = SimpleNamespace(defer_physical_release=True, capture_physical_release_marker=Mock(return_value=4),
                              release_retired_pages_through=Mock())
    scheduler.kv_cache_manager.block_pool = SimpleNamespace(kv_cache_manager=manager)
    scheduler.requests["request"] = original = request()
    scheduler.running.append(original)
    engine = SimpleNamespace(scheduler=scheduler, batch_queue=[], collective_rpc=Mock())
    modules.scheduler.bind_engine(engine)
    assert engine.placement_remove("epoch", "request") == {"mapped_bytes": 0, "requests": []}
    manager.release_retired_pages_through.assert_called_once_with(4)


def test_transfer_path_and_block_guards(modules, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setenv("DP_OPT_TRANSFER_ROOT", str(tmp_path))
    view = SimpleNamespace(data_ptr=lambda: 1, shape=(8, 32), stride=lambda: (32, 1))
    worker = SimpleNamespace(model_runner=SimpleNamespace(kv_caches=[
        SimpleNamespace(_kvcached_block_copy_view=view)]))
    with pytest.raises(ValueError, match="private job"):
        modules.scheduler.kv_file(worker, "export", [1], str(tmp_path.parent / "outside.pt"))
    for blocks in ([], [1, 1], ["1"], [0], [8]):
        with pytest.raises(ValueError):
            modules.scheduler.kv_file(worker, "export", blocks, str(tmp_path / "inside.pt"))


def test_host_transfer_roundtrip_checks_real_bytes(modules, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("DP_OPT_TRANSFER_ROOT", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    source = torch.arange(8 * 32, dtype=torch.int64).to(torch.uint8).reshape(8, 32)
    destination = torch.zeros_like(source)
    source_tensor = torch.empty(0)
    destination_tensor = torch.empty(0)
    source_tensor._kvcached_block_copy_view = source
    destination_tensor._kvcached_block_copy_view = destination
    source_worker = SimpleNamespace(device="cpu", model_runner=SimpleNamespace(kv_caches=[source_tensor, source_tensor]))
    destination_worker = SimpleNamespace(device="cpu", model_runner=SimpleNamespace(kv_caches=[destination_tensor]))
    path = str(tmp_path / "copy.pt")
    exported = modules.scheduler.kv_file(source_worker, "export", [1, 2], path)
    received = modules.scheduler.kv_file(destination_worker, "import", [4, 5], path)
    assert (exported["sha256"], exported["bytes"]) == (received["sha256"], received["bytes"])
    assert exported["bytes"] == 64
    assert "serialize_file" in exported["stage_s"]
    assert "readback_device_to_host" in received["stage_s"]
    assert all(value >= 0 for result in (exported, received) for value in result["stage_s"].values())
    assert torch.equal(source[[1, 2]], destination[[4, 5]])
    assert torch.count_nonzero(destination[[0, 1, 2, 3, 6, 7]]) == 0


def controller_with_calls(modules, tmp_path, failure=None, peak=False):
    engine = SimpleNamespace(engine_core=SimpleNamespace(core_engines=["source", "destination"],
                                                         reqs_in_flight={"request": "source"}))
    async def create():
        return modules.controller.PlacementController([engine, engine], {}, tmp_path)

    controller = asyncio.run(create())
    calls = []

    async def call(model, rank, method, *args):
        calls.append((rank, method, args))
        if method == failure:
            raise RuntimeError("RPC failed after possible mutation")
        if method == "placement_export":
            return {"request_id": "request", "blocks": [1, 2], "request": b"state"}
        if method == "placement_prepare":
            return {"request_id": "request", "blocks": [[7, 8]], "request": b"state"}
        if method == "placement_kv_file":
            return {"sha256": "good" if args[1] == "export" or failure != "checksum" else "bad", "bytes": 64}
        if method == "placement_remove":
            return {"mapped_bytes": 0}
        return True

    async def snapshots():
        return [[{"mapped_bytes": 1024**3 if peak else 0} for rank in (0, 1)] for model in (0, 1)]

    controller.call = call
    controller.snapshots = snapshots
    return controller, calls, engine


def test_live_migration_copy_commit_remove_order(modules, tmp_path):
    controller, calls, engine = controller_with_calls(modules, tmp_path)
    event = asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0,
                                                    "memory_gib": 1}, [1, 1]))
    assert [method for rank, method, args in calls] == ["placement_export", "placement_prepare",
        "placement_kv_file", "placement_kv_file", "placement_runner_state", "placement_commit",
        "placement_remove", "placement_runner_state"]
    assert engine.engine_core.reqs_in_flight["request"] == "destination"
    assert event["kv_bytes"] == 64
    assert event["copy_peak_gib"] == [0, 0]
    assert set(event["stage_s"]) == {"export_request_state", "source_capacity_check", "destination_prepare",
                                     "copy_peak_capacity_check", "export_kv_rpc", "import_kv_rpc",
                                     "install_destination_runner", "commit_destination", "release_source",
                                     "reroute_and_remove_source_runner"}
    assert all(value >= 0 for value in event["stage_s"].values())


@pytest.mark.parametrize("failure", ("checksum", "placement_prepare", "placement_commit"))
def test_pre_source_removal_failure_rolls_back_destination(modules, tmp_path, failure):
    controller, calls, engine = controller_with_calls(modules, tmp_path, failure)
    with pytest.raises(RuntimeError):
        asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert "placement_rollback" in [method for rank, method, args in calls]
    assert "placement_remove" not in [method for rank, method, args in calls]
    assert engine.engine_core.reqs_in_flight["request"] == "source"


def test_uncertain_source_removal_never_destroys_destination(modules, tmp_path):
    controller, calls, engine = controller_with_calls(modules, tmp_path, "placement_remove")
    with pytest.raises(RuntimeError, match="retain destination"):
        asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert "placement_rollback" not in [method for rank, method, args in calls]
    assert controller.events == []


def test_actual_copy_peak_checked_before_source_release(modules, tmp_path):
    controller, calls, engine = controller_with_calls(modules, tmp_path, peak=True)
    observed = 0

    async def snapshots():
        nonlocal observed
        observed += 1
        return [[{"mapped_bytes": 0 if observed == 1 else 1024**3} for rank in (0, 1)]
                for model in (0, 1)]

    controller.snapshots = snapshots
    with pytest.raises(MemoryError):
        asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0,
                                                "memory_gib": 1}, [1, 1]))
    assert "placement_rollback" in [method for rank, method, args in calls]
    assert "placement_kv_file" not in [method for rank, method, args in calls]
    assert "placement_remove" not in [method for rank, method, args in calls]


@pytest.mark.parametrize("fail_admission", (False, True))
def test_planning_and_admission_share_frozen_epoch(modules, monkeypatch, tmp_path, fail_admission):
    fit = {"status": "verified_curve_fit", "raw_verification": {"status": "raw_verified"}, "requests": 128,
           "capture_sizes": [1, 8, 16, 32, 64, 128, 256], "baseline_cadence_s": 0.025,
           "capture_shape_posthoc": {"status": "descriptive_posthoc_fit", "a": 0.3, "b": 0.2}}
    order = []
    observations = [[{"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
                      "block_bytes": 1, "page_bytes": 64, "uniform_page_geometry": True}
                     for rank in (0, 1)] for model in (0, 1)]

    def solve(snapshot, **kwargs):
        order.append("solve")
        assert snapshot["capacity_gib"] == [64 / 1024**3] * 2
        return {"winner": {"target": 0, "moves": []}}

    monkeypatch.setattr(modules.controller, "enumerate_targets", solve)

    async def run():
        controller = modules.controller.PlacementController([], fit, tmp_path, allow_extrapolation=True)

        async def pause():
            order.append("pause")

        async def begin(transaction):
            order.append("begin")
            return observations

        async def end(transaction):
            order.append("end")

        async def resume():
            order.append("resume")

        async def snapshots():
            return observations

        async def admit(target):
            order.append("admit")
            assert target == 0
            if fail_admission:
                raise RuntimeError("admission failed")

        controller.pause, controller.begin, controller.end = pause, begin, end
        controller.resume, controller.snapshots = resume, snapshots
        return await controller.place(16, headroom_pages=1, admit=admit)

    if fail_admission:
        with pytest.raises(RuntimeError, match="admission failed"):
            asyncio.run(run())
        assert order == ["pause", "begin", "solve", "admit"]
    else:
        result = asyncio.run(run())
        assert result["admitted_while_paused"]
        assert order == ["pause", "begin", "solve", "admit", "end", "resume"]


@pytest.mark.parametrize("allocation", (True, False, "error"))
def test_real_allocation_probe_restores_scheduler(modules, scheduler, monkeypatch, allocation):
    sampling = ModuleType("vllm")
    sampling.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", sampling)
    monkeypatch.setattr(sys.modules["vllm.v1.request"], "Request", SimpleNamespace, raising=False)
    scheduler.placement_snapshot.side_effect = lambda: {"requests": list(scheduler.requests), "mapped_bytes": 0}
    scheduler.kv_cache_manager.allocate_slots.return_value = object() if allocation is True else None
    if allocation == "error":
        scheduler.kv_cache_manager.allocate_slots.side_effect = RuntimeError("allocator failure")
        with pytest.raises(RuntimeError, match="allocator failure"):
            scheduler.placement_probe("epoch", [1, 2], 16)
    else:
        result = scheduler.placement_probe("epoch", [1, 2], 16)
        assert result["allocated"] is allocation
        assert len(result["during"]["requests"]) == int(allocation)
        assert result["blocks"] == ([7, 8] if allocation else [])
    assert scheduler.requests == {} and scheduler.placement_pending == {}
    scheduler.kv_cache_manager.free.assert_called_once()


def test_probe_drain_uses_worker_queue_and_epoch_guards(modules, scheduler):
    manager = SimpleNamespace(defer_physical_release=True, capture_physical_release_marker=Mock(return_value=9),
                              release_retired_pages_through=Mock())
    scheduler.kv_cache_manager.block_pool = SimpleNamespace(kv_cache_manager=manager)
    scheduler.placement_probe = Mock(return_value={"allocated": True})
    engine = SimpleNamespace(scheduler=scheduler, batch_queue=["in-flight"], collective_rpc=Mock())
    modules.scheduler.bind_engine(engine)
    with pytest.raises(RuntimeError, match="batches remain"):
        engine.placement_probe("epoch", [1], 16)
    scheduler.placement_probe.assert_not_called()
    engine.batch_queue = []
    assert engine.placement_probe("epoch", [1], 16)["after"]["mapped_bytes"] == 0
    manager.release_retired_pages_through.assert_called_once_with(9)


@pytest.mark.parametrize("mutation", (None, "pages", "blocks"))
def test_controller_probe_checks_physical_rollback(modules, tmp_path, mutation):
    observations = [[{"mapped_bytes": 96, "requests": [{"id": "request", "blocks": [1, 2]}],
                      "pages": [{"id": 0, "owners": ["request"], "pinned": True}]} for rank in (0, 1)]
                    for model in (0, 1)]
    restored = pickle.loads(pickle.dumps(observations))
    if mutation == "pages":
        restored[0][0]["mapped_bytes"] += 96
    if mutation == "blocks":
        restored[0][0]["requests"][0]["blocks"] = [3, 4]

    async def run():
        controller = modules.controller.PlacementController([], {}, tmp_path)
        snapshots = iter((observations, restored))

        async def snapshot():
            return next(snapshots)

        async def call(*args):
            return {"allocated": True, "during": {"mapped_bytes": 288}}

        controller.snapshots, controller.call = snapshot, call
        return await controller.probe("epoch", 0, 0, [1], 16, [384 / 1024**3, 192 / 1024**3])

    if mutation:
        with pytest.raises(RuntimeError, match="restore physical"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result["fits_budget"] and result["rollback_verified"]
        assert result["additional_mapped_bytes"] == 192


@pytest.mark.parametrize("scenario", ("packing", "pressure"))
def test_allocator_witness_uses_actual_peaks_not_conservative_plan(modules, tmp_path, scenario):
    counts = modules.benchmark.initial_counts(8, scenario)
    page_bytes = 96 * 1024**2
    observations = [[{"requests": [{"id": f"{model}-{rank}-{index}"} for index in range(counts[model][rank])],
                      "pages": [{"id": 0, "owners": ["pinned"], "pinned": True, "memory_gib": 0.09375}],
                      "mapped_bytes": page_bytes * (2 if rank == 0 and (scenario == "packing" or model == 1) else 1),
                      "page_bytes": page_bytes, "block_size": 16, "page_block_capacity": 64}
                     for rank in (0, 1)] for model in (0, 1)]
    if scenario == "pressure":
        observations[1][0]["pages"].append({"id": 1, "owners": ["exclusive-B"], "pinned": False,
                                            "memory_gib": 0.09375})
    order = []
    controller = SimpleNamespace()

    async def pause():
        order.append("pause")

    async def begin(transaction):
        order.append("begin")
        return observations

    async def probe(transaction, model, rank, prompt, output, budgets):
        order.append(f"probe{rank}")
        assert model == 0
        return {"rank": rank, "fits_budget": scenario == "packing" and rank == 0,
                "additional_mapped_bytes": 192 * 1024**2 if rank == 0 and scenario == "packing" else 288 * 1024**2,
                "allocation_peak_gib": [0.5625, 0.1875] if rank == 0 else [0.375, 0.46875]}

    async def snapshot():
        return observations

    async def capacity(budgets):
        order.append("capacity")

    async def end(transaction):
        order.append("end")

    async def resume():
        order.append("resume")

    async def admit(rank):
        order.append("admit")
        assert rank == 0

    controller.pause, controller.begin, controller.probe = pause, begin, probe
    controller.snapshots, controller.check_capacity = snapshot, capacity
    controller.end, controller.resume = end, resume
    args = SimpleNamespace(scenario=scenario, policy="no_move", output_tokens=256, output=tmp_path)
    async def run():
        controller.lock = asyncio.Lock()
        return await modules.benchmark.allocation_preflight(controller, args,
                                                           [1] * (2816 if scenario == "pressure" else 2048), counts, admit)

    result = asyncio.run(run())
    assert result["admitted_while_paused"] is (scenario == "packing")
    assert order == ["pause", "begin", "probe0", "probe1"] + (["admit", "capacity"] if scenario == "packing" else []) + ["end", "resume"]
    assert result["capacity_gib"] == ([0.5625, 0.375] if scenario == "packing" else [0.46875, 0.28125])


def test_pressure_handoff_requires_b_release_and_new_a_pages(modules):
    before = [[{"mapped_bytes": 96 * 1024**2}] * 2, [{"mapped_bytes": 192 * 1024**2}, {"mapped_bytes": 96 * 1024**2}]]
    after_migrations = [[{"mapped_bytes": 96 * 1024**2}] * 2, [{"mapped_bytes": 96 * 1024**2}] * 2]
    after = [[{"mapped_bytes": 384 * 1024**2, "requests": [{"id": "arriving-A-1234", "blocks": [1, 2]}]},
              {"mapped_bytes": 96 * 1024**2, "requests": []}],
             [{"mapped_bytes": 96 * 1024**2, "requests": []}] * 2]
    moves = [{"model": 1, "source": 0, "source_mapped_after": 96 * 1024**2}]
    result = modules.benchmark.validate_capacity_handoff(before, after_migrations, after, moves, "arriving-A")
    assert result["arrival_new_a_mapped_bytes"] == 288 * 1024**2
    moves[0]["source_mapped_after"] = 192 * 1024**2
    with pytest.raises(RuntimeError, match="release a B"):
        modules.benchmark.validate_capacity_handoff(before, after_migrations, after, moves, "arriving-A")


@pytest.mark.parametrize("prompt,output", (([], 16), ([1], 0), ([1], -1), ([-1], 16), ([1] * 4096, 16)))
def test_probe_rejects_invalid_reservations(modules, scheduler, monkeypatch, prompt, output):
    sampling = ModuleType("vllm")
    sampling.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", sampling)
    monkeypatch.setattr(sys.modules["vllm.v1.request"], "Request", SimpleNamespace, raising=False)
    with pytest.raises(ValueError, match="valid token IDs"):
        scheduler.placement_probe("epoch", prompt, output)
    scheduler.kv_cache_manager.allocate_slots.assert_not_called()


def test_pressure_fails_if_own_model_repacking_can_fit(modules):
    result = {"probes": [{"rank": 0, "fits_budget": False}, {"rank": 1, "fits_budget": False}],
              "capacity_gib": [0.46875, 0.28125], "no_b_migration_lower_bound_gib": [0.46875, 0.375],
              "releasable_pages": [{"model": 1, "rank": 0, "owners": ["B"]}]}
    with pytest.raises(RuntimeError, match="A-only reordering"):
        modules.benchmark.validate_probe_result(SimpleNamespace(scenario="pressure"), result)


@pytest.mark.parametrize("consensus", (False, True))
def test_fixed_fast_pause_sync_preserves_collective_consensus(modules, scheduler, monkeypatch, consensus):
    monkeypatch.setenv("DP_OPT_PAUSE_SYNC_INTERVAL", "1")
    sync = Mock(return_value=(not consensus, consensus))
    config = ModuleType("vllm.config")
    config.ParallelConfig = SimpleNamespace(sync_dp_state=sync)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    engine = SimpleNamespace(scheduler=scheduler, step_counter=4, dp_group="group", pending_pause=True,
                             ignore_start_dp_wave=False, _has_global_unfinished_reqs=Mock(), collective_rpc=Mock())
    modules.scheduler.bind_engine(engine)
    assert engine._has_global_unfinished_reqs(True) is not consensus
    sync.assert_called_once_with("group", has_unfinished=True, pending_pause=True)
    assert engine.step_counter == 5 and engine.ignore_start_dp_wave is consensus
    assert engine.pending_pause is not consensus


def test_fast_pause_interval_rejects_unsupported_scope(modules, scheduler, monkeypatch):
    monkeypatch.setenv("DP_OPT_PAUSE_SYNC_INTERVAL", "2")
    with pytest.raises(ValueError, match="interval"):
        modules.scheduler.bind_engine(SimpleNamespace(scheduler=scheduler))


@pytest.mark.parametrize("active", [True, False])
def test_default32_boundary_pause_keeps_decode_until_consensus(modules, scheduler, active):
    scheduler.pause_state = "running"
    scheduler.set_pause_state = lambda state: setattr(scheduler, "pause_state", state)
    engine = SimpleNamespace(scheduler=scheduler, pending_pause=False, engines_running=active,
                             ignore_start_dp_wave=False, step_counter=30, batch_queue=[], dp_rank=0,
                             _idle_state_callbacks=[])

    def sync(local_unfinished):
        engine.step_counter += 1
        if engine.step_counter % 32 == 0:
            engine.pending_pause = False
            engine.ignore_start_dp_wave = True
        return local_unfinished or engine.pending_pause

    engine._has_global_unfinished_reqs = sync
    modules.scheduler.install_boundary_pause(engine)
    future = engine.placement_pause_boundary()
    assert engine.engines_running and engine.pending_pause
    assert scheduler.pause_state == "running" and not future.done()
    assert engine._has_global_unfinished_reqs(active)
    assert scheduler.pause_state == "running" and not future.done()
    assert not engine._has_global_unfinished_reqs(active)
    assert scheduler.pause_state == "paused" and not engine.pending_pause
    assert engine.ignore_start_dp_wave and not engine.placement_boundary_pause_pending
    engine._idle_state_callbacks.pop()(engine)
    receipt = future.result()
    assert receipt["boundary_step_counter"] == 32 and receipt["rank"] == 0
    assert receipt["armed_at"] <= receipt["reached_at"] <= receipt["callback_at"]


@pytest.mark.parametrize("fault", ["pending", "paused", "inflight", "off_boundary"])
def test_boundary_pause_rejects_unsafe_state(modules, scheduler, fault):
    scheduler.pause_state = "paused" if fault == "paused" else "running"
    engine = SimpleNamespace(scheduler=scheduler, pending_pause=fault == "pending", engines_running=True,
                             ignore_start_dp_wave=False, step_counter=31, dp_rank=0,
                             batch_queue=["batch"] if fault == "inflight" else [], _idle_state_callbacks=[])

    def sync(local_unfinished):
        engine.step_counter += 0 if fault == "off_boundary" else 1
        engine.pending_pause = False
        engine.ignore_start_dp_wave = True
        return local_unfinished

    engine._has_global_unfinished_reqs = sync
    modules.scheduler.install_boundary_pause(engine)
    if fault in ("pending", "paused"):
        with pytest.raises(RuntimeError):
            engine.placement_pause_boundary()
    else:
        engine.placement_pause_boundary()
        with pytest.raises(RuntimeError, match="default32"):
            engine._has_global_unfinished_reqs(True)


def test_boundary_pause_controller_orders_idle_tenant_first(modules, tmp_path):
    order = []

    async def call(model, rank, method):
        assert method == "placement_pause_boundary"
        order.append(model)
        return {"rank": rank}

    observations = [[{"requests": ["active"]}, {"requests": []}],
                    [{"requests": []}, {"requests": []}]]

    async def run():
        controller = modules.controller.PlacementController([], {}, tmp_path)
        controller.call = call
        return await controller.pause_boundary(observations)

    assert asyncio.run(run()) == [[{"rank": 0}, {"rank": 1}]] * 2
    assert order == [1, 1, 0, 0]


@pytest.mark.parametrize("changed", [None, "generation", "ownership", "after_boundary", "infeasible"])
def test_preplan_solves_before_pause_and_rejects_stale_layout(modules, tmp_path, monkeypatch, changed):
    order = []
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
             "block_bytes": 1024, "page_bytes": 65536, "uniform_page_geometry": True,
             "mapped_free_block_slots": 0, "page_block_capacity": 64, "free_blocks": 10000}
    observed = [[dict(empty) for rank in (0, 1)] for model in (0, 1)]
    monkeypatch.setattr(modules.controller, "performance_tables", lambda *args: [[0, 0], [0]])

    def solve(snapshot, **kwargs):
        order.append("solve")
        return {"winner": None if changed == "infeasible" else
                {"target": 0, "moves": [], "objective_s": 0}}

    monkeypatch.setattr(modules.controller, "enumerate_targets", solve)

    async def run():
        controller = modules.controller.PlacementController([], {}, tmp_path, allow_extrapolation=True)

        async def snapshots(*args, **kwargs):
            return observed

        async def pause_boundary(current):
            order.append("pause_boundary")
            return [[{"reached_at": 0}] * 2] * 2

        async def begin(transaction):
            order.append("begin")
            if changed == "after_boundary":
                observed[0][0] = {**empty, "mapped_bytes": 65536}
            return observed

        async def end(transaction):
            order.append("end")

        async def resume():
            order.append("resume")

        async def admit(target):
            order.append("admit")

        controller.snapshots = snapshots
        controller.pause_boundary, controller.begin, controller.end, controller.resume = pause_boundary, begin, end, resume
        prepared = await controller.prepare_admission(16, 16, [1, 1])
        assert order == ["solve"]
        if changed == "generation":
            controller.admission_generation += 1
        elif changed == "ownership":
            observed[0][0] = {**empty, "mapped_bytes": 65536}
        return await controller.schedule_admission(16, 16, [1, 1], admit, prepared=prepared)

    result = asyncio.run(run())
    if changed in ("generation", "ownership", "infeasible"):
        assert order == ["solve"] and result["pause_call_s"] == 0
        assert not result["admitted"]
        assert result["observations"] == observed and result["plan"] is not None
        assert result["snapshot"] is not None and result["winner"] is None
    elif changed == "after_boundary":
        assert order == ["solve", "pause_boundary", "begin", "end", "resume"]
        assert result["reason"] == "preplan_stale_after_boundary"
        assert result["observations"] == observed and result["plan"] is not None
    else:
        assert order == ["solve", "pause_boundary", "begin", "admit", "end", "resume"]
        assert result["admitted"] and result["solve_s"] == 0 and result["preplan_s"] > 0


@pytest.mark.parametrize("limit", (None, 4096, 32768))
def test_engine_arguments_preserve_sparse_graph_with_configurable_length(modules, monkeypatch, limit):
    arguments = ModuleType("vllm.engine.arg_utils")
    arguments.AsyncEngineArgs = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arguments)
    options = {} if limit is None else {"max_model_len": limit}
    args = modules.benchmark.engine_arguments("checkpoint", 31000, 4294967296, **options)
    assert args.max_model_len == (32768 if limit is None else limit)
    assert args.max_num_batched_tokens == 4096
    assert args.kv_cache_memory_bytes == 4294967296
    assert args.compilation_config["cudagraph_capture_sizes"] == [1, 8, 16, 32, 64, 128, 256]
    assert args.compilation_config["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert args.data_parallel_size == 2 and args.tensor_parallel_size == 1


def test_benchmark_defaults_support_full_32k_sequence(modules):
    args = modules.benchmark.parse_args(["--checkpoint", "checkpoint", "--fit", "fit.json",
                                       "--output", "result", "--input-tokens", "32256",
                                       "--arrival-input-tokens", "32256", "--output-tokens", "512"])
    assert args.max_model_len == 32768 and args.cache_bytes == 4294967296


@pytest.mark.parametrize("options", (("--max-model-len", "0"), ("--max-model-len", "-1"),
                                     ("--input-tokens", "32257", "--output-tokens", "512"),
                                     ("--max-model-len", "4096", "--input-tokens", "8192")))
def test_benchmark_rejects_lengths_outside_configured_limit(modules, options):
    with pytest.raises(SystemExit) as error:
        modules.benchmark.parse_args(["--checkpoint", "checkpoint", "--fit", "fit.json",
                                      "--output", "result", *options])
    assert error.value.code == 2


@pytest.mark.parametrize("prompt_tokens,accepted", ((4097, True), (32752, True), (32753, False)))
def test_probe_uses_32k_full_reservation_limit(modules, scheduler, monkeypatch, prompt_tokens, accepted):
    scheduler.max_model_len = 32768
    sampling = ModuleType("vllm")
    sampling.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", sampling)
    monkeypatch.setattr(sys.modules["vllm.v1.request"], "Request", SimpleNamespace, raising=False)
    if accepted:
        result = scheduler.placement_probe("epoch", [1] * prompt_tokens, 16)
        assert result["allocated"]
        allocation = scheduler.kv_cache_manager.allocate_slots.call_args
        assert allocation.args[1] == prompt_tokens + 16
        scheduler.kv_cache_manager.free.assert_called_once()
        assert not scheduler.requests
    else:
        with pytest.raises(ValueError, match="supported full reservation"):
            scheduler.placement_probe("epoch", [1] * prompt_tokens, 16)
        scheduler.kv_cache_manager.allocate_slots.assert_not_called()
