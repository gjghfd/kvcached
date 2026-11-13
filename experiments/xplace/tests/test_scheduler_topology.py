import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def schedulers(monkeypatch):
    class Scheduler:
        def __init__(self, tp=1, dp=2, pp=1, asynchronous=False, prefix=False, groups=1,
                     connector=None, lookahead=0):
            self.parallel_config = SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp,
                                                   pipeline_parallel_size=pp)
            self.scheduler_config = SimpleNamespace(async_scheduling=asynchronous)
            self.cache_config = SimpleNamespace(enable_prefix_caching=prefix)
            self.kv_cache_config = SimpleNamespace(kv_cache_groups=[None] * groups)
            self.connector = connector
            self.num_lookahead_tokens = lookahead

    for name, contents in {
        "vllm.v1.core.sched.scheduler": {"Scheduler": Scheduler},
        "vllm.v1.core.sched.interface": {"PauseState": SimpleNamespace(PAUSED_ALL="paused")},
        "vllm.v1.request": {"RequestStatus": SimpleNamespace()},
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**contents))
    path = Path(__file__).resolve().parents[1] / "runtime/live_scheduler.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("scheduler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tp2_calibration_initializes_without_enabling_tp2_migration(schedulers):
    calibration = schedulers.CalibrationScheduler(tp=2)
    assert calibration.placement_mode == "full"
    assert calibration.placement_preemptions == 0
    with pytest.raises(ValueError, match="TP1"):
        schedulers.PlacementScheduler(tp=2)
    assert schedulers.PlacementScheduler(tp=1).placement_mode == calibration.placement_mode
    for method in (calibration.placement_begin, calibration._require_transaction):
        with pytest.raises(RuntimeError, match="does not support migration"):
            method("transaction")
    assert schedulers.TensorParallelPlacementScheduler(tp=2).placement_mode == "full"


@pytest.mark.parametrize("invalid", [
    {"tp": 0}, {"dp": 1}, {"pp": 2}, {"asynchronous": True}, {"prefix": True},
    {"groups": 2}, {"connector": object()}, {"lookahead": 1},
])
def test_calibration_keeps_other_scheduler_safety_checks(schedulers, invalid):
    with pytest.raises(ValueError):
        schedulers.CalibrationScheduler(**{"tp": 2, **invalid})


def test_snapshot_and_summary_keep_phase_distinct_from_movable(schedulers):
    scheduler = schedulers.CalibrationScheduler(tp=2)
    scheduler.parallel_config.data_parallel_rank = 0
    scheduler.cache_config.block_size = 16
    scheduler.placement_mode = "prompt"
    scheduler.pause_state = "paused"
    manager = SimpleNamespace(in_shrink=False, block_mem_size=16, num_layers=1, num_kv_buffers=1,
                              page_size=64, num_avail_blocks=1, reserved_blocks=[], full_pages=[1], avail_pages=[],
                              get_mapped_memory_size=lambda: 1024, _memory_limit_page_bundle_bytes=lambda: 1024,
                              page_allocator=SimpleNamespace(group_indices_by_page=lambda blocks, size: {0: [0]}
                                                             if blocks == [0] else {1: list(blocks)}))
    scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=lambda identifier: [[1]],
                                                block_pool=SimpleNamespace(kv_cache_manager=manager,
                                                                           get_num_free_blocks=lambda: 10))
    requests = []
    for name, computed, output, inflight in [("decode", 128, 28, 1), ("prefill", 50, 0, 0),
                                            ("waiting", 120, 20, 0)]:
        requests.append(SimpleNamespace(request_id=name, num_computed_tokens=computed, num_prompt_tokens=100,
                                        num_output_tokens=output, num_in_flight_tokens=inflight))
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.running = requests[:2]
    for view in (scheduler.placement_snapshot(), scheduler.placement_summary()):
        assert view["phase_schema"] == 1 and view["scheduler_paused"] is True
        records = {request["id"]: request for request in view["requests"]}
        assert {name: record["phase"] for name, record in records.items()} == {
            "decode": "decode", "prefill": "prefill", "waiting": "waiting"}
        assert records["decode"]["in_flight_tokens"] == 1
        assert not records["decode"].get("movable", records["decode"].get("decoding"))
        assert records["decode"]["prompt_tokens"] == 100
