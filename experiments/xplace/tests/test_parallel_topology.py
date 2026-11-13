import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from engine import engine_arguments
from imbalance_calibration import fits_budget, scan_cases
from main import measure_budget, parse_args
from parallel_topology import minimum_free_bytes, verify_workers, worker_filename


def write_workers(root, num_ranks, tp_size):
    paths = []
    for model in (0, 1):
        for rank in range(num_ranks):
            for lane in range(tp_size):
                index = rank * tp_size + lane
                record = {"service": f"instance-{model + 1}", "dp_rank": rank, "gpu_uuid": f"gpu{index}",
                          "groups": {"tp": {"rank": lane, "world_size": tp_size},
                                     "dp": {"rank": rank, "world_size": num_ranks},
                                     "ep": {"rank": index, "world_size": num_ranks * tp_size}},
                          "expert_ids": [list(range(index * 32, (index + 1) * 32))] * 2,
                          "kv_runtime": {"initialized": True, "world_size": tp_size},
                          "ipc_name": f"model{model}_dp{rank}", "kv_num_blocks": 65536,
                          "kv_block_size": 16, "parameter_bytes": 16 * 1024**3}
                path = root / worker_filename(record["service"], rank, lane, tp_size)
                path.write_text(json.dumps(record))
                paths.append(path)
    return paths


@pytest.mark.parametrize("num_ranks,tp_size", [(4, 1), (2, 2)])
def test_four_distinct_gpus_and_complementary_ep_shards(tmp_path, num_ranks, tp_size):
    paths = write_workers(tmp_path, num_ranks, tp_size)
    assert len(set(paths)) == 8
    topology = verify_workers(tmp_path, num_ranks, tp_size)
    assert topology["expert_parallel_size"] == topology["physical_gpus"] == 4
    assert topology["tensor_parallel_size"] == tp_size


@pytest.mark.parametrize("damage", ["gpu", "ep", "experts", "namespace", "capacity", "kv", "colocation"])
def test_corrupt_tp_topology_rejected(tmp_path, damage):
    paths = write_workers(tmp_path, 2, 2)
    path = paths[4] if damage == "colocation" else paths[1]
    record = json.loads(path.read_text())
    if damage == "gpu":
        record["gpu_uuid"] = "gpu0"
    elif damage == "ep":
        record["groups"]["ep"]["rank"] = 0
    elif damage == "experts":
        record["expert_ids"][0] = list(range(32))
    elif damage == "namespace":
        record["ipc_name"] = "independent_tp_worker"
    elif damage == "capacity":
        record["kv_num_blocks"] -= 1
    elif damage == "kv":
        record["kv_runtime"]["world_size"] = 1
    else:
        record["gpu_uuid"] = "other_gpu"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        verify_workers(tmp_path, 2, 2)


def memory_snapshots():
    return [[[dict(gpu_uuid=f"gpu{lane}", free_mib=free) for lane, free in enumerate(values)]]
            for values in ((10000, 2000), (9000, 3000))]


def test_budget_uses_worst_tp_lane():
    class Controller:
        num_ranks = 1
        tensor_parallel_size = 2

        async def memory(self, release_unused=False):
            return memory_snapshots()

        async def summaries(self):
            return [[{"mapped_bytes": 1024**3}], [{"mapped_bytes": 1024**3}]]

    result = asyncio.run(measure_budget(Controller(), 0.5))
    assert result["budget_gib"] == [3.25]
    assert result["capacity_schema"] == 1
    assert result["num_ranks"] == 1 and result["tensor_parallel_size"] == 2
    assert result["summary_snapshots"] == [[{"mapped_bytes": 1024**3}], [{"mapped_bytes": 1024**3}]]
    assert result["rank_calculations"] == [dict(rank=0, minimum_free_bytes=2000 * 1024**2,
        mapped_bytes_by_model=[1024**3, 1024**3], mapped_bytes=2 * 1024**3,
        unrounded_budget_gib=3.453125, rounded_budget_gib=3.25)]
    times = result["measurement_times"]
    assert times["memory_requested_at"] <= times["memory_received_at"] <= times["summaries_received_at"]


@pytest.mark.parametrize("mapped", [-1, True, 1.5, float("nan"), float("inf")])
def test_budget_rejects_invalid_mapped_capacity_input(mapped):
    class Controller:
        num_ranks = 1
        tensor_parallel_size = 2

        async def memory(self, release_unused=False):
            return memory_snapshots()

        async def summaries(self):
            return [[{"mapped_bytes": mapped}], [{"mapped_bytes": 0}]]

    with pytest.raises(ValueError, match="mapped-KV"):
        asyncio.run(measure_budget(Controller(), 0.5))


def test_missing_tp_lane_and_mismatched_colocation_rejected():
    memory = memory_snapshots()
    memory[1][0][1]["gpu_uuid"] = "wrong"
    with pytest.raises(RuntimeError):
        minimum_free_bytes(memory, 0, 2)
    memory[1][0].pop()
    with pytest.raises(RuntimeError):
        minimum_free_bytes(memory, 0, 2)


def test_calibration_budget_uses_per_shard_page_geometry():
    controller = SimpleNamespace(budgets=[0.501])
    shard = {"block_size": 16, "uniform_page_geometry": True, "page_block_capacity": 4,
             "page_bytes": 1024**2, "mapped_free_block_slots": 0, "mapped_bytes": 0, "free_blocks": 100}
    summaries = [[dict(shard)], [dict(shard)]]
    assert fits_budget(controller, summaries, 1, [1], [[16]], 16)
    summaries[1][0]["page_bytes"] *= 2
    assert not fits_budget(controller, summaries, 1, [1], [[16]], 16)


def tp2_config():
    return {"num_ranks": 2, "tensor_parallel_size": 2, "seed": 7, "context_tokens": 4096,
            "output_tokens": 1024, "repeats": 3, "case_timeout_s": 120, "max_scan_s": 2800,
            "series": [{"batch_size": 128, "deltas": [0, 1, 32, 64]}]}


def test_tp2_scan_rotates_two_attention_groups():
    cases = scan_cases(tp2_config(), 2, 2)
    assert len(cases) == 12
    for case in cases:
        assert len(case["layout"]) == 2 and sum(case["layout"]) == 128
        assert case["imbalance"] == 2 * case["delta"]
    extreme = {case["repeat"]: case["layout"] for case in cases if case["delta"] == 64}
    assert extreme == {0: [128, 0], 1: [0, 128], 2: [128, 0]}
    with pytest.raises(ValueError):
        scan_cases(tp2_config(), 4, 1)


def test_tp2_cli_rejects_unverified_phases_and_mismatched_config(tmp_path):
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(tp2_config()))
    args = ["--checkpoint", "test", "--output", "out", "--prompt-bank", "bank.json",
            "--tensor-parallel-size", "2", "--num-ranks", "2", "--imbalance-config", str(path)]
    parsed = parse_args(args + ["--phases", "imbalance"])
    assert parsed.tensor_parallel_size == parsed.num_ranks == 2
    for extra in (["--phases", "dataplane"], ["--phases", "trace", "--trace", "trace.json"],
                  ["--phases", "imbalance", "--num-ranks", "4"],
                  ["--phases", "imbalance", "--tensor-parallel-size", "1"]):
        with pytest.raises(SystemExit):
            parse_args(args + extra)


def test_engine_arguments_preserve_ep4_when_enabling_tp2(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", SimpleNamespace(AsyncEngineArgs=SimpleNamespace))
    args = engine_arguments("test", 31000, 48 * 1024**3, 131072, 2, tensor_parallel_size=2)
    assert args.tensor_parallel_size == args.data_parallel_size == 2
    assert args.enable_expert_parallel and args.enable_ep_weight_filter
    assert args.scheduler_cls == "live_scheduler.CalibrationScheduler"
    tp1 = engine_arguments("test", 31000, 48 * 1024**3, 131072, 4)
    assert tp1.tensor_parallel_size == 1
    assert tp1.scheduler_cls == "live_scheduler.PlacementScheduler"
    migrated = engine_arguments("test", 31000, 48 * 1024**3, 131072, 2,
                                tensor_parallel_size=2, enable_tp_migration=True)
    assert migrated.scheduler_cls == "live_scheduler.TensorParallelPlacementScheduler"


def test_tp2_migration_opt_in_requires_dataplane_and_measured_cost(tmp_path):
    from test_measured_cost import write_measured_cost_fixture

    base = ["--checkpoint", "test", "--output", "out", "--prompt-bank", "bank.json",
            "--tensor-parallel-size", "2", "--num-ranks", "2", "--enable-tp-migration"]
    assert parse_args(base + ["--phases", "dataplane"]).enable_tp_migration
    measured = write_measured_cost_fixture(tmp_path)
    trace = ["--trace", "trace.json", "--kv-mode", "prompt", "--step-params", str(measured)]
    assert parse_args(base + ["--phases", "dataplane,trace"] + trace).kv_mode == "prompt"
    for phases in ("trace", "trace,dataplane", "ilayout,dataplane,trace"):
        with pytest.raises(SystemExit):
            parse_args(base + ["--phases", phases] + trace)
    with pytest.raises(SystemExit):
        parse_args(base + ["--phases", "dataplane,trace", "--trace", "trace.json"])
    measured.write_text("{}")
    with pytest.raises(SystemExit):
        parse_args(base + ["--phases", "dataplane,trace"] + trace)


def test_tp2_grid_preserves_total_batch_under_imbalance():
    from calibrate import conv_grid

    cases = {case["name"]: case for case in conv_grid(2)}
    for count in (8, 16):
        for delta in (2, 4, 8):
            case = cases[f"skew{count}d{delta}_c4k"]
            assert case["layout"] == [count + delta, count - delta]
            assert sum(case["layout"]) == 2 * count
