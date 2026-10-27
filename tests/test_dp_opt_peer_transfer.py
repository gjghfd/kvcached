import asyncio
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

@pytest.fixture
def modules(monkeypatch):
    setup = importlib.import_module("test_dp_opt_live_placement").modules
    return setup.__wrapped__(monkeypatch)


@pytest.fixture
def peer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("peer_transfer")


def backing(shape=(100, 1536 * 1024), element_bytes=1):
    return SimpleNamespace(shape=shape, dtype="uint8", element_size=lambda: element_bytes)


def test_geometry_bounds_staging_and_fingerprints_layout(peer):
    source = peer.transfer_geometry([backing()], [1, 2, 3], 0, 0, 32)
    destination = peer.transfer_geometry([backing()], [7, 8, 9], 0, 1, 32)
    assert source == destination
    assert source["chunk_blocks"] == 21
    assert source["bytes"] == 3 * 1536 * 1024
    changed = peer.transfer_geometry([backing(element_bytes=2)], [1, 2, 3], 0, 0, 32)
    assert changed["metadata"][-1] != source["metadata"][-1]


@pytest.mark.parametrize("blocks", [[], [0], [-1], [100], [1, 1], [True], [1.5]])
def test_geometry_rejects_invalid_blocks(peer, blocks):
    with pytest.raises(ValueError):
        peer.transfer_geometry([backing()], blocks, 0, 0, 32)


@pytest.mark.parametrize("chunk", [0, 65, True, 1.5])
def test_geometry_rejects_invalid_chunk(peer, chunk):
    with pytest.raises(ValueError):
        peer.transfer_geometry([backing()], [1], 0, 0, chunk)


def test_geometry_never_uses_oversized_backing_blocks(peer):
    with pytest.raises(ValueError, match="exceeds"):
        peer.transfer_geometry([backing((10, 2 * 1024**2))], [1], 0, 0, 1)


def test_peer_controller_uses_concurrent_ranks_and_commits_after_integrity(modules, tmp_path):
    from test_dp_opt_live_placement import controller_with_calls

    controller, calls, engine = controller_with_calls(modules, tmp_path)
    controller.kv_transport = "dp_allgatherv"
    original = controller.call
    started = []

    async def call(model, rank, method, *args):
        if method == "placement_kv_peer":
            started.append(rank)
            await asyncio.sleep(0)
            assert set(started) == {0, 1}
            calls.append((rank, method, args))
            return {"sha256": "same", "bytes": 64, "stage_s": {"data": rank + 1}}
        return await original(model, rank, method, *args)

    controller.call = call
    result = asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert engine.engine_core.reqs_in_flight["request"] == "destination"
    assert result["kv_bytes"] == 64
    assert result["worker_transfer_stages_s"] == {"export": {"data": 1}, "import": {"data": 2}}
    assert "peer_kv_rpc" in result["stage_s"]
    assert "placement_kv_file" not in [method for rank, method, args in calls]
    assert [args[1] for rank, method, args in calls if method == "placement_kv_peer"] == [[1, 2], [7, 8]]


def test_uncertain_peer_collective_keeps_both_mapping_owners(modules, tmp_path):
    from test_dp_opt_live_placement import controller_with_calls

    controller, calls, engine = controller_with_calls(modules, tmp_path)
    controller.kv_transport = "dp_allgatherv"
    original = controller.call

    async def call(model, rank, method, *args):
        if method == "placement_kv_peer":
            raise RuntimeError("collective timeout")
        return await original(model, rank, method, *args)

    controller.call = call
    with pytest.raises(RuntimeError, match="retain source and destination"):
        asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert not controller.events
    assert engine.engine_core.reqs_in_flight["request"] == "source"
    assert not {"placement_rollback", "placement_remove"} & {method for rank, method, args in calls}


def test_completed_peer_checksum_failure_can_roll_back_destination(modules, tmp_path):
    from test_dp_opt_live_placement import controller_with_calls

    controller, calls, engine = controller_with_calls(modules, tmp_path)
    controller.kv_transport = "dp_allgatherv"
    original = controller.call

    async def call(model, rank, method, *args):
        if method == "placement_kv_peer":
            return {"sha256": str(rank), "bytes": 64}
        return await original(model, rank, method, *args)

    controller.call = call
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert "placement_rollback" in [method for rank, method, args in calls]
    assert "placement_remove" not in [method for rank, method, args in calls]


@pytest.mark.parametrize("proved", [True, False])
def test_echo_controller_commits_only_with_full_device_bytewise_proof(modules, tmp_path, proved):
    from test_dp_opt_live_placement import controller_with_calls

    controller, calls, engine = controller_with_calls(modules, tmp_path)
    controller.kv_transport = "dp_echo"
    original = controller.call

    async def call(model, rank, method, *args):
        if method == "placement_kv_peer":
            assert args[-1] == "device_echo"
            return {"sha256": "receipt", "bytes": 64, "integrity": "device_echo_full_bytewise",
                    "bitwise_verified": proved, "sha256_scope": "verified_integrity_receipt_not_KV_content"}
        return await original(model, rank, method, *args)

    controller.call = call
    if proved:
        result = asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
        assert result["bitwise_verified"] is True
        assert result["sha256_scope"] == "verified_integrity_receipt_not_KV_content"
    else:
        with pytest.raises(RuntimeError, match="bytewise integrity"):
            asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
        assert "placement_remove" not in [method for rank, method, args in calls]
        assert "placement_rollback" in [method for rank, method, args in calls]


def test_host_transport_does_not_compare_instrumentation_as_integrity(modules, tmp_path):
    from test_dp_opt_live_placement import controller_with_calls

    controller, calls, engine = controller_with_calls(modules, tmp_path)
    original = controller.call

    async def call(model, rank, method, *args):
        result = await original(model, rank, method, *args)
        if method == "placement_kv_file":
            result["stage_s"] = {"different_each_rank": rank + 1}
        return result

    controller.call = call
    result = asyncio.run(controller.migrate("epoch", {"id": "request", "model": 0, "source": 0}))
    assert result["kv_bytes"] == 64


@pytest.mark.parametrize("integrity", ["sha256", "device_echo"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_peer_data_roundtrip_scatter_and_real_sha(peer, monkeypatch, integrity, corrupt):
    import hashlib

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    barrier = threading.Barrier(2)
    local = threading.local()
    payloads, sizes_seen = {}, []

    def all_gatherv(tensors, dim, sizes):
        rank = local.rank
        payloads[rank] = tensors[0].clone()
        if corrupt and rank == 1 and sizes == [0, 3]:
            payloads[rank][0, 0] ^= 1
        barrier.wait(timeout=5)
        if rank == 0:
            sizes_seen.append(sizes)
        result = torch.cat([payloads[rank] for rank in (0, 1)], dim=dim)
        barrier.wait(timeout=5)
        return [result]

    groups = [SimpleNamespace(rank_in_group=rank, world_size=2, all_gatherv=all_gatherv) for rank in (0, 1)]
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_dp_group = lambda: groups[local.rank]
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)
    source = torch.arange(8 * 32, dtype=torch.int64).to(torch.uint8).reshape(8, 32)
    destination = torch.zeros_like(source)

    def transfer(rank):
        local.rank = rank
        view = source if rank == 0 else destination
        blocks = [1, 2, 3] if rank == 0 else [4, 5, 6]
        return peer.transfer_blocks(SimpleNamespace(device=torch.device("cpu")), [view], blocks, 0, 1, integrity)

    with ThreadPoolExecutor(max_workers=2) as executor:
        if corrupt and integrity == "device_echo":
            with pytest.raises(RuntimeError, match="differs bytewise"):
                list(executor.map(transfer, (0, 1)))
            return
        exported, received = list(executor.map(transfer, (0, 1)))
    expected = hashlib.sha256(source[[1, 2, 3]].numpy().tobytes()).hexdigest()
    assert exported["sha256"] == received["sha256"]
    if integrity == "sha256":
        assert exported["sha256"] == expected
    else:
        assert exported["bitwise_verified"] is received["bitwise_verified"] is True
        assert exported["sha256_scope"] == "verified_integrity_receipt_not_KV_content"
    assert exported["bytes"] == received["bytes"] == 96
    assert torch.equal(source[[1, 2, 3]], destination[[4, 5, 6]])
    assert torch.count_nonzero(destination[[0, 1, 2, 3, 7]]) == 0
    assert sizes_seen == ([[6, 6], [3, 0]] if integrity == "sha256" else [[6, 6], [3, 0], [0, 3], [1, 1]])
