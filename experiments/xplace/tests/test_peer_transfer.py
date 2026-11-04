import sys
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

import peer_transfer  # noqa: E402

WORLD = 4


class FakeGroup:
    """Thread-based all_gatherv: every rank contributes; each receives the concatenation along dim 0."""

    def __init__(self, world):
        self.world, self.barrier = world, threading.Barrier(world)
        self.slots, self.lock = {}, threading.Lock()

    def bind(self, rank):
        group = self

        class RankView:
            rank_in_group, world_size = rank, group.world

            def all_gatherv(self, tensors, dim=0, sizes=None):
                assert dim == 0 and len(sizes) == group.world and len(tensors) == 1
                assert tensors[0].shape[0] == sizes[rank], (rank, tensors[0].shape, sizes)
                with group.lock:
                    group.slots[rank] = tensors[0].clone()
                group.barrier.wait()
                result = torch.cat([group.slots[peer] for peer in range(group.world)], dim=0)
                group.barrier.wait()
                return [result]

        return RankView()


def run_transfer(source, destination, blocks_by_rank, num_blocks, views_by_rank):
    group = FakeGroup(WORLD)
    results, errors = {}, []
    local = threading.local()
    module = types.ModuleType("vllm.distributed.parallel_state")
    module.get_dp_group = lambda: local.group
    saved = {name: sys.modules.get(name) for name in ("vllm.distributed", "vllm.distributed.parallel_state")}
    sys.modules["vllm.distributed"] = types.ModuleType("vllm.distributed")
    sys.modules["vllm.distributed.parallel_state"] = module
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *args, **kwargs: None

    def worker(rank):
        local.group = group.bind(rank)
        try:
            results[rank] = peer_transfer.transfer_blocks(types.SimpleNamespace(device=torch.device("cpu")),
                                                          views_by_rank[rank], blocks_by_rank.get(rank, []),
                                                          source, destination, num_blocks, chunk_mib=1)
        except BaseException as error:  # surfaced below
            errors.append((rank, error))
            group.barrier.abort()

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(WORLD)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        torch.cuda.synchronize = original
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return results, errors


def make_views(seed, block_shape=(3, 4096)):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 255, (64, *block_shape), generator=generator, dtype=torch.uint8) for _ in range(2)]


def test_source_blocks_land_bytewise_on_destination_only():
    views = {rank: make_views(rank) for rank in range(WORLD)}
    before = {rank: [view.clone() for view in views[rank]] for rank in range(WORLD)}
    source_blocks, destination_blocks = [3, 9, 10, 40, 41], [50, 51, 52, 7, 8]
    results, errors = run_transfer(1, 3, {1: source_blocks, 3: destination_blocks}, 5, views)
    assert not errors, errors
    assert {results[rank]["role"] for rank in range(WORLD)} == {"source", "destination", "bystander"}
    for view_index in range(2):
        assert torch.equal(views[3][view_index][destination_blocks], before[1][view_index][source_blocks])
        for rank in (0, 1, 2):
            assert torch.equal(views[rank][view_index], before[rank][view_index])
    assert all(results[rank]["bitwise_verified"] for rank in range(WORLD))


def test_chunking_with_many_blocks():
    views = {rank: make_views(10 + rank, block_shape=(16, 4096)) for rank in range(WORLD)}
    source_blocks = list(range(1, 33))
    destination_blocks = list(range(32, 64))
    assert peer_transfer.transfer_geometry(views[0], source_blocks, 32, 0, 2, 0, WORLD, 1)["chunk_blocks"] == 16
    expected = [view[source_blocks].clone() for view in views[0]]
    results, errors = run_transfer(0, 2, {0: source_blocks, 2: destination_blocks}, 32, views)
    assert not errors, errors
    for view_index in range(2):
        assert torch.equal(views[2][view_index][destination_blocks], expected[view_index])


def test_bystander_with_blocks_is_rejected():
    views = {rank: make_views(rank) for rank in range(WORLD)}
    results, errors = run_transfer(0, 1, {0: [1, 2], 1: [3, 4], 2: [5, 6]}, 2, views)
    assert errors and any(rank == 2 for rank, _ in errors)
