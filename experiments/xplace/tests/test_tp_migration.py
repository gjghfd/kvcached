import asyncio
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from controller import GIB, PlacementController
from tp_migration import aggregate_peer, verify_runner_results, verify_transfer_endpoints


def peer_shards(rank, role):
    return [{"rank": rank, "tp_rank": lane, "tp_size": 2, "role": role, "num_blocks": 8,
             "integrity": "device_echo_full_bytewise", "transport": "dp", "bytes": 64,
             "bitwise_verified": True, "stage_s": {"copy": lane + 1.0}} for lane in range(2)]


def test_peer_aggregation_preserves_every_shard_and_checks_endpoints():
    source = aggregate_peer(list(reversed(peer_shards(0, "source"))), 2, 0)
    destination = aggregate_peer(peer_shards(1, "destination"), 2, 1)
    assert source["bytes"] == 128 and source["stage_s"]["copy"] == 2.0
    assert [shard["tp_rank"] for shard in source["shards"]] == [0, 1]
    verify_transfer_endpoints(source, destination, 2)
    destination["shards"][1]["bytes"] -= 1
    with pytest.raises(RuntimeError, match="shards do not match"):
        verify_transfer_endpoints(source, destination, 2)


@pytest.mark.parametrize("damage", ["missing", "identity", "group", "bytes", "proof", "role"])
def test_peer_aggregation_rejects_partial_or_inconsistent_results(damage):
    shards = peer_shards(0, "source")
    if damage == "missing":
        shards.pop()
    elif damage == "identity":
        shards[1]["tp_rank"] = 0
    elif damage == "group":
        shards[1]["rank"] = 1
    elif damage == "bytes":
        shards[1]["bytes"] = 0
    elif damage == "proof":
        shards[1]["bitwise_verified"] = False
    else:
        shards[1]["role"] = "destination"
    with pytest.raises(RuntimeError):
        aggregate_peer(shards, 2, 0)


def test_runner_acknowledgment_requires_all_tp_workers():
    shards = [{"rank": 0, "tp_rank": lane, "tp_size": 2, "installed": True} for lane in range(2)]
    assert verify_runner_results(shards, 2, 0)
    shards[1]["installed"] = False
    with pytest.raises(RuntimeError):
        verify_runner_results(shards, 2, 0)


def test_tp_ballast_caps_each_physical_card_independently():
    class Controller(PlacementController):
        def __init__(self):
            super().__init__([None, None], 2, [8.0, 8.0])
            self.tensor_parallel_size = 2
            self.free = [[30.0 * GIB, 25.0 * GIB], [29.0 * GIB, 27.0 * GIB]]
            self.held = {}

        async def summaries(self):
            return [[{"mapped_bytes": 0, "page_bytes": 48 * 1024**2} for _ in range(2)] for _ in range(2)]

        async def memory(self):
            return {model: [[{"gpu_uuid": f"gpu-{rank}-{lane}", "free_mib": self.free[rank][lane] / 1024**2,
                             "total_mib": 80 * 1024} for lane in range(2)] for rank in range(2)] for model in (0, 1)}

        async def call(self, model, rank, method, operation, timeout, args, kwargs):
            if operation == "placement_ballast":
                assert isinstance(args[0], list) and len(args[0]) == 2
                for lane, value in enumerate(args[0]):
                    self.free[rank][lane] -= value
                    key = model, rank, lane
                    self.held[key] = self.held.get(key, 0) + value
                return [{"held_bytes": self.held[model, rank, lane]} for lane in range(2)]
            assert operation == "placement_ballast_release"
            result = []
            for lane in range(2):
                released = self.held.pop((model, rank, lane), 0)
                self.free[rank][lane] += released
                result.append({"released_bytes": released})
            return result

    async def scenario():
        controller = Controller()
        original = copy.deepcopy(controller.free)
        applied = await controller.apply_ballast(8.0, 0.98)
        assert all(abs(value - applied["target_gib"]) < 1e-6
                   for group in applied["worker_capacity_gib"] for value in group)
        held = sum(controller.held.values())
        released = await controller.release_ballast(0.98)
        assert released["released_bytes"] == held
        assert controller.free == original

    asyncio.run(scenario())
