import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def controller_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("live_controller")


def record(mapped_mib=96, requests=()):
    return {"requests": list(requests), "mapped_bytes": mapped_mib * 1024**2,
            "uniform_page_geometry": True, "page_block_capacity": 64,
            "page_bytes": 96 * 1024**2, "block_size": 16,
            "mapped_free_block_slots": 63, "free_blocks": 1000}


@pytest.mark.parametrize("concurrent", (False, True))
@pytest.mark.parametrize("released_model", (0, 1))
def test_completion_reopens_empty_rank_even_if_hot_rank_still_fits(controller_module, tmp_path,
                                                                 concurrent, released_model):
    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        controller.concurrent_admission = concurrent
        arriving_model = 1 - released_model
        before = [[record(), record()], [record(), record()]]
        before[released_model][0] = record(900, ({"id": "completed"},))
        before[arriving_model][1] = record(96, ({"id": "busy"},))
        fresh = [list(ranks) for ranks in before]
        fresh[released_model][0] = record()
        controller.admission_observations = before
        controller.snapshots = AsyncMock(return_value=fresh)
        controller.call = AsyncMock(side_effect=lambda model, rank, method: fresh[model][rank])
        used, candidates = controller_module.admission_candidates(before, arriving_model, 2560, [1, 1])
        assert candidates == [1]
        await controller.notify_release(released_model)
        assert controller.admission_dirty_models == {released_model}
        assert controller.admission_generation == 1
        assert controller.model_admission_generations[released_model] == 1
        admit = AsyncMock(return_value={"reservation_pending": False, "rank_observed": 0,
                                       "rank_observation": fresh[arriving_model][0]})
        result = await controller.try_direct_admission(2048, 512, [1, 1], admit, arriving_model)
        assert result["admitted"] and result["target"] == 0
        assert result["observations_scope"] == "fresh_all_tenants"
        controller.snapshots.assert_awaited_once()
        assert controller.admission_dirty_models == set()
        admit.assert_awaited_once_with(0)

    asyncio.run(run())


@pytest.mark.parametrize("concurrent", (False, True))
def test_same_tenant_completion_keeps_other_tenant_prefill_rpc_off_fastpath(controller_module, tmp_path, concurrent):
    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path)
        controller.rank_local_admission = True
        controller.concurrent_admission = concurrent
        controller.admission_observations = [[record(), record()], [record(), record()]]
        controller.call = AsyncMock(return_value=record())
        controller.snapshots = AsyncMock(side_effect=RuntimeError("No unrelated Code RPC"))
        await controller.notify_release(1)
        result = await controller.try_direct_admission(64, 256, [1, 1], AsyncMock(return_value={
            "reservation_pending": False, "rank_observed": 0, "rank_observation": record()}), 1)
        assert result["admitted"]
        assert {call.args[0] for call in controller.call.await_args_list} == {1}
        controller.snapshots.assert_not_awaited()
        assert not controller.admission_dirty_models

    asyncio.run(run())


@pytest.mark.parametrize("cached", (False, True))
def test_release_during_concurrent_observation_forces_full_requery(controller_module, tmp_path, cached):
    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path)
        controller.rank_local_admission = controller.concurrent_admission = True
        before = [[record(900), record()], [record(), record(96, ({"id": "busy"},))]]
        fresh = [list(ranks) for ranks in before]
        fresh[0][0] = record()
        controller.admission_observations = before if cached else None
        released = False

        async def snapshot():
            nonlocal released
            assert not controller.lock.locked()
            if not released:
                released = True
                await controller.notify_release(0)
                return before
            return fresh

        async def call(model, rank, method):
            nonlocal released
            assert not controller.lock.locked()
            if not released:
                released = True
                await controller.notify_release(0)
            return fresh[model][rank]

        controller.snapshots = AsyncMock(side_effect=snapshot)
        controller.call = AsyncMock(side_effect=call)
        result = await controller.try_direct_admission(2048, 512, [1, 1], AsyncMock(return_value={
            "reservation_pending": False, "rank_observed": 0, "rank_observation": fresh[1][0]}), 1)
        assert result["admitted"] and result["target"] == 0
        assert result["observations_scope"] == "fresh_all_tenants"
        assert controller.snapshots.await_count == (1 if cached else 2)
        assert controller.admission_generation == 2
        assert not controller.lock.locked()

    asyncio.run(run())


@pytest.mark.parametrize("model", (-1, 2, True, "0"))
def test_release_rejects_unknown_tenant_without_invalidating_cache(controller_module, tmp_path, model):
    async def run():
        controller = controller_module.PlacementController([], {}, tmp_path)
        with pytest.raises(ValueError, match="known tenant"):
            await controller.notify_release(model)
        assert controller.admission_generation == 0

    asyncio.run(run())
