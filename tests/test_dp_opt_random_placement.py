import asyncio
import copy
import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from test_dp_opt_live_placement import modules
from test_dp_opt_rebalance import controller_fake, observations


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("random_placement")


@pytest.fixture
def symmetric_snapshot():
    return {"counts": [[0, 0], [2, 2]], "initial_gib": [6, 6], "capacity_gib": [10, 10],
            "new_gib": 5, "performance_cost_tables_s": [[0, 0], [0] * 5],
            "movable": [{"id": f"conv-{rank}-{index}", "model": 1, "source": rank,
                         "memory_gib": 2, "move_cost_s": 100 if rank else .01}
                        for rank in (0, 1) for index in (0, 1)]}


def test_random_rank_is_reproducible_not_cost_minimization(planner, symmetric_snapshot, monkeypatch):
    reference = importlib.import_module("ilp_reference")
    monkeypatch.setattr(reference, "solve_candidate", lambda *args, **kwargs: pytest.fail("Baseline must not solve ILP"))
    report = planner.random_target_plan(symmetric_snapshot, seed=0)
    assert report == planner.random_target_plan(symmetric_snapshot, seed=0)
    assert report["random_selection"]["feasible_targets"] == [0, 1]
    assert report["winner"]["target"] == 1
    assert report["winner"]["migration_cost_s"] == 100
    assert report["random_selection"]["release_rebalance"] is False
    choices = [planner.random_target_plan(symmetric_snapshot, seed=seed)["winner"]["target"] for seed in range(200)]
    assert 70 <= choices.count(0) <= 130


def test_only_feasible_rank_is_selected(planner, symmetric_snapshot):
    symmetric_snapshot["initial_gib"][1] = 9
    symmetric_snapshot["movable"] = symmetric_snapshot["movable"][:2]
    report = planner.random_target_plan(symmetric_snapshot, seed=0)
    assert report["winner"] is None
    assert report["proven_infeasible"] is False
    symmetric_snapshot["initial_gib"][1] = 5.5
    report = planner.random_target_plan(symmetric_snapshot, seed=0)
    assert report["random_selection"]["feasible_targets"] == [0]
    assert report["winner"]["target"] == 0


def test_shared_source_page_requires_all_owners_and_copy_peak(planner):
    snapshot = {"counts": [[0, 0], [2, 0]], "initial_gib": [6, 5.5], "capacity_gib": [10, 10],
                "new_gib": 5, "performance_cost_tables_s": [[0, 0], [0] * 3],
                "movable": [{"id": name, "model": 1, "source": 0, "memory_gib": .5, "move_cost_s": .01}
                            for name in ("owner-a", "owner-b")],
                "source_pages": [{"id": "shared", "rank": 0, "pool": "1:0", "owners": ["owner-a", "owner-b"],
                                  "memory_gib": 2, "pinned": False}]}
    assert planner.random_target_plan(snapshot, max_moves=1)["winner"] is None
    winner = planner.random_target_plan(snapshot, max_moves=2)["winner"]
    assert winner["moves"] == ["owner-a", "owner-b"]
    assert winner["released_source_pages"] == ["shared"]
    snapshot["source_pages"][0]["pinned"] = True
    assert planner.random_target_plan(snapshot, max_moves=2)["winner"] is None
    snapshot["source_pages"][0]["pinned"] = False
    snapshot["initial_gib"][1] = 9.5
    assert planner.random_target_plan(snapshot, max_moves=2)["winner"] is None


@pytest.mark.parametrize("maximum,seed", [(-1, 0), (True, 0), (2, -1), (2, True)])
def test_invalid_random_settings_rejected(planner, symmetric_snapshot, maximum, seed):
    with pytest.raises(ValueError):
        planner.random_target_plan(symmetric_snapshot, max_moves=maximum, seed=seed)


def test_seed_uses_request_identity_not_policy_order(planner):
    seed = planner.request_placement_seed(42, 0, "code-0")
    assert seed == planner.request_placement_seed(42, 0, "code-0")
    assert seed != planner.request_placement_seed(42, 0, "code-1")
    assert seed != planner.request_placement_seed(42, 1, "code-0")


@pytest.mark.parametrize("preplan", [False, True])
def test_live_cross_tenant_random_admission_and_revalidation(modules, tmp_path, monkeypatch, preplan):
    observed = observations(4)
    observed[1][1] = copy.deepcopy(observed[1][0])
    for request in observed[1][1]["requests"]:
        request["id"] += "-rank1"
    for page in observed[1][1]["pages"]:
        page["owners"] = [name + "-rank1" for name in page["owners"]]
    monkeypatch.setattr(modules.controller, "performance_tables", lambda snapshot, *args:
                        [[0] * (sum(snapshot["counts"][0]) + 2), [0] * (sum(snapshot["counts"][1]) + 1)])
    monkeypatch.setattr(modules.controller, "enumerate_targets", lambda *args, **kwargs: pytest.fail("Random admission ran ILP"))

    async def run():
        controller = controller_fake(modules.controller, {}, tmp_path, observed)
        controller.revalidate_prepared = preplan
        admit = AsyncMock(return_value={"reservation_pending": False})
        options = {"policy": "random_no_return", "max_moves": 2, "candidate_limit": 4, "placement_seed": 0}
        prepared = await controller.prepare_admission(6880, 16, [1, 1], **options) if preplan else None
        report = await controller.schedule_admission(6880, 16, [1, 1], admit, prepared=prepared, **options)
        assert report["admitted"]
        assert report["winner"]["target"] == 1
        assert len(report["moves"]) == 1 and report["moves"][0]["model"] == 1
        assert report["moves"][0]["source"] == 1 and report["moves"][0]["destination"] == 0
        if preplan:
            verifier = importlib.import_module("verify_candidate_revalidation")
            for move in report["moves"]:
                move.update({"integrity": "device_echo_full_bytewise", "bitwise_verified": True})
            verifier.verify_revalidation(report, {}, None)
        controller.resume.assert_awaited_once()
        admit.assert_awaited_once_with(1)

    asyncio.run(run())
