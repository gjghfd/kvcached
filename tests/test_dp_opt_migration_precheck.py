import importlib
import itertools
import random
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("live_controller")


def observations():
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
             "block_bytes": 1024**3, "page_bytes": 1024**3, "uniform_page_geometry": True}
    result = [[dict(empty) for rank in (0, 1)] for model in (0, 1)]
    for rank in (0, 1):
        records = [{"id": f"conv{rank}-{index}", "movable": True, "blocks": [index]}
                   for index in range(3)]
        result[1][rank] = {**empty, "mapped_bytes": 3 * 1024**3, "requests": records,
                           "pages": [{"id": index, "owners": [request["id"]],
                                      "pinned": False, "memory_gib": 1}
                                     for index, request in enumerate(records)]}
    return result


def test_own_model_with_no_resident_requests_cannot_migrate_other_tenant(controller):
    result = controller.migration_precheck(observations(), 0, 48, [4, 4], "own_model", 8, 8)
    assert result["selected_candidates"] == 0 and not result["may_fit"]


def test_more_than_two_moves_can_be_required(controller):
    assert not controller.migration_precheck(observations(), 0, 64, [4, 4], "ilp", 2, 2)["may_fit"]
    assert controller.migration_precheck(observations(), 0, 64, [4, 4], "ilp", 8, 8)["may_fit"]


def test_shared_page_release_requires_all_owners(controller):
    records = observations()
    for rank in (0, 1):
        records[1][rank]["pages"] = [{"id": 0, "owners": [request["id"] for request in records[1][rank]["requests"]],
                                      "pinned": False, "memory_gib": 3}]
    assert not controller.migration_precheck(records, 0, 64, [4, 4], "kv_only", 2, 8)["may_fit"]
    assert controller.migration_precheck(records, 0, 64, [4, 4], "kv_only", 3, 8)["may_fit"]
    for rank in (0, 1):
        records[1][rank]["pages"][0]["pinned"] = True
    assert not controller.migration_precheck(records, 0, 64, [4, 4], "ilp", 8, 8)["may_fit"]


def test_release_bound_never_understates_any_owner_complete_subset(controller):
    generator = random.Random(20261022)
    for iteration in range(40):
        records = observations()
        for rank in (0, 1):
            identifiers = [request["id"] for request in records[1][rank]["requests"]]
            records[1][rank]["pages"] = [{"id": index, "owners": generator.sample(identifiers, generator.randint(1, 3)),
                                          "pinned": False, "memory_gib": 1}
                                         for index in range(3)]
        limit = generator.randint(1, 3)
        result = controller.migration_precheck(records, 0, 64, [4, 4], "ilp", limit, 8)
        for rank in (0, 1):
            identifiers = [request["id"] for request in records[1][rank]["requests"]]
            for count in range(limit + 1):
                for subset in itertools.combinations(identifiers, count):
                    released = sum(page["memory_gib"] for page in records[1][rank]["pages"]
                                   if set(page["owners"]).issubset(subset))
                    assert released <= result["optimistic_source_release_gib"][rank] + 1e-6


@pytest.mark.parametrize("moves", [-1, True, 1.5])
def test_precheck_rejects_invalid_move_limit(controller, moves):
    with pytest.raises(ValueError, match="valid move limit"):
        controller.migration_precheck(observations(), 0, 64, [4, 4], "ilp", moves, 2)
