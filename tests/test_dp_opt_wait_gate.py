import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("live_controller")


def observations(remaining):
    empty = {"requests": [], "pages": [], "mapped_bytes": 0, "block_size": 16,
             "block_bytes": 1024**3, "page_bytes": 1024**3, "uniform_page_geometry": True}
    result = [[copy.deepcopy(empty) for rank in (0, 1)] for model in (0, 1)]
    for rank in (0, 1):
        requests = [{"id": f"conv-{rank}-{index}", "blocks": [index], "movable": True,
                     "estimated_remaining_decode_s": remaining}
                    for index in range(3)]
        result[1][rank] = {**empty, "requests": requests, "mapped_bytes": 3 * 1024**3,
                           "pages": [{"id": index, "owners": [request["id"]], "pinned": False,
                                      "memory_gib": 1} for index, request in enumerate(requests)]}
    return result


@pytest.mark.parametrize("remaining,worthwhile", [(1, False), (500, True), (None, False)])
def test_wait_gate_includes_all_active_cost_and_unknown_prediction(controller, remaining, worthwhile):
    result = controller.wait_migration_precheck(observations(remaining), 0, 48, [4, 4])
    assert result["may_reduce_latency_sum"] is worthwhile
    assert result["active_plus_incoming"] == 7
    assert result["estimated_wait_s"] == remaining
    assert result["migration_latency_sum_lower_s"] == 7 * (.35 + .15 + 10)


def test_shared_page_wait_requires_last_owner_completion(controller):
    records = observations(1)
    for rank in (0, 1):
        records[1][rank]["requests"][-1]["estimated_remaining_decode_s"] = 100
        records[1][rank]["pages"] = [{"id": 0, "owners": [request["id"] for request in records[1][rank]["requests"]],
                                      "pinned": False, "memory_gib": 3}]
    result = controller.wait_migration_precheck(records, 0, 48, [4, 4])
    assert result["estimated_wait_s"] == 100
    for rank in (0, 1):
        records[1][rank]["pages"][0]["pinned"] = True
    result = controller.wait_migration_precheck(records, 0, 48, [4, 4])
    assert not result["may_reduce_latency_sum"] and result["estimated_wait_s"] is None


def test_boundary_wait_charges_incoming_handshake_once_not_all_old_requests(controller):
    options = {"incoming_handshake_s": 3, "old_transaction_fixed_s": .1, "preplan_s": .5}
    result = controller.wait_migration_precheck(observations(10), 0, 48, [4, 4],
        copy_seconds_per_gib=.03, boundary_cost=options)
    assert result["migration_latency_sum_lower_s"] == pytest.approx(3 + .5 + 7 * (.1 + 2 * .03))
    assert result["may_reduce_latency_sum"]
    options["preplan_s"] = float("nan")
    with pytest.raises(ValueError):
        controller.wait_migration_precheck(observations(10), 0, 48, [4, 4], boundary_cost=options)


@pytest.mark.parametrize("options", [{"safety_fraction": 0}, {"safety_fraction": 1.1},
                                     {"pause_floor_s": -1}, {"copy_seconds_per_gib": float("inf")}])
def test_invalid_wait_costs_are_rejected(controller, options):
    with pytest.raises(ValueError):
        controller.wait_migration_precheck(observations(1), 0, 48, [4, 4], **options)
