import importlib
import math
from pathlib import Path

import pytest


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("verify_workaware_rebalance").same_replayed_cost


def test_only_machine_roundoff_is_permitted(compare):
    cost = .02534883076796341
    assert compare(cost, math.nextafter(cost, 0))
    assert compare({"cost": [cost], "rank": 1}, {"cost": [math.nextafter(cost, 0)], "rank": 1})
    shifted = cost
    for step in range(5):
        shifted = math.nextafter(shifted, 0)
    assert not compare(cost, shifted)
    assert not compare(cost, cost + 1e-12)


@pytest.mark.parametrize("left,right", [(1, 1.0), (True, 1), (False, True), (1, 2),
    ({"rank": 0}, {"rank": 1}), ([1], [1, 2]), ({"cost": 1}, {"other": 1}),
    (float("nan"), float("nan")), (float("inf"), float("inf")), (0.0, 1e-100)])
def test_structural_tokens_ranks_gates_and_invalid_numbers_remain_strict(compare, left, right):
    assert not compare(left, right)
